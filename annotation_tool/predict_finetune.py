"""
predict_finetune.py
===================
Runs inference with the fine-tuned SAM3 model over a folder of images.
Saves predictions in YOLO format, with a confidence score appended to each line.

Output format (per line):
    <class_id> <xc_norm> <yc_norm> <w_norm> <h_norm> <conf>

Usage:
  python predict_finetune.py \
    --images-dir ./datasets_test/dataset_test_3.0/images\
    --output-dir ./results/detector/SAM3_finetuned/res_test_3.0
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor


# ──────────────────────────────────────────────────────────────
# Prompts -> YOLO class
# The index in the list is the class_id SAM3 returns
# ──────────────────────────────────────────────────────────────
PROMPTS = [
    "a Squid",       # class 0
    "a Sardine",     # class 1
    "a Ray",         # class 2
    "a Sunfish",     # class 3
    "a Pilot Fish",  # class 4
    "a Shark",       # class 5
]


def get_box_from_mask(mask: np.ndarray, W: int, H: int):
    """Return the xyxy bbox derived from the SAM3 mask (original resolution)."""
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    coords = np.column_stack(np.where(m > 0.5))
    if coords.shape[0] == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def xyxy2xywhn(x1, y1, x2, y2, W, H):
    """Convert xyxy to normalized xywh (YOLO format)."""
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h


def main():
    parser = argparse.ArgumentParser(description="Prédictions SAM3 fine-tuné → YOLO + conf")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Dossier contenant les images à prédire")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dossier de sortie (contiendra un sous-dossier labels/)")
    parser.add_argument("--model",      type=str, default="sam3_finetune.pt",
                        help="Fichier modèle (défaut: sam3_finetune.pt)")
    parser.add_argument("--conf",       type=float, default=0.1,
                        help="Seuil de confiance (défaut: 0.1)")
    args = parser.parse_args()

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted([
        f for f in os.listdir(args.images_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    if not images:
        print(f"[ERREUR] Aucune image trouvée dans : {args.images_dir}")
        return

    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Modèle      : {args.model}")
    print(f"Confiance   : {args.conf}")
    print(f"Images      : {len(images)} fichiers dans {args.images_dir}")
    print(f"Sortie      : {labels_dir}")
    print(f"Prompts     :")
    for i, p in enumerate(PROMPTS):
        print(f"  [{i}] '{p}'")
    print()

    # Ultralytics can't load a raw state_dict like sam3_finetune.pt directly.
    # We first initialize with sam3.pt (full architecture), then inject the
    # fine-tuned weights manually after the first set_image call.
    base_model  = "sam3.pt"
    ft_weights  = args.model  # sam3_finetune.pt

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        imgsz=644,
        model=base_model,
        half=True,
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    ft_loaded = False  # inject the fine-tuned weights on the first frame

    n_frames_with_det = 0
    n_total_det       = 0
    start_time        = datetime.now()

    for idx, img_name in enumerate(images):
        img_path = os.path.join(args.images_dir, img_name)
        frame    = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] Impossible de lire : {img_path}")
            continue

        H, W = frame.shape[:2]

        predictor.set_image(frame)

        # Inject the fine-tuned weights exactly once, after the first set_image call.
        # Skipped if ft_weights == base_model (sam3.pt is already loaded by Ultralytics).
        if not ft_loaded and predictor.model is not None:
            if ft_weights != base_model:
                ckpt = torch.load(ft_weights, map_location="cpu", weights_only=False)
                # sam3_finetune.pt: {'model': OrderedDict}  /  other formats: flat dict
                state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                missing, unexpected = predictor.model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[WARN] Clés manquantes ({len(missing)}) : {missing[:5]}")
                if unexpected:
                    print(f"[WARN] Clés inattendues ({len(unexpected)}) : {unexpected[:5]}")
                predictor.model.to(predictor.device).half()
                print(f"[INFO] Poids fine-tunés chargés depuis : {ft_weights}\n")
            else:
                print(f"[INFO] Modèle de base utilisé : {base_model}\n")
            ft_loaded = True

        results = predictor(text=PROMPTS)

        lines = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            cls_arr  = r.boxes.cls.cpu().numpy()
            conf_arr = r.boxes.conf.cpu().numpy()

            for i, sam_idx in enumerate(cls_arr):
                sam_idx  = int(sam_idx)
                cls_id   = sam_idx  # index doubles as the YOLO class (same ordering)
                conf     = float(conf_arr[i])

                if r.masks is not None and i < len(r.masks.data):
                    box = get_box_from_mask(r.masks.data[i].cpu().numpy(), W, H)
                    if box is None:
                        continue
                    xc, yc, bw, bh = xyxy2xywhn(*box, W, H)
                else:
                    xc, yc, bw, bh = r.boxes.xywhn[i].cpu().numpy()

                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")

        stem     = os.path.splitext(img_name)[0]
        txt_path = os.path.join(labels_dir, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        if lines:
            n_frames_with_det += 1
            n_total_det       += len(lines)

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1:5d}/{len(images)}]  détections cumulées : {n_total_det}")

    elapsed = datetime.now() - start_time

    config_path = os.path.join(args.output_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modèle         : {args.model}\n")
        f.write(f"Confiance      : {args.conf}\n")
        f.write(f"Images dir     : {args.images_dir}\n")
        f.write(f"Prompts        : {PROMPTS}\n")
        f.write(f"Images totales : {len(images)}\n")
        f.write(f"Frames avec det: {n_frames_with_det}\n")
        f.write(f"Détections tot : {n_total_det}\n")
        f.write(f"Durée          : {elapsed}\n")

    print(f"\n{'='*50}")
    print(f"  Images traitées : {len(images)}")
    print(f"  Frames avec déт.: {n_frames_with_det}")
    print(f"  Détections tot. : {n_total_det}")
    print(f"  Durée           : {elapsed}")
    print(f"  Labels          : {labels_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
