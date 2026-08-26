"""
create_dataset.py
=================
Processes a video frame by frame with SAM3SemanticPredictor (text prompts)
and produces a YOLO bbox dataset.

The text -> YOLO class mapping is defined in TEXT_TO_CLASS below.
SAM3 returns the prompt's index in the list; we remap it to the real class ID.

Usage:
  python create_dataset.py \
    --Video-path /path/to/video.mp4 \
    --Output-dir /path/to/output/ \
    --Conf 0.60
"""

from ultralytics.models.sam import SAM3SemanticPredictor
import os
import cv2
from datetime import datetime
import numpy as np
import argparse


# ──────────────────────────────────────────────────────────────
# Text prompt -> YOLO class ID mapping
# Edit the prompts here to improve detection.
# The order determines the index SAM3 returns (idx -> class ID).
# ──────────────────────────────────────────────────────────────

TEXT_TO_CLASS = {
    "a squid":        0,
    "a sardine":      1,
    "a stingray":     2,
    "a sunfish":      3,
    "a pilot fish":   4,
    "a shark":        5,
}


def get_box_from_mask(mask, w_orig, h_orig):
    mask_float = mask.astype(np.float32)
    if len(mask_float.shape) == 3:
        mask_float = mask_float.squeeze()
    mask_resized = cv2.resize(mask_float, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    binary_mask  = (mask_resized > 0.5).astype(np.uint8)
    coords = np.column_stack(np.where(binary_mask > 0))
    if coords.shape[0] == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max


def xyxy2xywhn(x1, y1, x2, y2, W, H):
    w  = x2 - x1
    h  = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return xc / W, yc / H, w / W, h / H


def main():
    parser = argparse.ArgumentParser(description="Génère un dataset YOLO bbox avec SAM3 (prompts texte)")
    parser.add_argument("--Video-path",    type=str, required=True)
    parser.add_argument("--Output-dir",    type=str, required=True)
    parser.add_argument("--Conf",          type=float, default=0.50,
                        help="Seuil de confiance SAM3 (défaut: 0.50)")
    parser.add_argument("--classes",       type=str, nargs="+", default=None,
                        help="Sous-ensemble de classes à détecter, ex: --classes 'a stingray' 'a squid'. "
                             "Par défaut : toutes les classes de TEXT_TO_CLASS.")
    args = parser.parse_args()

    # Select the prompts
    if args.classes:
        unknown = [c for c in args.classes if c not in TEXT_TO_CLASS]
        if unknown:
            print(f"[ERREUR] Prompts inconnus : {unknown}")
            print(f"  Prompts disponibles : {list(TEXT_TO_CLASS.keys())}")
            return
        active = {p: TEXT_TO_CLASS[p] for p in args.classes}
    else:
        active = TEXT_TO_CLASS

    prompts    = list(active.keys())
    idx_to_cls = list(active.values())   # SAM index -> YOLO class

    print(f"Prompts actifs :")
    for i, (p, c) in enumerate(active.items()):
        print(f"  [{i}] '{p}'  →  class YOLO {c}")

    os.makedirs(args.Output_dir, exist_ok=True)
    labels_dir = os.path.join(args.Output_dir, "labels")
    imgs_dir   = os.path.join(args.Output_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(imgs_dir,   exist_ok=True)

    cap    = cv2.VideoCapture(args.Video_path)
    FPS    = cap.get(cv2.CAP_PROP_FPS)
    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVidéo : {WIDTH}x{HEIGHT} @ {FPS:.1f}fps — {TOTAL} frames")
    print(f"Sortie : {args.Output_dir}\n")

    overrides = dict(conf=args.Conf, task="segment", mode="predict",
                     imgsz=644, model="sam3.pt", half=True, save=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)

    n_detections = 0
    start_time   = datetime.now()
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictor.set_image(frame)
        results = predictor(text=prompts)

        lines = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            cls_tab = r.boxes.cls.cpu().numpy()

            for i in range(len(cls_tab)):
                sam_idx  = int(cls_tab[i])
                yolo_cls = idx_to_cls[sam_idx] if sam_idx < len(idx_to_cls) else sam_idx

                if r.masks is not None and i < len(r.masks.data):
                    box = get_box_from_mask(r.masks.data[i].cpu().numpy(), WIDTH, HEIGHT)
                    if box is None:
                        continue
                    xc, yc, w, h = xyxy2xywhn(*box, WIDTH, HEIGHT)
                else:
                    xc, yc, w, h = r.boxes.xywhn[i].cpu().numpy()

                lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if lines:
            img_path = os.path.join(imgs_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(img_path, frame)
            txt_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            n_detections += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx:5d}/{TOTAL}  —  {n_detections} détections")

        frame_idx += 1

    cap.release()
    elapsed = datetime.now() - start_time

    with open(os.path.join(args.Output_dir, "config.txt"), "w") as f:
        f.write(f"Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vidéo          : {args.Video_path}\n")
        f.write(f"Prompts        : {prompts}\n")
        f.write(f"Confiance      : {args.Conf}\n")
        f.write(f"Frames totales : {TOTAL}\n")
        f.write(f"Détections     : {n_detections}\n")
        f.write(f"Durée          : {elapsed}\n")

    print(f"\n{'='*45}")
    print(f"  Détections : {n_detections} / {frame_idx} frames")
    print(f"  Durée      : {elapsed}")
    print(f"  Labels     : {labels_dir}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
