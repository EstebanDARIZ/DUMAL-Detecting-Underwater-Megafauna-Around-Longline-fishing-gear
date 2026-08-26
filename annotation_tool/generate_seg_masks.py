"""
generate_seg_masks.py
=====================
Walks a multi-video dataset and generates SAM3 segmentation masks
from the existing YOLO bboxes.

Expected dataset structure:
  dataset_root/
    video_001/
      images/   frame_000001.jpg ...
      labels/   frame_000001.txt ...  (YOLO bbox: cls xc yc w h)
    video_002/
      ...

Output (added inside each video folder):
  dataset_root/
    video_001/
      images/       (unchanged)
      labels/       (unchanged)
      mask_labels/  frame_000001.txt ...  (YOLO seg: cls x1 y1 x2 y2 ...)

The existing images/ folder is reused as-is for fine-tuning.
Images with no mask label are simply skipped by Ultralytics.

Usage:
  # 500 images per video, evenly spaced
  python generate_seg_masks.py --dataset /path/to/dataset_root --per-video 500

  # Resume an interrupted run
  python generate_seg_masks.py --dataset /path/to/dataset_root --per-video 500 --resume

  # Test on 50 images per video
  python generate_seg_masks.py --dataset /path/to/dataset_root --per-video 50 --resume
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# YOLO bbox -> pixel conversion
# ──────────────────────────────────────────────────────────────

def yolo_bbox_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)


# ──────────────────────────────────────────────────────────────
# Mask -> normalized YOLO seg polygon
# ──────────────────────────────────────────────────────────────

def mask_to_polygon(mask: np.ndarray, W: int, H: int, epsilon_factor: float = 0.005):
    """
    Convert a SAM3 mask to a normalized polygon [x1,y1,x2,y2,...].
    Returns the largest contour (the main fish).
    """
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None

    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)

    pts = approx.reshape(-1, 2).astype(float)
    pts[:, 0] /= W
    pts[:, 1] /= H
    pts = np.clip(pts, 0.0, 1.0)

    if len(pts) < 3:
        return None

    return pts.flatten().tolist()


# ──────────────────────────────────────────────────────────────
# Read a YOLO bbox label file
# ──────────────────────────────────────────────────────────────

def load_bbox_labels(label_path: str):
    """Return a list of (class_id, xc, yc, w, h)."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    labels.append((cls, xc, yc, w, h))
                except ValueError:
                    continue
    return labels


# ──────────────────────────────────────────────────────────────
# Evenly select N images from a video
# ──────────────────────────────────────────────────────────────

def select_images(all_imgs: list[Path], n: int) -> list[Path]:
    """
    Select n evenly spaced images from the list.
    If n >= len(all_imgs), return everything.
    """
    if n <= 0 or n >= len(all_imgs):
        return all_imgs
    indices = np.linspace(0, len(all_imgs) - 1, n, dtype=int)
    return [all_imgs[i] for i in indices]


# ──────────────────────────────────────────────────────────────
# Process a single video
# ──────────────────────────────────────────────────────────────

def process_video_folder(model, video_dir: Path, per_video: int,
                         resume: bool, verbose: bool) -> dict:
    """
    Process one video folder. Returns a stats dict.
    """
    imgs_dir  = video_dir / "images"
    lbls_dir  = video_dir / "labels"
    masks_dir = video_dir / "mask_labels"

    if not imgs_dir.exists() or not lbls_dir.exists():
        return {"skipped": True, "reason": "pas de images/ ou labels/"}

    masks_dir.mkdir(exist_ok=True)

    # List images that have a non-empty bbox label
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_imgs = sorted(
        p for p in imgs_dir.iterdir()
        if p.suffix.lower() in exts
        and (lbls_dir / (p.stem + ".txt")).exists()
        and os.path.getsize(lbls_dir / (p.stem + ".txt")) > 0
    )

    if not all_imgs:
        return {"skipped": True, "reason": "aucune image avec label bbox"}

    selected = select_images(all_imgs, per_video)

    # Filter if --resume
    if resume:
        selected = [
            p for p in selected
            if not (masks_dir / (p.stem + ".txt")).exists()
        ]

    stats = {"total_selected": len(selected), "generated": 0, "failed": 0}

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            stats["failed"] += 1
            continue

        H, W = img.shape[:2]
        labels = load_bbox_labels(str(lbls_dir / (img_path.stem + ".txt")))
        if not labels:
            continue

        bboxes = [
            list(yolo_bbox_to_xyxy(xc, yc, w, h, W, H))
            for (_, xc, yc, w, h) in labels
        ]

        try:
            results = model(img, bboxes=bboxes, verbose=False)
        except Exception as e:
            if verbose:
                print(f"    [!] SAM3 error {img_path.name}: {e}")
            stats["failed"] += 1
            continue

        r = results[0]
        if r.masks is None or len(r.masks) == 0:
            stats["failed"] += 1
            continue

        masks_np = r.masks.data.cpu().numpy()

        seg_lines = []
        for det_i, (cls, *_) in enumerate(labels):
            if det_i >= len(masks_np):
                break
            polygon = mask_to_polygon(masks_np[det_i], W, H)
            if polygon is None:
                continue
            coords_str = " ".join(f"{v:.6f}" for v in polygon)
            seg_lines.append(f"{cls} {coords_str}\n")

        if not seg_lines:
            stats["failed"] += 1
            continue

        out_path = masks_dir / (img_path.stem + ".txt")
        with open(out_path, "w") as f:
            f.writelines(seg_lines)

        stats["generated"] += 1

    return stats


# ──────────────────────────────────────────────────────────────
# Discover video folders
# ──────────────────────────────────────────────────────────────

def find_video_folders(dataset_root: Path) -> list[Path]:
    """
    Return the subfolders that contain images/ and labels/.
    Also supports a flat dataset (dataset_root directly contains images/ and labels/).
    """
    # Flat dataset?
    if (dataset_root / "images").exists() and (dataset_root / "labels").exists():
        return [dataset_root]

    # Video subfolders
    folders = sorted(
        d for d in dataset_root.iterdir()
        if d.is_dir()
        and (d / "images").exists()
        and (d / "labels").exists()
    )
    return folders


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Génère des masques de segmentation SAM3 sur un dataset multi-vidéos"
    )
    parser.add_argument("--dataset",   type=str, required=True,
                        help="Dossier racine du dataset (contient les sous-dossiers vidéo)")
    parser.add_argument("--per-video", type=int, default=500,
                        help="Nombre d'images à traiter par vidéo, espacées uniformément (défaut: 500)")
    parser.add_argument("--model",     type=str, default="sam3.pt",
                        help="Poids SAM3 (défaut: sam3.pt)")
    parser.add_argument("--resume",    action="store_true",
                        help="Ignorer les images dont mask_labels/*.txt existe déjà")
    parser.add_argument("--verbose",   action="store_true",
                        help="Afficher les erreurs par image")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"[ERREUR] Dossier introuvable : {dataset_root}")
        sys.exit(1)

    # Discover the videos
    video_folders = find_video_folders(dataset_root)
    if not video_folders:
        print(f"[ERREUR] Aucun dossier vidéo trouvé dans {dataset_root}")
        print("  Attendu : chaque sous-dossier doit contenir images/ et labels/")
        sys.exit(1)

    print(f"Vidéos trouvées : {len(video_folders)}")
    print(f"Images par vidéo (max) : {args.per_video}")
    print(f"Mode resume : {args.resume}\n")

    # Load the model
    print(f"Chargement du modèle : {args.model}")
    from ultralytics import SAM
    model = SAM(args.model)

    total_generated = 0
    total_failed    = 0

    for i, vdir in enumerate(video_folders, 1):
        print(f"[{i:2d}/{len(video_folders)}] {vdir.name}")
        stats = process_video_folder(
            model, vdir, args.per_video, args.resume, args.verbose
        )

        if stats.get("skipped"):
            print(f"        → ignoré ({stats['reason']})")
            continue

        gen = stats["generated"]
        fail = stats["failed"]
        total_generated += gen
        total_failed    += fail
        print(f"        → {gen}/{stats['total_selected']} masques générés"
              + (f"  ({fail} échecs)" if fail else ""))

        torch.cuda.empty_cache()

    # Overall summary
    print(f"\n{'='*55}")
    print(f"  Vidéos traitées  : {len(video_folders)}")
    print(f"  Masques générés  : {total_generated}")
    print(f"  Échecs           : {total_failed}")
    print(f"{'='*55}")
    print("\nPour créer le dataset.yaml et lancer le fine-tuning :")
    print(f"  python finetune_sam3.py --prepare-yaml --data {dataset_root} --classes poisson")


if __name__ == "__main__":
    main()
