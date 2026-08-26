#!/usr/bin/env python3
"""
Step 3: randomly splits a train/ folder (flat images/ + labels/) into
train/ + val/, according to a given ratio (e.g. 90 for 90/10, 80 for 80/20).

Images/labels drawn for val are MOVED (cut, not copied) from
train/{images,labels} to a new val/{images,labels} created at the root
of the given path.

Usage:
    python3 e3_split_train_val.py --train-dir /path/to/dataset_sam_3.0/train --ratio 90
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def move_pair(img_path: Path, labels_dir: Path, dst_images: Path, dst_labels: Path) -> None:
    shutil.move(str(img_path), str(dst_images / img_path.name))

    lbl_src = labels_dir / f"{img_path.stem}.txt"
    dst_lbl = dst_labels / f"{img_path.stem}.txt"
    if lbl_src.exists():
        shutil.move(str(lbl_src), str(dst_lbl))
    else:
        dst_lbl.touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Dossier train/ (contenant images/ et labels/ à plat)")
    parser.add_argument("--ratio", type=int, required=True, help="Pourcentage gardé en train, ex: 90 pour 90/10, 80 pour 80/20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = Path(args.train_dir).resolve()
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"images/ ou labels/ introuvable dans {train_dir}")
    if not (0 < args.ratio < 100):
        raise ValueError("--ratio doit être entre 0 et 100 (exclus)")

    val_dir = train_dir.parent / "val"
    dst_images = val_dir / "images"
    dst_labels = val_dir / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    n_total = len(images)
    n_val = round(n_total * (100 - args.ratio) / 100)

    rng = random.Random(args.seed)
    rng.shuffle(images)
    val_images = images[:n_val]

    print(f"Total images en train : {n_total}")
    print(f"Ratio                 : {args.ratio}/{100 - args.ratio}")
    print(f"-> val                : {n_val} images")
    print(f"-> train (restant)    : {n_total - n_val} images")
    print(f"Déplacement en cours...")

    for img_path in val_images:
        move_pair(img_path, labels_dir, dst_images, dst_labels)

    print("\n" + "-" * 50)
    print(f"val/   : {dst_images} ({n_val} images)")
    print(f"train/ : {images_dir} ({n_total - n_val} images restantes)")
    print("-" * 50)


if __name__ == "__main__":
    main()
