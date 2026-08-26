#!/usr/bin/env python3
"""
Step 2: flattens a test/ or train/ folder (produced by e1) into a single
images/ + labels/ pair at its root.

Each frame from a run is renamed <run_name>_<original_frame_name>.ext
(e.g. Ray_07/images/frame_000000.jpg -> images/Ray_07_frame_000000.jpg) to
stay traceable to its source run. Frames in background/ are already named
<run>_bg_<n>.ext (by e1), so they are copied as-is, without re-prefixing.

The original extension (jpg/png) is kept as-is, no re-encoding.

Usage:
    python3 e2_flatten_and_rename.py --split-dir /path/to/dataset_sam_3.0/train
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_pair(img_src: Path, lbl_dir: Path, dst_images: Path, dst_labels: Path, new_stem: str) -> bool:
    """Copies an image/label pair under the new name. Returns False (and
    warns) if the destination already exists; never silently overwrite."""
    dst_img = dst_images / f"{new_stem}{img_src.suffix}"
    if dst_img.exists():
        print(f"    [SKIP] {dst_img.name} existe déjà")
        return False

    shutil.copy2(img_src, dst_img)

    lbl_src = lbl_dir / f"{img_src.stem}.txt"
    dst_lbl = dst_labels / f"{new_stem}.txt"
    if lbl_src.exists():
        shutil.copy2(lbl_src, dst_lbl)
    else:
        dst_lbl.touch()
    return True


def flatten_run(run_dir: Path, dst_images: Path, dst_labels: Path, prefix: str | None) -> int:
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    if not images_dir.is_dir():
        print(f"  [WARN] {run_dir.name}: pas de sous-dossier images/")
        return 0

    n = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        new_stem = f"{prefix}_{img_path.stem}" if prefix else img_path.stem
        if copy_pair(img_path, labels_dir, dst_images, dst_labels, new_stem):
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True,
                        help="Dossier test/ ou train/ produit par e1_split_test_train_folders.py")
    args = parser.parse_args()

    split_dir = Path(args.split_dir).resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {split_dir}")

    dst_images = split_dir / "images"
    dst_labels = split_dir / "labels"
    dst_images.mkdir(exist_ok=True)
    dst_labels.mkdir(exist_ok=True)

    run_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name not in ("images", "labels"))
    if not run_dirs:
        print(f"Aucun sous-dossier de run trouvé dans {split_dir}")
        return

    total = 0
    for run_dir in run_dirs:
        is_background = run_dir.name == "background"
        prefix = None if is_background else run_dir.name
        n = flatten_run(run_dir, dst_images, dst_labels, prefix)
        print(f"  {run_dir.name:20s} -> {n} frames")
        total += n

    print("\n" + "-" * 50)
    print(f"Total frames aplaties : {total}")
    print(f"Images : {dst_images}")
    print(f"Labels : {dst_labels}")
    print("-" * 50)


if __name__ == "__main__":
    main()
