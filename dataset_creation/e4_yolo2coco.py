#!/usr/bin/env python3
"""
Step 4: generates COCO annotations for an already-split YOLO dataset
(train/ val/ test/, each with flat images/ + labels/) and its data.yaml.

Writes root/annotations/instances_<split>.json for each split present
(train/val/test), reading class names from root/data.yaml.

Usage:
    python3 e4_yolo2coco.py --root /path/to/dataset_sam_3.0
"""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def load_class_names(data_yaml: Path) -> list[str]:
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    return [names[i] for i in sorted(names)]


def yolo_line_to_coco_bbox(line: str, img_w: int, img_h: int):
    """YOLO: class xc yc w h (normalized 0..1) -> COCO bbox [x_min, y_min, w, h] in pixels."""
    parts = line.strip().split()
    cls = int(float(parts[0]))
    xc, yc, w, h = (float(v) for v in parts[1:5])
    xc *= img_w; yc *= img_h; w *= img_w; h *= img_h

    x_min = max(0.0, xc - w / 2.0)
    y_min = max(0.0, yc - h / 2.0)
    w = max(0.0, min(w, img_w - x_min))
    h = max(0.0, min(h, img_h - y_min))

    return cls, [x_min, y_min, w, h]


def convert_split(images_dir: Path, labels_dir: Path, class_names: list[str]) -> dict:
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]

    images, annotations = [], []
    img_id = ann_id = 1

    img_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    for img_path in img_paths:
        with Image.open(img_path) as im:
            w, h = im.size

        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            txt = label_path.read_text(encoding="utf-8").strip()
            for line in txt.splitlines():
                if not line.strip():
                    continue
                cls, bbox = yolo_line_to_coco_bbox(line, w, h)
                if not (0 <= cls < len(class_names)):
                    raise ValueError(f"Classe {cls} hors de portée (0..{len(class_names)-1}) dans {label_path}")

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [float(b) for b in bbox],
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1

        img_id += 1

    return {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Racine du dataset (contient train/ val/ test/ et data.yaml)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml introuvable : {data_yaml}")

    class_names = load_class_names(data_yaml)
    print(f"{len(class_names)} classes lues dans {data_yaml}")

    annotations_dir = root / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"  [SKIP] {split}/ — images/ ou labels/ introuvable")
            continue

        print(f"  Conversion {split}...")
        coco = convert_split(images_dir, labels_dir, class_names)

        out_json = annotations_dir / f"instances_{split}.json"
        out_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        imgs_with_ann = len({a["image_id"] for a in coco["annotations"]})
        print(f"    -> {out_json}")
        print(f"    images: {n_img} | annotations: {n_ann} | images avec >=1 boîte: {imgs_with_ann} | vides: {n_img - imgs_with_ann}")

    print(f"\nAnnotations écrites dans {annotations_dir}")


if __name__ == "__main__":
    main()
