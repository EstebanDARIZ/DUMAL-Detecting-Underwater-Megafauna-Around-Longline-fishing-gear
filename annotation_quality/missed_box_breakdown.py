#!/usr/bin/env python3
"""
Breakdown of missed reference boxes by class and size, for a given
ref-vs-annotator pair.

A reference box is considered "missed" when the annotator drew STRICTLY
FEWER boxes than the reference on that image (pure counting, consistent
with aggregate_iou_iaa.py's counting metric): on these images, the full
Hungarian matching (with no IoU threshold) leaves n_ref - n_pred reference
boxes with no counterpart at all, and those are the ones we identify and
report the class and area for. This differs from the IoU >= 0.5 threshold
used for the localization score, which also rejects poor-quality matches,
not just unbalanced counts.

Used to document, for a given annotator, where under-counting concentrates
(rare small classes? common species? small boxes?). Complements
aggregate_iou_iaa.py's overall counting metric with the detail needed for
the paper's "What drives disagreement" section.

Usage:
    python IAA/missed_box_breakdown.py --pred Annotator_D
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.optimize import linear_sum_assignment

from compute_iou_iaa import iou

IOU_DIR = Path("./IAA/IoU")
IMAGES_DIR = IOU_DIR / "images"

CLASS_NAMES = {
    0: "Squid", 1: "Sardine", 2: "Ray", 3: "Sunfish", 4: "Pilot Fish",
    5: "Shark", 6: "JellyFish", 7: "Tuna", 8: "Mackerel", 9: "[cls_9]",
}


def read_boxes_with_class(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            out.append((int(parts[0]), tuple(float(v) for v in parts[1:5])))
    return out


def image_size(stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return Image.open(p).size
    raise FileNotFoundError(stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="Annotator_Ref")
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    ref_dir = IOU_DIR / args.ref
    pred_dir = IOU_DIR / args.pred

    all_ref_areas = []
    missed_areas = []
    missed_by_class = {}

    for ref_path in sorted(ref_dir.glob("*.txt")):
        stem = ref_path.stem
        pred_path = pred_dir / f"{stem}.txt"

        ref_labeled = read_boxes_with_class(ref_path)
        pred_labeled = read_boxes_with_class(pred_path)
        if not ref_labeled and not pred_labeled:
            continue

        W, H = image_size(stem)
        ref_boxes = [b for _, b in ref_labeled]
        pred_boxes = [b for _, b in pred_labeled]

        for cls, (cx, cy, w, h) in ref_labeled:
            all_ref_areas.append(w * W * h * H)

        n, m = len(ref_boxes), len(pred_boxes)
        if n == 0 or n <= m:
            continue  # "missing" (count-based) only occurs when ref has more boxes than pred

        cost = np.zeros((n, m)) if m > 0 else np.zeros((n, 0))
        for i, rb in enumerate(ref_boxes):
            for j, pb in enumerate(pred_boxes):
                cost[i, j] = 1.0 - iou(rb, pb)

        if m == 0:
            unmatched_ref_idx = set(range(n))
        else:
            row_ind, _ = linear_sum_assignment(cost)
            unmatched_ref_idx = set(range(n)) - set(row_ind)

        for i, (cls, (cx, cy, w, h)) in enumerate(ref_labeled):
            if i in unmatched_ref_idx:
                area = w * W * h * H
                missed_areas.append(area)
                missed_by_class[cls] = missed_by_class.get(cls, 0) + 1

    median_ref = float(np.median(all_ref_areas)) if all_ref_areas else float("nan")
    median_missed = float(np.median(missed_areas)) if missed_areas else float("nan")

    print(f"\n=== {args.pred} — boîtes de référence manquées (comptage pur, n_ref > n_pred) ===")
    print(f"Total boîtes ref             : {len(all_ref_areas)}")
    print(f"Total boîtes manquées        : {len(missed_areas)}")
    print(f"Médiane aire boîte ref       : {median_ref:.0f} px²  (≈ {median_ref**0.5:.0f}×{median_ref**0.5:.0f})")
    print(f"Médiane aire boîte manquée   : {median_missed:.0f} px²  (≈ {median_missed**0.5:.0f}×{median_missed**0.5:.0f})")
    if median_missed == median_missed and median_ref:  # not NaN
        print(f"Ratio (ref / manquée)        : {median_ref / median_missed:.1f}x plus petite")
    print("\nRépartition par classe des boîtes manquées :")
    for cls, count in sorted(missed_by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {CLASS_NAMES.get(cls, cls):12s} (id={cls}) : {count}")
    print()


if __name__ == "__main__":
    main()
