#!/usr/bin/env python3
"""
Compute inter-annotator agreement (IAA) IoU between two folders of YOLO labels.

For each image:
  - Boxes from the two annotators are matched optimally (Hungarian
    algorithm), maximizing per-pair IoU.
  - Unmatched boxes (when the two annotators drew different counts)
    contribute an IoU of 0.

Metrics produced:
  - Overall mean IoU
  - 95% standard bootstrap CI  (resampling individual pairs)
  - 95% cluster bootstrap CI  (resampling VIDEOS, which respects
    intra-video correlation: two frames from the same run share the same
    animal, turbidity, and lighting)

The manifest (--manifest) maps each anon_id to its source video; images
missing from the manifest are treated as their own separate video (fallback).

Usage:
    python IAA/compute_iou_iaa.py \\
        --ref  /path/to/Annotator_Ref \\
        --pred /path/to/Annotator_B
"""

import argparse
import csv
import re
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

DEFAULT_MANIFEST = "./IAA/IoU/selection_manifest.csv"


def base_video_name(video_path_str: str) -> str:
    stem = Path(video_path_str).stem
    return re.sub(r'(_sub_?clips?(_\d+)?|_event(_\d+)?)$', '', stem, flags=re.IGNORECASE)


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Return mapping anon_id → base_video_name (e.g. 'img_017' → 'run_66_2024...')."""
    if not manifest_path.exists():
        return {}
    mapping = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            vid = row.get("video_path", "") or row.get("folder", row["anon_id"])
            mapping[row["anon_id"]] = base_video_name(vid) if vid else row["anon_id"]
    return mapping


# ── IoU ───────────────────────────────────────────────────────────────────────

def iou(b1, b2):
    """IoU between two normalized YOLO boxes (cx, cy, w, h)."""
    cx1, cy1, w1, h1 = b1
    cx2, cy2, w2, h2 = b2
    ax1, ay1 = cx1 - w1/2, cy1 - h1/2
    ax2, ay2 = cx1 + w1/2, cy1 + h1/2
    bx1, by1 = cx2 - w2/2, cy2 - h2/2
    bx2, by2 = cx2 + w2/2, cy2 + h2/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def read_boxes(path: Path) -> list[tuple]:
    """Read a YOLO label file into a list of (cx, cy, w, h)."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(tuple(float(v) for v in parts[1:5]))
    return boxes


def _raw_match(ref_boxes, pred_boxes):
    """Raw Hungarian matching: returns (pair_ious, n_without_partner).

    pair_ious         : IoU of each assigned pair (min(n, m) values).
    n_without_partner : boxes with no possible partner (= |n - m|).
    """
    n, m = len(ref_boxes), len(pred_boxes)
    if n == 0 and m == 0:
        return [], 0
    if n == 0:
        return [], m
    if m == 0:
        return [], n

    cost = np.zeros((n, m))
    for i, rb in enumerate(ref_boxes):
        for j, pb in enumerate(pred_boxes):
            cost[i, j] = 1.0 - iou(rb, pb)

    row_ind, col_ind = linear_sum_assignment(cost)
    pair_ious = [1.0 - cost[r, c] for r, c in zip(row_ind, col_ind)]
    return pair_ious, abs(n - m)


def match_boxes(ref_boxes, pred_boxes, iou_threshold=0.5, loc_threshold_loose=0.1):
    """
    Returns (det_ious, loc_ious_strict, loc_ious_loose):
      - det_ious        : accepted pairs (>= iou_threshold) plus 0.0 for each
                          rejected or unpartnered box.
                          Penalizes detection errors.
      - loc_ious_strict : IoUs of accepted pairs (>= iou_threshold) only.
                          Pure localization, strict threshold.
      - loc_ious_loose  : IoUs of accepted pairs (>= loc_threshold_loose) only.
                          Pure localization, loose threshold.
    """
    pair_ious, n_without = _raw_match(ref_boxes, pred_boxes)

    accepted  = [v for v in pair_ious if v >= iou_threshold]
    rejected  = [v for v in pair_ious if v < iou_threshold]
    det_ious        = accepted + [0.0] * (2 * len(rejected) + n_without)
    loc_ious_strict = accepted
    loc_ious_loose  = [v for v in pair_ious if v >= loc_threshold_loose]
    return det_ious, loc_ious_strict, loc_ious_loose


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot=10_000, alpha=0.05, rng=None):
    """Percentile bootstrap CI over a vector of individual values."""
    rng = rng or np.random.default_rng(42)
    stats = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return lo, hi


def cluster_bootstrap_ci(clusters: list[np.ndarray], n_boot=10_000,
                          alpha=0.05, rng=None):
    """
    Cluster bootstrap CI: resamples VIDEOS (clusters), then computes the
    mean IoU over the pooled selected boxes.
    Respects intra-video correlation (same animal, turbidity, lighting).
    """
    rng = rng or np.random.default_rng(42)
    n = len(clusters)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = np.concatenate([clusters[i] for i in idx])
        stats.append(sample.mean() if len(sample) > 0 else 0.0)
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return lo, hi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref",  required=True, help="Path to the reference annotator's YOLO label folder (e.g. .../IoU/Annotator_Ref)")
    parser.add_argument("--pred", required=True, help="Path to the compared annotator's YOLO label folder (e.g. .../IoU/Annotator_B)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="Seuil minimum pour valider un appariement (défaut: 0.5)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="CSV manifest (colonne video_path) pour clusteriser par vidéo")
    parser.add_argument("--loc-threshold-loose", type=float, default=0.1,
                        help="Seuil permissif pour la métrique localisation large (défaut: 0.1)")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--csv",    default=None, help="Sauvegarder les IoU par image dans un CSV")
    args = parser.parse_args()

    ref_dir  = Path(args.ref)
    pred_dir = Path(args.pred)
    rng = np.random.default_rng(args.seed)

    anon_to_video = load_manifest(Path(args.manifest))

    all_det_ious:   list[float] = []
    all_loc_strict: list[float] = []
    all_loc_loose:  list[float] = []
    per_image: list[dict] = []
    video_det_ious:   dict[str, list[float]] = {}
    video_loc_strict: dict[str, list[float]] = {}
    video_loc_loose:  dict[str, list[float]] = {}

    label_files = sorted(ref_dir.glob("*.txt"))
    n_ref_only = n_pred_only = n_both = 0

    for ref_path in label_files:
        stem = ref_path.stem
        pred_path = pred_dir / f"{stem}.txt"

        ref_boxes  = read_boxes(ref_path)
        pred_boxes = read_boxes(pred_path)

        if not ref_boxes and not pred_boxes:
            continue

        if not pred_boxes:
            n_ref_only += 1
        elif not ref_boxes:
            n_pred_only += 1
        else:
            n_both += 1

        det_ious, loc_strict, loc_loose = match_boxes(
            ref_boxes, pred_boxes,
            iou_threshold=args.iou_threshold,
            loc_threshold_loose=args.loc_threshold_loose,
        )

        vid_key = anon_to_video.get(stem, stem)
        video_det_ious.setdefault(vid_key,   []).extend(det_ious)
        video_loc_strict.setdefault(vid_key, []).extend(loc_strict)
        video_loc_loose.setdefault(vid_key,  []).extend(loc_loose)

        all_det_ious.extend(det_ious)
        all_loc_strict.extend(loc_strict)
        all_loc_loose.extend(loc_loose)
        per_image.append({
            "image":           stem,
            "video":           vid_key,
            "n_ref":           len(ref_boxes),
            "n_pred":          len(pred_boxes),
            "n_matched_strict": len(loc_strict),
            "n_matched_loose":  len(loc_loose),
            "mean_iou_det":    float(np.mean(det_ious))  if det_ious  else float("nan"),
            "mean_iou_strict": float(np.mean(loc_strict)) if loc_strict else float("nan"),
            "mean_iou_loose":  float(np.mean(loc_loose))  if loc_loose  else float("nan"),
        })

    if not all_det_ious:
        print("[ERROR] Aucune boîte trouvée dans les deux dossiers.")
        return

    det_clusters    = [np.array(v) for v in video_det_ious.values()]
    strict_clusters = [np.array(v) for v in video_loc_strict.values() if v]
    loose_clusters  = [np.array(v) for v in video_loc_loose.values()  if v]

    det_arr    = np.array(all_det_ious)
    strict_arr = np.array(all_loc_strict) if all_loc_strict else np.array([])
    loose_arr  = np.array(all_loc_loose)  if all_loc_loose  else np.array([])

    det_mean     = det_arr.mean()
    det_ci_std   = bootstrap_ci(det_arr, n_boot=args.n_boot, rng=rng)
    det_ci_clust = cluster_bootstrap_ci(det_clusters, n_boot=args.n_boot, rng=rng)

    def _ci_pair(arr, clusters):
        if len(arr) == 0:
            return None, None, None
        return arr.mean(), bootstrap_ci(arr, n_boot=args.n_boot, rng=rng), \
               cluster_bootstrap_ci(clusters, n_boot=args.n_boot, rng=rng)

    strict_mean, strict_ci_std, strict_ci_clust = _ci_pair(strict_arr, strict_clusters)
    loose_mean,  loose_ci_std,  loose_ci_clust  = _ci_pair(loose_arr,  loose_clusters)

    W = 56
    print(f"\n{'─'*W}")
    print(f"  Référence  : {ref_dir.name}  ({len(label_files)} images)")
    print(f"  Prédiction : {pred_dir.name}")
    print(f"{'─'*W}")
    print(f"  Images annotées par les deux  : {n_both}")
    print(f"  Images ref seulement          : {n_ref_only}")
    print(f"  Images pred seulement         : {n_pred_only}")
    print(f"  Vidéos sources (clusters)     : {len(det_clusters)}")
    print(f"{'─'*W}")
    print(f"  ── Détection + Localisation (seuil={args.iou_threshold}) ──")
    print(f"     paires non matchées → IoU=0")
    print(f"     Paires totales               : {len(all_det_ious)}")
    print(f"     IoU moyen                    : {det_mean:.4f}")
    print(f"     IC 95 % bootstrap std        : [{det_ci_std[0]:.4f}, {det_ci_std[1]:.4f}]")
    print(f"     IC 95 % cluster bootstrap    : [{det_ci_clust[0]:.4f}, {det_ci_clust[1]:.4f}]")
    print(f"{'─'*W}")
    if strict_mean is not None:
        print(f"  ── Localisation pure (seuil={args.iou_threshold}) ────────────")
        print(f"     paires non matchées exclues")
        print(f"     Paires acceptées             : {len(all_loc_strict)}")
        print(f"     IoU moyen                    : {strict_mean:.4f}")
        print(f"     IC 95 % bootstrap std        : [{strict_ci_std[0]:.4f}, {strict_ci_std[1]:.4f}]")
        print(f"     IC 95 % cluster bootstrap    : [{strict_ci_clust[0]:.4f}, {strict_ci_clust[1]:.4f}]")
        print(f"{'─'*W}")
    if loose_mean is not None:
        print(f"  ── Localisation pure (seuil={args.loc_threshold_loose}) ───────────")
        print(f"     paires non matchées exclues")
        print(f"     Paires acceptées             : {len(all_loc_loose)}")
        print(f"     IoU moyen                    : {loose_mean:.4f}")
        print(f"     IC 95 % bootstrap std        : [{loose_ci_std[0]:.4f}, {loose_ci_std[1]:.4f}]")
        print(f"     IC 95 % cluster bootstrap    : [{loose_ci_clust[0]:.4f}, {loose_ci_clust[1]:.4f}]")
        print(f"{'─'*W}")
    print()

    per_image_sorted = sorted([r for r in per_image if not np.isnan(r["mean_iou_det"])],
                               key=lambda r: r["mean_iou_det"])
    print("  5 images avec le IoU det+loc le plus bas :")
    for r in per_image_sorted[:5]:
        s = f"{r['mean_iou_strict']:.3f}" if not np.isnan(r["mean_iou_strict"]) else "—"
        l = f"{r['mean_iou_loose']:.3f}"  if not np.isnan(r["mean_iou_loose"])  else "—"
        print(f"    {r['image']}  ref={r['n_ref']} pred={r['n_pred']}"
              f"  IoU_det={r['mean_iou_det']:.3f}  loc_strict={s}  loc_loose={l}")
    print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            w.writeheader(); w.writerows(per_image)
        print(f"  Détail par image → {args.csv}\n")


if __name__ == "__main__":
    main()
