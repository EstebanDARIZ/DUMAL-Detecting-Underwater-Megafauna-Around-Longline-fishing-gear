#!/usr/bin/env python3
"""
Aggregate the IoU IAA between Annotator_Ref (reference) and several annotators.

For each annotator, computes the per-image Hungarian matching (same as
compute_iou_iaa.py). Then pools all ref-vs-annotator pairs by source video
(union across all annotators) and runs a cluster bootstrap on that union:
this is the main score to report, not the simple average of individual CIs.

Also computes a pure COUNTING metric (independent of IoU): the difference
between the number of boxes drawn by the annotator and the number of
reference boxes, per image. This isolates detection misses (the human eye
missing a fish) from localization quality (an imprecise box placement),
since both are mixed together in the classic "Detection + Localization"
score.

Usage:
    python IAA/aggregate_iou_iaa.py --ref Annotator_Ref --pred Annotator_A Annotator_B Annotator_C Annotator_D
"""

import argparse
import csv
import numpy as np
from pathlib import Path

from compute_iou_iaa import (
    load_manifest, read_boxes, match_boxes, bootstrap_ci, cluster_bootstrap_ci,
    DEFAULT_MANIFEST,
)

IOU_DIR = Path("./IAA/IoU")


def compute_pair(ref_dir: Path, pred_dir: Path, anon_to_video: dict,
                  iou_threshold: float, loc_threshold_loose: float):
    video_det, video_strict, video_loose = {}, {}, {}
    video_ref_count, video_missing, video_extra = {}, {}, {}
    n_ref_only = n_pred_only = n_both = 0
    for ref_path in sorted(ref_dir.glob("*.txt")):
        stem = ref_path.stem
        pred_path = pred_dir / f"{stem}.txt"
        ref_boxes, pred_boxes = read_boxes(ref_path), read_boxes(pred_path)
        if not ref_boxes and not pred_boxes:
            continue
        if not pred_boxes:
            n_ref_only += 1
        elif not ref_boxes:
            n_pred_only += 1
        else:
            n_both += 1
        det, strict, loose = match_boxes(ref_boxes, pred_boxes,
                                          iou_threshold=iou_threshold,
                                          loc_threshold_loose=loc_threshold_loose)
        vid = anon_to_video.get(stem, stem)
        video_det.setdefault(vid, []).extend(det)
        video_strict.setdefault(vid, []).extend(strict)
        video_loose.setdefault(vid, []).extend(loose)

        n_ref, n_pred = len(ref_boxes), len(pred_boxes)
        video_ref_count.setdefault(vid, []).append(n_ref)
        video_missing.setdefault(vid, []).append(max(0, n_ref - n_pred))
        video_extra.setdefault(vid, []).append(max(0, n_pred - n_ref))
    count_stats = (video_ref_count, video_missing, video_extra)
    return video_det, video_strict, video_loose, (n_both, n_ref_only, n_pred_only), count_stats


def cluster_ratio_bootstrap_ci(num_clusters: dict, denom_clusters: dict,
                                n_boot=10_000, alpha=0.05, rng=None):
    """Cluster bootstrap CI on a ratio sum(num)/sum(denom), resampled by video."""
    rng = rng or np.random.default_rng(42)
    keys = list(denom_clusters.keys())
    n = len(keys)
    num_arr = np.array([sum(num_clusters.get(k, [])) for k in keys])
    den_arr = np.array([sum(denom_clusters[k]) for k in keys])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        num_sum, den_sum = num_arr[idx].sum(), den_arr[idx].sum()
        stats.append(num_sum / den_sum if den_sum > 0 else 0.0)
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="Annotator_Ref")
    parser.add_argument("--pred", nargs="+", default=["Annotator_A", "Annotator_B", "Annotator_C", "Annotator_D"])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--loc-threshold-loose", type=float, default=0.1)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Fichier markdown de sortie")
    args = parser.parse_args()

    ref_dir = IOU_DIR / args.ref
    anon_to_video = load_manifest(Path(args.manifest))
    rng = np.random.default_rng(args.seed)

    pooled_det: dict[str, list] = {}
    pooled_strict: dict[str, list] = {}
    pooled_loose: dict[str, list] = {}
    pooled_ref_count: dict[str, list] = {}
    pooled_missing: dict[str, list] = {}
    pooled_extra: dict[str, list] = {}

    per_pair_rows = []

    for ann in args.pred:
        pred_dir = IOU_DIR / ann
        vdet, vstrict, vloose, counts, (vref, vmiss, vextra) = compute_pair(
            ref_dir, pred_dir, anon_to_video, args.iou_threshold, args.loc_threshold_loose)

        total_ref = sum(v for lst in vref.values() for v in lst)
        total_missing = sum(v for lst in vmiss.values() for v in lst)
        total_extra = sum(v for lst in vextra.values() for v in lst)
        miss_rate = total_missing / total_ref if total_ref else 0.0
        extra_rate = total_extra / total_ref if total_ref else 0.0
        miss_ci = cluster_ratio_bootstrap_ci(vmiss, vref, n_boot=args.n_boot, rng=rng)
        extra_ci = cluster_ratio_bootstrap_ci(vextra, vref, n_boot=args.n_boot, rng=rng)

        det_arr = np.array([v for lst in vdet.values() for v in lst])
        strict_arr = np.array([v for lst in vstrict.values() for v in lst])
        loose_arr = np.array([v for lst in vloose.values() for v in lst])
        det_clusters = [np.array(v) for v in vdet.values()]
        strict_clusters = [np.array(v) for v in vstrict.values() if v]
        loose_clusters = [np.array(v) for v in vloose.values() if v]

        det_ci = cluster_bootstrap_ci(det_clusters, n_boot=args.n_boot, rng=rng)
        strict_ci = cluster_bootstrap_ci(strict_clusters, n_boot=args.n_boot, rng=rng)
        loose_ci = cluster_bootstrap_ci(loose_clusters, n_boot=args.n_boot, rng=rng)

        per_pair_rows.append({
            "annotateur": ann,
            "n_both": counts[0], "n_ref_only": counts[1], "n_pred_only": counts[2],
            "n_pairs_det": len(det_arr), "iou_det": det_arr.mean(), "ci_det": det_ci,
            "n_pairs_strict": len(strict_arr), "iou_strict": strict_arr.mean(), "ci_strict": strict_ci,
            "n_pairs_loose": len(loose_arr), "iou_loose": loose_arr.mean(), "ci_loose": loose_ci,
            "total_ref": total_ref, "total_missing": total_missing, "total_extra": total_extra,
            "miss_rate": miss_rate, "miss_ci": miss_ci,
            "extra_rate": extra_rate, "extra_ci": extra_ci,
        })

        for vid, lst in vdet.items():
            pooled_det.setdefault(vid, []).extend(lst)
        for vid, lst in vstrict.items():
            pooled_strict.setdefault(vid, []).extend(lst)
        for vid, lst in vloose.items():
            pooled_loose.setdefault(vid, []).extend(lst)
        for vid, lst in vref.items():
            pooled_ref_count.setdefault(vid, []).extend(lst)
        for vid, lst in vmiss.items():
            pooled_missing.setdefault(vid, []).extend(lst)
        for vid, lst in vextra.items():
            pooled_extra.setdefault(vid, []).extend(lst)

    pooled_det_arr = np.array([v for lst in pooled_det.values() for v in lst])
    pooled_strict_arr = np.array([v for lst in pooled_strict.values() for v in lst])
    pooled_loose_arr = np.array([v for lst in pooled_loose.values() for v in lst])
    pooled_det_clusters = [np.array(v) for v in pooled_det.values()]
    pooled_strict_clusters = [np.array(v) for v in pooled_strict.values() if v]
    pooled_loose_clusters = [np.array(v) for v in pooled_loose.values() if v]

    pooled_det_ci = cluster_bootstrap_ci(pooled_det_clusters, n_boot=args.n_boot, rng=rng)
    pooled_strict_ci = cluster_bootstrap_ci(pooled_strict_clusters, n_boot=args.n_boot, rng=rng)
    pooled_loose_ci = cluster_bootstrap_ci(pooled_loose_clusters, n_boot=args.n_boot, rng=rng)

    pooled_total_ref = sum(v for lst in pooled_ref_count.values() for v in lst)
    pooled_total_missing = sum(v for lst in pooled_missing.values() for v in lst)
    pooled_total_extra = sum(v for lst in pooled_extra.values() for v in lst)
    pooled_miss_rate = pooled_total_missing / pooled_total_ref if pooled_total_ref else 0.0
    pooled_extra_rate = pooled_total_extra / pooled_total_ref if pooled_total_ref else 0.0
    pooled_miss_ci = cluster_ratio_bootstrap_ci(pooled_missing, pooled_ref_count, n_boot=args.n_boot, rng=rng)
    pooled_extra_ci = cluster_ratio_bootstrap_ci(pooled_extra, pooled_ref_count, n_boot=args.n_boot, rng=rng)

    simple_mean_det = np.mean([r["iou_det"] for r in per_pair_rows])
    simple_mean_strict = np.mean([r["iou_strict"] for r in per_pair_rows])
    simple_mean_loose = np.mean([r["iou_loose"] for r in per_pair_rows])

    lines = []
    lines.append(f"# Résultats IoU IAA — Annotator_Ref (référence) vs {len(args.pred)} annotateurs\n")
    lines.append(f"Matching hongrois · Bootstrap {args.n_boot} répliques · Clusters = vidéos sources · seuils strict={args.iou_threshold}, loose={args.loc_threshold_loose}\n")

    lines.append("## Résultats par paire\n")
    lines.append("| Annotateur | Images (both/ref only/pred only) | Détection+Localisation (n paires) | IoU moyen | IC 95% cluster vidéo | Localisation stricte (n) | IoU moyen | IC 95% cluster vidéo | Localisation permissive (n) | IoU moyen | IC 95% cluster vidéo |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in per_pair_rows:
        lines.append(
            f"| {r['annotateur']} | {r['n_both']}/{r['n_ref_only']}/{r['n_pred_only']} "
            f"| {r['n_pairs_det']} | {r['iou_det']:.4f} | [{r['ci_det'][0]:.4f}, {r['ci_det'][1]:.4f}] "
            f"| {r['n_pairs_strict']} | {r['iou_strict']:.4f} | [{r['ci_strict'][0]:.4f}, {r['ci_strict'][1]:.4f}] "
            f"| {r['n_pairs_loose']} | {r['iou_loose']:.4f} | [{r['ci_loose'][0]:.4f}, {r['ci_loose'][1]:.4f}] |"
        )

    lines.append("\n## Moyenne simple des IC individuels (référence, informatif)\n")
    lines.append(f"- Détection+Localisation : {simple_mean_det:.4f}")
    lines.append(f"- Localisation stricte    : {simple_mean_strict:.4f}")
    lines.append(f"- Localisation permissive : {simple_mean_loose:.4f}\n")

    lines.append("## Score agrégé (méthode principale) — pool de toutes les paires ref-vs-annotateur par vidéo, cluster bootstrap sur l'union\n")
    lines.append("| Métrique | Paires totales | IoU moyen | IC 95% cluster vidéo |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Détection + Localisation | {len(pooled_det_arr)} | {pooled_det_arr.mean():.4f} | [{pooled_det_ci[0]:.4f}, {pooled_det_ci[1]:.4f}] |")
    lines.append(f"| Localisation stricte (seuil {args.iou_threshold}) | {len(pooled_strict_arr)} | {pooled_strict_arr.mean():.4f} | [{pooled_strict_ci[0]:.4f}, {pooled_strict_ci[1]:.4f}] |")
    lines.append(f"| Localisation permissive (seuil {args.loc_threshold_loose}) | {len(pooled_loose_arr)} | {pooled_loose_arr.mean():.4f} | [{pooled_loose_ci[0]:.4f}, {pooled_loose_ci[1]:.4f}] |")

    lines.append("\n## Métrique de comptage — boîtes manquées / en trop par rapport à la ref (indépendante de l'IoU)\n")
    lines.append("Par image : `manquant = max(0, n_ref - n_pred)`, `en trop = max(0, n_pred - n_ref)`. "
                  "Isole les erreurs de comptage (poisson non vu) de la qualité de localisation (boîte mal placée). "
                  "Taux = total / total de boîtes ref. IC 95 % cluster bootstrap (vidéos).\n")
    lines.append("| Annotateur | Boîtes ref | Boîtes manquées | Taux de manque | IC 95% | Boîtes en trop | Taux en trop | IC 95% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in per_pair_rows:
        lines.append(
            f"| {r['annotateur']} | {r['total_ref']} | {r['total_missing']} | {r['miss_rate']*100:.1f}% "
            f"| [{r['miss_ci'][0]*100:.1f}%, {r['miss_ci'][1]*100:.1f}%] "
            f"| {r['total_extra']} | {r['extra_rate']*100:.1f}% "
            f"| [{r['extra_ci'][0]*100:.1f}%, {r['extra_ci'][1]*100:.1f}%] |"
        )
    lines.append(
        f"| **Pool 4 annotateurs** | {pooled_total_ref} | {pooled_total_missing} | {pooled_miss_rate*100:.1f}% "
        f"| [{pooled_miss_ci[0]*100:.1f}%, {pooled_miss_ci[1]*100:.1f}%] "
        f"| {pooled_total_extra} | {pooled_extra_rate*100:.1f}% "
        f"| [{pooled_extra_ci[0]*100:.1f}%, {pooled_extra_ci[1]*100:.1f}%] |"
    )
    lines.append(
        "\nÀ comparer au score de localisation pure (seuil {:.1f}) = {:.3f} — un taux de manque élevé combiné "
        "à un IoU de localisation élevé confirme que le désaccord vient de boîtes non dessinées "
        "(limite de perception), pas d'un mauvais placement des boîtes existantes.".format(
            args.iou_threshold, pooled_strict_arr.mean())
    )

    text = "\n".join(lines) + "\n"
    print(text)

    if args.out:
        Path(args.out).write_text(text)
        print(f"\n[écrit] {args.out}")


if __name__ == "__main__":
    main()
