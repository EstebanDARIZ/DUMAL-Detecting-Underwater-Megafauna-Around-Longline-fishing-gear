#!/usr/bin/env python3
"""
Human agreement ceiling: IoU between PAIRS OF TEST ANNOTATORS (excluding the
Annotator_Ref reference), for every possible combination.

Unlike aggregate_iou_iaa.py (which only compares ref-vs-annotator, the only
score that matters for the published dataset's reliability), this script
compares test annotators against each other. This gives the "human agreement
ceiling": even two motivated, correctly briefed humans don't agree 100% of
the time, so the ref-vs-annotator score shouldn't be judged against 1.0 but
against this ceiling. Reported as CONTEXT in the paper, not as the main
score.

Usage:
    python IAA/human_agreement_ceiling.py --annotators Annotator_A Annotator_B Annotator_C Annotator_D
"""

import argparse
import itertools
import numpy as np
from pathlib import Path

from compute_iou_iaa import load_manifest, cluster_bootstrap_ci, DEFAULT_MANIFEST
from aggregate_iou_iaa import compute_pair, IOU_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotators", nargs="+", default=["Annotator_A", "Annotator_B", "Annotator_C", "Annotator_D"])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--loc-threshold-loose", type=float, default=0.1)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Fichier markdown de sortie")
    args = parser.parse_args()

    anon_to_video = load_manifest(Path(args.manifest))
    rng = np.random.default_rng(args.seed)

    pooled_det, pooled_strict, pooled_loose = {}, {}, {}
    per_pair_rows = []

    for a, b in itertools.combinations(args.annotators, 2):
        dir_a, dir_b = IOU_DIR / a, IOU_DIR / b
        vdet, vstrict, vloose, counts, _ = compute_pair(
            dir_a, dir_b, anon_to_video, args.iou_threshold, args.loc_threshold_loose)

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
            "pair": f"{a} vs {b}",
            "n_both": counts[0], "n_a_only": counts[1], "n_b_only": counts[2],
            "n_pairs_det": len(det_arr), "iou_det": det_arr.mean(), "ci_det": det_ci,
            "n_pairs_strict": len(strict_arr), "iou_strict": strict_arr.mean(), "ci_strict": strict_ci,
            "n_pairs_loose": len(loose_arr), "iou_loose": loose_arr.mean(), "ci_loose": loose_ci,
        })

        for vid, lst in vdet.items():
            pooled_det.setdefault(vid, []).extend(lst)
        for vid, lst in vstrict.items():
            pooled_strict.setdefault(vid, []).extend(lst)
        for vid, lst in vloose.items():
            pooled_loose.setdefault(vid, []).extend(lst)

    pooled_det_arr = np.array([v for lst in pooled_det.values() for v in lst])
    pooled_strict_arr = np.array([v for lst in pooled_strict.values() for v in lst])
    pooled_loose_arr = np.array([v for lst in pooled_loose.values() for v in lst])
    pooled_det_clusters = [np.array(v) for v in pooled_det.values()]
    pooled_strict_clusters = [np.array(v) for v in pooled_strict.values() if v]
    pooled_loose_clusters = [np.array(v) for v in pooled_loose.values() if v]

    pooled_det_ci = cluster_bootstrap_ci(pooled_det_clusters, n_boot=args.n_boot, rng=rng)
    pooled_strict_ci = cluster_bootstrap_ci(pooled_strict_clusters, n_boot=args.n_boot, rng=rng)
    pooled_loose_ci = cluster_bootstrap_ci(pooled_loose_clusters, n_boot=args.n_boot, rng=rng)

    lines = []
    lines.append(f"# Plafond d'accord humain — paires d'annotateurs test ({len(args.annotators)} annotateurs, "
                 f"{len(list(itertools.combinations(args.annotators, 2)))} paires)\n")
    lines.append("Annotator_Ref (référence) exclu — ces paires mesurent l'accord entre humains motivés et "
                 "correctement briefés, PAS la qualité du dataset publié. À reporter en CONTEXTE dans le "
                 "papier, pas comme score principal.\n")
    lines.append(f"Matching hongrois · Bootstrap {args.n_boot} répliques · Clusters = vidéos sources · "
                 f"seuils strict={args.iou_threshold}, loose={args.loc_threshold_loose}\n")

    lines.append("## Résultats par paire\n")
    lines.append("| Paire | Images (both/A only/B only) | Détection+Localisation (n) | IoU moyen | IC 95% | Localisation stricte (n) | IoU moyen | IC 95% | Localisation permissive (n) | IoU moyen | IC 95% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in per_pair_rows:
        lines.append(
            f"| {r['pair']} | {r['n_both']}/{r['n_a_only']}/{r['n_b_only']} "
            f"| {r['n_pairs_det']} | {r['iou_det']:.4f} | [{r['ci_det'][0]:.4f}, {r['ci_det'][1]:.4f}] "
            f"| {r['n_pairs_strict']} | {r['iou_strict']:.4f} | [{r['ci_strict'][0]:.4f}, {r['ci_strict'][1]:.4f}] "
            f"| {r['n_pairs_loose']} | {r['iou_loose']:.4f} | [{r['ci_loose'][0]:.4f}, {r['ci_loose'][1]:.4f}] |"
        )

    lines.append("\n## Plafond agrégé — pool de toutes les paires annotateur-annotateur par vidéo, cluster bootstrap sur l'union\n")
    lines.append("| Métrique | Paires totales | IoU moyen | IC 95% cluster vidéo |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Détection + Localisation | {len(pooled_det_arr)} | {pooled_det_arr.mean():.4f} | [{pooled_det_ci[0]:.4f}, {pooled_det_ci[1]:.4f}] |")
    lines.append(f"| Localisation stricte (seuil {args.iou_threshold}) | {len(pooled_strict_arr)} | {pooled_strict_arr.mean():.4f} | [{pooled_strict_ci[0]:.4f}, {pooled_strict_ci[1]:.4f}] |")
    lines.append(f"| Localisation permissive (seuil {args.loc_threshold_loose}) | {len(pooled_loose_arr)} | {pooled_loose_arr.mean():.4f} | [{pooled_loose_ci[0]:.4f}, {pooled_loose_ci[1]:.4f}] |")

    text = "\n".join(lines) + "\n"
    print(text)

    if args.out:
        Path(args.out).write_text(text)
        print(f"\n[écrit] {args.out}")


if __name__ == "__main__":
    main()
