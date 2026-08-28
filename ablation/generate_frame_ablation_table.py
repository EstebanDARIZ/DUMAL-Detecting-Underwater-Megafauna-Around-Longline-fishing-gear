#!/usr/bin/env python3
"""
Average the multiple training runs of the frame-spacing ablation (N=2..5)
for D-FINE-X and Boosting R-CNN, and print the corresponding LaTeX table
(tab:frame-ablation-results) with the standard "max per column, per model"
bolding convention.

Expects the evaluation results produced by benchmark/run_eval.py, one JSON
file per run, each containing at least "mAP" and "per_class_AP".

Usage:
    python3 generate_frame_ablation_table.py --results-dir /path/to/results
"""

import argparse
import json
import statistics
from pathlib import Path

CLASSES = ["Ray", "Tuna", "Shark", "Sunfish"]

# Run files per (model, configuration). Baselines have a single run.
RUN_GROUPS = {
    "D-FINE-X": {
        "100\\%": ["dfine_x"],
        "N=2": ["dfine_x_ablation_frame_n2", "dfine_x_ablation_frame_n2_h100", "dfine_x_ablation_frame_n2_run2"],
        "N=3": ["dfine_x_ablation_frame_n3", "dfine_x_ablation_frame_n3_h100", "dfine_x_ablation_frame_n3_run2"],
        "N=4": ["dfine_x_ablation_frame_n4", "dfine_x_ablation_frame_n4_h100", "dfine_x_ablation_frame_n4_run2"],
        "N=5": ["dfine_x_ablation_frame_n5_run1", "dfine_x_ablation_frame_n5_run2", "dfine_x_ablation_frame_n5_run3"],
    },
    "Boosting R-CNN": {
        "100\\%": ["boosting_rcnn"],
        "N=2": ["boosting_rcnn_ablation_frame_n2", "boosting_rcnn_ablation_frame_n2_run2", "boosting_rcnn_ablation_frame_n2_run3"],
        "N=3": ["boosting_rcnn_ablation_frame_n3", "boosting_rcnn_ablation_frame_n3_run2", "boosting_rcnn_ablation_frame_n3_run3"],
        "N=4": ["boosting_rcnn_ablation_frame_n4", "boosting_rcnn_ablation_frame_n4_run2", "boosting_rcnn_ablation_frame_n4_run3"],
        "N=5": ["boosting_rcnn_ablation_frame_n5", "boosting_rcnn_ablation_frame_n5_run2", "boosting_rcnn_ablation_frame_n5_run3"],
    },
}


def load(results_dir: Path, name: str) -> dict:
    return json.loads((results_dir / f"{name}.json").read_text())


def average_runs(results_dir: Path, run_files: list[str]) -> dict:
    runs = [load(results_dir, f) for f in run_files]
    avg = {"mAP": statistics.mean(r["mAP"] for r in runs)}
    avg["per_class_AP"] = {c: statistics.mean(r["per_class_AP"][c] for r in runs) for c in CLASSES}
    avg["n_runs"] = len(runs)
    avg["mAP_pstdev"] = statistics.pstdev(r["mAP"] for r in runs) if len(runs) > 1 else 0.0
    return avg


def format_table(results_dir: Path) -> str:
    cols = ["mAP"] + CLASSES
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Effect of frame spacing (N=2, N=3, N=4, N=5) on overall mAP and on the four "
        "target species' AP, compared to the full training set. Results are averaged over three "
        "independent training runs per configuration.}",
        "\\label{tab:frame-ablation-results}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Spacing & mAP & Ray & Tuna & Shark & Sunfish \\\\",
        "\\midrule",
    ]

    for model_idx, (model, configs) in enumerate(RUN_GROUPS.items()):
        rows = {cfg: average_runs(results_dir, files) for cfg, files in configs.items()}
        maxvals = {c: max(r["mAP"] if c == "mAP" else r["per_class_AP"][c] for r in rows.values()) for c in cols}

        for cfg, r in rows.items():
            cells = []
            for c in cols:
                v = r["mAP"] if c == "mAP" else r["per_class_AP"][c]
                s = f"{v:.3f}"
                if abs(v - maxvals[c]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            n_runs_note = "" if r["n_runs"] > 1 else "  % single run"
            lines.append(f"{model:<15} & {cfg:<6} & " + " & ".join(cells) + f" \\\\{n_runs_note}")

        if model_idx < len(RUN_GROUPS) - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, help="Directory containing the per-run evaluation JSON files (benchmark/run_eval.py output)")
    args = parser.parse_args()

    print(format_table(Path(args.results_dir)))


if __name__ == "__main__":
    main()
