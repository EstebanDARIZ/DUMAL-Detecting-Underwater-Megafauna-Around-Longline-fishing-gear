#!/usr/bin/env python3
"""
Compute Cohen's Kappa between the reference labels (Annotator_Ref, see
kappa_ground_truth.csv) and the expert's answers (the domain expert,
expert_answers.xlsx) on the Kappa study's per-box classification task.

Pairing: (anon_id, box_number) common to both sources.

Decoding the expert's answers (box_1..box_8 in the xlsx):
  - numeric value ("0".."8")    -> class_id as-is
  - "X"  (beyond n_boxes)       -> excluded (not an answer)
  - "?"  (expert doesn't know)  -> excluded
  - "10" (expert code for "everted stomach / stress reaction", not a species)
                                  -> excluded
  - "*"  (plastic lure mimicking a mackerel, per the user's clarification)
                                  -> decoded as class_id 8 (Mackerel)

Kappa is computed via scikit-learn (sklearn.metrics.cohen_kappa_score).

Usage:
    python3 compute_kappa.py \
        --ground-truth /path/to/kappa_ground_truth.csv \
        --expert-xlsx /path/to/expert_answers.xlsx
"""

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CLASS_NAMES = {
    0: "Squid", 1: "Sardine", 2: "Ray", 3: "Sunfish", 4: "Pilot Fish",
    5: "Shark", 6: "JellyFish", 7: "Tuna", 8: "Mackerel",
}


def col_to_idx(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx_rows(path: Path, sheet: str = "xl/worksheets/sheet1.xml"):
    """Minimal .xlsx reader (zip + XML), no external dependency."""
    with zipfile.ZipFile(path) as z:
        shared = []
        with z.open("xl/sharedStrings.xml") as f:
            root = ET.parse(f).getroot()
            for si in root.findall(f"{NS}si"):
                texts = si.findall(f".//{NS}t")
                shared.append("".join(t.text or "" for t in texts))

        with z.open(sheet) as f:
            root = ET.parse(f).getroot()

        rows = []
        for row in root.findall(f".//{NS}row"):
            row_vals = {}
            for c in row.findall(f"{NS}c"):
                ref = c.attrib.get("r", "")
                ctype = c.attrib.get("t")
                v_el = c.find(f"{NS}v")
                val = shared[int(v_el.text)] if (ctype == "s" and v_el is not None) else (v_el.text if v_el is not None else "")
                row_vals[col_to_idx(ref)] = val
            if row_vals:
                maxc = max(row_vals)
                rows.append([row_vals.get(i, "") for i in range(maxc + 1)])
    return rows


def load_ground_truth(path: Path) -> dict:
    """(anon_id, box_number) -> class_id (int), from kappa_ground_truth.csv."""
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["anon_id"], int(row["box_number"]))
            gt[key] = int(row["class_id"])
    return gt


def decode_expert_value(raw: str):
    """Returns a class_id (int), or None if the answer should be excluded."""
    raw = (raw or "").strip()
    if raw in ("", "X", "x", "?"):
        return None
    if raw == "*":
        return 8  # lure mimicking a mackerel -> Mackerel
    if raw == "10":
        return None  # code for "everted stomach / stress", not a species
    try:
        return int(raw)
    except ValueError:
        return None


def load_expert_answers(path: Path) -> dict:
    """(anon_id, box_number) -> class_id (int) or None, from expert_answers.xlsx."""
    rows = read_xlsx_rows(path)
    header = rows[0]
    box_cols = [i for i, h in enumerate(header) if isinstance(h, str) and h.startswith("box_")]

    answers = {}
    for row in rows[1:]:
        anon_id = row[0] if len(row) > 0 else ""
        if not anon_id or not anon_id.startswith("img_"):
            continue
        for box_idx, col in enumerate(box_cols, start=1):
            raw = row[col] if col < len(row) else ""
            answers[(anon_id, box_idx)] = decode_expert_value(raw)
    return answers




def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", required=True, help="Path to kappa_ground_truth.csv")
    parser.add_argument("--expert-xlsx", required=True, help="Path to the domain expert's answers xlsx (e.g. expert_answers.xlsx)")
    parser.add_argument("--confusion-png", required=True,
                        help="Chemin de sortie pour la matrice de confusion en PNG (vide pour desactiver)")
    args = parser.parse_args()

    gt = load_ground_truth(Path(args.ground_truth))
    expert = load_expert_answers(Path(args.expert_xlsx))

    ref_labels, expert_labels = [], []
    excluded_no_expert_answer = 0
    excluded_no_gt = 0

    for key, gt_class in gt.items():
        if key not in expert:
            excluded_no_gt += 1  # shouldn't happen if the two files are aligned
            continue
        exp_class = expert[key]
        if exp_class is None:
            excluded_no_expert_answer += 1
            continue
        ref_labels.append(gt_class)
        expert_labels.append(exp_class)

    print(f"Paires (anon_id, box) au total dans le ground truth : {len(gt)}")
    print(f"  exclues (réponse expert = '?'/'10'/vide)          : {excluded_no_expert_answer}")
    print(f"  exclues (boîte absente des réponses expert)       : {excluded_no_gt}")
    print(f"  paires utilisées pour le Kappa                    : {len(ref_labels)}")

    observed_agreement = sum(1 for a, b in zip(ref_labels, expert_labels) if a == b) / len(ref_labels)
    kappa = cohen_kappa_score(ref_labels, expert_labels)
    print(f"\nAccord observe          : {observed_agreement:.4f}")
    print(f"Cohen's Kappa (sklearn) : {kappa:.4f}")

    print("\n=== Matrice de confusion (lignes = Annotator_Ref, colonnes = expert) ===")
    classes = sorted(set(ref_labels) | set(expert_labels))
    cm = confusion_matrix(ref_labels, expert_labels, labels=classes)
    header = "        " + "".join(f"{CLASS_NAMES.get(c, c):>10}" for c in classes)
    print(header)
    for a, row in zip(classes, cm):
        row_str = f"{CLASS_NAMES.get(a, a):<8}" + "".join(f"{v:>10}" for v in row)
        print(row_str)

    if args.confusion_png:
        save_confusion_png(cm, classes, kappa, Path(args.confusion_png))
        print(f"\nMatrice de confusion sauvegardee : {args.confusion_png}")


def save_confusion_png(cm, classes, kappa, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [CLASS_NAMES.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Annotator")
    ax.set_title(f"Confusion matrix — Cohen's Kappa = {kappa:.3f}")

    vmax = cm.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            color = "white" if v > vmax / 2 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Number of boxes")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
