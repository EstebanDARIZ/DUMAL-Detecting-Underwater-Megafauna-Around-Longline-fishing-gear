#!/usr/bin/env python3
"""
Splits res_sam/ runs into test/ and train/ at the root of a new dataset,
based on the `for_testing` column of a CSV exported from
test_pool_selector.html (e.g. split_test_train.csv).

Only the images/ and labels/ subfolders of each run are copied.

The {res-sam}/background/ folder (unannotated frames, named
<run>_bg_<n>.ext) is split frame by frame: frames whose source run is in
test go to test/background/, the rest go to train/background/, so that no
background from a test video leaks into train.

Usage:
    python3 split_test_train_folders.py \
        --dataset-out /path/to/dataset_sam_3.1 \
        --split-csv /path/to/split_test_train.csv \
        [--res-sam /path/to/res_sam]
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

BG_NAME_RE = re.compile(r"^(?P<run>.+)_bg_\d+\.\w+$")


def read_run_splits(csv_path: Path) -> dict[str, bool]:
    """Returns {run_name: is_test} by expanding the 'runs' column (which can
    contain multiple names separated by commas) and deduplicating rows
    repeated per class."""
    run_is_test: dict[str, bool] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs_field = (row.get("runs") or "").strip()
            if not runs_field:
                continue
            for_testing = (row.get("for_testing") or "").strip()
            is_test = for_testing.lower().startswith("x")

            for run_name in [r.strip() for r in runs_field.split(",") if r.strip()]:
                if run_name in run_is_test and run_is_test[run_name] != is_test:
                    print(f"  [WARN] {run_name}: for_testing incohérent entre lignes du CSV "
                          f"({run_is_test[run_name]} vs {is_test}) — on garde {run_is_test[run_name]}")
                    continue
                run_is_test[run_name] = is_test

    return run_is_test


def resolve_run_dir(res_sam: Path, run_name: str) -> Path | None:
    """Finds a run's source folder: either res_sam/{run_name} directly, or
    res_sam/*/{run_name} for merged groups (e.g. Ray_04/Ray_04_1)."""
    direct = res_sam / run_name
    if direct.is_dir():
        return direct

    for candidate in res_sam.glob(f"*/{run_name}"):
        if candidate.is_dir():
            return candidate

    return None


def copy_images_labels(src_run_dir: Path, dst_run_dir: Path) -> None:
    for sub in ("images", "labels"):
        src_sub = src_run_dir / sub
        if not src_sub.is_dir():
            print(f"    [WARN] {src_run_dir.name}: pas de sous-dossier {sub}/")
            continue
        shutil.copytree(src_sub, dst_run_dir / sub)


def route_background_frames(background_dir: Path, dataset_out: Path, run_is_test: dict[str, bool]) -> None:
    images_dir = background_dir / "images"
    labels_dir = background_dir / "labels"

    if not images_dir.is_dir():
        print(f"  [WARN] pas de sous-dossier images/ dans {background_dir}")
        return

    n_test = n_train = 0

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue

        match = BG_NAME_RE.match(img_path.name)
        if not match:
            print(f"    [WARN] nom de fichier inattendu, ignoré : {img_path.name}")
            continue

        run_name = match.group("run")
        is_test = run_is_test.get(run_name)
        if is_test is None:
            print(f"    [WARN] {img_path.name}: run '{run_name}' absent du CSV — envoyé en train par défaut")
            is_test = False

        split = "test" if is_test else "train"
        dst_images = dataset_out / split / "background" / "images"
        dst_labels = dataset_out / split / "background" / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dst_images / img_path.name)

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_labels / lbl_path.name)
        else:
            (dst_labels / f"{img_path.stem}.txt").touch()

        if is_test:
            n_test += 1
        else:
            n_train += 1

    print(f"\nBackground — test: {n_test}  train: {n_train}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-out", required=True, help="Dossier du nouveau dataset (contiendra train/ et test/)")
    parser.add_argument("--split-csv", required=True, help="CSV avec les colonnes runs/for_testing")
    parser.add_argument("--res-sam", required=True, help="Racine des runs sources")
    args = parser.parse_args()

    dataset_out = Path(args.dataset_out).resolve()
    split_csv = Path(args.split_csv).resolve()
    res_sam = Path(args.res_sam).resolve()
    background_dir = res_sam / "background"

    if not split_csv.is_file():
        raise FileNotFoundError(f"CSV introuvable : {split_csv}")
    if not res_sam.is_dir():
        raise FileNotFoundError(f"res_sam introuvable : {res_sam}")

    run_is_test = read_run_splits(split_csv)
    print(f"{len(run_is_test)} runs listés dans le CSV.\n")

    (dataset_out / "test").mkdir(parents=True, exist_ok=True)
    (dataset_out / "train").mkdir(parents=True, exist_ok=True)

    n_test = n_train = n_missing = 0

    for run_name, is_test in sorted(run_is_test.items()):
        src_run_dir = resolve_run_dir(res_sam, run_name)
        if src_run_dir is None:
            print(f"  [MISSING] {run_name} — introuvable sous {res_sam}")
            n_missing += 1
            continue

        split = "test" if is_test else "train"
        dst_run_dir = dataset_out / split / run_name

        if dst_run_dir.exists():
            print(f"  [SKIP] {run_name} — {dst_run_dir} existe déjà")
            continue

        print(f"  [{split.upper():5s}] {run_name}  ({src_run_dir})")
        copy_images_labels(src_run_dir, dst_run_dir)

        if is_test:
            n_test += 1
        else:
            n_train += 1

    print("\n" + "-" * 50)
    print(f"Runs copiés en test  : {n_test}")
    print(f"Runs copiés en train : {n_train}")
    print(f"Runs introuvables    : {n_missing}")
    print("-" * 50)

    if not background_dir.is_dir():
        print(f"\n[WARN] dossier background introuvable, ignoré : {background_dir}")
    else:
        print(f"\nTraitement de {background_dir}...")
        route_background_frames(background_dir, dataset_out, run_is_test)

    print(f"\nSortie : {dataset_out}")


if __name__ == "__main__":
    main()
