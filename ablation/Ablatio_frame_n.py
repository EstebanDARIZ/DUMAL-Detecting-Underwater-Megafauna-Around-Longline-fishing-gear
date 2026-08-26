#!/usr/bin/env python3
"""
Frame-level ablation study, UNIFORM version (same N for every species),
run on dataset_sam_3.0.

Unlike select_frames_by_class.py (designed for a future version where N
varies per species), this version completely ignores frame content (which
classes are annotated in it). Since N is the same for everyone here, there's
no reason to track a separate threshold per class, doing so actually caused
a bug: two adjacent frames decided by two different classes (each with its
own independent threshold) could both end up kept, which defeats the whole
point of the experiment (never keep two redundant frames next to each
other).

Here: a single threshold per video, anchored on the frame number. A frame
is kept iff its number >= (last kept number for this video) + N. This
absorbs gaps in numbering by advancing the threshold to the actual value,
and guarantees that no two kept frames are ever closer than N apart in
number, regardless of content.

Computed over train+val combined per video (a video can have frames in
both splits, so the threshold must not reset at the split boundary), then
written back out separately per split. Test is never touched.

Usage:
    python3 Ablatio_frame_n.py --n 2 --tag N2
    python3 Ablatio_frame_n.py --n 3 --tag N3
    python3 Ablatio_frame_n.py --n 4 --tag N4
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

FRAME_RE = re.compile(r"^(.+?)_(?:frame|bg)_(\d+)\.")


def load_split(dataset_root: Path, split: str):
    return json.loads((dataset_root / "annotations" / f"instances_{split}.json").read_text())


def select_kept_files_uniform(dataset_root: Path, N: int):
    """Single threshold per video, anchored on frame number. Ignores frame
    content (classes) entirely."""
    all_files = []
    for split in ("train", "val"):
        d = load_split(dataset_root, split)
        all_files.extend(im["file_name"] for im in d["images"])

    run_frames = defaultdict(list)
    for fname in all_files:
        m = FRAME_RE.match(fname)
        if not m:
            continue
        run_frames[m.group(1)].append((int(m.group(2)), fname))

    kept_files = set()
    for frames in run_frames.values():
        frames.sort()
        last_kept = None
        for num, fname in frames:
            if last_kept is None or num >= last_kept + N:
                kept_files.add(fname)
                last_kept = num

    return kept_files


def write_filtered_split(dataset_root: Path, split: str, kept_files: set, out_path: Path):
    d = load_split(dataset_root, split)
    kept_images = [im for im in d["images"] if im["file_name"] in kept_files]
    kept_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in d["annotations"] if a["image_id"] in kept_ids]

    out = {"images": kept_images, "annotations": kept_anns, "categories": d["categories"]}
    out_path.write_text(json.dumps(out))
    return len(kept_images), len(d["images"]), len(kept_anns), len(d["annotations"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, required=True, help="Espacement minimal (en numero de frame) entre deux frames gardees")
    parser.add_argument("--tag", default=None, help="Suffixe des fichiers de sortie (defaut: N<n>)")
    args = parser.parse_args()

    if args.n < 1:
        raise ValueError(f"N doit etre >= 1 (recu {args.n})")

    dataset_root = Path(args.dataset_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"N{args.n}"

    kept_files = select_kept_files_uniform(dataset_root, args.n)
    print(f"N={args.n} (uniforme, sans distinction de classe)")
    print(f"Frames retenues (train+val) : {len(kept_files)}")

    for split in ("train", "val"):
        out_path = output / f"instances_{split}_{tag}.json"
        n_img_kept, n_img_total, n_ann_kept, n_ann_total = write_filtered_split(dataset_root, split, kept_files, out_path)
        print(f"  {split}: {n_img_kept}/{n_img_total} images ({100*n_img_kept/n_img_total:.1f}%), "
              f"{n_ann_kept}/{n_ann_total} annotations -> {out_path}")


if __name__ == "__main__":
    main()
