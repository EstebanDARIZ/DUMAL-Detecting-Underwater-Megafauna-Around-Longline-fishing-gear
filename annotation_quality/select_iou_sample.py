#!/usr/bin/env python3
"""
Select images for the IoU inter-annotator agreement task.

Goal: evaluate bounding-box localization quality — how similar are two
annotators' boxes for the same animal?  Species class is irrelevant here;
what matters is having a diverse set of annotated frames spread across as
many source videos as possible, with a minimum box size so tiny/ambiguous
detections don't pollute the metric.

Selection strategy:
  - For every source video that has annotated frames with at least one box
    ≥ min_box_px, pick --max-per-video evenly-spaced frames (maximising
    temporal spread inside the video).
  - A minimum inter-frame gap (--min-gap-seconds) prevents near-duplicate
    consecutive frames from the same clip.
  - No species stratification: any annotated frame qualifies.

Output (under --output):
  images/          Selected frames, anonymised (img_001.jpg …)
  Annotator_Ref/   Matching YOLO label files (reference, keep private)
  selection_manifest.csv

Usage:
    python IAA/select_iou_sample.py
    python IAA/select_iou_sample.py --max-per-video 3 --dry-run
"""

import argparse
import csv
import re
import random
import shutil
from pathlib import Path

from PIL import Image as PILImage

CLASS_NAMES = {
    0: "Squid", 1: "Sardine", 2: "Ray", 3: "Sunfish", 4: "Pilot Fish",
    5: "Shark", 6: "JellyFish", 7: "Tuna", 8: "Mackerel",
}

DEFAULT_ROOTS = [
    ("./res_sam", "train_pool"),
    ("./res_sam_video_test", "test"),
]
DEFAULT_OUTPUT = "./IAA/iou_set"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def parse_config(config_path: Path) -> dict:
    data = {}
    for line in config_path.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            data[k.strip()] = v.strip()
    return data


def base_video_name(video_path_str: str) -> str:
    stem = Path(video_path_str).stem
    return re.sub(r'(_sub_?clips?(_\d+)?|_event(_\d+)?)$', '', stem, flags=re.IGNORECASE)


def subclip_index(video_path_str: str) -> int:
    """Return the subclip/event number so subclips can be sorted chronologically."""
    m = re.search(r'(?:sub_?clips?|event)_(\d+)', video_path_str, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def read_classes(label_path: Path) -> frozenset[int]:
    if not label_path.exists():
        return frozenset()
    return frozenset(int(l.split()[0]) for l in label_path.read_text().splitlines() if l.strip())


def count_class_instances(label_path: Path, class_id: int) -> int:
    """Count boxes of class_id in a label file (no size filter — already applied at scan time)."""
    return sum(1 for l in label_path.read_text().splitlines()
               if l.strip() and int(l.split()[0]) == class_id)


def has_large_box(label_path: Path, img_w: int, img_h: int, min_px: int) -> bool:
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5 and float(parts[3]) * img_w >= min_px and float(parts[4]) * img_h >= min_px:
            return True
    return False


def frame_num(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def random_pick_with_gap(frames: list, k: int, min_gap: int, rng: random.Random) -> list:
    """Pick up to k frames randomly, removing a ±min_gap window after each pick."""
    pool = list(frames)
    picks = []
    while pool and len(picks) < k:
        chosen = rng.choice(pool)
        picks.append(chosen)
        pool = [f for f in pool
                if abs(f["abs_frame_idx"] - chosen["abs_frame_idx"]) > min_gap]
    return picks


def collect_frames(root: Path, split_label: str, min_box_px: int) -> list[dict]:
    """One entry per SOURCE VIDEO with eligible frames sorted by absolute position.

    When several run_XX folders are subclips of the same source video, their
    local frame indices (which all start at 0) are converted to absolute video
    frame indices using cumulative subclip offsets.  The offset for subclip N
    is estimated as the maximum frame index found in the preceding subclips + 1.
    Subclips are ordered by the numeric suffix in their video filename
    (subclip_1 < subclip_2 < …) before offsets are computed.
    """
    # Pass 1: collect per-folder, retaining subclip metadata
    by_src: dict[str, list[dict]] = {}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        images_dir = folder / "images"
        labels_dir = folder / "labels"
        config_path = folder / "config.txt"
        if not all(p.exists() for p in [images_dir, labels_dir, config_path]):
            continue

        cfg = parse_config(config_path)
        video_path = cfg.get("Video", "")
        src_key = base_video_name(video_path) if video_path else folder.name
        sc_idx = subclip_index(video_path)

        img_size: tuple[int, int] | None = None
        frames: list[dict] = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{p.stem}.txt"
            classes = read_classes(label_path)
            if not classes:
                continue
            if min_box_px > 0:
                if img_size is None:
                    try:
                        with PILImage.open(p) as img:
                            img_size = img.size
                    except Exception:
                        img_size = (1920, 1080)
                if not has_large_box(label_path, img_size[0], img_size[1], min_box_px):
                    continue
            frames.append({"path": p, "label_path": label_path, "classes": classes,
                           "folder_name": folder.name, "video_path": video_path,
                           "split": split_label})

        if not frames:
            continue

        by_src.setdefault(src_key, []).append({
            "folder_name": folder.name,
            "video_path": video_path,
            "split": split_label,
            "sc_idx": sc_idx,
            "frames": frames,
        })

    # Pass 2: sort subclips chronologically, assign abs_frame_idx
    entries = []
    for src_key, clip_entries in by_src.items():
        clip_entries.sort(key=lambda e: e["sc_idx"])

        cumulative_offset = 0
        all_frames: list[dict] = []
        for clip in clip_entries:
            local_frames = sorted(clip["frames"], key=lambda f: frame_num(f["path"]))
            for f in local_frames:
                all_frames.append({**f, "abs_frame_idx": frame_num(f["path"]) + cumulative_offset})
            # Estimate this subclip's length from its highest extracted frame index.
            # Using max + 1 is a lower bound — the clip may be longer, but this is
            # conservative (gaps across subclip boundaries appear smaller than they
            # really are, so we never accidentally accept frames that are too close).
            max_local = max(frame_num(f["path"]) for f in clip["frames"])
            cumulative_offset += max_local + 1

        all_frames.sort(key=lambda f: f["abs_frame_idx"])
        entries.append({
            "src_key": src_key,
            "frames": all_frames,
            "video_path": clip_entries[0]["video_path"],
            "split": clip_entries[0]["split"],
        })

    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-per-video", type=int, default=2,
                        help="Max frames per source video (default: 2)")
    parser.add_argument("--min-gap-seconds", type=float, default=10.0,
                        help="Min temporal gap between two frames from the same video (default: 10 s)")
    parser.add_argument("--min-box-pixels", type=int, default=15,
                        help="Exclude frames where no box has pixel width AND height >= this (default: 15)")
    parser.add_argument("--min-instances", type=int, default=5,
                        help="Min bounding-box instances per class after top-up (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    min_gap_frames = int(args.min_gap_seconds * 25)

    all_entries: list[dict] = []
    for root_str, split_label in DEFAULT_ROOTS:
        root = Path(root_str)
        if not root.exists():
            print(f"[SKIP] root not found: {root}")
            continue
        all_entries.extend(collect_frames(root, split_label, args.min_box_pixels))

    all_entries.sort(key=lambda e: e["src_key"])

    print(f"Source videos found: {len(all_entries)}")
    print(f"Max per video: {args.max_per_video}  |  Gap: {args.min_gap_seconds} s  |  Min box: {args.min_box_pixels} px\n")

    selected_keys: set[tuple[str, str]] = set()  # (folder_name, filename)
    selection: list[dict] = []
    for entry in all_entries:
        picks = random_pick_with_gap(entry["frames"], args.max_per_video, min_gap_frames, rng)
        for p in picks:
            selection.append(p)
            selected_keys.add((p["folder_name"], p["path"].name))

    # Coverage top-up: ensure at least --min-instances-per-class instances per class
    # Count current instances per class
    def instance_counts() -> dict[int, int]:
        counts = {c: 0 for c in CLASS_NAMES}
        for f in selection:
            for c in CLASS_NAMES:
                if c in f["classes"]:
                    counts[c] += count_class_instances(f["label_path"], c)
        return counts

    # Build flat candidate index (all eligible frames not yet selected), per class
    class_index: dict[int, list[dict]] = {c: [] for c in CLASS_NAMES}
    for entry in all_entries:
        for f in entry["frames"]:
            for c in f["classes"]:
                if c in CLASS_NAMES and (f["folder_name"], f["path"].name) not in selected_keys:
                    class_index[c].append(f)

    inst = instance_counts()
    for c in sorted(CLASS_NAMES):
        needed = args.min_instances - inst[c]
        if needed <= 0:
            continue
        added_frames, added_inst = 0, 0
        for candidate in class_index[c]:
            if needed <= 0:
                break
            key = (candidate["folder_name"], candidate["path"].name)
            if key in selected_keys:
                continue
            n = count_class_instances(candidate["label_path"], c)
            if n == 0:
                continue
            selection.append(candidate)
            selected_keys.add(key)
            needed -= n
            added_inst += n
            added_frames += 1
        if added_frames:
            print(f"  [top-up] {CLASS_NAMES[c]}: +{added_frames} frame(s), +{added_inst} instance(s)")

    # Shuffle for anonymisation
    rng.shuffle(selection)

    print(f"Total frames selected: {len(selection)}")

    # Count frames and instances per class
    final_inst = instance_counts()
    class_frame_counts = {c: 0 for c in CLASS_NAMES}
    for f in selection:
        for c in f["classes"]:
            if c in CLASS_NAMES:
                class_frame_counts[c] += 1
    print(f"\n{'Espèce':<12} {'Frames':>7} {'Instances':>10}")
    print("-" * 32)
    for c, name in CLASS_NAMES.items():
        ok = final_inst[c] >= args.min_instances
        mark = " ✓" if ok else f" ✗ ({final_inst[c]}<{args.min_instances})"
        print(f"{name:<12} {class_frame_counts[c]:>7} {final_inst[c]:>10}{mark}")

    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        return

    out_root = Path(args.output)
    images_out = out_root / "images"
    labels_out = out_root / "Annotator_Ref"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for stale in list(images_out.iterdir()) + list(labels_out.iterdir()):
        stale.unlink()

    width = len(str(len(selection)))
    manifest_rows = []
    for i, f in enumerate(selection, start=1):
        anon_id = f"img_{i:0{width}d}"
        dst_img = images_out / f"{anon_id}{f['path'].suffix.lower()}"
        shutil.copy2(f["path"], dst_img)
        dst_lbl = labels_out / f"{anon_id}.txt"
        shutil.copy2(f["label_path"], dst_lbl)
        manifest_rows.append({
            "anon_id": anon_id,
            "classes_present": "+".join(CLASS_NAMES.get(c, f"cls{c}") for c in sorted(f["classes"])),
            "split": f["split"],
            "folder": f["folder_name"],
            "frame": f["path"].name,
            "video_path": f["video_path"],
            "original_image_path": str(f["path"]),
            "original_label_path": str(f["label_path"]),
        })

    manifest_rows.sort(key=lambda r: r["anon_id"])
    manifest_path = out_root / "selection_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nImages  → {images_out}")
    print(f"Labels  → {labels_out}")
    print(f"Manifest→ {manifest_path}")


if __name__ == "__main__":
    main()
