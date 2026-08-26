#!/usr/bin/env python3
"""
Select frames for the Cohen's Kappa inter-annotator agreement task.

Each selected frame is saved with ALL kappa-class bounding boxes drawn and
numbered (no species label visible). The expert annotator watches the context
clip and independently assigns a species name to each numbered box.

Excluded from target selection:
  - Classes 0 (Squid) and 1 (Sardine): bait fish, present in almost all videos
    → not targeted but drawn/counted if present in selected frames
  - Classes 6 (JellyFish) and 9: single video, identified by direct expert consensus

Target order: [2, 3, 5, 7] then [4, 8]
  Main targets first (Ray, Sunfish, Shark, Tuna), then secondary species
  (Pilot Fish, Mackerel) which often co-occur in frames selected for main targets.

Selection algorithm (per target class c):
  1. Build pool: only videos where c appears, only frames containing c (≥ 15px).
     Keep ALL eligible frames — no pre-gap filtering.
  2. Shuffle video order (seed = BASE_SEED + class_id).
  3. Round-robin across videos:
       - Draw a random frame from the current video's pool
       - Remove that frame + ±gap window from that video's pool
       - Move to next video (skip if pool empty)
       - When all videos done, start a new round
  4. Count instances of c across ALL selected frames (current + previous classes).
     If ≥ target → next class. Else → continue round-robin.

Output (under --output):
  images/                 Annotated frames with numbered boxes (img_001.jpg …)
  Annotator_Ref/          Reference YOLO labels (keep private)
  kappa_manifest.csv      One row per image
  kappa_ground_truth.csv  box_number → species per image  ⚠ do not share with expert

Usage:
    python IAA/select_kappa_sample.py
    python IAA/select_kappa_sample.py --target 50 --dry-run
"""

import argparse
import csv
import re
import random
import shutil
from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont

# All classes drawn on the freeze frame and counted in ground truth
KAPPA_CLASSES = {
    0: "Squid", 1: "Sardine", 2: "Ray", 3: "Sunfish",
    4: "Pilot Fish", 5: "Shark", 7: "Tuna", 8: "Mackerel",
}

# Order in which instance targets are pursued
# [main targets] → [secondary, likely co-occurring] → [bait fish top-up]
TARGET_ORDER = [8, 4, 2, 3, 5, 7, 0, 1]

BASE_SEED = 42

# Fast-moving species get a shorter temporal gap between selected frames
CLASS_GAP_SECONDS: dict[int, float] = {
    4: 1.0,   # Pilot Fish
    7: 1.0,   # Tuna
    8: 1.0,   # Mackerel
}

DEFAULT_ROOTS = [
    ("./res_sam",            "train_pool"),
    ("./res_sam_video_test", "test"),
]
DEFAULT_OUTPUT = "./IAA/kappa"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

BOX_COLOR = (255, 0, 0)
BOX_LW    = 3

FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default(size=size)


# ── utilities ─────────────────────────────────────────────────────────────────

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
    m = re.search(r'(?:sub_?clips?|event)_(\d+)', video_path_str, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def frame_num(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def read_kappa_boxes(label_path: Path, img_w: int, img_h: int, min_px: int) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        if cid not in KAPPA_CLASSES:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if w * img_w >= min_px and h * img_h >= min_px:
            boxes.append({"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def draw_numbered_boxes(img: PILImage.Image, boxes: list[dict]) -> PILImage.Image:
    out = img.copy()
    W, H = out.size
    draw = ImageDraw.Draw(out)
    font_size = max(32, int(H * 0.045))   # ~49px for 1080p
    font = _get_font(font_size)
    pad = 6
    for i, b in enumerate(boxes, start=1):
        x1 = max(0, int((b["cx"] - b["w"] / 2) * W))
        y1 = max(0, int((b["cy"] - b["h"] / 2) * H))
        x2 = min(W - 1, int((b["cx"] + b["w"] / 2) * W))
        y2 = min(H - 1, int((b["cy"] + b["h"] / 2) * H))
        for t in range(BOX_LW):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=BOX_COLOR)
        label = str(i)
        bb = draw.textbbox((0, 0), label, font=font)
        th = bb[3] - bb[1] + pad * 2
        lx = max(0, x1)
        ly = max(0, y1 - th)
        tx, ty = lx + pad, ly + pad - bb[1]
        # White outline for legibility against any background
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,-2),(-2,2),(2,2)]:
            draw.text((tx + dx, ty + dy), label, font=font, fill=(255, 255, 255))
        draw.text((tx, ty), label, font=font, fill=BOX_COLOR)
    return out


# ── collect eligible frames per source video ──────────────────────────────────

def collect_entries(root: Path, split_label: str, min_box_px: int) -> list[dict]:
    """One entry per source video, with all of its eligible frames."""
    by_src: dict[str, list[dict]] = {}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        images_dir  = folder / "images"
        labels_dir  = folder / "labels"
        config_path = folder / "config.txt"
        if not all(p.exists() for p in [images_dir, labels_dir, config_path]):
            continue

        cfg        = parse_config(config_path)
        video_path = cfg.get("Video", "")
        src_key    = base_video_name(video_path) if video_path else folder.name
        sc_idx     = subclip_index(video_path)

        img_size: tuple[int, int] | None = None
        frames: list[dict] = []

        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{p.stem}.txt"
            if not label_path.exists():
                continue
            if img_size is None:
                try:
                    with PILImage.open(p) as img:
                        img_size = img.size
                except Exception:
                    img_size = (1920, 1080)
            W, H = img_size
            boxes = read_kappa_boxes(label_path, W, H, min_box_px)
            if not boxes:
                continue
            inst = {}
            for b in boxes:
                inst[b["class_id"]] = inst.get(b["class_id"], 0) + 1
            frames.append({
                "path": p, "label_path": label_path, "boxes": boxes,
                "inst": inst,
                "folder_name": folder.name, "video_path": video_path,
                "split": split_label,
            })

        if not frames:
            continue
        by_src.setdefault(src_key, []).append({
            "sc_idx": sc_idx, "frames": frames,
            "video_path": video_path, "split": split_label,
        })

    entries = []
    for src_key, clips in by_src.items():
        clips.sort(key=lambda e: e["sc_idx"])
        offset = 0
        all_frames: list[dict] = []
        for clip in clips:
            local = sorted(clip["frames"], key=lambda f: frame_num(f["path"]))
            for f in local:
                all_frames.append({**f, "abs_frame_idx": frame_num(f["path"]) + offset})
            offset += max(frame_num(f["path"]) for f in clip["frames"]) + 1
        all_frames.sort(key=lambda f: f["abs_frame_idx"])
        entries.append({
            "src_key": src_key, "frames": all_frames,
            "video_path": clips[0]["video_path"], "split": clips[0]["split"],
        })
    return entries


# ── round-robin selection per class ───────────────────────────────────────────

def select_for_class(all_entries: list[dict], class_id: int, target: int,
                     all_selected: list[dict], min_gap_frames: int,
                     global_excluded: set) -> list[dict]:
    """
    Round-robin over the videos containing class_id.
    Each round draws one frame per video, then removes a ±gap window around it.
    The gap is permanent: excluded frames are added to global_excluded, so
    later classes can't draw them either.
    Stops as soon as the cumulative instance count (all_selected + new) hits target.
    """
    rng = random.Random(BASE_SEED + class_id)

    selected_keys = {(f["folder_name"], f["path"].name) for f in all_selected}
    all_excluded = selected_keys | global_excluded

    # All frames per source video (to compute the gap windows)
    video_all_frames = {entry["src_key"]: entry["frames"] for entry in all_entries}

    # Pool: only videos and frames containing class_id
    video_pools: dict[str, list[dict]] = {}
    for entry in all_entries:
        eligible = [
            f for f in entry["frames"]
            if class_id in f["inst"]
            and (f["folder_name"], f["path"].name) not in all_excluded
        ]
        if eligible:
            video_pools[entry["src_key"]] = list(eligible)

    if not video_pools:
        return []

    video_order = list(video_pools.keys())
    rng.shuffle(video_order)

    new_selections: list[dict] = []

    def count() -> int:
        return (sum(f["inst"].get(class_id, 0) for f in all_selected) +
                sum(f["inst"].get(class_id, 0) for f in new_selections))

    while count() < target:
        made_progress = False
        for vk in video_order:
            if count() >= target:
                break
            pool = video_pools.get(vk, [])
            if not pool:
                continue

            chosen = rng.choice(pool)
            chosen_idx = chosen["abs_frame_idx"]

            new_selections.append(chosen)
            made_progress = True

            # Permanently exclude the chosen frame + its ±gap window
            for f in video_all_frames.get(vk, []):
                fkey = (f["folder_name"], f["path"].name)
                if abs(f["abs_frame_idx"] - chosen_idx) <= min_gap_frames:
                    global_excluded.add(fkey)
                    all_excluded.add(fkey)

            video_pools[vk] = [
                f for f in pool
                if (f["folder_name"], f["path"].name) not in all_excluded
            ]

        if not made_progress:
            break

    return new_selections


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target",          type=int,   default=50)
    parser.add_argument("--min-gap-seconds", type=float, default=5.0)
    parser.add_argument("--min-box-pixels",  type=int,   default=15)
    parser.add_argument("--output",          default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run",         action="store_true")
    args = parser.parse_args()

    min_gap_frames = int(args.min_gap_seconds * 25)

    all_entries: list[dict] = []
    for root_str, split_label in DEFAULT_ROOTS:
        root = Path(root_str)
        if not root.exists():
            print(f"[SKIP] {root}")
            continue
        all_entries.extend(collect_entries(root, split_label, args.min_box_pixels))

    print(f"Vidéos sources : {len(all_entries)}")
    print(f"Cible : {args.target} instances  |  gap : {args.min_gap_seconds} s\n")
    print(f"Ordre target : {[KAPPA_CLASSES[c] for c in TARGET_ORDER]}\n")

    all_selected: list[dict] = []
    global_excluded: set = set()

    for class_id in TARGET_ORDER:
        gap_s = CLASS_GAP_SECONDS.get(class_id, args.min_gap_seconds)
        gap_f = int(gap_s * 25)
        before = sum(f["inst"].get(class_id, 0) for f in all_selected)
        new    = select_for_class(all_entries, class_id, args.target,
                                   all_selected, gap_f, global_excluded)
        all_selected.extend(new)
        after  = sum(f["inst"].get(class_id, 0) for f in all_selected)
        mark   = "✓" if after >= args.target else "✗ insuffisant"
        print(f"  {KAPPA_CLASSES[class_id]:<14}  "
              f"{before} déjà présentes + {after - before} nouvelles = {after} inst  {mark}  "
              f"(+{len(new)} frames)  [gap {gap_s}s]")

    # Final shuffle for anonymization (fixed seed)
    rng_final = random.Random(BASE_SEED)
    rng_final.shuffle(all_selected)

    # Summary
    print(f"\nFrames totales    : {len(all_selected)}")
    total_boxes = sum(sum(f["inst"].values()) for f in all_selected)
    print(f"Boîtes kappa      : {total_boxes}")
    print(f"\n{'Espèce':<14} {'Instances':>9}")
    print("─" * 26)
    for c, name in KAPPA_CLASSES.items():
        n = sum(f["inst"].get(c, 0) for f in all_selected)
        if n:
            print(f"{name:<14} {n:>9}")

    if args.dry_run:
        print("\n[DRY RUN] Aucun fichier écrit.")
        return

    out_root      = Path(args.output)
    images_out    = out_root / "images"
    labels_og_out = out_root / "Annotator_Ref_raw"   # raw labels (all classes)
    labels_out    = out_root / "Annotator_Ref"        # filtered labels (kappa classes, >=15px)
    for d in (images_out, labels_og_out, labels_out):
        d.mkdir(parents=True, exist_ok=True)
    for stale in list(images_out.iterdir()) + list(labels_og_out.iterdir()) + list(labels_out.iterdir()):
        stale.unlink()

    width = len(str(len(all_selected)))
    manifest_rows     = []
    ground_truth_rows = []

    for i, f in enumerate(all_selected, start=1):
        anon_id = f"img_{i:0{width}d}"

        img       = PILImage.open(f["path"]).convert("RGB")
        annotated = draw_numbered_boxes(img, f["boxes"])
        annotated.save(images_out / f"{anon_id}.jpg", quality=93)

        # Annotator_Ref_raw: raw copy of the original label file (all classes)
        shutil.copy2(f["label_path"], labels_og_out / f"{anon_id}.txt")

        # Annotator_Ref: filtered labels, kappa boxes only (already filtered to >=15px)
        filt_lines = [
            f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in f["boxes"]
        ]
        (labels_out / f"{anon_id}.txt").write_text("\n".join(filt_lines))

        classes_present = sorted({b["class_id"] for b in f["boxes"]})
        manifest_rows.append({
            "anon_id":         anon_id,
            "n_boxes":         len(f["boxes"]),
            "classes_present": "+".join(KAPPA_CLASSES[c] for c in classes_present),
            "split":           f["split"],
            "folder":          f["folder_name"],
            "frame":           f["path"].name,
            "video_path":      f["video_path"],
        })
        for box_num, b in enumerate(f["boxes"], start=1):
            ground_truth_rows.append({
                "anon_id":    anon_id,
                "box_number": box_num,
                "class_id":   b["class_id"],
                "class_name": KAPPA_CLASSES[b["class_id"]],
            })

    manifest_rows.sort(key=lambda r: r["anon_id"])
    ground_truth_rows.sort(key=lambda r: (r["anon_id"], r["box_number"]))

    manifest_path = out_root / "kappa_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader(); w.writerows(manifest_rows)

    gt_path = out_root / "kappa_ground_truth.csv"
    with gt_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ground_truth_rows[0].keys()))
        w.writeheader(); w.writerows(ground_truth_rows)

    print(f"\nImages     → {images_out}")
    print(f"Annotator_Ref_raw → {labels_og_out}  (labels bruts)")
    print(f"Annotator_Ref     → {labels_out}  (labels filtrés kappa)")
    print(f"Manifest   → {manifest_path}")
    print(f"GT         → {gt_path}  ⚠ ne pas partager avec l'expert")


if __name__ == "__main__":
    main()
