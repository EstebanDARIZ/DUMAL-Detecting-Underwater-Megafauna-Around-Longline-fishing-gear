#!/usr/bin/env python3
"""
Build clips_iou for the IoU inter-annotator agreement task.

Timeline per clip:  [5 s lead] → [1.5 s freeze, no boxes] → [5 s tail]
The freeze frame is the selected image shown plain (no overlay) so the
annotator finds the animal from context and draws their own box.

Special cases handled:
  - run_35: local frame numbering offset +8400 (calibrated by template-match)
  - run_65: source video is wrong → clip built from the run's image sequence

Usage:
    python IAA/extract_iou_clips.py
    python IAA/extract_iou_clips.py --dry-run
"""

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

MANIFEST      = "./IAA/iou_set/selection_manifest.csv"
IMAGES_DIR    = "./IAA/iou_set/images"
OUTPUT_DIR    = "./IAA/iou_set/clips_iou"

# Folders where DataSam used session-relative frame numbering.
# true_frame = local_frame + offset
FOLDER_FRAME_OFFSET: dict[str, int] = {
    "Raie_2_event": 3600,
    "Raie_test":    8100,
    "Shark_event":  13800,
    "run_35":       8400,
}

# Folders where the video path is wrong — clips built from image sequence instead.
IMAGE_SEQUENCE_FOLDERS = {"run_65"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
RES_SAM_ROOT = Path("./res_sam")


def resolve_video(video_path_str: str) -> Path | None:
    p = Path(video_path_str)
    if p.exists():
        return p
    for candidate in (p.parent / "subclips" / p.name,
                      p.parent.parent / "subclips" / p.name):
        if candidate.exists():
            return candidate
    return None


def get_fps(video_path: Path, cache: dict) -> float:
    if video_path in cache:
        return cache[video_path]
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den or 1)
    except Exception:
        fps = 25.0
    cache[video_path] = fps
    return fps


def build_clip_from_video(video_path: Path, start: float, lead_len: float,
                           freeze_image: Path, freeze_dur: float,
                           tail_dur: float, size: tuple[int, int],
                           dst: Path, dry_run: bool):
    W, H = size
    raw_dur = lead_len + tail_dur
    fc = (
        f"[0:v]trim=0:{lead_len:.3f},setpts=PTS-STARTPTS,fps=25,scale={W}:{H},format=yuv420p[vA];"
        f"[0:v]trim={lead_len:.3f}:{raw_dur:.3f},setpts=PTS-STARTPTS,fps=25,scale={W}:{H},format=yuv420p[vC];"
        f"[1:v]trim=0:{freeze_dur:.3f},setpts=PTS-STARTPTS,fps=25,scale={W}:{H},format=yuv420p[vB];"
        f"[vA][vB][vC]concat=n=3:v=1:a=0[outv]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{raw_dur:.3f}", "-i", str(video_path),
        "-loop", "1", "-t", f"{freeze_dur:.3f}", "-i", str(freeze_image),
        "-filter_complex", fc,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-an", "-movflags", "+faststart", str(dst),
    ]
    if dry_run:
        print("  $", " ".join(cmd[:8]), "…")
        return
    subprocess.run(cmd, capture_output=True, check=True)


def build_clip_from_images(folder_name: str, frame_name: str,
                            freeze_image: Path, freeze_dur: float,
                            lead_dur: float, tail_dur: float,
                            size: tuple[int, int], dst: Path,
                            dry_run: bool):
    """Build clip from the run's image sequence (for folders with wrong video)."""
    images_dir = RES_SAM_ROOT / folder_name / "images"
    all_imgs = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not all_imgs:
        print(f"  [SKIP] no images in {images_dir}")
        return

    # Find selected frame index in the sequence
    names = [p.name for p in all_imgs]
    try:
        sel_pos = names.index(frame_name)
    except ValueError:
        print(f"  [SKIP] {frame_name} not found in {images_dir}")
        return

    fps = 25.0
    lead_frames = int(lead_dur * fps)
    tail_frames = int(tail_dur * fps)
    freeze_frames = max(1, int(freeze_dur * fps))

    lead_imgs  = all_imgs[max(0, sel_pos - lead_frames): sel_pos]
    tail_imgs  = all_imgs[sel_pos + 1: sel_pos + 1 + tail_frames]

    W, H = size

    with tempfile.TemporaryDirectory(prefix="iou_seq_") as tmp:
        tmp = Path(tmp)
        concat_txt = tmp / "concat.txt"
        lines = []

        for p in lead_imgs:
            lines.append(f"file '{p}'\nduration {1/fps:.6f}")
        # freeze (JPEG to avoid PNG/JPEG mixing in concat)
        freeze_jpg = tmp / "freeze.jpg"
        Image.open(freeze_image).convert("RGB").save(freeze_jpg, quality=95)
        lines.append(f"file '{freeze_jpg}'\nduration {freeze_dur:.3f}")
        for p in tail_imgs:
            lines.append(f"file '{p}'\nduration {1/fps:.6f}")
        # repeat last frame to avoid truncation
        last = tail_imgs[-1] if tail_imgs else freeze_jpg
        lines.append(f"file '{last}'")

        concat_txt.write_text("\n".join(lines))

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-vf", f"fps=25,scale={W}:{H},format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-an", "-movflags", "+faststart", str(dst),
        ]
        if dry_run:
            print("  $ ffmpeg [image sequence concat]", str(dst))
            return
        subprocess.run(cmd, capture_output=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lead-duration",   type=float, default=5.0)
    parser.add_argument("--freeze-duration", type=float, default=1.5)
    parser.add_argument("--tail-duration",   type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir)
    fps_cache: dict = {}

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    n_ok = n_skip = 0
    skipped = []

    with tempfile.TemporaryDirectory(prefix="iou_freeze_") as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for row in rows:
            anon_id   = row["anon_id"]
            folder    = row["folder"]
            frame_name= row["frame"]
            dst       = out_dir / f"{anon_id}.mp4"

            # Source image
            img_candidates = list(images_dir.glob(f"{anon_id}.*"))
            if not img_candidates:
                skipped.append((anon_id, "image not found")); n_skip += 1; continue
            img_path = img_candidates[0]

            # Freeze frame (JPEG copy, plain — no boxes)
            freeze = tmp_dir / f"{anon_id}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
                W, H = img.size
                W, H = W - W % 2, H - H % 2
                img.save(freeze, quality=95)
            except Exception as e:
                skipped.append((anon_id, f"PIL error: {e}")); n_skip += 1; continue

            size = (W, H)

            # ── image-sequence folders ────────────────────────────────────
            if folder in IMAGE_SEQUENCE_FOLDERS:
                print(f"  [img-seq] {anon_id} ({folder}/{frame_name})")
                try:
                    build_clip_from_images(
                        folder, frame_name, freeze,
                        args.freeze_duration, args.lead_duration, args.tail_duration,
                        size, dst, args.dry_run,
                    )
                    n_ok += 1
                except Exception as e:
                    skipped.append((anon_id, str(e))); n_skip += 1
                continue

            # ── video-based folders ───────────────────────────────────────
            video_path = resolve_video(row["video_path"])
            if video_path is None:
                skipped.append((anon_id, f"video not found: {row['video_path']}")); n_skip += 1; continue

            fps = get_fps(video_path, fps_cache)

            m = re.search(r"(\d+)", frame_name)
            if not m:
                skipped.append((anon_id, "cannot parse frame index")); n_skip += 1; continue

            local_idx  = int(m.group(1))
            abs_idx    = local_idx + FOLDER_FRAME_OFFSET.get(folder, 0)
            timestamp  = abs_idx / fps
            start      = max(0.0, timestamp - args.lead_duration)
            lead_len   = timestamp - start

            try:
                build_clip_from_video(
                    video_path, start, lead_len,
                    freeze, args.freeze_duration, args.tail_duration,
                    size, dst, args.dry_run,
                )
                n_ok += 1
            except subprocess.CalledProcessError as e:
                skipped.append((anon_id, e.stderr.decode(errors="ignore")[:120]))
                n_skip += 1

    print(f"\n{n_ok} clip(s) {'planned' if args.dry_run else 'written'}, {n_skip} skipped.")
    if skipped:
        print("\nSkipped:")
        for anon_id, reason in skipped:
            print(f"  {anon_id}: {reason}")


if __name__ == "__main__":
    main()
