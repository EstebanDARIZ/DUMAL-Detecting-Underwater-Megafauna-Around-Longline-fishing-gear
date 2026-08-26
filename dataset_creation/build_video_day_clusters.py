#!/usr/bin/env python3
import csv
import re
from pathlib import Path

RES_SAM = Path("./res_sam")
OUT_CSV = Path("./video_day_clusters.csv")

DATE_RE = re.compile(r"^(\d{2})[-_](\d{2})[-_](\d{2,4})$")

def parse_video_path(video_path: str):
    parts = Path(video_path).parts
    if "BORIS" not in parts:
        return None
    idx = parts.index("BORIS")
    try:
        year = parts[idx + 2]
        date_str = parts[idx + 3]
    except IndexError:
        return None
    m = DATE_RE.match(date_str)
    if not m:
        return None
    dd, mm, yy = m.groups()
    date_iso = f"{year}-{mm}-{dd}"

    # Segments between the date folder and the video filename (excluded).
    # If >=2 segments remain, the first one is the camera/gangion id.
    # If only 1 remains, the video file sits directly under the date folder
    # (no resolvable camera id in the path).
    segments_after_date = parts[idx + 4:]
    camera_id = segments_after_date[0] if len(segments_after_date) >= 2 else ""

    return date_iso, camera_id

def read_config(config_path: Path):
    video_line = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("Video"):
            video_line = line.partition(":")[2].strip()
            break
    if not video_line:
        return None
    return parse_video_path(video_line)

rows = []
skipped = []

for species_dir in sorted(RES_SAM.iterdir()):
    if not species_dir.is_dir():
        continue
    name = species_dir.name
    if name == "background":
        continue
    m = re.match(r"^([A-Za-z]+)_(\d+)$", name)
    if not m:
        continue  # not a species_XX folder
    species, run_num = m.group(1), m.group(2)
    video_key = f"{species}_{run_num}"

    direct_config = species_dir / "config.txt"
    subdirs = sorted(p for p in species_dir.iterdir() if p.is_dir())

    config_paths = []
    if direct_config.exists():
        config_paths.append(direct_config)
    if subdirs:
        for sub in subdirs:
            sub_cfg = sub / "config.txt"
            if sub_cfg.exists():
                config_paths.append(sub_cfg)

    if not config_paths:
        skipped.append((video_key, "no config.txt found"))
        continue

    for cfg in config_paths:
        parsed = read_config(cfg)
        if parsed is None:
            skipped.append((str(cfg), "could not parse Video path"))
            continue
        date_iso, camera_id = parsed
        rows.append({
            "video_key": video_key,
            "date": date_iso,
            "camera_model": camera_id,
            "cluster_id": date_iso,
        })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["video_key", "date", "camera_model", "cluster_id"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"{len(rows)} lignes écrites dans {OUT_CSV}")
if skipped:
    print(f"\n{len(skipped)} entrées ignorées :")
    for s in skipped:
        print(" ", s)
