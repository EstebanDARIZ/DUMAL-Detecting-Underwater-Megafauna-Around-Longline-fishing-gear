#!/usr/bin/env python3
"""
Compute the two dataset-statistics tables reported in the paper (Section 3.4):
  1. Per-split counts: images, instances, distinct source videos, same-camera/same-day clusters.
  2. Per-class breakdown: instances, videos, and clusters, for Train+Val (pooled) vs Test.

A "source video" merges sub-clips of the same original recording (e.g. Ray_04_1/Ray_04_2
-> a single "Ray_04" video), resolved from the res_sam folder structure.

A "cluster" groups videos recorded by the same camera on the same day (see
dataset_creation/build_video_day_clusters.py, which produces the clusters CSV consumed here).
Note: the CSV's own "cluster_id" column is date-only and does not account for camera, so
clustering here is done on the (date, camera_model) pair directly, not that column.

Usage:
    python3 dataset_splits_and_clusters.py
    python3 dataset_splits_and_clusters.py --dataset-root ... --res-sam ... --clusters-csv ...
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

FRAME_RE = re.compile(r"^(.+?)_(?:frame|bg)_(\d+)\.")
SPLITS = ("train", "val", "test")


def build_name_to_video_key(res_sam: Path) -> dict:
    """Reproduce the sub-clip -> source video merge (see build_video_day_clusters.py)."""
    mapping = {}
    for species_dir in sorted(res_sam.iterdir()):
        if not species_dir.is_dir() or species_dir.name == "background":
            continue
        if not re.match(r"^[A-Za-z]+_\d+$", species_dir.name):
            continue
        video_key = species_dir.name
        direct_config = species_dir / "config.txt"
        subdirs = sorted(p for p in species_dir.iterdir() if p.is_dir())
        if direct_config.exists():
            mapping[video_key] = video_key
        for sub in subdirs:
            if (sub / "config.txt").exists():
                mapping[sub.name] = video_key
    return mapping


def build_video_to_cluster(csv_path: Path) -> dict:
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["video_key"]] = (row["date"], row["camera_model"])
    return mapping


def load_split(dataset_root: Path, split: str) -> dict:
    path = dataset_root / "annotations" / f"instances_{split}.json"
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Path to the COCO-format dataset root (contains annotations/instances_{train,val,test}.json)")
    parser.add_argument("--res-sam", required=True, help="Path to the res_sam folder (per-run raw annotation output)")
    parser.add_argument("--clusters-csv", required=True, help="Path to video_day_clusters.csv (from dataset_creation/build_video_day_clusters.py)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    name_to_video_key = build_name_to_video_key(Path(args.res_sam))
    video_to_cluster = build_video_to_cluster(Path(args.clusters_csv))

    class_names = {}
    per_split = {}
    per_split_class = {}

    for split in SPLITS:
        d = load_split(dataset_root, split)
        class_names.update({c["id"]: c["name"] for c in d["categories"]})

        image_to_video = {}
        for im in d["images"]:
            m = FRAME_RE.match(im["file_name"])
            if m:
                run = m.group(1)
                image_to_video[im["id"]] = name_to_video_key.get(run, run)

        videos = set(image_to_video.values())
        clusters = {video_to_cluster[vk] for vk in videos if vk in video_to_cluster}

        cls = defaultdict(lambda: dict(instances=0, videos=set(), clusters=set()))
        for a in d["annotations"]:
            vk = image_to_video.get(a["image_id"])
            entry = cls[a["category_id"]]
            entry["instances"] += 1
            if vk:
                entry["videos"].add(vk)
                if vk in video_to_cluster:
                    entry["clusters"].add(video_to_cluster[vk])

        per_split[split] = dict(
            n_images=len(d["images"]),
            n_instances=len(d["annotations"]),
            videos=videos,
            clusters=clusters,
        )
        per_split_class[split] = cls

    # --- Table 1: per split ---
    print("=" * 70)
    print("TABLE — Dataset split statistics")
    print("=" * 70)
    print(f"{'Split':<8}{'Images':>10}{'Instances':>12}{'Source videos':>16}{'Clusters':>10}")
    tot_img = tot_inst = 0
    all_videos, all_clusters = set(), set()
    for split in SPLITS:
        s = per_split[split]
        tot_img += s["n_images"]
        tot_inst += s["n_instances"]
        all_videos |= s["videos"]
        all_clusters |= s["clusters"]
        print(f"{split.capitalize():<8}{s['n_images']:>10,}{s['n_instances']:>12,}{len(s['videos']):>16}{len(s['clusters']):>10}")
    print("-" * 56)
    print(f"{'Total':<8}{tot_img:>10,}{tot_inst:>12,}{len(all_videos):>16}{len(all_clusters):>10}")

    # --- Table 2: per class, Train+Val vs Test ---
    print("\n" + "=" * 70)
    print("TABLE — Per-class instances/videos/clusters, Train+Val vs Test")
    print("=" * 70)
    print(f"{'Class':<12}{'TV inst':>9}{'TV vid':>8}{'TV clu':>8}   {'Test inst':>10}{'Test vid':>9}{'Test clu':>9}")
    for c in sorted(class_names, key=lambda c: class_names[c]):
        tv = per_split_class["train"][c]
        val = per_split_class["val"][c]
        tv_inst = tv["instances"] + val["instances"]
        tv_videos = tv["videos"] | val["videos"]
        tv_clusters = tv["clusters"] | val["clusters"]
        te = per_split_class["test"][c]
        print(f"{class_names[c]:<12}{tv_inst:>9,}{len(tv_videos):>8}{len(tv_clusters):>8}   "
              f"{te['instances']:>10,}{len(te['videos']):>9}{len(te['clusters']):>9}")


if __name__ == "__main__":
    main()
