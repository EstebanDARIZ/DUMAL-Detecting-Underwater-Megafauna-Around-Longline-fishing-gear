import csv
import re

TARGET_SPECIES = ["Ray", "Tuna", "Shark", "Sunfish"]


def video_key(file_name):
    """Extract the (species_tag, video_number) key from a filename like
    'Ray_04_1_frame_000628.jpg' -> 'Ray_04' (split-file suffixes _1/_2 merged
    back into the same source video)."""
    m = re.match(r"([A-Za-z]+)_(.+?)_(frame|bg)_\d+\.jpg$", file_name)
    if not m:
        return None
    species, vid = m.group(1), m.group(2)
    vid = re.sub(r"_[12]$", "", vid)
    return f"{species}_{vid}"


def video_species(vkey):
    """'Ray_04' -> 'Ray'"""
    return vkey.rsplit("_", 1)[0]


def load_cluster_map(csv_path):
    """Load a video_key -> cluster_id mapping from CSV (columns:
    video_key,date,camera_model,cluster_id). cluster_id is derived here as
    date+camera_model combined (matching the same hermetic criterion used for
    the original train/test video split), not read from the CSV's own
    cluster_id column, which only reflects date. Returns None if csv_path is
    None (caller should then treat each video as its own cluster)."""
    if csv_path is None:
        return None
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            camera = (row.get("camera_model") or "").strip()
            mapping[row["video_key"]] = f"{row['date']}|{camera}"
    return mapping


def cluster_of(vkey, cluster_map):
    if cluster_map is None:
        return vkey  # fallback: each video is its own cluster
    return cluster_map.get(vkey, vkey)  # unmapped videos fall back to being their own cluster
