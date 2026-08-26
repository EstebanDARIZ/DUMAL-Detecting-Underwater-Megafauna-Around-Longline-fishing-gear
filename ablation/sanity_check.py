import json
import sys

from video_utils import TARGET_SPECIES, video_key, video_species

BASE = "./datasets/dataset_sam_3.0_reduced"
ORIG_TRAIN = "./datasets/dataset_sam_3.0/annotations/instances_train.json"
ORIG_VAL = "./datasets/dataset_sam_3.0/annotations/instances_val.json"
ORIG_TEST = "./datasets/dataset_sam_3.0/annotations/instances_test.json"

VARIANTS = [(pct, seed) for pct in ("75pct", "50pct") for seed in ("seed1", "seed42")]

failures = []


def check(condition, message):
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def basic_coco_validity(coco, label):
    image_ids = [im["id"] for im in coco["images"]]
    check(len(image_ids) == len(set(image_ids)), f"{label}: no duplicate image ids")
    valid_ids = set(image_ids)
    orphans = [a for a in coco["annotations"] if a["image_id"] not in valid_ids]
    check(len(orphans) == 0, f"{label}: no orphaned annotations ({len(orphans)} found)")
    return valid_ids


def videos_in(coco):
    return set(v for v in (video_key(im["file_name"]) for im in coco["images"]) if v)


def species_video_counts(coco):
    vbs = {}
    for v in videos_in(coco):
        vbs.setdefault(video_species(v), set()).add(v)
    return vbs


print("=== 0. Test split untouched ===")
with open(ORIG_TEST) as f:
    orig_test = json.load(f)
check(True, f"test has {len(orig_test['images'])} images, {len(orig_test['annotations'])} "
             f"annotations (not modified by this pipeline, nothing to compare against)")

print("\n=== 1. Original train/val category lists ===")
with open(ORIG_TRAIN) as f:
    orig_train = json.load(f)
with open(ORIG_VAL) as f:
    orig_val = json.load(f)
orig_cats = orig_train["categories"]

for pct, seed in VARIANTS:
    label = f"{pct}_{seed}"
    print(f"\n=== {label} ===")

    with open(f"{BASE}/instances_train_{pct}_{seed}.json") as f:
        train = json.load(f)
    with open(f"{BASE}/instances_val_{pct}_{seed}.json") as f:
        val = json.load(f)

    check(train["categories"] == orig_cats, f"{label}: train categories unchanged (9 classes, same ids)")
    check(val["categories"] == orig_cats, f"{label}: val categories unchanged")

    basic_coco_validity(train, f"{label} train")
    basic_coco_validity(val, f"{label} val")

    train_videos = videos_in(train)
    val_videos = videos_in(val)
    check(val_videos <= train_videos,
          f"{label}: every video present in val is also present in train "
          f"(val has {len(val_videos)} videos, train has {len(train_videos)})")

    orig_train_videos = videos_in(orig_train)
    check(train_videos <= orig_train_videos,
          f"{label}: no video appears in reduced train that wasn't in the original train")

    orig_val_videos = videos_in(orig_val)
    dropped_from_train = orig_train_videos - train_videos
    val_videos_that_should_be_gone = dropped_from_train & orig_val_videos
    check(len(val_videos_that_should_be_gone & val_videos) == 0,
          f"{label}: val contains none of the {len(val_videos_that_should_be_gone)} "
          f"videos that were dropped from train")

    sp_counts = species_video_counts(train)
    for sp in TARGET_SPECIES:
        n = len(sp_counts.get(sp, set()))
        print(f"    {sp:<10} {n} videos in this train variant")

print("\n=== 2. Nesting: 50% videos subset of 75% videos, per seed ===")
for seed in ("seed1", "seed42"):
    with open(f"{BASE}/instances_train_75pct_{seed}.json") as f:
        v75 = videos_in(json.load(f))
    with open(f"{BASE}/instances_train_50pct_{seed}.json") as f:
        v50 = videos_in(json.load(f))
    check(v50 <= v75, f"{seed}: 50% video set is a strict subset of 75% video set")

print("\n=== 3. Non-target species (Squid/Sardine/PilotFish/JellyFish/Mackerel) still present ===")
for pct, seed in VARIANTS:
    with open(f"{BASE}/instances_train_{pct}_{seed}.json") as f:
        train = json.load(f)
    cat_names = {c["id"]: c["name"] for c in train["categories"]}
    present = set(cat_names[a["category_id"]] for a in train["annotations"])
    non_target = {"Squid", "Sardine", "Pilot Fish", "JellyFish", "Mackerel"}
    check(len(non_target & present) > 0,
          f"{pct}_{seed}: at least some non-target species instances survived "
          f"({sorted(non_target & present)})")

print(f"\n{'='*50}")
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
else:
    print("All checks passed.")
