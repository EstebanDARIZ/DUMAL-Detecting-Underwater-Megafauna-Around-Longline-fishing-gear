import argparse
import collections
import json
import math
import os
import random

from video_utils import TARGET_SPECIES, cluster_of, load_cluster_map, video_key, video_species

TRAIN_ANN = "./datasets/dataset_sam_3.0/annotations/instances_train.json"
VAL_ANN = "./datasets/dataset_sam_3.0/annotations/instances_val.json"


def pick_kept_videos(coco, cluster_map, fraction, seed):
    """For each target species, group its own videos into clusters, randomly
    keep `fraction` of the clusters (all videos in a kept cluster survive
    with all their frames), and return the union of kept video keys."""
    img_to_video = {im["id"]: video_key(im["file_name"]) for im in coco["images"]}

    videos_by_species = collections.defaultdict(set)
    for vkey in img_to_video.values():
        if vkey is not None:
            videos_by_species[video_species(vkey)].add(vkey)

    kept_videos = set()
    summary = {}
    rng = random.Random(seed)

    for species in TARGET_SPECIES:
        species_videos = videos_by_species.get(species, set())

        clusters = collections.defaultdict(set)
        for vkey in species_videos:
            clusters[cluster_of(vkey, cluster_map)].add(vkey)

        cluster_ids = sorted(clusters.keys())
        rng.shuffle(cluster_ids)
        n_keep = max(1, round(fraction * len(cluster_ids)))
        kept_cluster_ids = cluster_ids[:n_keep]

        species_kept_videos = set()
        for cid in kept_cluster_ids:
            species_kept_videos |= clusters[cid]
        kept_videos |= species_kept_videos

        summary[species] = {
            "total_clusters": len(cluster_ids),
            "kept_clusters": len(kept_cluster_ids),
            "total_videos": len(species_videos),
            "kept_videos": len(species_kept_videos),
        }

    return kept_videos, img_to_video, summary


def filter_coco(coco, kept_videos, img_to_video):
    kept_image_ids = {
        im_id for im_id, vkey in img_to_video.items() if vkey in kept_videos
    }
    new_images = [im for im in coco["images"] if im["id"] in kept_image_ids]
    new_annotations = [
        a for a in coco["annotations"] if a["image_id"] in kept_image_ids
    ]
    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }


def filter_val_to_match_train(kept_videos):
    """Val shares its video pool with train (unlike test, which is fully
    video-hermetic) -- it's held-out *frames* from the same videos, not
    held-out videos. If we don't also drop from val whatever videos were
    dropped from train, val ends up a mix of frames from videos the reduced
    model has never seen at all (behaving like a mini test set) and frames
    from videos it trained on elsewhere (behaving like a normal val set),
    and that mix ratio would differ between the 50%/75%/100% variants,
    making the training-time monitoring signal not comparable across them.
    So val is filtered with the exact same kept_videos set as train, dropping
    a video's content wholesale (all species) just like for train."""
    with open(VAL_ANN) as f:
        val_coco = json.load(f)
    val_img_to_video = {im["id"]: video_key(im["file_name"]) for im in val_coco["images"]}
    return filter_coco(val_coco, kept_videos, val_img_to_video), val_coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, required=True, help="e.g. 0.75 or 0.5")
    parser.add_argument("--cluster-csv", default=None,
                         help="CSV with columns video_key,cluster_id. "
                              "Omit to treat each video as its own cluster.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(TRAIN_ANN) as f:
        coco = json.load(f)

    cluster_map = load_cluster_map(args.cluster_csv)
    kept_videos, img_to_video, summary = pick_kept_videos(
        coco, cluster_map, args.fraction, args.seed
    )
    reduced = filter_coco(coco, kept_videos, img_to_video)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reduced, f)

    print(f"fraction={args.fraction} seed={args.seed} "
          f"cluster_csv={'none (1 video = 1 cluster)' if cluster_map is None else args.cluster_csv}")
    for species, s in summary.items():
        print(f"  {species:<10} clusters {s['kept_clusters']}/{s['total_clusters']}  "
              f"videos {s['kept_videos']}/{s['total_videos']}")
    print(f"images: {len(reduced['images'])} / {len(coco['images'])}")
    print(f"annotations: {len(reduced['annotations'])} / {len(coco['annotations'])}")
    print(f"written to {args.output}")

    reduced_val, full_val = filter_val_to_match_train(kept_videos)
    val_output = args.output.replace("instances_train_", "instances_val_")
    assert val_output != args.output, "--output must contain 'instances_train_' for the val filename to be derived"
    with open(val_output, "w") as f:
        json.dump(reduced_val, f)
    print(f"val images: {len(reduced_val['images'])} / {len(full_val['images'])}")
    print(f"val annotations: {len(reduced_val['annotations'])} / {len(full_val['annotations'])}")
    print(f"written to {val_output}")


if __name__ == "__main__":
    main()
