import argparse
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from configs import ALL_MODELS, GT_ANNOTATION_FILE, MODELS

STATS_KEYS = [
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def per_class_ap(coco_gt, coco_dt, cat_ids):
    result = {}
    for cat_id in cat_ids:
        cat_name = coco_gt.cats[cat_id]["name"]
        n_gt = len(coco_gt.getAnnIds(catIds=[cat_id]))
        if n_gt == 0:
            result[cat_name] = None
            continue
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.catIds = [cat_id]
        ev.evaluate()
        ev.accumulate()
        # skip summarize() (it prints); read AP@[.5:.95] straight from precision
        precision = ev.eval["precision"]
        ap = precision[:, :, 0, 0, -1]
        ap = ap[ap > -1]
        result[cat_name] = float(ap.mean()) if ap.size else None
    return result


def evaluate_model(model_key, model_cfg, coco_gt, base_dir):
    pred_path = os.path.join(base_dir, model_cfg["predictions"])
    if not os.path.exists(pred_path):
        print(f"[skip] {model_cfg['display_name']}: no predictions file at {pred_path}")
        return None

    coco_dt = coco_gt.loadRes(pred_path)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = dict(zip(STATS_KEYS, ev.stats.tolist()))
    stats["per_class_AP"] = per_class_ap(coco_gt, coco_dt, coco_gt.getCatIds())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                         help="subset of model keys to evaluate (default: all)")
    parser.add_argument("--fps-file", default="results/fps.json",
                         help="optional JSON with {model_key: fps} to merge into the summary")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    coco_gt = COCO(GT_ANNOTATION_FILE)

    fps_by_model = {}
    fps_path = os.path.join(base_dir, args.fps_file)
    if os.path.exists(fps_path):
        with open(fps_path) as f:
            fps_by_model = json.load(f)

    keys = args.models if args.models else list(MODELS.keys())
    summary = {}

    for key in keys:
        if key not in ALL_MODELS:
            print(f"[warn] unknown model key: {key}")
            continue
        model_cfg = ALL_MODELS[key]
        print(f"\n=== {model_cfg['display_name']} ===")
        stats = evaluate_model(key, model_cfg, coco_gt, base_dir)
        if stats is None:
            continue
        stats["fps"] = fps_by_model.get(key)
        summary[key] = stats

        with open(os.path.join(base_dir, "results", f"{key}.json"), "w") as f:
            json.dump(stats, f, indent=2)

    with open(os.path.join(base_dir, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n=== Summary ===")
    header = f"{'Model':<20}{'mAP':>8}{'mAP50':>8}{'mAP75':>8}{'FPS':>8}"
    print(header)
    for key, stats in summary.items():
        name = ALL_MODELS[key]["display_name"]
        fps = stats["fps"]
        fps_str = f"{fps:.1f}" if fps is not None else "N/A"
        print(f"{name:<20}{stats['mAP']:>8.3f}{stats['mAP50']:>8.3f}{stats['mAP75']:>8.3f}{fps_str:>8}")


if __name__ == "__main__":
    main()
