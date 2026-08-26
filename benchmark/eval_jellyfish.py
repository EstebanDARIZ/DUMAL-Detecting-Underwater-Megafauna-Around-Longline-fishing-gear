import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_FILE = "./datasets/dataset_sam_3.0/cls_6/instances_test_cls6.json"
PRED_DIR = os.path.join(BASE_DIR, "predictions_jellyfish")
RESULTS_DIR = os.path.join(BASE_DIR, "results_jellyfish")
JELLYFISH_CAT_ID = 7  # per cls_6/instances_test_cls6.json categories

MODEL_KEYS = [
    "dfine_x", "dfine_s", "boosting_rcnn", "gccnet",
    "retinanet", "faster_rcnn_fpn", "yolov12_x", "yolov12_s",
]

STATS_KEYS = [
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    coco_gt = COCO(GT_FILE)

    summary = {}
    for key in MODEL_KEYS:
        pred_path = os.path.join(PRED_DIR, f"{key}.json")
        if not os.path.exists(pred_path):
            print(f"[skip] {key}: no predictions file at {pred_path}")
            continue

        print(f"\n=== {key} (JellyFish only) ===")
        coco_dt = coco_gt.loadRes(pred_path)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.catIds = [JELLYFISH_CAT_ID]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        stats = dict(zip(STATS_KEYS, ev.stats.tolist()))
        summary[key] = stats
        with open(os.path.join(RESULTS_DIR, f"{key}.json"), "w") as f:
            json.dump(stats, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n=== Summary (JellyFish only, test_cls6, 60 images) ===")
    print(f"{'Model':<20}{'mAP':>8}{'mAP50':>8}{'mAP75':>8}")
    for key, stats in summary.items():
        print(f"{key:<20}{stats['mAP']:>8.3f}{stats['mAP50']:>8.3f}{stats['mAP75']:>8.3f}")


if __name__ == "__main__":
    main()
