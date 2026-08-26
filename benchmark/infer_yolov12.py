import argparse
import json
import os

from ultralytics import YOLO

from configs import GT_ANNOTATION_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--model-key", required=True, help="e.g. yolov12_x, yolov12_s")
    parser.add_argument("--data", required=True)
    parser.add_argument("--gt-file", default=GT_ANNOTATION_FILE,
                         help="COCO GT file used for the stem->id remap (must match --data's test split)")
    parser.add_argument("--pred-dir", default=os.path.join(BASE_DIR, "predictions"))
    parser.add_argument("--fps-file", default=os.path.join(BASE_DIR, "results", "fps.json"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="test", save_json=True, batch=1)

    # Ultralytics saves predictions.json under its own run dir (metrics.save_dir),
    # keyed by filename stem rather than the COCO integer image id, so remap
    # before writing out to keep the file directly usable by pycocotools.
    src_json = os.path.join(str(metrics.save_dir), "predictions.json")
    with open(src_json) as f:
        preds = json.load(f)

    with open(args.gt_file) as f:
        gt = json.load(f)
    stem_to_id = {im["file_name"].rsplit(".", 1)[0]: im["id"] for im in gt["images"]}
    for p in preds:
        p["image_id"] = stem_to_id[p["image_id"]]

    os.makedirs(args.pred_dir, exist_ok=True)
    dst_json = os.path.join(args.pred_dir, f"{args.model_key}.json")
    with open(dst_json, "w") as f:
        json.dump(preds, f)
    print(f"Predictions copied to {dst_json}")

    speed = metrics.speed
    total_ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
    fps = 1000.0 / total_ms

    fps_path = args.fps_file
    os.makedirs(os.path.dirname(fps_path), exist_ok=True)
    fps_data = {}
    if os.path.exists(fps_path):
        with open(fps_path) as f:
            fps_data = json.load(f)
    fps_data[args.model_key] = fps
    fps_data[f"{args.model_key}_speed_ms"] = speed
    with open(fps_path, "w") as f:
        json.dump(fps_data, f, indent=2)

    print(f"{args.model_key}: {fps:.2f} FPS "
          f"(preprocess {speed['preprocess']:.2f}ms, "
          f"inference {speed['inference']:.2f}ms, "
          f"postprocess {speed['postprocess']:.2f}ms)")


if __name__ == "__main__":
    main()
