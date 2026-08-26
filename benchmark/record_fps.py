import argparse
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Merge a model's FPS into results/fps.json")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--fps-file", default="results/fps.json",
                         help="path (relative to this script) to merge into, default results/fps.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--avg-ms", type=float, help="average ms per image")
    group.add_argument("--fps", type=float, help="images per second, if already computed")
    args = parser.parse_args()

    fps_path = os.path.join(BASE_DIR, args.fps_file)
    os.makedirs(os.path.dirname(fps_path), exist_ok=True)
    fps_data = {}
    if os.path.exists(fps_path):
        with open(fps_path) as f:
            fps_data = json.load(f)

    fps_data[args.model_key] = args.fps if args.fps is not None else 1000.0 / args.avg_ms
    with open(fps_path, "w") as f:
        json.dump(fps_data, f, indent=2)

    print(f"{args.model_key}: {fps_data[args.model_key]:.2f} FPS")


if __name__ == "__main__":
    main()
