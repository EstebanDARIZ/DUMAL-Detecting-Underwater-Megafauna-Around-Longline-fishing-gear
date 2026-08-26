from ultralytics import YOLO

for name, weights in [
    ("YOLOv12-X", "./weights/yolov12_x/best.pt"),
    ("YOLOv12-S", "./weights/yolov12_s/best.pt"),
]:
    model = YOLO(weights)
    n_l, n_p, n_g, flops = model.model.info(verbose=False)
    print(f"=== {name} ===")
    print(f"Params: {n_p}  GFLOPs: {flops}")
    print()
