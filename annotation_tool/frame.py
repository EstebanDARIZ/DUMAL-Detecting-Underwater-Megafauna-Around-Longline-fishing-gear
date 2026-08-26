from ultralytics.models.sam import SAM3SemanticPredictor
import os
import cv2
from datetime import datetime
import numpy as np


VIDEO_PATH   = "./video_raw/Raie_event_2.0.mp4"
OUTPUT_DIR   = "./output/test_20"
TEXT_PROMPTS = ["a little squid", "a stingray"]
CONF         = 0.60

os.makedirs(OUTPUT_DIR, exist_ok=True)
labels_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(labels_dir, exist_ok=True)
imgs_dir = os.path.join(OUTPUT_DIR, "images")
os.makedirs(imgs_dir, exist_ok=True)


def get_box_from_mask(mask, w_orig, h_orig):
    mask_float = mask.astype(np.float32)
    if len(mask_float.shape) == 3: mask_float = mask_float.squeeze()
    mask_resized = cv2.resize(mask_float, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    coords = np.column_stack(np.where(binary_mask > 0))
    if coords.shape[0] > 0:
        # y_min, x_min are the minimum values; y_max, x_max are the maximums
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max


def xyxy2xywhn(x1, y1, x2, y2, WIDTH, HEIGHT):
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w/2
    yc = y1 + h/2
    return xc/WIDTH, yc/HEIGHT, w/WIDTH, h/HEIGHT


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    FPS    = cap.get(cv2.CAP_PROP_FPS)
    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames are {WIDTH}x{HEIGHT} @ {FPS}fps — {TOTAL} frames")

    overrides = dict(
        conf=CONF,
        task="segment",
        mode="predict",
        imgsz=644,
        model="sam3.pt",
        half=True,
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    video_out = os.path.join(OUTPUT_DIR, "output_frame_by_frame.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    print("Loading...\n")
    n_detections = 0
    start_time   = datetime.now()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictor.set_image(frame)
        results = predictor(text=TEXT_PROMPTS)

        txt_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            n_detections += len(r.boxes)
            # boxes_xywhn  = r.boxes.xywhn.cpu().numpy()
            confidences  = r.boxes.conf.cpu().numpy()
            cls_tab      = r.boxes.cls.cpu().numpy()

            # with open(txt_path, "w", encoding="utf-8") as f:
            #     for i, cls in enumerate(cls_tab):
            #         box = " ".join(map(str, boxes_xywhn[i]))
            #         f.write(f"{int(cls)} {box} {confidences[i]:.6f}\n")
            #         print(f"{int(cls)} {box} {confidences[i]:.6f}")


            if r.masks is not None:
                img_path = os.path.join(imgs_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(img_path, frame)
                for i, mask in enumerate(r.masks.data.cpu().numpy()):
                    x1, y1, x2, y2 = get_box_from_mask(mask, WIDTH, HEIGHT)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Cls : {cls_tab[i]}",
                            (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    xc, yc, w, h = xyxy2xywhn(x1, y1, x2, y2, WIDTH, HEIGHT)
                    print(f"{int(cls_tab[i])} {xc} {yc} {w} {h}\n")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"{int(cls_tab[i ])} {xc} {yc} {w} {h}\n")

                    overlay = annotated.copy()
                    overlay[mask > 0.5] = [0, 180, 255]
                    annotated = cv2.addWeighted(annotated, 0.65, overlay, 0.35, 0)

            # for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
            #     x1, y1, x2, y2 = box
            #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(annotated, f"Box_box_box {confidences[i]:.2f}",
            #                 (x1, max(y1 - 8, 20)),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if not os.path.exists(txt_path):
            open(txt_path, "w").close()

        writer.write(annotated)

        # if frame_idx % 100 == 0:
            # elapsed = (datetime.now() - start_time).seconds
            # fps_proc = frame_idx / max(elapsed, 1)
            # eta = int((TOTAL - frame_idx) / max(fps_proc, 0.01))
            # print(f"  ↳ Frame {frame_idx:5d}/{TOTAL}  "
            #       f"détections cumulées : {n_detections}  "
            #       f"vitesse : {fps_proc:.1f}fps  "
            #       f"ETA : {eta//60}m{eta%60:02d}s")

        frame_idx += 1

    cap.release()
    writer.release()

    elapsed_total = datetime.now() - start_time
    with open(os.path.join(OUTPUT_DIR, "config.txt"), "w") as f:
        f.write(f"Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vidéo          : {VIDEO_PATH}\n")
        f.write(f"Prompts        : {TEXT_PROMPTS}\n")
        f.write(f"Confiance      : {CONF}\n")
        f.write(f"Frames totales : {TOTAL}\n")
        f.write(f"Détections     : {n_detections}\n")
        f.write(f"Durée          : {elapsed_total}\n")

    print(f"Labels YOLO   : {labels_dir}")
    print(f"{n_detections} détections sur {frame_idx} frames en {elapsed_total}")



if __name__ == "__main__":
    main()