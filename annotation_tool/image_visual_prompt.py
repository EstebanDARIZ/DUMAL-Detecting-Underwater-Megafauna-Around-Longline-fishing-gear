from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoSemanticPredictor
import os
import cv2
from datetime import datetime

# === CONFIGURATION ===
video_path = "./video_raw/Raie_event.mp4"
reference_image_path = "./dataset/images/R_frame_13621.jpg"
reference_bbox = [[893.0, 347.0, 1264.0, 681.0]]
output_dir = "./output/test_7"

os.makedirs(output_dir, exist_ok=True)

overrides = dict(
    conf=0.6,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    save=False,  # we handle saving manually
)

# === STEP 1: Encode the reference image ===
img_predictor = SAM3SemanticPredictor(overrides={**overrides, "save": False})
img_predictor.setup_model()
img_predictor.set_image(reference_image_path)

# === STEP 2: Video setup ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"📹 Vidéo : {width}x{height} @ {fps:.1f}fps — {total_frames} frames")

# VideoWriter for the annotated video
video_out_path = os.path.join(output_dir, "output_annotated.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

# Folder for the YOLO .txt files
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)

# Config log
# config_path = os.path.join(output_dir, "config.txt")
# with open(config_path, "w", encoding="utf-8") as f:
#     f.write("=== CONFIGURATION ===\n")
#     f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#     f.write(f"Vidéo : {video_path}\n")
#     f.write(f"Image de référence : {reference_image_path}\n")
#     f.write(f"BBox de référence : {reference_bbox}\n")
#     f.write(f"Confiance : {overrides['conf']}\n")
#     f.write("=====================\n")

# === STEP 3: Video inference, loop over the generator ===
video_predictor = SAM3VideoSemanticPredictor(overrides=overrides)

results = video_predictor(
    source=video_path,
    exemplar_imgs=[reference_image_path],
    exemplar_bboxes=[reference_bbox],
    stream=True,  # frame-by-frame generator
)

print("🚀 Démarrage du tracking...")

for frame_idx, r in enumerate(results):  # <- the loop that was missing
    # --- Original frame ---
    frame = r.orig_img.copy()  # BGR numpy array

    # --- YOLO annotations (.txt) ---
    txt_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
    if r.boxes is not None and len(r.boxes) > 0:
        boxes_xywhn = r.boxes.xywhn.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        cls_tab = r.boxes.cls.cpu().numpy()

        with open(txt_path, "w", encoding="utf-8") as f:
            for i, cls in enumerate(cls_tab):
                box = " ".join(map(str, boxes_xywhn[i]))
                f.write(f"{int(cls)} {box} {confidences[i]:.6f}\n")

        # --- Draw the bboxes on the frame ---
        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            conf = confidences[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"raie {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # --- Draw the masks (if available) ---
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                colored = frame.copy()
                colored[mask > 0.5] = [0, 180, 255]  # orange
                frame = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
    else:
        # Empty frame -> empty .txt file
        open(txt_path, "w").close()

    # --- Write the frame to the output video ---
    writer.write(frame)

    if frame_idx % 50 == 0:
        print(f"  ↳ Frame {frame_idx}/{total_frames} traitée")

writer.release()
print(f"\n✅ Vidéo annotée : {video_out_path}")
print(f"✅ Labels YOLO   : {labels_dir}")