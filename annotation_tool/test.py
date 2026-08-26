from ultralytics.models.sam import SAM3VideoSemanticPredictor, SAM3VideoPredictor
import os
import cv2
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================
VIDEO_PATH    = "./video_raw/Raie_event_2.0.mp4"
OUTPUT_DIR    = "./output/test_9"
INITIAL_BBOXES = [
    [53, 372, 272, 648],
    [1151, 163, 1222, 199],
]
TEXT_PROMPTS   = ["a stingray", "a squid"]
LOST_THRESHOLD = 60   # frames with no detection before switching over (= 2s at 30fps)
CONF_TRACKING  = 0.35
CONF_SEMANTIC  = 0.40

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# UTILITIES
# ================================================================
def draw_detections(frame, r, color, label_prefix):
    if r.masks is not None:
        for mask in r.masks.data.cpu().numpy():
            overlay = frame.copy()
            overlay[mask > 0.5] = list(color)
            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
    if r.boxes is not None:
        confs = r.boxes.conf.cpu().numpy()
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_prefix} {confs[i]:.2f}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return frame

def save_labels(r, txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        if r.boxes is not None and len(r.boxes) > 0:
            for i, cls in enumerate(r.boxes.cls.cpu().numpy()):
                box  = " ".join(map(str, r.boxes.xywhn[i].cpu().numpy()))
                conf = r.boxes.conf[i].item()
                f.write(f"{int(cls)} {box} {conf:.6f}\n")

# ================================================================
# VIDEO INFO
# ================================================================
cap = cv2.VideoCapture(VIDEO_PATH)
FPS    = cap.get(cv2.CAP_PROP_FPS)
WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f"📹 {WIDTH}x{HEIGHT} @ {FPS}fps — {TOTAL} frames")

# Visual check of the initial bboxes
cap = cv2.VideoCapture(VIDEO_PATH)
_, frame0 = cap.read()
cap.release()
for i, (x1, y1, x2, y2) in enumerate(INITIAL_BBOXES):
    cv2.rectangle(frame0, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame0, f"init {i}", (x1, max(y1-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "frame0_check.jpg"), frame0)
print("📸 frame0_check.jpg sauvegardé — vérifie que les bboxes sont correctes avant de continuer")

# ================================================================
# OVERRIDES
# ================================================================
BASE_OVERRIDES = dict(imgsz=644, model="sam3.pt", half=True, save=False,
                      task="segment", mode="predict")

# ================================================================
# PASS 1: SAM3 TRACKING
# Tracks the specific individuals initialized on frame 0.
# Detection stops once the rays leave the frame.
# ================================================================
print("\n🎯 Passe 1 — Tracking individuel (SAM3VideoPredictor)")

tracker  = SAM3VideoPredictor(overrides={**BASE_OVERRIDES, "conf": CONF_TRACKING})
results1 = tracker(source=VIDEO_PATH, bboxes=INITIAL_BBOXES, stream=True)

labels_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(labels_dir, exist_ok=True)

# We keep all of pass 1's annotated frames in memory
# (if your video is very long, >20min, write frame by frame instead)
frames_pass1 = {}   # frame_idx -> annotated BGR frame
lost_since   = 0
lost_at      = None  # frame where tracking is lost for good

for frame_idx, r in enumerate(results1):
    frame = r.orig_img.copy()
    has_det = r.boxes is not None and len(r.boxes) > 0

    if has_det:
        lost_since = 0
        frame = draw_detections(frame, r, color=(0, 255, 0), label_prefix="track")
        save_labels(r, os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt"))
    else:
        lost_since += 1
        open(os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt"), "w").close()
        cv2.putText(frame, f"LOST ({lost_since}f)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # Mark the start of the loss (first frame with no detection)
        if lost_at is None:
            lost_at = frame_idx

    # If the rays reappear after a loss, reset lost_at
    if has_det and lost_at is not None:
        lost_at = None

    frames_pass1[frame_idx] = frame

    if frame_idx % 150 == 0:
        print(f"  ↳ Frame {frame_idx:5d}/{TOTAL}  lost_since={lost_since}")

# Determine which frame to restart semantic detection from
# We start at (lost_at - 5 frames) so we don't miss the exact moment they reappear
redet_start = max(0, (lost_at or TOTAL) - 5) if lost_at is not None else None
print(f"\n  ↳ Tracking terminé. Perte détectée à la frame {lost_at}, "
      f"re-détection sémantique à partir de la frame {redet_start}")

# ================================================================
# PASS 2: SEMANTIC RE-DETECTION (if a loss was detected)
# Re-runs over the entire video with the text prompt and overwrites
# only the frames from redet_start onward
# ================================================================
if redet_start is not None:
    print(f"\n🔍 Passe 2 — Re-détection sémantique '{TEXT_PROMPTS}' depuis frame {redet_start}")

    semantic  = SAM3VideoSemanticPredictor(overrides={**BASE_OVERRIDES, "conf": CONF_SEMANTIC})
    results2  = semantic(source=VIDEO_PATH, text=TEXT_PROMPTS, stream=True)

    for frame_idx, r in enumerate(results2):
        if frame_idx < redet_start:
            continue   # keep pass 1's result

        frame = r.orig_img.copy()
        has_det = r.boxes is not None and len(r.boxes) > 0

        if has_det:
            frame = draw_detections(frame, r, color=(0, 140, 255), label_prefix="redet")
            save_labels(r, os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt"))
        else:
            open(os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt"), "w").close()

        frames_pass1[frame_idx] = frame  # overwrite pass 1's frame

        if frame_idx % 150 == 0:
            print(f"  ↳ Frame {frame_idx:5d}/{TOTAL}")

    print("  ↳ Passe 2 terminée")

# ================================================================
# WRITE THE FINAL VIDEO
# ================================================================
print("\n💾 Écriture de la vidéo finale...")
video_out = os.path.join(OUTPUT_DIR, "output_hybrid.mp4")
writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

for frame_idx in range(TOTAL):
    if frame_idx in frames_pass1:
        writer.write(frames_pass1[frame_idx])

writer.release()

# Config log
with open(os.path.join(OUTPUT_DIR, "config.txt"), "w") as f:
    f.write(f"Date             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Vidéo            : {VIDEO_PATH}\n")
    f.write(f"Bboxes initiales : {INITIAL_BBOXES}\n")
    f.write(f"Prompts texte    : {TEXT_PROMPTS}\n")
    f.write(f"Lost threshold   : {LOST_THRESHOLD} frames\n")
    f.write(f"Perte détectée   : frame {lost_at}\n")
    f.write(f"Re-détection dès : frame {redet_start}\n")
    f.write(f"Conf tracking    : {CONF_TRACKING}\n")
    f.write(f"Conf sémantique  : {CONF_SEMANTIC}\n")

print(f"\n✅ Vidéo finale  : {video_out}")
print(f"✅ Labels YOLO   : {labels_dir}")