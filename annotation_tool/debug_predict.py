"""
debug_predict.py: single-image diagnostic to figure out why predictions come back empty.
"""
import cv2
import os
from ultralytics.models.sam import SAM3SemanticPredictor

IMAGES_DIR = "./datasets_test/dataset_test_3.0/images"
MODEL      = "sam3_finetune.pt"
CONF       = 0.01

PROMPTS = [
    "a Squid",
    "a Sardine",
    "a Ray",
    "a Sunfish",
    "a Pilot Fish",
    "a Shark",
]

# --- grab the first available image ---
images = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
assert images, f"Aucune image dans {IMAGES_DIR}"
img_path = os.path.join(IMAGES_DIR, images[0])
print(f"Image : {img_path}")

frame = cv2.imread(img_path)
H, W  = frame.shape[:2]
print(f"Taille : {W}x{H}")

overrides = dict(conf=CONF, task="segment", mode="predict",
                 imgsz=644, model=MODEL, half=True, save=False)
predictor = SAM3SemanticPredictor(overrides=overrides)

predictor.set_image(frame)
results = predictor(text=PROMPTS)

print(f"\nNombre de résultats : {len(results)}")

for ri, r in enumerate(results):
    print(f"\n--- Résultat [{ri}] ---")
    print(f"  type(r)       : {type(r)}")
    print(f"  r.boxes       : {r.boxes}")
    print(f"  r.masks       : {r.masks}")
    print(f"  r.probs       : {r.probs}")
    print(f"  r.names       : {r.names}")

    if r.boxes is not None:
        print(f"  r.boxes.shape : {r.boxes.data.shape}")
        print(f"  r.boxes.cls   : {r.boxes.cls}")
        print(f"  r.boxes.conf  : {r.boxes.conf}")
        print(f"  r.boxes.xywhn : {r.boxes.xywhn}")
    if r.masks is not None:
        print(f"  r.masks.shape : {r.masks.data.shape}")
