from ultralytics.models.sam import SAM3SemanticPredictor
import os
from datetime import datetime

folder_path = "./datasets_test/dataset_test_2.0/images"
folder_pred_path = "./output/test_17"

prompts_utilises = ["a squid", "a fish bait", "a stingray", "a sunfish", "a pilotfish"]

overrides = dict(
    conf=0.2,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True, 
    save=True,
)

if not os.path.exists(folder_pred_path):
    os.makedirs(folder_pred_path)
    print(f"Dossier créé : {folder_pred_path}")

config_file_path = os.path.join(folder_pred_path, "results.txt")
with open(config_file_path, "w", encoding="utf-8") as f:
    f.write("=== CONFIGURATION DU TEST ===\n")
    f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Modèle : {overrides['model']}\n")
    f.write(f"Confiance (Seuil) : {overrides['conf']}\n")
    f.write(f"Half Precision (FP16) : {overrides['half']}\n")
    f.write(f"Prompts utilisés : {', '.join(prompts_utilises)}\n")
    f.write("==============================\n")


predictor = SAM3SemanticPredictor(overrides=overrides)

for image_name in os.listdir(folder_path):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    image_path = os.path.join(folder_path, image_name)
    predictor.set_image(image_path)
    
    results = predictor(text=prompts_utilises)

    for r in results:
        boxes = r.boxes.xywhn.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        cls_tab = r.boxes.cls.cpu().numpy()
        preds_name = image_name.replace('.png', ".txt").replace('.jpg', '.txt')
        preds_path = os.path.join(folder_pred_path, preds_name)
        with open(preds_path, "w", encoding="utf-8") as preds:
            for i, cls in enumerate(cls_tab):
                box = " ".join(map(str, boxes[i]))
                line = f"{int(cls)} {box} {confidences[i]}\n"
                preds.write(line)