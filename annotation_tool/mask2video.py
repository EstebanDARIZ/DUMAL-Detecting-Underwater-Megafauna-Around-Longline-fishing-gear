import cv2
import os
import glob
import argparse
import numpy as np


CLASS_NAMES = {
    "0": "Squid",
    "1": "Sardine",
    "2": "Ray",
    "3": "Sunfish",
    "4": "Pilot Fish",
    "5": "Shark",
}

CLASS_COLORS = {
    "0": (255,   0,   0),
    "1": (  0, 255,   0),
    "2": (  0,   0, 255),
    "3": (255, 255,   0),
    "4": (255,   0, 255),
    "5": (255, 255, 255),
}


def draw_masks(img: np.ndarray, label_path: str, alpha: float = 0.35) -> np.ndarray:
    """Draw the YOLO segmentation masks on the image."""
    H, W = img.shape[:2]
    overlay = img.copy()

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # cls + at least 3 (x, y) points
                continue

            class_id = parts[0]
            coords   = [float(v) for v in parts[1:]]

            # Unpack normalized (x, y) pairs into pixels
            pts = np.array([
                [int(coords[i] * W), int(coords[i + 1] * H)]
                for i in range(0, len(coords) - 1, 2)
            ], dtype=np.int32)

            color = CLASS_COLORS.get(class_id, (255, 255, 255))

            # Semi-transparent fill
            cv2.fillPoly(overlay, [pts], color)

            # Opaque outline
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            # Text label above the first point
            label = CLASS_NAMES.get(class_id, f"class {class_id}")
            x, y  = pts[0]
            cv2.putText(img, label, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Merge overlay (fill) with image (outlines + text)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Crée une vidéo annotée avec les masques de segmentation YOLO. "
                    "Input : dossier contenant images/ et mask_labels/."
    )
    parser.add_argument("--folder", type=str, required=True,
                        help="Dossier contenant images/ et mask_labels/")
    parser.add_argument("--fps",    type=int, default=25)
    parser.add_argument("--alpha",  type=float, default=0.35,
                        help="Opacité du remplissage des masques (défaut: 0.35)")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin de sortie de la vidéo (défaut: <folder>/resultat_masks.mp4)")
    args = parser.parse_args()

    image_folder = os.path.join(args.folder, "images")
    mask_folder  = os.path.join(args.folder, "mask_labels")
    video_name   = args.output or os.path.join(args.folder, "resultat_masks.mp4")

    # Start from the masks, not the images
    mask_files = sorted(glob.glob(os.path.join(mask_folder, "*.txt")))
    mask_files = [p for p in mask_files if os.path.getsize(p) > 0]
    if not mask_files:
        print("Aucun masque trouvé dans mask_labels/")
        return

    # Resolve the matching images
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    pairs = []
    for mask_path in mask_files:
        stem = os.path.basename(mask_path).rsplit(".", 1)[0]
        img_path = None
        for ext in exts:
            candidate = os.path.join(image_folder, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"  [!] Image introuvable pour {stem}, skip.")
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        print("Aucune image trouvée pour les masques.")
        return

    first = cv2.imread(pairs[0][0])
    H, W  = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video  = cv2.VideoWriter(video_name, fourcc, args.fps, (W, H))

    print(f"Génération de la vidéo : {W}x{H} @ {args.fps}fps — {len(pairs)} frames (masques uniquement)...")

    for img_path, label_path in pairs:
        img = cv2.imread(img_path)
        img = draw_masks(img, label_path, args.alpha)
        video.write(img)

    video.release()
    print(f"Frames dans la vidéo : {len(pairs)}")
    print(f"Vidéo sauvegardée    : {video_name}")


if __name__ == "__main__":
    main()
