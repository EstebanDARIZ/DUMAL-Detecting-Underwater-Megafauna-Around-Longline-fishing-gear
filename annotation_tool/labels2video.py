import cv2
import os
import glob
import argparse

from config import CLASS_NAMES, _PALETTE_BGR

def yolo_to_xml(yolo_coords, img_w, img_h):
    """Convert YOLO format (center_x, center_y, width, height) to pixel coords (x1, y1, x2, y2)."""
    c_x, c_y, w, h = yolo_coords
    x1 = int((c_x - w / 2) * img_w)
    y1 = int((c_y - h / 2) * img_h)
    x2 = int((c_x + w / 2) * img_w)
    y2 = int((c_y + h / 2) * img_h)
    return x1, y1, x2, y2



def main():
    parser = argparse.ArgumentParser(description="Create a video from image and labels. Input : path of the folder containing images and labels.")
    parser.add_argument("--folder", type=str, help="Path to the folder containing images and labels")
    parser.add_argument("--fps", type=int, default= 25)

    args = parser.parse_args()

    folder = args.folder 
    image_folder = os.path.join(folder, 'images')
    label_folder = os.path.join(folder, 'labels')
    video_name = os.path.join(folder, 'output.mp4')
    fps = args.fps

    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        print("Images not found, check extension")
        exit()

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"Generating video: {width}x{height} @ {fps}fps...")

    for img_path in images:
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(label_folder, f"{img_name}.txt")
    
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    class_id = int(parts[0])

                    coords = [float(x) for x in parts[1:]]
                    x1, y1, x2, y2 = yolo_to_xml(coords, width, height)

                    color = _PALETTE_BGR[class_id % len(_PALETTE_BGR)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = CLASS_NAMES.get(class_id, f"class {class_id}")
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        video.write(img)

    video.release()
    print(f"Done! Video saved to: {video_name}")

if __name__ == "__main__":
    main()
