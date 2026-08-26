import cv2
import os
import re
from pathlib import Path
import sys


def parse_video_path(config_path: str) -> str | None:
    """Extract the video path from a config.txt file."""
    with open(config_path, "r") as f:
        for line in f:
            if line.startswith("Video"):
                # Format: "Video          : /path/to/video.mp4"
                match = re.split(r"\s*:\s*", line.strip(), maxsplit=1)
                if len(match) == 2:
                    return match[1].strip()
    return None


def get_frame_numbers(images_dir: str) -> list[tuple[int, str]]:
    """
    Return the list of (frame number, filename) pairs from the images
    folder, sorted by frame number.
    """
    frames = []
    pattern = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)", re.IGNORECASE)

    for filename in os.listdir(images_dir):
        m = pattern.match(filename)
        if m:
            frame_number = int(m.group(1))
            frames.append((frame_number, filename))

    frames.sort(key=lambda x: x[0])
    return frames


def extract_frames(video_path: str, frames_info: list[tuple[int, str]], output_dir: str):
    """
    Open the video and extract the requested frames at maximum quality.
    Save them into output_dir under the same filenames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Vidéo ouverte : {video_path}")
    print(f"  Nombre total de frames dans la vidéo : {total_frames}")

    os.makedirs(output_dir, exist_ok=True)

    # Index for walking through the requested frames sequentially
    frame_dict = {num: name for num, name in frames_info}
    target_numbers = sorted(frame_dict.keys())

    saved = 0

    for frame_number in target_numbers:
        filename = frame_dict[frame_number]
        output_path = os.path.join(output_dir, filename)

        # Seek directly to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            print(f"  [ATTENTION] Frame {frame_number} illisible, ignorée.")
            continue

        # Determine the output format
        ext = Path(filename).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            # Maximum JPEG quality (100)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif ext == ".png":
            # Minimal PNG compression (0 = lossless)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(output_path, frame)

        saved += 1

    cap.release()
    print(f"  {saved}/{len(target_numbers)} frames sauvegardées dans : {output_dir}")
    return saved


def process_root_folder(root_dir: str):
    """
    Walk each subfolder of the root folder and process the ones that
    contain images/, labels/, and config.txt.
    """
    root = Path(root_dir)
    subdirs = [d for d in root.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"Aucun sous-dossier trouvé dans : {root_dir}")
        return

    print(f"=== {len(subdirs)} sous-dossier(s) détecté(s) ===\n")

    total_saved = 0

    for subdir in sorted(subdirs):
        config_file = subdir / "config.txt"
        images_dir  = subdir / "images"
        new_images_dir = subdir / "new_images"

        # Check the expected structure
        if not config_file.exists():
            print(f"[IGNORÉ] {subdir.name} — config.txt introuvable")
            continue
        if not images_dir.exists():
            print(f"[IGNORÉ] {subdir.name} — dossier images/ introuvable")
            continue

        print(f"--- Traitement : {subdir.name} ---")

        # Read the video path
        video_path = parse_video_path(str(config_file))
        if not video_path:
            print(f"  [ERREUR] Chemin vidéo introuvable dans config.txt")
            continue
        if not os.path.exists(video_path):
            print(f"  [ERREUR] Vidéo introuvable sur le disque : {video_path}")
            continue

        # Get the frames to extract
        frames_info = get_frame_numbers(str(images_dir))
        if not frames_info:
            print(f"  [ATTENTION] Aucune frame trouvée dans images/")
            continue

        print(f"  {len(frames_info)} frames à extraire (de frame_{frames_info[0][0]:05d} à frame_{frames_info[-1][0]:05d})")

        saved = extract_frames(video_path, frames_info, str(new_images_dir))
        total_saved += saved
        print()

    print(f"=== Terminé : {total_saved} frames extraites au total ===")


if __name__ == "__main__":
    

    if len(sys.argv) < 2:
        print("Usage : python extract_frames.py <chemin_dossier_racine>")
        print("Exemple : python extract_frames.py /home/esteban/dataset/sessions")
        sys.exit(1)

    root_directory = sys.argv[1]

    if not os.path.isdir(root_directory):
        print(f"[ERREUR] Dossier introuvable : {root_directory}")
        sys.exit(1)

    process_root_folder(root_directory)