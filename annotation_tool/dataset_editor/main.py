import argparse
import os
import sys
from datetime import datetime

from dataset_editor import DatasetEditor


def main():
    parser = argparse.ArgumentParser(description="Editeur YOLO interactif avec integration SAM3")
    parser.add_argument("--folder", type=str, required=True,help="Dossier contenant images/ et labels/")
    parser.add_argument("--fps", type=float, default=10.0,help="FPS en mode play (defaut : 10)")
    parser.add_argument("--video", type=str, required=True, help="Video source (navigation + SAM3)")
    parser.add_argument("--model", type=str, default="sam3.pt",
                        help="Poids SAM3 a utiliser (defaut: sam3.pt)")

    args = parser.parse_args()

    os.makedirs(args.folder, exist_ok=True)
    if not os.path.exists(args.video):
        print(f"[ERREUR] Video introuvable : {args.video}")
        sys.exit(1)

    editor = DatasetEditor(folder=args.folder, fps=args.fps, video_path=args.video,
                           model=args.model)
    t0 = datetime.now()
    editor.run()
    elapsed = datetime.now() - t0

    imgs_dir   = os.path.join(args.folder, "images")
    labels_dir = os.path.join(args.folder, "labels")
    n_imgs   = len(os.listdir(imgs_dir))   if os.path.isdir(imgs_dir)   else 0
    n_labels = sum(
        1 for f in os.listdir(labels_dir)
        if os.path.getsize(os.path.join(labels_dir, f)) > 0
    ) if os.path.isdir(labels_dir) else 0

    config_path = os.path.join(args.folder, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video          : {args.video}\n")
        f.write(f"N objects      : {len(editor._sam_class_ids)}\n")
        f.write(f"Class-ids      : {editor._sam_class_ids}\n")
        f.write(f"Conf           : {editor._sam_conf}\n")
        f.write(f"Lost threshold : {editor._sam_lost_thr}\n")
        f.write(f"Merge mode     : {editor._merge_mode}\n")
        f.write(f"Segments       : {editor._sam_segment_count}\n")
        f.write(f"Images         : {n_imgs}\n")
        f.write(f"Labels         : {n_labels}\n")
        f.write(f"Time           : {elapsed}\n")
    print(f"[INFO] config.txt écrit : {config_path}")


if __name__ == "__main__":
    main()