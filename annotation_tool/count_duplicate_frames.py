"""
count_duplicate_frames.py
Walks every 'images/' subfolder under a root folder and counts consecutive
frame pairs (N, N+1) with identical content.

Usage:
    python count_duplicate_frames.py --folder /path/to/root
    python count_duplicate_frames.py --folder /path/to/root --threshold 8
    python count_duplicate_frames.py --folder /path/to/root --threshold 8 --delete
"""

import argparse
import glob
import hashlib
import os
import shutil

import cv2


EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
EXT_PRIORITY = {".jpg": 0, ".jpeg": 1, ".png": 2, ".bmp": 3}


def scan_images(images_dir: str) -> list[str]:
    all_paths = []
    for ext in EXTS:
        all_paths += glob.glob(os.path.join(images_dir, ext))
    seen = {}
    for p in all_paths:
        stem = os.path.basename(p).rsplit(".", 1)[0]
        ext  = os.path.splitext(p)[1].lower()
        if stem not in seen or EXT_PRIORITY.get(ext, 9) < EXT_PRIORITY.get(
                os.path.splitext(seen[stem])[1].lower(), 9):
            seen[stem] = p
    return sorted(seen.values())


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def are_identical(p1: str, p2: str, threshold: int) -> bool:
    if threshold == 0:
        return file_md5(p1) == file_md5(p2)
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return False
    return int(cv2.absdiff(img1, img2).max()) <= threshold


def label_path_for(img_path: str) -> str | None:
    """Return the path to the image's matching YOLO label, or None."""
    images_dir = os.path.dirname(img_path)
    labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    if not os.path.isdir(labels_dir):
        return None
    stem = os.path.basename(img_path).rsplit(".", 1)[0]
    lbl  = os.path.join(labels_dir, stem + ".txt")
    return lbl if os.path.exists(lbl) else None


def delete_duplicate(img_path: str, trash_dir: str):
    """Move the image (and its label, if any) into trash_dir."""
    os.makedirs(trash_dir, exist_ok=True)
    shutil.move(img_path, os.path.join(trash_dir, os.path.basename(img_path)))
    lbl = label_path_for(img_path)
    if lbl:
        shutil.move(lbl, os.path.join(trash_dir, os.path.basename(lbl)))


def process_folder(images_dir: str, threshold: int, delete: bool) -> dict:
    paths = scan_images(images_dir)
    if len(paths) < 2:
        return {"total": len(paths), "duplicates": 0, "pairs": [], "deleted": 0}

    pairs   = []
    deleted = 0
    i = 0
    while i < len(paths) - 1:
        p1, p2 = paths[i], paths[i + 1]
        if are_identical(p1, p2, threshold):
            pairs.append((os.path.basename(p1), os.path.basename(p2)))
            if delete:
                trash = os.path.join(os.path.dirname(images_dir), "_duplicates")
                delete_duplicate(p2, trash)
                deleted += 1
                paths.pop(i + 1)   # don't compare p3 against the now-deleted p2
                continue
        i += 1

    return {"total": len(paths), "duplicates": len(pairs),
            "pairs": pairs, "deleted": deleted}


def find_image_dirs(root: str) -> list[str]:
    results = []
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) == "images":
            results.append(dirpath)
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compte (et supprime) les paires de frames consécutives identiques.")
    parser.add_argument("--folder", required=True,
                        help="Dossier racine à parcourir")
    parser.add_argument("--threshold", type=int, default=0,
                        help="Diff max par pixel tolérée (0=strict MD5, ex: 8). Défaut: 0")
    parser.add_argument("--delete", action="store_true",
                        help="Supprimer le doublon N+1 (déplacé dans _duplicates/)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERREUR] Dossier introuvable : {args.folder}")
        return

    image_dirs = find_image_dirs(args.folder)
    if not image_dirs:
        print(f"Aucun sous-dossier 'images/' trouvé dans {args.folder}")
        return

    mode = "strict MD5" if args.threshold == 0 else f"diff max={args.threshold}/255"
    action = "SUPPRESSION" if args.delete else "comptage seul"
    print(f"Threshold : {args.threshold} ({mode})  |  Mode : {action}")
    if args.delete:
        print("  Les doublons sont déplacés dans <run>/_duplicates/ (récupérables)")
    print(f"{'─'*72}")

    total_frames = 0
    total_dups   = 0
    total_deleted = 0

    for images_dir in image_dirs:
        rel = os.path.relpath(images_dir, args.folder)
        res = process_folder(images_dir, args.threshold, args.delete)
        total_frames  += res["total"]
        total_dups    += res["duplicates"]
        total_deleted += res["deleted"]

        del_info = f"  [{res['deleted']} supprimé(s)]" if args.delete and res["deleted"] else ""
        status   = f"{res['duplicates']:4d} paire(s) / {res['total']} frames{del_info}"
        print(f"  {rel:<45}  {status}")

        if res["pairs"] and res["duplicates"] <= 20:
            for a, b in res["pairs"]:
                marker = "  [del]" if args.delete else ""
                print(f"      {a}  ==  {b}{marker}")

    print(f"{'─'*72}")
    summary = f"  TOTAL : {total_dups} paire(s) dupliquée(s) sur {total_frames} frames"
    if args.delete:
        summary += f"  —  {total_deleted} fichier(s) supprimé(s)"
    print(summary)


if __name__ == "__main__":
    main()
