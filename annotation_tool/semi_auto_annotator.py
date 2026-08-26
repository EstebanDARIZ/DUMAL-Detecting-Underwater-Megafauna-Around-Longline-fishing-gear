"""
sam3_interactive_dataset.py
============================
Interactive workflow for building a detection dataset with SAM3.

Steps:
  1. First frame is shown; 2 mouse clicks define the initial bbox
  2. SAM3VideoPredictor tracks the object frame by frame
  3. When the object disappears (N consecutive frames with no detection),
     manual keyboard navigation to find where it comes back
  4. New 2-click bbox, SAM3 resumes tracking
  5. Repeat until the end of the video

Keyboard controls (navigation mode):
  ← / →        : -1 / +1 frame
  A / D        : -10 / +10 frames
  Q / E        : -100 / +100 frames
  Enter        : confirm the current frame, then draw the bbox
  S            : skip the reappearance (move to the next frame with no bbox)
  Esc          : quit

Output:
  <output_dir>/images/frame_XXXXXX.jpg
  <output_dir>/labels/frame_XXXXXX.txt  (YOLO format: cls xc yc w h)

Usage:
  python sam3_interactive_dataset.py \
      --video  path/to/video.mp4 \
      --output path/to/output_dir \
      --lost-threshold 15 \
      --conf 0.35 \
      --class-id 0
"""

import argparse
import os
import sys
import cv2
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")          # GTK3Agg or Qt5Agg if TkAgg isn't available
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

from ultralytics.models.sam import SAM3VideoPredictor



def mask_to_xyxy(mask: np.ndarray, w_orig: int, h_orig: int):
    """Convert a SAM3 mask to an xyxy bbox in the original image space."""
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    m_resized = cv2.resize(m, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    binary = (m_resized > 0.5).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def xyxy_to_xywhn(x1, y1, x2, y2, W, H):
    """Normalize an xyxy bbox to xywh normalized (YOLO format)."""
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return xc / W, yc / H, w / W, h / H




class BBoxSelector:
    """
    Shows the frame in a matplotlib window and waits for the user
    to draw a rectangle (click-and-drag).
    - Close the window OR press Enter to confirm
    - Close without drawing anything to get None back
    """

    def __init__(self):
        self._bbox = None

    def _on_select(self, eclick, erelease):
        x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
        if x2 > x1 and y2 > y1:
            self._bbox = [x1, y1, x2, y2]

    def select(self, frame: np.ndarray, window_name: str = "Select BBox") -> list | None:
        """Return [x1, y1, x2, y2], or None if cancelled."""
        self._bbox = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        fig.canvas.manager.set_window_title(window_name)
        ax.imshow(rgb)
        ax.set_title("Cliquer-glisser pour tracer la bbox  |  Fermer la fenêtre pour valider",
                     fontsize=11, color="steelblue")
        ax.axis("off")

        rs = RectangleSelector(
            ax, self._on_select,
            useblit=True,
            button=[1],          # left mouse button only
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        # Pressing Enter also closes the window
        def on_key(event):
            if event.key in ("enter", "escape"):
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show(block=True)   # blocks until the window is closed

        return self._bbox


def navigate_to_reappearance(cap, start_frame: int, total_frames: int,
                             width: int, height: int):
    """
    Interactive matplotlib window for navigating the video to find
    the moment the object reappears.

    Returns (frame_idx, frame_bgr), or (None, None) on Esc/close,
            (-1, None) if the user chooses Skip.

    Keyboard controls:
      ← / →      : ±1 frame
      a / d      : ±10 frames
      q / e      : ±100 frames
      Enter      : confirm this frame
      s          : skip (auto-advance to the next frame)
      Esc        : quit the script
    """
    state = {"idx": max(0, start_frame), "result": "pending"}

    def read_frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        return f if ret else None

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    fig.canvas.manager.set_window_title("Navigation — trouver la réapparition")

    frame0 = read_frame(state["idx"])
    if frame0 is None:
        plt.close(fig)
        return None, None

    img_display = ax.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    title = ax.set_title("", fontsize=10, color="steelblue")

    def update_display():
        f = read_frame(state["idx"])
        if f is None:
            return None
        img_display.set_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        title.set_text(
            f"Frame {state['idx']}/{total_frames-1}  |  "
            "←/→ ±1  a/d ±10  q/e ±100  |  Entrée=valider  s=skip  Echap=quitter"
        )
        fig.canvas.draw_idle()
        return f

    update_display()

    def on_key(event):
        k = event.key
        i = state["idx"]

        if k == "escape":
            state["result"] = "quit"
            plt.close(fig)
        elif k in ("enter", " "):
            state["result"] = "validate"
            plt.close(fig)
        elif k == "s":
            state["result"] = "skip"
            plt.close(fig)
        elif k == "left":
            state["idx"] = max(0, i - 1);      update_display()
        elif k == "right":
            state["idx"] = min(total_frames-1, i + 1); update_display()
        elif k == "a":
            state["idx"] = max(0, i - 10);     update_display()
        elif k == "d":
            state["idx"] = min(total_frames-1, i + 10); update_display()
        elif k == "q":
            state["idx"] = max(0, i - 100);    update_display()
        elif k == "e":
            state["idx"] = min(total_frames-1, i + 100); update_display()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show(block=True)

    if state["result"] == "quit":
        return None, None
    elif state["result"] == "skip":
        return -1, None
    else:  # validate
        final_frame = read_frame(state["idx"])
        return state["idx"], final_frame




def extract_subclip(video_path: str, start_frame: int, fps: float,
                    width: int, height: int, tmp_dir: str) -> str:
    """
    Extract frames [start_frame ... end] into a temporary MP4 file.
    Returns the sub-clip's path.
    The caller is responsible for deleting the file after use.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"_subclip_from_{start_frame:06d}.mp4")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    print(f"  ↳ Sous-clip extrait : {written} frames → {tmp_path}")
    return tmp_path



def run_tracking_segment(
    video_path: str,
    start_frame: int,
    fps: float,
    bbox: list,
    conf: float,
    class_id: int,
    imgs_dir: str,
    labels_dir: str,
    lost_threshold: int,
    width: int,
    height: int,
    total_frames: int,
    tmp_dir: str,
    preview: bool = True,
) -> int:
    """
    Extracts a sub-clip starting at start_frame, runs SAM3VideoPredictor
    on it (local frame 0 = global frame start_frame), then translates local
    indices back to global indices for saving.

    Returns the global index of the last frame processed.
    """
    # ── 1. Extract the sub-clip ──────────────────────────────────────────
    print(f"  ↳ Extraction du sous-clip depuis la frame {start_frame}...")
    subclip_path = extract_subclip(video_path, start_frame, fps, width, height, tmp_dir)

    # ── 2. Run the predictor on the sub-clip ─────────────────────────────
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=644,
        model="sam3.pt",
        half=True,
        save=False,
    )
    predictor = SAM3VideoPredictor(overrides=overrides)
    results = predictor(source=subclip_path, bboxes=[bbox], stream=True)

    lost_since = 0
    last_local_idx = 0
    seg_detections = 0

    for local_idx, r in enumerate(results):
        # Global index in the original video
        global_idx = start_frame + local_idx

        has_det = (r.boxes is not None and len(r.boxes) > 0)

        if has_det:
            lost_since = 0
            seg_detections += 1

            img_path = os.path.join(imgs_dir, f"frame_{global_idx:06d}.jpg")
            cv2.imwrite(img_path, r.orig_img)

            txt_path = os.path.join(labels_dir, f"frame_{global_idx:06d}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                if r.masks is not None:
                    for i, mask in enumerate(r.masks.data.cpu().numpy()):
                        box = mask_to_xyxy(mask, width, height)
                        if box is None:
                            continue
                        xc, yc, w, h = xyxy_to_xywhn(*box, width, height)
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                else:
                    for i, box in enumerate(r.boxes.xywhn.cpu().numpy()):
                        xc, yc, w, h = box
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            if local_idx % 30 == 0:
                print(f"  ↳ Frame {global_idx:5d}/{total_frames}  (locale {local_idx})  [✓ {seg_detections} det]")
        else:
            lost_since += 1
            if lost_since % 5 == 1:
                print(f"  ↳ Frame {global_idx:5d}  PERDU {lost_since}/{lost_threshold}")

        last_local_idx = local_idx

        if lost_since >= lost_threshold:
            print(f"  ↳ Objet perdu à la frame globale {global_idx} (locale {local_idx}).")
            break

        if global_idx >= total_frames - 1:
            break

    try:
        os.remove(subclip_path)
    except OSError:
        pass

    last_global_idx = start_frame + last_local_idx
    print(f"  ↳ Segment terminé — {seg_detections} détections, dernière frame globale={last_global_idx}")
    return last_global_idx



def main():
    parser = argparse.ArgumentParser(
        description="Dataset interactif SAM3 : bbox manuelle + tracking + re-annotation"
    )
    parser.add_argument("--video",           type=str,   required=True,
                        help="Chemin vers la vidéo source")
    parser.add_argument("--output",          type=str,   required=True,
                        help="Répertoire de sortie (images/ et labels/)")
    parser.add_argument("--lost-threshold",  type=int,   default=15,
                        help="Nb de frames consécutives sans détection avant navigation (défaut: 15)")
    parser.add_argument("--conf",            type=float, default=0.35,
                        help="Seuil de confiance SAM3 (défaut: 0.35)")
    parser.add_argument("--class-id",        type=int,   default=0,
                        help="Classe YOLO à écrire dans les labels (défaut: 0)")
    parser.add_argument("--no-preview",      action="store_true",
                        help="Désactiver la fenêtre de preview pendant le tracking")
    args = parser.parse_args()

    # ── Prepare the output directories ────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    imgs_dir   = os.path.join(args.output, "images")
    labels_dir = os.path.join(args.output, "labels")
    os.makedirs(imgs_dir,   exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # ── Video info ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {args.video}")
        sys.exit(1)

    FPS    = cap.get(cv2.CAP_PROP_FPS)
    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    TMP_DIR = os.path.join(args.output, "_tmp_subclips")

    print(f"Vidéo : {WIDTH}x{HEIGHT} @ {FPS:.1f}fps — {TOTAL} frames")
    print(f"Seuil de perte : {args.lost_threshold} frames | Conf : {args.conf} | Classe : {args.class_id}\n")

    selector    = BBoxSelector()
    start_time  = datetime.now()
    current_frame = 0          # frame from which tracking (re)starts
    segment_count = 0
    
    print("Navigation initiale — choisissez la frame de départ.")
    print("   ← → (±1)  a/d (±10)  q/e (±100)  Entrée=valider  Echap=quitter\n")
 
    nav_idx, _ = navigate_to_reappearance(
        cap          = cap,
        start_frame  = 0,
        total_frames = TOTAL,
        width        = WIDTH,
        height       = HEIGHT,
    )
    if nav_idx is None:
        print("\n[!] Annulé. Fin du script.")
        cap.release()
        return
    elif nav_idx == -1:
        current_frame = 0   # skip on the first navigation -> frame 0
    else:
        current_frame = nav_idx
 
    print(f"   Frame de départ : {current_frame}\n")

    while current_frame < TOTAL:

        # Read the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"[!] Impossible de lire la frame {current_frame}. Fin.")
            break

        # Select the object to track with a bbox
        segment_count += 1
        print(f"\n══ Segment {segment_count} — Frame {current_frame}/{TOTAL-1} ══")
        print("   Tracez une bbox autour de l'objet (2 clics) ou appuyez sur Echap pour quitter.")

        ann_frame = frame.copy()
        cv2.putText(ann_frame, f"Segment {segment_count} — Frame {current_frame}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)

        bbox = selector.select(ann_frame, window_name=f"Segment {segment_count} — Sélectionner BBox")
        if bbox is None:
            print("\n[!] Sélection annulée. Fin du script.")
            break

        x1, y1, x2, y2 = bbox
        print(f"   BBox : [{x1}, {y1}, {x2}, {y2}]")

        # Start tracking with SAM3VideoPredictor
        print(f"   Lancement du tracking depuis la frame {current_frame}...")
        last_frame = run_tracking_segment(
            video_path      = args.video,
            start_frame     = current_frame,
            fps             = FPS,
            bbox            = bbox,
            conf            = args.conf,
            class_id        = args.class_id,
            imgs_dir        = imgs_dir,
            labels_dir      = labels_dir,
            lost_threshold  = args.lost_threshold,
            width           = WIDTH,
            height          = HEIGHT,
            total_frames    = TOTAL,
            tmp_dir         = TMP_DIR,
            preview         = not args.no_preview,
        )

        
        if last_frame >= TOTAL - 1:
            print("\n✅ Fin de la vidéo atteinte.")
            break

        # Finding the reappearance of the object after loss
        print(f"\n🔍 Navigation manuelle depuis la frame {last_frame}...")
        print("   Utilisez ← → (±1), A/D (±10), Q/E (±100), Entrée pour valider, S pour ignorer.")

        nav_idx, nav_frame = navigate_to_reappearance(
            cap          = cap,
            start_frame  = last_frame,
            total_frames = TOTAL,
            width        = WIDTH,
            height       = HEIGHT,
        )

        if nav_idx is None:
            print("\n[!] Navigation annulée. Fin du script.")
            break
        elif nav_idx == -1:
            print("   Skip demandé — avance d'une frame.")
            current_frame = last_frame + 1
        else:
            print(f"   Frame {nav_idx} validée comme point de réapparition.")
            current_frame = nav_idx


    cap.release()

    elapsed = datetime.now() - start_time
    n_imgs   = len(os.listdir(imgs_dir))
    n_labels = len([f for f in os.listdir(labels_dir) if os.path.getsize(os.path.join(labels_dir, f)) > 0])

    with open(os.path.join(args.output, "config.txt"), "w") as f:
        f.write(f"Date             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vidéo            : {args.video}\n")
        f.write(f"Conf             : {args.conf}\n")
        f.write(f"Classe YOLO      : {args.class_id}\n")
        f.write(f"Seuil de perte   : {args.lost_threshold} frames\n")
        f.write(f"Segments         : {segment_count}\n")
        f.write(f"Images sauvées   : {n_imgs}\n")
        f.write(f"Labels non vides : {n_labels}\n")
        f.write(f"Durée totale     : {elapsed}\n")

    print(f"\n{'='*55}")
    print(f"  Images sauvées   : {n_imgs}  →  {imgs_dir}")
    print(f"  Labels non vides : {n_labels}  →  {labels_dir}")
    print(f"  Segments         : {segment_count}")
    print(f"  Durée totale     : {elapsed}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()