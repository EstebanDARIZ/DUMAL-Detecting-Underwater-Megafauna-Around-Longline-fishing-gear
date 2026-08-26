import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import cv2
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")          # switch to GTK3Agg if TkAgg isn't available
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import tkinter as tk
from tkinter import simpledialog

import gc
import torch
from ultralytics.models.sam import SAM3VideoPredictor

_PALETTE = [
    (220,  50,  50),
    ( 50, 130, 220),
    ( 50, 200,  80),
    (230, 160,  30),
    (160,  60, 220),
    ( 30, 200, 200),
]




"""
annotator_multi_box.py
============================
Interactive workflow for building a detection dataset with SAM3.
Supports tracking N objects simultaneously.

Steps:
  1. Initial navigation to pick the starting frame
  2. Select N bboxes (one window per object, previous ones shown)
  3. SAM3VideoPredictor tracks all objects in parallel over a sub-clip
  4. As soon as an object exceeds lost_threshold frames with no detection:
     navigate to find the frame where the objects reappear,
     re-select the N bboxes, start a new segment
  5. Repeat until the end of the video

Keyboard controls (navigation):
  ← / →   : ±1 frame        a / d : ±10 frames
  q / e   : ±100 frames
  Enter   : confirm the current frame
  s       : skip (resumes from last_frame + 1)
  Esc     : quit the script

Output:
  <output_dir>/images/frame_XXXXXX.jpg
  <output_dir>/labels/frame_XXXXXX.txt  (YOLO: cls xc yc w h, one line per object)

Usage:
  # 3 objects, same class for all
  python sam3_interactive_dataset.py --video v.mp4 --output out/ \
      --n-objects 3 --class-ids 0 0 0 --lost-threshold 20 --conf 0.40

  # Replace mode (overwrites existing files)
  python sam3_interactive_dataset.py --video v.mp4 --output out/ --merge-mode replace
"""



def mask_to_xyxy(mask: np.ndarray, w: int, h: int):
    """Convert a SAM3 mask to an xyxy bbox in the original image space."""
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    coords = np.column_stack(np.where(m > 0.5))
    if coords.shape[0] == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def xyxy_to_xywhn(x1, y1, x2, y2, W, H):
    w = x2 - x1;  h = y2 - y1
    return (x1 + w/2)/W, (y1 + h/2)/H, w/W, h/H


def obj_color_rgb(i):
    """Matplotlib color (0-1 range) for object i."""
    r, g, b = _PALETTE[i % len(_PALETTE)]
    return r/255, g/255, b/255


def ask_tracking_params() -> tuple[int, list[int]] | None:
    """
    Tkinter dialogs asking for:
      - the number of objects to track
      - the YOLO class for each object
    Returns (n_objects, class_ids), or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    n = simpledialog.askinteger(
        "SAM3 — Paramètres",
        "Nombre d'objets à tracker :",
        minvalue=1, maxvalue=20, parent=root
    )
    if n is None:
        root.destroy()
        return None

    class_ids = []
    for k in range(n):
        cid = simpledialog.askinteger(
            "SAM3 — Paramètres",
            f"Classe YOLO pour l'objet {k+1}/{n} :",
            minvalue=0, maxvalue=999, parent=root
        )
        if cid is None:
            root.destroy()
            return None
        class_ids.append(cid)

    root.destroy()
    return n, class_ids


class _SingleSelector:
    """Select a single bbox via click-and-drag."""
    def __init__(self):
        self._bbox = None

    def _cb(self, ec, er):
        x1 = int(min(ec.xdata, er.xdata));  y1 = int(min(ec.ydata, er.ydata))
        x2 = int(max(ec.xdata, er.xdata));  y2 = int(max(ec.ydata, er.ydata))
        if x2 > x1 and y2 > y1:
            self._bbox = [x1, y1, x2, y2]

    def select(self, frame_rgb: np.ndarray, title: str,
               done_bboxes: list, class_ids: list) -> list | None:
        self._bbox = None
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.canvas.manager.set_window_title(title)
        ax.imshow(frame_rgb);  ax.axis("off")
        ax.set_title(
            f"{title}  |  Click and drag to draw  |  Enter=validate  |  Echap=Quit",
            fontsize=10, color="steelblue"
        )

        for k, b in enumerate(done_bboxes):
            c = obj_color_rgb(k)
            ax.add_patch(plt.Rectangle(
                (b[0], b[1]), b[2]-b[0], b[3]-b[1],
                linewidth=2, edgecolor=c, facecolor="none"
            ))
            ax.text(b[0], b[1]-5,
                    f"Obj {k+1} cls={class_ids[k]}",
                    color=c, fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))

        rs = RectangleSelector(
            ax, self._cb, useblit=False, button=[1],
            minspanx=5, minspany=5, spancoords="pixels", interactive=True,
        )

        hline = ax.axhline(y=0, color="white", lw=0.7, ls="--", alpha=0.5,
                           visible=False)
        vline = ax.axvline(x=0, color="white", lw=0.7, ls="--", alpha=0.5,
                           visible=False)

        def on_move(ev):
            if ev.inaxes != ax or ev.xdata is None:
                hline.set_visible(False)
                vline.set_visible(False)
            else:
                hline.set_ydata([ev.ydata, ev.ydata])
                vline.set_xdata([ev.xdata, ev.xdata])
                hline.set_visible(True)
                vline.set_visible(True)
            fig.canvas.draw_idle()

        def on_key(ev):
            if ev.key in ("enter", "escape"):
                plt.close(fig)

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show(block=True)
        return self._bbox


def select_multiple_bboxes(frame_bgr: np.ndarray,
                            n_objects: int,
                            class_ids: list[int]) -> list[list] | None:
    """
    Opens n_objects windows in sequence.
    Returns [[x1,y1,x2,y2], ...], or None if the user cancels.
    """
    sel    = _SingleSelector()
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bboxes = []

    for k in range(n_objects):
        c_label = f"Object {k+1}/{n_objects}  (classe {class_ids[k]})"
        print(f"   → Draw a box for the class {c_label}")
        bbox = sel.select(rgb, c_label, bboxes, class_ids)
        if bbox is None:
            print(f"   [!] Cancelled on object {k+1}.")
            return None
        bboxes.append(bbox)
        print(f"  {c_label} → {bbox}")

    return bboxes


def navigate(cap, start_frame: int, total_frames: int,
             width: int, height: int):
    """
    Return (idx, frame_bgr) | (-1, None) for skip | (None, None) for quit.
    """
    state = {"idx": max(0, start_frame), "result": "pending"}

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        return f if ok else None

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.canvas.manager.set_window_title("Navigation")
    f0 = read(state["idx"])
    if f0 is None:
        plt.close(fig);  return None, None

    disp = ax.imshow(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ttl = ax.set_title("", fontsize=10, color="steelblue")

    def refresh():
        f = read(state["idx"])
        if f is None: return
        disp.set_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        ttl.set_text(
            f"Frame {state['idx']}/{total_frames-1}  |  "
            "←/→ ±1  a/d ±10  q/e ±100  |  Enter=validate  s=skip  Echap=Quit"
        )
        fig.canvas.draw_idle()

    refresh()

    def on_key(ev):
        k = ev.key;  i = state["idx"]
        if   k == "escape":       state["result"] = "quit";     plt.close(fig)
        elif k in ("enter", " "): state["result"] = "validate"; plt.close(fig)
        elif k == "s":            state["result"] = "skip";     plt.close(fig)
        elif k == "left":   state["idx"] = max(0, i-1);                refresh()
        elif k == "right":  state["idx"] = min(total_frames-1, i+1);   refresh()
        elif k == "a":      state["idx"] = max(0, i-10);               refresh()
        elif k == "d":      state["idx"] = min(total_frames-1, i+10);  refresh()
        elif k == "q":      state["idx"] = max(0, i-100);              refresh()
        elif k == "e":      state["idx"] = min(total_frames-1, i+100); refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout();  plt.show(block=True)

    if   state["result"] == "quit":     return None, None
    elif state["result"] == "skip":     return -1, None
    else:
        f = read(state["idx"])
        return state["idx"], f


def extract_subclip(video_path: str, start_frame: int, fps: float,
                    w: int, h: int, tmp_dir: str) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"_sub_{start_frame:06d}.mp4")
    cap  = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    wr   = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n    = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        wr.write(f);  n += 1
    cap.release();  wr.release()
    print(f"  ↳ Sub-Video : {n} frames → {os.path.basename(path)}")
    return path

def run_tracking_segment(
    video_path: str,
    start_frame: int,
    fps: float,
    bboxes: list[list],
    class_ids: list[int],
    conf: float,
    imgs_dir: str,
    labels_dir: str,
    lost_threshold: int,
    width: int,
    height: int,
    total_frames: int,
    tmp_dir: str,
    merge_mode: str = "merge",
    neg_bboxes: list[list] = None,
    model: str = "sam3.pt",
) -> int:
    n = len(bboxes)

    subclip = extract_subclip(video_path, start_frame, fps, width, height, tmp_dir)

    # ── Read raw frames from the source video ──
    cap_src = cv2.VideoCapture(video_path)

    overrides = dict(conf=conf, task="segment", mode="predict",
                     imgsz=644, model=model, half=True, save=False)
    predictor = SAM3VideoPredictor(overrides=overrides)
    # Build bboxes + labels (1=positive, 0=negative)
    neg_bboxes = neg_bboxes or []
    all_bboxes = bboxes + neg_bboxes
    all_labels = [1] * len(bboxes) + [0] * len(neg_bboxes)

    results = predictor(
        source=subclip,
        bboxes=all_bboxes,
        labels=all_labels,
        stream=True,
    )

    lost        = [0] * n
    dets        = [0] * n
    last_local  = 0
    stop_reason = None

    for local_idx, r in enumerate(results):
        global_idx = start_frame + local_idx

        # ── Raw frame at the original resolution ──
        cap_src.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
        ok, frame_raw = cap_src.read()
        if not ok or frame_raw is None:
            frame_raw = r.orig_img  # fallback only if the read fails

        detected = [False] * n
        if r.boxes is not None and len(r.boxes) > 0:
            for cls_val in r.boxes.cls.cpu().numpy():
                obj_i = int(cls_val)
                if 0 <= obj_i < n:
                    detected[obj_i] = True

        for i in range(n):
            if detected[i]: lost[i] = 0;  dets[i] += 1
            else:           lost[i] += 1

        if any(detected):
            img_path = os.path.join(imgs_dir, f"frame_{global_idx:06d}.jpg")
            # ── frame_raw (source), max quality ──
            cv2.imwrite(img_path, frame_raw)

            txt_path = os.path.join(labels_dir, f"frame_{global_idx:06d}.txt")

            nouvelles_lignes = []
            src_masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            cls_np    = r.boxes.cls.cpu().numpy()
            xywhn_np  = r.boxes.xywhn.cpu().numpy()

            for di in range(len(cls_np)):
                obj_i = int(cls_np[di])
                if obj_i >= n:
                    continue
                if src_masks is not None:
                    box = mask_to_xyxy(src_masks[di], width, height)
                    if box is None:
                        continue
                    xc, yc, bw, bh = xyxy_to_xywhn(*box, width, height)
                else:
                    xc, yc, bw, bh = xywhn_np[di]
                nouvelles_lignes.append(
                    (class_ids[obj_i], f"{class_ids[obj_i]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                )

            if not os.path.exists(txt_path):
                with open(txt_path, "w", encoding="utf-8") as f:
                    for _, ligne in nouvelles_lignes:
                        f.write(ligne)

            elif merge_mode == "replace":
                classes_a_remplacer = {cls_id for cls_id, _ in nouvelles_lignes}
                with open(txt_path, "r", encoding="utf-8") as f:
                    lignes_conservees = [
                        l for l in f
                        if l.strip() and int(l.split()[0]) not in classes_a_remplacer
                    ]
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.writelines(lignes_conservees)
                    for _, ligne in nouvelles_lignes:
                        f.write(ligne)
            
            # ##############################################################################
            else:  # merge
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.writelines(ligne for _, ligne in nouvelles_lignes)
            # ##############################################################################



        if local_idx % 30 == 0:
            st = "  ".join(
                f"obj{i+1}={'✓' if detected[i] else f'✗{lost[i]}/{lost_threshold}'}"
                for i in range(n)
            )
            print(f"  ↳ Frame {global_idx:5d}/{total_frames}  [{st}]")

        last_local = local_idx

        trigger = [i for i in range(n) if lost[i] >= lost_threshold]
        if trigger:
            names = ", ".join(f"obj{i+1}(cls {class_ids[i]})" for i in trigger)
            print(f"  ↳ Lost objects : {names}  at frame {global_idx}")
            stop_reason = "lost"
            try: results.close()
            except Exception: pass
            break

        if global_idx >= total_frames - 1:
            stop_reason = "end"
            break

    # ── Clean up after the loop ──
    cap_src.release()

    try:
        del predictor
        del results
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    try: os.remove(subclip)
    except OSError: pass

    last_global = start_frame + last_local
    summary     = "  ".join(f"obj{i+1}:{dets[i]}det" for i in range(n))
    print(f"  ↳ Segment completed ({stop_reason}) — {summary} — frame={last_global}")

    return last_global


def main():
    parser = argparse.ArgumentParser(description="Dataset interactif SAM3 — tracking multi-objets")
    parser.add_argument("--video",          type=str,   required=True)
    parser.add_argument("--output",         type=str,   required=True)
    parser.add_argument("--lost-threshold", type=int,   default=15,
                        help="Frames consécutives sans détection avant navigation (défaut: 15)")
    parser.add_argument("--conf",           type=float, default=0.35,
                        help="Seuil de confiance SAM3 (défaut: 0.35)")
    parser.add_argument("--merge-mode",     type=str,   default="merge",
                        choices=["merge", "replace"],
                        help="Fusion des labels existants : 'merge' ajoute si classe absente, "
                             "'replace' écrase tout (défaut: merge)")
    parser.add_argument("--model",          type=str,   default="sam3.pt",
                        help="Poids SAM3 a utiliser (defaut: sam3.pt)")
    args = parser.parse_args()

    print(f"Merge mode  : {args.merge_mode}")

    os.makedirs(args.output, exist_ok=True)
    imgs_dir   = os.path.join(args.output, "images")
    labels_dir = os.path.join(args.output, "labels")
    tmp_dir    = os.path.join(args.output, "_tmp")
    for d in (imgs_dir, labels_dir): os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERREUR] Can't open : {args.video}"); sys.exit(1)

    FPS    = cap.get(cv2.CAP_PROP_FPS)
    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info  : {WIDTH}x{HEIGHT} @ {FPS:.1f}fps — {TOTAL} frames")
    print(f"Video path  : {args.video}")
    print(f"Tracker     : lost-threshold={args.lost_threshold}  conf={args.conf}\n")

    t0 = datetime.now()
    segment_count = 0

    print("Choose a frame to start from:\n")
    nav_idx, _ = navigate(cap, 0, TOTAL, WIDTH, HEIGHT)
    if nav_idx is None:
        print("[!] Annulé."); cap.release(); return
    current_frame = 0 if nav_idx == -1 else nav_idx
    print(f"You chose frame : {current_frame}\n")

    while current_frame < TOTAL:

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok:
            print(f"[!] Can't read frame {current_frame}."); break

        segment_count += 1
        print(f"\n══ Iteration {segment_count} — Frame {current_frame}/{TOTAL-1} ══")
        params = ask_tracking_params()
        if params is None:
            print("[!] Cancelled. End."); break
        n, class_ids = params
        print(f"  Objets : {n}   classes : {class_ids}")

        bboxes = select_multiple_bboxes(frame, n, class_ids)
        if bboxes is None:
            print("[!] Cancelled. End."); break

        last_frame = run_tracking_segment(
            video_path      = args.video,
            start_frame     = current_frame,
            fps             = FPS,
            bboxes          = bboxes,
            class_ids       = class_ids,
            conf            = args.conf,
            imgs_dir        = imgs_dir,
            labels_dir      = labels_dir,
            lost_threshold  = args.lost_threshold,
            width           = WIDTH,
            height          = HEIGHT,
            total_frames    = TOTAL,
            tmp_dir         = tmp_dir,
            merge_mode      = args.merge_mode,
            model           = args.model,
        )

        if last_frame >= TOTAL - 1:
            print("\nEnd of video."); break

        print(f"\nNavigation depuis la frame {last_frame}.")
        nav_idx, _ = navigate(cap, last_frame, TOTAL, WIDTH, HEIGHT)

        if nav_idx is None:
            print("[!] End."); break
        elif nav_idx == -1:
            current_frame = last_frame + 1
        else:
            current_frame = nav_idx
            print(f"  Frame {nav_idx} choisie.\n")

    cap.release()

    elapsed  = datetime.now() - t0
    n_imgs   = len(os.listdir(imgs_dir))
    n_labels = sum(
        1 for f in os.listdir(labels_dir)
        if os.path.getsize(os.path.join(labels_dir, f)) > 0
    )

    with open(os.path.join(args.output, "config.txt"), "w") as f:
        f.write(f"Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video          : {args.video}\n")
        f.write(f"Conf           : {args.conf}\n")
        f.write(f"Lost threshold : {args.lost_threshold}\n")
        f.write(f"Merge mode     : {args.merge_mode}\n")
        f.write(f"Segments       : {segment_count}\n")
        f.write(f"Images         : {n_imgs}\n")
        f.write(f"Labels         : {n_labels}\n")
        f.write(f"Time           : {elapsed}\n")

    print(f"\n{'='*50}")
    print(f"  Images     : {n_imgs}  →  {imgs_dir}")
    print(f"  Labels     : {n_labels}  →  {labels_dir}")
    print(f"  Segments   : {segment_count}")
    print(f"  Merge mode : {args.merge_mode}")
    print(f"  Durée      : {elapsed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()