#!/usr/bin/env python3
"""
Bounding-box annotation tool, IAA (dual panel, N pairs).

LEFT panel: video clip (clips_iou/<stem>.mp4), auto-loops, no controls.
RIGHT panel: frozen image (images/<stem>.jpg) to annotate, with zoom and pan.

Each drawn bbox gets a sequential number (1, 2, 3...), no dialog.

Usage:
    python bbox_annotator.py /path/to/IAA_folder
    python bbox_annotator.py /path/to/IAA_folder --output /path/to/labels_dir
    python bbox_annotator.py /path/to/IAA_folder --output /path/to/labels_dir --start 42

Keyboard shortcuts:
    Enter / Down   NEXT pair (auto-saves)
    Up / B         PREVIOUS pair
    N              toggle bbox draw mode
    K              toggle eraser mode
    + / =          zoom in (centered on the view)
    -              zoom out
    .              reset zoom
    Ctrl+Scroll    zoom centered on the mouse
    Left click     draw a bbox (draw mode) / move-resize a bbox
                   click on background in normal mode -> pan the view
    Right click    select, then delete a bbox
    Ctrl+Z         undo
    R              toggle box visibility
    Delete         delete the selected bbox
    Esc            cancel current action / quit
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button


# ─── Constants ─────────────────────────────────────────────────────────────────

_PALETTE = [
    (0.20, 0.78, 0.31), (0.20, 0.51, 0.86), (0.90, 0.63, 0.12),
    (0.86, 0.20, 0.20), (0.63, 0.24, 0.86), (0.12, 0.78, 0.78),
    (0.86, 0.20, 0.63), (0.55, 0.82, 0.12), (0.78, 0.51, 0.20),
]

IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}
HANDLE_RADIUS = 8
NEW_BOX_MIN   = 5
FRAME_MS      = 33          # ~30 fps
PROG_X0       = 0.03
PROG_W        = 0.94
ZOOM_STEP     = 1.5         # zoom factor per key press
MIN_VIEW_PX   = 20          # minimum view size in image pixels (avoids over-zooming)


def palette(box_num: int):
    return _PALETTE[(box_num - 1) % len(_PALETTE)]


# ─── YOLO utilities ────────────────────────────────────────────────────────────

def yolo_to_xyxy(xc, yc, w, h, W, H):
    return (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H


def read_labels(path):
    if not os.path.exists(path):
        return []
    labels = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                labels.append([int(p[0])] + [float(v) for v in p[1:]])
    return labels


def write_labels(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for lb in labels:
            f.write(f"{int(lb[0])} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f} {lb[4]:.6f}\n")


def get_handle(mx, my, x1, y1, x2, y2):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for name, (hx, hy) in {
        "tl": (x1, y1), "tr": (x2, y1), "bl": (x1, y2), "br": (x2, y2),
        "tc": (cx, y1), "bc": (cx, y2), "lc": (x1, cy), "rc": (x2, cy),
    }.items():
        if abs(mx - hx) < HANDLE_RADIUS and abs(my - hy) < HANDLE_RADIUS:
            return name
    if x1 <= mx <= x2 and y1 <= my <= y2:
        return "inside"
    return None


def apply_handle_drag(handle, dx, dy, x1, y1, x2, y2, W, H):
    if   handle == "inside": x1 += dx; x2 += dx; y1 += dy; y2 += dy
    elif handle == "tl":  x1 += dx; y1 += dy
    elif handle == "tr":  x2 += dx; y1 += dy
    elif handle == "bl":  x1 += dx; y2 += dy
    elif handle == "br":  x2 += dx; y2 += dy
    elif handle == "tc":  y1 += dy
    elif handle == "bc":  y2 += dy
    elif handle == "lc":  x1 += dx
    elif handle == "rc":  x2 += dx
    x1 = max(0, min(x1, W - 1));  x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1));  y2 = max(0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


# ─── Image/clip pairs ───────────────────────────────────────────────────────────

def build_pairs(iaa_folder):
    imgs_dir  = os.path.join(iaa_folder, "images")
    clips_dir = os.path.join(iaa_folder, "clips_iou")
    for d, name in [(imgs_dir, "images/"), (clips_dir, "clips_iou/")]:
        if not os.path.isdir(d):
            print(f"[ERROR] Dossier {name} introuvable dans : {iaa_folder}"); sys.exit(1)

    clip_stems = {}
    for f in os.listdir(clips_dir):
        stem, ext = os.path.splitext(f)
        if ext.lstrip(".") in {e.lstrip(".") for e in VIDEO_EXTS}:
            clip_stems[stem] = os.path.join(clips_dir, f)

    pairs = []
    for f in sorted(os.listdir(imgs_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMAGE_EXTS:
            continue
        if stem not in clip_stems:
            print(f"  [SKIP] {f} — pas de clip dans clips_iou/"); continue
        pairs.append({"stem": stem,
                      "img_path":  os.path.join(imgs_dir, f),
                      "clip_path": clip_stems[stem]})

    if not pairs:
        print(f"[ERROR] Aucune paire dans : {iaa_folder}"); sys.exit(1)
    print(f"  {len(pairs)} paires image/clip trouvées.")
    return pairs


# ─── Main tool ──────────────────────────────────────────────────────────────────

class BboxAnnotator:

    def __init__(self, iaa_folder, output_dir):
        self.iaa_folder = iaa_folder
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pairs    = build_pairs(iaa_folder)
        self.pair_idx = 0
        self.n_pairs  = len(self.pairs)

        # ── Video (auto-play, no controls) ────
        self.cap             = None
        self.total_frames    = 1
        self.vid_idx         = 0
        self._cap_next       = 0       # index of the next cap.read() (used by the reader thread)
        self._freeze_idx     = 0       # clip frame matching the annotated image
        self._loop_start     = 0       # loop start (~3s before the freeze)
        self._timer          = None
        self._rendering      = False
        self._vid_bg         = None    # blit background captured after each full draw
        self._vid_img_artist = None    # animated AxesImage (excluded from the blit background)
        self._vid_frame_shape= None
        self._prog_fill      = None    # animated Rectangle (progress bar)
        self._prog_label     = None    # animated Text
        self._prog_dragging  = False   # True while dragging on the progress bar
        # Reader thread: decodes frames ahead of time into a buffer
        self._frame_buf      = deque(maxlen=4)   # (frame_idx, bgr_array)
        self._reader_stop    = threading.Event()
        self._reader_thread  = None
        # Cache freeze_idx to avoid re-searching on every pair change
        self._freeze_cache: dict[str, int] = {}

        # ── Annotation ─────────────────────────────────────
        self.img_W = self.img_H    = 720
        self.labels: list          = []
        self.sel_bbox              = None
        self.drag_handle           = None
        self.drag_start            = None
        self.drag_orig             = None
        self.new_box_start         = None
        self._last_mouse           = None
        self._draw_mode            = False
        self._erase_mode           = False
        self._hide_boxes           = False
        self._ctrl                 = False
        self._undo_stack: list     = []
        self._current_bgr          = None
        # Pan
        self._panning              = False
        self._pan_start_px         = None
        self._pan_xlim             = None
        self._pan_ylim             = None
        self._pan_axes_w           = 1.0
        self._pan_axes_h           = 1.0
        # Blit overlay for the annotation panel (crosshair + draw/drag preview)
        self._ch_h                 = None   # animated axhline
        self._ch_v                 = None   # animated axvline
        self._drag_rect_artist     = None   # animated rectangle (draw preview + drag ghost)
        self._current_rgb          = None   # BGR->RGB cache computed once per pair
        self._last_drag_time       = 0.0    # throttle drag redraw
        self._last_pan_time        = 0.0    # throttle pan redraw

        self._build_ui()
        self._load_pair(0)

    def _next_num(self):
        return max((int(lb[0]) for lb in self.labels), default=0) + 1

    # ═══════════════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════════════
    def _build_ui(self):
        plt.rcParams.update({
            "figure.facecolor": "#0d1117",
            "axes.facecolor":   "#0d1117",
            "text.color":       "#c9d1d9",
            "axes.edgecolor":   "#30363d",
        })

        self.fig = plt.figure(figsize=(19, 9))
        self.fig.canvas.manager.set_window_title(
            f"IAA Annotator — {os.path.basename(self.iaa_folder)}")
        self.fig.patch.set_facecolor("#0d1117")

        gs = gridspec.GridSpec(
            3, 2,
            height_ratios=[0.875, 0.052, 0.073],
            wspace=0.025, hspace=0.05,
            left=0.02, right=0.98, top=0.945, bottom=0.055,
        )

        self.ax_vid = self.fig.add_subplot(gs[0, 0])
        self.ax_vid.axis("off")
        self.ax_ann = self.fig.add_subplot(gs[0, 1])
        self.ax_ann.axis("off")

        # ── Video progress bar (display-only, animated) ──
        ax_prog = self.fig.add_subplot(gs[1, 0])
        ax_prog.set_facecolor("#161b22")
        ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1)
        ax_prog.axis("off")
        ax_prog.add_patch(mpatches.Rectangle(
            (PROG_X0, 0.25), PROG_W, 0.50, facecolor="#21262d", zorder=0))
        self._prog_fill = mpatches.Rectangle(
            (PROG_X0, 0.25), 0.0, 0.50, facecolor="#1f6feb", zorder=1, animated=True)
        ax_prog.add_patch(self._prog_fill)
        self._prog_label = ax_prog.text(
            0.5, 0.5, "0 / 0", ha="center", va="center",
            fontsize=8, color="#58a6ff", fontfamily="monospace",
            zorder=2, animated=True)
        self.ax_prog = ax_prog

        # ── Pair slider (right side, interactive) ──────────────
        ax_ps = self.fig.add_subplot(gs[1, 1])
        ax_ps.set_facecolor("#161b22")
        self.pair_slider = Slider(ax_ps, "", 0, max(self.n_pairs - 1, 1),
                                  valinit=0, valstep=1, color="#238636")
        self.pair_slider.label.set_color("#8b949e")
        self.pair_slider.valtext.set_color("#00ff88")
        self.pair_slider.on_changed(self._on_pair_slider)

        # ── Buttons (annotation only, no video controls) ──
        fl, fr = self.fig.subplotpars.left, self.fig.subplotpars.right
        fw = fr - fl
        btn_defs = [
            ("◁ paire",   "#1f3a1f", lambda _: self._pair_go(-1)),
            ("paire ▷",   "#1f3a1f", lambda _: self._pair_go(1)),
            ("Draw [N]",  "#2d1b4e", self._btn_draw),
            ("Erase [K]", "#4a1010", self._btn_erase),
            ("Del bbox",  "#4a1010", self._btn_delete_bbox),
            ("Zoom [.]",  "#1a2a3a", lambda _: self._zoom_reset()),
        ]
        self._buttons = {}
        n = len(btn_defs)
        margin, bw = 0.008, (1.0 - 0.008 * (n + 1)) / n
        for k, (label, color, cb) in enumerate(btn_defs):
            bx = margin + k * (bw + margin)
            ax_b = self.fig.add_axes([fl + bx * fw, 0.005, bw * fw, 0.044])
            btn = Button(ax_b, label, color=color, hovercolor="#2d333b")
            btn.label.set_color("#c9d1d9"); btn.label.set_fontsize(8)
            btn.label.set_fontfamily("monospace")
            btn.on_clicked(cb)
            self._buttons[label] = btn

        # ── Status texts ────────────────────────────────
        self._status_ann = self.fig.text(
            0.75, 0.984, "", ha="center", va="top",
            color="#00ff88", fontsize=8, fontfamily="monospace")
        self._status_vid_txt = self.fig.text(
            0.26, 0.984, "▶ boucle auto", ha="center", va="top",
            color="#484f58", fontsize=8, fontfamily="monospace")
        self.fig.text(
            0.02, 0.002,
            "Entrée/↓=suivant  ↑/B=précédent  N=draw  K=erase  +/-/.=zoom  "
            "Ctrl+Scroll=zoom  pan=clic fond  Ctrl+Z=undo  R=boxes  Esc=quitter",
            ha="left", va="bottom", fontsize=6.5,
            color="#484f58", fontfamily="monospace")

        # ── Animated overlay artists for annotation ──────────────
        self._rebuild_animated_artists()

        # ── Event connections ───────────────────────────────────────
        self.fig.canvas.mpl_connect("draw_event",           self._on_draw_event)
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)
        self.fig.canvas.mpl_connect("key_release_event",    self._on_key_release)
        self.fig.canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("scroll_event",         self._on_scroll)

    def _on_draw_event(self, _ev):
        """Recapture the blit background after every full redraw."""
        try:
            self._vid_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._vid_bg = None

    # ═══════════════════════════════════════════════════════
    #  Loading a pair
    # ═══════════════════════════════════════════════════════
    def _load_pair(self, idx):
        self.pair_idx = max(0, min(idx, self.n_pairs - 1))
        pair = self.pairs[self.pair_idx]

        # Reset annotation state
        self.sel_bbox      = None
        self._undo_stack   = []
        self.new_box_start = None
        self._draw_mode    = False
        self._erase_mode   = False
        self._panning      = False
        self._set_cursor()

        # Image
        img = cv2.imread(pair["img_path"])
        if img is not None:
            self.img_H, self.img_W = img.shape[:2]
            self._current_bgr = img
            self._current_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self._current_bgr = None
            self._current_rgb = None
            print(f"[WARNING] Image illisible : {pair['img_path']}")
        self.labels = read_labels(self._label_path())

        # Clip
        self._stop_timer()
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(pair["clip_path"])
        if self.cap.isOpened():
            self.total_frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        else:
            print(f"[WARNING] Clip illisible : {pair['clip_path']}")
            self.cap = None
            self.total_frames = 1

        # Reset video artist (forces recreation)
        self._vid_img_artist  = None
        self._vid_frame_shape = None
        self._vid_bg          = None
        self.vid_idx          = 0
        self._cap_next        = 0

        # Pair slider
        self.pair_slider.eventson = False
        self.pair_slider.set_val(self.pair_idx)
        self.pair_slider.eventson = True

        # Find the freeze frame (cached, template-matched only once per pair)
        stem = pair["stem"]
        if stem in self._freeze_cache:
            self._freeze_idx = self._freeze_cache[stem]
        else:
            self._freeze_idx = self._find_freeze_frame()
            self._freeze_cache[stem] = self._freeze_idx
        clip_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 25.0
        pre_frames = int(3.0 * clip_fps)   # 3 seconds of context before the freeze
        self._loop_start = max(0, self._freeze_idx - pre_frames)
        self._show_frame_static(self._freeze_idx)

        # Annotation panel + zoom reset
        self._reset_ann_view()     # set_xlim/ylim based on the image size
        self._update_annot()       # draw_idle

        # Full render -> draw_event -> _vid_bg captured
        self.fig.canvas.draw()

        # Start the reader thread (it positions the cap at _loop_start itself)
        self._start_timer()

        print(f"  Paire {self.pair_idx + 1}/{self.n_pairs} : {pair['stem']}  "
              f"({len(self.labels)} bbox(s))")

    def _find_freeze_frame(self) -> int:
        """Template-match the annotated image against the clip's frames.

        The clip layout is [lead][frozen freeze][tail]. We look for the first
        frame of the freeze section (= the plain image, no boxes) by minimizing
        mean squared error on a 64x36 thumbnail.
        Typical cost: ~50 seeks on a 200-frame clip, under 200 ms.
        """
        if not self.cap or self._current_bgr is None:
            return self.total_frames // 2

        thumb_w, thumb_h = 64, 36
        ref = cv2.resize(self._current_bgr, (thumb_w, thumb_h),
                         interpolation=cv2.INTER_AREA).astype("float32")

        best_err = float("inf")
        best_idx = self.total_frames // 2
        step = max(1, self.total_frames // 50)

        # Coarse pass
        for i in range(0, self.total_frames, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            f = cv2.resize(frame, (thumb_w, thumb_h),
                           interpolation=cv2.INTER_AREA).astype("float32")
            err = float(((f - ref) ** 2).mean())
            if err < best_err:
                best_err, best_idx = err, i

        # Fine pass around the best candidate
        lo = max(0, best_idx - step)
        hi = min(self.total_frames - 1, best_idx + step)
        for i in range(lo, hi + 1):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            f = cv2.resize(frame, (thumb_w, thumb_h),
                           interpolation=cv2.INTER_AREA).astype("float32")
            err = float(((f - ref) ** 2).mean())
            if err < best_err:
                best_err, best_idx = err, i

        print(f"  freeze détecté : frame {best_idx}/{self.total_frames} "
              f"(err={best_err:.1f})")
        return best_idx

    def _show_frame_static(self, frame_idx):
        """Seek + display in ax_vid without blit (for initialization)."""
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None: return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax_vid.clear(); self.ax_vid.axis("off")
        stem = self.pairs[self.pair_idx]["stem"]
        self.ax_vid.set_title(f"CLIP   {stem}.mp4",
                              color="#484f58", fontsize=8,
                              fontfamily="monospace", pad=3)
        self._vid_img_artist  = self.ax_vid.imshow(rgb, aspect="equal", animated=True)
        self._vid_frame_shape = rgb.shape
        self.vid_idx = frame_idx
        self._set_prog(frame_idx)

    def _pair_go(self, delta):
        self._save_annotation()
        self._load_pair(self.pair_idx + delta)

    def _on_pair_slider(self, val):
        new = int(round(val))
        if new != self.pair_idx:
            self._save_annotation()
            self._load_pair(new)

    # ═══════════════════════════════════════════════════════
    #  LEFT PANEL, video blit at 30 fps, auto-loop
    # ═══════════════════════════════════════════════════════
    def _set_prog(self, frame_idx):
        self._prog_fill.set_width(PROG_W * frame_idx / max(self.total_frames - 1, 1))
        self._prog_label.set_text(f"{frame_idx} / {self.total_frames - 1}")

    def _blit_vid(self, frame_bgr, frame_idx):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._vid_img_artist is None or rgb.shape != self._vid_frame_shape:
            self.ax_vid.clear(); self.ax_vid.axis("off")
            stem = self.pairs[self.pair_idx]["stem"]
            self.ax_vid.set_title(f"CLIP   {stem}.mp4",
                                  color="#484f58", fontsize=8,
                                  fontfamily="monospace", pad=3)
            self._vid_img_artist  = self.ax_vid.imshow(rgb, aspect="equal", animated=True)
            self._vid_frame_shape = rgb.shape
            self._vid_bg = None

        self._vid_img_artist.set_data(rgb)
        self._set_prog(frame_idx)

        if self._vid_bg is not None:
            self.fig.canvas.restore_region(self._vid_bg)
            self.ax_vid.draw_artist(self._vid_img_artist)
            self.ax_prog.draw_artist(self._prog_fill)
            self.ax_prog.draw_artist(self._prog_label)
            # Keep the annotation overlay (crosshair, draw preview) alive
            # so it doesn't get erased on every video tick
            if self._ch_h is not None and self._ch_h.get_alpha() > 0:
                self.ax_ann.draw_artist(self._ch_h)
                self.ax_ann.draw_artist(self._ch_v)
            if self._drag_rect_artist is not None and self._drag_rect_artist.get_visible():
                self.ax_ann.draw_artist(self._drag_rect_artist)
            self.fig.canvas.blit(self.fig.bbox)
        else:
            self.fig.canvas.draw_idle()

    def _reader_loop(self, cap, start_pos, loop_start, total_frames, stop_event, buf):
        """Dedicated thread: reads frames ahead of time and pushes them into the buffer.
        Starts at start_pos, loops back to loop_start."""
        idx = start_pos
        while not stop_event.is_set():
            if len(buf) >= buf.maxlen:
                stop_event.wait(0.005)   # buffer full, wait 5 ms
                continue
            if idx >= total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start)
                idx = loop_start
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start)
                idx = loop_start
                continue
            buf.append((idx, frame))
            idx += 1

    def _tick(self):
        """Timer callback: takes the next frame from the buffer (non-blocking)."""
        if self._rendering or not self._frame_buf:
            return
        self._rendering = True
        try:
            frame_idx, frame = self._frame_buf.popleft()
            self.vid_idx = frame_idx
            self._blit_vid(frame, frame_idx)
        finally:
            self._rendering = False

    def _start_timer(self):
        self._stop_timer()
        # Start the reader thread
        self._reader_stop.clear()
        self._frame_buf.clear()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._loop_start)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self.cap, self._loop_start, self._loop_start, self.total_frames,
                  self._reader_stop, self._frame_buf),
            daemon=True,
        )
        self._reader_thread.start()
        self._timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _stop_timer(self):
        if self._timer:
            self._timer.stop(); self._timer = None
        self._reader_stop.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.08)   # 80ms max, the thread notices the event quickly
        self._reader_thread = None
        self._frame_buf.clear()

    def _prog_hit(self, event):
        """Return the frame_idx if the event falls on ax_prog, else None."""
        if event.inaxes is not self.ax_prog or event.xdata is None:
            return None
        if self.total_frames <= 0:
            return None
        frac = (event.xdata - PROG_X0) / PROG_W
        return int(max(0.0, min(1.0, frac)) * (self.total_frames - 1))

    def _seek_video(self, frame_idx):
        """Reposition the reader thread at frame_idx without stopping the timer."""
        self._reader_stop.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.08)
        self._frame_buf.clear()
        self._reader_stop.clear()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self.cap, frame_idx, self._loop_start, self.total_frames,
                  self._reader_stop, self._frame_buf),
            daemon=True,
        )
        self._reader_thread.start()

    def _pause_at(self, frame_idx):
        """Stop the reader and display only frame_idx (scrub pause)."""
        self._reader_stop.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.08)
        self._frame_buf.clear()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.vid_idx = frame_idx
                self._blit_vid(frame, frame_idx)

    def _resume_from(self, frame_idx):
        """Restart the reader from frame_idx after a scrub pause."""
        self._reader_stop.clear()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self.cap, frame_idx, self._loop_start, self.total_frames,
                  self._reader_stop, self._frame_buf),
            daemon=True,
        )
        self._reader_thread.start()

    # ═══════════════════════════════════════════════════════
    #  Animated overlay artists for annotation (fast blit)
    # ═══════════════════════════════════════════════════════
    def _rebuild_animated_artists(self):
        """Recreate the animated artists after each ax_ann.clear()."""
        W, H = self.img_W, self.img_H
        # Lines in data coordinates, exactly covering the whole image
        self._ch_h = Line2D([0, W], [0, 0],
                            color="white", lw=1.2, ls="-", alpha=0.0,
                            animated=True, zorder=20)
        self._ch_v = Line2D([0, 0], [0, H],
                            color="white", lw=1.2, ls="-", alpha=0.0,
                            animated=True, zorder=20)
        self.ax_ann.add_line(self._ch_h)
        self.ax_ann.add_line(self._ch_v)
        self._drag_rect_artist = mpatches.Rectangle(
            (0, 0), 0, 0, lw=2, edgecolor=(0.0, 1.0, 0.5),
            facecolor=(0.0, 1.0, 0.5, 0.10), animated=True, visible=False, zorder=20)
        self.ax_ann.add_patch(self._drag_rect_artist)

    def _blit_ann_overlay(self):
        """Fast blit: crosshair + draw/drag preview, without redrawing the whole figure."""
        if self._vid_bg is None:
            self.fig.canvas.draw_idle()
            return
        self.fig.canvas.restore_region(self._vid_bg)
        # Keep the video visible (animated artist not included in _vid_bg)
        if self._vid_img_artist is not None:
            self.ax_vid.draw_artist(self._vid_img_artist)
        self.ax_prog.draw_artist(self._prog_fill)
        self.ax_prog.draw_artist(self._prog_label)
        # Annotation overlay
        if self._ch_h is not None:
            self.ax_ann.draw_artist(self._ch_h)
            self.ax_ann.draw_artist(self._ch_v)
        if self._drag_rect_artist is not None:
            self.ax_ann.draw_artist(self._drag_rect_artist)
        self.fig.canvas.blit(self.fig.bbox)

    # ═══════════════════════════════════════════════════════
    #  RIGHT PANEL, annotation with zoom/pan
    # ═══════════════════════════════════════════════════════
    def _label_path(self):
        return os.path.join(self.output_dir, self.pairs[self.pair_idx]["stem"] + ".txt")

    def _save_annotation(self):
        if self.labels:
            write_labels(self._label_path(), self.labels)
        else:
            p = self._label_path()
            if os.path.exists(p): os.remove(p)

    def _reset_ann_view(self):
        """Reset the zoom to show the full image."""
        self.ax_ann.set_xlim(0, self.img_W)
        self.ax_ann.set_ylim(self.img_H, 0)   # Y axis inverted (imshow)

    def _zoom_at(self, factor, cx=None, cy=None):
        """
        Zoom centered on (cx, cy) in image coordinates.
        factor > 1 = zoom in, factor < 1 = zoom out.
        """
        xl, xr = self.ax_ann.get_xlim()
        yb, yt = self.ax_ann.get_ylim()    # yb > yt (inverted axis)
        if cx is None: cx = (xl + xr) / 2
        if cy is None: cy = (yb + yt) / 2

        new_xl = cx + (xl - cx) / factor
        new_xr = cx + (xr - cx) / factor
        new_yb = cy + (yb - cy) / factor
        new_yt = cy + (yt - cy) / factor

        # Image bounds
        new_xl = max(0, new_xl);     new_xr = min(self.img_W, new_xr)
        new_yt = max(0, new_yt);     new_yb = min(self.img_H, new_yb)

        # Minimum view size
        if new_xr - new_xl < MIN_VIEW_PX or new_yb - new_yt < MIN_VIEW_PX:
            return

        self.ax_ann.set_xlim(new_xl, new_xr)
        self.ax_ann.set_ylim(new_yb, new_yt)
        self._vid_bg = None
        self.fig.canvas.draw_idle()

    def _zoom_reset(self):
        self._reset_ann_view()
        self._vid_bg = None
        self.fig.canvas.draw_idle()

    def _update_annot(self):
        """Redraw the right panel. Invalidates the blit background."""
        W, H = self.img_W, self.img_H

        # Save the current zoom
        xlim = self.ax_ann.get_xlim()
        ylim = self.ax_ann.get_ylim()

        self.ax_ann.clear(); self.ax_ann.axis("off")

        # Restore the zoom
        self.ax_ann.set_xlim(xlim)
        self.ax_ann.set_ylim(ylim)

        if self._draw_mode:
            title_str, title_col = "DRAW", "#00ff88"
        elif self._erase_mode:
            title_str, title_col = "ERASE", "#ff4444"
        else:
            title_str, title_col = "ANNOTATION", "#8b949e"
        self.ax_ann.set_title(title_str, color=title_col,
                               fontsize=9, fontfamily="monospace", pad=3)

        if self._current_rgb is not None:
            self.ax_ann.imshow(self._current_rgb, aspect="equal", extent=[0, W, H, 0])
        else:
            self.ax_ann.text(0.5, 0.5, "Image illisible", ha="center", va="center",
                             transform=self.ax_ann.transAxes,
                             color="#ff4444", fontsize=12, fontfamily="monospace")

        if not self._hide_boxes:
            for i, lb in enumerate(self.labels):
                num = int(lb[0])
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4], W, H)
                col = palette(num)
                self.ax_ann.add_patch(mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2.5 if i == self.sel_bbox else 1.5,
                    edgecolor=col, facecolor=(*col, 0.08),
                    linestyle="-" if i == self.sel_bbox else "--"))
                self.ax_ann.text(
                    x1 + 3, y1 + 3, str(num),
                    color=col, fontsize=8, fontfamily="monospace",
                    fontweight="bold", va="top",
                    bbox=dict(facecolor="#0d1117", alpha=0.65, pad=1, linewidth=0))
                if i == self.sel_bbox:
                    cx_b, cy_b = (x1 + x2) / 2, (y1 + y2) / 2
                    for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
                                   (cx_b, y1), (cx_b, y2), (x1, cy_b), (x2, cy_b)]:
                        self.ax_ann.plot(hx, hy, "o", ms=5, color="white",
                                          markerfacecolor=col, markeredgewidth=1)

        stem = self.pairs[self.pair_idx]["stem"]
        self._status_ann.set_text(
            f"{stem}  |  {self.pair_idx + 1}/{self.n_pairs}  |  "
            f"{'✓ ' if os.path.exists(self._label_path()) else ''}"
            f"{len(self.labels)} bbox(s)  [next: #{self._next_num()}]")

        self._rebuild_animated_artists()
        self._vid_bg = None
        self.fig.canvas.draw_idle()

    # ═══════════════════════════════════════════════════════
    #  Mouse
    # ═══════════════════════════════════════════════════════
    def _ann_coords(self, event):
        if event.inaxes is self.ax_ann:
            return event.xdata, event.ydata
        if (self.drag_handle is not None or self.new_box_start is not None) \
                and event.x is not None:
            try:
                inv = self.ax_ann.transData.inverted()
                x, y = inv.transform((event.x, event.y))
                return (max(0.0, min(float(self.img_W - 1), x)),
                        max(0.0, min(float(self.img_H - 1), y)))
            except Exception:
                pass
        return None

    def _on_mouse_press(self, event):
        # Click on the progress bar -> scrub pause
        frame = self._prog_hit(event)
        if frame is not None:
            self._prog_dragging = True
            self._pause_at(frame)
            return

        coords = self._ann_coords(event)
        if coords is None: return
        mx, my = coords
        W, H = self.img_W, self.img_H

        # Right click: select / delete
        if event.button == 3:
            for i, lb in enumerate(self.labels):
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4], W, H)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    if self.sel_bbox == i:
                        self._push_undo(); del self.labels[i]
                        self._save_annotation()
                        print(f"  [-] bbox #{int(lb[0])} supprimée")
                        self.sel_bbox = None
                    else:
                        self.sel_bbox = i
                    self._update_annot(); return
            return

        if event.button != 1: return

        # Eraser mode
        if self._erase_mode:
            for i, lb in enumerate(self.labels):
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4], W, H)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    self._push_undo(); del self.labels[i]
                    self._save_annotation()
                    print(f"  [-] bbox #{int(lb[0])} effacée")
                    self.sel_bbox = None; self._update_annot(); return
            return

        # Draw mode
        if self._draw_mode:
            self.sel_bbox = None; self.new_box_start = (mx, my)
            self._update_annot(); return

        # Hit-test existing bboxes
        hit_resize = hit_inside = None
        indices = list(range(len(self.labels)))
        if self.sel_bbox is not None:
            indices = [self.sel_bbox] + [i for i in indices if i != self.sel_bbox]
        for i in indices:
            lb = self.labels[i]
            x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4], W, H)
            h = get_handle(mx, my, x1, y1, x2, y2)
            if h is None: continue
            if h == "inside":
                if hit_inside is None: hit_inside = (i, x1, y1, x2, y2)
            else:
                hit_resize = (i, h, x1, y1, x2, y2); break

        if hit_resize:
            i, h, x1, y1, x2, y2 = hit_resize
            self.sel_bbox = i; self.drag_handle = h
            self.drag_start = (mx, my); self.drag_orig = (x1, y1, x2, y2)
            self._push_undo()
        elif hit_inside:
            i, x1, y1, x2, y2 = hit_inside
            self.sel_bbox = i; self.drag_handle = "inside"
            self.drag_start = (mx, my); self.drag_orig = (x1, y1, x2, y2)
            self._push_undo()
        else:
            # Background clicked -> start panning
            self.sel_bbox = None
            self._panning    = True
            self._pan_start_px = (event.x, event.y)
            self._pan_xlim   = self.ax_ann.get_xlim()
            self._pan_ylim   = self.ax_ann.get_ylim()
            bbox = self.ax_ann.get_window_extent()
            self._pan_axes_w = max(bbox.width,  1.0)
            self._pan_axes_h = max(bbox.height, 1.0)
            return          # skip _update_annot so we don't break the background
        self._update_annot()

    def _on_mouse_move(self, event):
        # Drag on the progress bar -> paused scrub
        if self._prog_dragging:
            frame = self._prog_hit(event)
            if frame is not None:
                self._pause_at(frame)
            return

        coords = self._ann_coords(event)
        if coords is not None:
            self._last_mouse = coords

        # Bbox drag, throttled to ~30 fps to avoid a full render on every pixel
        if self.drag_handle is not None and coords is not None:
            mx, my = coords
            dx, dy = mx - self.drag_start[0], my - self.drag_start[1]
            ox1, oy1, ox2, oy2 = self.drag_orig
            nx1, ny1, nx2, ny2 = apply_handle_drag(
                self.drag_handle, dx, dy, ox1, oy1, ox2, oy2, self.img_W, self.img_H)
            self.labels[self.sel_bbox][1:5] = list(
                xyxy_to_yolo(nx1, ny1, nx2, ny2, self.img_W, self.img_H))
            if event.inaxes is self.ax_ann:
                now = time.monotonic()
                if now - self._last_drag_time >= 0.033:
                    self._last_drag_time = now
                    self._update_annot()
            return

        # Pan, throttled to ~30 fps
        if self._panning and event.x is not None:
            xl, xr = self._pan_xlim
            yb, yt = self._pan_ylim    # yb > yt (inverted axis)
            dx_px = event.x - self._pan_start_px[0]
            dy_px = event.y - self._pan_start_px[1]  # positive = upward
            shift_x = dx_px / self._pan_axes_w * (xr - xl)
            shift_y = dy_px / self._pan_axes_h * (yb - yt)
            vw, vh = xr - xl, yb - yt
            new_xl = xl - shift_x; new_xr = xr - shift_x
            new_yt = yt + shift_y; new_yb = yb + shift_y
            if new_xl < 0:            new_xl, new_xr = 0, vw
            elif new_xr > self.img_W: new_xr, new_xl = self.img_W, self.img_W - vw
            if new_yt < 0:            new_yt, new_yb = 0, vh
            elif new_yb > self.img_H: new_yb, new_yt = self.img_H, self.img_H - vh
            self.ax_ann.set_xlim(new_xl, new_xr)
            self.ax_ann.set_ylim(new_yb, new_yt)
            now = time.monotonic()
            if now - self._last_pan_time >= 0.033:
                self._last_pan_time = now
                self._vid_bg = None
                self.fig.canvas.draw_idle()
            return

        # Crosshair in draw mode, animated blit (no full redraw)
        # Also handle the case where the mouse is outside the axes while drawing
        if self._draw_mode and (event.inaxes is self.ax_ann or self.new_box_start is not None):
            if self._ch_h is not None and coords is not None:
                mx, my = coords
                if self.new_box_start is None:
                    # Before the first click: full-image crosshair, no rectangle
                    self._ch_h.set_xdata([0, self.img_W])
                    self._ch_h.set_ydata([my, my])
                    self._ch_h.set_alpha(0.75)
                    self._ch_v.set_xdata([mx, mx])
                    self._ch_v.set_ydata([0, self.img_H])
                    self._ch_v.set_alpha(0.75)
                    self._drag_rect_artist.set_visible(False)
                else:
                    # After the first click: crosshair hidden, live rectangle
                    self._ch_h.set_alpha(0.0)
                    self._ch_v.set_alpha(0.0)
                    sx, sy = self.new_box_start
                    self._drag_rect_artist.set_xy((min(sx, mx), min(sy, my)))
                    self._drag_rect_artist.set_width(abs(mx - sx))
                    self._drag_rect_artist.set_height(abs(my - sy))
                    self._drag_rect_artist.set_visible(True)
                self._blit_ann_overlay()
            else:
                self._update_annot()

    def _on_mouse_release(self, event):
        if self._prog_dragging:
            self._prog_dragging = False
            self._resume_from(self.vid_idx)
            return

        if self._panning:
            self._panning = False; return

        if self.drag_handle is not None:
            coords = self._ann_coords(event) or self._last_mouse
            if coords:
                mx, my = coords
                dx, dy = mx - self.drag_start[0], my - self.drag_start[1]
                ox1, oy1, ox2, oy2 = self.drag_orig
                nx1, ny1, nx2, ny2 = apply_handle_drag(
                    self.drag_handle, dx, dy, ox1, oy1, ox2, oy2, self.img_W, self.img_H)
                self.labels[self.sel_bbox][1:5] = list(
                    xyxy_to_yolo(nx1, ny1, nx2, ny2, self.img_W, self.img_H))
            self.drag_handle = None; self.drag_start = None; self.drag_orig = None
            self._save_annotation()
            print(f"  [OK] bbox mise à jour — {self.pairs[self.pair_idx]['stem']}")
            self._update_annot(); return

        if self.new_box_start is None: return
        # Get the coords BEFORE resetting new_box_start:
        # _ann_coords only clamps out-of-axes points when new_box_start is not None
        coords = self._ann_coords(event)
        sx, sy = self.new_box_start; self.new_box_start = None
        if not coords: coords = self._last_mouse
        if not coords: self._update_annot(); return
        mx, my = coords
        x1, y1 = min(sx, mx), min(sy, my)
        x2, y2 = max(sx, mx), max(sy, my)
        if (x2 - x1) <= NEW_BOX_MIN and (y2 - y1) <= NEW_BOX_MIN:
            self._update_annot(); return

        num = self._next_num()
        self._push_undo()
        self.labels.append([num] + list(
            xyxy_to_yolo(x1, y1, x2, y2, self.img_W, self.img_H)))
        self.sel_bbox   = len(self.labels) - 1
        self._draw_mode = False
        self._set_cursor()
        self._save_annotation()
        print(f"  [+] bbox #{num} — {self.pairs[self.pair_idx]['stem']}")
        self._update_annot()

    def _on_scroll(self, event):
        if event.inaxes is self.ax_ann:
            if self._ctrl:
                # Ctrl+scroll -> zoom centered on the mouse
                factor = ZOOM_STEP if event.button == "up" else 1 / ZOOM_STEP
                cx = event.xdata if event.xdata is not None else None
                cy = event.ydata if event.ydata is not None else None
                self._zoom_at(factor, cx, cy)
            # Scroll without Ctrl: ignored

    # ═══════════════════════════════════════════════════════
    #  Keyboard
    # ═══════════════════════════════════════════════════════
    def _on_key(self, ev):
        k = ev.key
        if   k in ("control", "ctrl"):        self._ctrl = True
        elif k == "ctrl+z":                   self._undo()
        elif k in ("enter", "return", "down"):
            self._save_annotation()
            self._load_pair(self.pair_idx + 1)
        elif k in ("up", "b", "B"):           self._pair_go(-1)
        elif k == "n":
            self._draw_mode  = not self._draw_mode
            self._erase_mode = False; self.new_box_start = None
            self._set_cursor(); self._update_annot()
        elif k in ("k", "K"):
            self._erase_mode = not self._erase_mode
            self._draw_mode  = False; self.new_box_start = None
            self._set_cursor(); self._update_annot()
        elif k == "r":
            self._hide_boxes = not self._hide_boxes; self._update_annot()
        elif k == "delete" and self.sel_bbox is not None:
            self._push_undo(); del self.labels[self.sel_bbox]
            self.sel_bbox = None; self._save_annotation(); self._update_annot()
        # ── Zoom ──
        elif k in ("+", "=", "num_add"):
            self._zoom_at(ZOOM_STEP)
        elif k in ("-", "num_subtract"):
            self._zoom_at(1 / ZOOM_STEP)
        elif k == ".":
            self._zoom_reset()
        elif k == "escape":
            if self._draw_mode or self._erase_mode or self.new_box_start is not None \
                    or self._panning:
                self._draw_mode = False; self._erase_mode = False
                self.new_box_start = None; self._panning = False
                self._set_cursor(); self._update_annot()
            else:
                self._save_annotation(); self._stop_timer(); plt.close("all")

    def _on_key_release(self, ev):
        if ev.key in ("control", "ctrl"):
            self._ctrl = False

    def _set_cursor(self):
        try:
            cur = "crosshair" if self._draw_mode else ("X_cursor" if self._erase_mode else "")
            self.fig.canvas.get_tk_widget().config(cursor=cur)
        except Exception:
            pass

    def _push_undo(self):
        self._undo_stack.append([list(lb) for lb in self.labels])
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            print("  [undo] Rien à annuler."); return
        self.labels = self._undo_stack.pop()
        self.sel_bbox = None; self._save_annotation()
        print(f"  [undo] {len(self.labels)} bbox(s) restaurées")
        self._update_annot()

    def _btn_draw(self, _):
        self._draw_mode  = not self._draw_mode
        self._erase_mode = False; self.new_box_start = None
        self._set_cursor(); self._update_annot()

    def _btn_erase(self, _):
        self._erase_mode = not self._erase_mode
        self._draw_mode  = False; self.new_box_start = None
        self._set_cursor(); self._update_annot()

    def _btn_delete_bbox(self, _):
        if self.sel_bbox is not None:
            self._push_undo(); del self.labels[self.sel_bbox]
            self.sel_bbox = None; self._save_annotation(); self._update_annot()

    def run(self):
        plt.show(block=True)
        self._save_annotation()
        if self.cap: self.cap.release()
        print(f"\nSession terminée. Labels dans : {self.output_dir}")


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder",
                        help="Dossier IAA contenant images/ et clips_iou/")
    parser.add_argument("--output", default=None,
                        help="Dossier de sortie pour les labels YOLO "
                             "(défaut: <folder>/labels/)")
    parser.add_argument("--start", type=int, default=0,
                        help="Index de la paire de départ 0-based (défaut: 0)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERROR] Dossier introuvable : {args.folder}"); sys.exit(1)

    output_dir = args.output or os.path.join(args.folder, "labels")
    ann = BboxAnnotator(iaa_folder=args.folder, output_dir=output_dir)
    if args.start > 0:
        ann._load_pair(args.start)
    ann.run()


if __name__ == "__main__":
    main()
