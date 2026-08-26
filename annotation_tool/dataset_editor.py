"""
dataset_editor.py
=================
Interactive editor for a YOLO dataset (images + labels).

Features:
  - "Video" playback of the frames with Play / Pause
  - Frame-by-frame navigation while paused
  - Bounding boxes shown with a color per class
  - Edit a bounding box with the mouse (click-and-drag an edge/corner)
  - Delete a single bbox (right click on the box)
  - Delete an entire image+label pair (Delete key)
  - Automatic save to the matching .txt file

Usage:
  python dataset_editor.py --folder /path/to/folder
  python dataset_editor.py --folder /path/to/folder --fps 12

Keyboard controls (always active):
  Space         : Play / Pause
  ←             : -1 frame    (auto-pauses)
  →             : +1 frame    (auto-pauses)
  a             : -10 frames  (auto-pauses)
  d             : +10 frames  (auto-pauses)
  q             : -100 frames (auto-pauses)
  e             : +100 frames (auto-pauses)
  Delete        : Delete the current frame's image + label
  Esc           : Quit

Mouse controls (paused only):
  Left click on a bbox        : select the bbox
  Click-and-drag a corner     : resize the selected bbox
  Click-and-drag the center   : move the selected bbox
  Right click on a bbox       : delete that bbox
  Click-and-drag (empty area) : draw a new bbox
                                 a window then asks for the class
"""

import argparse
import os
import glob
import sys
import shutil
import cv2
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────
# Per-class color palette (BGR for cv2, RGB for mpl)
# ──────────────────────────────────────────────────────────
_PALETTE_RGB = [
    (0.86, 0.20, 0.20),   # red          cls 0
    (0.20, 0.51, 0.86),   # blue         cls 1
    (0.20, 0.78, 0.31),   # green        cls 2
    (0.90, 0.63, 0.12),   # orange       cls 3
    (0.63, 0.24, 0.86),   # purple       cls 4
    (0.12, 0.78, 0.78),   # cyan         cls 5
    (0.86, 0.20, 0.63),   # pink         cls 6
    (0.55, 0.82, 0.12),   # lime         cls 7
]

HANDLE_RADIUS = 8   # screen px for detecting corners/edges
NEW_BOX_MIN   = 10  # minimum px to create a new bbox


def palette(cls_id: int):
    return _PALETTE_RGB[int(cls_id) % len(_PALETTE_RGB)]


# ──────────────────────────────────────────────────────────
# YOLO <-> pixel conversion
# ──────────────────────────────────────────────────────────
def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h


# ──────────────────────────────────────────────────────────
# Read / write labels
# ──────────────────────────────────────────────────────────
def read_labels(path: str):
    """Return a list of [cls_id, xc, yc, w, h] (float)."""
    if not os.path.exists(path):
        return []
    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(v) for v in parts[1:]])
    return labels


def write_labels(path: str, labels: list):
    """Write YOLO labels to the txt file."""
    with open(path, "w") as f:
        for lb in labels:
            cls_id = int(lb[0])
            f.write(f"{cls_id} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f} {lb[4]:.6f}\n")


# ──────────────────────────────────────────────────────────
# Detect a bbox's corners / center
# ──────────────────────────────────────────────────────────
HANDLES = ["tl", "tr", "bl", "br", "tc", "bc", "lc", "rc", "inside"]


def get_handle(mx, my, x1, y1, x2, y2, r=HANDLE_RADIUS):
    """
    Return which handle is under the mouse:
      'tl','tr','bl','br' : corner
      'tc','bc','lc','rc' : edge midpoint
      'inside'            : interior (move)
      None                : not on the bbox
    """
    cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2

    handles = {
        "tl": (x1, y1), "tr": (x2, y1),
        "bl": (x1, y2), "br": (x2, y2),
        "tc": (cx, y1), "bc": (cx, y2),
        "lc": (x1, cy), "rc": (x2, cy),
    }
    for name, (hx, hy) in handles.items():
        if abs(mx - hx) < r and abs(my - hy) < r:
            return name
    if x1 <= mx <= x2 and y1 <= my <= y2:
        return "inside"
    return None


def apply_handle_drag(handle, dx, dy, x1, y1, x2, y2, W, H):
    """Return (x1,y1,x2,y2) after moving according to the handle."""
    if handle == "inside":
        x1 += dx; x2 += dx; y1 += dy; y2 += dy
    elif handle == "tl":  x1 += dx; y1 += dy
    elif handle == "tr":  x2 += dx; y1 += dy
    elif handle == "bl":  x1 += dx; y2 += dy
    elif handle == "br":  x2 += dx; y2 += dy
    elif handle == "tc":  y1 += dy
    elif handle == "bc":  y2 += dy
    elif handle == "lc":  x1 += dx
    elif handle == "rc":  x2 += dx
    # Clamp
    x1 = max(0, min(x1, W - 1));  x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1));  y2 = max(0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


# ──────────────────────────────────────────────────────────
# Ask for the class (tkinter window, genuinely blocking)
# ──────────────────────────────────────────────────────────
def ask_class_id(default=0) -> int | None:
    """
    Opens a small modal tkinter window to enter a class integer.
    Uses Toplevel (not Tk) since TkAgg has already instantiated a Tk root.
    wait_window() blocks until it closes without conflicting with the event loop.
    """
    import tkinter as tk

    result = {"cls": None}

    # TkAgg already has a Tk root, so we create a Toplevel on top of it
    top = tk.Toplevel()
    top.title("Nouvelle bbox — Classe ?")
    top.configure(bg="#0d1117")
    top.resizable(False, False)
    top.grab_set()   # modal: blocks interaction with the parent window

    w, h = 300, 130
    top.update_idletasks()
    sw = top.winfo_screenwidth()
    sh = top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    top.lift()
    top.focus_force()

    tk.Label(top, text="Entrez l'ID de classe :", bg="#0d1117",
             fg="#c9d1d9", font=("Courier", 11)).pack(pady=(18, 4))

    entry_var = tk.StringVar(value=str(default))
    entry = tk.Entry(top, textvariable=entry_var, bg="#161b22", fg="#00ff88",
                     insertbackground="#00ff88", font=("Courier", 14, "bold"),
                     justify="center", relief="flat", width=10,
                     highlightthickness=1, highlightcolor="#238636",
                     highlightbackground="#30363d")
    entry.pack(pady=4)
    entry.select_range(0, tk.END)
    entry.focus_set()

    def validate(event=None):
        val = entry_var.get().strip()
        try:
            result["cls"] = int(val)
        except ValueError:
            result["cls"] = default
        top.destroy()

    def cancel(event=None):
        top.destroy()

    entry.bind("<Return>",   validate)
    entry.bind("<KP_Enter>", validate)
    entry.bind("<Escape>",   cancel)

    btn_frame = tk.Frame(top, bg="#0d1117")
    btn_frame.pack(pady=6)
    tk.Button(btn_frame, text="Valider", command=validate,
              bg="#238636", fg="white", font=("Courier", 9, "bold"),
              relief="flat", padx=10, pady=3).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Annuler", command=cancel,
              bg="#4a1010", fg="white", font=("Courier", 9),
              relief="flat", padx=10, pady=3).pack(side="left", padx=6)

    top.wait_window(top)
    return result["cls"]


# ──────────────────────────────────────────────────────────
# Main class: the editor
# ──────────────────────────────────────────────────────────
class DatasetEditor:
    def __init__(self, folder: str, fps: float = 10.0):
        self.folder      = folder
        self.imgs_dir    = os.path.join(folder, "images")
        self.labels_dir  = os.path.join(folder, "labels")
        self.fps         = fps
        self.interval    = 1.0 / fps   # seconds between frames

        # Load the frames
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        for ext in exts:
            self.image_paths += glob.glob(os.path.join(self.imgs_dir, ext))
        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            print(f"[ERREUR] Aucune image trouvée dans {self.imgs_dir}")
            sys.exit(1)

        print(f"  {len(self.image_paths)} frames chargées depuis {self.imgs_dir}")

        # State
        self.idx         = 0
        self.playing     = False
        self.labels      = []      # current frame's labels
        self.sel_bbox    = None    # index of the selected bbox
        self.drag_handle = None
        self.drag_start  = None    # (mx, my) at the start of the drag
        self.drag_orig   = None    # original bbox before the drag
        self.new_box_start   = None   # (mx,my) for drawing a new bbox
        self._last_mouse_pos = None   # last known mouse position within the axes
        self._timer          = None

        # Image dimensions (read from the 1st frame)
        img0 = cv2.imread(self.image_paths[0])
        self.H, self.W = img0.shape[:2]

        self._build_ui()
        self._load_frame(0)

    # ── Build the UI ───────────────────────────────────────
    def _build_ui(self):
        plt.rcParams.update({
            "figure.facecolor":  "#0d1117",
            "axes.facecolor":    "#0d1117",
            "text.color":        "#c9d1d9",
            "axes.edgecolor":    "#30363d",
            "xtick.color":       "#8b949e",
            "ytick.color":       "#8b949e",
        })

        self.fig = plt.figure(figsize=(15, 8.5))
        self.fig.canvas.manager.set_window_title("Dataset Editor — YOLO Label Viewer")
        self.fig.patch.set_facecolor("#0d1117")

        gs = gridspec.GridSpec(
            3, 1,
            height_ratios=[0.92, 0.045, 0.035],
            hspace=0.06,
            figure=self.fig
        )

        # ── Image area
        self.ax = self.fig.add_subplot(gs[0])
        self.ax.axis("off")

        # ── Progress bar (slider)
        ax_slider = self.fig.add_subplot(gs[1])
        ax_slider.set_facecolor("#161b22")
        self.slider = Slider(
            ax_slider, "", 0, max(len(self.image_paths) - 1, 1),
            valinit=0, valstep=1, color="#238636",
        )
        self.slider.label.set_color("#8b949e")
        self.slider.valtext.set_color("#00ff88")
        self.slider.on_changed(self._on_slider)

        # ── Buttons
        ax_btns = self.fig.add_subplot(gs[2])
        ax_btns.axis("off")

        btn_defs = [
            ("⏮  -100", "#1f2937", self._btn_m100),
            ("◀  -10",  "#1f2937", self._btn_m10),
            ("◀  -1",   "#1f2937", self._btn_m1),
            ("⏸  Pause" if not self.playing else "▶  Play",
             "#0d4f3c", self._btn_play),
            ("▶  +1",   "#1f2937", self._btn_p1),
            ("▶▶  +10", "#1f2937", self._btn_p10),
            ("▶▶  +100","#1f2937", self._btn_p100),
            ("🗑 Delete","#4a1010", self._btn_delete),
        ]

        self._buttons = {}
        n_btns = len(btn_defs)
        margin = 0.01
        bw = (1.0 - margin * (n_btns + 1)) / n_btns

        for k, (label, color, cb) in enumerate(btn_defs):
            bx = margin + k * (bw + margin)
            ax_b = self.fig.add_axes([
                self.fig.subplotpars.left + bx * (
                    self.fig.subplotpars.right - self.fig.subplotpars.left),
                0.005,
                bw * (self.fig.subplotpars.right - self.fig.subplotpars.left),
                0.028,
            ])
            btn = Button(ax_b, label,
                         color=color, hovercolor="#2d333b")
            btn.label.set_color("#c9d1d9")
            btn.label.set_fontsize(8)
            btn.label.set_fontfamily("monospace")
            btn.on_clicked(cb)
            self._buttons[label] = btn

        # ── Mouse / keyboard event connections
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event",self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event",self._on_mouse_release)

        # ── Status text
        self._status = self.fig.text(
            0.5, 0.973, "", ha="center", va="top",
            color="#8b949e", fontsize=9, fontfamily="monospace",
        )

        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.97, bottom=0.045
        )

    # ── Load a frame ───────────────────────────────────────
    def _load_frame(self, idx: int):
        self.idx = max(0, min(idx, len(self.image_paths) - 1))
        img_path   = self.image_paths[self.idx]
        base       = os.path.basename(img_path).rsplit(".", 1)[0]
        label_path = os.path.join(self.labels_dir, base + ".txt")

        self.current_img_path   = img_path
        self.current_label_path = label_path
        self.labels             = read_labels(label_path)
        self.sel_bbox           = None

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return
        self.H, self.W = img_bgr.shape[:2]
        self._current_img_bgr = img_bgr

        self._update_display()

        # Sync the slider without triggering its callback
        self.slider.eventson = False
        self.slider.set_val(self.idx)
        self.slider.eventson = True

        # Status
        n_boxes = len(self.labels)
        self._status.set_text(
            f"Frame {self.idx+1}/{len(self.image_paths)}  |  "
            f"{os.path.basename(img_path)}  |  "
            f"{n_boxes} bbox(s)  |  "
            f"{'▶ PLAY' if self.playing else '⏸ PAUSE'}  |  "
            "Espace=play/pause  ←/→=nav  Delete=suppr frame  "
            "Clic droit sur bbox=suppr bbox"
        )

    def _update_display(self):
        """Redraw the image + bboxes in self.ax."""
        img_bgr = self._current_img_bgr.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(img_rgb, aspect="equal")

        for i, lb in enumerate(self.labels):
            cls_id            = int(lb[0])
            xc, yc, bw, bh   = lb[1], lb[2], lb[3], lb[4]
            x1, y1, x2, y2   = yolo_to_xyxy(xc, yc, bw, bh, self.W, self.H)
            color             = palette(cls_id)
            lw                = 2.5 if i == self.sel_bbox else 1.5
            ls                = "-"  if i == self.sel_bbox else "--"

            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=lw, edgecolor=color, facecolor=(*color, 0.08),
                linestyle=ls
            )
            self.ax.add_patch(rect)

            # Label is shown ABOVE the bbox (outside the box)
            # If the bbox is at the very top of the image, put it below instead
            label_y  = y1 - 4
            va       = "bottom"
            if label_y < 12:          # not enough room above, put it below
                label_y = y2 + 4
                va      = "top"

            self.ax.text(
                x1, label_y,
                f"cls {cls_id}",
                color="white", fontsize=8, fontfamily="monospace",
                va=va, ha="left",
                bbox=dict(facecolor=color, alpha=0.82, pad=2, linewidth=0),
            )

            # Corners of the selected bbox
            if i == self.sel_bbox:
                cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
                for hx, hy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2),
                                (cx,y1),(cx,y2),(x1,cy),(x2,cy)]:
                    self.ax.plot(hx, hy, "o",
                                 ms=5, color="white",
                                 markerfacecolor=color, markeredgewidth=1)

        self.fig.canvas.draw_idle()

    # ── Timer (play) ───────────────────────────────────────
    def _start_timer(self):
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.fig.canvas.new_timer(
            interval=int(self.interval * 1000)
        )
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self):
        if not self.playing:
            self._stop_timer()
            return
        if self.idx >= len(self.image_paths) - 1:
            self.playing = False
            self._stop_timer()
            return
        self._load_frame(self.idx + 1)

    # ── Controls ───────────────────────────────────────────
    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._start_timer()
        else:
            self._stop_timer()
        self._load_frame(self.idx)   # refresh the status text

    def _go(self, delta):
        """Go to idx + delta and pause."""
        self.playing = False
        self._stop_timer()
        self._load_frame(self.idx + delta)

    # ── Button callbacks ───────────────────────────────────
    def _btn_m100(self, _): self._go(-100)
    def _btn_m10(self, _):  self._go(-10)
    def _btn_m1(self, _):   self._go(-1)
    def _btn_p1(self, _):   self._go(1)
    def _btn_p10(self, _):  self._go(10)
    def _btn_p100(self, _): self._go(100)
    def _btn_play(self, _): self._toggle_play()

    def _btn_delete(self, _):
        self._delete_current_frame()

    # ── Keyboard ─────────────────────────────────────────────
    def _on_key(self, ev):
        if ev.key == " ":
            self._toggle_play()
        elif ev.key == "left":
            self._go(-1)
        elif ev.key == "right":
            self._go(1)
        elif ev.key == "a":
            self._go(-10)
        elif ev.key == "d":
            self._go(10)
        elif ev.key == "q":
            self._go(-100)
        elif ev.key == "e":
            self._go(100)
        elif ev.key == "delete":
            self._delete_current_frame()
        elif ev.key == "escape":
            self._stop_timer()
            plt.close("all")

    # ── Slider ─────────────────────────────────────────────
    def _on_slider(self, val):
        new_idx = int(round(val))
        if new_idx != self.idx:
            self.playing = False
            self._stop_timer()
            self._load_frame(new_idx)

    # ── Mouse ──────────────────────────────────────────────
    def _ax_coords(self, event):
        """
        Return (x_pixel, y_pixel) in image space.
        - If the event is inside the axes: use the coordinates directly.
        - If outside the axes but a drag/draw is in progress: convert from
          display coordinates and clamp to the image bounds.
        - Otherwise: None.
        """
        if event.inaxes is self.ax:
            return event.xdata, event.ydata

        # Outside the axes but a drag is in progress, extrapolate from display coords
        if (self.drag_handle is not None or self.new_box_start is not None) \
                and event.x is not None and event.y is not None:
            try:
                # Convert display coordinates to image data coordinates
                inv = self.ax.transData.inverted()
                x, y = inv.transform((event.x, event.y))
                # Clamp to the image dimensions
                x = max(0.0, min(float(self.W - 1), x))
                y = max(0.0, min(float(self.H - 1), y))
                return x, y
            except Exception:
                pass
        return None

    def _on_mouse_press(self, event):
        if self.playing:
            return
        coords = self._ax_coords(event)
        if coords is None:
            return
        mx, my = coords

        # ── Right click: delete bbox
        if event.button == 3:
            for i, lb in enumerate(self.labels):
                cls_id = int(lb[0])
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4],
                                                self.W, self.H)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    del self.labels[i]
                    write_labels(self.current_label_path, self.labels)
                    print(f"  [−] Bbox cls{cls_id} supprimée  "
                          f"(frame {self.idx+1}/{len(self.image_paths)})")
                    self.sel_bbox = None
                    self._update_display()
                    return
            return

        # ── Left click
        if event.button != 1:
            return

        # 1) Look for a corner/edge handle (priority) or interior
        hit_resize = None
        hit_inside = None

        for i, lb in enumerate(self.labels):
            x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4],
                                            self.W, self.H)
            handle = get_handle(mx, my, x1, y1, x2, y2)
            if handle is None:
                continue
            if handle == "inside":
                if hit_inside is None:
                    hit_inside = (i, x1, y1, x2, y2)
            else:
                hit_resize = (i, handle, x1, y1, x2, y2)
                break

        if hit_resize is not None:
            i, handle, x1, y1, x2, y2 = hit_resize
            self.sel_bbox    = i
            self.drag_handle = handle
            self.drag_start  = (mx, my)
            self.drag_orig   = (x1, y1, x2, y2)
            self._update_display()
            return

        if hit_inside is not None:
            i, x1, y1, x2, y2 = hit_inside
            self.sel_bbox    = i
            self.drag_handle = "inside"
            self.drag_start  = (mx, my)
            self.drag_orig   = (x1, y1, x2, y2)
            self._update_display()
            return

        # 2) Click on empty space: deselect and start a new bbox
        self.sel_bbox      = None
        self.new_box_start = (mx, my)
        self._update_display()

    def _on_mouse_move(self, event):
        if self.playing:
            return
        coords = self._ax_coords(event)
        if coords is not None:
            self._last_mouse_pos = coords   # always remember it
        if coords is None:
            return
        mx, my = coords

        # Dragging an existing bbox
        if self.drag_handle is not None and self.drag_start is not None:
            dx = mx - self.drag_start[0]
            dy = my - self.drag_start[1]
            ox1, oy1, ox2, oy2 = self.drag_orig
            nx1, ny1, nx2, ny2 = apply_handle_drag(
                self.drag_handle, dx, dy, ox1, oy1, ox2, oy2, self.W, self.H
            )
            # Temporary update for display
            lb = self.labels[self.sel_bbox]
            lb[1], lb[2], lb[3], lb[4] = xyxy_to_yolo(
                nx1, ny1, nx2, ny2, self.W, self.H
            )
            self._update_display()
            return

        # Drawing a new bbox
        if self.new_box_start is not None:
            # Show a temporary rectangle
            sx, sy = self.new_box_start
            x1, y1 = min(sx, mx), min(sy, my)
            x2, y2 = max(sx, mx), max(sy, my)
            # Redraw with the temporary rect
            img_rgb = cv2.cvtColor(self._current_img_bgr, cv2.COLOR_BGR2RGB)
            self.ax.clear()
            self.ax.axis("off")
            self.ax.imshow(img_rgb, aspect="equal")
            # Existing bboxes
            for i, lb in enumerate(self.labels):
                cls_id = int(lb[0])
                bx1, by1, bx2, by2 = yolo_to_xyxy(
                    lb[1], lb[2], lb[3], lb[4], self.W, self.H
                )
                color = palette(cls_id)
                self.ax.add_patch(mpatches.Rectangle(
                    (bx1, by1), bx2 - bx1, by2 - by1,
                    linewidth=1.5, edgecolor=color,
                    facecolor=(*color, 0.08), linestyle="--"
                ))
            # Temporary new bbox
            self.ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=(0.0, 1.0, 0.5),
                facecolor=(0.0, 1.0, 0.5, 0.1), linestyle="-"
            ))
            self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event):
        if self.playing:
            return

        # ── End of dragging an existing bbox ──────────────────
        if self.drag_handle is not None:
            self.drag_handle = None
            self.drag_start  = None
            self.drag_orig   = None
            write_labels(self.current_label_path, self.labels)
            print(f"  [✓] Bbox mise à jour — frame {self.idx+1}")
            self._update_display()
            return

        # ── End of drawing a new bbox ──────────────────────────
        if self.new_box_start is None:
            return

        sx, sy = self.new_box_start
        self.new_box_start = None

        # Use the release coordinates if available, otherwise the last move's
        coords = self._ax_coords(event)
        if coords is not None:
            mx, my = coords
        elif self._last_mouse_pos is not None:
            mx, my = self._last_mouse_pos
        else:
            self._update_display()
            return

        x1, y1 = min(sx, mx), min(sy, my)
        x2, y2 = max(sx, mx), max(sy, my)

        print(f"  [?] Nouvelle bbox tracée : ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})  taille=({x2-x1:.0f}x{y2-y1:.0f})")

        if (x2 - x1) <= NEW_BOX_MIN or (y2 - y1) <= NEW_BOX_MIN:
            print(f"  [!] Bbox trop petite, ignorée (min={NEW_BOX_MIN}px)")
            self._update_display()
            return

        # Ask for the class via simpledialog (thread-safe with TkAgg)
        cls_id = ask_class_id(default=0)
        print(f"  [?] Classe saisie : {cls_id}")

        if cls_id is not None:
            xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, self.W, self.H)
            self.labels.append([cls_id, xc, yc, bw, bh])
            write_labels(self.current_label_path, self.labels)
            print(f"  [+] Nouvelle bbox cls{cls_id} sauvegardée — frame {self.idx+1}")
        else:
            print("  [!] Ajout annulé.")

        self._update_display()

    # ── Delete a frame ──────────────────────────────────────
    def _delete_current_frame(self):
        img_path   = self.current_img_path
        label_path = self.current_label_path

        # Quick confirmation in the title
        self._status.set_text(
            f"⚠  Suppression de {os.path.basename(img_path)} "
            f"— confirmé (fichiers déplacés vers _deleted/)"
        )
        self.fig.canvas.draw_idle()

        # Move to _deleted/ rather than deleting outright (safety)
        del_dir = os.path.join(self.folder, "_deleted")
        os.makedirs(del_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S%f")

        if os.path.exists(img_path):
            shutil.move(img_path,
                        os.path.join(del_dir,
                                     ts + "_" + os.path.basename(img_path)))
        if os.path.exists(label_path):
            shutil.move(label_path,
                        os.path.join(del_dir,
                                     ts + "_" + os.path.basename(label_path)))

        print(f"  [🗑] Frame supprimée → _deleted/ : "
              f"{os.path.basename(img_path)}")

        # Reload the file list
        old_idx = self.idx
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        for ext in exts:
            self.image_paths += glob.glob(os.path.join(self.imgs_dir, ext))
        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            print("[!] Plus aucune image. Fermeture.")
            plt.close("all")
            return

        self.slider.valmax = max(len(self.image_paths) - 1, 1)
        new_idx = min(old_idx, len(self.image_paths) - 1)
        self._load_frame(new_idx)

    # ── Run ────────────────────────────────────────────────
    def run(self):
        plt.show(block=True)


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Éditeur interactif de dataset YOLO"
    )
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Dossier contenant images/ et labels/"
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="FPS en mode play (défaut : 10)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERREUR] Dossier introuvable : {args.folder}")
        sys.exit(1)
    if not os.path.isdir(os.path.join(args.folder, "images")):
        print(f"[ERREUR] Sous-dossier 'images/' absent dans {args.folder}")
        sys.exit(1)

    editor = DatasetEditor(folder=args.folder, fps=args.fps)
    editor.run()


if __name__ == "__main__":
    main()