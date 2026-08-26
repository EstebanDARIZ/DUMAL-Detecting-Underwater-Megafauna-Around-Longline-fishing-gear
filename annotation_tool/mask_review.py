"""
mask_review.py
==============
Interactive tool for reviewing and fixing segmentation masks.

Global controls:
  S            : save changes and move to the next image
  E            : delete the mask_label file (image kept) and move to the next one
  ← / →        : previous / next image without saving
  R (held)     : temporarily hide the masks to see the raw image
  N            : new mask, click to place points, Enter to confirm
  Q / Esc      : quit

Editing an existing mask:
  Left click on a mask       : select it (control points shown)
  Left click on a point      : select it, drag to move
  Left click on an edge      : insert a new point
  Right click on a point     : delete that point (min. 3 left)
  Right click outside a mask : deselect

Drawing mode (N key):
  Left click      : add a point
  Right click     : cancel the drawing
  Enter           : confirm (dialog to pick the class)
  Esc             : cancel

Usage:
  python mask_review.py --folder /path/to/video_001/
  python mask_review.py --folder /path/to/video_001/ --start 50
"""

import argparse
import os
import sys
import glob
import numpy as np
import cv2
import tkinter as tk
from tkinter import simpledialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt      # used in main() to create the figure
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

PICK_RADIUS = 10
EDGE_RADIUS = 8

CLASS_NAMES = {
    0: "Squid",
    1: "Sardine",
    2: "Ray",
    3: "Sunfish",
    4: "Pilot Fish",
    5: "Shark",
}

_PALETTE = [
    (220,  50,  50),
    ( 50, 130, 220),
    ( 50, 200,  80),
    (230, 160,  30),
    (160,  60, 220),
    ( 30, 200, 200),
]


def cls_color(cls_id: int) -> tuple:
    r, g, b = _PALETTE[cls_id % len(_PALETTE)]
    return r / 255, g / 255, b / 255


# ──────────────────────────────────────────────────────────────
# YOLO seg label I/O
# ──────────────────────────────────────────────────────────────

def load_seg_labels(path: str) -> list:
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls    = int(parts[0])
            coords = np.array([float(v) for v in parts[1:]]).reshape(-1, 2)
            entries.append([cls, coords])
    return entries


def save_seg_labels(path: str, entries: list):
    with open(path, "w") as f:
        for cls, pts in entries:
            coords_str = " ".join(f"{v:.6f}" for v in pts.flatten())
            f.write(f"{cls} {coords_str}\n")


# ──────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────

def pt_seg_dist(p, a, b) -> float:
    ab    = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-10:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


# ──────────────────────────────────────────────────────────────
# Mask editor
# ──────────────────────────────────────────────────────────────

class MaskEditor:
    def __init__(self, ax, img_rgb: np.ndarray, entries: list, W: int, H: int,
                 mask_path: str = ""):
        self.ax        = ax
        self.img_rgb   = img_rgb
        self.W         = W
        self.H         = H
        self.mask_path = mask_path
        self.entries   = [[cls, pts.copy()] for cls, pts in entries]

        self.sel_mask = -1
        self.sel_pt   = -1
        self.dragging = False
        self._did_drag = False        # true if a point moved during the drag

        self._hidden        = False   # R held down -> masks hidden
        self._drawing       = False   # new-mask drawing mode
        self._draw_pts: list = []     # points collected so far (pixels)

        self._artists       = []      # artists for existing masks
        self._draw_artists  = []      # artists for the drawing preview

        self._redraw()

    # ── Render existing masks ─────────────────────────────────

    def _redraw(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

        if self._hidden:
            self._update_title()
            self.ax.figure.canvas.draw_idle()
            return

        for i, (cls, pts) in enumerate(self.entries):
            pts_px = pts * np.array([self.W, self.H])
            color  = cls_color(cls)
            sel    = (i == self.sel_mask)

            poly = MplPolygon(
                pts_px, closed=True,
                facecolor=(*color, 0.40 if sel else 0.25),
                edgecolor=color,
                linewidth=2.5 if sel else 1.5,
                zorder=2,
            )
            self.ax.add_patch(poly)
            self._artists.append(poly)

            if sel:
                # scatter: size in points^2 (screen units), independent of zoom
                colors = ["white" if j == self.sel_pt else color
                          for j in range(len(pts_px))]
                sizes  = [120 if j == self.sel_pt else 60
                          for j in range(len(pts_px))]
                sc = self.ax.scatter(
                    pts_px[:, 0], pts_px[:, 1],
                    s=sizes, c=colors,
                    edgecolors="black", linewidths=0.8,
                    zorder=5,
                )
                self._artists.append(sc)

        self._update_title()
        self.ax.figure.canvas.draw_idle()

    # ── Autosave ───────────────────────────────────────────────

    def _autosave(self):
        if self.mask_path:
            save_seg_labels(self.mask_path, self.entries)

    # ── Render the drawing preview ────────────────────────────

    def _redraw_drawing(self):
        for a in self._draw_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._draw_artists.clear()

        if not self._draw_pts:
            self.ax.figure.canvas.draw_idle()
            return

        pts = np.array(self._draw_pts)

        sc = self.ax.scatter(pts[:, 0], pts[:, 1],
                             s=40, c="yellow", zorder=10, edgecolors="black")
        self._draw_artists.append(sc)

        if len(pts) > 1:
            line, = self.ax.plot(pts[:, 0], pts[:, 1],
                                 color="yellow", linewidth=1.5, zorder=9)
            self._draw_artists.append(line)

        if len(pts) >= 3:
            close = np.array([pts[-1], pts[0]])
            dline, = self.ax.plot(close[:, 0], close[:, 1],
                                  color="yellow", linestyle="--", linewidth=1, zorder=9)
            self._draw_artists.append(dline)

        self._update_title()
        self.ax.figure.canvas.draw_idle()

    def _update_title(self):
        if self._drawing:
            n = len(self._draw_pts)
            self.ax.set_title(
                f"MODE DESSIN — {n} point(s)  |  Clic gauche = ajouter  "
                f"Entrée = valider (≥3 pts)  Clic droit / Echap = annuler",
                fontsize=9, color="darkorange",
            )
            return

        if self._hidden:
            self.ax.set_title(
                "Image brute (masques cachés — relâcher R pour restaurer)",
                fontsize=9, color="gray",
            )
            return

        if self.sel_mask >= 0:
            cls = self.entries[self.sel_mask][0]
            n   = len(self.entries[self.sel_mask][1])
            info = f"Masque : {CLASS_NAMES.get(cls, f'cls{cls}')} ({n} pts)"
            if self.sel_pt >= 0:
                info += f"  —  point {self.sel_pt} sélectionné  |  Clic droit = supprimer"
        else:
            info = "Clic sur un masque pour le sélectionner  |  N = nouveau masque"

        self.ax.set_title(
            f"{info}\n"
            "S / → = suivant   E = supprimer   ← = précédent   "
            "R (maintenu) = image brute   N = nouveau masque   Q = quitter   "
            "[auto-sauvegarde active]",
            fontsize=9, color="steelblue",
        )

    # ── Picking ────────────────────────────────────────────────

    def _find_point(self, x, y) -> int:
        if self.sel_mask < 0:
            return -1
        pts_px = self.entries[self.sel_mask][1] * np.array([self.W, self.H])
        dists  = np.linalg.norm(pts_px - [x, y], axis=1)
        j      = int(np.argmin(dists))
        return j if dists[j] < PICK_RADIUS else -1

    def _find_edge(self, x, y) -> int:
        if self.sel_mask < 0:
            return -1
        pts_px = self.entries[self.sel_mask][1] * np.array([self.W, self.H])
        n = len(pts_px)
        best_d, best_i = EDGE_RADIUS, -1
        for i in range(n):
            d = pt_seg_dist(np.array([x, y]), pts_px[i], pts_px[(i + 1) % n])
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _find_mask(self, x, y) -> int:
        for i, (_, pts) in enumerate(self.entries):
            pts_px = (pts * np.array([self.W, self.H])).astype(np.float32)
            if cv2.pointPolygonTest(
                pts_px.reshape(-1, 1, 2), (float(x), float(y)), False
            ) >= 0:
                return i
        return -1

    # ── Mouse events ───────────────────────────────────────────

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata

        # Drawing mode
        if self._drawing:
            if event.button == 1:
                self._draw_pts.append((x, y))
                self._redraw_drawing()
            elif event.button == 3:
                self._cancel_drawing()
            return

        if event.button == 1:
            j = self._find_point(x, y)
            if j >= 0:
                self.sel_pt   = j
                self.dragging = True
                self._redraw()
                return

            ei = self._find_edge(x, y)
            if ei >= 0:
                pts = self.entries[self.sel_mask][1]
                xn  = np.clip(x / self.W, 0, 1)
                yn  = np.clip(y / self.H, 0, 1)
                self.entries[self.sel_mask][1] = np.insert(pts, ei + 1, [[xn, yn]], axis=0)
                self.sel_pt   = ei + 1
                self.dragging = True
                self._autosave()
                self._redraw()
                return

            self.sel_mask = self._find_mask(x, y)
            self.sel_pt   = -1
            self._redraw()

        elif event.button == 3:
            if self.sel_mask >= 0:
                j = self._find_point(x, y)
                if j >= 0:
                    self.sel_pt = j
                    self._delete_selected_point()
                    return
            self.sel_mask = -1
            self.sel_pt   = -1
            self._redraw()

    def on_motion(self, event):
        if not self.dragging:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.sel_mask < 0 or self.sel_pt < 0:
            return
        self.entries[self.sel_mask][1][self.sel_pt] = [
            np.clip(event.xdata / self.W, 0, 1),
            np.clip(event.ydata / self.H, 0, 1),
        ]
        self._did_drag = True
        self._redraw()

    def on_release(self, *_):
        if self.dragging and self._did_drag:
            self._autosave()
        self.dragging  = False
        self._did_drag = False

    # ── Keyboard ───────────────────────────────────────────────

    def on_key_press(self, ev) -> bool:
        """
        Handle the editor's own key bindings.
        Returns True if the key was consumed (don't propagate further).
        """
        k = ev.key.lower() if ev.key else ""

        # R held down: hide the masks
        if k == "r":
            if not self._hidden:
                self._hidden = True
                self._redraw()
            return True

        # Drawing mode is active
        if self._drawing:
            if k in ("enter", " "):
                self._finalize_drawing()
            elif k == "escape":
                self._cancel_drawing()
            return True  # consume every key while in drawing mode

        # Enter drawing mode
        if k == "n":
            self._start_drawing()
            return True

        return False

    def on_key_release(self, ev):
        if ev.key and ev.key.lower() == "r":
            self._hidden = False
            self._redraw()

    # ── Drawing actions ─────────────────────────────────────────

    def _start_drawing(self):
        self._drawing  = True
        self._draw_pts = []
        self.sel_mask  = -1
        self.sel_pt    = -1
        self._redraw()
        self._redraw_drawing()

    def _cancel_drawing(self):
        self._drawing  = False
        self._draw_pts = []
        for a in self._draw_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._draw_artists.clear()
        self._redraw()

    def _finalize_drawing(self):
        if len(self._draw_pts) < 3:
            print("  [!] Minimum 3 points pour créer un masque.")
            return

        # Tkinter dialog to pick the class
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        classes_hint = "  ".join(f"{k}={v}" for k, v in CLASS_NAMES.items())
        cls_id = simpledialog.askinteger(
            "Classe du nouveau masque",
            f"Classes : {classes_hint}\n\nEntrer l'ID de la classe :",
            minvalue=0, maxvalue=99, parent=root,
        )
        root.destroy()

        if cls_id is None:
            return  # dialog cancelled, stay in drawing mode

        pts_norm = np.array([
            [x / self.W, y / self.H] for x, y in self._draw_pts
        ])
        self.entries.append([cls_id, pts_norm])
        self._autosave()
        print(f"  ↳ Nouveau masque ajouté : {CLASS_NAMES.get(cls_id, f'cls{cls_id}')} "
              f"({len(pts_norm)} pts)")

        self._drawing  = False
        self._draw_pts = []
        for a in self._draw_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._draw_artists.clear()
        self._redraw()

    # ── Delete a point ─────────────────────────────────────────

    def _delete_selected_point(self):
        if self.sel_mask < 0 or self.sel_pt < 0:
            return
        pts = self.entries[self.sel_mask][1]
        if len(pts) <= 3:
            print("  [!] Minimum 3 points — suppression annulée.")
            return
        self.entries[self.sel_mask][1] = np.delete(pts, self.sel_pt, axis=0)
        self.sel_pt = -1
        self._autosave()
        self._redraw()


# ──────────────────────────────────────────────────────────────
# Review window for a single image
# ──────────────────────────────────────────────────────────────

def review_image(fig, ax, img_bgr: np.ndarray, entries: list,
                 stem: str, idx: int, total: int, mask_path: str = ""):
    """
    Show an image in the existing window (fig/ax) and wait for an action.
    The window is never closed between images, so its size is preserved.
    """
    ax.cla()

    W, H    = img_bgr.shape[1], img_bgr.shape[0]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax.imshow(img_rgb)
    ax.axis("off")
    fig.canvas.manager.set_window_title(f"Review masques [{idx}/{total}] — {stem}")

    editor = MaskEditor(ax, img_rgb, entries, W, H, mask_path=mask_path)

    patches = [
        mpatches.Patch(color=cls_color(cls),
                       label=f"{CLASS_NAMES.get(cls, f'cls{cls}')} ({cls})")
        for cls, _ in entries
    ]
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=8,
                  framealpha=0.75, facecolor="black", labelcolor="white")

    result = {"action": "next"}
    cids   = []

    def on_key_press(ev):
        consumed = editor.on_key_press(ev)
        if consumed:
            return
        k = ev.key.lower() if ev.key else ""
        if k == "s":
            result["action"] = "save"
            fig.canvas.stop_event_loop()
        elif k == "e":
            result["action"] = "delete"
            fig.canvas.stop_event_loop()
        elif k in ("right", "d", " "):
            result["action"] = "next"
            fig.canvas.stop_event_loop()
        elif k in ("left", "a"):
            result["action"] = "prev"
            fig.canvas.stop_event_loop()
        elif k in ("q", "escape"):
            result["action"] = "quit"
            fig.canvas.stop_event_loop()

    cids.append(fig.canvas.mpl_connect("key_press_event",     on_key_press))
    cids.append(fig.canvas.mpl_connect("key_release_event",   editor.on_key_release))
    cids.append(fig.canvas.mpl_connect("button_press_event",  editor.on_press))
    cids.append(fig.canvas.mpl_connect("motion_notify_event", editor.on_motion))
    cids.append(fig.canvas.mpl_connect("button_release_event", editor.on_release))

    fig.canvas.draw()
    fig.canvas.start_event_loop(0)   # blocks until stop_event_loop()

    for cid in cids:
        fig.canvas.mpl_disconnect(cid)

    if result["action"] == "save":
        return "save", editor.entries
    return result["action"], None


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Review interactif des masques de segmentation")
    parser.add_argument("--folder", type=str, required=True,
                        help="Dossier vidéo contenant images/ et mask_labels/")
    parser.add_argument("--start",  type=int, default=0,
                        help="Index de départ (défaut: 0)")
    args = parser.parse_args()

    image_dir = os.path.join(args.folder, "images")
    mask_dir  = os.path.join(args.folder, "mask_labels")

    if not os.path.isdir(mask_dir):
        print(f"[ERREUR] Dossier mask_labels/ introuvable : {mask_dir}")
        sys.exit(1)

    mask_files = sorted(
        p for p in glob.glob(os.path.join(mask_dir, "*.txt"))
        if os.path.getsize(p) > 0
    )
    if not mask_files:
        print("Aucun masque trouvé.")
        return

    total = len(mask_files)
    print(f"Masques à reviewer : {total}")
    print("S=sauvegarder  E=supprimer  ←→=naviguer  R=image brute  N=nouveau masque  Q=quitter\n")

    stats = {"saved": 0, "deleted": 0, "skipped": 0}

    # Single window reused across all images, keeps its size
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.show(block=False)

    i = args.start

    while 0 <= i < total:
        mask_path = mask_files[i]
        stem      = os.path.basename(mask_path).rsplit(".", 1)[0]

        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            c = os.path.join(image_dir, stem + ext)
            if os.path.exists(c):
                img_path = c
                break

        if img_path is None:
            print(f"  [!] Image introuvable pour {stem}, skip.")
            i += 1
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [!] Lecture impossible : {img_path}, skip.")
            i += 1
            continue

        try:
            entries = load_seg_labels(mask_path)
        except Exception as e:
            print(f"  [!] Label illisible {stem}: {e}, skip.")
            i += 1
            continue

        if not entries:
            i += 1
            continue

        action, new_entries = review_image(fig, ax, img_bgr, entries, stem, i + 1, total,
                                           mask_path=mask_path)

        if action == "quit":
            print("\nReview interrompue.")
            break
        elif action == "save":
            save_seg_labels(mask_path, new_entries)
            stats["saved"] += 1
            print(f"  [{i+1}/{total}] {stem} → sauvegardé")
            i += 1
        elif action == "delete":
            os.remove(mask_path)
            stats["deleted"] += 1
            print(f"  [{i+1}/{total}] {stem} → masque supprimé (image conservée)")
            i += 1
        elif action == "next":
            stats["skipped"] += 1
            i += 1
        elif action == "prev":
            i = max(0, i - 1)

    print(f"\n{'='*45}")
    print(f"  Sauvegardés : {stats['saved']}")
    print(f"  Supprimés   : {stats['deleted']}")
    print(f"  Ignorés     : {stats['skipped']}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
