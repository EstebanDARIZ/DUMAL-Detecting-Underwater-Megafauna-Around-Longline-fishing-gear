import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

from config import _PALETTE_BGR

from utils import yolo_to_xyxy



def palette_bgr(cls_id: int):
    return _PALETTE_BGR[int(cls_id) % len(_PALETTE_BGR)]

# ──────────────────────────────────────────────────────────
# Live Tkinter window (replaces the OpenCV LiveWindow)
# ──────────────────────────────────────────────────────────
class LiveWindow:
    """
    Shows SAM frames live in a Tkinter window.
    Compatible with opencv-headless, no cv2.imshow calls.
    Shows SAM3's predicted bboxes with a colored overlay.
    """

    MAX_W = 960
    MAX_H = 540

    def __init__(self):
        self._top    = tk.Toplevel()
        self._top.title("SAM3 - Tracking live  |  bouton Stop pour arreter")
        self._top.configure(bg="#0d1117")
        self._top.resizable(True, True)
        self._top.grab_release()

        self._canvas = tk.Canvas(self._top, bg="#0d1117",
                                 highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        self._status = tk.Label(
            self._top, text="", bg="#161b22", fg="#8b949e",
            font=("Courier", 9))
        self._status.pack(fill="x")

        self._photo  = None
        self._closed = False
        self._top.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._closed = True
        try:
            self._top.destroy()
        except Exception:
            pass

    def show(self, frame_bgr: np.ndarray, global_idx: int,
             total: int, class_ids: list,
             detected: list, lost: list, lost_thr: int,
             live_labels: list = None):
        """
        Show the frame with SAM3's bboxes drawn as an overlay.

        live_labels: list of [cls_id, xc, yc, bw, bh] (normalized YOLO format)
                     for the current frame's detections.
        """
        if self._closed:
            return

        H, W = frame_bgr.shape[:2]
        scale = min(1.0, self.MAX_W / W, self.MAX_H / H)
        dw    = int(W * scale)
        dh    = int(H * scale)

        # ── Text + bbox overlay via OpenCV ──
        vis = frame_bgr.copy()

        # Draw the detected SAM bboxes
        if live_labels:
            for lb in live_labels:
                cls_id = int(lb[0])
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4], W, H)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color_bgr = palette_bgr(cls_id)

                # Find whether this object is detected
                obj_i = -1
                for idx_c, cid in enumerate(class_ids):
                    if cid == cls_id:
                        obj_i = idx_c
                        break

                is_det = detected[obj_i] if 0 <= obj_i < len(detected) else True
                thickness = 3 if is_det else 1

                # Bbox rectangle
                cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, thickness)

                # Semi-transparent fill
                overlay = vis.copy()
                alpha = 0.18 if is_det else 0.05
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
                cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

                # Bbox label
                lo_val = lost[obj_i] if 0 <= obj_i < len(lost) else 0
                sym    = "OK" if is_det else f"PERDU {lo_val}/{lost_thr}"
                label  = f"cls{cls_id} {sym}"
                lx, ly = x1, max(y1 - 6, 14)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + 2),
                               color_bgr, -1)
                cv2.putText(vis, label, (lx + 2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)

        # Per-object status (left column)
        y_off = 28
        for i, (det, lo) in enumerate(zip(detected, lost)):
            color_bgr = palette_bgr(class_ids[i]) if det else (80, 80, 80)
            sym = "OK" if det else f"PERDU {lo}/{lost_thr}"
            txt = f"Obj {i+1} cls={class_ids[i]}  {sym}"
            cv2.putText(vis, txt, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color_bgr, 2, cv2.LINE_AA)
            y_off += 24

        cv2.putText(vis,
                    f"Frame {global_idx}  |  Clic Stop pour arreter",
                    (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb).resize(
            (dw, dh), Image.BILINEAR)

        try:
            self._top.geometry(f"{dw}x{dh + 24}")
            self._canvas.config(width=dw, height=dh)
            self._photo = ImageTk.PhotoImage(pil_img)
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

            n_det = sum(detected)
            n_tot = len(detected)
            lost_parts = "  ".join(
                f"Obj{i+1}={'OK' if det else f'PERDU {lo}'}"
                for i, (det, lo) in enumerate(zip(detected, lost))
            )
            self._status.config(
                text=f"Frame {global_idx} / {total}  |  "
                     f"{n_det}/{n_tot} detectes  |  {lost_parts}"
            )
            self._top.update_idletasks()
            self._top.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self._top.destroy()
            except Exception:
                pass
            self._closed = True
