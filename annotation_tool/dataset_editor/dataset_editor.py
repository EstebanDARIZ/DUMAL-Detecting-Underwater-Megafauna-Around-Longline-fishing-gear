import os
import sys
import shutil
import queue

import cv2
import numpy as np
from datetime import datetime

import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec



from config import NEW_BOX_MIN, CLASS_NAMES, _PALETTE_BGR

from utils import palette_rgb, ask_class_id, yolo_to_xyxy, xyxy_to_yolo, read_labels, write_labels, select_bboxes_for_sam, ask_sam_config, get_handle, apply_handle_drag

from sam_thread import SAMThread


# ──────────────────────────────────────────────────────────
# Main class: editor + SAM
# ──────────────────────────────────────────────────────────
class DatasetEditor:
    def __init__(self, folder: str, fps: float = 10.0, video_path: str = None,
                 model: str = "sam3.pt"):
        self.folder     = folder
        self.imgs_dir   = os.path.join(folder, "images")
        self.labels_dir = os.path.join(folder, "labels")
        self.tmp_dir    = os.path.join(folder, "_tmp")
        self.fps        = fps
        self.interval   = 1.0 / fps
        self.video_path = video_path
        self.model      = model

        if not video_path or not os.path.exists(video_path):
            print(f"[ERREUR] --video requis et doit exister : {video_path}")
            sys.exit(1)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
            sys.exit(1)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(self.imgs_dir,   exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        print(f"  Vidéo : {os.path.basename(video_path)}  "
              f"({self.total_frames} frames, {self.W}x{self.H})")

        # Editor state
        self.idx             = 0
        self.playing         = False
        self.labels          = []
        self.sel_bbox        = None
        self.drag_handle     = None
        self.drag_start      = None
        self.drag_orig       = None
        self.new_box_start   = None
        self._last_mouse_pos = None
        self._timer          = None
        self._hide_boxes     = False
        self._drag_bg        = None   # cached background for drag blitting
        self._drag_rect      = None   # animated patch during the drag
        self._zoom_xlim      = None   # zoom limits kept across frames
        self._zoom_ylim      = None
        self._ctrl_pressed   = False
        self._draw_mode      = False  # True = drawing a bbox, False = pan
        self._draw_class     = None   # pre-selected class (None = ask)
        self._erase_mode     = False  # True = left click deletes the bbox
        self._pan_start      = None   # (mx, my) in data coords at the start of the pan
        self._pan_xlim0      = None
        self._pan_ylim0      = None

        # SAM state
        self._sam_thread         = None
        self._sam_running        = False
        self._sam_live_win       = None
        self._merge_mode         = "merge"
        self._poll_timer         = None
        self._sam_class_ids      = []
        self._sam_total_frames   = 0
        self._sam_lost_thr       = 15
        self._sam_conf           = 0.35
        self._sam_segment_count  = 0

        # Latest SAM live_labels (for the main window display)
        self._sam_live_labels  = []
        self._sam_detected     = []
        self._sam_lost         = []
        self._sam_global_idx   = 0

        self._build_ui()
        self._load_frame(0)

    def _read_video_frame(self, idx: int) -> np.ndarray | None:
        """Read frame idx from the source video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def _save_current_annotation(self):
        """Save the image+label if bboxes exist, otherwise delete the files."""
        if self.labels:
            os.makedirs(self.imgs_dir,   exist_ok=True)
            os.makedirs(self.labels_dir, exist_ok=True)
            if not os.path.exists(self.current_img_path):
                cv2.imwrite(self.current_img_path, self._current_img_bgr)
            write_labels(self.current_label_path, self.labels)
        else:
            for p in (self.current_label_path, self.current_img_path):
                if os.path.exists(p):
                    os.remove(p)

    # ── Build the UI ───────────────────────────────────────
    def _build_ui(self):
        plt.rcParams.update({
            "figure.facecolor": "#0d1117",
            "axes.facecolor":   "#0d1117",
            "text.color":       "#c9d1d9",
            "axes.edgecolor":   "#30363d",
        })

        self.fig = plt.figure(figsize=(15, 8.5))
        self.fig.canvas.manager.set_window_title(
            "Dataset Editor + SAM3 - YOLO Label Viewer")
        self.fig.patch.set_facecolor("#0d1117")

        gs = gridspec.GridSpec(
            3, 1,
            height_ratios=[0.905, 0.045, 0.05],
            hspace=0.06,
            figure=self.fig
        )

        self.ax = self.fig.add_subplot(gs[0])
        self.ax.axis("off")

        # Slider
        ax_slider = self.fig.add_subplot(gs[1])
        ax_slider.set_facecolor("#161b22")
        self.slider = Slider(
            ax_slider, "", 0, max(self.total_frames - 1, 1),
            valinit=0, valstep=1, color="#238636",
        )
        self.slider.label.set_color("#8b949e")
        self.slider.valtext.set_color("#00ff88")
        self.slider.on_changed(self._on_slider)

        # Buttons
        fig_left  = self.fig.subplotpars.left
        fig_right = self.fig.subplotpars.right
        fig_w     = fig_right - fig_left

        nav_btns = [
            ("-100",  "#1f2937", self._btn_m100),
            ("-10",   "#1f2937", self._btn_m10),
            ("-1",    "#1f2937", self._btn_m1),
            ("Pause", "#0d4f3c", self._btn_play),
            ("+1",    "#1f2937", self._btn_p1),
            ("+10",   "#1f2937", self._btn_p10),
            ("+100",  "#1f2937", self._btn_p100),
            ("Delete","#4a1010", self._btn_delete),
            ("SAM",   "#1a3a5c", self._btn_launch_sam),
            ("Stop",  "#4a2800", self._btn_stop_sam),
        ]

        self._buttons = {}
        n_btns = len(nav_btns)
        margin = 0.008
        bw = (1.0 - margin * (n_btns + 1)) / n_btns

        for k, (label, color, cb) in enumerate(nav_btns):
            bx = margin + k * (bw + margin)
            ax_b = self.fig.add_axes([
                fig_left + bx * fig_w,
                0.005,
                bw * fig_w,
                0.034,
            ])
            btn = Button(ax_b, label, color=color, hovercolor="#2d333b")
            btn.label.set_color("#c9d1d9")
            btn.label.set_fontsize(8)
            btn.label.set_fontfamily("monospace")
            btn.on_clicked(cb)
            self._buttons[label] = btn

        # Toggle merge/replace (clickable text)
        self._merge_text = self.fig.text(
            fig_left + 0.78 * fig_w + 0.005,
            0.052,
            "Mode SAM:  [Merge]  Replace",
            ha="left", va="center",
            color="#8b949e", fontsize=8, fontfamily="monospace",
            picker=True
        )
        self.fig.canvas.mpl_connect("pick_event", self._on_pick_merge)

        # Event connections
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)
        self.fig.canvas.mpl_connect("key_release_event",    self._on_key_release)
        self.fig.canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("scroll_event",         self._on_scroll)

        self._status = self.fig.text(
            0.5, 0.980, "", ha="center", va="top",
            color="#c9d1d9", fontsize=9, fontfamily="monospace",
            bbox=dict(facecolor="#161b22", alpha=0.0, pad=0, linewidth=0),
        )

        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.935, bottom=0.055
        )

    # ── Toggle merge/replace ───────────────────────────────
    def _on_pick_merge(self, event):
        self._merge_mode = "replace" if self._merge_mode == "merge" else "merge"
        self._refresh_merge_label()

    def _refresh_merge_label(self):
        if self._merge_mode == "merge":
            txt = "Mode SAM:  [Merge]  Replace"
        else:
            txt = "Mode SAM:   Merge  [Replace]"
        self._merge_text.set_text(txt)
        self.fig.canvas.draw_idle()

    # ── Load a frame ───────────────────────────────────────
    def _load_frame(self, idx: int):
        self.idx = max(0, min(idx, self.total_frames - 1))
        frame_name = f"frame_{self.idx:06d}"
        self.current_img_path   = os.path.join(self.imgs_dir,   frame_name + ".jpg")
        self.current_label_path = os.path.join(self.labels_dir, frame_name + ".txt")
        self.labels   = read_labels(self.current_label_path)
        self.sel_bbox = None

        img_bgr = self._read_video_frame(self.idx)
        if img_bgr is None:
            placeholder = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            placeholder[:, :] = (30, 30, 80)
            cv2.putText(placeholder, f"Frame illisible : {self.idx}",
                        (self.W//2 - 160, self.H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
            self._current_img_bgr = placeholder
            self._update_display()
            self._status.set_color("#ff4444")
            self._status.set_text(f"[ERREUR] Frame illisible : {self.idx}")
            return
        self._current_img_bgr = img_bgr

        self._update_display()
        self.slider.eventson = False
        self.slider.set_val(self.idx)
        self.slider.eventson = True

        annotated = os.path.exists(self.current_label_path)
        sam_info  = "  SAM en cours... X=stopper" if self._sam_running else \
                    "  S=lancer SAM3"
        self._status.set_color("#c9d1d9")
        self._status.set_text(
            f"Frame {self.idx}/{self.total_frames - 1}  |  "
            f"{frame_name}  |  "
            f"{'[annotée] ' if annotated else ''}"
            f"{len(self.labels)} bbox(s)  |  "
            f"{'PLAY' if self.playing else 'PAUSE'}"
            f"{sam_info}"
        )

    def _update_display(self):
        img_rgb = cv2.cvtColor(self._current_img_bgr, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(img_rgb, aspect="equal")
        if self._zoom_xlim is not None:
            self.ax.set_xlim(self._zoom_xlim)
            self.ax.set_ylim(self._zoom_ylim)
    
        if not self._hide_boxes:  # <- the whole boxes section lives inside this if
            for i, lb in enumerate(self.labels):
                cls_id          = int(lb[0])
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4],
                                               self.W, self.H)
                color = palette_rgb(cls_id)
                lw = 2.5 if i == self.sel_bbox else 1.5
                ls = "-"  if i == self.sel_bbox else "--"
    
                self.ax.add_patch(mpatches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=lw, edgecolor=color,
                    facecolor=(*color, 0.08), linestyle=ls
                ))
                if i == self.sel_bbox:
                    cx = (x1+x2)/2; cy = (y1+y2)/2
                    for hx, hy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2),
                                    (cx,y1),(cx,y2),(x1,cy),(x2,cy)]:
                        self.ax.plot(hx, hy, "o", ms=5, color="white",
                                     markerfacecolor=color, markeredgewidth=1)
    
            # Color legend, bottom left
            present_cls = sorted({int(lb[0]) for lb in self.labels})
            if present_cls:
                legend_handles = [
                    mpatches.Patch(
                        facecolor=palette_rgb(c), edgecolor="white", linewidth=0.5,
                        label=CLASS_NAMES.get(c, f"cls {c}")
                    )
                    for c in present_cls
                ]
                self.ax.legend(
                    handles=legend_handles,
                    loc="lower left",
                    fontsize=8,
                    framealpha=0.55,
                    facecolor="#0d1117",
                    edgecolor="#30363d",
                    labelcolor="white",
                    handlelength=1.2,
                    handleheight=1.0,
                    borderpad=0.5,
                    labelspacing=0.3,
                )

            if self._sam_running:
                self._status.set_text(
                    "SAM3 EN COURS  —  touche X ou bouton Stop pour arreter"
                )
                self._status.set_color("#ff8c00")

        else:
            # Small discreet indicator, bottom left
            self.ax.text(
                8, self.H - 8, "image brute  [r]",
                ha="left", va="bottom", fontsize=8,
                color="white", fontfamily="monospace",
                bbox=dict(facecolor="#0d1117", alpha=0.55, pad=2, linewidth=0)
            )

        # Eraser mode indicator
        if self._erase_mode:
            self.ax.text(
                self.W - 8, self.H - 8, "GOMME  [k]",
                ha="right", va="bottom", fontsize=9,
                color="#ff4444", fontfamily="monospace",
                bbox=dict(facecolor="#0d1117", alpha=0.65, pad=3, linewidth=0)
            )

        # Temporary rectangle + crosshair in draw mode
        if not self.playing and self._draw_mode \
                and self._last_mouse_pos is not None:
            mx, my = self._last_mouse_pos
            if self.new_box_start is not None:
                sx, sy = self.new_box_start
                x1t, y1t = min(sx, mx), min(sy, my)
                x2t, y2t = max(sx, mx), max(sy, my)
                self.ax.add_patch(mpatches.Rectangle(
                    (x1t, y1t), x2t - x1t, y2t - y1t,
                    linewidth=2, edgecolor=(0.0, 1.0, 0.5),
                    facecolor=(0.0, 1.0, 0.5, 0.1), linestyle="-"
                ))
            self.ax.axhline(my, color="white", lw=0.6, ls="--", alpha=0.45)
            self.ax.axvline(mx, color="white", lw=0.6, ls="--", alpha=0.45)

        self.fig.canvas.draw_idle()

    def _update_display_sam(self, img_bgr: np.ndarray, global_idx: int,
                             detected: list, lost: list,
                             live_labels: list):
        """
        Update the matplotlib display during SAM tracking.
        The boxes are drawn directly onto the pixel buffer (cv2), which
        avoids any matplotlib coordinate-system headaches.
        """
        H_disp, W_disp = img_bgr.shape[:2]

        vis = img_bgr.copy()
        for lb in live_labels:
            cls_id = int(lb[0])
            x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4],
                                            W_disp, H_disp)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color_bgr = _PALETTE_BGR[cls_id % len(_PALETTE_BGR)]

            obj_i = next((i for i, cid in enumerate(self._sam_class_ids)
                          if cid == cls_id), -1)
            is_det = detected[obj_i] if 0 <= obj_i < len(detected) else True
            lo_val = lost[obj_i]     if 0 <= obj_i < len(lost)     else 0

            thickness = 2 if is_det else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, thickness)

            overlay = vis.copy()
            alpha = 0.18 if is_det else 0.05
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            sym = "OK" if is_det else f"PERDU {lo_val}"
            label = f"cls{cls_id} {sym}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx = x1
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + 2),
                          color_bgr, -1)
            cv2.putText(vis, label, (lx + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

        img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(img_rgb, aspect="equal")

        n_det = sum(detected)
        n_tot = len(detected)

        # ── Update the slider ──
        self.slider.eventson = False
        self.slider.set_val(global_idx)
        self.slider.eventson = True
        self.idx = global_idx

        self._status.set_text(
            f"SAM3 EN COURS  |  frame {global_idx}  |  "
            f"{n_det}/{n_tot} detecte(s)  |  "
            + "  ".join(
                f"Obj{i+1}={'OK' if det else f'PERDU {lo}'}"
                for i, (det, lo) in enumerate(zip(detected, lost))
            )
            + "  |  X=stopper"
        )
        self._status.set_color("#ff4444" if n_det < n_tot else "#ff8c00")

        self.fig.canvas.draw_idle()

    # ── Playback timer ──────────────────────────────────────
    def _start_timer(self):
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.fig.canvas.new_timer(
            interval=int(self.interval * 1000))
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self):
        if not self.playing:
            self._stop_timer(); return
        if self.idx >= self.total_frames - 1:
            self.playing = False; self._stop_timer(); return
        self._load_frame(self.idx + 1)

    # ── SAM polling timer ───────────────────────────────────
    def _start_poll_timer(self):
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._poll_timer = self.fig.canvas.new_timer(interval=80)
        self._poll_timer.add_callback(self._poll_sam_queue)
        self._poll_timer.start()

    def _stop_poll_timer(self):
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None

    def _poll_sam_queue(self):
        if self._sam_thread is None:
            return
        last_frame_msg = None
        terminal_msgs = []
        try:
            while True:
                msg = self._sam_thread.result_queue.get_nowait()
                if msg["type"] == "frame":
                    last_frame_msg = msg
                else:
                    terminal_msgs.append(msg)
        except queue.Empty:
            pass
        if last_frame_msg is not None:
            self._handle_sam_msg(last_frame_msg)
        for msg in terminal_msgs:
            self._handle_sam_msg(msg)

    def _handle_sam_msg(self, msg):
        t = msg["type"]

        if t == "frame":
            live_labels = msg.get("live_labels", [])
            detected    = msg["detected"]
            lost        = msg["lost"]
            global_idx  = msg["global_idx"]

            # ── Store the current SAM state ──
            self._sam_live_labels = live_labels
            self._sam_detected    = detected
            self._sam_lost        = lost
            self._sam_global_idx  = global_idx
            self._current_img_bgr = msg["img_bgr"]

            # ── Update the main matplotlib window ──
            self._update_display_sam(
                msg["img_bgr"],
                global_idx,
                detected,
                lost,
                live_labels,
            )

            # ── Update the live Tkinter window ──
            if self._sam_live_win is not None:
                try:
                    self._sam_live_win.show(
                        msg["img_bgr"],
                        global_idx,
                        self._sam_total_frames,
                        self._sam_class_ids,
                        detected,
                        lost,
                        self._sam_lost_thr,
                        live_labels=live_labels,
                    )
                except Exception:
                    pass

        elif t == "done":
            self._on_sam_done(
                msg["last_frame"],
                msg["stopped"],
                msg.get("reason", "termine"),
            )

        elif t == "error":
            print(f"[SAM ERREUR]\n{msg['msg']}")
            self._on_sam_done(-1, True, "erreur")

    def _on_sam_done(self, last_frame: int, stopped: bool, reason: str = ""):
        self._sam_running = False
        self._stop_poll_timer()

        if self._sam_live_win is not None:
            self._sam_live_win.close()
            self._sam_live_win = None

        # Reset the live SAM data
        self._sam_live_labels = []
        self._sam_detected    = []
        self._sam_lost        = []

        print(f"  [SAM] Tracking terminé ({reason}). Dernière frame : {last_frame}")

        # Jump to the last processed frame
        self._load_frame(last_frame if last_frame >= 0 else self.idx)
        self._update_display()

    # ── Launch SAM ───────────────────────────────────────────
    def _launch_sam(self):
        if self._sam_running:
            print("  [SAM] Deja en cours.")
            return
    
        if not self.video_path or not os.path.exists(self.video_path):
            print("  [SAM] Aucun fichier video source.")
            self._show_no_video_dialog()
            return
    
        self.playing = False
        self._stop_timer()
    
        # 1) Config: number of objects + classes + number of negatives
        cfg = ask_sam_config(
            default_n=max(1, len(self.labels)),
            default_classes=[int(lb[0]) for lb in self.labels] or [0]
        )
        if cfg is None:
            print("  [SAM] Annule.")
            return
        n_objects, class_ids, n_neg = cfg
        print(f"  [SAM] {n_objects} objet(s), classes={class_ids}, "
              f"{n_neg} bbox(s) negative(s)")
    
        # 2) Current frame from memory (already read from the video)
        frame_bgr = self._current_img_bgr
        if frame_bgr is None:
            print("  [SAM] Impossible de lire la frame courante.")
            return

        # 3) Draw positive bboxes, then negative ones
        result = select_bboxes_for_sam(frame_bgr, class_ids, n_neg=n_neg)
        if result is None:
            print("  [SAM] Annule lors du dessin des bboxes.")
            return
        bboxes_pos, bboxes_neg = result

        # 4) Starting frame = current idx in the video
        start_frame = self.idx
        print(f"  [SAM] Depart frame video : {start_frame}")

        # 5) Source video info (already known)
        fps          = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = self.W
        height       = self.H
        total_frames = self.total_frames
    
        # 6) Store for the live window
        self._sam_class_ids     = class_ids
        self._sam_total_frames  = total_frames
        self._sam_lost_thr      = 15
        self._sam_conf          = 0.35
        self._sam_segment_count += 1
    
        # 7) Open the live Tkinter window
        self._sam_live_win = None  # live window disabled

        # 8) Launch the SAM thread
        self._sam_thread = SAMThread(
            video_path     = self.video_path,
            start_frame    = start_frame,
            fps            = fps,
            width          = width,
            height         = height,
            total_frames   = total_frames,
            bboxes         = bboxes_pos,
            neg_bboxes     = bboxes_neg,
            class_ids      = class_ids,
            conf           = 0.35,
            imgs_dir       = self.imgs_dir,
            labels_dir     = self.labels_dir,
            tmp_dir        = self.tmp_dir,
            merge_mode     = self._merge_mode,
            lost_threshold = self._sam_lost_thr,
            model          = self.model,
        )
        self._sam_running = True
        self._sam_thread.start()
        self._start_poll_timer()
    
        print(f"  [SAM] Thread lance - merge_mode={self._merge_mode} "
              f"- lost_threshold={self._sam_lost_thr} "
              f"- neg_bboxes={len(bboxes_neg)}")
        self._update_display()

    def _show_no_video_dialog(self):
        top = tk.Toplevel()
        top.title("Video source manquante")
        top.configure(bg="#0d1117")
        top.resizable(False, False)
        top.grab_set()
        w, h = 380, 100
        top.update_idletasks()
        sw = top.winfo_screenwidth(); sh = top.winfo_screenheight()
        top.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        tk.Label(top,
                 text="Aucun fichier video source.\n"
                      "Relancer avec :  --video /chemin/video.mp4",
                 bg="#0d1117", fg="#ff6060",
                 font=("Courier", 10), justify="center").pack(pady=20)
        tk.Button(top, text="OK", command=top.destroy,
                  bg="#238636", fg="white", font=("Courier", 9),
                  relief="flat", padx=10, pady=3).pack()
        top.wait_window(top)

    def _stop_sam(self):
        if self._sam_thread is not None and self._sam_running:
            print("  [SAM] Arret demande...")
            self._sam_thread.stop()
        else:
            print("  [SAM] Pas de tracking en cours.")

    # ── Navigation controls ─────────────────────────────────
    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._start_timer()
        else:
            self._stop_timer()
        self._load_frame(self.idx)

    def _go(self, delta):
        self.playing = False
        self._stop_timer()
        self._load_frame(self.idx + delta)

    # ── Button callbacks ────────────────────────────────────
    def _btn_m100(self, _): self._go(-100)
    def _btn_m10(self, _):  self._go(-10)
    def _btn_m1(self, _):   self._go(-1)
    def _btn_p1(self, _):   self._go(1)
    def _btn_p10(self, _):  self._go(10)
    def _btn_p100(self, _): self._go(100)
    def _btn_play(self, _): self._toggle_play()
    def _btn_delete(self, _): self._delete_current_frame()
    def _btn_launch_sam(self, _): self._launch_sam()
    def _btn_stop_sam(self, _):   self._stop_sam()

    def _set_cursor(self):
        try:
            if self._draw_mode:
                cur = "crosshair"
            elif self._erase_mode:
                cur = "X_cursor"
            else:
                cur = ""
            self.fig.canvas.get_tk_widget().config(cursor=cur)
        except Exception:
            pass

    def _pan_update(self):
        """Fast view update with no re-imshow (just set_xlim/ylim)."""
        if self._zoom_xlim is not None:
            self.ax.set_xlim(self._zoom_xlim)
            self.ax.set_ylim(self._zoom_ylim)
        self.fig.canvas.draw_idle()

    def _zoom(self, factor, cx=None, cy=None):
        """factor > 1 = zoom in, < 1 = zoom out. cx/cy = center in image coordinates."""
        # Current limits (or the initial ones if not zoomed yet)
        xlim = self._zoom_xlim if self._zoom_xlim is not None else self.ax.get_xlim()
        ylim = self._zoom_ylim if self._zoom_ylim is not None else self.ax.get_ylim()

        x0, x1 = xlim
        y0, y1 = ylim  # y0 > y1 (image axis is inverted)

        x_span = x1 - x0          # positive
        y_span = y0 - y1          # positive (roughly H)

        if cx is None:
            cx = (x0 + x1) / 2
        if cy is None:
            cy = (y0 + y1) / 2

        new_x_span = x_span / factor
        new_y_span = y_span / factor

        # Minimum zoom limit (no smaller than 20px per side)
        MIN_SPAN = 20
        if new_x_span < MIN_SPAN or new_y_span < MIN_SPAN:
            return

        # Fraction of the cursor's position within the current span
        fx = (cx - x0) / x_span if x_span else 0.5
        fy = (y0 - cy) / y_span if y_span else 0.5  # inverted

        new_x0 = cx - fx * new_x_span
        new_x1 = cx + (1 - fx) * new_x_span
        new_y0 = cy + fy * new_y_span       # y0 side is the "bottom of screen" (larger)
        new_y1 = cy - (1 - fy) * new_y_span # y1 side is the "top of screen" (smaller)

        # Clamp to the image, don't go past the edges
        img_x0, img_x1 = -0.5, self.W - 0.5
        img_y0, img_y1 = self.H - 0.5, -0.5  # inverted

        new_x0 = max(img_x0, new_x0)
        new_x1 = min(img_x1, new_x1)
        new_y0 = min(img_y0, new_y0)
        new_y1 = max(img_y1, new_y1)

        # Max zoom out = full view
        if factor < 1 and (new_x1 - new_x0 >= img_x1 - img_x0) \
                       and (new_y0 - new_y1 >= img_y0 - img_y1):
            self._zoom_xlim = None
            self._zoom_ylim = None
        else:
            self._zoom_xlim = (new_x0, new_x1)
            self._zoom_ylim = (new_y0, new_y1)

        self._update_display()

    def _on_scroll(self, event):
        if not self._ctrl_pressed or event.inaxes is not self.ax:
            return
        factor = 1.25 if event.button == "up" else 1 / 1.25
        self._zoom(factor, event.xdata, event.ydata)

    def _on_key_release(self, ev):
        if ev.key in ("control", "ctrl"):
            self._ctrl_pressed = False
        if ev.key == "r" and self._hide_boxes:
            self._hide_boxes = False
            self._update_display()

    # ── Keyboard ─────────────────────────────────────────────
    def _on_key(self, ev):
        k = ev.key
        if   k in ("control", "ctrl"): self._ctrl_pressed = True
        elif k == " ":        self._toggle_play()
        elif k == "left":     self._go(-1)
        elif k == "right":    self._go(1)
        elif k == "a":        self._go(-10)
        elif k == "d":        self._go(10)
        elif k == "q":        self._go(-100)
        elif k == "e":        self._go(100)
        elif k == "delete":   self._delete_current_frame()
        elif k in ("s", "S"): self._launch_sam()
        elif k in ("x", "X"): self._stop_sam()
        elif k in ("=", "+", "plus"):   self._zoom(1.25)
        elif k in ("-", "minus"):       self._zoom(1 / 1.25)
        elif k == ".":
            self._zoom_xlim = None
            self._zoom_ylim = None
            self._update_display()
        elif k in ("k", "K") and not self.playing:
            self._erase_mode = not self._erase_mode
            self._set_cursor()
            self._update_display()
        elif k == "n" and not self.playing:
            self._draw_mode  = not self._draw_mode
            self._draw_class = None
            self.new_box_start = None
            self._set_cursor()
            self._update_display()
        elif k.isdigit() and not self.playing:
            new_cls = int(k)
            if self._draw_mode and self._draw_class == new_cls:
                self._draw_mode  = False
                self._draw_class = None
            else:
                self._draw_class = new_cls
                self._draw_mode  = True
            self.new_box_start = None
            self._set_cursor()
            self._update_display()
        elif k == "r" and not self.playing:
            self._hide_boxes = True
            self._update_display()
        elif k == "escape":
            if self._draw_mode or self._erase_mode:
                self._draw_mode  = False
                self._draw_class = None
                self._erase_mode = False
                self.new_box_start = None
                self._set_cursor()
                self._update_display()
            else:
                self._stop_sam()
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
        if event.inaxes is self.ax:
            return event.xdata, event.ydata
        if (self.drag_handle is not None or self.new_box_start is not None) \
                and event.x is not None and event.y is not None:
            try:
                inv = self.ax.transData.inverted()
                x, y = inv.transform((event.x, event.y))
                x = max(0.0, min(float(self.W-1), x))
                y = max(0.0, min(float(self.H-1), y))
                return x, y
            except Exception:
                pass
        return None

    def _start_drag_blit(self, x1, y1, x2, y2):
        """Render the image + static bboxes (excluding sel_bbox), cache it, and create the animated patch."""
        img_rgb = cv2.cvtColor(self._current_img_bgr, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(img_rgb, aspect="equal")
        if self._zoom_xlim is not None:
            self.ax.set_xlim(self._zoom_xlim)
            self.ax.set_ylim(self._zoom_ylim)

        for i, lb in enumerate(self.labels):
            if i == self.sel_bbox:
                continue
            cls_id = int(lb[0])
            bx1, by1, bx2, by2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4], self.W, self.H)
            color = palette_rgb(cls_id)
            self.ax.add_patch(mpatches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=1.5, edgecolor=color,
                facecolor=(*color, 0.08), linestyle="--"
            ))

        self.fig.canvas.draw()
        self._drag_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        color = palette_rgb(int(self.labels[self.sel_bbox][0]))
        self._drag_rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color,
            facecolor=(*color, 0.15), linestyle="-",
            animated=True
        )
        self.ax.add_patch(self._drag_rect)

    def _on_mouse_press(self, event):
        if self.playing:
            return
        coords = self._ax_coords(event)
        if coords is None:
            return
        mx, my = coords

        # Right click: select first, only delete if already selected
        if event.button == 3:
            for i, lb in enumerate(self.labels):
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4],
                                               self.W, self.H)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    if self.sel_bbox == i:
                        del self.labels[i]
                        self._save_current_annotation()
                        print(f"  [-] Bbox cls{int(lb[0])} supprimee")
                        self.sel_bbox = None
                    else:
                        self.sel_bbox = i
                    self._update_display()
                    return
            return

        if event.button != 1:
            return

        # Eraser mode: left click immediately deletes the touched bbox
        if self._erase_mode:
            for i, lb in enumerate(self.labels):
                x1, y1, x2, y2 = yolo_to_xyxy(lb[1], lb[2], lb[3], lb[4],
                                               self.W, self.H)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    del self.labels[i]
                    self._save_current_annotation()
                    print(f"  [-] Bbox cls{int(lb[0])} supprimee (gomme)")
                    self.sel_bbox = None
                    self._update_display()
                    return
            return

        # In drawing mode, ignore existing bboxes
        if self._draw_mode:
            self.sel_bbox = None
            self.new_box_start = (mx, my)
            self._update_display()
            return

        hit_resize = None
        hit_inside = None

        indices = list(range(len(self.labels)))
        if self.sel_bbox is not None and self.sel_bbox in indices:
            indices.remove(self.sel_bbox)
            indices = [self.sel_bbox] + indices

        for i in indices:
            lb = self.labels[i]
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
            self._start_drag_blit(x1, y1, x2, y2)
            return

        if hit_inside is not None:
            i, x1, y1, x2, y2 = hit_inside
            self.sel_bbox    = i
            self.drag_handle = "inside"
            self.drag_start  = (mx, my)
            self.drag_orig   = (x1, y1, x2, y2)
            self._start_drag_blit(x1, y1, x2, y2)
            return

        self.sel_bbox = None
        if self._zoom_xlim is not None:
            # Pan: remember the current position + limits
            self._pan_start = (mx, my)
            self._pan_xlim0 = self._zoom_xlim
            self._pan_ylim0 = self._zoom_ylim

    def _on_mouse_move(self, event):
        if self.playing:
            return
        coords = self._ax_coords(event)
        if coords is not None:
            self._last_mouse_pos = coords
        if coords is None:
            return

        if self._pan_start is not None:
            mx, my = self._last_mouse_pos
            dx = mx - self._pan_start[0]
            dy = my - self._pan_start[1]
            x0, x1 = self._pan_xlim0
            y0, y1 = self._pan_ylim0          # y0 > y1 (inverted)
            x_span = x1 - x0
            y_span = y0 - y1                  # positif
            nx0, nx1 = x0 - dx, x1 - dx
            ny0, ny1 = y0 - dy, y1 - dy
            # Clamp to the image bounds
            img_x0, img_x1 = -0.5, self.W - 0.5
            img_y0, img_y1 = self.H - 0.5, -0.5
            if nx0 < img_x0:   nx0 = img_x0;  nx1 = img_x0 + x_span
            elif nx1 > img_x1: nx1 = img_x1;  nx0 = img_x1 - x_span
            if ny0 > img_y0:   ny0 = img_y0;  ny1 = img_y0 - y_span
            elif ny1 < img_y1: ny1 = img_y1;  ny0 = img_y1 + y_span
            self._zoom_xlim = (nx0, nx1)
            self._zoom_ylim = (ny0, ny1)
            self._pan_update()              # rapide : pas de re-imshow
            return

        if self.drag_handle is not None and self.drag_start is not None \
                and self._drag_bg is not None and self._drag_rect is not None:
            mx, my = self._last_mouse_pos
            dx = mx - self.drag_start[0]
            dy = my - self.drag_start[1]
            ox1, oy1, ox2, oy2 = self.drag_orig
            nx1, ny1, nx2, ny2 = apply_handle_drag(
                self.drag_handle, dx, dy, ox1, oy1, ox2, oy2, self.W, self.H)
            self._drag_rect.set_xy((nx1, ny1))
            self._drag_rect.set_width(nx2 - nx1)
            self._drag_rect.set_height(ny2 - ny1)
            self.fig.canvas.restore_region(self._drag_bg)
            self.ax.draw_artist(self._drag_rect)
            self.fig.canvas.blit(self.ax.bbox)
            return

        # Crosshair + temporary rect handled in _update_display
        self._update_display()

    def _on_mouse_release(self, event):
        if self.playing:
            return

        if self._pan_start is not None:
            self._pan_start = None
            self._pan_xlim0 = None
            self._pan_ylim0 = None
            self._update_display()  # full redraw to sync the state
            return

        if self.drag_handle is not None:
            coords = self._ax_coords(event)
            if coords is None:
                coords = self._last_mouse_pos
            if coords is not None:
                mx, my = coords
                dx = mx - self.drag_start[0]
                dy = my - self.drag_start[1]
                ox1, oy1, ox2, oy2 = self.drag_orig
                nx1, ny1, nx2, ny2 = apply_handle_drag(
                    self.drag_handle, dx, dy, ox1, oy1, ox2, oy2, self.W, self.H)
                lb = self.labels[self.sel_bbox]
                lb[1], lb[2], lb[3], lb[4] = xyxy_to_yolo(
                    nx1, ny1, nx2, ny2, self.W, self.H)
            # Clean up the animated patch
            if self._drag_rect is not None:
                self._drag_rect.remove()
                self._drag_rect = None
            self._drag_bg   = None
            self.drag_handle = None
            self.drag_start  = None
            self.drag_orig   = None
            self._save_current_annotation()
            print(f"  [OK] Bbox mise a jour - frame {self.idx+1}")
            self._update_display()
            return

        if self.new_box_start is None:
            return

        sx, sy = self.new_box_start
        self.new_box_start = None

        coords = self._ax_coords(event)
        if coords is not None:
            mx, my = coords
        elif self._last_mouse_pos is not None:
            mx, my = self._last_mouse_pos
        else:
            self._update_display(); return

        x1, y1 = min(sx, mx), min(sy, my)
        x2, y2 = max(sx, mx), max(sy, my)

        if (x2 - x1) <= NEW_BOX_MIN and (y2 - y1) <= NEW_BOX_MIN:
            self._update_display(); return

        cls_id = self._draw_class if self._draw_class is not None else ask_class_id(default=0)
        if cls_id is not None:
            xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, self.W, self.H)
            self.labels.append([cls_id, xc, yc, bw, bh])
            self.sel_bbox    = len(self.labels) - 1  # select it immediately
            self._draw_mode  = False                  # back to pan mode
            self._draw_class = None
            self._set_cursor()
            self._save_current_annotation()
            print(f"  [+] Nouvelle bbox cls{cls_id} - frame {self.idx+1}")
        else:
            print("  [!] Ajout annule.")
        self._update_display()

    
    def _delete_current_frame(self):
        del_dir = os.path.join(self.folder, "_deleted")
        os.makedirs(del_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S%f")
        moved = False
        for p in (self.current_img_path, self.current_label_path):
            if os.path.exists(p):
                shutil.move(p, os.path.join(del_dir, ts + "_" + os.path.basename(p)))
                moved = True
        if moved:
            print(f"  [del] Annotation frame {self.idx} -> _deleted/")
        else:
            print(f"  [del] Rien à supprimer pour la frame {self.idx}")
        self.labels   = []
        self.sel_bbox = None
        self._update_display()

    
    def run(self):
        plt.show(block=True)
        self.cap.release()