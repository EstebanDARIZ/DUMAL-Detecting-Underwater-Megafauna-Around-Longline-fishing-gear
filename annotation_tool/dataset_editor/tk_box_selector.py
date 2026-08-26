import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

from config import _PALETTE_PIL

def palette_pil(cls_id: int):
    return _PALETTE_PIL[int(cls_id) % len(_PALETTE_PIL)]

class TKBoxSelector:
    """
    Tkinter Canvas window for drawing a bbox via drag-and-drop.
    Compatible with opencv-headless (no cv2.namedWindow calls).

    Keys:
      Enter / Space    : confirm
      Esc              : cancel (returns None)
      R                : reset (clears the current bbox)
      + / =            : zoom in
      -                : zoom out
      .                : reset zoom
      Ctrl + wheel     : zoom centered on the cursor
      Ctrl + drag      : pan (move the view)
    """

    MAX_W = 1100
    MAX_H = 700

    def __init__(self):
        self._result  = None
        self._sx = self._sy = self._ex = self._ey = None
        self._drawing = False
        self._photo   = None

    def select(self, frame_bgr: np.ndarray, title: str,
           done_bboxes: list, class_ids: list,
           extra_boxes: list = None,
           box_color: str = "#00ff88",
           label_prefix: str = ""):
        self._result  = None
        self._sx = self._sy = self._ex = self._ey = None
        self._drawing = False

        H_img, W_img = frame_bgr.shape[:2]

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_source = Image.fromarray(frame_rgb)

        # ── Zoom / pan state ──
        zoom     = [1.0]       # multiplier on top of the fit-to-window scale
        img_ox   = [0]         # image's X origin within the canvas (canvas pixels)
        img_oy   = [0]         # Y origin
        pan_anch = [None]      # (cx, cy, ox0, oy0) for an active pan

        rect_id  = [None]

        # Window
        top = tk.Toplevel()
        top.title(title)
        top.configure(bg="#0d1117")
        top.resizable(True, True)
        top.grab_set()

        sw = top.winfo_screenwidth()
        sh = top.winfo_screenheight()
        init_s = min(1.0, self.MAX_W / W_img, self.MAX_H / H_img)
        init_w = int(W_img * init_s)
        init_h = int(H_img * init_s)
        wx = (sw - init_w) // 2
        wy = max(0, (sh - init_h - 60) // 2)
        top.geometry(f"{init_w}x{init_h + 26}+{wx}+{wy}")
        top.lift()
        top.focus_force()

        canvas = tk.Canvas(top, bg="#0d1117", cursor="crosshair",
                           highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        status_var = tk.StringVar(
            value="Ctrl+drag=pan  Ctrl+molette/+/-=zoom  .=reset  "
                  "Drag=dessiner  Entrée=valider  Échap=annuler  R=reset bbox")
        tk.Label(top, textvariable=status_var,
                 bg="#161b22", fg="#8b949e",
                 font=("Courier", 9)).pack(fill="x", side="bottom")

        # ── Zoom/pan helpers ─────────────────────────────────────
        def _fit_scale():
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 2 or ch < 2:
                return min(1.0, self.MAX_W / W_img, self.MAX_H / H_img)
            return min(cw / W_img, ch / H_img)

        def _scale():
            return _fit_scale() * zoom[0]

        def _clamp_pan():
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 2: return
            s  = _scale()
            dw = int(W_img * s)
            dh = int(H_img * s)
            if dw <= cw:
                img_ox[0] = (cw - dw) // 2
            else:
                img_ox[0] = max(cw - dw, min(0, img_ox[0]))
            if dh <= ch:
                img_oy[0] = (ch - dh) // 2
            else:
                img_oy[0] = max(ch - dh, min(0, img_oy[0]))

        def _to_img(cx, cy):
            s = _scale()
            ix = (cx - img_ox[0]) / s
            iy = (cy - img_oy[0]) / s
            ix = max(0.0, min(float(W_img - 1), ix))
            iy = max(0.0, min(float(H_img - 1), iy))
            return ix, iy

        def _to_canvas(ix, iy):
            s = _scale()
            return ix * s + img_ox[0], iy * s + img_oy[0]

        def _update_rect_coords():
            if rect_id[0] is not None and self._sx is not None \
                    and self._ex is not None:
                rx1, ry1 = _to_canvas(self._sx, self._sy)
                rx2, ry2 = _to_canvas(self._ex, self._ey)
                canvas.coords(rect_id[0], rx1, ry1, rx2, ry2)

        def _do_zoom(factor, cx=None, cy=None):
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 2: return
            fit_s = _fit_scale()
            s_old = fit_s * zoom[0]
            new_zoom = max(0.5, min(20.0, zoom[0] * factor))
            s_new    = fit_s * new_zoom
            if cx is None: cx = cw / 2
            if cy is None: cy = ch / 2
            img_ox[0] = int(cx - (cx - img_ox[0]) * s_new / s_old)
            img_oy[0] = int(cy - (cy - img_oy[0]) * s_new / s_old)
            zoom[0] = new_zoom
            _clamp_pan()
            _render()
            _update_rect_coords()
            _update_status_zoom()

        def _update_status_zoom():
            pct = int(zoom[0] * 100)
            status_var.set(
                f"Zoom {pct}%  |  Ctrl+drag=pan  +/-=zoom  .=reset  "
                "Drag=dessiner  Entrée=valider  Échap=annuler  R=reset bbox"
            )

        # ── Rendering ─────────────────────────────────────────────
        def _render():
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 2 or ch < 2: return

            _clamp_pan()
            s  = _scale()
            dw = int(W_img * s)
            dh = int(H_img * s)

            pil_disp = pil_source.resize((dw, dh), Image.BILINEAR)
            draw = ImageDraw.Draw(pil_disp)

            for k, b in enumerate(done_bboxes):
                color = palette_pil(class_ids[k])
                bx1 = int(b[0] * s); by1 = int(b[1] * s)
                bx2 = int(b[2] * s); by2 = int(b[3] * s)
                draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
                draw.text((bx1 + 4, max(by1 - 16, 2)),
                          f"{label_prefix} Obj {k+1} cls={class_ids[k]}", fill=color)

            if extra_boxes:
                for idx_neg, b in enumerate(extra_boxes):
                    bx1 = int(b[0] * s); by1 = int(b[1] * s)
                    bx2 = int(b[2] * s); by2 = int(b[3] * s)
                    draw.rectangle([bx1, by1, bx2, by2],
                                   outline=(220, 50, 50), width=2)
                    draw.text((bx1 + 4, max(by1 - 16, 2)),
                              f"NEG {idx_neg+1}", fill=(220, 50, 50))

            self._photo = ImageTk.PhotoImage(pil_disp)
            ox = img_ox[0]
            oy = img_oy[0]
            canvas.delete("img")
            canvas.create_image(ox, oy, anchor="nw",
                                image=self._photo, tags="img")
            if rect_id[0] is not None:
                canvas.tag_raise(rect_id[0])

        # ── Window resize ────────────────────────────────────────
        def _on_resize(ev):
            _render()
            _update_rect_coords()

        canvas.bind("<Configure>", _on_resize)

        top.update_idletasks()
        top.update()
        _clamp_pan()
        _render()

        # ── Draw the bbox ────────────────────────────────────────
        def _start(ev):
            if ev.state & 4:   # Ctrl: pan
                return
            self._drawing = True
            ix, iy = _to_img(ev.x, ev.y)
            self._sx, self._sy = ix, iy
            self._ex, self._ey = ix, iy
            if rect_id[0]:
                canvas.delete(rect_id[0])
            cx1, cy1 = _to_canvas(ix, iy)
            rect_id[0] = canvas.create_rectangle(
                cx1, cy1, cx1, cy1, outline=box_color, width=2)

        def _drag(ev):
            if not self._drawing:
                return
            ix, iy = _to_img(ev.x, ev.y)
            self._ex, self._ey = ix, iy
            if rect_id[0]:
                cx1, cy1 = _to_canvas(self._sx, self._sy)
                cx2, cy2 = _to_canvas(ix, iy)
                canvas.coords(rect_id[0], cx1, cy1, cx2, cy2)

        def _end(ev):
            if not self._drawing:
                return
            self._drawing = False
            ix, iy = _to_img(ev.x, ev.y)
            self._ex, self._ey = ix, iy
            if rect_id[0]:
                cx1, cy1 = _to_canvas(self._sx, self._sy)
                cx2, cy2 = _to_canvas(ix, iy)
                canvas.coords(rect_id[0], cx1, cy1, cx2, cy2)
            pw = abs(self._ex - self._sx)
            ph = abs(self._ey - self._sy)
            if pw > 2 and ph > 2:
                status_var.set(
                    f"Bbox : ({int(min(self._sx,self._ex))},"
                    f"{int(min(self._sy,self._ey))}) -> "
                    f"({int(max(self._sx,self._ex))},"
                    f"{int(max(self._sy,self._ey))})"
                    "  |  Entrée=valider  Échap=annuler  R=reset")
            else:
                status_var.set("Trop petite, recommence  |  R=reset")

        # ── Pan (Ctrl + drag) ──────────────────────────────────────
        def _pan_start(ev):
            if not (ev.state & 4): return
            pan_anch[0] = (ev.x, ev.y, img_ox[0], img_oy[0])
            canvas.config(cursor="fleur")

        def _pan_motion(ev):
            if pan_anch[0] is None: return
            ax, ay, ox0, oy0 = pan_anch[0]
            img_ox[0] = ox0 + (ev.x - ax)
            img_oy[0] = oy0 + (ev.y - ay)
            _clamp_pan()
            _render()
            _update_rect_coords()

        def _pan_end(ev):
            if pan_anch[0] is None: return
            pan_anch[0] = None
            canvas.config(cursor="crosshair")

        # ── Crosshair ─────────────────────────────────────────────
        ch_h = canvas.create_line(0, 0, 0, 0, fill="white", width=1,
                                  dash=(5, 4), state="hidden")
        ch_v = canvas.create_line(0, 0, 0, 0, fill="white", width=1,
                                  dash=(5, 4), state="hidden")

        def _on_motion(ev):
            # Pan motion (takes priority if Ctrl is held)
            _pan_motion(ev)
            # Draw drag
            _drag(ev)
            # Crosshair
            ch_h_var = ch_h; ch_v_var = ch_v
            ox, oy_ = img_ox[0], img_oy[0]
            s  = _scale()
            dw = int(W_img * s)
            dh = int(H_img * s)
            canvas.coords(ch_h_var, ox, ev.y, ox + dw, ev.y)
            canvas.coords(ch_v_var, ev.x, oy_, ev.x, oy_ + dh)
            canvas.itemconfigure(ch_h_var, state="normal")
            canvas.itemconfigure(ch_v_var, state="normal")
            canvas.tag_raise(ch_h_var)
            canvas.tag_raise(ch_v_var)
            if rect_id[0] is not None:
                canvas.tag_raise(rect_id[0])

        def _on_leave(ev):
            canvas.itemconfigure(ch_h, state="hidden")
            canvas.itemconfigure(ch_v, state="hidden")

        canvas.bind("<Motion>",          _on_motion)
        canvas.bind("<Leave>",           _on_leave)
        canvas.bind("<ButtonPress-1>",   _pan_start)
        canvas.bind("<ButtonPress-1>",   _start,    add="+")
        canvas.bind("<B1-Motion>",       _on_motion)
        canvas.bind("<ButtonRelease-1>", _pan_end)
        canvas.bind("<ButtonRelease-1>", _end,      add="+")

        # ── Mouse wheel ───────────────────────────────────────────
        def _on_scroll(ev):
            if not (ev.state & 4): return   # only with Ctrl held
            factor = 1.25 if (ev.delta > 0 or ev.num == 4) else 1 / 1.25
            _do_zoom(factor, ev.x, ev.y)

        canvas.bind("<MouseWheel>", _on_scroll)          # Windows/Mac
        canvas.bind("<Button-4>",   _on_scroll)          # Linux scroll up
        canvas.bind("<Button-5>",   lambda e: _on_scroll(
            type("E", (), {"state": e.state, "delta": -1,
                           "num": 5, "x": e.x, "y": e.y})()))

        # ── Confirm / cancel / reset ───────────────────────────────
        def _validate(ev=None):
            if self._sx is None or self._ex is None:
                status_var.set("Dessine d'abord une bbox !")
                return
            pw = abs(self._ex - self._sx)
            ph = abs(self._ey - self._sy)
            if pw <= 2 or ph <= 2:
                status_var.set("Bbox trop petite, recommence !")
                return
            x1 = int(min(self._sx, self._ex))
            y1 = int(min(self._sy, self._ey))
            x2 = int(max(self._sx, self._ex))
            y2 = int(max(self._sy, self._ey))
            x1 = max(0, min(x1, W_img - 1))
            y1 = max(0, min(y1, H_img - 1))
            x2 = max(0, min(x2, W_img - 1))
            y2 = max(0, min(y2, H_img - 1))
            self._result = [x1, y1, x2, y2]
            top.destroy()

        def _cancel(ev=None):
            self._result = None
            top.destroy()

        def _reset(ev=None):
            self._sx = self._sy = self._ex = self._ey = None
            self._drawing = False
            if rect_id[0]:
                canvas.delete(rect_id[0])
                rect_id[0] = None
            status_var.set(
                "Ctrl+drag=pan  Ctrl+molette/+/-=zoom  .=reset  "
                "Drag=dessiner  Entrée=valider  Échap=annuler  R=reset bbox")

        def _on_key(ev):
            k = ev.keysym
            if k in ("Return", "KP_Enter"):
                _validate()
            elif k == "Escape":
                _cancel()
            elif k in ("r", "R"):
                _reset()
            elif k in ("equal", "plus"):
                _do_zoom(1.25)
            elif k == "minus":
                _do_zoom(1 / 1.25)
            elif k == "period":
                zoom[0] = 1.0
                _clamp_pan()
                _render()
                _update_rect_coords()
                status_var.set(
                    "Zoom 100%  |  Ctrl+drag=pan  +/-=zoom  .=reset  "
                    "Drag=dessiner  Entrée=valider  Échap=annuler  R=reset bbox")

        top.bind("<Key>", _on_key)
        top.protocol("WM_DELETE_WINDOW", _cancel)

        top.wait_window(top)
        return self._result
