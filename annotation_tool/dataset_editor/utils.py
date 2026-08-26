from config import _PALETTE_RGB, HANDLE_RADIUS

import os
import cv2
import numpy as np
import tkinter as tk

from tk_box_selector import TKBoxSelector

def palette_rgb(cls_id: int):
    return _PALETTE_RGB[int(cls_id) % len(_PALETTE_RGB)]


# ──────────────────────────────────────────────────────────
# YOLO <-> pixel conversions
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


def xyxy_to_xywhn(x1, y1, x2, y2, W, H):
    return xyxy_to_yolo(x1, y1, x2, y2, W, H)


# ──────────────────────────────────────────────────────────
# Read / write labels
# ──────────────────────────────────────────────────────────
def read_labels(path: str):
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
    with open(path, "w") as f:
        for lb in labels:
            cls_id = int(lb[0])
            f.write(f"{cls_id} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f} {lb[4]:.6f}\n")


# ──────────────────────────────────────────────────────────
# SAM3 helpers
# ──────────────────────────────────────────────────────────
def mask_to_xyxy(mask: np.ndarray, w: int, h: int):
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


def write_labels_sam(txt_path, nouvelles_lignes, merge_mode, class_ids):
    """Write the SAM labels according to merge_mode."""
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

    else:  # merge
        with open(txt_path, "a", encoding="utf-8") as f:
            for _, ligne in nouvelles_lignes:
                f.write(ligne)


def select_bboxes_for_sam(frame_bgr: np.ndarray,
                           class_ids: list,
                           n_neg: int = 0) -> tuple | None:
    """
    Return (positive_bboxes, negative_bboxes), or None if cancelled.
    Negative bboxes are shown in red in the selector.
    """
    n   = len(class_ids)
    sel = TKBoxSelector()

    # ── Positive bboxes ──
    bboxes_pos = []
    for k in range(n):
        title = f"SAM3 - Objet POSITIF {k+1}/{n}  (classe {class_ids[k]})"
        print(f"   -> Bbox positive pour {title}")
        bbox = sel.select(frame_bgr, title, bboxes_pos, class_ids,
                          box_color="#00ff88", label_prefix="POS")
        if bbox is None:
            print(f"   [!] Annule sur l'objet positif {k+1}.")
            return None
        bboxes_pos.append(bbox)
        print(f"   Positif {k+1} -> {bbox}")

    # ── Negative bboxes ──
    bboxes_neg = []
    for k in range(n_neg):
        title = f"SAM3 - Bbox NEGATIVE {k+1}/{n_neg}  (zone a ignorer)"
        print(f"   -> Bbox negative {k+1}/{n_neg}")
        # Show the already-drawn positives in the background
        bbox = sel.select(frame_bgr, title, bboxes_pos, class_ids,
                          extra_boxes=bboxes_neg,
                          box_color="#ff4444", label_prefix="NEG")
        if bbox is None:
            print(f"   [!] Annule sur la bbox negative {k+1}.")
            return None
        bboxes_neg.append(bbox)
        print(f"   Negatif {k+1} -> {bbox}")

    return bboxes_pos, bboxes_neg


# ──────────────────────────────────────────────────────────
# Ask for number of objects + classes (Tkinter)
# ──────────────────────────────────────────────────────────
def ask_sam_config(default_n=1, default_classes=None) -> tuple | None:
    result = {"ok": False, "n": default_n, "classes": default_classes or [0],
              "n_neg": 0}

    top = tk.Toplevel()
    top.title("Configuration SAM3")
    top.configure(bg="#0d1117")
    top.resizable(False, False)
    top.grab_set()

    w, h = 340, 240  # a bit taller to fit the negative field
    top.update_idletasks()
    sw = top.winfo_screenwidth()
    sh = top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    top.lift()
    top.focus_force()

    style       = dict(bg="#0d1117", fg="#c9d1d9", font=("Courier", 10))
    entry_style = dict(bg="#161b22", fg="#00ff88", insertbackground="#00ff88",
                       font=("Courier", 12, "bold"), justify="center",
                       relief="flat", width=20,
                       highlightthickness=1, highlightcolor="#238636",
                       highlightbackground="#30363d")
    neg_style   = dict(bg="#161b22", fg="#ff6060", insertbackground="#ff6060",
                       font=("Courier", 12, "bold"), justify="center",
                       relief="flat", width=20,
                       highlightthickness=1, highlightcolor="#8b0000",
                       highlightbackground="#30363d")

    tk.Label(top, text="Nombre d'objets a tracker :", **style).pack(pady=(12, 2))
    n_var = tk.StringVar(value=str(default_n))
    tk.Entry(top, textvariable=n_var, **entry_style).pack(pady=2)

    tk.Label(top, text="IDs de classe (ex: 0 1 2) :", **style).pack(pady=(8, 2))
    cls_default = " ".join(str(c) for c in (default_classes or [0]))
    cls_var = tk.StringVar(value=cls_default)
    tk.Entry(top, textvariable=cls_var, **entry_style).pack(pady=2)

    tk.Label(top, text="Nombre de bboxes negatives (0 = aucune) :", **style).pack(pady=(8, 2))
    neg_var = tk.StringVar(value="0")
    tk.Entry(top, textvariable=neg_var, **neg_style).pack(pady=2)

    def validate(event=None):
        try:
            n = int(n_var.get().strip())
            classes = [int(x) for x in cls_var.get().strip().split()]
            while len(classes) < n:
                classes.append(classes[-1] if classes else 0)
            classes = classes[:n]
            n_neg = max(0, int(neg_var.get().strip()))
            result["ok"]      = True
            result["n"]       = n
            result["classes"] = classes
            result["n_neg"]   = n_neg
        except ValueError:
            pass
        top.destroy()

    def cancel(event=None):
        top.destroy()

    top.bind("<Return>",   validate)
    top.bind("<KP_Enter>", validate)
    top.bind("<Escape>",   cancel)

    btn_frame = tk.Frame(top, bg="#0d1117")
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="  Lancer SAM3  ", command=validate,
              bg="#238636", fg="white", font=("Courier", 10, "bold"),
              relief="flat", padx=8, pady=4).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Annuler", command=cancel,
              bg="#4a1010", fg="white", font=("Courier", 9),
              relief="flat", padx=6, pady=4).pack(side="left", padx=6)

    top.wait_window(top)
    if result["ok"]:
        return result["n"], result["classes"], result["n_neg"]
    return None


def get_handle(mx, my, x1, y1, x2, y2, r=HANDLE_RADIUS):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
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
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def ask_class_id(default=0) -> int | None:
    result = {"cls": None}
    top = tk.Toplevel()
    top.title("Nouvelle bbox - Classe ?")
    top.configure(bg="#0d1117")
    top.resizable(False, False)
    top.grab_set()
    w, h = 300, 130
    top.update_idletasks()
    sw = top.winfo_screenwidth(); sh = top.winfo_screenheight()
    top.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    top.lift(); top.focus_force()
    tk.Label(top, text="Entrez l'ID de classe :", bg="#0d1117",
             fg="#c9d1d9", font=("Courier", 11)).pack(pady=(18, 4))
    entry_var = tk.StringVar(value=str(default))
    entry = tk.Entry(top, textvariable=entry_var, bg="#161b22", fg="#00ff88",
                     insertbackground="#00ff88", font=("Courier", 14, "bold"),
                     justify="center", relief="flat", width=10,
                     highlightthickness=1, highlightcolor="#238636",
                     highlightbackground="#30363d")
    entry.pack(pady=4)
    entry.select_range(0, tk.END); entry.focus_set()

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