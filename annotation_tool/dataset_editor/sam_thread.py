import threading
import queue
import cv2
import os
import gc
import torch

from utils import mask_to_xyxy, xyxy_to_xywhn, write_labels_sam




# ──────────────────────────────────────────────────────────
# SAM3 thread
# ──────────────────────────────────────────────────────────
class SAMThread(threading.Thread):
    """
    Runs SAM3VideoPredictor in a separate thread.
    result_queue receives dicts {"type": ..., ...}:
      {"type": "frame", "global_idx": int, "img_bgr": array,
       "detected": [...], "lost": [...], "img_path": str,
       "label_path": str, "live_labels": [...]}
      {"type": "done",  "last_frame": int, "stopped": bool, "reason": str}
      {"type": "error", "msg": str}

    Stops automatically if lost[i] >= lost_threshold for any object.
    """

    def __init__(self, video_path, start_frame, fps, width, height,
                 total_frames, bboxes, class_ids, conf,
                 imgs_dir, labels_dir, tmp_dir, merge_mode,
                 lost_threshold=15, neg_bboxes=None, model="sam3.pt"):
        super().__init__(daemon=True)
        self.video_path     = video_path
        self.start_frame    = start_frame
        self.fps            = fps
        self.width          = width
        self.height         = height
        self.total_frames   = total_frames
        self.bboxes         = bboxes
        self.class_ids      = class_ids
        self.conf           = conf
        self.imgs_dir       = imgs_dir
        self.labels_dir     = labels_dir
        self.tmp_dir        = tmp_dir
        self.merge_mode     = merge_mode
        self.lost_threshold = lost_threshold
        self.neg_bboxes = neg_bboxes or []
        self.model      = model

        self._stop_event  = threading.Event()
        self.result_queue = queue.Queue()

    def stop(self):
        self._stop_event.set()

    def _extract_subclip(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        path = os.path.join(self.tmp_dir,
                            f"_sam_sub_{self.start_frame:06d}.mp4")
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        wr = cv2.VideoWriter(path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (self.width, self.height))
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            wr.write(f)
            n += 1
        cap.release()
        wr.release()
        print(f"  [SAM] Sous-clip : {n} frames -> {os.path.basename(path)}")
        return path

    def run(self):
        print(f"  [DEBUG RUN] run() appelé — fichier : {__file__}")
        subclip    = None
        stop_reason = "termine"
        try:
            from ultralytics.models.sam import SAM3VideoPredictor

            subclip = self._extract_subclip()
            n = len(self.bboxes)

            cap_src = cv2.VideoCapture(self.video_path)
            cap_src.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            overrides = dict(conf=self.conf, task="segment", mode="predict",
                             imgsz=644, model=self.model, half=True, save=False)
            predictor = SAM3VideoPredictor(overrides=overrides)
            all_bboxes = self.bboxes + self.neg_bboxes
            all_labels = [1] * len(self.bboxes) + [0] * len(self.neg_bboxes)

            results = predictor(
                source=subclip,
                bboxes=all_bboxes,
                labels=all_labels,
                stream=True,
            )

            lost       = [0] * n
            last_local = 0

            for local_idx, r in enumerate(results):
                if self._stop_event.is_set():
                    stop_reason = "stoppe manuellement"
                    try:
                        results.close()
                    except Exception:
                        pass
                    break

                global_idx = self.start_frame + local_idx

                ok, frame_raw = cap_src.read()
                if not ok or frame_raw is None:
                    frame_raw = r.orig_img

                h_raw, w_raw = frame_raw.shape[:2]

                detected = [False] * n
                if r.boxes is not None and len(r.boxes) > 0:
                    for cls_val in r.boxes.cls.cpu().numpy():
                        obj_i = int(cls_val)
                        if 0 <= obj_i < n:
                            detected[obj_i] = True

                for i in range(n):
                    if detected[i]:
                        lost[i] = 0
                    else:
                        lost[i] += 1

                live_labels      = []
                nouvelles_lignes = []
                img_path         = None
                label_path       = None

                src_masks = (r.masks.data.cpu().numpy()
                             if r.masks is not None else None)

                if r.boxes is not None and len(r.boxes) > 0:
                    cls_np   = r.boxes.cls.cpu().numpy()
                    xywhn_np = r.boxes.xywhn.cpu().numpy()

                    for di in range(len(cls_np)):
                        obj_i = int(cls_np[di])
                        if obj_i >= n:
                            continue

                        if src_masks is not None:
                            box = mask_to_xyxy(src_masks[di], w_raw, h_raw)
                            if box is None:
                                continue
                            xc, yc, bw, bh = xyxy_to_xywhn(*box, w_raw, h_raw)
                        else:
                            xc, yc, bw, bh = xywhn_np[di]

                        real_cls = self.class_ids[obj_i]
                        live_labels.append([real_cls, xc, yc, bw, bh])
                        ligne = (f"{real_cls} "
                                 f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                        nouvelles_lignes.append((real_cls, ligne))

                if nouvelles_lignes:
                    img_path   = os.path.join(self.imgs_dir,
                                              f"frame_{global_idx:06d}.jpg")
                    label_path = os.path.join(self.labels_dir,
                                              f"frame_{global_idx:06d}.txt")
                    # raw frame, max quality
                    cv2.imwrite(img_path, frame_raw)
                    write_labels_sam(label_path, nouvelles_lignes,
                                     self.merge_mode, self.class_ids)

                # ── cap_src.release() removed from here ──

                self.result_queue.put({
                    "type":        "frame",
                    "global_idx":  global_idx,
                    "img_bgr":     frame_raw,
                    "detected":    detected,
                    "lost":        lost[:],
                    "img_path":    img_path,
                    "label_path":  label_path,
                    "live_labels": live_labels,
                })

                last_local = local_idx

                lost_objects = [i for i in range(n)
                                if lost[i] >= self.lost_threshold]
                if lost_objects:
                    names = ", ".join(
                        f"obj{i+1}(cls{self.class_ids[i]})"
                        for i in lost_objects
                    )
                    print(f"  [SAM] Arret auto — objets perdus : {names} "
                          f"a la frame {global_idx} "
                          f"(seuil={self.lost_threshold})")
                    stop_reason = (
                        f"arret auto — {names} perdus "
                        f">{self.lost_threshold} frames"
                    )
                    try:
                        results.close()
                    except Exception:
                        pass
                    break

                if global_idx >= self.total_frames - 1:
                    break

            cap_src.release()  # <- released ONCE, after the loop

            last_global = self.start_frame + last_local
            stopped     = self._stop_event.is_set() or (stop_reason != "termine")

            try:
                del predictor
                del results
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

            self.result_queue.put({
                "type":       "done",
                "last_frame": last_global,
                "stopped":    stopped,
                "reason":     stop_reason,
            })

        except Exception as e:
            import traceback
            self.result_queue.put({
                "type": "error",
                "msg":  traceback.format_exc(),
            })
        finally:
            if subclip and os.path.exists(subclip):
                try:
                    os.remove(subclip)
                except OSError:
                    pass