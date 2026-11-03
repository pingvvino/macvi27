"""Torch-free NCNN inference for the maritime model (SEA-6).

Loads the ultralytics-exported NCNN model directly via the `ncnn` package — no
ultralytics, no torch — and decodes the YOLO11 head. The export already does the
DFL/anchor decode, so `out0` is [4+nc, num_anchors]: rows 0-3 = box xywh (centre,
in letterbox-pixel space), rows 4.. = per-class scores (sigmoid). We threshold,
NMS (class-aware), and map boxes back to original-image coords — same detection
dicts the ultralytics path produced, so the rest of the pipeline is unchanged.
"""
import os
import re

import numpy as np
import ncnn


def _read_metadata_imgsz(model_dir):
    """Parse ultralytics export metadata.yaml for the model's natural input shape.
    Returns (H, W) if found, else (None, None). Tiny flat-yaml parser -- no pyyaml
    dep, matches the bytetrack.yaml approach in watchkeeper._load_tracker."""
    path = os.path.join(model_dir, "metadata.yaml")
    if not os.path.isfile(path):
        return None, None
    in_block = False
    vals = []
    for line in open(path):
        s = line.rstrip()
        if s.startswith("imgsz:"):
            in_block = True
            continue
        if in_block:
            t = s.strip()
            if t.startswith("- "):
                try:
                    vals.append(int(t[2:].strip()))
                except ValueError:
                    return None, None
            else:
                break
    if len(vals) == 2:
        return vals[0], vals[1]                  # ultralytics convention: (H, W)
    return None, None


def _read_metadata_names(model_dir):
    """Parse the exporter's class list from metadata.yaml. Returns {id: name} or {}.

    The band model has its OWN class list (4 classes -- no person/fishing_net, which
    cannot appear in a 192 px strip anyway) and it differs from the full-frame model's.
    Passing the caller's map to a model exported with another one renames every class
    silently: index 0 would read 'person' instead of 'sailboat'. Same flat-yaml parser
    as _read_metadata_imgsz -- no pyyaml dep.
    """
    path = os.path.join(model_dir, "metadata.yaml")
    if not os.path.isfile(path):
        return {}
    in_block = False
    out = {}
    for line in open(path):
        s = line.rstrip()
        if s.startswith("names:"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"\s+(\d+):\s*(\S+)\s*$", s)
            if not m:
                break
            out[int(m.group(1))] = m.group(2)
    return out


def _nms(xyxy, scores, cls_id, iou_thr):
    """Class-aware greedy NMS. Returns kept row indices (highest score first)."""
    if len(xyxy) == 0:
        return []
    # Offset boxes per class so different classes never suppress each other
    # (matches ultralytics agnostic_nms=False).
    off = cls_id.astype(np.float32) * (float(xyxy.max()) + 1.0)
    b = xyxy + off[:, None]
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clip(0); h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_thr]
    return keep


def _box_iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def _box_contain(a, b):
    """Intersection over the SMALLER box's area (1.0 = smaller box fully inside
    the larger). Catches NESTED duplicates that IoU misses when the boxes differ
    a lot in size (big outer box vs small inner box -> low IoU but high IoS)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (smaller + 1e-9)


def _drop_ghost_boxes(dets, iou_thr=0.6, conf_ratio=1.4, contain_thr=0.8):
    """Drop same-class duplicate boxes that class-aware NMS leaves behind, keeping
    the highest-confidence box of each duplicate cluster. 'person' is never
    dropped (MOB safety).

    A weaker same-class box j is dropped vs a stronger box i when either:
      (a) NESTED: j is mostly inside (or contains) i -- intersection-over-smaller
          >= contain_thr. Fires REGARDLESS of the conf gap, because nested boxes
          are one object detected at different extents (e.g. a hull box inside a
          hull+mast box -> low IoU, so the IoU rule alone misses them). Keeps the
          highest-conf box of the nested stack.
      (b) OVERLAP+GAP: high IoU (>= iou_thr) AND a clear conf gap (i >= conf_ratio
          * j) -- a much weaker partially-overlapping ghost.
    Real distinct boats overlap only partially (low IoS) at comparable conf, so
    neither rule merges them; one box ~fully inside another is a duplicate.
    """
    order = sorted(range(len(dets)), key=lambda i: dets[i]["confidence"], reverse=True)
    drop = set()
    for ai in range(len(order)):
        i = order[ai]
        if i in drop:
            continue
        for bj in range(ai + 1, len(order)):
            j = order[bj]
            if j in drop:
                continue
            if dets[i]["class"] != dets[j]["class"] or dets[i]["class"] == "person":
                continue
            a_box, b_box = dets[i]["bbox"], dets[j]["bbox"]
            nested = _box_contain(a_box, b_box) >= contain_thr
            overlap_gap = (_box_iou(a_box, b_box) >= iou_thr
                           and dets[i]["confidence"] >= conf_ratio * dets[j]["confidence"])
            if nested or overlap_gap:
                drop.add(j)
    if not drop:
        return dets
    return [d for k, d in enumerate(dets) if k not in drop]


class NcnnDetector:
    """Drop-in YOLO replacement for the maritime NCNN model, ultralytics-free."""

    def __init__(self, model_dir, classes, imgsz=800, num_threads=2):
        self.classes = classes          # {cls_id: name}
        # Resolve (W, H). metadata.yaml is the ground truth -- the exporter bakes a
        # fixed-shape head reshape at imgsz, so a rect-exported model (e.g. v8 800x448)
        # REQUIRES the rect input. Falls back to a square `imgsz` arg for legacy v5/v7.
        h_meta, w_meta = _read_metadata_imgsz(model_dir)
        if w_meta is not None and h_meta is not None:
            self.W, self.H = int(w_meta), int(h_meta)
        elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            self.H, self.W = int(imgsz[0]), int(imgsz[1])  # ultralytics convention: (H, W)
        else:
            self.W = self.H = int(imgsz)
        self.imgsz = self.W                       # back-compat for callers reading .imgsz
        self.net = ncnn.Net()
        self.net.opt.num_threads = int(num_threads)
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(f"{model_dir}/model.ncnn.param")
        self.net.load_model(f"{model_dir}/model.ncnn.bin")

    def _letterbox(self, img):
        import cv2
        h0, w0 = img.shape[:2]
        r = min(self.W / w0, self.H / h0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        canvas = np.full((self.H, self.W, 3), 114, np.uint8)
        top, left = (self.H - nh) // 2, (self.W - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, r, left, top

    def detect(self, img, conf=0.25, iou=0.7, ramka=None):
        """img: BGR ndarray. Returns list of {class, class_id, confidence, bbox(x1,y1,x2,y2)}.

        `ramka=(r, padx, pady)` — uzyj TEGO odwzorowania zamiast liczyc wlasne, a `img`
        potraktuj jako gotowe pole wejsciowe modelu. Sluzy wylacznie zdalnemu wezlowi
        (`wezel_pelnej_klatki.py`): Pi robi letterbox u siebie i wysyla juz przeskalowane
        800x448, bo surowy kadr 2560x1440 to 11 MB zamiast 1,05 MB.

        Dlaczego nie wystarczy policzyc letterboxa po stronie wezla: wspolrzedne sa
        ZAOKRAGLANE dopiero po przeliczeniu na kadr oryginalny (`round((x - padx) / r)`).
        Gdyby wezel zaokraglil we wlasnej przestrzeni, a Pi dopiero potem przeskalowalo,
        wynik rozniłby sie od lokalnego o ulamki piksela — a caly sens tego wezla jest
        taki, ze oddaje DOKLADNIE te liczbe, ktora policzyloby Pi.
        """
        if ramka is None:
            canvas, r, padx, pady = self._letterbox(img)
        else:
            canvas, (r, padx, pady) = img, ramka
            if canvas.shape[0] != self.H or canvas.shape[1] != self.W:
                raise ValueError(f"z `ramka` oczekuje pola {self.W}x{self.H}, "
                                 f"dostalem {canvas.shape[1]}x{canvas.shape[0]}")
        mat = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                                   self.W, self.H)
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, out = ex.extract("out0")
        if ret != 0:                            # failed forward: drop frame, never np.array a bad Mat (segfault)
            return []
        return decode_output(np.array(out), self.classes, conf, iou, r, padx, pady)   # [4+nc, N]


def decode_output(a, classes, conf, iou, r, padx, pady):
    """Raw head output [4+nc, N] (xywh in letterbox px, per-class sigmoid) -> detections in the
    ORIGINAL frame: threshold, class-aware NMS, un-letterbox, rounding, ghost-box drop.
    Shared by NcnnDetector.detect (the boat) and the .pt epoch ranking on Kaggle
    (models/ocena_toru_lodzi.py detekcje-pt), so both decode a network output identically."""
    if a.ndim != 2 or a.size == 0:
        return []
    boxes = a[:4].T                          # [N,4] xywh (letterbox px)
    scores = a[4:].T                         # [N,nc] sigmoid
    cls_id = scores.argmax(1)
    cf = scores.max(1)
    keep = cf >= conf
    boxes, cls_id, cf = boxes[keep], cls_id[keep], cf[keep]
    if len(boxes) == 0:
        return []
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    dets = []
    for i in _nms(xyxy, cf, cls_id, iou):
        c = int(cls_id[i])
        if c not in classes:
            continue
        x1 = (xyxy[i, 0] - padx) / r
        y1 = (xyxy[i, 1] - pady) / r
        x2 = (xyxy[i, 2] - padx) / r
        y2 = (xyxy[i, 3] - pady) / r
        dets.append({
            "class": classes[c],
            "class_id": c,
            "confidence": round(float(cf[i]), 3),
            "bbox": (round(x1), round(y1), round(x2), round(y2)),
        })
    return _drop_ghost_boxes(dets)
