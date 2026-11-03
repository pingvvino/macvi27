#!/usr/bin/env python3
"""CPU visual odometry for the MaCVi LOOKOUT challenge -- predictions in the starter's format.

Challenge: https://macvi.org/workshop/macvi27/challenges/lookout . Starter code (evaluator,
ClipPreprocessor, data loaders) comes with the dataset in `release/starter/`; this script only
adds the estimator on top of it and writes `<output>/predictions/<clip>.json`.

Methods (--method):
  line     straight line, constant speed. No vision at all -- the floor every method must beat.
  horizon  course from vision, CONSTANT speed. Rotation is fitted only to points on and above
           the horizon (shoreline, clouds -- effectively at infinity), because water texture
           moves on its own and breaks rigid-motion estimation.
  flow     course as in `horizon` + SPEED from water flow below the horizon. A water point
           at angle v below the horizon lies at Z = h/v (h = camera height); after removing
           rotation, forward motion dz moves it down by dv = v^2 * dz/h. So dz/h =
           median(dv / v^2). h is unknown but constant per clip -- the global Sim(3) fit of
           the metric absorbs it, so we never need it.
  auto     `flow` on cameras where water texture carries the speed, `horizon` elsewhere
           (FLOW_MODES; chosen on the TRAIN split -- see the table there).
  adaptive water speed trusted per clip by how much it really varies (ADAPT_*); best so far.

The horizon comes from camera/postprocess.estimate_horizon (validated on our own footage).

--wasr-dir <dir>  per-clip <clip>.npz with WaSR water probabilities (frames, prob, W, H) at ~1 Hz;
           the course masks then follow the WATER EDGE column by column (interpolated in time)
           instead of the fitted horizon line. Speed is unchanged.

--yolo-dir <NCNN band model>  (e.g. models/maritime-yolo11s-v11g-best-ncnn-608x192)
           cuts detected vessels (sailboat/motorboat/ship) out of BOTH measurements: they sit
           right at the horizon, where the rotation points are, and move on their own; their
           hulls and wakes also pollute the water flow. Buoys stay -- they are fixed points.
           Detection runs on 608x192 tiles along the horizon in the FULL-resolution frame
           (the band model's native scale); the VO frame is the same half-resolution image
           the plain path uses, so results are directly comparable.

    python tools/lookout_vo.py --dataset release/release_manifest.json --split val \\
        --method flow --output runs/flow
    python release/starter/evaluate.py --dataset release/release_manifest.json --split val \\
        --predictions runs/flow

A clip whose processing raises gets the `line` prediction instead (still scored); the run ends
with a SUMMARY line and exits non-zero only if some clip has no prediction at all.
Environment: tools/lookout_vo_requirements.txt.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'camera'))
from postprocess import estimate_horizon_line  # noqa: E402

# Pixel thresholds below are written for the half-resolution VO frame (the starter's default);
# --full-res multiplies them by PX = 2 so they keep the same angle, and adds one pyramid level.
PX = 1.0
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
WATER_EDGE_MARGIN = 6       # px above the WaSR water edge where course points may start
HORIZON_MIN_CONF = 0.3      # below this estimate_horizon says "no trustworthy line" -> no mask
# The horizon is a TILTED line (boat roll, camera mount): estimate_horizon_line gives its slope.
# For the cameras in TILT_MODES (method adaptive) the masks above the horizon follow the line and
# the water flow is measured in the frame rotated level about the horizon, so a row again means
# one angle below the horizon; a level row is wrong by slope*W/2 at the frame edge (~33 px at
# 4 deg on 960 px), right where dv = s v^2 is most sensitive. Slope from confident lines only,
# median over HORIZON_SMOOTH_N. Chosen on TRAIN per camera (68 clips, median / mean drift %):
# wide 9.08/11.78 -> 8.44/10.83 (better 27, worse 18); flir-VIS 4.95/8.09 -> 7.43/10.40 and
# sionyx 5.73/7.33 -> 8.19/8.69 -- there the line's slope follows tree lines and shores, not
# the water. All train 8.68/10.98 -> 8.44/10.34; val checked once 9.93/12.09 -> 8.28/10.70
# (the 21 val clips never looked at during development: 11.04/12.30 -> 8.80/10.78).
TILT_MODES = {'wide'}
MAX_DYAW = np.deg2rad(15)   # per step at ~8 Hz; anything larger is a bad fit, not a turn
# Test server eligibility (RULES.md, 2026-10-10): a reference time counts as covered only between predicted
# samples at most 1 s apart, and every clip needs >= 95% coverage. Flagged per clip in the SUMMARY as gaps=.
SERVER_MAX_GAP_S = 1.0
# water band used for speed, as angle below the horizon (rad): rows right under the horizon
# amplify flow noise by 1/v^2, rows far below are near-field spray/bow
WATER_V_MIN, WATER_V_MAX = 0.04, 0.35
BOTTOM_CUT = 0.97           # the camera can be pitched up (horizon at 0.85 H): keep the band
HORIZON_SMOOTH_N = 15       # frames (~2 s) of rolling median for the speed-step horizon
SPEED_WINDOW_S = 6.0        # median window for speed; a boat does not change speed faster
# Cameras whose water texture gives a usable speed. Chosen on TRAIN (68 clips), median drift
# constant-speed vs water-speed: wide 12.2/12.1 (n=42), zoom+infrared 38.7-42.8/16.9 (n=3),
# but sionyx 7.9/33.1 (n=6), flir-IR 8.1/33.4 (n=4), flir-VIS 21.8/31.9 (n=6) -- night and
# thermal water has no trackable texture, a constant speed is far better there.
FLOW_MODES = {'wide', 'zoom', 'infrared'}
# Cameras whose course comes from the ESSENTIAL MATRIX over EMAT_N-frame chunks instead of
# the pure-rotation fit. Rotation-only treats far points as at infinity; a city a few km to
# one side moves in the frame with the boat's FORWARD motion (~0.14 deg/s at 10 m/s, 2 km),
# which it reads as a steady turn -- clip_090: +108 deg heading error after 290 s, with the
# bow rail fixed in frame (not a pan head). Chosen on TRAIN per camera, median drift %
# rotation-only -> E16: flir-VIS 19.6 -> 10.6 (better on 7/8 clips; 090 38.4 -> 6.1), but
# wide 11.6 -> 17.9, sionyx 9.3 -> 17.5 -- the short-baseline E is noisier there. No
# flir-VIS clip in val, so this rule is checked on train only.
EMAT_MODES = {'flir-VIS'}
EMAT_N = 16
# Every OTHER camera (method adaptive): course from an INVERSE-DEPTH windowed bundle
# adjustment (Civera, Davison, Montiel, T-RO 2008) over BA_N-frame chunks of KLT tracks above
# the horizon. Rotation-only is the rho -> 0 limit of the same model; BA lets each track have a
# finite depth, so shore parallax from the boat's own forward motion stops reading as a turn.
# Chosen on TRAIN (68 clips), median/mean drift % rotation-only -> BA: wide 11.6/14.5 ->
# 9.3/12.4 (better on 35/46), sionyx 9.3/7.9 -> 5.5/7.3, flir-IR 11.1/10.5 -> 8.8/9.2, zoom
# 18.8 -> 18.4, infrared 16.8 -> 15.2; flir-VIS 19.6 -> 17.6 but E16 is 10.6 there, so E16
# stays. All train 11.1/13.5 -> 9.4/11.9; val checked once 10.9/13.4 -> 9.9/12.3.
BA_N = 16
# Course multiplier per camera. The release calibration is 'nominal-fov-derived', not measured;
# a focal length that is too short inflates every measured turn. Chosen on TRAIN (68 clips,
# grid 0.3-1.5, by mean drift; only cameras with >= 4 train clips): flir-VIS 0.65 (all 8
# clips want <= 0.7; 10.6/12.5 -> 5.0/8.1), wide 0.9 (9.3/12.4 -> 9.1/11.8); sionyx/flir-IR
# best ~1.05, gain < 0.2, left at 1. Independent video-only check: the BA cost vs focal
# length bottoms out at >= 1.5x nominal fx on 5 of 6 flir-VIS clips. All train 9.4/11.9 ->
# 8.7/11.0; val checked once 9.9/12.3 -> 9.9/12.1 (no flir-VIS clip in val).
YAW_MULT = {'flir-VIS': 0.65, 'wide': 0.9}
BA_NFEV = 100               # course settles in ~30-100 evaluations (<=0.1 deg change after)
BA_RHO_MAX = 0.5            # inverse depth in 1/(camera heights): nearest point 2 h away
BA_AZ_MAX, BA_EL_MAX = 0.6, 0.3   # travel direction vs camera axis (rad); +-1.5 changed nothing
BA_MIN_OBS = 4              # frames a track must be seen in within a chunk
BA_MAX_TRACKS = 200
BA_MIN_INLIERS = 40         # reprojection < 2 px; fewer -> chunk keeps rotation-only
TRACK_MAX, TRACK_MIN = 300, 200   # KLT points kept alive above the horizon; refill below MIN
YOLO_MOVING = {'sailboat', 'motorboat', 'ship'}   # buoy is NOT here: a fixed point is wanted
YOLO_MARGIN = 0.3           # box grown by this fraction of its size on every side


class BoatMasker:
    """Vessel boxes from the NCNN band model on tiles along the horizon, every N-th frame."""

    def __init__(self, model_dir, every=4, conf=0.25, threads=2):
        from ncnn_infer import NcnnDetector, _read_metadata_names
        from byte_track import ByteTracker
        names = _read_metadata_names(model_dir)
        if not names:
            raise ValueError(f'{model_dir}: no class names in metadata.yaml')
        self.det = NcnnDetector(model_dir, names, num_threads=threads)
        self.class_ids = {v: k for k, v in names.items()}
        # Tracks link one object across detection frames -- triangulation needs the SAME
        # object seen from several boat positions. GMC compensates the boat's own yaw.
        self.tracker = ByteTracker(cmc=True)
        self.every, self.conf = max(1, int(every)), conf
        self.n, self.boxes = 0, []
        self.frames_with_boats = 0
        self.obs = []       # (step, track_id, class_id, cx, y_bottom) full-res px, every class

    def update(self, full, hy_full, step):
        """Vessel boxes (x1, y1, x2, y2) in full-res pixels; recomputed every `every` frames.
        Also records tracked observations of ALL classes (buoys too) for triangulation."""
        if self.n % self.every == 0:
            self.boxes = []
            dets = []
            if hy_full is not None:
                fh, fw = full.shape[:2]
                tw, th = self.det.W, self.det.H
                y0 = int(np.clip(hy_full - th / 2, 0, max(0, fh - th)))
                n = max(2, int(np.ceil((fw - tw) / (0.8 * tw))) + 1) if fw > tw else 1
                for x0 in np.linspace(0, max(0, fw - tw), n).astype(int):
                    for d in self.det.detect(full[y0:y0 + th, x0:x0 + tw], conf=self.conf):
                        x1, y1, x2, y2 = d['bbox']
                        d = dict(d, bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0))
                        dets.append(d)
                        if d['class'] in YOLO_MOVING:
                            self.boxes.append(d['bbox'])
            for t in self.tracker.update(dets, frame=full, horizon_y=hy_full):
                x1, y1, x2, y2 = t['bbox']
                self.obs.append((step, t['track_id'], self.class_ids.get(t['class'], -1),
                                 (x1 + x2) / 2.0, float(y2)))
        self.n += 1
        self.frames_with_boats += bool(self.boxes)
        return self.boxes

    @staticmethod
    def mask(shape, boxes, scale):
        """Boolean `shape` mask, True on (grown) vessel boxes scaled into that image."""
        m = np.zeros(shape, bool)
        for x1, y1, x2, y2 in boxes:
            gx, gy = YOLO_MARGIN * (x2 - x1), YOLO_MARGIN * (y2 - y1)
            a = int(max(0, (x1 - gx) * scale)); b = int(min(shape[1], (x2 + gx) * scale) + 1)
            c = int(max(0, (y1 - gy) * scale)); d = int(min(shape[0], (y2 + gy) * scale) + 1)
            m[c:d, a:b] = True
        return m


def track_points(prev, cur, mask, max_corners, fb_px):
    """LK with forward-backward check. Returns matched (a, b) pixel arrays."""
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners, qualityLevel=0.005,
                                 minDistance=6 * PX, mask=mask)
    if p0 is None or len(p0) < 8:
        return None, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **LK)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **LK)
    ok = ((st.ravel() == 1) & (st2.ravel() == 1)
          & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < fb_px))
    return p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)


def bearings(p, Ki):
    v = (Ki @ np.c_[p, np.ones(len(p))].T).T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def above_line_mask(shape, y_c, slope, margin=None):
    """uint8 mask, 255 above the line y = y_c + slope*(x - W/2) + margin (default 6 px x PX;
    at least 8 rows)."""
    margin = 6 * PX if margin is None else margin
    h, w = shape[:2]
    lim = np.maximum(8, y_c + slope * (np.arange(w) - w / 2.0) + margin)
    return np.where(np.arange(h)[:, None] < lim[None, :], 255, 0).astype(np.uint8)


def water_edge(prob, out_w, out_h):
    """Per-column row (VO px, length out_w) where water starts, from a water probability map
    (rows x cols, uint8 0..255): the split minimising water above + non-water below, median-
    smoothed over 9 columns. A column with no water gives out_h (no edge in the frame)."""
    P = prob.astype(np.float64) / 255.0
    nr, nc = P.shape
    c = np.vstack([np.zeros((1, nc)), np.cumsum(P, 0)])
    cost = c + ((nr - np.arange(nr + 1))[:, None] - (c[-1] - c))
    e = np.argmin(cost, 0).astype(np.float64) * out_h / nr
    e = np.array([np.median(e[max(0, i - 4):i + 5]) for i in range(nc)])
    return np.interp(np.arange(out_w), (np.arange(nc) + 0.5) * out_w / nc, e)


class WaterEdges:
    """Water edge per VO frame, linearly interpolated between the WaSR samples."""

    def __init__(self, path):
        d = np.load(path)
        W, H = int(d['W']), int(d['H'])
        self.frames = np.asarray(d['frames'], float)
        self.edges = np.array([water_edge(p, W, H) for p in d['prob']])

    def at(self, f):
        i = int(np.searchsorted(self.frames, f))
        if i <= 0:
            return self.edges[0]
        if i >= len(self.frames):
            return self.edges[-1]
        a = (f - self.frames[i - 1]) / (self.frames[i] - self.frames[i - 1])
        return (1 - a) * self.edges[i - 1] + a * self.edges[i]


def edge_mask(shape, edge, margin=None):
    """uint8 mask, 255 above the per-column water edge minus `margin` (default WATER_EDGE_MARGIN,
    read at call time; at least 8 rows)."""
    lim = np.maximum(8, np.asarray(edge, float) - (WATER_EDGE_MARGIN if margin is None else margin))
    return np.where(np.arange(shape[0])[:, None] < lim[None, :], 255, 0).astype(np.uint8)


def kabsch(X, Y):
    """Rotation R with Y ~= X @ R.T."""
    U, _, Vt = np.linalg.svd(Y.T @ X)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ S @ Vt


def rotation_far(prev, cur, K, mask, rng):
    """Pure rotation (cur <- prev) from far points: RANSAC Kabsch on bearing vectors."""
    a, b = track_points(prev, cur, mask, 600, 0.7 * PX)
    if a is None or len(a) < 8:
        return None
    Ki = np.linalg.inv(K)
    A, B = bearings(a, Ki), bearings(b, Ki)
    thr = 1.5 * PX / K[0, 0]
    best, best_n = None, 0
    for _ in range(100):
        i = rng.choice(len(A), 2, replace=False)
        R = kabsch(A[i], B[i])
        inl = np.linalg.norm(B - A @ R.T, axis=1) < thr
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best_n < 6:
        return None
    return kabsch(A[best], B[best])


_DIS_P = None
PROFILE_TOP, PROFILE_BOT = 0.05, 0.97   # rows covered by the water-flow profile (frac of H)


def water_profile(prev, cur, K, R, boat=None, roll=0.0, pivot_y=None):
    """Median derotated vertical flow per image row (normalised by fy), NaN where a row has
    too little texture. Independent of the horizon ROW, so any horizon rule can be applied
    offline: speed = LS fit of dv = s v^2 over the rows below that horizon. `roll` = horizon
    slope (dy/dx): both frames are first rotated level about (W/2, pivot_y), so rows are
    parallel to the horizon and the row of the horizon stays pivot_y."""
    global _DIS_P
    h = prev.shape[0]
    H = K @ R @ np.linalg.inv(K)
    prev_w = cv2.warpPerspective(prev, H, (prev.shape[1], h))
    valid = None
    if roll and pivot_y is not None:
        A = cv2.getRotationMatrix2D((prev.shape[1] / 2.0, float(pivot_y)), float(np.degrees(np.arctan(roll))), 1.0)
        size = (prev.shape[1], h)
        prev_w = cv2.warpAffine(prev_w, A, size)
        cur = cv2.warpAffine(cur, A, size)
        valid = cv2.warpAffine(np.full(cur.shape[:2], 255, np.uint8), A, size, flags=cv2.INTER_NEAREST) > 0
        if boat is not None:
            boat = cv2.warpAffine(boat.astype(np.uint8), A, size, flags=cv2.INTER_NEAREST) > 0
    y0, y1 = int(PROFILE_TOP * h), int(PROFILE_BOT * h)
    if _DIS_P is None:
        _DIS_P = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = _DIS_P.calc(prev_w[y0:y1], cur[y0:y1], None)[:, :, 1] / K[1, 1]
    g = (np.abs(cv2.Sobel(prev_w[y0:y1], cv2.CV_32F, 0, 1, ksize=3)) +
         np.abs(cv2.Sobel(prev_w[y0:y1], cv2.CV_32F, 1, 0, ksize=3)))
    tex = (g > 12 / PX) & (prev_w[y0:y1] > 0)
    if valid is not None:
        tex &= cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8))[y0:y1] > 0
    if boat is not None:
        tex &= ~boat[y0:y1]                    # hulls and wakes move on their own
    out = np.full(h, np.nan, np.float32)
    for r in range(fl.shape[0]):
        m = tex[r]
        if m.sum() >= 10:
            out[y0 + r] = np.median(fl[r, m])
    return out


def speed_from_profile(profile, hy, fy):
    """dz/h for this step: least squares of dv = s v^2 over the water rows below `hy`.
    Chosen on TRAIN over the earlier per-pixel median of dv/v^2 (train 11.9/15.1 ->
    11.8/14.6, val 12.0/13.7 -> 10.9/13.4, median/mean drift %, method adaptive)."""
    h = len(profile)
    rows = np.arange(h)
    v = (rows - hy) / fy
    m = np.isfinite(profile) & (v >= WATER_V_MIN) & (v <= WATER_V_MAX) & (rows < BOTTOM_CUT * h)
    if m.sum() < 10:
        return None
    return float(np.sum(profile[m] * v[m] ** 2) / np.sum(v[m] ** 4))


def profile_fit_quality(profile, hy, fy):
    """Relative RMS residual of the step's profile around its own fit P = s v^2 (NaN if no fit):
    how well this step obeys the water law. Same band as speed_from_profile."""
    h = len(profile)
    rows = np.arange(h)
    v = (rows - hy) / fy
    m = np.isfinite(profile) & (v >= WATER_V_MIN) & (v <= WATER_V_MAX) & (rows < BOTTOM_CUT * h)
    if m.sum() < 10:
        return np.nan
    P = np.asarray(profile, float)[m]
    s = float(np.sum(P * v[m] ** 2) / np.sum(v[m] ** 4))
    if s <= 0:
        return np.nan
    return float(np.sqrt(np.mean((P - s * v[m] ** 2) ** 2)) / (s * np.mean(v[m] ** 2)))


# Horizon of the WATER PLANE from the flow itself (method adaptive). After removing rotation the
# vertical flow at row r is P(r) = s ((r - h)/f)^2, so sqrt(P) is LINEAR in r and crosses zero at
# the water's own horizon h -- independent of tree lines, bridges, rails or a horizon out of frame,
# which is where the line detector goes wrong. Per WATER_HZ_WIN-step window: median profile,
# Huber line fit of sqrt(P) over the water band. It replaces the detector's row only where the fit
# is reliable (R^2 >= WATER_HZ_R2, slope >= WATER_HZ_SLOPE x the clip's median slope -- i.e. enough
# flow) and disagrees by > WATER_HZ_GAP px. Chosen on TRAIN (grid R^2 0/0.8/0.9/0.95, slope 0/0.5/1,
# gap 10/20/40): 8.44/10.34 -> 6.88/9.94 (better 13, worse 8); val checked once 8.28/10.70 ->
# 8.28/10.48 (21 untouched val clips 10.78 -> 10.53). Without the gates the median improved but the
# mean got worse: at a near standstill the flow is noise and the fit is not to be trusted.
# The course BA keeps the speed from the detector's row (as measured); only the trajectory's
# speed uses the corrected row.
WATER_HZ_WIN = 80
WATER_HZ_R2 = 0.95
WATER_HZ_SLOPE = 1.0
WATER_HZ_GAP = 40.0
WATER_HZ_V = (0.06, 0.35)


def water_plane_horizon(profiles, hy_frames, fy):
    """Per-step horizon row for the speed step: the detector's row (hy_frames[i+1]), replaced by
    the water-plane horizon where that fit is reliable. profiles: (steps, H) rows of
    water_profile; hy_frames: (steps+1,) detector rows per frame. Returns (rows, replaced mask)."""
    P_all = np.asarray(profiles, float)
    hA = np.asarray(hy_frames, float)
    n, H = P_all.shape
    rows = np.arange(H)
    hw = np.full(n, np.nan); r2 = np.full(n, np.nan); slope = np.full(n, np.nan)
    for w0 in range(0, n, WATER_HZ_WIN):
        sl = slice(w0, min(n, w0 + WATER_HZ_WIN))
        if not np.isfinite(P_all[sl]).any():
            continue
        P = np.nanmedian(P_all[sl], axis=0)
        h0 = np.nanmedian(hA[1:][sl])
        if not np.isfinite(h0):
            continue
        v = (rows - h0) / fy
        m = np.isfinite(P) & (P > 0) & (v >= WATER_HZ_V[0]) & (v <= WATER_HZ_V[1]) & (rows < BOTTOM_CUT * H)
        if m.sum() < 20:
            continue
        x = rows[m].astype(float); y = np.sqrt(P[m]); w = np.ones(len(x))
        for _ in range(10):                                   # IRLS, Huber at 1.5 robust sigma
            A = np.c_[x, np.ones(len(x))] * np.sqrt(w)[:, None]
            a, b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
            r = np.abs(y - a * x - b)
            sc = 1.4826 * np.median(r) + 1e-12
            w = np.where(r <= 1.5 * sc, 1.0, 1.5 * sc / np.maximum(r, 1e-12))
        if a <= 0:
            continue
        res = y - a * x - b
        hw[sl] = -b / a
        r2[sl] = 1 - np.sum(w * res ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
        slope[sl] = a
    if not np.isfinite(slope).any():
        return hA[1:].copy(), np.zeros(n, bool)
    ok = (np.isfinite(hw) & (r2 >= WATER_HZ_R2) & (slope >= WATER_HZ_SLOPE * np.nanmedian(slope))
          & (np.abs(hw - hA[1:]) > WATER_HZ_GAP * PX))
    return np.where(ok, hw, hA[1:]), ok


def chunk_yaw_essential(frames, K, hy, conf, slope=0.0, edge=None):
    """Yaw between the first and last of `frames` from the essential matrix of points above
    the horizon, LK-chained through every frame (forward-backward checked). NaN on failure."""
    g0 = frames[0]
    mask = None
    if edge is not None:
        mask = edge_mask(g0.shape, edge)
    elif conf >= HORIZON_MIN_CONF:
        mask = above_line_mask(g0.shape, hy, slope)
    p = cv2.goodFeaturesToTrack(g0, maxCorners=800, qualityLevel=0.005, minDistance=6 * PX, mask=mask)
    if p is None or len(p) < 20:
        return np.nan
    p0, alive = p.copy(), np.ones(len(p), bool)
    for a, b in zip(frames[:-1], frames[1:]):
        q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **LK)
        qb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **LK)
        alive &= ((st.ravel() == 1) & (st2.ravel() == 1)
                  & (np.linalg.norm((qb - p).reshape(-1, 2), axis=1) < 0.7 * PX))
        p = q
    a, b = p0[alive].reshape(-1, 2), p[alive].reshape(-1, 2)
    if len(a) < 20:
        return np.nan
    cv2.setRNGSeed(0)        # RANSAC draws from OpenCV's global RNG: same chunk -> same answer
    E, m = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=0.7 * PX)
    if E is None or E.shape != (3, 3):
        return np.nan
    n, R, _t, _ = cv2.recoverPose(E, a, b, K, mask=m)
    return float(np.arctan2(R[0, 2], R[2, 2])) if n >= 15 else np.nan


class HorizonTracks:
    """Long KLT tracks of corners ABOVE the horizon, one row (frame, id, x, y) per sighting."""

    def __init__(self):
        self.pts = np.empty((0, 2), np.float32)
        self.ids = np.empty(0, int)
        self.nid, self.rows, self.prev, self.f = 0, [], None, 0

    def update(self, gray, hy_raw, conf, slope=0.0, edge=None):
        if self.prev is not None and len(self.pts):
            p0 = self.pts.reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **LK)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **LK)
            ok = ((st.ravel() == 1) & (st2.ravel() == 1)
                  & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.7 * PX))
            self.pts, self.ids = p1.reshape(-1, 2)[ok], self.ids[ok]
        trusted = hy_raw is not None and conf >= HORIZON_MIN_CONF
        # drop points that sank into the water (or left the frame bottom without a line)
        if edge is not None:
            xi = np.clip(self.pts[:, 0].astype(int), 0, gray.shape[1] - 1)
            lim = edge[xi] - WATER_EDGE_MARGIN + 6
        else:
            lim = (hy_raw + slope * (self.pts[:, 0] - gray.shape[1] / 2.0) + 6 * PX) if trusted else gray.shape[0]
        keep = self.pts[:, 1] < lim
        self.pts, self.ids = self.pts[keep], self.ids[keep]
        if len(self.pts) < TRACK_MIN:
            mask = (edge_mask(gray.shape, edge) if edge is not None
                    else above_line_mask(gray.shape, hy_raw, slope) if trusted
                    else np.full_like(gray, 255))
            for x, y in self.pts.astype(int):
                cv2.circle(mask, (int(x), int(y)), 8, 0, -1)
            new = cv2.goodFeaturesToTrack(gray, maxCorners=TRACK_MAX - len(self.pts),
                                          qualityLevel=0.005, minDistance=8 * PX, mask=mask)
            if new is not None:
                new = new.reshape(-1, 2).astype(np.float32)
                self.pts = np.vstack([self.pts, new])
                self.ids = np.r_[self.ids, np.arange(self.nid, self.nid + len(new))]
                self.nid += len(new)
        for i, (x, y) in zip(self.ids, self.pts):
            self.rows.append((self.f, i, x, y))
        self.prev = gray
        self.f += 1

    def array(self):
        return np.asarray(self.rows, np.float32).reshape(-1, 4)


def _skew(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                     np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _left_jacobian(w):
    """SO(3) left Jacobian: Exp(w + d) = Exp(J_l(w) d) Exp(w) to first order."""
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    W = _skew(w)
    small = th < 1e-6
    t = np.where(small, 1.0, th)
    a = np.where(small, 0.5, (1 - np.cos(t)) / t ** 2)
    b = np.where(small, 1 / 6, (t - np.sin(t)) / t ** 3)
    return np.eye(3) + a * W + b * W @ W


def ba_window(obs, K, s, w0):
    """Inverse-depth BA over one chunk. Frame 0 is the gauge (R_0 = I, c_0 = 0).
    Unknowns: rotation vectors w_j (frames 1..L), travel direction (az, el), inverse depth
    rho_i >= 0 of each track anchored at its first frame a. Known: distance s_j along that
    direction from OUR water speed, in camera heights -- the units of rho, so no scale needed.
      x_ij ~ R_j (R_a^T m_i + rho_i (c_a - c_j)),  c_j = s_j u(az, el)
    obs: (n, 4) track, frame, x, y; s: (L+1,); w0: (L+1, 3) initial rotation vectors.
    Returns (yaw of frame L w.r.t. frame 0, inliers) or (None, 0)."""
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix
    from scipy.spatial.transform import Rotation
    if len(obs) < BA_MIN_INLIERS:
        return None, 0
    fx, fy, cx, cy = K
    L = len(s) - 1
    tr, tk = np.unique(obs[:, 0].astype(int), return_inverse=True)
    nT = len(tr)
    fj, ox, oy = obs[:, 1].astype(int), obs[:, 2].astype(float), obs[:, 3].astype(float)
    anc = np.full(nT, 10 ** 9)
    np.minimum.at(anc, tk, fj)
    first = fj == anc[tk]
    m = np.zeros((nT, 3))
    m[tk[first], 0] = (ox[first] - cx) / fx
    m[tk[first], 1] = (oy[first] - cy) / fy
    m[:, 2] = 1.0
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    tk, fj, ox, oy = tk[~first], fj[~first], ox[~first], oy[~first]   # anchors carry no info
    if len(tk) < BA_MIN_INLIERS:
        return None, 0
    a = anc[tk]
    nO, nW = len(tk), 3 * L
    nP = nW + 2 + nT

    def fwd(p):
        w = np.vstack([np.zeros(3), p[:nW].reshape(L, 3)])
        az, el = p[nW], p[nW + 1]
        u = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        rho = p[nW + 2:]
        R = Rotation.from_rotvec(w).as_matrix()
        c = s[:, None] * u[None, :]
        v = np.einsum('nji,nj->ni', R[a], m[tk]) + rho[tk][:, None] * (c[a] - c[fj])
        return w, u, rho, R, np.einsum('nij,nj->ni', R[fj], v)

    def resid(p):
        xc = fwd(p)[4]
        z = np.maximum(xc[:, 2], 1e-6)
        return np.r_[fx * xc[:, 0] / z + cx - ox, fy * xc[:, 1] / z + cy - oy]

    def jac(p):
        w, u, rho, R, xc = fwd(p)
        az, el = p[nW], p[nW + 1]
        z = np.maximum(xc[:, 2], 1e-6)
        P = np.zeros((nO, 2, 3))                       # d pixel / d camera point
        P[:, 0, 0] = fx / z
        P[:, 0, 2] = -fx * xc[:, 0] / z ** 2
        P[:, 1, 1] = fy / z
        P[:, 1, 2] = -fy * xc[:, 1] / z ** 2
        Jl = _left_jacobian(w)
        Rj = R[fj]
        D_obs = -_skew(xc) @ Jl[fj]
        D_anc = Rj @ np.transpose(R[a], (0, 2, 1)) @ _skew(m[tk]) @ Jl[a]
        du = np.array([[np.cos(el) * np.cos(az), 0, -np.cos(el) * np.sin(az)],
                       [-np.sin(el) * np.sin(az), np.cos(el), -np.sin(el) * np.cos(az)]]).T
        ds = (s[a] - s[fj])[:, None, None]
        D_dir = rho[tk][:, None, None] * ds * (Rj @ du)
        D_rho = np.einsum('nij,nj->ni', Rj, u[None, :] * ds[:, :, 0])
        rr, cc, vv = [], [], []
        for k in range(2):
            row = np.arange(nO) + k * nO
            for frm, D in ((fj, D_obs), (a, D_anc)):
                ok = frm > 0
                blk = np.einsum('ni,nij->nj', P[:, k], D)
                for q in range(3):
                    rr.append(row[ok]); cc.append(3 * (frm[ok] - 1) + q); vv.append(blk[ok, q])
            bd = np.einsum('ni,nij->nj', P[:, k], D_dir)
            for q in range(2):
                rr.append(row); cc.append(np.full(nO, nW + q)); vv.append(bd[:, q])
            rr.append(row); cc.append(nW + 2 + tk); vv.append(np.einsum('ni,ni->n', P[:, k], D_rho))
        return csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                          shape=(2 * nO, nP))

    lo = np.r_[np.full(nW, -np.inf), -BA_AZ_MAX, -BA_EL_MAX, np.zeros(nT)]
    hi = np.r_[np.full(nW, np.inf), BA_AZ_MAX, BA_EL_MAX, np.full(nT, BA_RHO_MAX)]
    p0 = np.clip(np.r_[w0[1:].ravel(), 0.0, 0.0, np.zeros(nT)], lo + 1e-9, hi - 1e-9)
    try:
        # x_scale=1: with 'jac' scaling the solver never left the rotation-only start
        r = least_squares(resid, p0, jac=jac, bounds=(lo, hi), loss='huber', f_scale=1.5 * PX,
                          x_scale=1.0, max_nfev=BA_NFEV, method='trf')
    except Exception:
        return None, 0
    res = r.fun.reshape(2, -1)
    inl = int((np.hypot(res[0], res[1]) < 2.0 * PX).sum())
    Rl = Rotation.from_rotvec(r.x[nW - 3:nW]).as_matrix()
    return float(np.arctan2(Rl[0, 2], Rl[2, 2])), inl


def ba_course(rows, K, dyaw, step_dist):
    """Per-step yaw: BA per non-overlapping BA_N-frame chunk (its yaw spread evenly over the
    chunk's steps), rotation-only `dyaw` where a chunk fails. Returns (dyaw, chunks used)."""
    fr = rows[:, 0].astype(int)
    order = np.argsort(fr, kind='stable')
    rows, fr = rows[order], fr[order]
    n = len(dyaw)
    out = np.array(dyaw, float)
    starts = np.searchsorted(fr, np.arange(0, n + 2))
    used = 0
    for c0 in range(0, n - BA_N + 1, BA_N):
        ch = rows[starts[c0]:starts[min(c0 + BA_N + 1, len(starts) - 1)]].astype(float)
        ch[:, 0] -= c0
        ch = ch[:, [1, 0, 2, 3]]                       # track, frame, x, y
        u, cnt = np.unique(ch[:, 0], return_counts=True)
        good = u[cnt >= BA_MIN_OBS]
        if len(good) > BA_MAX_TRACKS:
            good = good[np.argsort(-cnt[cnt >= BA_MIN_OBS])[:BA_MAX_TRACKS]]
        ch = ch[np.isin(ch[:, 0], good)]
        s = np.r_[0.0, np.cumsum(step_dist[c0:c0 + BA_N])]
        w0 = np.zeros((BA_N + 1, 3))
        w0[:, 1] = np.r_[0.0, np.cumsum(dyaw[c0:c0 + BA_N])]
        y, inl = ba_window(ch, K, s, w0)
        if y is not None and inl >= BA_MIN_INLIERS:
            out[c0:c0 + BA_N] = y / BA_N
            used += 1
    return out, used


def run_clip(pre, method, raw_path=None, masker=None, vo=None, emat=False, ba=False,
             yaw_mult=1.0, tilt=False, edges=None, edge_with_line=False, speed_from=None):
    """`masker` + `vo=(height, width, fx, fy, cx, cy)`: `pre` yields FULL-res frames, which
    the masker sees; VO runs on their 2x INTER_AREA reduction cropped to the half-res
    pipeline's size -- the same pixels ClipPreprocessor(half_res=True) would give.
    `speed_from`: raw npz of an earlier run on the same frames (--speed-from): its per-step water
    speed and quality replace this run's for the integration; the course (incl. BA) is this run's."""
    times, dec, dyaw, dts, spd = [], [], [], [], []
    profiles, dy_rot = [], []   # horizon-independent water flow per row + rotation pitch shift
    prof_all = []               # every step's profile (NaN where none) for the water-plane horizon
    prev = prev_t = None
    rng = np.random.default_rng(0)
    hy_prev = boat_prev = None
    slope_prev = 0.0
    chunk, chunk_hz = [], None    # essential-matrix course (emat=True)
    tracks = HorizonTracks() if ba else None
    hy_hist = []            # raw horizon rows, any confidence -- for the speed step only
    slope_hist, slope_track = [], []   # horizon slope (roll), confident lines only
    hy_track = []           # hy_speed per frame (VO px), saved for triangulation
    hy_frame = []           # raw per-frame horizon + its confidence: follows pitch, unlike hy_speed
    edge_prev = None        # WaSR water edge of the previous frame (edges given)
    for f_i, (_index, image, intr, pts, decoded) in enumerate(pre.frames_with_time()):
        full = None
        if masker is not None:
            full = image
            vh, vw = vo[0], vo[1]
            image = cv2.resize(full, (full.shape[1] // 2, full.shape[0] // 2),
                               interpolation=cv2.INTER_AREA)[:vh, :vw]
            intr = np.asarray(vo[2:], np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if image.ndim == 3:
            yc_, slope_raw, conf = estimate_horizon_line(image)
            hy_raw = int(round(yc_))
        else:
            hy_raw, slope_raw, conf = None, 0.0, 0.0
        if tilt and hy_raw is not None and conf >= HORIZON_MIN_CONF:
            slope_hist = (slope_hist + [slope_raw])[-HORIZON_SMOOTH_N:]
        slope = float(np.median(slope_hist)) if slope_hist else 0.0
        slope_track.append(slope)
        hy = hy_raw if conf >= HORIZON_MIN_CONF else None
        # The rotation mask needs a TRUSTED line (a wrong mask hides the shoreline). The speed
        # step only needs roughly where water starts, and a low-confidence line behind a city
        # skyline is still at the waterline -- so it takes every estimate, median-smoothed.
        if hy_raw is not None:
            hy_hist = (hy_hist + [hy_raw])[-HORIZON_SMOOTH_N:]
        hy_speed = float(np.median(hy_hist)) if hy_hist else None
        hy_track.append(np.nan if hy_speed is None else hy_speed)
        hy_frame.append((np.nan, 0.0) if hy_raw is None else (hy_raw, conf))
        edge = edges.at(f_i) if edges is not None else None
        if edge is not None and edge_with_line and hy_raw is not None and conf >= HORIZON_MIN_CONF:
            # points must sit above BOTH the water edge and the trusted horizon line
            # (+ margin + 6: after edge_mask subtracts the margin, the line keeps its usual 6 px)
            edge = np.minimum(edge, hy_raw + slope * (np.arange(gray.shape[1]) - gray.shape[1] / 2.0)
                              + WATER_EDGE_MARGIN + 6)
        if tracks is not None:
            tracks.update(gray, hy_raw, conf, slope, edge)
        boat = None
        if masker is not None:
            boxes = masker.update(full, None if hy_speed is None else 2 * hy_speed, len(times))
            boat = BoatMasker.mask(gray.shape, boxes, 0.5) if boxes else None
        if prev is not None:
            fx, fy, cx, cy = intr[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            mask = None
            if edge_prev is not None:
                mask = edge_mask(prev.shape, edge_prev)
            elif hy_prev is not None:
                mask = above_line_mask(prev.shape, hy_prev, slope_prev)
            if boat_prev is not None:
                if mask is None:
                    mask = np.full_like(prev, 255)
                mask[boat_prev] = 0
            R = rotation_far(prev, gray, K, mask, rng)
            d = 0.0
            if R is not None:
                d = float(np.arctan2(R[0, 2], R[2, 2]))    # yaw about camera y (down)
                if abs(d) > MAX_DYAW:
                    d = 0.0
            s = None
            if method in ('flow', 'adaptive') and R is not None and hy_speed is not None:
                prof = water_profile(prev, gray, K, R, boat, roll=slope, pivot_y=hy_speed)
                s = speed_from_profile(prof, hy_speed, K[1, 1])
                prof_all.append(prof)
                if raw_path is not None:
                    profiles.append(prof)
                    dy_rot.append(K[1, 1] * R[1, 2] / R[2, 2])   # horizon row shift from pitch
            else:
                prof_all.append(None)
                if raw_path is not None and method in ('flow', 'adaptive'):
                    profiles.append(np.full(gray.shape[0], np.nan, np.float32))
                    dy_rot.append(np.nan)
            dyaw.append(d); dts.append(pts - prev_t); spd.append(s)
        times.append(float(pts)); dec.append(int(decoded))
        if emat:
            # chunk k covers frames k*N .. k*N+N (steps k*N .. k*N+N-1); its yaw replaces the
            # rotation-only steps, spread evenly; a failed chunk keeps rotation-only
            if not chunk:
                chunk_hz = (hy_raw, conf, slope, edge)
            chunk.append(gray)
            if len(chunk) == EMAT_N + 1:
                fx_, fy_, cx_, cy_ = intr[:4]
                Kc = np.array([[fx_, 0, cx_], [0, fy_, cy_], [0, 0, 1.0]])
                y = chunk_yaw_essential(chunk, Kc, *chunk_hz)
                if np.isfinite(y):
                    dyaw[-EMAT_N:] = [y / EMAT_N] * EMAT_N
                chunk, chunk_hz = [gray], (hy_raw, conf, slope, edge)
        prev, prev_t, hy_prev, boat_prev, slope_prev = gray, pts, hy, boat, slope
        edge_prev = edge
    ba_used = None
    if tracks is not None and len(dts) >= BA_N:
        v, _ = adaptive_speed(times, dts, spd)
        i_ = pre.intrinsics
        Kba = np.asarray(vo[2:6] if vo is not None else (i_.fx, i_.fy, i_.cx, i_.cy), float)
        dyaw, ba_used = ba_course(tracks.array(), Kba, dyaw, v * np.asarray(dts))
        dyaw = list(dyaw)
    hz_water = quality = None
    if method == 'adaptive' and any(p is not None for p in prof_all):
        fy_ = float(pre.intrinsics.fy if vo is None else vo[3])
        H_ = next(len(p) for p in prof_all if p is not None)
        P_ = np.array([p if p is not None else np.full(H_, np.nan, np.float32) for p in prof_all], float)
        h_step, hz_water = water_plane_horizon(P_, hy_track, fy_)
        spd = [speed_from_profile(P_[i], h_step[i], fy_) if s_ is not None else None
               for i, s_ in enumerate(spd)]
        quality = [profile_fit_quality(P_[i], h_step[i], fy_) if s_ is not None else np.nan
                   for i, s_ in enumerate(spd)]
    if speed_from is not None:
        assert np.array_equal(np.asarray(speed_from['times'], float), np.asarray(times, float)), \
            '--speed-from run has different frame times'
        spd = [None if not np.isfinite(x) else float(x) for x in speed_from['spd']]
        quality = list(np.asarray(speed_from['quality'], float))
    if raw_path is not None:
        # per-step measurements, so integration variants can be scored without re-decoding
        extra = {}
        if masker is not None:
            # full-res geometry = VO geometry x2 (VO frame is the exact 2x reduction)
            extra = dict(obs=np.asarray(masker.obs, float).reshape(-1, 5), hy=hy_track,
                         hy_frame=np.asarray(hy_frame, float),
                         k_full=2 * np.asarray(vo[2:], float))
        np.savez(raw_path, times=times, dec=dec, dyaw=dyaw, dts=dts,
                 spd=[np.nan if x is None else x for x in spd],
                 **({'profile': np.asarray(profiles, np.float16), 'dy_rot': np.asarray(dy_rot),
                     'fy': float(pre.intrinsics.fy if vo is None else vo[3])} if profiles else {}),
                 slope=np.asarray(slope_track, np.float32),
                 **({'hz_water_used': hz_water} if hz_water is not None else {}),
                 # the speed horizon row actually used per step (detector median or water-plane), exact,
                 # so offline consumers (Colab top view) need not rebuild it from partial horizon tracks
                 **({'h_step': np.asarray(h_step, float)} if hz_water is not None else {}),
                 **({'quality': np.asarray(quality, float)} if quality is not None else {}),
                 **extra, **({'ba_used': ba_used} if ba_used is not None else {}),
                 **({'speed_from': True} if speed_from is not None else {}))
    return integrate(times, dec, [yaw_mult * d for d in dyaw], dts, spd, method, quality)


def smooth_speed(times, dts, spd):
    """Per-step speed (dz/h per second), median over SPEED_WINDOW_S, gaps filled."""
    t = np.asarray(times[1:])
    raw = np.array([np.nan if s is None else s / dt for s, dt in zip(spd, dts)])
    out = np.full(len(raw), np.nan)
    for i in range(len(raw)):
        w = raw[np.abs(t - t[i]) <= SPEED_WINDOW_S / 2]
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            out[i] = np.median(w)
    good = np.isfinite(out)
    if not good.any():
        return np.ones(len(raw)), 0.0
    out = np.interp(t, t[good], out[good])
    return np.clip(out, 0.0, None), float(good.mean())


# --method adaptive: how much to trust the water speed is decided PER CLIP from the water
# speed itself (no camera rule, no GPS). Chosen on TRAIN (grid over window, block, t0, t1,
# min validity), val checked once: train 11.4/17.4 -> 11.9/15.1, val 14.4/14.9 -> 12.0/13.7
# (median/mean drift % vs `auto`). Why: water speed wins big when the boat really changes
# speed (clip_085 86 -> 11.5 %) and only adds noise when it does not (clip_051 2.4 -> 40 %).
ADAPT_WINDOW_S = 12.0       # median window on log water speed
ADAPT_BLOCK_S = 30.0        # signal = spread of block medians of log water speed
ADAPT_T0, ADAPT_T1 = 0.25, 0.6   # spread <= T0 -> constant speed, >= T1 -> full water speed
ADAPT_MIN_VALID = 0.6       # fewer valid water steps -> constant speed


# The water speed's error is slow and has no simple structure (heading, time, speed, pitch explain
# none of it out of sample), but a step that fits the water law badly (profile_fit_quality) is
# more often wrong (Spearman vs 10 s block error +0.38 pooled, +0.23 within clips). So the 12 s
# window takes a median WEIGHTED by 1/max(q, ADAPT_Q_FLOOR)^2 instead of a plain one. Fixed
# constants, no grid. Train 6.88/9.94 -> 6.45/9.58 (both halves of train agree on it); val checked
# once 8.28/10.48 -> 8.19/10.12 (21 untouched val clips 10.53 -> 10.38). Rejected on train: dropping
# bad steps, shrinking toward the clip median where quality is low, longer windows, slew limits,
# an even/odd-step agreement gate.
ADAPT_Q_FLOOR = 0.05


def _weighted_median(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def adaptive_speed(times, dts, spd, quality=None):
    """exp(m + alpha * (smoothed log water speed - m)); alpha from the clip's own
    speed variation relative to T0/T1. `quality` (per step, profile_fit_quality) switches the
    smoothing to the quality-weighted median. Returns (v, alpha)."""
    t = np.asarray(times[1:], float)
    w = np.array([np.nan if s is None else s / dt for s, dt in zip(spd, dts)])
    lw = np.where(np.isfinite(w) & (w > 0), np.log(np.where(w > 0, w, 1.0)), np.nan)
    valid = float(np.isfinite(lw).mean())
    if valid == 0:
        return np.ones(len(t)), 0.0
    m = float(np.nanmedian(lw))
    blocks = []
    for s0 in np.arange(t[0], t[-1] - ADAPT_BLOCK_S / 2, ADAPT_BLOCK_S):
        x = lw[(t >= s0) & (t < s0 + ADAPT_BLOCK_S)]
        if np.isfinite(x).sum() > ADAPT_BLOCK_S * 2:
            blocks.append(np.nanmedian(x))
    S = float(np.std(blocks)) if len(blocks) > 2 else 0.0
    a = 0.0 if valid < ADAPT_MIN_VALID else float(np.clip((S - ADAPT_T0) / (ADAPT_T1 - ADAPT_T0), 0, 1))
    sm = np.full(len(t), np.nan)
    wq = None if quality is None else 1.0 / np.maximum(np.asarray(quality, float), ADAPT_Q_FLOOR) ** 2
    for i in range(len(t)):
        j = (np.abs(t - t[i]) <= ADAPT_WINDOW_S / 2) & np.isfinite(lw)
        if wq is not None:
            j &= np.isfinite(wq)
            if j.sum() >= 3:
                sm[i] = _weighted_median(lw[j], wq[j])
            continue
        x = lw[j]
        if len(x) >= 3:
            sm[i] = np.median(x)
    g = np.isfinite(sm)
    sm = np.interp(t, t[g], sm[g]) if g.any() else np.full(len(t), m)
    return np.exp(m + a * (sm - m)), a


def integrate(times, dec, dyaw, dts, spd, method, quality=None):
    if method == 'flow':
        v, valid = smooth_speed(times, dts, spd)
    elif method == 'adaptive':
        v, valid = adaptive_speed(times, dts, spd, quality)
    else:
        v, valid = np.ones(len(dts)), None
    yaw, xy, pos = 0.0, np.zeros(2), [[0.0, 0.0, 0.0]]
    for d, dt, vi in zip(dyaw, dts, v):
        yaw += d
        xy = xy + vi * dt * np.array([np.sin(yaw), np.cos(yaw)])
        pos.append([xy[0], xy[1], 0.0])
    return dec, times, pos, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, type=Path, help='release/release_manifest.json')
    ap.add_argument('--split', default='val')
    ap.add_argument('--clips', help='comma-separated clip ids; overrides --split')
    ap.add_argument('--method', choices=['line', 'horizon', 'flow', 'auto', 'adaptive'], required=True)
    ap.add_argument('--target-fps', type=float, default=8.0)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--save-raw', action='store_true',
                    help='also write <output>/raw/<clip>.npz (per-step yaw, dt, water speed)')
    ap.add_argument('--skip-existing', action='store_true',
                    help='keep predictions already in <output> (resume after a disconnect)')
    ap.add_argument('--no-ba', action='store_true',
                    help='method adaptive: keep rotation-only course instead of the BA')
    ap.add_argument('--full-res', action='store_true',
                    help='VO on full-resolution frames (pixel thresholds x2, one more LK pyramid level)')
    ap.add_argument('--speed-from', type=Path,
                    help='run dir with raw/ of a --full-res --save-raw run: its water speed replaces this '
                         "run's for cameras in --speed-from-modes (course stays this run's)")
    ap.add_argument('--speed-from-modes', default='wide',
                    help='comma-separated camera modes that take --speed-from (default: wide)')
    ap.add_argument('--wasr-dir', type=Path,
                    help='dir of <clip>.npz WaSR water probabilities: course masks follow the water edge')
    ap.add_argument('--wasr-margin', type=float,
                    help=f'px above the water edge where course points may start (default {WATER_EDGE_MARGIN})')
    ap.add_argument('--wasr-with-line', action='store_true',
                    help='course points must also sit above the trusted horizon line')
    ap.add_argument('--yolo-dir', type=Path,
                    help='NCNN band model dir; cut detected vessels out of the measurements')
    ap.add_argument('--yolo-every', type=int, default=4, help='detect every N-th VO frame')
    ap.add_argument('--yolo-conf', type=float, default=0.25)
    ap.add_argument('--yolo-threads', type=int, default=2)
    a = ap.parse_args()
    if a.full_res:
        if a.yolo_dir is not None:
            ap.error('--full-res and --yolo-dir cannot be combined')
        globals()['PX'] = 2.0
        LK['maxLevel'] = 4
    if a.speed_from is not None and (a.full_res or a.method != 'adaptive'):
        ap.error('--speed-from goes with a half-resolution --method adaptive run')
    if a.wasr_margin is not None:
        globals()['WATER_EDGE_MARGIN'] = a.wasr_margin
    sys.path.insert(0, str(a.dataset.resolve().parent / 'starter'))
    from preprocessing import ClipPreprocessor
    from public_dataset import (asset_path, load_manifest, load_timestamps, select_clips,
                                validate_prediction, write_json)

    manifest = load_manifest(a.dataset)
    root = a.dataset.resolve().parent
    out = a.output / 'predictions'
    out.mkdir(parents=True, exist_ok=True)
    if a.save_raw:
        (a.output / 'raw').mkdir(exist_ok=True)
    def line_prediction(ft):
        """Straight line at constant speed on every 3rd released frame: the no-vision floor."""
        idx = list(range(0, len(ft), 3))
        times = [float(ft[i]) for i in idx]
        return idx, times, [[0.0, t - times[0], 0.0] for t in times], None

    def gap_report(times, ft):
        """Largest gap between predicted samples, and the frame-time range left uncovered at the ends (s)."""
        t, f = np.asarray(times, float), np.asarray(ft, float)
        return (float(np.diff(t).max()) if len(t) > 1 else np.inf), float(max(t[0] - f[0], f[-1] - t[-1]))

    counts = dict(ok=0, fallback=0, failed=0, skipped=0, gaps=0)
    if a.speed_from is not None:
        # a missing file must stop the run, not become a straight-line fallback inside the loop
        need = [c['key'] for c in select_clips(manifest, a.split, a.clips)
                if c['camera_mode'] in a.speed_from_modes.split(',')]
        missing = [k for k in need if not (a.speed_from / 'raw' / f'{k}.npz').exists()]
        if missing:
            ap.error(f'--speed-from {a.speed_from}: no raw for {", ".join(missing)}')
    for clip in select_clips(manifest, a.split, a.clips):
        key = clip['key']
        if a.skip_existing and (out / f'{key}.json').exists():
            print(f'{key}: exists, skipped', flush=True)
            counts['skipped'] += 1
            continue
        ft = None
        try:
            ft = load_timestamps(root, clip)
            if a.method == 'line':
                dec, times, pos, valid = line_prediction(ft)
            else:
                video = str(asset_path(root, clip['video']))
                pre = ClipPreprocessor(video, clip['camera_mode'], target_fps=a.target_fps,
                                       half_res=not a.full_res,
                                       calibration=clip['calibration'], frame_timestamps=ft)
                masker = vo = None
                if a.yolo_dir is not None:
                    i = pre.intrinsics         # half-res geometry: what VO keeps using
                    vo = (i.height, i.width, i.fx, i.fy, i.cx, i.cy)
                    pre = ClipPreprocessor(video, clip['camera_mode'], target_fps=a.target_fps,
                                           half_res=False, calibration=clip['calibration'],
                                           frame_timestamps=ft)
                    masker = BoatMasker(str(a.yolo_dir), a.yolo_every, a.yolo_conf,
                                        a.yolo_threads)
                raw = a.output / 'raw' / f'{key}.npz' if a.save_raw else None
                method = a.method
                if method == 'auto':
                    method = 'flow' if clip['camera_mode'] in FLOW_MODES else 'horizon'
                emat = clip['camera_mode'] in EMAT_MODES
                ba = method == 'adaptive' and not emat and not a.no_ba
                adaptive = method == 'adaptive'
                dec, times, pos, valid = run_clip(
                    pre, method, raw, masker, vo, emat=emat, ba=ba,
                    yaw_mult=YAW_MULT.get(clip['camera_mode'], 1.0) if adaptive else 1.0,
                    tilt=adaptive and clip['camera_mode'] in TILT_MODES,
                    edges=WaterEdges(a.wasr_dir / f'{key}.npz') if a.wasr_dir is not None else None,
                    edge_with_line=a.wasr_with_line,
                    speed_from=(np.load(a.speed_from / 'raw' / f'{key}.npz') if a.speed_from is not None
                                and clip['camera_mode'] in a.speed_from_modes.split(',') else None))
            pred = dict(key=key, frame_indices=dec, times=times, positions=pos)
            validate_prediction(pred, key, ft)
            write_json(out / f'{key}.json', pred)
            extra = '' if valid is None else (f' alpha={valid:.2f}' if method == 'adaptive'
                                               else f' speed_valid={valid:.2f}')
            if a.method != 'line' and masker is not None:
                extra += f' boats_in={masker.frames_with_boats / max(1, masker.n):.2f}'
            gap, ends = gap_report(times, ft)
            if gap > SERVER_MAX_GAP_S or ends > SERVER_MAX_GAP_S:
                extra += f' GAP max={gap:.2f}s ends={ends:.2f}s (test server needs <= {SERVER_MAX_GAP_S:.0f} s)'
                counts['gaps'] += 1
            print(f'{key}: {len(times)} poses{extra}', flush=True)
            counts['ok'] += 1
        except Exception as exc:
            # A clip without a prediction is not scored at all; a straight line still is.
            try:
                if ft is None:
                    raise
                dec, times, pos, _ = line_prediction(ft)
                pred = dict(key=key, frame_indices=dec, times=times, positions=pos)
                validate_prediction(pred, key, ft)
                write_json(out / f'{key}.json', pred)
                print(f'{key}: FALLBACK straight line ({exc})', flush=True)
                counts['fallback'] += 1
            except Exception as exc2:
                print(f'{key}: FAILED {exc} / fallback: {exc2}', flush=True)
                counts['failed'] += 1
    print('SUMMARY ' + ' '.join(f'{k}={v}' for k, v in counts.items()), flush=True)
    if counts['failed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
