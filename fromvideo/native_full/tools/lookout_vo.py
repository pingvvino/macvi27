import os
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'camera'))
# Pixel thresholds below are written for the half-resolution VO frame (the starter's default);
# --full-res multiplies them by PX = 2 so they keep the same angle, and adds one pyramid level.
PX = 1.0
# water band used for speed, as angle below the horizon (rad): rows right under the horizon
# amplify flow noise by 1/v^2, rows far below are near-field spray/bow
WATER_V_MIN, WATER_V_MAX = 0.04, 0.35
# Owner 25.10 ('for slow ones cut a share of the lower water'): when a step's flow at v = 0.2 is below LOOKOUT_WATER_CUT_PX px,
# speed_from_profile drops the bottom LOOKOUT_WATER_CUT_SLOW share of the band rows (error_breakdown.py --water-var: 0.5 on the
# slow dev windows train |log err| 0.552 -> 0.482, val 0.717 -> 0.736; 0 = off)
WATER_CUT_SLOW = float(os.environ.get('LOOKOUT_WATER_CUT_SLOW', '0'))
WATER_CUT_PX = float(os.environ.get('LOOKOUT_WATER_CUT_PX', '0.4'))
# research only (owner 18.10): extra water profiles from a CENTRAL column sector, as fractions of the width, e.g.
# LOOKOUT_WATER_X_SECTORS=0.5,0.25 -- saved in --save-raw as profile_x50, profile_x25; the speed itself is unchanged
# LOOKOUT_WATER_X_CUT_DEG=5,10 -- the opposite: the central +-5 / +-10 deg (column angle from the principal point) LEFT
# OUT, the sides kept -- saved as profile_cut050, profile_cut100 (tenths of a degree)
WATER_X_SECTORS = tuple(float(x) for x in os.environ.get('LOOKOUT_WATER_X_SECTORS', '').split(',') if x.strip())
WATER_X_CUT_DEG = tuple(float(x) for x in os.environ.get('LOOKOUT_WATER_X_CUT_DEG', '').split(',') if x.strip())
SECTOR_NAMES = [f'x{int(round(fr * 100))}' for fr in WATER_X_SECTORS] + [f'cut{int(round(dg * 10)):03d}' for dg in WATER_X_CUT_DEG]
# LOOKOUT_WATER_U=1 (research, owner 18.10 -- wave advection of the water texture): also the HORIZONTAL flow per row
# (derotated, normalised by fx), fitted per row as u(x) = a + b (x - cx) / fx over the same water pixels; a = the
# row's sideways texture motion at the optical axis. On the water plane a lateral texture velocity L gives a ~ (L/h) v,
# linear in the angle v below the horizon (a residual yaw adds a constant). Saved in --save-raw as profile_u.
# LOOKOUT_WATER_U=b (owner 19.10 -- columns): instead the row's SLOPE b (saved as profile_ub). Forward motion over the water
# plane spreads the texture sideways, u = s x v (x = column offset /fx, v = angle below the horizon), so b = s v: a second
# speed measurement, from the columns, untouched by pitch (vertical) and by any uniform sideways motion (yaw, waves).
# '1' = profile_u only, 'b' = profile_ub only, 'ub' = both.
_WU = os.environ.get('LOOKOUT_WATER_U', '')
WATER_U = _WU in ('1', 'b', 'ub')
# LOOKOUT_WATER_GRID=N (owner 19.10 -- 2-D map of where the water carries the speed): the band also split into N equal
# column blocks; per row and block the median vertical (gv<c>) and horizontal (gu<c>) derotated flow, normalised by fy / fx,
# saved in --save-raw as profile_gv<c> / profile_gu<c>. Rows are binned and the horizon applied offline. Default off.
WATER_GRID = int(os.environ.get('LOOKOUT_WATER_GRID', '0') or 0)
SECTOR_NAMES = SECTOR_NAMES + [f'g{a}{c:02d}' for a in 'vu' for c in range(WATER_GRID)]
# LOOKOUT_WATER_FIXED=1 (research, owner 20.10 -- clip_104: the bow and lens drops fill the lower water rows, so the row
# median reads THEIR flow): a pixel is BOAT-FIXED when its blurred brightness stays within FIXED_STILL grey levels over
# FIXED_LAG steps while the scene changes (75th percentile of the band's change > FIXED_MOVE), in >= FIXED_SHARE of those
# steps (running mean, weight FIXED_EMA, after FIXED_MIN_N of them); the mask is dilated 2 px. The profile is then also
# taken without those pixels, saved in --save-raw as profile_nf, plus profile_fxfrac (share of each row's textured pixels
# flagged). The speed itself is unchanged.
WATER_FIXED = os.environ.get('LOOKOUT_WATER_FIXED', '') == '1'
_SECTOR_LAST = {}
BOTTOM_CUT = 0.97           # the camera can be pitched up (horizon at 0.85 H): keep the band
_DIS_P = None
PROFILE_TOP, PROFILE_BOT = 0.05, 0.97   # rows covered by the water-flow profile (frac of H)
# DIS preset MEDIUM computes the flow at finest scale 1 (the frame halved, then upsampled); None keeps it,
# 0 computes it at the frame's own resolution (--dis-finest-scale)
DIS_FINEST_SCALE = None
def water_profile(prev, cur, K, R, boat=None, roll=0.0, pivot_y=None, fixed=None):
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
        if fixed is not None:
            fixed = cv2.warpAffine(fixed.astype(np.uint8), A, size, flags=cv2.INTER_NEAREST) > 0
    y0, y1 = int(PROFILE_TOP * h), int(PROFILE_BOT * h)
    if _DIS_P is None:
        _DIS_P = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        if DIS_FINEST_SCALE is not None:
            _DIS_P.setFinestScale(int(DIS_FINEST_SCALE))
    F_ = _DIS_P.calc(prev_w[y0:y1], cur[y0:y1], None)
    fl = F_[:, :, 1] / K[1, 1]
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
    _SECTOR_LAST.clear()
    if WATER_FIXED:                                       # the same rows without the boat-fixed pixels
        onf, ofr = np.full(h, np.nan, np.float32), np.full(h, np.nan, np.float32)
        if fixed is not None:
            fm = fixed[y0:y1]
            for r in range(fl.shape[0]):
                m = tex[r]
                if m.sum() >= 10:
                    ofr[y0 + r] = (m & fm[r]).sum() / m.sum()
                    mk = m & ~fm[r]
                    if mk.sum() >= 10:
                        onf[y0 + r] = np.median(fl[r, mk])
        _SECTOR_LAST['nf'], _SECTOR_LAST['fxfrac'] = onf, ofr
    W_ = fl.shape[1]
    ang = np.degrees(np.abs(np.arctan((np.arange(W_) - K[0, 2]) / K[0, 0])))
    masks = [np.abs(np.arange(W_) - W_ / 2.0) <= fr * W_ / 2.0 for fr in WATER_X_SECTORS] + [ang > dg for dg in WATER_X_CUT_DEG]
    if WATER_U:                                           # per-row LS u = a + b X, X = (x - cx) / fx, water pixels only
        fu = F_[:, :, 0] / K[0, 0]; Xc = (np.arange(W_) - K[0, 2]) / K[0, 0]
        m_ = tex.astype(np.float64); n_ = m_.sum(1); sx = m_ @ Xc; sxx = m_ @ (Xc * Xc)
        su = (m_ * fu).sum(1); sxu = (m_ * fu) @ Xc
        den = n_ * sxx - sx * sx; ok_ = (n_ >= 10) & (den > 1e-12)
        b_ = np.where(ok_, (n_ * sxu - sx * su) / np.where(ok_, den, 1), np.nan)
        a_ = np.where(ok_, (su - b_ * sx) / np.where(n_ > 0, n_, 1), np.nan)
        ou = np.full(h, np.nan, np.float32); ou[y0:y1] = a_
        _SECTOR_LAST['u'] = ou
        ob = np.full(h, np.nan, np.float32); ob[y0:y1] = b_
        _SECTOR_LAST['ub'] = ob
    if WATER_GRID:
        fu = F_[:, :, 0] / K[0, 0]; edges = np.linspace(0, W_, WATER_GRID + 1).astype(int)
        for c in range(WATER_GRID):
            tc = tex[:, edges[c]:edges[c + 1]]; ok_ = tc.sum(1) >= 10
            for a, f in (('v', fl), ('u', fu)):
                fc = np.where(tc, f[:, edges[c]:edges[c + 1]], np.nan)
                o = np.full(h, np.nan, np.float32)
                if ok_.any():
                    o[y0:y1][ok_] = np.nanmedian(fc[ok_], 1)
                _SECTOR_LAST[f'g{a}{c:02d}'] = o
    for name, xs in zip([nm for nm in SECTOR_NAMES if nm not in ('u', 'ub') and not nm.startswith(('gv', 'gu'))], masks):            # same rows, only these columns
        o = np.full(h, np.nan, np.float32)
        for r in range(fl.shape[0]):
            m = tex[r] & xs
            if m.sum() >= 10:
                o[y0 + r] = np.median(fl[r, m])
        _SECTOR_LAST[name] = o
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
    s = float(np.sum(profile[m] * v[m] ** 2) / np.sum(v[m] ** 4))
    if WATER_CUT_SLOW > 0 and fy * s * 0.04 < WATER_CUT_PX:      # slow (flow at v = 0.2 below the floor): the band's bottom rows
        idx = np.flatnonzero(m); keep = idx[:len(idx) - int(WATER_CUT_SLOW * len(idx))]     # under-read -- fit without them
        if len(keep) >= 10:
            s = float(np.sum(profile[keep] * v[keep] ** 2) / np.sum(v[keep] ** 4))
    return s
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
WATER_HZ_V = (0.06, 0.35)
