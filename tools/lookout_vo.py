"""The parts of the LOOKOUT VO tool that the final chain re-runs on the tool's saved raws: the water law per step
(speed_from_profile), the per-clip trusted water speed (adaptive_speed) and the position integration (integrate).

Water speed: a water point at angle v below the horizon lies at Z = h/v (h = camera height); after removing rotation,
forward motion dz moves it down by dv = v^2 * dz/h, so dz/h is the least-squares s of dv = s v^2 over the water rows.
h is unknown but constant per clip -- the metric's per-clip similarity fit absorbs it.
The full tool (video -> raws) is the code snapshot named in REPRODUCE_FROM_VIDEO.md."""
import numpy as np

WATER_V_MIN, WATER_V_MAX = 0.04, 0.35   # water band, angle below the horizon (rad)
WATER_HZ_V = (0.06, 0.35)               # the band of the speed horizon scan (course_combo.hz_scan)
BOTTOM_CUT = 0.97                       # rows below this share of the image are never water (own bow)
YAW_MULT = {'flir-VIS': 0.65, 'wide': 0.9}


def speed_from_profile(profile, hy, fy):
    """dz/h for this step: least squares of dv = s v^2 over the water rows below `hy`."""
    h = len(profile)
    rows = np.arange(h)
    v = (rows - hy) / fy
    m = np.isfinite(profile) & (v >= WATER_V_MIN) & (v <= WATER_V_MAX) & (rows < BOTTOM_CUT * h)
    if m.sum() < 10:
        return None
    s = float(np.sum(profile[m] * v[m] ** 2) / np.sum(v[m] ** 4))
    return s


ADAPT_WINDOW_S = 12.0       # median window on log water speed
ADAPT_BLOCK_S = 30.0        # signal = spread of block medians of log water speed
ADAPT_T0, ADAPT_T1 = 0.25, 0.6   # spread <= T0 -> constant speed, >= T1 -> full water speed
ADAPT_MIN_VALID = 0.6
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
    """positions from per-step yaw changes and the adaptive water speed (method 'adaptive', the only one the chain uses)"""
    assert method == 'adaptive', method
    v, valid = adaptive_speed(times, dts, spd, quality)
    yaw, xy, pos = 0.0, np.zeros(2), [[0.0, 0.0, 0.0]]
    for d, dt, vi in zip(dyaw, dts, v):
        yaw += d
        xy = xy + vi * dt * np.array([np.sin(yaw), np.cos(yaw)])
        pos.append([xy[0], xy[1], 0.0])
    return dec, times, pos, valid
