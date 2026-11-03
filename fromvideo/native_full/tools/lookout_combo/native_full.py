import sys, os, io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()): import course_combo as CC
import lookout_vo as LV, metrics
FV_RAW = os.environ.get('FV_RAW', 'runs/fv_hstep/raw')     # flir-VIS raws for --fv-signals / --fv-water / --hz-pitch (dev, test);
FVT_RAW = os.environ.get('FVT_RAW', 'runs/fvt_hstep/raw')  # runs/roll/fv(t)/raw = the re-run with the roll from the camera rotation
FV_TBUF = []   # fv_frames: the times (s) of the frames in its yielded buffer
FV_BGR = []    # fv_frames: the newest frame as decoded (colour where the camera has it)
def fv_frames(k, every, keep=1, fps=8.0, hz=None, half=True):
    """Gray frames of clip k, half-res, at the tool's ~8 Hz step (fps=8.0) or another --target-fps (1000 = every native frame),
    every `every`-th frame with a finite raw horizon: yields (frame index, the last `keep` + 1 frames, K, horizon row, time s).
    Raw horizon from FVT_RAW / FV_RAW: at 8 Hz the step ending at this frame, otherwise that of the nearest step; hz = (rows per
    step, step end times) instead (any camera, e.g. --water-lag)."""
    import json, cv2
    from preprocessing import ClipPreprocessor
    from public_dataset import asset_path, load_timestamps
    tcl = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
    test = k in tcl; c = tcl[k] if test else CC.cl[k]; root = 'data/test' if test else 'data/release'
    if hz is None: d = np.load(f"{FVT_RAW if test else FV_RAW}/{k}.npz"); hs = np.asarray(d['h_step'], float); ts = np.asarray(d['times'], float)[1:len(hs) + 1]
    else: hs, ts = np.asarray(hz[0], float), np.asarray(hz[1], float)
    pre = ClipPreprocessor(str(asset_path(root, c['video'])), c['camera_mode'], target_fps=fps, half_res=half,   # half=False: full-res frames, horizon still in half-res rows
                           calibration=c['calibration'], frame_timestamps=load_timestamps(root, c))
    buf = []; FV_TBUF.clear()
    for f_i, (_i, image, intr, pts, _d) in enumerate(pre.frames_with_time()):
        buf = (buf + [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image])[-(keep + 1):]
        FV_TBUF[:] = (FV_TBUF + [float(pts)])[-(keep + 1):]; FV_BGR[:] = [image]
        if len(buf) <= keep or f_i % every: continue
        h = (hs[f_i - 1] if f_i - 1 < len(hs) else np.nan) if fps == 8.0 else hs[int(np.argmin(np.abs(ts - float(pts))))]
        if not np.isfinite(h): continue
        fx, fy, cx, cy = [float(x) for x in np.asarray(intr)[:4]]
        yield f_i, buf, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), h, float(pts)
_DIS_B = None
def band_profile(prev, cur, K, y0):
    """LV.water_profile with R = I and no roll, its DIS on rows y0 .. PROFILE_BOT H only (the rows above are not used by the water
    law): per row the median vertical flow / fy over textured pixels (the same texture test), NaN elsewhere"""
    global _DIS_B
    import cv2
    H = prev.shape[0]; y1 = int(LV.PROFILE_BOT * H)
    if _DIS_B is None: _DIS_B = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = _DIS_B.calc(prev[y0:y1], cur[y0:y1], None)[:, :, 1] / K[1, 1]
    g = np.abs(cv2.Sobel(prev[y0:y1], cv2.CV_32F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(prev[y0:y1], cv2.CV_32F, 1, 0, ksize=3))
    tex = (g > 12 / LV.PX) & (prev[y0:y1] > 0); out = np.full(H, np.nan)
    for r in range(fl.shape[0]):
        if tex[r].sum() >= 10: out[y0 + r] = np.median(fl[r, tex[r]])
    return out
def native_full_job(k):
    """MACVI1 25.10 (owner: 'the test tracks at full resolution'): native_water_job's estimator (every native frame pair, R = I, the
    pair's c out, summed into the 8-Hz steps as a rate) on FULL-resolution frames (LV.PX = 2 as --full-res; the half-res frame is the
    full one halved and trimmed at the bottom / right, so rows and fy x2) -> the 8-Hz raw with spd / quality from that water
    (speed_from_profile with 2 x h_step) and the 8-Hz profiles kept -- as the submission's --speed-from full-res speed; NF_OUT / NFT_OUT"""
    import json
    LV.PX = 2.0
    tcl = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}; test = k in tcl
    d8 = np.load(f"{FVT_RAW if test else FV_RAW}/{k}.npz"); T8 = np.asarray(d8['times'], float); n = len(d8['dts']); dt8 = np.asarray(d8['dts'], float)
    h8 = np.asarray(d8['h_step'], float); hfit = 2 * np.asarray(CC.hz_rescue(d8)[0], float); I3 = np.eye(3); acc = dts_ = None; npair = np.zeros(n, int)
    for f_i, buf, K, h, t in fv_frames(k, 1, 1, fps=1000.0, half=False):
        j = int(np.searchsorted(T8, t, 'left')) - 1
        if j < 0 or j >= n: continue
        H = buf[-1].shape[0]
        if acc is None: acc, dts_ = np.zeros((n, H)), np.zeros((n, H)); assert abs(K[1, 1] - 2 * float(d8['fy'])) < 0.5, (k, K[1, 1], d8['fy'])
        if os.environ.get('NF_BAND'):   # owner 25.10 ('faster'): DIS only on the rows from 32 px above the water band (v >= WATER_V_MIN under the
            y0 = max(int(LV.PROFILE_TOP * H), int(min(hfit[j], 2 * h8[j]) + LV.WATER_V_MIN * K[1, 1]) - 32)   # higher of the two horizons) down
            p = np.full(H, np.nan) if y0 >= int(LV.PROFILE_BOT * H) - 16 else band_profile(buf[0], buf[1], K, y0)
        else: p = LV.water_profile(buf[0], buf[1], K, I3).astype(float)
        rows = np.arange(H); v = (rows - hfit[j]) / K[1, 1]
        m = (v >= LV.WATER_V_MIN) & (v <= LV.WATER_V_MAX) & (rows < LV.BOTTOM_CUT * H) & np.isfinite(p)
        if m.sum() < 10: continue
        c = np.linalg.lstsq(np.c_[np.ones(m.sum()), v[m] ** 2], p[m], rcond=None)[0][0]; ok = np.isfinite(p); dt = FV_TBUF[-1] - FV_TBUF[-2]
        acc[j, ok] += p[ok] - c; dts_[j, ok] += dt; npair[j] += 1
    with np.errstate(all='ignore'):
        P = np.where(dts_ >= 0.5 * dt8[:, None], acc / dts_ * dt8[:, None], np.nan)
    fy2 = 2 * float(d8['fy']); r = {q: d8[q] for q in d8.files}
    r.update(spd=np.array([np.nan if not np.isfinite(h8[j]) or (x := LV.speed_from_profile(P[j], 2 * h8[j], fy2)) is None else x for j in range(n)]),
             quality=np.array([LV.profile_fit_quality(P[j], 2 * h8[j], fy2) if np.isfinite(h8[j]) else np.nan for j in range(n)]), native_full_pairs=npair)
    out = os.environ.get('NFT_OUT' if test else 'NF_OUT', 'runs/native/fullt/raw' if test else 'runs/native/full/raw'); os.makedirs(out, exist_ok=True)   # NF_BAND: DIS on the band rows only
    np.savez(f'{out}/{k}.npz.tmp.npz', **r); os.replace(f'{out}/{k}.npz.tmp.npz', f'{out}/{k}.npz')
    return k, dict(test=test, pairs=float(np.median(npair)), water=float(np.isfinite(r['spd']).mean()))
if __name__ == '__main__' and '--native-full' in sys.argv:
    # --native-full: native_full_job on NF_CLIPS (comma list, required)
    from multiprocessing import Pool
    with Pool(int(os.environ.get('NF_POOL', '3'))) as p:
        for k, o in p.imap_unordered(native_full_job, os.environ['NF_CLIPS'].split(',')): print(f"{k} {'test' if o['test'] else 'dev '}: native full-res pairs per 8 Hz step {o['pairs']:.0f}, steps with water {o['water']:.2f}", flush=True)
    sys.exit()
