"""Owner 27.10 ('your own model based on the flow, trained on train, on Kaggle GPU'): the per-step inputs and targets for a learned
speed-SHAPE model on wide clips. Per 8-Hz step: the vertical-flow profile of the raw (half-res rows, the tool's derotation) resampled on 24
angles below the speed horizon v in [0.04, 0.35] (h_step, fy), divided by v^2 (constant for a perfect water law) and by the clip median of
the tool's water s, asinh-compressed; + the tool's fit quality, pitch rate (dy_rot / dt / fy), yaw rate, horizon-row change, the share of
finite band rows. Target (dev only): log GPS speed per step minus its clip median (the metric sees no clip scale). Usage: flowmodel_build.py OUT"""
import sys, io, os, json, contextlib, numpy as np
OUT = sys.argv[1]
sys.argv += ['--best']
sys.path[:0] = ['tools/lookout_combo', 'tools', 'data/release/starter']
from multiprocessing import Pool
with contextlib.redirect_stdout(io.StringIO()): import error_breakdown as EB
LV = EB.LV; VG = np.linspace(0.04, 0.35, 24)
TCL = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
def feats(d):
    n = len(d['dts']); P = d['profile'].astype(np.float32)[:n]; h = np.asarray(d['h_step'], float)[:n]; fy = float(d['fy']); dt = np.asarray(d['dts'], float)[:n]
    rows = np.arange(P.shape[1]); s = np.asarray(d['spd'], float)[:n]; smed = np.nanmedian(s[np.isfinite(s) & (s > 0)]) if np.isfinite(s).any() else 1.0
    X = np.full((n, len(VG) + 5), np.nan, np.float32)
    for i in range(n):
        if not np.isfinite(h[i]): continue
        v = (rows - h[i]) / fy; m = np.isfinite(P[i]) & (v > 0.02) & (v < 0.4) & (rows < LV.BOTTOM_CUT * P.shape[1])
        if m.sum() < 10: continue
        q = np.interp(VG, v[m], P[i, m], left=np.nan, right=np.nan) / VG ** 2 / smed
        X[i, :len(VG)] = np.arcsinh(q); X[i, len(VG)] = float(np.isfinite(q).mean())
    X[:, len(VG) + 1] = np.nan_to_num(np.asarray(d['quality'], float)[:n], nan=1.0)
    X[:, len(VG) + 2] = np.nan_to_num(np.asarray(d['dy_rot'], float)[:n]) / dt / fy
    X[:, len(VG) + 3] = np.nan_to_num(np.asarray(d['dyaw'], float)[:n]) / dt
    X[:, len(VG) + 4] = np.r_[0.0, np.diff(np.nan_to_num(h))] / fy
    return X
def job(k):
    test = k in TCL; d = np.load(f"runs/{'test_v5' if test else 'final_v5'}/raw/{k}.npz"); n = len(d['dts']); X = feats(d); y = np.full(n, np.nan, np.float32)
    if not test:
        T, dts, yo, v, gh, gv, sc = EB.clip(k, parts_only=True); gv = np.asarray(gv, float)[:n]
        ok = gv > 0.3; y[:len(gv)] = np.where(ok, np.log(np.maximum(gv, 1e-3)) - np.median(np.log(gv[ok])), np.nan)
    sp = 'test' if test else EB.CC.cl[k]['split']
    np.savez_compressed(f'{OUT}/{k}.npz', X=X, y=y, t=np.asarray(d['times'], float)[1:n + 1], dts=np.asarray(d['dts'], float)[:n], split=sp)
    return k, sp, n, float(np.isfinite(X[:, 0]).mean())
if __name__ == '__main__':
    ks = sorted(k for k, c in EB.CC.cl.items() if c['camera_mode'] == 'wide') + sorted(k for k, c in TCL.items() if c['camera_mode'] == 'wide')
    with Pool(3) as p: R = p.map(job, ks)
    for sp in ('train', 'val', 'test'): print(sp, sum(r[1] == sp for r in R), 'clips', sum(r[2] for r in R if r[1] == sp), 'steps, finite profile share', round(np.mean([r[3] for r in R if r[1] == sp]), 3))
