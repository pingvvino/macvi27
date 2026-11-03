"""All cameras (owner 27.10: 'on all clips'): the per-step profile features (flowmodel_build.feats) for every dev clip and every test clip,
+ a 6-channel camera one-hot (wide, flir-VIS, zoom, sionyx, flir-IR, infrared). Wide / flir-VIS copied from prof_all (tool water; flir-VIS
horizon hz_best after rescue_raw); other cameras from runs/final_v4 / test_v4 raws, which have no speed horizon -> the water-plane horizon
from the profile itself: per 80-step window sqrt(median profile) fitted by a line in the row over the band, the zero = the horizon (two
passes), median over 3 windows. Usage: flowmodel_build_all.py PROF_ALL OUT"""
import sys, io, os, json, contextlib, numpy as np
PALL, OUT = sys.argv[1], sys.argv[2]
sys.argv += ['--best']
sys.path[:0] = ['tools/lookout_combo', 'tools', 'data/release/starter', os.path.dirname(os.path.abspath(__file__))]
from multiprocessing import Pool
with contextlib.redirect_stdout(io.StringIO()): import error_breakdown as EB
import flowmodel_build as FB
LV = EB.LV; CAMS = ['wide', 'flir-VIS', 'zoom', 'sionyx', 'flir-IR', 'infrared']
TCL = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
CAM = {**{k: c['camera_mode'] for k, c in EB.CC.cl.items()}, **{k: c['camera_mode'] for k, c in TCL.items()}}
def plane_horizon(P, fy):
    n, H = P.shape; rows = np.arange(H); hw = []
    for w0 in range(0, n, 80):
        Pm = np.nanmedian(P[w0:w0 + 80], 0); h = 0.4 * H
        for _ in range(2):
            v = (rows - h) / fy; m = np.isfinite(Pm) & (Pm > 0) & (v >= 0.04) & (v <= 0.35) & (rows < 0.97 * H)
            if m.sum() < 20: h = np.nan; break
            a, b = np.polyfit(rows[m].astype(float), np.sqrt(Pm[m]), 1)
            h = -b / a if a > 0 else np.nan
            if not np.isfinite(h) or h < 0 or h > 0.9 * H: h = np.nan; break
        hw.append(h)
    hw = np.array(hw); hs = np.array([np.nanmedian(hw[max(0, i - 1):i + 2]) if np.isfinite(hw[max(0, i - 1):i + 2]).any() else np.nan for i in range(len(hw))])
    if np.isfinite(hs).any(): hs = np.where(np.isfinite(hs), hs, np.nanmedian(hs))
    return np.repeat(hs, 80)[:n]
def onehot(k, n): o = np.zeros((n, len(CAMS)), np.float32); o[:, CAMS.index(CAM[k])] = 1.0; return o
def job(k):
    test = k in TCL; d = dict(np.load(f"runs/{'test_v4' if test else 'final_v4'}/raw/{k}.npz")); n = len(d['dts'])
    d['h_step'] = plane_horizon(d['profile'].astype(float)[:n], float(d['fy'])); X = FB.feats(d); y = np.full(n, np.nan, np.float32)
    if not test:
        T, dts, yo, v, gh, gv, sc = EB.clip(k, parts_only=True); gv = np.asarray(gv, float)[:n]
        ok = gv > 0.3; y[:len(gv)] = np.where(ok, np.log(np.maximum(gv, 1e-3)) - np.median(np.log(gv[ok])), np.nan)
    sp = 'test' if test else EB.CC.cl[k]['split']
    np.savez_compressed(f'{OUT}/{k}.npz', X=np.c_[X, onehot(k, n)], y=y, t=np.asarray(d['times'], float)[1:n + 1], dts=np.asarray(d['dts'], float)[:n], split=sp)
    return k, CAM[k], sp, float(np.isfinite(X[:, 0]).mean()), float(np.nanmedian(d['h_step']) / d['profile'].shape[1])
if __name__ == '__main__':
    other = sorted(k for k in CAM if CAM[k] not in ('wide', 'flir-VIS'))
    with Pool(3) as p: R = p.map(job, other)
    for r in R: print(f'{r[0]} {r[1]:8s} {r[2]:5s} finite profile {r[3]:.2f} horizon {r[4]:.2f} H')
    for f in sorted(os.listdir(PALL)):                    # wide / flir-VIS: the 29 features (drop prof_all's flir flag) + one-hot
        if f.startswith('clip_') and f.endswith('.npz'):
            z = np.load(f'{PALL}/{f}'); k = f[:-4]
            np.savez_compressed(f'{OUT}/{f}', X=np.c_[z['X'][:, :-1], onehot(k, len(z['X']))], y=z['y'], t=z['t'], dts=z['dts'], split=z['split'])
    print('files', len([f for f in os.listdir(OUT) if f.startswith('clip_')]))
