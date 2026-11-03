"""Stage 10: the wide low-flow correction. For every wide clip of BASE (the stage-9 file, frame 0 included) the water speed
is corrected with course_combo.WATER_LOWPOW = LOWPOW (P0:beta:F) and the steps are scaled by the ratio of the trusted
water speed (alpha as in stage 7: U15 over the tool's) with / without the correction; speed values from the native water
(runs/native/fullt/raw) where it exists. Frame 0 is re-extrapolated from the first step. -> OUT"""
import sys, os, json, io, contextlib, warnings, numpy as np
warnings.filterwarnings('ignore')
_a = sys.argv[:]; sys.argv = [sys.argv[0]]
with contextlib.redirect_stdout(io.StringIO()): import speed_rules as EB
sys.argv = _a
F0 = json.load(open(os.environ['BASE'])); out = json.loads(json.dumps(F0))
tcl = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
for k in sorted(x for x, c in tcl.items() if c['camera_mode'] == 'wide'):
    d = np.load(f'runs/test_v5/raw/{k}.npz'); T = np.asarray(d['times'], float); dts = np.asarray(d['dts'], float); n = len(dts)
    Tb = np.asarray(F0[k]['times'], float); P0 = np.asarray(F0[k]['positions'], float)[1:]; assert np.array_equal(Tb[1:], T), k
    o = EB.u_features(T, dts, d); a = EB.u_alpha(o, EB.GAIN_U); A = a if a > o['alpha'] + 1e-9 else o['alpha']
    nat = os.path.exists(f'runs/native/fullt/raw/{k}.npz'); dv = d
    if nat:
        zs = np.load(f'runs/native/fullt/raw/{k}.npz'); dv = {q: d[q] for q in d.files}; dv['spd'], dv['quality'] = zs['spd'], zs['quality']
    ctrl = EB.CC.water_speed_all(d, np.asarray(d['h_step'], float))
    EB.CC.WATER_LOWPOW = tuple(float(x) for x in os.environ['LOWPOW'].split(':'))
    w2 = EB.CC.water_speed_all(d, np.asarray(d['h_step'], float)); EB.CC.WATER_LOWPOW = None
    q = w2[:n] / ctrl[:n]; okq = np.isfinite(q) & (q > 0)
    dv2 = dict({x: dv[x] for x in (dv.files if hasattr(dv, 'files') else dv)}); s0 = np.asarray(dv['spd'], float).copy(); s0[:n] = np.where(okq, s0[:n] * np.where(okq, q, 1.0), s0[:n]); dv2['spd'] = s0
    r = np.asarray(EB.forced_alpha_speed(T, dts, dv2, A)[0], float) / np.asarray(EB.forced_alpha_speed(T, dts, dv, A)[0], float); r[~np.isfinite(r)] = 1.0
    dP = np.diff(P0[:, :2], axis=0) * r[:, None]
    P2 = np.c_[np.vstack([P0[:1, :2], P0[:1, :2] + np.cumsum(dP, 0)]), P0[:, 2]]
    f0 = lambda P: P[0] - (P[1] - P[0]) / (T[1] - T[0]) * (T[0] - Tb[0])
    assert np.array_equal(f0(P0), np.asarray(F0[k]['positions'], float)[0]), k   # the frame-0 rule reproduces BASE
    print(f'{k}: alpha {A:.2f}, changed steps {np.mean(np.abs(r - 1) > 1e-6):.2f}, values {"native" if nat else "8 Hz"}')
    if np.array_equal(P2, P0): continue
    out[k]['positions'] = np.vstack([f0(P2), P2]).tolist(); print(f'{k}: written (max shift {np.abs(P2 - P0).max():.2f} m)')
json.dump(out, open(os.environ['OUT'], 'w'), separators=(',', ':'), allow_nan=False); print('wrote', os.environ['OUT'])
