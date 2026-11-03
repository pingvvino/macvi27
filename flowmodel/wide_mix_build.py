"""Stage 11: the wide clips' speed of the base file WM_BASE mixed with the temporal CNN: per step speed^(1-w) * m^w, m = exp of the
CNN's log prediction (mean over its checkpoints, and over the models of several comma-separated PRED_NPZ) scaled to the speed's
median; w = W, or with WM_CONF_S0 = s0 the per-step weight W * s0^2 / (s0^2 + spread^2) of the models' spread. Every step keeps its
direction (reversing stays), stops stay 0, the frame-0 step like the first; other cameras unchanged. Writes OUT_JSON + RUN_DIR/predictions.
Usage: WM_BASE=base.json [WM_CONF_S0=s0] wide_mix_build.py PRED_NPZ[,NPZ..] W OUT_JSON RUN_DIR"""
import sys, os, json, numpy as np
PRED, W, OUTJ, RUN = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
BASE = os.environ['WM_BASE']   # the base file (reproduce.sh: the stage-10 output)
SUB = json.load(open(BASE)); TCL = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
E = {}
for f_ in PRED.split(','):
    z = np.load(f_); per = {}
    for key in z.files:
        k, ep = key.split('|'); per.setdefault(k, []).append(np.asarray(z[key], float))
    for k, ps in per.items(): E.setdefault(k, []).append(np.mean(ps, 0))
NM = len(PRED.split(',')); SPR = {k: np.std([x - np.median(x) for x in v], 0) for k, v in E.items() if len(v) == NM}   # seeds' spread
E = {k: np.mean(v, 0) for k, v in E.items() if len(v) == NM}
S0 = float(os.environ['WM_CONF_S0']) if os.environ.get('WM_CONF_S0') else None
NEW = {}
for k in sorted(SUB):
    if TCL[k]['camera_mode'] != 'wide': NEW[k] = SUB[k]; continue
    P0 = np.asarray(SUB[k]['positions'], float); dQ = np.diff(P0, axis=0); dt = np.diff(np.asarray(SUB[k]['times'], float))
    sp = np.linalg.norm(dQ, axis=1)[1:] / dt[1:]; mv = sp > 1e-9; p = E[k]; assert len(p) == len(sp), (k, len(p), len(sp))
    m = np.exp(p - np.median(p)); r = sp / np.median(sp[mv])
    Wi = W * S0 ** 2 / (S0 ** 2 + SPR[k] ** 2) if S0 else W
    f = np.where(mv, np.where(mv, r, 1.0) ** (1 - Wi) * (m * 1.0 / np.median(m)) ** Wi / np.where(mv, r, 1.0), 1.0); f = np.r_[f[0], f]
    Pn = np.vstack([P0[:1], P0[:1] + np.cumsum(dQ * f[:, None], 0)]); assert np.isfinite(Pn).all() and Pn.shape == P0.shape
    NEW[k] = dict(SUB[k], positions=Pn.tolist())
    stops = float((~mv).mean())
    print(f'  {k}: stops {stops:.1%}, factor p5/p50/p95 {np.percentile(f, 5):.2f}/{np.median(f):.2f}/{np.percentile(f, 95):.2f}')
json.dump(NEW, open(OUTJ, 'w')); os.makedirs(f'{RUN}/predictions', exist_ok=True)
for k, v in NEW.items(): json.dump(v, open(f'{RUN}/predictions/{k}.json', 'w'))
print('changed', sum(NEW[k] != SUB[k] for k in SUB), 'of', len(SUB), '->', OUTJ)
