"""Stage 12: the flir-VIS slow-down rule (stage 8: SLR water cond + slrsel:10) with windows every STRIDE s instead of 20 s.
For every flir-VIS clip the steps of BASE930 (the stage-9 file) are rescaled by rel_STRIDE / rel_20 where the fine strides
agree (mean |rel_STRIDE - rel_10| < AGREE), else left as they are; the other clips are copied from BASE unchanged.
Usage: BASE930=09.json LOOKOUT_STOP_THR=0.635 FV_STOP_OK=all [AGREE=0.05] python flirvis_stride.py CODE_DIR STRIDE BASE.json OUT.json"""
import sys, os, json, numpy as np
SP, STRIDE, BASE, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
AGREE = float(os.environ.get('AGREE', '0.05'))
sys.path[:0] = [f'{SP}/tools/lookout_combo', f'{SP}/tools']; sys.argv = sys.argv[:1]
import speed_rules as EB
tst = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
S = json.load(open(os.environ['BASE930']))
B = json.load(open(BASE)); out = json.loads(json.dumps(B)); base, slr = 'runs/fvt_hstep/raw', 'runs/dk_slr2_fvt/raw'
def rule(k, T, n, w):
    os.environ['LOOKOUT_STOP_STRIDE'] = f'{w:g}'; ra, ta = EB.fv_rule_rel(slr, k, T, n, 0.35, 'slrsel:10')
    r = np.asarray(EB.fv_combine(ra, ta, EB.fv_rule_rel(base, k, T, n, 0.35, 'slrsel:10')[0], T, n), float); os.environ['LOOKOUT_STOP_STRIDE'] = '20'; return r
changed = []
for k in sorted(x for x, c in tst.items() if c['camera_mode'] == 'flir-VIS'):
    d = np.load(f'{base}/{k}.npz'); n = len(d['dts']); T = np.asarray(d['times'], float)
    r20, rs, r10 = rule(k, T, n, 20.0), rule(k, T, n, STRIDE), rule(k, T, n, 10.0); ag = float(np.mean(np.abs(rs - r10)))
    why = 'disagree' if ag >= AGREE else 'stride'
    if why != 'stride': rs = r20
    Ts = np.asarray(S[k]['times'], float); P = np.asarray(S[k]['positions'], float); off = len(Ts) - 1 - n; dP = np.diff(P[:, :2], axis=0)
    assert np.array_equal(Ts, np.asarray(B[k]['times'], float)) and np.allclose(P, np.asarray(B[k]['positions'], float)), f'{k}: BASE differs from BASE930'
    f = np.r_[np.ones(off), rs / np.where(r20 > 0, r20, 1.0)]
    Pn = np.c_[np.vstack([P[:1, :2], P[:1, :2] + np.cumsum(dP * f[:, None], 0)]), P[:, 2:]]
    sh = float(np.max(np.hypot(*(Pn[:, :2] - P[:, :2]).T)))
    if sh > 1e-9: out[k]['positions'] = Pn.tolist(); changed.append(k)
    line = f'{k}: |rel{STRIDE:g}-rel10| {ag:.3f} -> {why:8s}; steps {n}, rel changed on {int((np.abs(rs - r20) > 1e-9).sum())} steps, max shift {sh:7.2f} m'
    print(line, flush=True)
json.dump(out, open(OUT, 'w')); import hashlib
print(f'{OUT}: {len(out)} clips, changed {changed}, {os.path.getsize(OUT)} B, sha256 {hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:12]}; unchanged clips bit-equal to BASE: '
      f'{all(out[k] == B[k] for k in B if k not in changed)}')
