"""Speed stages of the final chain on the saved tool outputs (run from the work layout of reproduce.sh: data/, runs/).
  --u-test       stage 5 (a): the U15 trust override -- wide clips whose 15-s block water is smooth and persistent get alpha 1;
                 prints TV_ALPHA_CLIPS for (b). env U_RULE (step:B:J:Smin:A:AC).
  --test-variant stage 5 (b): every wide clip of SUB rebuilt with its alpha (TV_ALPHA_CLIPS; 'base' = as in SUB) -> TV_OUT.
  --native-test  stage 7: every wide clip of SUB rebuilt with its alpha and the speed VALUES from the native-resolution water
                 (NT_SPEED_ONLY raws; alpha, Z and reversing still from the 8-Hz water) -> NT_OUT.
  --fv-test-var  stages 6 / 8: the flir-VIS steps of SUB rescaled by the slow-down rule on another water (FVTV
                 'label=A+B@-@VMAX@cond[@HZ]', the '-' and 'cond' fields fixed: A where A's 20-s window obeys the water law, else B) relative to the rule on the
                 8-Hz water that SUB was built with -> FVTV_OUT (FVTV_WRITE = the label written)."""
import sys, os, io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()): import course_combo as CC
import lookout_vo as LV

GAIN_U = 'step:15:0.06:0:1:0.5'   # U15


def wide_rules(T, d, v, a):
    """The wide speed rules on speed v of raw d: Z (CC.stop_rel, law fit 0.5, 20-s windows) when the trust a < 0.5, then reversing
    from the water with 40-s ramps (reversing_rule / reversing_ramp)."""
    argv = sys.argv[:]; from reversing_rule import windows; from reversing_ramp import ramp; sys.argv[:] = argv   # they reset sys.argv
    n = len(v); za, zg, zw = 0.5, 0.5, 20.0
    rs = np.asarray(CC.stop_rel(T, dict(d), win=zw, min_good=zg)[0], float)[:n] if a < za else np.ones(n)
    tw, st, keep, rel = windows(T, d)
    u = ramp(tw, st, keep, rel, v, 40.0, True) if keep.any() else v.copy()
    return np.where(u > 0, u * rs, u)


def gate_spread(T, dts, spd):
    """The trust gate's signal S: std of the block medians of the log water speed (blocks LV.ADAPT_BLOCK_S)."""
    t = np.asarray(T[1:], float); w = np.array([s / dt if np.isfinite(s) and s > 0 else np.nan for s, dt in zip(spd, dts)]); lw = np.log(w)
    B = LV.ADAPT_BLOCK_S; bl = [np.nanmedian(lw[(t >= s0) & (t < s0 + B)]) for s0 in np.arange(t[0], t[-1] - B / 2, B)
                               if np.isfinite(lw[(t >= s0) & (t < s0 + B)]).sum() > B * 2]
    return float(np.std(bl)) if len(bl) > 2 else 0.0


def block_series(T, dts, spd, B=10.0):
    """B-s block medians of the positive water speed (log), block starts; blocks with <= 2 B valid steps -> NaN."""
    t = np.asarray(T[1:], float); wr = np.asarray(spd, float) / np.asarray(dts, float); st = np.arange(t[0], t[-1] - B / 2, B)
    lw = np.array([np.log(np.nanmedian(np.where(wr[m] > 0, wr[m], np.nan))) if (np.isfinite(wr[m]) & (wr[m] > 0)).sum() > 2 * B else np.nan
                   for m in [(t >= x) & (t < x + B) for x in st]])
    return st, lw


def u_features(T, dts, d):
    """Water features of a wide clip for the U rule: the tool's alpha and S; for 15-s blocks the median |change| of consecutive
    valid block log medians (jit15), the valid block count (nb15) and the lag-1 persistence (ac15)."""
    out = {}; _, out['alpha'] = forced_alpha_speed(T, dts, d, 'gate'); out['S'] = gate_spread(T, dts, d['spd'])
    B = 15.0
    st, lw = block_series(T, dts, d['spd'], B); ok = np.isfinite(lw); x = lw[ok]
    out[f'jit{B:g}'] = float(np.median(np.abs(np.diff(x)))) if ok.sum() > 2 else np.nan; out[f'nb{B:g}'] = int(ok.sum())
    out[f'ac{B:g}'] = float(np.corrcoef(x[:-1] - x.mean(), x[1:] - x.mean())[0, 1]) if ok.sum() > 4 else np.nan
    return out


def u_alpha(o, spec):
    """Alpha of the rule 'step:B:J:Smin:A:AC' on the features o: A where jit_B <= J, S >= Smin and ac_B >= AC (needs > 4 valid
    blocks); never below the tool's alpha."""
    _, B, *p = spec.split(':'); B = int(B); x = [float(v) for v in p]
    if not (o[f'nb{B}'] > 4 and np.isfinite(o[f'jit{B}'])): return o['alpha']
    j = o[f'jit{B}']
    a = x[2] if (j <= x[0] and o['S'] >= x[1] and (len(x) < 4 or (np.isfinite(o[f'ac{B}']) and o[f'ac{B}'] >= x[3]))) else 0.0
    return max(o['alpha'], a)


def forced_alpha_speed(T, dts, d, a):
    """The tool's adaptive water speed with the trust alpha forced to a ('gate' = as the tool decides) -> (v, alpha used)."""
    sp = [None if not np.isfinite(x) else x for x in d['spd']]; q = d['quality'] if 'quality' in (d if isinstance(d, dict) else d.files) else None
    if a == 'gate': v, al = LV.adaptive_speed(T, dts, sp, q); return np.asarray(v, float), al
    t0, t1 = LV.ADAPT_T0, LV.ADAPT_T1; LV.ADAPT_T0, LV.ADAPT_T1 = -1.0, -0.5
    try: v1, _ = LV.adaptive_speed(T, dts, sp, q)
    finally: LV.ADAPT_T0, LV.ADAPT_T1 = t0, t1
    LV.ADAPT_T0, LV.ADAPT_T1 = 1e9, 1e9 + 1
    try: vc, _ = LV.adaptive_speed(T, dts, sp, q)
    finally: LV.ADAPT_T0, LV.ADAPT_T1 = t0, t1
    a = float(a); return np.asarray(vc, float) ** (1 - a) * np.asarray(v1, float) ** a, a


def test_variant_clip(p, k, alk='base', dspeed=None):
    """A wide clip p (times / positions; course 1/4 BA x0.9 + 1/4 VGGT x0.8 + 1/2 Omega) rebuilt: the course is recovered from p, the
    |speed| replaced by the water with trust alk forced (its rules applied, in the clip's scale; 'base' = as in p). dspeed: a raw
    whose speed values replace the 8-Hz ones. -> new positions list"""
    P = np.asarray(p['positions'], float); dP = np.diff(P[:, :2], axis=0); n = len(dP); L = np.hypot(*dP.T)
    d = np.load(f'runs/test_v5/raw/{k}.npz'); B = np.asarray(d['dyaw'], float); fill = lambda y: B if y is None else np.where(np.isfinite(y), y, B)
    dyb = 0.25 * 0.9 * B + 0.25 * 0.8 * fill(CC.vggt_steps(k, n, 'kf4_w17')) + 0.5 * fill(CC.vggt_steps(k, n, 'kf4_w17', src='runs/vggt_omega'))
    y = np.cumsum(dyb); mv = L > 1e-9; j0 = int(np.flatnonzero(mv)[0]); psi0 = np.arctan2(dP[j0, 0], dP[j0, 1]) - y[j0]
    if np.dot(dP[j0], [np.sin(y[j0] + psi0), np.cos(y[j0] + psi0)]) < 0: psi0 += np.pi
    ub = dP[:, 0] * np.sin(y + psi0) + dP[:, 1] * np.cos(y + psi0); cr = np.abs(dP[:, 0] * np.cos(y + psi0) - dP[:, 1] * np.sin(y + psi0))
    assert np.max(cr[mv] / L[mv]) < 1e-3, (k, 'course not reproduced')
    if alk != 'base':
        a_ = np.abs(ub); va, af = forced_alpha_speed(d['times'], np.asarray(d['dts'], float), d if dspeed is None else dspeed, alk)
        u = wide_rules(d['times'], d, va, af)
        u = u * np.asarray(d['dts'], float)
        ub = u * np.median(a_[a_ > 1e-9]) / np.median(np.abs(u)[np.abs(u) > 1e-9])
    hh = psi0 + np.cumsum(1.0 * dyb); Xn = np.vstack([P[:1, :2], P[:1, :2] + np.cumsum(np.c_[ub * np.sin(hh), ub * np.cos(hh)], 0)])
    return np.c_[Xn, P[:, 2]].tolist()


def wide_clips():
    import json
    return sorted(c['key'] for c in json.load(open('data/test/release_manifest.json'))['clips'] if c['camera_mode'] == 'wide')


if __name__ == '__main__' and '--u-test' in sys.argv:
    al = {}
    for k in wide_clips():
        d = np.load(f'runs/test_v5/raw/{k}.npz'); o = u_features(d['times'], np.asarray(d['dts'], float), d); al[k] = u_alpha(o, os.environ['U_RULE'])
        print(f"{k}: tool alpha {o['alpha']:.2f} -> {al[k]:.3f}")
        if al[k] <= o['alpha'] + 1e-9: del al[k]               # unchanged clips keep their speed (Z, reversing) untouched
    print('TV_ALPHA_CLIPS=' + ','.join(f'{k}:{a:.3f}' for k, a in al.items()))
    sys.exit()

if __name__ == '__main__' and '--test-variant' in sys.argv:
    import json
    sub = json.load(open(os.environ['SUB'])); out = dict(sub)
    alc = dict(x.split(':') for x in os.environ.get('TV_ALPHA_CLIPS', '').split(',') if x)
    for k in wide_clips():
        alk = alc.get(k, 'base'); out[k] = dict(sub[k], positions=test_variant_clip(sub[k], k, alk)); print(f'{k}: speed {alk}')
    json.dump(out, open(os.environ['TV_OUT'], 'w')); print('written', os.environ['TV_OUT']); sys.exit()

if __name__ == '__main__' and '--native-test' in sys.argv:
    import json
    sub = json.load(open(os.environ['SUB'])); out = dict(sub)
    for k in wide_clips():
        d = np.load(f'runs/test_v5/raw/{k}.npz'); o = u_features(d['times'], np.asarray(d['dts'], float), d)
        a = u_alpha(o, GAIN_U); al = a if a > o['alpha'] + 1e-9 else o['alpha']
        zs = np.load(f"{os.environ['NT_SPEED_ONLY']}/{k}.npz"); dsp = {q: d[q] for q in d.files}; dsp['spd'], dsp['quality'] = zs['spd'], zs['quality']
        out[k] = dict(sub[k], positions=test_variant_clip(sub[k], k, al, dspeed=dsp)); print(f"{k}: alpha {o['alpha']:.2f} -> {al:.2f}", flush=True)
    json.dump(out, open(os.environ['NT_OUT'], 'w')); print('written', os.environ['NT_OUT'])
    sys.exit()


def fv_combine(ra, tb, rb, T, n):
    """Two slow-down rule profiles (1 with a closed gate): ra where ra's 20-s window obeys the water law (R^2 >= 0.9 in its window
    table tb), else rb."""
    ts = np.asarray(T, float)[1:n + 1]; okA = np.zeros(n, bool)
    for c_, _, r2_, *_ in (tb if tb is not None and len(tb) else []):
        if np.isfinite(r2_) and r2_ >= 0.9: okA |= (ts >= c_ - 10.0) & (ts < c_ + 10.0)
    return np.where(okA, ra, rb)


FV_STOP_OK = os.environ.get('FV_STOP_OK', 'rescue')   # the stop-ok guard: 'rescue' = only where the horizon rescue fired, 'all'
def fv_stop_ok(hv):
    return {'rescue': 'bow_row' in hv, 'all': True}[FV_STOP_OK]


def fv_rule_rel(rd, k, T, n, vm, hz='best'):
    """The flir-VIS slow-down rule (hard gate, soft profile, horizon rescue, stop-ok guard) on the water of raw dir rd -> (relative
    speed per step, 1 with a closed gate; the window table). vm: the water band's lower edge, hz: the speed horizon."""
    LV.WATER_V_MAX = vm; LV.WATER_V_MIN = 0.04
    try:
        hv = CC.rescue_raw(np.load(f'{rd}/{k}.npz')); hv = dict(hv) if not isinstance(hv, dict) else hv; sv = fv_stop_ok(hv)
        assert np.array_equal(np.asarray(hv['times'], float), np.asarray(T, float)), f'{k}: {rd} steps differ'
        r0, tb0, fs = CC.stop_rel(T, hv, win=20.0, min_good=0.5, hz=hz, stop_ok=sv)
        if fs == 0: return np.ones(n), tb0
        r, tb, _ = CC.stop_rel(T, hv, win=20.0, min_good=0.5, soft=True, hz=hz, stop_ok=sv)
        return np.asarray(r, float)[:n], tb
    finally: LV.WATER_V_MAX = 0.35


if __name__ == '__main__' and '--fv-test-var' in sys.argv:
    import json
    sub = json.load(open(os.environ['SUB']))
    ks = sorted(c['key'] for c in json.load(open('data/test/release_manifest.json'))['clips'] if c['camera_mode'] == 'flir-VIS')
    VAR = [(lab, f[0], float(f[2]), f[4] if len(f) > 4 and f[4] else 'best')
           for lab, f in ((lab, a.split('@')) for lab, a in (x.split('=', 1) for x in os.environ['FVTV'].split(';') if x))]
    def rel_of(T, h, sok, hz='best'):
        """the rule's profile and window table on raw h; (None, None) with a closed gate"""
        if CC.stop_rel(T, h, win=20.0, min_good=0.5, hz=hz, stop_ok=sok)[2] == 0: return None, None
        r, tb, _ = CC.stop_rel(T, h, win=20.0, min_good=0.5, soft=True, hz=hz, stop_ok=sok)
        return np.asarray(r, float), tb
    def water_rel(rd, k, T, n, vm, hz='best'):
        LV.WATER_V_MAX = vm; LV.WATER_V_MIN = 0.04
        hv = CC.rescue_raw(np.load(f'{rd}/{k}.npz')); hv = dict(hv) if not isinstance(hv, dict) else hv; sv = fv_stop_ok(hv)
        assert np.array_equal(np.asarray(hv['times'], float), T), f'{k}: {rd} steps differ from SUB'
        r, tb = rel_of(T, hv, sv, hz=hz)
        if tb is None: tb = CC.stop_rel(T, hv, win=20.0, min_good=0.5, hz=hz, stop_ok=sv)[1]
        r = np.ones(n) if r is None else r[:n]
        return r, tb
    OUTSUB = dict(sub)
    for k in ks:
        LV.WATER_V_MAX = 0.35
        h = CC.rescue_raw(np.load(f'runs/fvt_hstep/raw/{k}.npz')); h = dict(h) if not isinstance(h, dict) else h; sok = 'bow_row' in h
        T = np.asarray(h['times'], float); P = np.asarray(sub[k]['positions'], float); dQ = np.diff(P, axis=0); n = len(dQ)
        assert len(P) == len(T) and np.allclose(np.asarray(sub[k]['times'], float), T), k
        thr_, CC.STOP_THR = CC.STOP_THR, 0.5              # the profile SUB was built with: 8-Hz water, gate threshold 0.5
        r0 = rel_of(T, h, sok)[0]; r0 = np.ones(n) if r0 is None else r0[:n]; CC.STOP_THR = thr_
        line = f'{k} rescue {sok!s:5s}'
        for lab, rd, vm, hz in VAR:
            ra, tb = water_rel(rd.split('+')[0], k, T, n, vm, hz=hz); rb, _ = water_rel(rd.split('+')[1], k, T, n, vm, hz=hz)
            r1 = fv_combine(ra, tb, rb, T, n)
            ch = float(np.max(np.abs(r1 / r0 - 1)))
            if ch <= 1e-9: line += f' | {lab}: same'
            else:
                Pn = np.vstack([P[:1], P[:1] + np.cumsum(dQ * (r1 / r0)[:, None], 0)])
                if lab == os.environ.get('FVTV_WRITE'): OUTSUB[k] = dict(sub[k], positions=Pn.tolist())
                line += f' | {lab}: changed ({ch:.2f}, slow {(r1 < 0.95).mean():.2f})'
        LV.WATER_V_MAX = 0.35
        print(line, flush=True)
    json.dump(OUTSUB, open(os.environ['FVTV_OUT'], 'w')); print('written', os.environ['FVTV_OUT'], 'changed:', sorted(k for k in OUTSUB if OUTSUB[k] is not sub[k]), flush=True)
    sys.exit()
