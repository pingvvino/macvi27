"""Stage 1 of the final chain (test_predictions) and the water helpers the later stages share: the speed horizon rescue
(hz_scan / hz_rescue / hz_best / rescue_raw), the water law on every step (water_speed_all), the slow-down rule (stop_rel)
and the split-band consensus gate (splitband_C)."""
import sys, os, numpy as np
from pathlib import Path

sys.path.insert(0,'data/test/starter'); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import lookout_vo as LV
from public_dataset import load_manifest
N=16

VSRC='runs/vggt'
def vggt_steps(k,n_steps,cfg,src=VSRC):
    """Per-step yaw from VGGT window poses (lookout_vggt_colab.ipynb): for consecutive keyframes a, b of a window the relative
    rotation R = R_b R_a^T (camera from world, OpenCV), yaw atan2(R[0,2], R[2,2]) spread evenly over the steps between them.
    NaN where no window covers a step; None without the clip's file."""
    f=f'{src}/{cfg}/{k}.npz'
    if not os.path.exists(f): return None
    z=np.load(f); y=np.full(n_steps,np.nan)
    for e,st in zip(z['ext'],z['steps']):
        st=[int(x) for x in st if x>=0]; R=[x[:,:3].astype(np.float64) for x in e[:len(st)]]
        for a,b,s0,s1 in zip(R[:-1],R[1:],st[:-1],st[1:]):
            Rr=b@a.T; y[s0:s1]=np.arctan2(Rr[0,2],Rr[2,2])/(s1-s0)
    return y

def offset_law(d):
    """Water speed per step with the law P = s v^2 + c (a constant offset per step) on a raw with profiles and the speed
    horizon h_step; steps without a tool speed stay None. Quality = relative RMS of the fit. -> (spd list, quality list)"""
    prof=d['profile'].astype(np.float32); fy=float(d['fy']); h=d['h_step']; rows=np.arange(prof.shape[1]); sp=[None]*len(d['dts']); q=[np.nan]*len(d['dts'])
    for i in range(len(sp)):
        if not np.isfinite(d['spd'][i]): continue
        P=prof[i].astype(float); v=(rows-h[i])/fy; m=np.isfinite(P)&(v>=LV.WATER_V_MIN)&(v<=LV.WATER_V_MAX)&(rows<LV.BOTTOM_CUT*len(P))
        if m.sum()<10: continue
        c=np.linalg.lstsq(np.c_[v[m]**2,np.ones(m.sum())],P[m],rcond=None)[0]; sp[i]=float(c[0]); pred=c[0]*v[m]**2+c[1]
        q[i]=float(np.sqrt(np.mean((P[m]-pred)**2))/(np.abs(pred).mean()+1e-12)) if c[0]>0 else np.nan
    return sp,q
def hz_scan(d,win=20.0,gap=80.0,r2min=0.95,step=4):
    """Speed horizon scan: per WIN-s window the median water profile is scanned for the horizon h0 whose water law is
    self-consistent (sqrt(P) linear in the row over v = (row - h0)/fy in LV.WATER_HZ_V, IRLS; fitted horizon -b/a within 8 px
    of h0; best R^2); the band stops at the first row where the flow drops below 0.7 of its running maximum (bow, rail).
    Where that R^2 >= r2min and the tool's h_step (window median) is more than `gap` px away, the scan's horizon replaces it.
    -> (h_step per step, replaced mask, per-window table (start, tool row, scan row, R^2, band cut row))"""
    P_all=d['profile'].astype(float); fy=float(d['fy']); h=np.asarray(d['h_step'],float).copy(); T=np.asarray(d['times'],float)
    n,H=P_all.shape; rows=np.arange(H); t=T[1:n+1]; rep=np.zeros(n,bool); tab=[]
    for w0 in np.arange(t[0],t[-1]+1e-9,win):
        sl=(t>=w0)&(t<w0+win)
        if sl.sum()<5: continue
        P=np.nanmedian(P_all[sl],0); best=None
        for h0 in range(int(0.08*H),int(0.9*H),step):
            v=(rows-h0)/fy; m=np.isfinite(P)&(P>0)&(v>=LV.WATER_HZ_V[0])&(v<=LV.WATER_HZ_V[1])&(rows<LV.BOTTOM_CUT*H)
            if m.sum()<20: continue
            idx=np.flatnonzero(m); run=np.maximum.accumulate(P[idx]); cut=np.flatnonzero((P[idx]<0.7*run)&(np.arange(len(idx))>=10))
            cr=idx[cut[0]] if len(cut) else H
            if len(cut): m[cr:]=False
            if m.sum()<20: continue
            x=rows[m].astype(float); y=np.sqrt(P[m]); w=np.ones(len(x))
            for _ in range(6):
                A=np.c_[x,np.ones(len(x))]*np.sqrt(w)[:,None]; a,b=np.linalg.lstsq(A,y*np.sqrt(w),rcond=None)[0]
                r=np.abs(y-a*x-b); sc=1.4826*np.median(r)+1e-12; w=np.where(r<=1.5*sc,1.0,1.5*sc/np.maximum(r,1e-12))
            if a<=0 or abs(-b/a-h0)>8: continue
            res=y-a*x-b; r2=1-np.sum(w*res**2)/np.sum(w*(y-np.average(y,weights=w))**2)
            if best is None or r2>best[0]: best=(r2,-b/a,cr)
        hd=float(np.nanmedian(h[sl]))
        if best is not None and best[0]>=r2min and abs(best[1]-hd)>gap: h[sl]=best[1]; rep[sl]=True
        tab.append((w0,hd,np.nan if best is None else best[1],np.nan if best is None else best[0],np.nan if best is None else best[2]))
    return h,rep,np.array(tab)
def hz_rescue(d):
    """Clip-level use of hz_scan: when in >= half of the reliable scan windows (R^2 >= 0.95) the tool's horizon is more than
    80 px off, the detector is taken as stuck on something else (e.g. the own bow's edge) and every step gets the scan
    horizon of the reliable windows, interpolated between their centres. -> (h_step, triggered, table)"""
    _,_,tab=hz_scan(d)
    ok=np.isfinite(tab[:,3])&(tab[:,3]>=0.95)
    if ok.sum()>=2 and np.mean(np.abs(tab[ok,1]-tab[ok,2])>80.0)>=0.5:
        t=np.asarray(d['times'],float)[1:len(d['dts'])+1]
        return np.interp(t,tab[ok,0]+20.0/2,tab[ok,2]),True,tab
    return np.asarray(d['h_step'],float),False,tab
def hz_best(d,r2min=0.95,win=20.0):
    """Speed horizon per step: hz_scan's window horizon where its water law fits (R^2 >= r2min), the tool's h_step elsewhere."""
    h=np.asarray(d['h_step'],float).copy(); _,_,tab=hz_scan(d,gap=0,win=win); t=np.asarray(d['times'],float)[1:len(d['dts'])+1]
    for w0,_,hs,r2,_ in tab:
        if np.isfinite(r2) and r2>=r2min: h[(t>=w0)&(t<w0+win)]=hs
    return h
def rescue_raw(h):
    """An h_step raw with the speed horizon rescued (hz_rescue) and, on such a clip, the profile rows from the bow edge down
    (hz_scan's band cut, median over the reliable windows) blanked; the water law then on every step (bow_row marks it).
    Untriggered clips come back unchanged."""
    hh,trig,tab=hz_rescue(h)
    if not trig: return h
    r={q:h[q] for q in h.files}; ok=np.isfinite(tab[:,3])&(tab[:,3]>=0.95); bow=int(np.median(tab[ok,4]))
    P=r['profile'].astype(float).copy(); P[:,bow:]=np.nan; r['profile']=P; r['h_step']=hh; r['bow_row']=bow
    r['spd']=water_speed_all(r,hh); return r
WATER_LOWPOW = None   # (P0, beta, F): the wide low-flow correction of stage 10 (lowflow_correction.py), None = off
def water_speed_all(d,h):
    """The tool's water law P = s v^2 on every step (also where the tool gave no speed) with the speed horizon h.
    WATER_LOWPOW (P0, beta, F): on a clip whose median 20-s flow is below F px, in 20-s windows with median flow < P0 px
    the water x (P0 / flow)^beta."""
    P=d['profile'].astype(float); fy=float(d['fy'])
    s=np.array([np.nan if (s_:=LV.speed_from_profile(P[i],h[i],fy)) is None else s_ for i in range(len(d['dts']))])
    if WATER_LOWPOW:
        n=len(s); px=fy*s*0.04; t=np.asarray(d['times'],float)[1:n+1]; P0_,B_=WATER_LOWPOW[:2]
        W0_=np.arange(t[0],t[-1]+1e-9,20.0); F_=WATER_LOWPOW[2] if len(WATER_LOWPOW)>2 else 0.0
        if F_>0 and not np.nanmedian([np.nanmedian(px[(t>=a)&(t<a+20.0)]) if ((t>=a)&(t<a+20.0)).any() else np.nan for a in W0_])>=F_: W0_=W0_[:0]
        for w0 in W0_:
            sl=(t>=w0)&(t<w0+20.0)
            if sl.any() and np.isfinite(px[sl]).sum()>=5 and 0<np.nanmedian(px[sl])<P0_: s[sl]=s[sl]*(P0_/max(float(np.nanmedian(px[sl])),0.2))**B_
    return s
STOP_THR = float(os.environ.get('LOOKOUT_STOP_THR', '0.5'))   # the hard gate's slow window: water < STOP_THR * S
def stop_rel(T, d, win=20.0, min_good=0.0, soft=False, hz='rescue', stop_ok=False):
    """Constant speed except 20-s windows where the water says the boat slows / stops. Per window (every LOOKOUT_STOP_STRIDE s,
    default win) the median profile's water law sqrt(P) = a v gives R^2; S = median water of the windows where it fits
    (R^2 >= 0.9). Hard rule: a fitting window with water < STOP_THR * S, or a broken law (R^2 < 0.5) with ~0 flow (< 0.3 S,
    a stop), is slow: water / S (>= 0.05). soft: every window with R^2 >= 0.5 takes water / S within [0.3, 1], stops
    0.05, the rest from the neighbours. min_good: the rule only where the law fits in that share of the windows (stop_ok:
    stop-like windows count as fitting). hz: the speed horizon -- 'best' (hz_best), 'rescue' (hz_rescue), 'slrsel:G' (the
    SLR water edge where its median agrees with hz_best within G px). -> (per-step relative speed, window table (centre s,
    water speed, R^2), fraction of slow windows)"""
    if str(hz).startswith('slrsel:'):
        P_ = d['profile']; e_ = np.array([np.argmax(np.isfinite(p_)) if np.isfinite(p_).any() else np.nan for p_ in P_], float)
        ok_ = np.isfinite(e_); e_ = np.interp(np.arange(len(e_)), np.flatnonzero(ok_), e_[ok_]) if ok_.any() else np.asarray(d['h_step'], float)
        h = e_ if abs(np.median(e_ - hz_best(d))) <= float(str(hz)[7:]) else hz_best(d)
    else: h = hz_best(d) if hz == 'best' else hz_rescue(d)[0]
    w = water_speed_all(d, h); t = np.asarray(T[1:len(w) + 1], float); n = len(w)
    P_all = d['profile'].astype(float); fy = float(d['fy']); rows = np.arange(P_all.shape[1]); tab = []
    stride = float(os.environ.get('LOOKOUT_STOP_STRIDE', '0') or 0) or win
    for w0 in np.arange(t[0], t[-1] + 1e-9, stride):
        sl = (t >= w0) & (t < w0 + win); ok = sl & np.isfinite(w)
        if sl.sum() < 5 or ok.sum() < 5: continue
        P = np.nanmedian(P_all[sl], 0); hh = float(np.median(h[sl])); v = (rows - hh) / fy
        m = np.isfinite(P) & (v >= LV.WATER_V_MIN) & (v <= LV.WATER_V_MAX) & (rows < LV.BOTTOM_CUT * len(P))
        r2 = np.nan
        if m.sum() >= 10:
            y = np.sqrt(np.clip(P[m], 0, None)); x = v[m]; a = float(x @ y / (x @ x)); res = y - a * x
            r2 = 1 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        tab.append((w0 + win / 2, float(np.median(w[ok])), r2))
    tab = np.array(tab)
    if len(tab) < 2: return np.ones(n), tab, 0.0
    good = tab[:, 2] >= 0.9
    if not good.any(): return np.ones(n), tab, 0.0
    S = float(np.median(tab[good, 1]))
    brk = (tab[:, 2] < 0.5) & (np.abs(tab[:, 1]) < 0.3 * S)          # a broken law with ~0 flow = a stop
    if (good | brk if stop_ok else good).mean() < min_good: return np.ones(n), tab, 0.0
    slow = (good & (tab[:, 1] < STOP_THR * S)) | brk
    rel = np.where(slow, np.maximum(tab[:, 1] / S, 0.05), 1.0)
    if soft:
        ok = tab[:, 2] >= 0.5
        rel = np.where(brk, 0.05, np.where(ok, np.clip(np.maximum(tab[:, 1] / S, 1e-3), 0.3, 1.0), np.nan))
        k = np.isfinite(rel)
        if not k.any(): return np.ones(n), tab, 0.0
        rel = np.interp(tab[:, 0], tab[k, 0], rel[k])
        slow = rel < 0.95
    return np.interp(t, tab[:, 0], rel), tab, float(slow.mean())
def splitband_C(d,split=0.19,lo=None,hi=None):
    """Split-band consensus of the water speed: per step the law P = s v^2 fitted separately on the near (v < split) and far
    (v >= split) half of the water band; per LV.ADAPT_BLOCK_S block (the gate's validity rule) the median log speed of each
    half; C = std(near - far block medians) / S, S = the gate's spread of the full-band block medians. Real speed changes move
    both halves together, a wrong horizon row or band-specific junk moves them apart. -> (C, S)"""
    lo=LV.WATER_V_MIN if lo is None else lo; hi=LV.WATER_V_MAX if hi is None else hi
    prof=d['profile'].astype(np.float32); fy=float(d['fy']); h=d['h_step']; rows=np.arange(prof.shape[1]); dts=d['dts']
    t=np.asarray(d['times'][1:],float); n=len(dts); ln,lf=np.full(n,np.nan),np.full(n,np.nan)
    for i in range(n):
        if not np.isfinite(d['spd'][i]): continue
        P=prof[i].astype(float); v=(rows-h[i])/fy; ok=np.isfinite(P)&(rows<LV.BOTTOM_CUT*len(P))
        for out,m in ((ln,ok&(v>=lo)&(v<split)),(lf,ok&(v>=split)&(v<=hi))):
            if m.sum()>=10:
                sv=float(np.sum(P[m]*v[m]**2)/np.sum(v[m]**4))/dts[i]
                if sv>0: out[i]=np.log(sv)
    w=np.asarray(d['spd'],float)/dts; lw=np.where(np.isfinite(w)&(w>0),np.log(np.where(w>0,w,1.0)),np.nan)
    B=LV.ADAPT_BLOCK_S; bw,bd=[],[]
    for s0 in np.arange(t[0],t[-1]-B/2,B):
        m=(t>=s0)&(t<s0+B)
        if np.isfinite(lw[m]).sum()>B*2:
            bw.append(np.nanmedian(lw[m]))
            if np.isfinite(ln[m]).any() and np.isfinite(lf[m]).any(): bd.append(np.nanmedian(ln[m])-np.nanmedian(lf[m]))
    S=float(np.std(bw)) if len(bw)>2 else 0.0
    return (float(np.std(bd))/S if S>0 and len(bd)>2 else np.nan), S
def rebuilt(pos,ref,atol=1e-9):
    """positions rebuilt by LV.integrate equal the saved ones (bit-identical on the machine that made them; ~1 ulp elsewhere)"""
    return np.allclose(np.asarray(pos),np.asarray(ref),rtol=0,atol=atol)
def chunks(dy):
    n=len(dy)//N; return dy[:n*N].reshape(n,N).sum(1)
def spread(c,tail):
    return np.r_[np.repeat(c/N,N),tail]
def test_predictions(out,mE,mB,mV,wideV,wideO,fscale_flirvis,c5_flirvis,offset_flirvis,stop_min_good,stop_hz,stop_ok,
                     src='runs/test_v5',root='data/test',raw='runs/test_v4/raw',vcfg='kf4_w17',osrc='runs/vggt_omega'):
    """Stage 1: new course on the wide and flir-VIS clips of the tool's test run `src`; every other clip copied.
    wide: per step (1-w) * mean(YAW_MULT * BA, wideV * VGGT) + w * m * Omega, wideO = (m, w) (VGGT / Omega windows vcfg; BA
      where no window), speed as in src (the tool's adaptive water speed on src's raw); src's own positions rebuilt first.
    flir-VIS (raw: test_v4): per 16-step chunk the median of mE*E16, mV*VGGT, mB*BA with the boat-fixed image areas masked
      (runs/lt_half_fx_test, runs/lt_ba_fx2_test; E16 chunk where no VGGT window), tail steps mE*E16; mE and mB x
      fscale_flirvis (the test flir-VIS focal is 1/g x the train one; VGGT is K-free). Speed: the offset law (offset_law) on
      offset_flirvis raws with the horizon rescued; clips whose split-band consensus C >= c5_flirvis get constant speed and
      then the slow-down rule stop_rel (soft, gated by the hard rule, min_good stop_min_good, horizon stop_hz; stop_ok
      'rescued' = the stop-ok guard only where the horizon rescue fired)."""
    import json,shutil
    sys.path.insert(0,root+'/starter')
    from public_dataset import load_timestamps,validate_prediction,write_json
    man=load_manifest(root+'/release_manifest.json'); os.makedirs(out+'/predictions',exist_ok=True)
    for c in man['clips']:
        k=c['key']; dst=f'{out}/predictions/{k}.json'
        if c['camera_mode']=='wide':
            d=np.load(f'{src}/raw/{k}.npz'); old=json.load(open(f'{src}/predictions/{k}.json'))
            T=d['times']; dec=[int(x) for x in d['dec']]; spd=list(d['spd']); q=d['quality']; wm=LV.YAW_MULT['wide']
            _,_,pos0,_=LV.integrate(list(T),dec,list(wm*d['dyaw']),list(d['dts']),spd,'adaptive',q)
            assert rebuilt(pos0,old['positions']), k
            yv=vggt_steps(k,len(d['dyaw']),vcfg); assert np.array_equal(np.load(f'{VSRC}/{vcfg}/{k}.npz')['times'],T), k
            B=d['dyaw']
            y=(wm*B+wideV*np.where(np.isnan(yv),B,yv))/2
            mO,wO=wideO; yo=vggt_steps(k,len(d['dyaw']),vcfg,src=osrc); assert np.array_equal(np.load(f'{osrc}/{vcfg}/{k}.npz')['times'],T), k
            y=(1-wO)*y+wO*mO*np.where(np.isnan(yo),B,yo)
            _,_,pos,_=LV.integrate(list(T),dec,list(y),list(d['dts']),spd,'adaptive',q)
            pred=dict(key=k,frame_indices=old['frame_indices'],times=old['times'],positions=pos)
            validate_prediction(pred,k,load_timestamps(root,c)); write_json(Path(dst),pred)
            print(f'{k} wide: VGGT steps {np.mean(~np.isnan(yv)):.2f}, Omega steps {np.mean(~np.isnan(yo)):.2f}',flush=True)
            continue
        if c['camera_mode']!='flir-VIS': shutil.copyfile(f'{src}/predictions/{k}.json',dst); continue
        d=np.load(f'{raw}/{k}.npz'); old=json.load(open(f'{src}/predictions/{k}.json'))
        T=d['times']; dec=[int(x) for x in d['dec']]
        _,_,pos0,_=LV.integrate(list(T),dec,list(LV.YAW_MULT['flir-VIS']*d['dyaw']),list(d['dts']),list(d['spd']),'adaptive',d['quality'])
        assert rebuilt(pos0,old['positions']), k
        zE=np.load(f'runs/lt_half_fx_test/{k}.npz'); assert np.array_equal(zE['times'],T), k
        E,B=zE['dyaw'],np.load(f'runs/lt_ba_fx2_test/{k}.npz')['dyaw']; n=len(d['dyaw'])//N
        mEf,mBf=mE*fscale_flirvis,mB*fscale_flirvis
        yv=vggt_steps(k,len(E),vcfg); assert np.array_equal(np.load(f'{VSRC}/{vcfg}/{k}.npz')['times'],T), k
        bad=chunks(np.isnan(yv).astype(float))>0; Vc=np.where(bad,chunks(E),chunks(np.nan_to_num(yv)))
        y=spread(np.median([mEf*chunks(E),mV*Vc,mBf*chunks(B)],0),mEf*E[n*N:])
        h=rescue_raw(np.load(f'{offset_flirvis}/{k}.npz'))
        assert np.array_equal(h['times'],T) and (np.allclose(h['spd'],d['spd'],equal_nan=True) or 'spd' in h and not hasattr(h,'files')), k
        C,_=splitband_C(h); force0=bool(np.isfinite(C) and C>=c5_flirvis)
        h=np.load(f'{offset_flirvis}/{k}.npz')
        assert np.array_equal(h['times'],T) and np.array_equal(h['dyaw'],d['dyaw']) and np.allclose(h['spd'],d['spd'],equal_nan=True), k
        sp_,q_=offset_law(rescue_raw(h))
        t0,t1=LV.ADAPT_T0,LV.ADAPT_T1
        if force0: LV.ADAPT_T0,LV.ADAPT_T1=1e9,1e9+1
        try: _,_,pos,_=LV.integrate(list(T),dec,list(y),list(d['dts']),sp_,'adaptive',q_)
        finally: LV.ADAPT_T0,LV.ADAPT_T1=t0,t1
        tag=f'C={C:.3f}'
        if force0:                 # constant speed -> each step scaled by the slow-down rule's relative speed
            h=rescue_raw(np.load(f'{offset_flirvis}/{k}.npz')); h=dict(h) if not isinstance(h,dict) else h
            sok=('bow_row' in h) if stop_ok=='rescued' else stop_ok
            rel,_,fs=stop_rel(T,h,min_good=stop_min_good,soft=True,hz=stop_hz,stop_ok=sok)
            if stop_rel(T,h,min_good=stop_min_good,soft=False,hz=stop_hz,stop_ok=sok)[2]==0: rel=np.ones_like(rel); fs=0.0
            Q=np.asarray(pos,float); dQ=np.diff(Q,axis=0)*rel[:len(Q)-1,None]
            pos=np.vstack([Q[:1],Q[:1]+np.cumsum(dQ,0)]).tolist(); tag+=f' -> constant speed, slow windows {fs:.2f}'+(' (horizon rescued)' if sok else '')
        pred=dict(key=k,frame_indices=old['frame_indices'],times=old['times'],positions=pos)
        validate_prediction(pred,k,load_timestamps(root,c)); write_json(Path(dst),pred)
        print(f'{k} flir-VIS: VGGT chunks {1-bad.mean():.2f}, {tag}',flush=True)
