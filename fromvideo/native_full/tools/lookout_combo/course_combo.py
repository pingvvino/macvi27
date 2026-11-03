import sys,os,itertools,numpy as np,cv2
sys.path.insert(0,'data/release/starter'); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.insert(0,'data/release/starter'); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import metrics, lookout_vo as LV
from public_dataset import load_manifest,load_reference
cl={c['key']:c for c in load_manifest('data/release/release_manifest.json')['clips']}
def hz_scan(d,win=20.0,gap=80.0,r2min=0.95,step=4):
    """Speed horizon rescue (owner 19.10, clip_109: the tool's horizon detector sat on the bow edge for the whole clip, so the
    water band was the bow -> 7 % valid speeds). Per WIN-s window the median water profile is scanned for the horizon h0
    whose water law is self-consistent: sqrt(P) linear in the row over v = (row - h0)/fy in [WATER_HZ_V] (IRLS), fitted
    horizon -b/a within 8 px of h0, the best R^2; the band stops at the first row where the flow drops below 0.7 of its running
    maximum (bow, rail). Where that R^2 >= r2min and the tool's h_step (window median) is more than
    `gap` VO px away, the scan's horizon replaces it. Returns (h_step per step, replaced mask, per-window table)."""
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
            if len(cut): m[cr:]=False                             # the band stops where the flow first drops: the bow / rail
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
def hz_rescue(d,frac=0.5,**kw):
    """Clip-level use of hz_scan: when in >= frac of the reliable scan windows (R^2 >= r2min) the tool's horizon is more than
    `gap` px off, the detector is taken as stuck on something else (clip_109: the bow edge) and EVERY step gets the scan
    horizon of the reliable windows, interpolated between their centres (a clip median left R^2 0.74-0.86 on 109: the boat pitches). Returns (h_step, triggered, table). Train flir-VIS: never triggers;
    test: only clip_109 (5 / 5 reliable windows ~258 px off)."""
    h,rep,tab=hz_scan(d,**kw); r2min=kw.get('r2min',0.95); gap=kw.get('gap',80.0)
    ok=np.isfinite(tab[:,3])&(tab[:,3]>=r2min)
    if ok.sum()>=2 and np.mean(np.abs(tab[ok,1]-tab[ok,2])>gap)>=frac:   # per-window scan horizon where reliable (the boat
        win=kw.get('win',20.0); t=np.asarray(d['times'],float)[1:len(d['dts'])+1]   # pitches: 221-253 px on 109), interpolated between
        return np.interp(t,tab[ok,0]+win/2,tab[ok,2]),True,tab
    return np.asarray(d['h_step'],float),False,tab
