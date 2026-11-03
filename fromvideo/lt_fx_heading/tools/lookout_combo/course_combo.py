"""flir-VIS course from SEVERAL estimators combined per 16-step chunk (train, 8 clips):
  E = tool's E16 (runs/lt_half = final_v4 dyaw, checked identical), P = planar model (runs/lt_half_planar),
  R = rotation-only (runs/lt_rot, e16_course.py --no-emat), B = the tool's inverse-depth BA run offline on R's
  tracks (ba_course, rotation-only fallback, step length from the tool's v4 speed as in run_clip).
Each estimator gets its own multiplier, chosen leave-one-clip-out (best mean drift on the other 7 clips, alone,
with our speed); combos then use those multipliers: median of 3, mean of 2, etc. L<x><t> = LightGlue chunk yaw
(runs/lt_lg from e16_course.py --lg; x = a(liked)/(su)p(erpoint)/(si)f(t), t = RANSAC px), E16 chunk where it fails. Scored with OUR speed (v5 =
v4 for flir-VIS) and with GPS speed (course only). python course_combo.py
python course_combo.py test: test predictions with the 'mean EB' flir-VIS course (needs runs/lt_rot_test).
python course_combo.py test-cons: the same with the 2-of-3 consensus E16/BA/DPVO (needs runs/dpvo_poses).
python course_combo.py test-ebm: 'mean EB' with the boat-fixed areas masked, multipliers E16 0.60, BA 0.45 (chosen on train).
python course_combo.py test-evb: flir-VIS course = median of E16 x0.65, VGGT kf4 x0.55, BA x0.50 (chosen on train);
python course_combo.py test-evbw: the same + wide course = mean(BA x0.9, VGGT kf4 x0.80) (chosen on train, val once).
python course_combo.py test-evbmw: as test-evbw with the boat-fixed mask ('still'): E16 x0.60, VGGT x0.55, BA x0.45.
python course_combo.py test-evbmwg: test-evbmw + speed gate T0 0.40 (was 0.25) on flir-VIS (chosen on train, 7/8 folds).
python course_combo.py test-evbmwo: test-evbmw + flir-VIS water speed with P = s v^2 + c (offset_law; runs/fvt_hstep).
python course_combo.py test-evbmwc / test-evbmwco: test-evbmw + C5 split-band gate (C >= 0.45 -> alpha 0) on flir-VIS / + offset law too.
python course_combo.py test-var NAME MASK REFINE mE mB: the same with MASK = none|still|follow|slow (fixed_tracks.py rule) and
REFINE = 0|1 (BA second pass, LV.BA_REFINE), output runs/test_v5_NAME.
python course_combo.py wide [--val]: VGGT / DPVO course on wide (runs/vggt from lookout_vggt_colab.ipynb, runs/dpvo_poses from
lookout_dpvo_poses_v2_kaggle.ipynb); --val = the ONE val check.
python course_combo.py wide-ba [--val]: wide BA variants (second pass, without 'follow' tracks) from runs/lt_rot_wide."""
import sys,os,itertools,numpy as np,cv2
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0,'data/release/starter'); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import metrics, lookout_vo as LV
from public_dataset import load_manifest,load_reference
cl={c['key']:c for c in load_manifest('data/release/release_manifest.json')['clips']}
KS=sorted(k for k,c in cl.items() if c['split']=='train' and c['camera_mode']=='flir-VIS')
M=np.round(np.arange(0.3,1.31,0.05),2); N=16
def path(dts,yaw,v):
    st=np.c_[v*np.sin(yaw),v*np.cos(yaw)]*dts[:,None]; xy=np.vstack([[0,0],np.cumsum(st,0)]); return np.c_[xy,np.zeros(len(xy))]
def ba(k):
    f=f'runs/lt_ba/{k}.npz'
    if not os.path.exists(f):
        r=np.load(f'runs/lt_rot/{k}.npz'); d=np.load(f'runs/final_v4/raw/{k}.npz'); assert np.allclose(r['times'],d['times'])
        v,_=LV.adaptive_speed(d['times'],d['dts'],[None if not np.isfinite(s) else s for s in d['spd']])   # as run_clip: no quality
        y,used=LV.ba_course(r['rows'],r['K'],r['dyaw'],np.asarray(v)*d['dts'])
        os.makedirs('runs/lt_ba',exist_ok=True); np.savez(f,dyaw=y,used=used,n=len(y)//LV.BA_N); print(k,'BA chunks',used,'/',len(y)//LV.BA_N,flush=True)
    return np.load(f)['dyaw']
LGX=('aliked','superpoint','sift'); LGT=(0.7,1.0,1.5,2.5); AB={'aliked':'a','superpoint':'p','sift':'f'}
def lg_chunks(k,n_chunks,src='runs/lt_lg'):
    """Per E16 chunk: yaw from the saved LightGlue matches (above the horizon mask), the tool's essential-matrix
    RANSAC + recoverPose at threshold t px (tool: 0.7); NaN where < 20 matches or < 15 pose inliers."""
    f=f'{src}/{k}.npz'
    if not os.path.exists(f): return {}
    z=np.load(f); K=z['Kc']; out={}
    for ex in LGX:
        if 'lg_'+ex not in z.files: continue
        r=z['lg_'+ex]; r=r[r[:,5]>0]; idx=r[:,0].astype(int)
        for t in LGT:
            cf=f'runs/lt_lg_yaw/{k}_{ex}_{t}.npy'
            if os.path.exists(cf): out[f'L{AB[ex]}{t}']=np.load(cf); continue
            y=np.full(n_chunks,np.nan)
            for i in range(n_chunks):
                a=r[idx==i]
                if len(a)<20: continue
                p0=a[:,1:3].astype(np.float64); p1=a[:,3:5].astype(np.float64)
                cv2.setRNGSeed(0); E,m=cv2.findEssentialMat(p0,p1,K,method=cv2.RANSAC,prob=0.999,threshold=t)
                if E is None or E.shape!=(3,3): continue
                nn,R,_,_=cv2.recoverPose(E,p0,p1,K,mask=m)
                if nn>=15: y[i]=float(np.arctan2(R[0,2],R[2,2]))
            os.makedirs('runs/lt_lg_yaw',exist_ok=True); np.save(cf,y); out[f'L{AB[ex]}{t}']=y
    return out
VCFG=('kf4_w17','kf2_w9'); VSRC='runs/vggt'
PCFG={'XK':'pi3xK_kf4_w17','X0':'pi3x_kf4_w17'}; PSRC='runs/pi3'   # Pi3X (lookout_pi3_colab.ipynb): with / without our K
def vggt_steps(k,n_steps,cfg,src=VSRC):
    """Per-step yaw from lookout_vggt_colab.ipynb poses: for consecutive keyframes a, b of a window the relative rotation
    R = R_b R_a^T (camera from world = first frame, OpenCV) -- the tool's E16 convention (recoverPose: x_b = R x_a + t) --
    yaw atan2(R[0,2], R[2,2]) spread evenly over the steps between them. NaN where no window covers a step."""
    f=f'{src}/{cfg}/{k}.npz'
    if not os.path.exists(f): return None
    z=np.load(f); y=np.full(n_steps,np.nan)
    for e,st in zip(z['ext'],z['steps']):
        st=[int(x) for x in st if x>=0]; R=[x[:,:3].astype(np.float64) for x in e[:len(st)]]
        for a,b,s0,s1 in zip(R[:-1],R[1:],st[:-1],st[1:]):
            Rr=b@a.T; y[s0:s1]=np.arctan2(Rr[0,2],Rr[2,2])/(s1-s0)
    return y
DSRC='runs/dpvo_poses'
def dpvo_steps(k,T,src=DSRC):
    """Per-step yaw from lookout_dpvo_poses_v2_kaggle.ipynb: DPVO poses are world from camera (x y z qx qy qz qw), so the
    step rotation in the tool's E16 convention is R = Q_{i+1}^T Q_i; yaw atan2(R[0,2], R[2,2]). None if absent or if the
    frames differ from the tool's."""
    f=f'{src}/{k}.npz'
    if not os.path.exists(f): return None
    z=np.load(f)
    if not np.array_equal(z['times'],T): print(f'  {k}: DPVO frames differ from the tool\'s -- skipped'); return None
    from scipy.spatial.transform import Rotation
    Q=Rotation.from_quat(z['poses'][:,3:7]).as_matrix(); Rr=np.einsum('nji,njk->nik',Q[1:],Q[:-1])
    return np.arctan2(Rr[:,0,2],Rr[:,2,2])
PAIRS=(('E','B'),('E','D'),('B','D')); CW=4
def consensus_pair(Y):
    """2-of-3 consensus (consensus.py): Y = {'E','B','D': chunk yaws, multipliers applied}; disagreement of a pair = RMS of
    the difference of their heading changes over CW-chunk (8 s) windows; returns (per-chunk mean of the closest pair,
    pair, disagreements). GPS-free."""
    win=lambda y: y[:len(y)//CW*CW].reshape(-1,CW).sum(1)
    dis={p:float(np.sqrt(np.mean((win(Y[p[0]])-win(Y[p[1]]))**2))) for p in PAIRS}
    p=min(dis,key=dis.get); return (Y[p[0]]+Y[p[1]])/2,p,dis
def ba_fixed(k,src='runs/lt_rot',cache='runs/lt_ba_fx',vraw=None,rule='still'):
    """BA (as ba()) on the same tracks WITHOUT the boat-fixed ones (fixed_tracks.py: still while the scene moves --
    camera overlays, lens drops, bow, rail, antenna)."""
    f=f'{cache}/{k}.npz'
    if not os.path.exists(f):
        from fixed_tracks import fixed_track_ids
        r=np.load(f'{src}/{k}.npz'); d=np.load(vraw or f'runs/final_v4/raw/{k}.npz'); assert np.allclose(r['times'],d['times'])
        rows=r['rows']; fx=fixed_track_ids(rows,rule); keep=~np.isin(rows[:,1].astype(int),list(fx))
        v,_=LV.adaptive_speed(d['times'],d['dts'],[None if not np.isfinite(s) else s for s in d['spd']])
        y,used=LV.ba_course(rows[keep],r['K'],r['dyaw'],np.asarray(v)*d['dts'])
        os.makedirs(cache,exist_ok=True); np.savez(f,dyaw=y,used=used,n=len(y)//LV.BA_N,dropped=float(1-keep.mean()))
        print(k,'BA without boat-fixed tracks: dropped',f'{1-keep.mean():.1%}','of observations, chunks',used,flush=True)
    return np.load(f)['dyaw']
def ba_any(k,src='runs/lt_rot',cache='runs/lt_ba_r',vraw=None,rule=None,refine=True):
    """BA as ba()/ba_fixed() with options: rule = drop tracks flagged by fixed_tracks.py ('still'/'follow'/None),
    refine = the tool's optional second pass (LV.BA_REFINE: drop tracks with large reprojection error, solve again)."""
    f=f'{cache}/{k}.npz'
    if not os.path.exists(f):
        r=np.load(f'{src}/{k}.npz'); d=np.load(vraw or f'runs/final_v4/raw/{k}.npz'); assert np.allclose(r['times'],d['times'])
        rows=r['rows']
        if rule:
            from fixed_tracks import fixed_track_ids
            rows=rows[~np.isin(rows[:,1].astype(int),list(fixed_track_ids(rows,rule)))]
        v,_=LV.adaptive_speed(d['times'],d['dts'],[None if not np.isfinite(s) else s for s in d['spd']])
        old=LV.BA_REFINE; LV.BA_REFINE=refine
        try: y,used=LV.ba_course(rows,r['K'],r['dyaw'],np.asarray(v)*d['dts'])
        finally: LV.BA_REFINE=old
        os.makedirs(cache,exist_ok=True); np.savez(f,dyaw=y,used=used)
        print(k,cache,'chunks',used,flush=True)
    return np.load(f)['dyaw']
def offset_law(d):
    """Water speed per step with the law P = s v^2 + c (a constant offset per step; the tool fits P = s v^2) on a raw that
    keeps the profiles and the speed horizon h_step (lookout_vo.py --save-raw); steps without a tool speed stay None.
    Quality = relative RMS of that fit (as wave_term.qual). Returns (spd list, quality list)."""
    prof=d['profile'].astype(np.float32); fy=float(d['fy']); h=d['h_step']; rows=np.arange(prof.shape[1]); sp=[None]*len(d['dts']); q=[np.nan]*len(d['dts'])
    for i in range(len(sp)):
        if not np.isfinite(d['spd'][i]): continue
        P=prof[i].astype(float); v=(rows-h[i])/fy; m=np.isfinite(P)&(v>=LV.WATER_V_MIN)&(v<=LV.WATER_V_MAX)&(rows<LV.BOTTOM_CUT*len(P))
        if m.sum()<10: continue
        c=np.linalg.lstsq(np.c_[v[m]**2,np.ones(m.sum())],P[m],rcond=None)[0]; sp[i]=float(c[0]); pred=c[0]*v[m]**2+c[1]
        q[i]=float(np.sqrt(np.mean((P[m]-pred)**2))/(np.abs(pred).mean()+1e-12)) if c[0]>0 else np.nan
    return sp,q
def splitband_C(d,split=0.19,lo=None,hi=None):
    """Split-band consensus of the water speed (C5_REPORT.md, outside review 15.10): per step the tool's law P = s v^2
    fitted separately on the near (v < split) and far (v >= split) half of the water band; per 30 s block (the gate's
    ADAPT_BLOCK_S, same validity rule) the median log speed of each half; C = std(near - far block medians) / S, S = the
    gate's own spread (std of the full-band block medians of log speed). Real speed changes move both halves together,
    a wrong horizon row or band-specific junk moves them apart. Needs a raw with profiles + h_step."""
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
def chunks(dy):
    n=len(dy)//N; return dy[:n*N].reshape(n,N).sum(1)
def spread(c,tail):
    return np.r_[np.repeat(c/N,N),tail]
def test_predictions(src='runs/test_v5',out='runs/test_v5_eb',root='data/test',mE=0.65,mB=0.50,raw='runs/test_v4/raw',mode='eb',mD=0.50,masked=False,refine=False,
                     mV=0.55,vcfg='kf4_w17',wideV=None,t0_flirvis=None,offset_flirvis=None,c5_flirvis=None,c5_dir='runs/fvt_hstep/raw',
                     fscale_flirvis=None,t0_wide=None):
    """Test variant 'mean EB' for flir-VIS: per 16-step chunk yaw = mean(mE*E16, mB*BA) (multipliers chosen on all 8
    train clips), tail steps mE*E16; speed and every other camera unchanged from `src`. Positions with the tool's own
    integrate() on `raw` (flir-VIS raw lives in test_v4: v5 changed only wide); first checks that it rebuilds src's flir-VIS predictions exactly.
    mode 'cons': 2-of-3 consensus of mE*E16, mB*BA, mD*DPVO (consensus_pair; runs/dpvo_poses), 'eb' where DPVO is missing.
    masked=True: E16 and BA with the boat-fixed image areas excluded (e16_course.py --fixed-from: runs/lt_half_fx_test,
    runs/lt_rot_fx_test; BA re-filtered by fixed_tracks.py), multipliers re-chosen on train. masked='follow': the same with
    fixed_tracks.py rule 'follow' (lens drops, people; runs/lt_half_fx2_test, runs/lt_rot_fx2_test); 'slow' likewise (fx3). refine=True: BA with
    the second pass (LV.BA_REFINE).
    mode 'evb': per chunk MEDIAN of mE*E16, mV*VGGT (vcfg window setting, runs/vggt; E16 chunk where no VGGT window),
    mB*BA -- 'med E+Vkf4+B' of main(). wideV=m: wide clips too, per step mean(YAW_MULT['wide']*BA, m*VGGT; BA where no
    window) -- wide_eval's 'mean B+Vkf4' -- on src's raw (test_v5/raw), first checking it rebuilds src's wide predictions.
    t0_flirvis=x: the speed trust gate's lower threshold (LV.ADAPT_T0) for flir-VIS clips only (train: 0.25 -> 0.40).
    offset_flirvis=DIR: flir-VIS water speed with the law P = s v^2 + c (offset_law) from DIR/<clip>.npz (a --save-raw rerun
    keeping h_step; checked identical to `raw` in frames, course and speed first).
    c5_flirvis=x: flir-VIS clips whose split-band consensus C (splitband_C on c5_dir) >= x get constant speed (alpha 0)
    -- C5_REPORT.md's gate (0.45).
    fscale_flirvis=g: flir-VIS E16 and BA multipliers (mE, mB; also the tail's mE) times g -- the test flir-VIS focal is
    1/g x the train one (the train multipliers absorb the train focal); VGGT's mV unchanged (K-free).
    mode 'v': flir-VIS chunk yaw = mV*VGGT alone (E16 chunk where no window; tail mE*E16 as before) -- the course
    independent of the release calibration (E16 and BA need K; the test flir-VIS focal differs from train)."""
    import json,shutil
    sys.path.insert(0,root+'/starter')
    from public_dataset import load_timestamps,validate_prediction,write_json
    man=load_manifest(root+'/release_manifest.json'); os.makedirs(out+'/predictions',exist_ok=True)
    for c in man['clips']:
        k=c['key']; dst=f'{out}/predictions/{k}.json'
        if c['camera_mode']=='wide' and wideV:
            d=np.load(f'{src}/raw/{k}.npz'); old=json.load(open(f'{src}/predictions/{k}.json'))
            T=d['times']; dec=[int(x) for x in d['dec']]; spd=list(d['spd']); q=d['quality']; wm=LV.YAW_MULT['wide']
            _,_,pos0,_=LV.integrate(list(T),dec,list(wm*d['dyaw']),list(d['dts']),spd,'adaptive',q)
            assert np.array_equal(np.asarray(pos0),np.asarray(old['positions'])), k
            yv=vggt_steps(k,len(d['dyaw']),vcfg); assert np.array_equal(np.load(f'{VSRC}/{vcfg}/{k}.npz')['times'],T), k
            y=(wm*d['dyaw']+wideV*np.where(np.isnan(yv),d['dyaw'],yv))/2
            t0w=LV.ADAPT_T0
            if t0_wide is not None: LV.ADAPT_T0=t0_wide
            try: _,_,pos,_=LV.integrate(list(T),dec,list(y),list(d['dts']),spd,'adaptive',q)
            finally: LV.ADAPT_T0=t0w
            if t0_wide is not None: _,aw=LV.adaptive_speed(T,d['dts'],[None if not np.isfinite(x) else x for x in spd],q); LV.ADAPT_T0=t0_wide; _,aw2=LV.adaptive_speed(T,d['dts'],[None if not np.isfinite(x) else x for x in spd],q); LV.ADAPT_T0=t0w; print(f'{k} wide gate T0 {t0_wide}: alpha {aw:.2f} -> {aw2:.2f}')
            pred=dict(key=k,frame_indices=old['frame_indices'],times=old['times'],positions=pos)
            validate_prediction(pred,k,load_timestamps(root,c)); write_json(Path(dst),pred)
            P=np.asarray(pos); P0=np.asarray(pos0)
            print(f'{k} wide B+V{vcfg[:3]}  VGGT steps {np.mean(~np.isnan(yv)):.2f}  turn total B {np.degrees(np.sum(wm*d["dyaw"])):7.1f}  new {np.degrees(np.sum(y)):7.1f} deg  '
                  f'end shift {np.hypot(*(P[-1,:2]-P0[-1,:2]))/np.sum(np.hypot(*np.diff(P0[:,:2],axis=0).T))*100:5.1f}% of path',flush=True)
            continue
        if c['camera_mode']!='flir-VIS': shutil.copyfile(f'{src}/predictions/{k}.json',dst); continue
        d=np.load(f'{raw}/{k}.npz'); old=json.load(open(f'{src}/predictions/{k}.json'))
        T=d['times']; dec=[int(x) for x in d['dec']]; spd=list(d['spd']); q=d['quality']
        _,_,pos0,_=LV.integrate(list(T),dec,list(LV.YAW_MULT['flir-VIS']*d['dyaw']),list(d['dts']),spd,'adaptive',q)
        assert np.array_equal(np.asarray(pos0),np.asarray(old['positions'])), k
        r=np.load(f'runs/lt_rot_test/{k}.npz'); assert np.allclose(r['times'],T), k
        v,_=LV.adaptive_speed(T,d['dts'],[None if not np.isfinite(s) else s for s in d['spd']])
        bf=f'runs/lt_ba_test/{k}.npz'
        rule={True:'still',False:None}.get(masked,masked)
        if rule:
            hx={'still':'fx','follow':'fx2','slow':'fx3'}[rule]
            zE=np.load(f'runs/lt_half_{hx}_test/{k}.npz'); assert np.array_equal(zE['times'],T), k
            if rule=='still' and not refine: Bmask=ba_fixed(k,src='runs/lt_rot_fx_test',cache='runs/lt_ba_fx2_test',vraw=f'{raw}/{k}.npz')
            else: Bmask=ba_any(k,src=f'runs/lt_rot_{hx}_test',cache={'still':'runs/lt_ba_mr_test','follow':'runs/lt_ba_fl_test','slow':'runs/lt_ba_sl_test'}[rule]+'r'*(rule!='still' and refine),
                               vraw=f'{raw}/{k}.npz',rule=rule,refine=refine)
        elif refine: Bmask=ba_any(k,src='runs/lt_rot_test',cache='runs/lt_ba_r_test',vraw=f'{raw}/{k}.npz',rule=None,refine=True)
        if not os.path.exists(bf):
            B,used=LV.ba_course(r['rows'],r['K'],r['dyaw'],np.asarray(v)*d['dts']); os.makedirs('runs/lt_ba_test',exist_ok=True); np.savez(bf,dyaw=B,used=used)
        B,used=np.load(bf)['dyaw'],int(np.load(bf)['used'])
        E=d['dyaw']; n=len(E)//N; Dy=dpvo_steps(k,T) if mode=='cons' else None; tag='EB'
        if rule: E,B,tag=zE['dyaw'],Bmask,{'still':'EmBm','follow':'EfBf','slow':'EsBs'}[rule]+'r'*refine
        elif refine: B,tag=Bmask,'EBr'
        mE_,mB_=mE,mB
        if fscale_flirvis: mE,mB=mE*fscale_flirvis,mB*fscale_flirvis
        if mode in ('evb','v'):
            yv=vggt_steps(k,len(E),vcfg); assert np.array_equal(np.load(f'{VSRC}/{vcfg}/{k}.npz')['times'],T), k
            bad=chunks(np.isnan(yv).astype(float))>0; Vc=np.where(bad,chunks(E),chunks(np.nan_to_num(yv)))
            if mode=='v': yc=mV*Vc; tag=f'V{vcfg[:3]} only (VGGT chunks {1-bad.mean():.2f})'
            else: yc=np.median([mE*chunks(E),mV*Vc,mB*chunks(B)],0); tag=f'med E+V{vcfg[:3]}+B (VGGT chunks {1-bad.mean():.2f})'
        elif Dy is None: yc=(mE*chunks(E)+mB*chunks(B))/2
        else:
            yc,pr,dis=consensus_pair({'E':mE*chunks(E),'B':mB*chunks(B),'D':mD*chunks(Dy)}); tag=''.join(pr)
            tag+=' ('+' '.join(f'{a}{b} {np.degrees(dis[(a,b)]):.2f}' for a,b in PAIRS)+' deg/8s)'
        y=spread(yc,mE*E[n*N:])
        if fscale_flirvis: tag+=f' focal: E16/BA x{fscale_flirvis}'
        t0=LV.ADAPT_T0
        if t0_flirvis is not None: LV.ADAPT_T0=t0_flirvis; tag+=f' gate T0 {t0_flirvis}'
        sp_,q_=spd,q
        force0=False
        if c5_flirvis is not None:
            h=np.load(f'{c5_dir}/{k}.npz')
            assert np.array_equal(h['times'],T) and np.allclose(h['spd'],d['spd'],equal_nan=True), k
            C,_=splitband_C(h); force0=bool(np.isfinite(C) and C>=c5_flirvis); tag+=f' C5 C={C:.3f}'+(' -> alpha 0' if force0 else '')
        if offset_flirvis:
            h=np.load(f'{offset_flirvis}/{k}.npz')
            assert np.array_equal(h['times'],T) and np.array_equal(h['dyaw'],d['dyaw']) and np.allclose(h['spd'],d['spd'],equal_nan=True), k
            sp_,q_=offset_law(h); tag+=' offset law'
        t1=LV.ADAPT_T1
        if force0: LV.ADAPT_T0,LV.ADAPT_T1=1e9,1e9+1
        try: _,_,pos,_=LV.integrate(list(T),dec,list(y),list(d['dts']),sp_,'adaptive',q_)
        finally: LV.ADAPT_T0,LV.ADAPT_T1=t0,t1
        pred=dict(key=k,frame_indices=old['frame_indices'],times=old['times'],positions=pos)
        validate_prediction(pred,k,load_timestamps(root,c)); write_json(Path(dst),pred)
        P=np.asarray(pos); P0=np.asarray(pos0); net=lambda Q: np.hypot(*(Q[-1,:2]-Q[0,:2]))/np.sum(np.hypot(*np.diff(Q[:,:2],axis=0).T))
        print(f'{k} {tag}  BA chunks {used}/{n}  turn total E {np.degrees(np.sum(mE_*E)):7.1f}  new {np.degrees(np.sum(y)):7.1f} deg  '
              f'end shift {np.hypot(*(P[-1,:2]-P0[-1,:2]))/np.sum(np.hypot(*np.diff(P0[:,:2],axis=0).T))*100:5.1f}% of path  net/path {net(P0):.2f}->{net(P):.2f}',flush=True)
        mE,mB=mE_,mB_

def pose_speed(k,T,cfg='kf4_w17',src=VSRC,per_window=False):
    """Per-step speed from a window model's poses (VGGT / Pi3X): camera centres c = -R^T t of consecutive keyframes,
    |c_b - c_a| / (T_b - T_a) spread over the steps between them; windows share one keyframe, so their scales are
    independent -- per_window=True rescales each window to mean 1 (within-window modulation only), False trusts the
    model's scale across windows. NaN where no window; None if the file is missing."""
    f=f'{src}/{cfg}/{k}.npz'
    if not os.path.exists(f): return None
    z=np.load(f); assert np.array_equal(z['times'],T), k
    v=np.full(len(T)-1,np.nan)
    for e,st in zip(z['ext'],z['steps']):
        st=[int(x) for x in st if x>=0]; R=e[:len(st),:,:3].astype(np.float64); t=e[:len(st),:,3].astype(np.float64)
        c=-np.einsum('nji,nj->ni',R,t); sp=np.linalg.norm(np.diff(c,axis=0),axis=1)/(T[st[1:]]-T[st[:-1]])
        if per_window and np.nanmean(sp)>0: sp=sp/np.nanmean(sp)
        for a,b,x in zip(st[:-1],st[1:],sp): v[a:b]=x
    return v
def pose_speed_eval(cam='flir-VIS',split='train'):
    """Speed source without the water: VGGT / Pi3X translation per keyframe interval (pose_speed), with the course as
    submitted (flir-VIS: median of masked E16 x0.60, VGGT x0.55, masked BA x0.45; wide: BA x0.9), vs constant speed and
    vs the water speed through the gate. Also the correlation of window-mean pose speed with window-mean GPS speed
    (is the scale consistent across windows?)."""
    ks=sorted(k for k,c in cl.items() if c['split']==split and c['camera_mode']==cam); R={}
    for k in ks:
        raw=f'runs/final_v4/raw/{k}.npz' if cam=='flir-VIS' else f'runs/final_v5/raw/{k}.npz'
        d=np.load(raw); T=d['times']; dts=d['dts']; rt,rx=load_reference('data/release',cl[k]); n=len(dts)
        gv=np.hypot(np.diff(np.interp(T,rt,rx[:,0])),np.diff(np.interp(T,rt,rx[:,1])))/dts
        if cam=='flir-VIS':
            E=np.load(f'runs/lt_half_fx/{k}.npz')['dyaw']; B=ba_fixed(k,src='runs/lt_rot_fx',cache='runs/lt_ba_fx2'); nn=n//N
            y=vggt_steps(k,n,'kf4_w17'); bad=chunks(np.isnan(y).astype(float))>0; V=np.where(bad,chunks(E),chunks(np.nan_to_num(y)))
            yaw=np.cumsum(spread(np.median([0.60*chunks(E),0.55*V,0.45*chunks(B)],0),0.60*E[nn*N:]))
        else: yaw=np.cumsum(LV.YAW_MULT['wide']*d['dyaw'])
        wv,_=LV.adaptive_speed(T,dts,[None if not np.isfinite(x) else x for x in d['spd']],d['quality'])
        S={'const':np.ones(n),'water gate':np.asarray(wv)}
        for nm,src,cfg in (('V','runs/vggt','kf4_w17'),('X0',PSRC,'pi3x_kf4_w17')):
            for pw in (False,True):
                v=pose_speed(k,T,cfg,src,per_window=pw)
                if v is None: continue
                v=np.where(np.isfinite(v),v,np.nanmedian(v)); S[f'{nm} {"window-norm" if pw else "raw"}']=v
                if not pw:
                    z=np.load(f'{src}/{cfg}/{k}.npz'); wm=[];gm=[]
                    for st in z['steps']:
                        st=[int(x) for x in st if x>=0]; wm.append(np.nanmean(v[st[0]:st[-1]])); gm.append(np.mean(gv[st[0]:st[-1]]))
                    S[f'{nm} corr']=float(np.corrcoef(wm,gm)[0,1])
        for nm,v in S.items():
            if nm.endswith('corr'): R.setdefault(nm,[]).append(v); continue
            R.setdefault(nm,[]).append(metrics.score_trajectory(T,path(dts,yaw,v),rt,rx)['drift_pct'])
        R.setdefault('GPS speed',[]).append(metrics.score_trajectory(T,path(dts,yaw,gv),rt,rx)['drift_pct'])
        print(f'  {k}: '+'  '.join(f'{nm} {R[nm][-1]:5.2f}' for nm in R),flush=True)
    print(f'{cam} {split} ({len(ks)} clips), course as submitted; drift median / mean per speed source:')
    for nm,x in R.items():
        x=np.array(x); print(f'  {nm:16s} {np.median(x):5.2f} / {np.mean(x):5.2f}' if not nm.endswith('corr') else f'  {nm:16s} window-mean speed vs GPS: median r {np.median(x):+.2f}, min {x.min():+.2f}')
def straight_group(base,out,modes=('wide',),root='data/test'):
    """Localization submission: clips of camera `modes` replaced by a STRAIGHT LINE at constant speed (positions
    (0, t - t0, 0) on the base prediction's own frames; the Sim(3) alignment makes speed and heading irrelevant, and a
    straight line resamples exactly, so each such clip scores as the organizers' straight-line baseline), every other
    clip copied byte for byte from `base`. 20 x (score - base score) = sum over the replaced clips of
    (straight-line drift - our drift): how much better than a straight line our predictions are on that group."""
    import json,shutil
    sys.path.insert(0,root+'/starter')
    from public_dataset import load_timestamps,validate_prediction,write_json
    man=load_manifest(root+'/release_manifest.json'); os.makedirs(out+'/predictions',exist_ok=True); n=0
    for c in man['clips']:
        k=c['key']; src=f'{base}/predictions/{k}.json'; dst=f'{out}/predictions/{k}.json'
        if c['camera_mode'] not in modes: shutil.copyfile(src,dst); continue
        old=json.load(open(src)); t=np.asarray(old['times'],float)
        pos=np.c_[np.zeros(len(t)),t-t[0],np.zeros(len(t))].tolist()
        pred=dict(key=k,frame_indices=old['frame_indices'],times=old['times'],positions=pos)
        validate_prediction(pred,k,load_timestamps(root,c)); write_json(Path(dst),pred); n+=1
        print(f'{k} {c["camera_mode"]}: straight line, {len(t)} samples, {t[-1]-t[0]:.1f} s',flush=True)
    print(f'{n} clips replaced by straight lines; the rest copied from {base}')

SPEEDS=('v','gv'); ONLY=None
def main():
    with Pool(4) as p: p.map(ba,KS)
    with Pool(4) as p: p.map(ba_fixed,KS)
    C={}; OK={}
    for k in KS:
        d=np.load(f'runs/final_v4/raw/{k}.npz'); T=d['times']; dts=d['dts']; rt,rx=load_reference('data/release',cl[k])
        v,_=LV.adaptive_speed(T,dts,[None if not np.isfinite(s) else s for s in d['spd']],d['quality'])
        gv=np.hypot(np.diff(np.interp(T,rt,rx[:,0])),np.diff(np.interp(T,rt,rx[:,1])))/dts
        E=np.load(f'runs/lt_half/{k}.npz')['dyaw']; est=dict(E=E,P=np.load(f'runs/lt_half_planar/{k}.npz')['dyaw'],
                 R=np.load(f'runs/lt_rot/{k}.npz')['dyaw'],B=ba(k),F=ba_fixed(k))
        if os.path.exists(f'runs/lt_half_fx/{k}.npz') and os.path.exists(f'runs/lt_rot_fx/{k}.npz'):
            est['Em']=np.load(f'runs/lt_half_fx/{k}.npz')['dyaw']                        # E16, boat-fixed areas masked
            est['Bm']=ba_fixed(k,src='runs/lt_rot_fx',cache='runs/lt_ba_fx2')            # BA on masked tracks, re-filtered
        if os.path.exists(f'runs/lt_ba_r/{k}.npz'): est['Br']=ba_any(k)                                  # BA + second pass
        if os.path.exists(f'runs/lt_ba_mr/{k}.npz'): est['Bmr']=ba_any(k,src='runs/lt_rot_fx',cache='runs/lt_ba_mr',rule='still')
        if os.path.exists(f'runs/lt_half_fx2/{k}.npz') and os.path.exists(f'runs/lt_rot_fx2/{k}.npz'):   # rule 'follow' (drops, people)
            est['Ef']=np.load(f'runs/lt_half_fx2/{k}.npz')['dyaw']
            est['Bf']=ba_any(k,src='runs/lt_rot_fx2',cache='runs/lt_ba_fl',rule='follow',refine=False)
            est['Bfr']=ba_any(k,src='runs/lt_rot_fx2',cache='runs/lt_ba_flr',rule='follow',refine=True)
        if os.path.exists(f'runs/lt_half_cl/{k}.npz') and os.path.exists(f'runs/lt_rot_cl/{k}.npz'):     # CLAHE (e16_course --clahe 2,8)
            est['Ec']=np.load(f'runs/lt_half_cl/{k}.npz')['dyaw']
            est['Bc']=ba_any(k,src='runs/lt_rot_cl',cache='runs/lt_ba_cl',rule=None,refine=False)
        if os.path.exists(f'runs/lt_half_fx3/{k}.npz') and os.path.exists(f'runs/lt_rot_fx3/{k}.npz'):   # rule 'slow'
            est['Es']=np.load(f'runs/lt_half_fx3/{k}.npz')['dyaw']
            est['Bs']=ba_any(k,src='runs/lt_rot_fx3',cache='runs/lt_ba_sl',rule='slow',refine=False)
            est['Bsr']=ba_any(k,src='runs/lt_rot_fx3',cache='runs/lt_ba_slr',rule='slow',refine=True)
        n=len(E)//N; ch={e:chunks(x) for e,x in est.items()}
        for cfg in VCFG:
            y=vggt_steps(k,len(E),cfg)
            if y is None: continue
            z=np.load(f'{VSRC}/{cfg}/{k}.npz'); assert np.array_equal(z['times'],T), k
            bad=chunks(np.isnan(y).astype(float))>0; ch['V'+cfg[:3]]=np.where(bad,ch['E'],chunks(np.nan_to_num(y))); OK.setdefault('V'+cfg[:3],[]).append(1-bad.mean())
            print(f'  {k} V{cfg[:3]} vs E16 chunk yaw: corr {np.corrcoef(ch["V"+cfg[:3]],ch["E"])[0,1]:+.2f}, total turn V {np.degrees(np.nansum(y)):7.1f} E16 {np.degrees(E.sum()):7.1f} deg')
        for nm,cfg in PCFG.items():
            y=vggt_steps(k,len(E),cfg,src=PSRC)
            if y is None: continue
            assert np.array_equal(np.load(f'{PSRC}/{cfg}/{k}.npz')['times'],T), k
            bad=chunks(np.isnan(y).astype(float))>0; ch[nm]=np.where(bad,ch['E'],chunks(np.nan_to_num(y)))
        if 'Vkf4' in ch:      # focal self-calibration against VGGT (independent of our K): E16 / BA chunk turns put on VGGT's scale
            vk=ch['Vkf4']
            for nm in ('E','B','Em','Bm'):
                if nm in ch: ch[nm+'n']=ch[nm]*float(vk@vk)/float(ch[nm]@vk)
        y=dpvo_steps(k,T)
        if y is not None:
            ch['D']=chunks(y); OK.setdefault('D',[]).append(1.0)
            print(f'  {k} D vs E16 chunk yaw: corr {np.corrcoef(ch["D"],ch["E"])[0,1]:+.2f}, total turn D {np.degrees(y.sum()):7.1f} E16 {np.degrees(E.sum()):7.1f} deg')
        for name,y in lg_chunks(k,n).items():
            ok=np.isfinite(y); ch[name]=np.where(ok,y,ch['E']); OK.setdefault(name,[]).append(ok.mean())
        C[k]=dict(T=T,dts=dts,v=np.asarray(v),gv=gv,c=np.ones(len(dts)),ref=(rt,rx),tail=E[n*N:],ch=ch)   # c: constant speed
    def score(k,yaw_chunks,spd):
        c=C[k]; yaw=np.cumsum(spread(yaw_chunks,0.65*c['tail'])); return metrics.score_trajectory(c['T'],path(c['dts'],yaw,c[spd]),*c['ref'])['drift_pct']
    ES=[e for e in C[KS[0]]['ch'] if all(e in C[k]['ch'] for k in KS)]
    for e in OK: print(f'  {e}: LightGlue chunk succeeded {np.mean(OK[e]):.2f} (else E16 chunk)')
    if ONLY: ES=[e for e in ES if e in ONLY]
    single={(e,m,k,s):score(k,m*C[k]['ch'][e],s) for e in ES for m in M for k in KS for s in SPEEDS}
    print('single estimators, best multiplier on all 8 (in-sample), our speed / GPS speed (median/mean):')
    for e in ES:
        for s in SPEEDS:
            bm=min(M,key=lambda m: np.mean([single[(e,m,k,s)] for k in KS])); x=[single[(e,bm,k,s)] for k in KS]
            print(f'  {e} {s:2s} m={bm:.2f} {np.median(x):5.2f}/{np.mean(x):5.2f}  at 0.65: {np.mean([single[(e,0.65,k,s)] for k in KS]):5.2f}  per clip '+' '.join(f'{y:5.2f}' for y in x))
    COMBOS={'E':('E',),'P':('P',),'R':('R',),'B':('B',),'med EPR':('E','P','R'),'med EPB':('E','P','B'),'med PRB':('P','R','B'),
            'med EPRB':('E','P','R','B'),'mean EP':('E','P'),'mean EPB':('E','P','B'),'mean EB':('E','B'),
            'F':('F',),'mean EF':('E','F'),'mean DF':('D','F'),'mean EDF':('E','D','F'),
            'Em':('Em',),'Bm':('Bm',),'mean EmBm':('Em','Bm'),'mean DBm':('D','Bm'),'mean EmDBm':('Em','D','Bm'),
            'Br':('Br',),'mean EBr':('E','Br'),'Bmr':('Bmr',),'mean EmBmr':('Em','Bmr'),
            'Ef':('Ef',),'Bf':('Bf',),'Bfr':('Bfr',),'mean EfBf':('Ef','Bf'),'mean EfBfr':('Ef','Bfr'),
            'Es':('Es',),'Bs':('Bs',),'Bsr':('Bsr',),'mean EsBs':('Es','Bs'),'mean EsBsr':('Es','Bsr'),'mean EsBr':('Es','Br'),
            'med Em+Vkf4+Bm':('Em','Vkf4','Bm'),'mean Em+Vkf4+Bm':('Em','Vkf4','Bm'),'med Em+Vkf2+Bm':('Em','Vkf2','Bm'),
            'med E+Vkf4+B+D':('E','Vkf4','B','D'),'mean E+Vkf4+B+D':('E','Vkf4','B','D'),
            'Ec':('Ec',),'Bc':('Bc',),'mean EcBc':('Ec','Bc'),'med Ec+Vkf4+Bc':('Ec','Vkf4','Bc'),'mean EcB':('Ec','B'),
            'XK':('XK',),'X0':('X0',),'med E+XK+B':('E','XK','B'),'med E+X0+B':('E','X0','B'),'mean E+XK+B':('E','XK','B'),
            'med Em+XK+Bm':('Em','XK','Bm'),'med E+Vkf4+XK':('E','Vkf4','XK'),'med E+Vkf4+XK+B':('E','Vkf4','XK','B'),
            'med Vkf4+XK+B':('Vkf4','XK','B'),'mean Vkf4+XK':('Vkf4','XK'),
            'Bn':('Bn',),'En':('En',),'med En+Vkf4+Bn':('En','Vkf4','Bn'),'med E+Vkf4+Bn':('E','Vkf4','Bn'),'mean Vkf4+Bn':('Vkf4','Bn'),
            'med Emn+Vkf4+Bmn':('Emn','Vkf4','Bmn'),'med Em+Vkf4+Bmn':('Em','Vkf4','Bmn')}
    COMBOS={n:es for n,es in COMBOS.items() if all(e in ES for e in es)}
    for e in ES:
        if e[0] in 'LVD': COMBOS.update({e:(e,),f'mean {e}+B':(e,'B'),f'mean E+{e}':('E',e),f'mean E+{e}+B':('E',e,'B'),f'med E+{e}+B':('E',e,'B')})
    for s in SPEEDS:
        print(f'\nLEAVE-ONE-CLIP-OUT, speed = {dict(v="ours (v5)",gv="GPS (course only)",c="CONSTANT")[s]}; per-estimator multiplier chosen on the other 7 with '
              +('constant speed' if s=='c' else 'OUR speed'))
        res={}
        for name,es in COMBOS.items():
            out=[]
            for k in KS:
                oth=[j for j in KS if j!=k]
                mm={e:min(M,key=lambda m: np.mean([single[(e,m,j,'c' if s=='c' else 'v')] for j in oth])) for e in es}
                Y=np.array([mm[e]*C[k]['ch'][e] for e in es])
                y=np.median(Y,0) if name.startswith('med') or len(es)==1 else Y.mean(0)
                out.append(score(k,y,s))
            res[name]=out
            print(f'  {name:15s} {np.median(out):5.2f}/{np.mean(out):5.2f}   '+' '.join(f'{x:5.2f}' for x in out))
        print('  clips:   '+' '.join(k[-3:] for k in KS))
        tool=[single[('E',0.65,k,s)] for k in KS]; print(f'  tool (E16 x0.65, chosen on all 8): {np.median(tool):5.2f}/{np.mean(tool):5.2f}   '+' '.join(f'{x:5.2f}' for x in tool))

def wide_eval(cam='wide',val=False,val_names=None):
    """VGGT course on `cam` (default wide; tool course = BA x YAW_MULT from final_v5 raw, our v5 speed): per-step yaw of
    each estimator (VGGT with BA fallback where no window), alone and averaged with BA; multipliers chosen on TRAIN
    (leave-one-clip-out for the train numbers); with val=True ONE val check with the multipliers from all train.
    val_names: only these combinations on val (the candidate chosen on train + the submitted one), and val clips
    without every Pi3X setting are left out (named) instead of dropping Pi3X everywhere."""
    TR=sorted(k for k,c in cl.items() if c['split']=='train' and c['camera_mode']==cam); VA=sorted(k for k,c in cl.items() if c['split']=='val' and c['camera_mode']==cam)
    D={}
    for k in TR+(VA if val else []):
        f=f'runs/final_v5/raw/{k}.npz' if os.path.exists(f'runs/final_v5/raw/{k}.npz') else f'runs/final_v4/raw/{k}.npz'
        d=np.load(f); T=d['times']; dts=d['dts']; rt,rx=load_reference('data/release',cl[k])
        v,_=LV.adaptive_speed(T,dts,[None if not np.isfinite(x) else x for x in d['spd']],d['quality'])
        gv=np.hypot(np.diff(np.interp(T,rt,rx[:,0])),np.diff(np.interp(T,rt,rx[:,1])))/dts
        est={'B':d['dyaw']}
        for cfg in VCFG:
            y=vggt_steps(k,len(dts),cfg)
            if y is None: continue
            assert np.array_equal(np.load(f'{VSRC}/{cfg}/{k}.npz')['times'],T), k
            est['V'+cfg[:3]]=np.where(np.isnan(y),d['dyaw'],y)
        for nm,cfg in PCFG.items():
            y=vggt_steps(k,len(dts),cfg,src=PSRC)
            if y is None: continue
            assert np.array_equal(np.load(f'{PSRC}/{cfg}/{k}.npz')['times'],T), k
            est[nm]=np.where(np.isnan(y),d['dyaw'],y)
        y=dpvo_steps(k,T)
        est['D']=y if y is not None else d['dyaw']                      # no DPVO (out of memory on a T4): BA's steps
        if y is None: print(f'  {k}: no DPVO -> BA steps in D')
        D[k]=dict(T=T,dts=dts,v=np.asarray(v),gv=gv,ref=(rt,rx),est=est)
    if val and val_names:
        gone=[k for k in VA if not all(nm in D[k]['est'] for nm in PCFG)]
        if gone: print(f'  val clips without every Pi3X setting, left out of the val check: {gone}')
        VA=[k for k in VA if k not in gone]
        for k in gone: del D[k]
    ES=[e for e in D[TR[0]]['est'] if all(e in D[k]['est'] for k in D)] if D else []
    if len(ES)<2: print('no VGGT results for',cam); return
    sc=lambda k,y,s: metrics.score_trajectory(D[k]['T'],path(D[k]['dts'],np.cumsum(y),D[k][s]),*D[k]['ref'])['drift_pct']
    COMB={e:(e,) for e in ES}; COMB.update({f'mean B+{e}':('B',e) for e in ES if e!='B'})
    if 'D' in ES: COMB.update({'mean B+Vkf4+D':('B','Vkf4','D'),'med B+Vkf4+D':('B','Vkf4','D')})
    for x in ('XK','X0'):
        if x in ES: COMB.update({f'mean B+Vkf4+{x}':('B','Vkf4',x),f'med B+Vkf4+{x}':('B','Vkf4',x)})
    single={(e,m,k):sc(k,m*D[k]['est'][e],'v') for e in ES for m in M for k in D}
    def mult(e,ks): return min(M,key=lambda m: np.mean([single[(e,m,j)] for j in ks]))
    def comb(k,es,mm,s,name=''):
        Y=[mm[e]*D[k]['est'][e] for e in es]
        return sc(k,(np.median(Y,0) if name.startswith('med') else np.mean(Y,0)),s)
    tool=LV.YAW_MULT.get(cam,1.0)
    print(f'{cam}: {len(TR)} train clips; tool = BA x {tool}')
    for s in ('v','gv'):
        print(f' TRAIN leave-one-clip-out, speed = {"ours (v5)" if s=="v" else "GPS (course only)"}')
        x=[sc(k,tool*D[k]['est']['B'],s) for k in TR]; print(f'   tool (BA x {tool})        {np.median(x):5.2f}/{np.mean(x):5.2f}')
        for name,es in COMB.items():
            x=[comb(k,es,{e:mult(e,[j for j in TR if j!=k]) for e in es},s,name) for k in TR]
            print(f'   {name:22s} {np.median(x):5.2f}/{np.mean(x):5.2f}   multipliers on all train: '+' '.join(f'{e} {mult(e,TR):.2f}' for e in es))
    if val:
        un=[k for k in VA if k not in ('clip_005','clip_016','clip_066')]
        print(f' VAL ONCE ({len(VA)} clips; 21 unseen in development: {len(un)}), multipliers from all train, our speed')
        x=np.array([sc(k,tool*D[k]['est']['B'],'v') for k in VA]); xu=np.array([sc(k,tool*D[k]['est']['B'],'v') for k in un])
        print(f'   tool (BA x {tool})        {np.median(x):5.2f}/{np.mean(x):5.2f}   unseen {np.median(xu):5.2f}/{np.mean(xu):5.2f}')
        R={}
        for name,es in COMB.items():
            if val_names and name not in val_names: continue
            mm={e:mult(e,TR) for e in es}; y=np.array([comb(k,es,mm,'v',name) for k in VA]); yu=np.array([comb(k,es,mm,'v',name) for k in un]); R[name]=y
            print(f'   {name:22s} {np.median(y):5.2f}/{np.mean(y):5.2f}   unseen {np.median(yu):5.2f}/{np.mean(yu):5.2f}   better {np.sum(y<x-1e-6)} worse {np.sum(y>x+1e-6)} (vs tool)')
        if val_names and len(val_names)==2 and all(n in R for n in val_names):
            a,b=R[val_names[0]],R[val_names[1]]; i=int(np.argmax(b-a))
            print(f'   {val_names[1]} vs {val_names[0]}: better {np.sum(b<a-1e-6)} worse {np.sum(b>a+1e-6)}, mean {a.mean():.2f} -> {b.mean():.2f}, '
                  f'largest worsening {VA[i]} {a[i]:.2f} -> {b[i]:.2f}')
            for k,p,q in zip(VA,a,b): print(f'     {k} {p:6.2f} -> {q:6.2f}'+('   (seen in development)' if k not in un else ''))

WBA={'B0':(None,False),'Br':(None,True),'Bf':('follow',False),'Bfr':('follow',True)}
def wide_ba_job(kv):
    k,e=kv; rule,ref=WBA[e]
    return ba_any(k,src='runs/lt_rot_wide',cache=f'runs/lt_wba_{e}',vraw=f'runs/final_v4/raw/{k}.npz',rule=rule,refine=ref)
def wide_ba(val=False,variants=('B0','Br','Bf','Bfr'),procs=3):
    """Wide (tool course = BA x YAW_MULT['wide']): BA run offline on the tool's tracks (runs/lt_rot_wide, e16_course.py
    --no-emat --save-tracks), step length from final_v4 raw speed. B0 = plain (as the tool; differs from the tool's course by
    <= 0.0013 rad/step because the tool's BA gets the horizon-method speed and the raw keeps the water-flow one), Br = with
    the second pass (LV.BA_REFINE), Bf = without the tracks fixed_tracks.py rule 'follow' flags, Bfr = both. Scored with
    the v5 speed (as wide_eval); multiplier per variant chosen on TRAIN (leave-one-clip-out for the train numbers);
    val=True: the ONE val check with multipliers from all train."""
    TR=sorted(k for k,c in cl.items() if c['split']=='train' and c['camera_mode']=='wide'); VA=sorted(k for k,c in cl.items() if c['split']=='val' and c['camera_mode']=='wide')
    ks=TR+(VA if val else []); miss=[k for k in ks if not os.path.exists(f'runs/lt_rot_wide/{k}.npz')]
    assert not miss, f'no tracks for {miss}'
    with Pool(procs) as p: p.map(wide_ba_job,[(k,e) for e in variants for k in ks],chunksize=1)
    D={}
    for k in ks:
        d=np.load(f'runs/final_v5/raw/{k}.npz'); T=d['times']; dts=d['dts']; rt,rx=load_reference('data/release',cl[k])
        v,_=LV.adaptive_speed(T,dts,[None if not np.isfinite(x) else x for x in d['spd']],d['quality'])
        gv=np.hypot(np.diff(np.interp(T,rt,rx[:,0])),np.diff(np.interp(T,rt,rx[:,1])))/dts
        est={'tool':d['dyaw']}; est.update({e:wide_ba_job((k,e)) for e in variants})
        D[k]=dict(T=T,dts=dts,v=np.asarray(v),gv=gv,ref=(rt,rx),est=est)
    sc=lambda k,y,s: metrics.score_trajectory(D[k]['T'],path(D[k]['dts'],np.cumsum(y),D[k][s]),*D[k]['ref'])['drift_pct']
    ES=('tool',)+tuple(variants)
    single={(e,m,k):sc(k,m*D[k]['est'][e],'v') for e in ES for m in M for k in ks}
    mult=lambda e,kk: min(M,key=lambda m: np.mean([single[(e,m,j)] for j in kk]))
    tm=LV.YAW_MULT.get('wide',1.0)
    print(f'wide: {len(TR)} train clips; tool = BA x {tm} (course as submitted)')
    for s in ('v','gv'):
        print(f' TRAIN, speed = {"ours (v5)" if s=="v" else "GPS (course only)"}: at x{tm} | leave-one-clip-out multiplier | multiplier on all train')
        for e in ES:
            a=[sc(k,tm*D[k]['est'][e],s) for k in TR]; b=[sc(k,mult(e,[j for j in TR if j!=k])*D[k]['est'][e],s) for k in TR]
            print(f'   {e:5s} {np.median(a):5.2f}/{np.mean(a):5.2f} | {np.median(b):5.2f}/{np.mean(b):5.2f} | {mult(e,TR):.2f}')
    if val:
        un=[k for k in VA if k not in ('clip_005','clip_016','clip_066')]
        print(f' VAL ONCE ({len(VA)} clips; unseen {len(un)}), our speed, multiplier from all train')
        x=np.array([sc(k,tm*D[k]['est']['tool'],'v') for k in VA]); xu=np.array([sc(k,tm*D[k]['est']['tool'],'v') for k in un])
        print(f'   tool x{tm}  {np.median(x):5.2f}/{np.mean(x):5.2f}   unseen {np.median(xu):5.2f}/{np.mean(xu):5.2f}')
        for e in variants:
            m=mult(e,TR); y=np.array([sc(k,m*D[k]['est'][e],'v') for k in VA]); yu=np.array([sc(k,m*D[k]['est'][e],'v') for k in un])
            print(f'   {e:5s} x{m:.2f} {np.median(y):5.2f}/{np.mean(y):5.2f}   unseen {np.median(yu):5.2f}/{np.mean(yu):5.2f}   better {np.sum(y<x-1e-6)} worse {np.sum(y>x+1e-6)}')

if __name__=='__main__':
    if sys.argv[1:2]==['test']: test_predictions()
    elif sys.argv[1:2]==['test-cons']: test_predictions(out='runs/test_v5_cons',mode='cons')
    elif sys.argv[1:2]==['test-ebm']: test_predictions(out='runs/test_v5_ebm',mE=0.60,mB=0.45,masked=True)
    elif sys.argv[1:2]==['test-var']:
        nm,mk,rf,mE,mB=sys.argv[2:7]; test_predictions(out=f'runs/test_v5_{nm}',mE=float(mE),mB=float(mB),masked={'none':False}.get(mk,mk),refine=rf=='1')
    elif sys.argv[1:2]==['test-evb']: test_predictions(out='runs/test_v5_evb',mode='evb')           # VGGT on flir-VIS only
    elif sys.argv[1:2]==['test-evbw']: test_predictions(out='runs/test_v5_evbw',mode='evb',wideV=0.80)  # + VGGT on wide
    elif sys.argv[1:2]==['test-evbmw']: test_predictions(out='runs/test_v5_evbmw',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80)  # mask + VGGT
    elif sys.argv[1:2]==['pi3']:        # flir-VIS: Pi3X (with / without our K) vs VGGT in the course combinations
        ONLY={'E','B','Vkf4','Em','Bm','XK','X0'}; main()
    elif sys.argv[1:2]==['focal']:      # flir-VIS: E16 / BA rescaled per clip to VGGT's turn scale (focal self-calibration)
        ONLY={'E','B','Vkf4','Em','Bm','En','Bn','Emn','Bmn'}; main()
    elif sys.argv[1:2]==['const']:      # flir-VIS: our water speed vs CONSTANT speed (the leaders' choice on FLIR)
        SPEEDS=('v','c','gv'); ONLY={'E','B','Vkf4','Em','Bm'}; main()
    elif sys.argv[1:2]==['test-evbmwg']: test_predictions(out='runs/test_v5_evbmwg',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,t0_flirvis=0.40)
    elif sys.argv[1:2]==['test-evbmwo']: test_predictions(out='runs/test_v5_evbmwo',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,offset_flirvis='runs/fvt_hstep/raw')
    elif sys.argv[1:2]==['test-evbmwc']: test_predictions(out='runs/test_v5_evbmwc',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,c5_flirvis=0.45)
    elif sys.argv[1:2]==['test-evbmwco']: test_predictions(out='runs/test_v5_evbmwco',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,c5_flirvis=0.45,offset_flirvis='runs/fvt_hstep/raw')
    elif sys.argv[1:2]==['test-focal']:   # NAME G [co]: evbmw (or evbmwco with 'co') with flir-VIS E16/BA multipliers x G
        nm,g=sys.argv[2],float(sys.argv[3]); co=sys.argv[4:5]==['co']
        test_predictions(out=f'runs/test_v5_{nm}',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,fscale_flirvis=g,
                         **(dict(c5_flirvis=0.45,offset_flirvis='runs/fvt_hstep/raw') if co else {}))
    elif sys.argv[1:2]==['test-vco']:     # flir-VIS course = VGGT alone (K-free), speed as evbmwco (C5 + offset), wide as evbmw
        test_predictions(out='runs/test_v5_vco',mode='v',masked=True,mE=0.60,mB=0.45,wideV=0.80,c5_flirvis=0.45,offset_flirvis='runs/fvt_hstep/raw')
    elif sys.argv[1:2]==['pspeed']: pose_speed_eval(*(sys.argv[2:4] or ('flir-VIS','train')))   # pspeed [cam [split]]
    elif sys.argv[1:2]==['test-c5all']:    # evbmwco with alpha 0 on EVERY flir-VIS clip with a finite C (adds 107, 111)
        test_predictions(out='runs/test_v5_c5all',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,c5_flirvis=0.0,offset_flirvis='runs/fvt_hstep/raw')
    elif sys.argv[1:2]==['test-wt0']:      # evbmwco + wide speed gate T0 = argv[2] (train 0.25; 0.10 keeps water on 6/8 test wide)
        g=float(sys.argv[2]); test_predictions(out=f'runs/test_v5_wt0_{int(round(g*100)):02d}',mode='evb',masked=True,mE=0.60,mB=0.45,wideV=0.80,c5_flirvis=0.45,offset_flirvis='runs/fvt_hstep/raw',t0_wide=g)
    elif sys.argv[1:2]==['test-straight']:   # BASE OUT [mode ...]: localization file, e.g. test-straight runs/test_v5_evbmw runs/test_v5_evbmw_swide wide
        straight_group(sys.argv[2],sys.argv[3],tuple(sys.argv[4:]) or ('wide',))
    elif sys.argv[1:2]==['wide-ba']: wide_ba(val='--val' in sys.argv)
    elif sys.argv[1:2]==['wide']:       # wide [--val [--only NAME1 NAME2]]: NAME1 = submitted, NAME2 = candidate from train
        vn=sys.argv[sys.argv.index('--only')+1:] if '--only' in sys.argv else None
        wide_eval(val='--val' in sys.argv,val_names=vn)
    else: main()
