"""Course only (the tool's own run_clip, method 'horizon' = no water flow; E16 for flir-VIS, rotation-only
elsewhere), half or full resolution, with parameter overrides, to compare course variants quickly.
Same pattern as rot_capture.py (run_clip with a monkeypatch): integrate() is replaced by a recorder of the
raw per-step yaw. python e16_course.py out_dir clip,clip,... [--full-res] [--lk-win N] [--fscale s]
--fscale: focal length (fx, fy) given to the course x s (the release calibration is nominal, FOV-derived).
--save-tracks: also keep the tool's long KLT tracks above the horizon (HorizonTracks, as the BA gets them;
the BA itself is replaced by a recorder, so the course stays E16/rotation-only) -> rows (frame,id,x,y), K.
--no-emat: rotation-only course also for flir-VIS (with --save-tracks: the BA's input, as for other cameras).
--root: dataset root (default data/release; data/test for the test clips).
--fixed-from DIR: exclude image areas FIXED TO THE BOAT from every corner detection (E16, rotation, BA tracks): tracks
of DIR/<clip>.npz (an earlier --save-tracks run) flagged by fixed_tracks.py, each observation dilated by 8 px.
--clahe CLIP,TILE: contrast-limited histogram equalisation (cv2.createCLAHE) of the gray frame for the COURSE only
(LV.COURSE_GRAY: rotation, E16, BA tracks); the water profile keeps the raw frame.
--fixed-rule still|follow|slow: fixed_tracks.py rule ('follow' also catches lens drops and people on deck; 'slow' = 'follow'
limited to tracks that move little -- drops, boat, overlays, not near scene objects).
--lg aliked,superpoint,sift: ALSO match each E16 chunk's first and last frame with that extractor + LightGlue
(cvg/LightGlue, pretrained weights; needs .venv_lg) and save every match: rows (chunk, x0, y0, x1, y1, above the
chunk's horizon mask in frame 0) per extractor as lg_<name>, plus Kc (3x3). The course itself stays the tool's E16."""
import sys,os,argparse,numpy as np,cv2
sys.path.insert(0,'data/release/starter'); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import lookout_vo as LV
from preprocessing import ClipPreprocessor
from public_dataset import load_manifest,load_timestamps
ap=argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('clips'); ap.add_argument('--full-res',action='store_true')
ap.add_argument('--lk-win',type=int); ap.add_argument('--fscale',type=float,default=1.0); ap.add_argument('--emat-n',type=int); ap.add_argument('--save-tracks',action='store_true'); ap.add_argument('--points-scale',type=float,default=1.0); ap.add_argument('--no-emat',action='store_true'); ap.add_argument('--root',default='data/release'); ap.add_argument('--lg'); ap.add_argument('--lg-max-kp',type=int,default=1024); ap.add_argument('--fixed-from'); ap.add_argument('--fixed-rule',default='still'); ap.add_argument('--clahe'); a=ap.parse_args()
if a.clahe:
    _cl=cv2.createCLAHE(clipLimit=float(a.clahe.split(',')[0]),tileGridSize=(int(a.clahe.split(',')[1]),)*2); LV.COURSE_GRAY=_cl.apply
for n in ('TRACK_MAX','TRACK_MIN','BA_MAX_TRACKS','ROT_MAX_CORNERS','EMAT_MAX_CORNERS'): setattr(LV,n,int(round(getattr(LV,n)*a.points_scale)))   # as the tool's --points-scale
if a.emat_n: LV.EMAT_N=a.emat_n                                # E16 chunk length (frames)
if a.full_res: LV.PX=2.0; LV.LK['maxLevel']=4                 # exactly what main() does for --full-res
if a.lk_win: LV.LK['winSize']=(a.lk_win,a.lk_win)
cl={c['key']:c for c in load_manifest(a.root+'/release_manifest.json')['clips']}
REC={}
def rec(times,dec,dyaw,dts,spd,method,quality=None):
    REC.update(times=np.asarray(times,float),dyaw=np.asarray(dyaw,float)); return dec,times,[[0,0,0]]*len(times),None
LV.integrate=rec
def ba_rec(rows,K,dyaw,step_dist):
    REC.update(rows=np.asarray(rows,np.float32),K=np.asarray(K,float)); return dyaw,None
LV.ba_course=ba_rec
LG={}
if a.lg:
    import torch
    from lightglue import LightGlue,ALIKED,SuperPoint,SIFT
    from lightglue.utils import rbd
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    MK={'aliked':ALIKED,'superpoint':SuperPoint,'sift':SIFT}
    for n in a.lg.split(','): LG[n]=(MK[n](max_num_keypoints=a.lg_max_kp).eval(),LightGlue(features=n).eval())
    _chunk=LV.chunk_yaw_essential; CACHE={}; CH={'i':0}
    def feats(n,g):
        key=(n,id(g))
        if key not in CACHE:
            x=torch.from_numpy(np.repeat(g[None],3,0) if n=='aliked' else g[None]).float()/255
            CACHE[key]=LG[n][0].extract(x)
        return CACHE[key]
    def chunk_lg(frames,K,hy,conf,slope=0.0,edge=None):
        y=_chunk(frames,K,hy,conf,slope,edge); g0,g1=frames[0],frames[-1]; i=CH['i']; CH['i']+=1
        mask=LV.edge_mask(g0.shape,edge) if edge is not None else (LV.above_line_mask(g0.shape,hy,slope) if conf>=LV.HORIZON_MIN_CONF else None)
        for n,(ex,m) in LG.items():
            f0,f1=feats(n,g0),feats(n,g1); r=m({'image0':f0,'image1':f1}); f0_,f1_,r=rbd(f0),rbd(f1),rbd(r)
            mt=r['matches'].numpy(); p0=f0_['keypoints'].numpy()[mt[:,0]]; p1=f1_['keypoints'].numpy()[mt[:,1]]
            up=np.ones(len(p0),bool) if mask is None else mask[np.clip(p0[:,1].astype(int),0,g0.shape[0]-1),np.clip(p0[:,0].astype(int),0,g0.shape[1]-1)]>0
            REC.setdefault('lg_'+n,[]).append(np.c_[np.full(len(p0),i),p0,p1,up].astype(np.float32))
        for kk in [kk for kk in CACHE if kk[1]!=id(g1)]: del CACHE[kk]
        REC['Kc']=np.asarray(K,float); return y
    LV.chunk_yaw_essential=chunk_lg
os.makedirs(a.out,exist_ok=True)
FM={'m':None,'pts':None}
if a.fixed_from:
    from fixed_tracks import fixed_track_ids
    _gftt=cv2.goodFeaturesToTrack
    def gftt(img,*args,mask=None,**kw):
        if FM['pts'] is not None:
            if FM['m'] is None or FM['m'].shape!=img.shape[:2]:          # built at the frames' own size
                FM['m']=np.full(img.shape[:2],255,np.uint8)
                for x,y in FM['pts']: cv2.circle(FM['m'],(int(round(x)),int(round(y))),8,0,-1)
                print('  fixed mask', img.shape[:2], f'{(FM["m"]==0).mean():.1%} of the image excluded', flush=True)
            mask=FM['m'] if mask is None else cv2.bitwise_and(mask,FM['m'])
        return _gftt(img,*args,mask=mask,**kw)
    cv2.goodFeaturesToTrack=gftt                                 # lookout_vo calls cv2.goodFeaturesToTrack at run time
for k in a.clips.split(','):
    if os.path.exists(f'{a.out}/{k}.npz'): continue
    c=cl[k]
    if a.fixed_from:
        rr=np.load(f'{a.fixed_from}/{k}.npz')['rows']; fx=fixed_track_ids(rr,a.fixed_rule); sel=rr[np.isin(rr[:,1].astype(int),list(fx))]
        FM['pts']=sel[:,2:4]; FM['m']=None; print(k,'fixed tracks:',len(fx),flush=True)
    pre=ClipPreprocessor(a.root+'/'+c['video'],c['camera_mode'],8.0,half_res=not a.full_res,calibration=c['calibration'],frame_timestamps=load_timestamps(a.root,c))
    if a.fscale!=1.0:
        class Scaled:
            def __init__(s,p): s.p=p; s.intrinsics=p.intrinsics
            def frames_with_time(s):
                for i,img,intr,t,d in s.p.frames_with_time():
                    intr=np.array(intr,np.float32).copy(); intr[:2]*=a.fscale; yield i,img,intr,t,d
        pre=Scaled(pre)
    LV.run_clip(pre,'horizon',None,emat=c['camera_mode'] in LV.EMAT_MODES and not a.no_emat,ba=a.save_tracks,yaw_mult=1.0,tilt=False)
    for n in LG: REC['lg_'+n]=np.concatenate(REC['lg_'+n]) if REC.get('lg_'+n) else np.zeros((0,6),np.float32)
    np.savez(f'{a.out}/{k}.npz',**REC,emat_n=LV.EMAT_N); print(k,len(REC['dyaw']),flush=True)
    REC.clear()
    if LG: CH['i']=0; CACHE.clear()
