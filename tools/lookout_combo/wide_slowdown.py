# Stage 2: the wide slow-down rule Z (course_combo.stop_rel, detector horizon) on the wide clips with tool alpha < 0.5 -- each step of SUB x the rule's
# relative speed (course unchanged) -> OUT.
import os,sys,io,json,contextlib,numpy as np
sys.argv=['x']; sys.path.insert(0,'tools/lookout_combo')
with contextlib.redirect_stdout(io.StringIO()): import course_combo as CC
import lookout_vo as LV
sub=json.load(open(os.environ['SUB'])); man={c['key']:c for c in json.load(open('data/test/release_manifest.json'))['clips']}
od=os.environ['OUT']+'/predictions'; os.makedirs(od,exist_ok=True)
for k,p in sorted(sub.items()):
    q=dict(p)
    if man[k]['camera_mode']=='wide':
        d=dict(np.load(f'runs/test_v5/raw/{k}.npz')); T=d['times']; _,a=LV.adaptive_speed(T,d['dts'],[None if not np.isfinite(s) else s for s in d['spd']],d['quality'])
        P=np.asarray(p['positions'],float); assert len(P)==len(T), k
        if a<0.5:
            rel,_,fs=CC.stop_rel(T,d,min_good=0.5); rel=np.asarray(rel,float)[:len(P)-1]; dP=np.diff(P[:,:2],axis=0)
            Q=np.vstack([P[:1,:2],P[:1,:2]+np.cumsum(dP*rel[:,None],0)]); q['positions']=np.c_[Q,P[:,2]].tolist()
            t=np.asarray(T[1:],float); sl=[f'{int(s)//60}:{int(s)%60:02d}' for s in range(0,300,10) if np.median(rel[(t>=s)&(t<s+10)])<0.95]
            print(f'{k} alpha {a:.2f}: rule, slow windows {fs:.2f}; 10-s windows below 0.95: {", ".join(sl) if sl else "none"}; min {rel.min():.2f}')
        else: print(f'{k} alpha {a:.2f}: unchanged (alpha >= 0.5)')
    json.dump(q,open(f'{od}/{k}.json','w'))
