# Stage 4: the wide reversing rule (reversing_rule) with L-s ramps before and after each run, on the wide clips of SUB -> OUT.
import os,sys,json,numpy as np
sys.argv=['x']; sys.path[:0]=['tools/lookout_combo',os.path.dirname(os.path.abspath(__file__))]
from reversing_rule import windows
L=float(os.environ['L'])
sub=json.load(open(os.environ['SUB'])); man={c['key']:c for c in json.load(open('data/test/release_manifest.json'))['clips']}
od=os.environ['OUT']+'/predictions'; os.makedirs(od,exist_ok=True)
for k,p in sorted(sub.items()):
    q=dict(p)
    if man[k]['camera_mode']=='wide':
        d=dict(np.load(f'runs/test_v5/raw/{k}.npz')); T=d['times']; t,st,keep,rel=windows(T,d)
        if keep.any():
            P=np.asarray(p['positions'],float); dP=np.diff(P[:,:2],axis=0); dt=np.diff(np.asarray(p['times'],float)); Ls=np.hypot(*dP.T); vm=np.median(Ls/dt)
            tt=np.asarray(p['times'],float)[1:]-p['times'][0]; sc=np.ones(len(dP)); i=0; R=[]
            while i<len(keep):
                if keep[i]:
                    j=i
                    while j<len(keep) and keep[j]: j+=1
                    R.append((st[i],st[j-1]+5)); i=j
                else: i+=1
            for a,b in R:
                m=(tt>=a-L)&(tt<a); sc[m]=np.clip((a-tt[m])/L,0,1); m=(tt>=b)&(tt<b+L); sc[m]=np.clip((tt[m]-b)/L,0,1)
            for s,kk,r in zip(st,keep,rel):
                if kk: m=(tt>=s)&(tt<s+5); sc[m]=-abs(r)*vm*dt[m]/np.maximum(Ls[m],1e-9)
            Q=np.vstack([P[:1,:2],P[:1,:2]+np.cumsum(dP*sc[:,None],0)]); q['positions']=np.c_[Q,P[:,2]].tolist()
            print(k,'reversing',[(f'{int(a)//60}:{int(a)%60:02d}',f'{int(b)//60}:{int(b)%60:02d}') for a,b in R],f'ramps {L:.0f} s before and after')
    json.dump(q,open(f'{od}/{k}.json','w'))
