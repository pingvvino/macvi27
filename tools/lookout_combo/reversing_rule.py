# The wide reversing rule: a 5-s window with negative raw water on >= 60 % of its steps AND the water law's R^2 (5 s, detector
# horizon) >= 0.9 AND median water <= -0.1 of the clip's level; only runs >= 10 s. There the boat goes astern at
# |water / level| * median(v).
import sys, io, contextlib, numpy as np
sys.argv=['x']; sys.path.insert(0,'tools/lookout_combo')
with contextlib.redirect_stdout(io.StringIO()): import course_combo as CC
def windows(T,d):
    t=np.asarray(T[1:],float)-T[0]; w=d['spd']/d['dts']; lvl=np.nanmedian(w[w>0]); st=np.arange(0,t[-1],5.0)
    _,tb,_=CC.stop_rel(T,dict(d),win=5.0,hz='rescue'); tb=np.asarray(tb); flag=[]; rel=[]
    for s in st:
        m=(t>=s)&(t<s+5); x=w[m]; x=x[np.isfinite(x)]
        if not len(x) or tb.ndim<2 or not len(tb): flag.append(False); rel.append(np.nan); continue
        r2=tb[np.argmin(np.abs(tb[:,0]-T[0]-(s+2.5))),2]; r=np.median(x)/lvl
        flag.append(np.mean(x<0)>=0.6 and r2>=0.9 and r<=-0.1); rel.append(r)
    flag=np.array(flag); keep=np.zeros_like(flag); i=0
    while i<len(flag):
        if flag[i]:
            j=i
            while j<len(flag) and flag[j]: j+=1
            if (j-i)*5>=10: keep[i:j]=True
            i=j
        else: i+=1
    return t,st,keep,np.array(rel)
def apply(t,st,keep,rel,v):
    u=v.copy(); vm=np.median(v)
    for s,kk,r in zip(st,keep,rel):
        if kk: m=(t>=s)&(t<s+5); u[m]=-abs(r)*vm
    return u
