# The reversing rule (reversing_rule) with ramps: L s before a reversing run the speed falls linearly to 0; po: after it rises
# from 0 over L s.
import os, sys, numpy as np
sys.argv=['x']; sys.path[:0]=['tools/lookout_combo',os.path.dirname(os.path.abspath(__file__))]
from reversing_rule import apply
def runs(st,keep):
    out=[]; i=0
    while i<len(keep):
        if keep[i]:
            j=i
            while j<len(keep) and keep[j]: j+=1
            out.append((st[i],st[j-1]+5.0)); i=j
        else: i+=1
    return out
def ramp(t,st,keep,rel,v,L,po):
    u=apply(t,st,keep,rel,v)
    for a,b in runs(st,keep):
        m=(t>=a-L)&(t<a); u[m]=v[m]*np.clip((a-t[m])/L,0,1)
        if po: m=(t>=b)&(t<b+L); u[m]=v[m]*np.clip((t[m]-b)/L,0,1)
    return u
