"""B 的现实性：BIPM 上把 h 与 δ 解耦、把恒定注入换成间歇/噪声/混合符号/异幅/漂移。规律一字不改。"""
from rtc import *
import json, collections, itertools, torch
import law_v6 as L, validate_bipm as V
from analytic import sigma_c
from hetero import measure
from validate_v61 import baselines
Wn=list(V.windows("test"))
rg=np.random.default_rng(7); R=[]
def cell(tag,cal,ev,h,sh,zeta=None,judges=("软","硬2"),**meta):
    c=[(False,a,r,2*h) for _,a,_,r in cal]
    e=[(True,a,r+torch.tensor(sh[t],dtype=r.dtype),2*h) for t,(_,a,_,r) in enumerate(ev)]
    out=[]
    for j in judges:
        s0=max(sigma_c(c)[1],1e-6)
        pt=L.predict(c,e,j,s0); act=measure(c+e,2*h,j,s0)["miss"]; b1,b2=baselines(c,e,j,s0)
        out.append(dict(tag=tag,judge=j,h=h,act=act,pt=pt,b2=b2,b1=b1,**meta))
    return out
for wi,(gi,w,cal,ev) in enumerate(Wn):
    T=len(ev);n=9
    for k,dl in itertools.product((2,4,6),(3.,6.,12.)):
        idx=rg.choice(n,k,replace=False); base=np.zeros((T,n)); base[:,idx]=dl
        R+=cell("基线 h=δ/2",cal,ev,dl/2,base,wi=wi,k=k,dl=dl,w=w)
        R+=cell("解耦 h=1.5",cal,ev,1.5,base,wi=wi,k=k,dl=dl,w=w)
        R+=cell("解耦 h=3",cal,ev,3.0,base,wi=wi,k=k,dl=dl,w=w)
        mask=(rg.random((T,1))<.5); s=base*mask
        R+=cell("间歇50% h=δ/2",cal,ev,dl/2,s,wi=wi,k=k,dl=dl,w=w)
        mask=(rg.random((T,n))<.5); s=base*mask
        R+=cell("逐节点间歇50% h=δ/2",cal,ev,dl/2,s,wi=wi,k=k,dl=dl,w=w)
        s=base*(1+rg.normal(0,1,(T,n)))
        R+=cell("逐题噪声CV=1 h=δ/2",cal,ev,dl/2,s,wi=wi,k=k,dl=dl,w=w)
        sg=np.ones(n); sg[idx[:len(idx)//2]]=-1; s=base*sg
        R+=cell("混合符号 h=δ/2",cal,ev,dl/2,s,wi=wi,k=k,dl=dl,w=w)
        s=base*np.exp(rg.normal(0,1,n))
        R+=cell("异幅 σlog=1 h=δ/2",cal,ev,dl/2,s,wi=wi,k=k,dl=dl,w=w)
        ramp=(np.arange(1,T+1)/T*2)[:,None]; s=base*ramp
        R+=cell("漂移0→2δ h=δ/2",cal,ev,dl/2,s,wi=wi,k=k,dl=dl,w=w)
    print(wi,flush=True)
json.dump(R,open(W+"/b_real.json","w"),ensure_ascii=False)
