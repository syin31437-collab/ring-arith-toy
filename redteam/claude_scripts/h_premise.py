"""C：ζ 故障的大小、ζ<1、间歇、对照实际预测误差的扫描（只测 S 不成立，软裁判）。"""
from rtc import *
import torch,json,collections
import law_v6 as L, validate_bipm as V
from premise2 import decide
from analytic import sigma_c
from hetero import measure
Wn=list(V.windows("test")); rg=np.random.default_rng(13); R=[]
def run(cal,ev,h,shn,sha):
    c=[(False,a,r,2*h) for _,a,_,r in cal]
    e=[(True,a+float(sha[t]),r+torch.tensor(shn[t],dtype=r.dtype),2*h) for t,(_,a,_,r) in enumerate(ev)]
    d=decide(c,e,[b for _,_,b,_ in cal],[b for _,_,b,_ in ev],h)
    s0=max(sigma_c(c)[1],1e-6)
    return d["refuse"],d["s1"],L.predict(c,e,"软",s0),measure(c+e,2*h,"软",s0)["miss"]
for gi,w,cal,ev in Wn:
    T=len(ev);n=9
    for dl in(1.,2.,3.,6.,12.):
        idx=rg.choice(n,4,replace=False); S=np.zeros((T,n));S[:,idx]=dl
        for tag,sa,h in (("ζ=1 恒定 h=δ/2",np.full(T,dl),dl/2),("ζ=.5 h=δ/2",np.full(T,.5*dl),dl/2),("ζ=.25 h=δ/2",np.full(T,.25*dl),dl/2),("ζ=1 间歇50% h=δ/2",dl*(rg.random(T)<.5),dl/2),("ζ=1 逐题噪声CV=1 h=δ/2",dl*(1+rg.normal(0,1,T)),dl/2),("ζ=1 恒定 h=3 固定",np.full(T,dl),3.0)):
            r=run(cal,ev,h,S,sa); R.append(dict(tag=tag,dl=dl,w=w,refuse=r[0],s1=r[1],pt=r[2],act=r[3]))
json.dump(R,open(W+"/h_premise.json","w"),ensure_ascii=False)
tags=[]
for r in R:
    if r["tag"] not in tags:tags.append(r["tag"])
print("场景 / δ：拒答率 | S1 | 实测−预测（正=预测乐观） | 乐观≥.2 且没拒的比例")
for t in tags:
    for dl in(1.,2.,3.,6.,12.):
        s=[r for r in R if r["tag"]==t and r["dl"]==dl]
        print(f" {t:24s} δ={dl:<4g} 拒 {np.mean([r['refuse'] for r in s]):.2f} S1 {np.mean([r['s1'] for r in s]):.2f} 实测 {np.mean([r['act'] for r in s]):.2f} 预测 {np.mean([r['pt'] for r in s]):.2f} 乐观≥.2且未拒 {np.mean([(r['act']-r['pt']>=.2) and not r['refuse'] for r in s]):.2f}")
