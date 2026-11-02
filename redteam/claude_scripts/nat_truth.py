"""E 的自然版：不注入，用真值算节点真实偏移，预测 vs 实际（真值只用于这个检验）。SLQA 报警的幅度。L2 归因口径分叉。S 未拒的窗口。"""
from rtc import *
import json,collections
from statistics import NormalDist
N=NormalDist()
def pover(z): return 1-(N.cdf(3-z)-N.cdf(-3-z))
A1=json.load(open(W+"/a1_z.json"))
for name in("SLQA","BIPM"):
    z=np.concatenate([np.array(v) for v in A1[name].values()])
    lam=np.median(z**2)/.4549
    print(f"{name}: λ {lam:.2f} 除以 √λ 后 |z|>3 {np.mean(np.abs(z)/math.sqrt(lam)>3):.3f}（原 {np.mean(np.abs(z)>3):.3f}）")
for name in("SLQA","BIPM"):
    W_=windows_cached(name); P=[];Al=[];shs=[];mds=[]
    for cal,ev in W_:
        Rc,Ac,Bc=unpack(cal);Re,Ae,Be=unpack(ev)
        Sc,Se=series_np(Rc,Ac,Bc),series_np(Re,Ae,Be)
        for j in range(Rc.shape[1]):
            c,e=Sc[("nodeA",j)],Se[("nodeA",j)]
            cc=c[~np.isnan(c)];ee=e[~np.isnan(e)]
            if len(cc)<5 or len(ee)<5: continue
            m=mad(cc); se=1.2533*m*math.sqrt(1/len(cc)+1/len(ee))
            t=np.nanmedian(Re[:,j])-np.nanmedian(Rc[:,j])   # 相对真值的真实偏移
            P.append(pover(abs(t)/se)); Al.append(fstat(c,e)[0]>3); shs.append(t); mds.append(m)
    P,Al,shs,mds=np.array(P),np.array(Al),np.array(shs),np.array(mds)
    print(f"\n{name} 自然：{len(P)} 条 节点−秤 序列；预测均值 {P.mean():.3f} 实际 {Al.mean():.3f}；AUC(P→报警) {sum((a>b)+.5*(a==b) for a in P[Al] for b in P[~Al])/(Al.sum()*(~Al).sum()):.3f}")
    for lo,hi in((0,.01),(.01,.1),(.1,.5),(.5,.9),(.9,1.001)):
        s=(P>=lo)&(P<hi)
        if s.sum(): print(f"  P∈[{lo},{hi}): {s.sum():4d} 条 预测 {P[s].mean():.3f} 实际 {Al[s].mean():.3f}")
    print(f"  报警序列的真实偏移 / 节点MAD 中位 {np.median(np.abs(shs[Al])/mds[Al]):.2f}；真实偏移绝对值中位 {np.median(np.abs(shs[Al])):.4g}；全部序列 {np.median(np.abs(shs)):.4g}；节点 MAD 中位 {np.median(mds):.4g}")
# S 未拒
MB=json.load(open(W+"/map_bipm.json")); JB=json.load(open(REPO+"/bipm/validate_bipm_test.json"))
S=JB["S"]
print("\nS 格未拒：",[(MB[i//2]["g"],MB[i//2]["w"],S[i]["h"]*2) for i in range(len(S)) if not S[i]["refuse"]])
# L2 归因口径分叉
import ladder2 as LL
for name,w_,ds,a,b in (("SLQA",windows_cached("SLQA"),(0.05,0.10,0.20),3,5),("BIPM",windows_cached("BIPM"),(6.,12.,24.),4,6)):
    for dl in ds:
        A_=[];S_=[]
        for cal,ev in w_:
            A_.append(LL.scores(cal,ev,lambda t,j,dl=dl:dl,0.)["ca"]); S_.append(LL.scores(cal,ev,lambda t,j:0.,dl)["ca"])
        cand=sorted(set(A_+S_))
        best=max((np.mean(np.array(A_)>c)+np.mean(np.array(S_)<=c))/2 for c in cand)
        best=max(best,max((np.mean(np.array(A_)<=c)+np.mean(np.array(S_)>c))/2 for c in cand))
        def fold(a1,s1,a2,s2):
            c=max((np.mean(np.array(a1)>x)+np.mean(np.array(s1)<=x),x) for x in sorted(set(a1+s1)))[1]
            return (np.mean(np.array(a2)>c)+np.mean(np.array(s2)<=c))/2
        cv=(fold(A_[::2],S_[::2],A_[1::2],S_[1::2])+fold(A_[1::2],S_[1::2],A_[::2],S_[::2]))/2
        print(f"{name} δ={dl}: L2 归因 同样本最优门槛（v1 口径）{best:.0%}；两折交叉（v2 口径）{cv:.0%}；门槛 ≤60%")
