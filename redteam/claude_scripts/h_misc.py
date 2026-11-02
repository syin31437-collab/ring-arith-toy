"""剩余：E 噪声地板/有效组数；B 自然格 LOO、ECE 分层；C 拒答扫描 + 拒答是否对应错预测。"""
from rtc import *
import json,collections,pickle,torch
from e_eval import table,score,O
print("=== E 噪声地板 + 有效组数 ===")
rng=np.random.default_rng(11)
for name in("SLQA","BIPM"):
    G=table(name); K=list(G); sp,m,P,A,_=score(G)
    # 模型完全正确时（按窗 Bernoulli(p_w)，组内窗独立）MAE 的分布
    Mb=[];Sp=[]
    for _ in range(400):
        Asim=[np.mean(rng.random(len(G[k]))<np.array([1-(1-x[0])*(1-x[3]) for x in G[k]])) for k in K]
        Mb.append(np.mean(np.abs(np.array(Asim)-np.array(P)))); Sp.append(spearman(Asim,P))
    print(f" {name}: 观测 MAE {m:.3f} ρ {sp:.3f}；模型完全正确时 MAE {np.mean(Mb):.3f} [{np.percentile(Mb,2.5):.3f},{np.percentile(Mb,97.5):.3f}]，ρ {np.mean(Sp):.3f} [{np.percentile(Sp,2.5):.3f},{np.percentile(Sp,97.5):.3f}]；P(ρ≥.8)={np.mean(np.array(Sp)>=.8):.2f}")
    # 窗×组 矩阵的有效组数
    wins=sorted({x[2] for k in K for x in G[k]})
    M=np.array([[next(x[1] for x in G[k] if x[2]==w) for w in wins] for k in K],float)
    v=M.std(1)>0; C=np.corrcoef(M[v]); ev=np.linalg.eigvalsh(C); ev=ev[ev>0]
    print(f"   检出矩阵 组×窗：有效独立组数（参与率）{ev.sum()**2/(ev**2).sum():.1f} / {v.sum()}；窗口均值 SD {M.mean(0).std():.2f}")
print("\n=== B 自然格 ===")
JB=json.load(open(REPO+"/bipm/validate_bipm_test.json")); RB=JB["cells"]; MB=json.load(open(W+"/map_bipm.json"))
for i,r in enumerate(RB): r["c"]=i//22; r["w"]=MB[i//22]["w"]; r["g"]=MB[i//22]["g"]
N=[r for r in RB if r["tag"]=="自然"]
for j in("软","硬2"):
    for h in(1.5,3.0):
        s=[r for r in N if r["judge"]==j and r["h"]==h]
        a=np.array([r["act"] for r in s]);p=np.array([r["pt"] for r in s])
        lo=np.array([np.delete(a,i).mean() for i in range(len(a))])
        print(f" {j} h={h}: v6.1 MAE {np.abs(p-a).mean():.3f} ρ {spearman(p,a):+.2f}；留一常数 MAE {np.abs(lo-a).mean():.3f}；只用常数比 v6.1 好? {np.abs(lo-a).mean()<np.abs(p-a).mean()}")
s=N; a=np.array([r["act"] for r in s]);p=np.array([r["pt"] for r in s]);print(f" 自然合计 34 格 v6.1 MAE {np.abs(p-a).mean():.3f} ρ {spearman(p,a):.2f}")
# 留一常数 across
lo=[];
for r in s:
    o=[x["act"] for x in s if x["judge"]==r["judge"] and x["h"]==r["h"] and x["c"]!=r["c"]]; lo.append(np.mean(o))
print(f" 留一常数（裁判×h）MAE {np.mean(np.abs(np.array(lo)-a)):.3f} ρ {spearman(lo,a):.2f}")
# 配对 簇 bootstrap: v6.1 − 留一常数
cl=collections.defaultdict(list)
for r,l in zip(s,lo): cl[r["w"]].append(abs(r["pt"]-r["act"])-abs(l-r["act"]))
ks=list(cl); B=[np.mean(np.concatenate([cl[ks[i]] for i in rng.integers(len(ks),size=len(ks))])) for _ in range(3000)]
print(f" MAE(v6.1)−MAE(留一常数) {np.mean([x for v in cl.values() for x in v]):+.3f} 按时段 95% {np.round(np.percentile(B,[2.5,97.5]),3)}")
print("\n=== 校准 ECE 分层（是否被 q≈0 的题撑起来）===")
def ece_tab(items):
    bins=[[] for _ in range(10)]
    for q,y,_ in items: bins[min(int(q*10),9)].append((q,y))
    return sum(len(b)*abs(np.mean([x[0] for x in b])-np.mean([x[1] for x in b])) for b in bins if b)/len(items)
J6=json.load(open(REPO+"/validate_v61.json"))
for nm,I,R,per in (("BIPM",JB["items"],RB,None),("FMNIST",J6["items"],J6["cells"],None)):
    I=[(q,bool(y),p) for q,y,p in I]; q=np.array([x[0] for x in I]);y=np.array([x[1] for x in I],float)
    print(f" {nm}: 题 {len(I)} ECE {ece_tab(I):.3f}；q<0.1 的题 {np.mean(q<.1):.0%}（实际 {y[q<.1].mean():.3f}）；q∈[.1,.9] {np.mean((q>=.1)&(q<=.9)):.0%} ECE {ece_tab([x for x in I if .1<=x[0]<=.9]):.3f}；q>.9 {np.mean(q>.9):.0%} 平均预测 {q[q>.9].mean():.3f} 实际 {y[q>.9].mean():.3f}")
    print("   Brier",round(float(np.mean((q-y)**2)),3),"参考(用总体均值) Brier",round(float(y.mean()*(1-y.mean())),3),"BSS",round(1-float(np.mean((q-y)**2))/float(y.mean()*(1-y.mean())),3))
# BIPM ECE 按时段 bootstrap
I=JB["items"]; 
# items 顺序: 每 one() 的每个裁判 ev 条
pos=0; bycell=[]
# 窗口 eval 长度
evlen=[m["ne"] for m in MB]
cl=collections.defaultdict(list)
for ci,ne in enumerate(evlen):
    for c in range(22):
        cl[MB[ci]["w"]].append((pos,pos+ne)); pos+=ne
print(" items 对齐检查",pos,len(I))
ks=list(cl);E=[]
for _ in range(500):
    s=[x for k in rng.choice(ks,len(ks)) for a,b in cl[k] for x in I[a:b]]
    E.append(ece_tab(s))
print(" BIPM ECE 按时段 bootstrap 95%",np.round(np.percentile(E,[2.5,97.5]),3),"P(≤.05)",np.mean(np.array(E)<=.05))
ks=list(range(len(evlen)))
print("\n=== C：自检 ===")
ra=[r for r in RB]
err=np.array([abs(r["pt"]-r["act"]) for r in ra]);ref=np.array([r["refuse"] for r in ra],bool)
print(f" BIPM 前提成立格 {len(ra)}：拒 {ref.mean():.1%}；没拒 MAE {err[~ref].mean():.3f}，拒了 MAE {err[ref].mean():.3f}；误差>.15 的 {np.mean(err>.15):.0%}，其中被拒 {ref[err>.15].mean():.0%}（拒答的召回）；误差≤.15 被拒 {ref[err<=.15].mean():.1%}")
print(" 拒答对象：自然格拒",np.mean([r["refuse"] for r in ra if r["tag"]=="自然"]),"注入格",np.mean([r["refuse"] for r in ra if r["tag"]=="注入"]),"；S1 触发",np.mean([r["s1"] for r in ra]))
for nm,v in (("S 不成立",JB["S"]),("B 不成立",JB["B"])):
    e=np.array([abs(r["pt"]-r["act"]) for r in v]);f=np.array([r["refuse"] for r in v],bool)
    print(f" {nm} {len(v)} 格：拒 {f.mean():.0%}；误差中位 {np.median(e):.3f}；没拒的 MAE {e[~f].mean():.3f}（{(~f).sum()} 格）；s1 {np.mean([r['s1'] for r in v]):.0%} m1 {np.mean([r['m1'] for r in v]):.0%}")
