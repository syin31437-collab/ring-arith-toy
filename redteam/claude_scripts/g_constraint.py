"""G：约束 AUC 的驱动因素。"""
from rtc import *
sys.path.insert(0,REPO+"/llm")
import constraint as C, collections, json
A=C.load(); pop,area=C.wb("SP.POP.TOTL"),C.wb("AG.SRF.TOTL.K2")
land=C.wb("AG.LND.TOTL.K2")
nodes=sorted(A); ids=sorted(i for i in pop if i in area and any(i in A[n]["pop"] for n in nodes))
T=math.log10(1.2); R=[]
for i in ids:
    tr={"pop":math.log10(pop[i]/1e6),"area":math.log10(area[i]),"dens":math.log10(pop[i]/area[i])}
    have=[n for n in nodes if all(i in A[n][q] for q in tr)]
    if len(have)<3: continue
    Lg={n:{q:math.log10(A[n][q][i]) for q in tr} for n in have}
    for n in have:
        errs={q:Lg[n][q]-tr[q] for q in tr}
        err=max(abs(v) for v in errs.values())
        res=abs(Lg[n]["dens"]-(Lg[n]["pop"]+6-Lg[n]["area"]))
        peer=max(abs(Lg[n][q]-st.median(Lg[m][q] for m in have if m!=n)) for q in tr)
        worst=max(errs,key=lambda q:abs(errs[q]))
        R.append(dict(i=i,n=n,wrong=err>T,res=res,peer=peer,err=err,worst=worst,e=errs,pop=pop[i],area=area[i],land=land.get(i),raw={q:A[n][q][i] for q in tr},shared=max(abs(st.median(Lg[m][q] for m in have)-tr[q]) for q in tr)>T))
print(len(R),"节点-国家",len({r['i'] for r in R}),"国；错",sum(r['wrong'] for r in R),round(np.mean([r['wrong'] for r in R]),3))
auc=C.auc
AUC=lambda s,k="res":auc([r[k] for r in s if r["wrong"]],[r[k] for r in s if not r["wrong"]])
print("AUC",round(AUC(R),3))
by=collections.defaultdict(list)
for r in R: by[r["i"]].append(r)
wc=sorted(((sum(x["wrong"] for x in v),len(v),k) for k,v in by.items()),reverse=True)
print("错例按国家(前25):",[(k,a,b) for a,b,k in wc[:25]])
print("有错例的国家",sum(a>0 for a,_,_ in wc))
print("错例最差量:",collections.Counter(r["worst"] for r in R if r["wrong"]))
print("按节点错例:",collections.Counter(r["n"] for r in R if r["wrong"]))
# 同量定义问题：共有错例的国家，节点中位数 vs WB
print("\n前 15 国详情（WB 面积、WB 陆地面积、各节点面积中位、节点人口中位/WB、错的节点数）")
for a,b,k in wc[:15]:
    v=by[k]; ma=np.median([x["raw"]["area"] for x in v]); mp=np.median([x["raw"]["pop"] for x in v]); md=np.median([x["raw"]["dens"] for x in v])
    print(f"  {k}: 错{a}/{b} WB面积 {area[k]:.0f} WB陆地 {land.get(k)} 模型面积中位 {ma:.0f}; WB人口M {pop[k]/1e6:.3f} 模型 {mp:.3f}; 模型密度 {md:.1f} WB密度 {pop[k]/area[k]:.1f} 陆地算密度 {(pop[k]/land[k] if land.get(k) else float('nan')):.1f}; 残差中位 {np.median([x['res'] for x in v]):.3f}")
# 残差分析：定义失配 = 面积 vs 陆地
dfrac={k:(math.log10(area[k]/land[k]) if land.get(k) else 0) for k in by}
for thr in (0.01,0.02,0.05,0.08):
    inc=[r for r in R if abs(dfrac[r["i"]])<thr]
    print(f"只留 |log10(总面积/陆地)|<{thr}: {len(inc)} 例 错 {sum(r['wrong'] for r in inc)} AUC {AUC(inc):.3f}")
# 留一法影响
print("\n逐个移除错例最多的 k 国：")
order=[k for a,b,k in wc]
for m in (1,3,5,10,15,20,30):
    ex=set(order[:m]); s=[r for r in R if r["i"] not in ex]
    print(f"  移除前{m}国：{len(s)} 例，错 {sum(r['wrong'] for r in s)}，AUC {AUC(s):.3f}")
# 小国/取整
for lim,lab in ((1e6,"pop≥1M"),(5e6,"pop≥5M")):
    s=[r for r in R if r["pop"]>=lim]; print(f"  {lab}: {len(s)} 例 错 {sum(r['wrong'] for r in s)} AUC {AUC(s):.3f}")
s=[r for r in R if r["pop"]<1e6]; print(f"  pop<1M: {len(s)} 例 错 {sum(r['wrong'] for r in s)} 错率 {np.mean([r['wrong'] for r in s]):.3f} AUC {AUC(s):.3f}")
# 按错的量
for q in ("pop","area","dens"):
    p=[r["res"] for r in R if r["wrong"] and r["worst"]==q]; n=[r["res"] for r in R if not r["wrong"]]
    print(f"  最差量={q}: {len(p)} 例 AUC {auc(p,n):.3f}")
# 残差最大的例子
print("\n残差 > 20% 的节点-国家：")
hi=[r for r in R if r["res"]>T]
print(" 共",len(hi),"其中 wrong",sum(r["wrong"] for r in hi),"；国家",collections.Counter(r["i"] for r in hi).most_common(12),"节点",collections.Counter(r["n"] for r in hi).most_common())
neg=[r for r in R if not r["wrong"]]
print(" 没错的",len(neg)," 残差>20%：",sum(r["res"]>T for r in neg)," 没错的残差最大值",round(max(r["res"] for r in neg),4),f"(= {10**max(r['res'] for r in neg)-1:.1%})")
# 数学上界：没错意味着三量 |e|<T；残差 ≤ 3T；残差>T 需要误差恰好累积。自洽真值使 FP 受算术限制
print(" 没错的里最大 sum|e|:",round(max(sum(abs(v) for v in r["e"].values()) for r in neg),3),"，T=",round(T,3))
print(" 没错例的 |e| 中位:",round(float(np.median([r['err'] for r in neg])),4),"→ 残差 >T 需要 |e_d−e_p+e_a|>T；误差这么小时几乎不可能")
# 零模型：把错误的符号/量打乱（不靠模型自相矛盾，只靠真值自洽的算术）
rng=np.random.default_rng(0)
def sim_null(mode,B=200):
    fps=[];aucs=[];rec=[]
    for _ in range(B):
        ps,ns=[],[]; fp=0; hit=0
        for r in R:
            e=r["e"]
            if mode=="符号随机":
                v=[e[q]*rng.choice([-1,1]) for q in ("pop","area","dens")]
            res=abs(v[2]-(v[0]-v[1]))
            (ps if r["wrong"] else ns).append(res)
        aucs.append(auc(ps,ns)); fps.append(np.mean(np.array(ns)>T)); rec.append(np.mean(np.array(ps)>T))
    return np.mean(aucs),np.mean(fps),np.mean(rec)
print("\n零模型(保持每个节点-国家三量的误差大小，符号随机→错误不自洽)：AUC, 没错的残差>20%比例, 错的残差>20%比例:",np.round(sim_null("符号随机",60),3))
# 错得最小的基线：一个不看约束的预测器
def rank_pos(key,sgn=1):
    return auc([sgn*key(r) for r in R if r["wrong"]],[sgn*key(r) for r in R if not r["wrong"]])
print("\n无约束的平凡信号 AUC：")
print("  国家人口小(用模型自己报的) :",round(rank_pos(lambda r:r["raw"]["pop"],-1),3))
print("  面积小(模型自报)          :",round(rank_pos(lambda r:r["raw"]["area"],-1),3))
print("  密度极端|log−中位|(自报)  :",round(rank_pos(lambda r:abs(math.log10(r["raw"]["dens"])-2)),3))
print("  同伴信号                  :",round(AUC(R,"peer"),3))
# 组内 AUC：同国家内 / 同节点内
def strat(f):
    g=collections.defaultdict(list)
    for r in R: g[f(r)].append(r)
    num=den=0
    for k,v in g.items():
        p=[x["res"] for x in v if x["wrong"]];n=[x["res"] for x in v if not x["wrong"]]
        if p and n:
            w=len(p)*len(n); num+=auc(p,n)*w; den+=w
    return num/den,den
print("分层 AUC：同国家内",np.round(strat(lambda r:r["i"]),3),"同节点内",np.round(strat(lambda r:r["n"]),3))
# 国家 bootstrap
rg=random.Random(1); keys=list(by); bs=[]
for _ in range(1000):
    s=[r for k in (rg.choice(keys) for _ in keys) for r in by[k]]
    bs.append(AUC(s))
print("按国家 bootstrap 95%:",np.round(np.percentile(bs,[2.5,97.5]),3),"P(≥0.70)",np.mean(np.array(bs)>=.7))
# 0.0% 假阳性上界（按国家簇）
nc=len({r['i'] for r in neg}); nn=len(neg)
print(f"假阳性 0/{nn}：独立假设 95% 上界 {3/nn:.2%}；按国家（{nc}）上界 {3/nc:.2%}。阈值敏感：")
for th in (1.05,1.1,1.15,1.2,1.3,1.5):
    t=math.log10(th); fp=np.mean([r["res"]>t for r in neg]); hit=np.mean([r["res"]>t for r in R if r["wrong"]])
    fn=sum(r["res"]>t for r in neg); tp=sum(r["res"]>t for r in R if r["wrong"])
    print(f"  残差>{th-1:.0%}: 没错的 {fp:.2%}({fn}) 错的 {hit:.0%}({tp}) 精确率 {tp/max(tp+fn,1):.2f}")
# 错的定义阈值敏感
print("“错”定义改变：")
for th in (1.05,1.1,1.2,1.3,1.5):
    t=math.log10(th); p=[r["res"] for r in R if r["err"]>t]; n=[r["res"] for r in R if r["err"]<=t]
    fp=np.mean(np.array(n)>T); print(f"  错=误差>{th-1:.0%}: 正例 {len(p)} 比例 {len(p)/len(R):.1%} AUC {auc(p,n):.3f} 残差>20%的假阳性 {fp:.2%}")
json.dump([{k:r[k] for k in("i","n","wrong","res","peer","err","worst","shared")} for r in R],open(W+"/g_rows.json","w"))
