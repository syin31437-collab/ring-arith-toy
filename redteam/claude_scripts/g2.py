from rtc import *
sys.path.insert(0,REPO+"/llm")
import constraint as C, collections
A=C.load(); pop,area=C.wb("SP.POP.TOTL"),C.wb("AG.SRF.TOTL.K2"); land=C.wb("AG.LND.TOTL.K2")
nodes=sorted(A); ids=sorted(i for i in pop if i in area and any(i in A[n]["pop"] for n in nodes))
T=math.log10(1.2)
def build(fix):
    R=[]
    for i in ids:
        ar=area[i]
        if fix and land.get(i) and area[i]/land[i]>1.2: ar=land[i]      # 世界银行总面积 ≫ 陆地面积（挪威/克罗地亚/多米尼加/法罗/百慕大/摩纳哥），用陆地面积当真值
        tr={"pop":math.log10(pop[i]/1e6),"area":math.log10(ar),"dens":math.log10(pop[i]/ar)}
        have=[n for n in nodes if all(i in A[n][q] for q in tr)]
        if len(have)<3: continue
        Lg={n:{q:math.log10(A[n][q][i]) for q in tr} for n in have}
        for n in have:
            e=max(abs(Lg[n][q]-tr[q]) for q in tr)
            R.append(dict(i=i,n=n,wrong=e>T,res=abs(Lg[n]["dens"]-(Lg[n]["pop"]+6-Lg[n]["area"])),rp=A[n]["pop"][i],pop=pop[i],peer=max(abs(Lg[n][q]-st.median(Lg[m][q] for m in have if m!=n)) for q in tr)))
    return R
for fix in (False,True):
    R=build(fix)
    AU=lambda s,k="res",sg=1:C.auc([sg*r[k] for r in s if r["wrong"]],[sg*r[k] for r in s if not r["wrong"]])
    wc=collections.Counter(r["i"] for r in R if r["wrong"])
    print(f"\n修标签={fix}: 正例 {sum(r['wrong'] for r in R)}/{len(R)} 国 {len(wc)} {wc.most_common(14)}")
    print(f"  约束 AUC {AU(R):.3f}；只看自报人口小 AUC {AU(R,'rp',-1):.3f}；同伴 AUC {AU(R,'peer'):.3f}；每国人口大小(真) {AU(R,'pop',-1):.3f}")
    for lo,hi,l in ((0,1e6,"<1M"),(1e6,1e12,"≥1M"),(5e6,1e12,"≥5M")):
        s=[r for r in R if lo<=r["pop"]<hi]; p=sum(r["wrong"] for r in s)
        print(f"  人口{l}: {len(s)} 例 正例 {p} 国 {len({r['i'] for r in s if r['wrong']})} 约束 AUC {AU(s) if p else float('nan'):.3f} 同伴 {AU(s,'peer') if p else float('nan'):.3f} 自报人口 {AU(s,'rp',-1) if p else float('nan'):.3f}")
    by=collections.defaultdict(list)
    for r in R: by[r["i"]].append(r)
    keys=list(by); rg=random.Random(2); bs=[];db=[]
    for _ in range(1000):
        s=[r for k in (rg.choice(keys) for _ in keys) for r in by[k]]
        a=AU(s);bs.append(a);db.append(a-AU(s,'rp',-1))
    print("  按国家 bootstrap AUC 95%",np.round(np.percentile(bs,[2.5,97.5]),3),"；约束−自报人口 95%",np.round(np.percentile(db,[2.5,97.5]),3))
    # 留一国
    lo=[AU([r for r in R if r["i"]!=k]) for k in wc]
    print("  留一有错国 AUC 范围",round(min(lo),3),round(max(lo),3))
    # 残差>20%命中：
    neg=[r for r in R if not r["wrong"]]; pos=[r for r in R if r["wrong"]]
    print(f"  残差>20%：没错的 {sum(r['res']>T for r in neg)}/{len(neg)}，错的 {sum(r['res']>T for r in pos)}/{len(pos)}；≥1M 国的错例中残差>20% {sum(r['res']>T for r in pos if r['pop']>=1e6)}/{sum(1 for r in pos if r['pop']>=1e6)}")
