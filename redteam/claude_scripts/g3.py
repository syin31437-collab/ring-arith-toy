from rtc import *
sys.path.insert(0,REPO+"/llm")
import constraint as C, collections, json
R=json.load(open(W+"/g_rows.json"))
A=C.load(); pop=C.wb("SP.POP.TOTL")
T=math.log10(1.2); BAD={"NOR","HRV","FRO","BMU","MCO","GRL","DOM","CAN"}
for r in R: r["rp"]=A[r["n"]]["pop"][r["i"]]; r["pop"]=pop[r["i"]]
AU=lambda s,k="res",sg=1:C.auc([sg*r[k] for r in s if r["wrong"]],[sg*r[k] for r in s if not r["wrong"]])
print("全部：正例",sum(r["wrong"] for r in R),"其中 WB 面积异常 8 国",sum(r["wrong"] and r["i"] in BAD for r in R))
def rep(tag,s):
    p=[r for r in s if r["wrong"]]; n=[r for r in s if not r["wrong"]]
    print(f"{tag}: {len(s)} 例，正例 {len(p)}（{len({r['i'] for r in p})} 国），约束 AUC {AU(s):.3f}，同伴 {AU(s,'peer'):.3f}，自报人口小 {AU(s,'rp',-1):.3f}；自洽地错(残差<5%) {np.mean([r['res']<math.log10(1.05) for r in p]):.0%}；残差>20% 抓到 {np.mean([r['res']>T for r in p]):.0%}；假阳 {sum(r['res']>T for r in n)}/{len(n)}；共有 {np.mean([r['shared'] for r in p]):.0%}")
rep("原",R)
S=[r for r in R if r["i"] not in BAD]; rep("去 8 国",S)
rep("  其中只有 WB 8 国的正例+全部负例",[r for r in R if r["i"] in BAD or not r["wrong"]])
by=collections.defaultdict(list)
for r in S: by[r["i"]].append(r)
keys=list(by); rg=random.Random(3); bs=[];bp=[]
for _ in range(2000):
    s=[r for k in (rg.choice(keys) for _ in keys) for r in by[k]]
    if sum(r["wrong"] for r in s)==0: continue
    bs.append(AU(s));bp.append(AU(s)-AU(s,"rp",-1))
print(" 去 8 国按国家 bootstrap AUC 95%",np.round(np.percentile(bs,[2.5,97.5]),3),"；约束−自报人口",np.round(np.percentile(bp,[2.5,97.5]),3))
wc=collections.Counter(r["i"] for r in S if r["wrong"]); print(" 正例国",wc.most_common())
lo=[(k,round(AU([r for r in S if r["i"]!=k]),3)) for k in wc]; print(" 留一国 AUC",lo)
for lim,l in((1e6,"≥1M"),):
    s=[r for r in S if r["pop"]>=lim]; print(f" 人口{l}: {len(s)} 例 正例 {sum(r['wrong'] for r in s)}（{ {r['i'] for r in s if r['wrong']} }）AUC {AU(s):.3f}")
s=[r for r in S if r["pop"]<1e6]; print(f" 人口<1M: {len(s)} 正例 {sum(r['wrong'] for r in s)} 约束 {AU(s):.3f} 同伴 {AU(s,'peer'):.3f} 自报 {AU(s,'rp',-1):.3f}")
# 残差分布：负例按人口
print(" 负例残差中位 ≥1M vs <1M：",round(np.median([r['res'] for r in S if not r['wrong'] and r['pop']>=1e6]),4),round(np.median([r['res'] for r in S if not r['wrong'] and r['pop']<1e6]),4)," → 小国残差大（取整）")
