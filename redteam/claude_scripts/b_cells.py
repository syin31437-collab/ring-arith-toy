"""B：validate_v61 / validate_bipm 分层；信息格；基线；簇 bootstrap；平凡对手。"""
from rtc import *
import json, collections
R6=json.load(open(REPO+"/validate_v61.json"))["cells"]
JB=json.load(open(REPO+"/bipm/validate_bipm_test.json")); RB=JB["cells"]
MB=json.load(open(W+"/map_bipm.json"))
for i,r in enumerate(RB):
    m=MB[i//22]; r["g"]=m["g"]; r["w"]=m["w"]; r["c"]=i//22
# 注入格的 k：按 main 中顺序 h 1.5,3 自然；然后 product(k,dl)
ORD=[("自然",0,1.5),("自然",0,3.0)]+[("注入",k,dl) for k in(2,4,6) for dl in(3.0,6.0,12.0)]
for i,r in enumerate(RB):
    tag,k,dl=ORD[(i%22)//2]; assert r["tag"]==tag and (r["h"]==(dl if tag=="自然" else dl/2)); r["k"]=k; r["dl"]=dl if tag=="注入" else 0
for r in R6: r["cfg"]=(r["n"],r["f"],r["attack"],r["m"])
def mae(s,k="pt"): return float(np.mean([abs(r[k]-r["act"]) for r in s]))
def sp(s,k="pt"): return spearman([r[k] for r in s],[r["act"] for r in s])
def bias(s,k="pt"): return float(np.mean([r[k]-r["act"] for r in s]))
def line(tag,s):
    a=np.array([r["act"] for r in s])
    print(f"  {tag:28s} {len(s):4d}格 实测{a.mean():.3f} | v6.1 MAE {mae(s):.3f} 偏差{bias(s):+.3f} ρ{sp(s):+.2f} | B2 {mae(s,'b2'):.3f} ρ{sp(s,'b2'):+.2f} | d5 {mae(s,'d5'):.3f} | B1 {mae(s,'b1'):.3f} | 常数(用实测均值) {float(np.mean(np.abs(a-a.mean()))):.3f}")
for name,R in(("FMNIST",R6),("BIPM",RB)):
    print(f"\n=== {name} 全部 ===")
    print(f" argsort 排名的 Spearman 为何：平均秩 Spearman {sp(R):.3f}；act 恰为 0 的格 {np.mean([r['act']==0 for r in R]):.0%}，act<0.05 {np.mean([r['act']<.05 for r in R]):.0%}，act<0.1 {np.mean([r['act']<.1 for r in R]):.0%}，0.1≤act≤0.9 {np.mean([.1<=r['act']<=.9 for r in R]):.0%}")
    line("全部",R)
    for j in("软","硬2"): line("裁判 "+j,[r for r in R if r["judge"]==j])
    line("易格 act<0.05",[r for r in R if r["act"]<.05])
    line("易格 pt<0.05",[r for r in R if r["pt"]<.05])
    line("信息格 0.1≤act≤0.9",[r for r in R if .1<=r["act"]<=.9])
    line("信息格 0.2≤act≤0.8",[r for r in R if .2<=r["act"]<=.8])
    line("非易 pt≥0.1 或 act≥0.1",[r for r in R if r["act"]>=.1 or r["pt"]>=.1])
    line("B2 也认为非平凡 b2≥.1",[r for r in R if r["b2"]>=.1])
print("\n=== FMNIST 按攻击/m ===")
for a in["ALIE0.5","ALIE1","ALIE2","ALIE4","noise","flip"]:
    line(a,[r for r in R6 if r["attack"]==a])
    for m in(32,512): line(f"  {a} m={m}",[r for r in R6 if r["attack"]==a and r["m"]==m])
print("\n=== BIPM 按 tag / δ / k / 裁判 ===")
for j in("软","硬2"):
    line(j+" 自然 h=1.5",[r for r in RB if r["judge"]==j and r["tag"]=="自然" and r["h"]==1.5])
    line(j+" 自然 h=3",[r for r in RB if r["judge"]==j and r["tag"]=="自然" and r["h"]==3])
    for dl in(3.,6.,12.):
        line(f"{j} 注入 δ={dl:g}",[r for r in RB if r["judge"]==j and r["tag"]=="注入" and r["dl"]==dl])
for k in(2,4,6): line(f"注入 k={k}",[r for r in RB if r["tag"]=="注入" and r["k"]==k])
for dl in(3.,6.,12.):
  for k in(2,4,6): line(f"注入 δ={dl:g} k={k}",[r for r in RB if r["tag"]=="注入" and r["k"]==k and r["dl"]==dl])
# 平凡对手：留一表
def loo_table(R,keyf,clusterf):
    for r in R:
        s=[x["act"] for x in R if keyf(x)==keyf(r) and clusterf(x)!=clusterf(r)]
        r["_t"]=float(np.mean(s)) if s else float(np.mean([x["act"] for x in R]))
    return "_t"
print("\n=== 平凡对手（留一）===")
def compare(R,tag):
    v=[abs(r["pt"]-r["act"]) for r in R]; t=[abs(r["_t"]-r["act"]) for r in R]
    print(f"  {tag}: 对手 MAE {np.mean(t):.3f} ρ {spearman([r['_t'] for r in R],[r['act'] for r in R]):.2f} ｜ v6.1 MAE {np.mean(v):.3f} ρ {sp(R):.2f}；信息格：对手 {np.mean([abs(r['_t']-r['act']) for r in R if .1<=r['act']<=.9]):.3f} v6.1 {np.mean([abs(r['pt']-r['act']) for r in R if .1<=r['act']<=.9]):.3f}")
loo_table(R6,lambda r:(r["attack"],r["m"],r["judge"]),lambda r:(r["n"],r["f"]));compare(R6,"FMNIST 同攻击×m×裁判(其他 n,f)")
loo_table(R6,lambda r:(r["attack"],r["m"]),lambda r:(r["n"],r["f"]));compare(R6,"FMNIST 同攻击×m(其他 n,f)")
loo_table(R6,lambda r:(r["attack"]),lambda r:(r["n"],r["f"],r["m"]));compare(R6,"FMNIST 同攻击(其他 n,f,m)")
loo_table(RB,lambda r:(r["tag"],r["h"],r["judge"]),lambda r:r["c"]);compare(RB,"BIPM 同tag×h×裁判(其他窗)")
loo_table(RB,lambda r:(r["tag"],r["h"],r["k"],r["judge"]),lambda r:r["c"]);compare(RB,"BIPM 同tag×h×k×裁判(其他窗)")
loo_table(RB,lambda r:(r["tag"],r["h"]),lambda r:r["c"]);compare(RB,"BIPM 同tag×h(其他窗)")
loo_table(RB,lambda r:(r["h"]),lambda r:r["c"]);compare(RB,"BIPM 只按 h(其他窗)")
# 把 v6.1 的残差和窗口差异：窗内相关
print("\n=== BIPM 组内技能：同 tag×h×k×裁判 内，v6.1 能否解释窗口差异 ===")
gr=collections.defaultdict(list)
for r in RB: gr[(r["tag"],r["h"],r["k"],r["judge"])].append(r)
pa=[];aa=[];ba=[]
for k,v in gr.items():
    ma=np.mean([r["act"] for r in v]); mp=np.mean([r["pt"] for r in v]); mb=np.mean([r["b2"] for r in v])
    for r in v: pa.append(r["pt"]-mp); aa.append(r["act"]-ma); ba.append(r["b2"]-mb)
print(f"  去组均值后：v6.1 Pearson {np.corrcoef(pa,aa)[0,1]:.3f}，B2 Pearson {np.corrcoef(ba,aa)[0,1]:.3f}，实测组内 SD {np.std(aa):.3f}，v6.1 组内 SD {np.std(pa):.3f}，残差 v6.1 {np.std(np.array(aa)-np.array(pa)):.3f}")
gr=collections.defaultdict(list)
for r in R6: gr[(r["attack"],r["m"],r["judge"])].append(r)
pa=[];aa=[];ba=[]
for k,v in gr.items():
    ma=np.mean([r["act"] for r in v]); mp=np.mean([r["pt"] for r in v]); mb=np.mean([r["b2"] for r in v])
    for r in v: pa.append(r["pt"]-mp); aa.append(r["act"]-ma); ba.append(r["b2"]-mb)
print(f"  FMNIST 去 攻击×m×裁判 均值后：v6.1 Pearson {np.corrcoef(pa,aa)[0,1]:.3f}，B2 {np.corrcoef(ba,aa)[0,1]:.3f}，实测组内 SD {np.std(aa):.3f}，残差 v6.1 {np.std(np.array(aa)-np.array(pa)):.3f}")
# 簇 bootstrap
rg=np.random.default_rng(3)
def boot(R,keyf,B=4000,sel=lambda r:True):
    cl=collections.defaultdict(list)
    for r in R:
        if sel(r): cl[keyf(r)].append(r)
    ks=list(cl); M,Sp,D=[],[],[]
    for _ in range(B):
        s=[r for k in rg.choice(len(ks),len(ks)) for r in cl[ks[k]]]
        v=np.array([abs(r["pt"]-r["act"]) for r in s]); b=np.array([abs(r["b2"]-r["act"]) for r in s])
        M.append(v.mean()); D.append(b.mean()-v.mean()); Sp.append(spearman([r["pt"] for r in s],[r["act"] for r in s]))
    f=lambda a:tuple(np.round(np.percentile(a,[2.5,97.5]),3))
    return f"{len(ks)}簇 MAE{f(M)} P(≤.10)={np.mean(np.array(M)<=.1):.2f}; ρ{f(Sp)} P(≥.8)={np.mean(np.array(Sp)>=.8):.2f}; MAE(B2)−MAE(v6.1){f(D)} P(>0)={np.mean(np.array(D)>0):.3f}"
print("\n=== 簇 bootstrap ===")
print(" BIPM 全部 按窗口-组(论断用):",boot(RB,lambda r:r["c"]))
print(" BIPM 全部 按时段 w:",boot(RB,lambda r:r["w"]))
print(" BIPM 全部 按实验室组 g:",boot(RB,lambda r:r["g"]))
info=lambda r:.1<=r["act"]<=.9
print(" BIPM 信息格 按时段 w:",boot(RB,lambda r:r["w"],sel=info))
print(" BIPM 信息格 按窗口-组:",boot(RB,lambda r:r["c"],sel=info))
print(" BIPM 软 按时段:",boot(RB,lambda r:r["w"],sel=lambda r:r["judge"]=="软"))
print(" BIPM 硬2 按时段:",boot(RB,lambda r:r["w"],sel=lambda r:r["judge"]=="硬2"))
print(" BIPM 自然 按时段:",boot(RB,lambda r:r["w"],sel=lambda r:r["tag"]=="自然"))
print(" BIPM 注入 δ=3 按时段:",boot(RB,lambda r:r["w"],sel=lambda r:r["dl"]==3))
print(" FMNIST 按配置:",boot(R6,lambda r:r["cfg"]))
print(" FMNIST 信息格 按配置:",boot(R6,lambda r:r["cfg"],sel=info))
print(" FMNIST 按攻击(只 6 簇):",boot(R6,lambda r:r["attack"]))
print(" FMNIST 按 (n,f):",boot(R6,lambda r:(r["n"],r["f"])))
print(" FMNIST 信息格 按攻击:",boot(R6,lambda r:r["attack"],sel=info))
