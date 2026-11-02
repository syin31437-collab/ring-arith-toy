"""F：实验室内分析。"""
from rtc import *
import os, collections, json
os.chdir(REPO+"/bipm")
import bipm as B, events as E
ev=E.parse_notes(); W_,keep=E.windows()
lab_ev={k:[e for e in ev.get(k,[]) if e[0]>=B.SANDBOX_END] for k in keep}
print(len(keep),"实验室",len(W_),"窗")
print("事件分布前10",sorted(((len(v),k) for k,v in lab_ev.items()),reverse=True)[:10])
print("za 例子：");
for l in [l for l in open("data/notes.utc",errors="ignore")][:0]: pass
txt=open("data/notes.utc",errors="ignore").read().splitlines()
# 找 ZA 段前几行
i0=[i for i,l in enumerate(txt) if l.strip().startswith("ZA")][0]
print("\n".join(txt[i0:i0+6]))
i1=[i for i,l in enumerate(txt) if l.strip().startswith("DMDM")][0]
print("\n".join(txt[i1:i1+4]))
LV=("L2","L0")
names={0:"EWMA",1:"单点"}
def has(w,E_=lab_ev,sft=0): return any(w["lo"]+sft<=m<=w["hi"]+sft for m,_,_ in E_.get(w["lab"],[]))
rows=lambda L,i:[(w["lab"],bool(w[L][i]),has(w),w["lo"]) for w in W_]
def lift(a,b): return (np.mean(a)/np.mean(b)) if len(a) and len(b) and np.mean(b)>0 else float('inf')
def MH(rs):
    """Mantel-Haenszel OR 按实验室分层"""
    g=collections.defaultdict(list)
    for lab,al,evn,_ in rs: g[lab].append((al,evn))
    num=den=0
    for lab,v in g.items():
        n=len(v); a=sum(x and y for x,y in v); b=sum(x and not y for x,y in v); c=sum((not x) and y for x,y in v); d=sum((not x) and (not y) for x,y in v)
        num+=a*d/n; den+=b*c/n
    return num/den if den else float('inf')
rng=np.random.default_rng(0)
# 窗口的时间索引：每个实验室每窗按 lo
by=collections.defaultdict(list)
for w in W_: by[w["lab"]].append(w)
for L in LV:
  for i in (0,1):
    rs=rows(L,i)
    a=[e for _,al,e,_ in rs if al]; b=[e for _,al,e,_ in rs if not al]
    pa,pb=np.mean(a),np.mean(b)
    or_all=(sum(a)*(len(b)-sum(b)))/max(((len(a)-sum(a))*sum(b)),1)
    mh=MH(rs)
    # 实验室内置换：每实验室内，窗口的事件标签循环移位（保留该实验室的事件数和时序自相关）
    obs=mh; perm=[]; lfs=[]
    for _ in range(2000):
        rr=[]
        for lab,ws in by.items():
            s=rng.integers(len(ws)); ee=[has(x) for x in ws]; ee=ee[s:]+ee[:s]
            rr+= [(lab,bool(x[L][i]),e,x["lo"]) for x,e in zip(ws,ee)]
        perm.append(MH(rr))
        ax=[e for _,al,e,_ in rr if al];bx=[e for _,al,e,_ in rr if not al]
        lfs.append(lift(ax,bx))
    perm=np.array(perm);lfs=np.array(lfs)
    lf=lift(a,b)
    # 实验室固定效应下的 组内差
    dif=[]
    wts=[]
    for lab,ws in by.items():
        al=[has(x) for x in ws if x[L][i]]; nl=[has(x) for x in ws if not x[L][i]]
        if al and nl:
            dif.append(np.mean(al)-np.mean(nl)); wts.append(len(al)*len(nl)/(len(al)+len(nl)))
    print(f"\n{L} {names[i]}：报警{len(a)} 窗事件率 {pa:.0%}；不报警{len(b)} {pb:.0%}；提升度 {lf:.2f}（实验室内循环移位零分布中位 {np.median(lfs):.2f}，95% {np.percentile(lfs,95):.2f}，p {np.mean(lfs>=lf):.4f}）")
    print(f"   粗 OR {or_all:.2f}；按实验室分层 MH-OR {mh:.2f}，循环移位零分布中位 {np.median(perm[np.isfinite(perm)]):.2f}，p {np.mean(perm>=mh):.4f}；实验室内率差加权 {np.average(dif,weights=wts):+.3f}（{len(dif)} 个实验室两类都有）")
    # 去掉 za, dmdm（事件爆炸的两个）
    s=[r for r in rs if r[0] not in("za","dmdm")]
    a2=[e for _,al,e,_ in s if al];b2=[e for _,al,e,_ in s if not al]
    print(f"   去 za、dmdm：提升度 {lift(a2,b2):.2f} MH {MH(s):.2f}（报警 {len(a2)} 窗）")
    # 实验室水平混杂：实验室报警率 vs 事件率 Spearman
    labs=list(by); ar=[np.mean([x[L][i] for x in by[k]]) for k in labs]; er=[np.mean([has(x) for x in by[k]]) for k in labs]; nz=[np.median([1]) for k in labs]
    print(f"   实验室水平：报警率 vs 事件率 Spearman {spearman(ar,er):.2f}（{len(labs)} 个实验室）")
    # 安慰剂：事件时间平移 ±k 窗（26 历元*5 天 =130 天 窗长 78 历元=390 天）
    pl={}
    for sft in (-780,-390,-130,0,130,390,780):
        aa=[has(w,sft=sft) for w in W_ if w[L][i]];bb=[has(w,sft=sft) for w in W_ if not w[L][i]]
        pl[sft]=round(lift(aa,bb),2)
    print("   安慰剂(事件平移天数→提升度):",pl)
    # 簇 bootstrap（按实验室）
    bs=[]
    for _ in range(2000):
        pick=rng.choice(labs,len(labs))
        aa=[];bb=[]
        for k in pick:
            for x in by[k]:
                (aa if x[L][i] else bb).append(has(x))
        bs.append(lift(aa,bb))
    print("   按实验室 bootstrap 提升度 95%",np.round(np.percentile(bs,[2.5,97.5]),2))
# 噪声 vs 事件
print("\n实验室 MAD(nodeA 校准) vs 事件率 vs 报警率：")
labs_=B.all_labs(); gps=B.gnss("gps")
eps=[t for t in sorted({t for d in labs_.values() for t in d}) if t>=B.SANDBOX_END]; pg=B.diffs(gps,eps)
m={}
for k in keep:
    d=B.diffs(labs_[k],eps); x=[d[t]-pg[t] for t in eps if t in d and t in pg]; m[k]=mad(x)
ks=list(by)
print(" Spearman(MAD,事件率)",round(spearman([m[k] for k in ks],[np.mean([has(x) for x in by[k]]) for k in ks]),2)," Spearman(MAD,EWMA报警率)",round(spearman([m[k] for k in ks],[np.mean([x['L2'][0] for x in by[k]]) for k in ks]),2)," Spearman(MAD,单点报警率)",round(spearman([m[k] for k in ks],[np.mean([x['L2'][1] for x in by[k]]) for k in ks]),2))
# 时间：逐年事件率与报警率
yr=lambda t:2015+int((t-57023)/365.25)
Y=collections.defaultdict(list)
for w in W_: Y[yr(w["lo"])].append(w)
print(" 年份 窗数 事件率 单点率 EWMA率:",[(y,len(v),round(np.mean([has(x) for x in v]),2),round(np.mean([x['L2'][1] for x in v]),2),round(np.mean([x['L2'][0] for x in v]),2)) for y,v in sorted(Y.items())])
# 单点报警落点 vs 事件日期：时间距离
print(" apparent 事件占比", )
