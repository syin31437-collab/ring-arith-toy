"""基础率的成因：SLQA study 簇；BIPM 长记忆 + 记录事件 + 集中度 + 幅度。E 的算术重构。"""
from rtc import *
import json, os, pickle, collections
os.chdir(REPO+"/bipm")
import bipm as B, events as EV, validate_bipm as V, slqa as S
T=3.0
rng=np.random.default_rng(2)
# ---------- SLQA
E_,items=S.load()
MS=json.load(open(W+"/map_slqa.json"))
print("=== SLQA：study 簇 ===")
zs=[];zb=[];zperm=[];icc=[];zcl=[]
for m in MS:
    it=m["items"]; st_=[i.split("|")[0] for i in it]
    nodes,a,b=m["nodes"],m["a"],m["b"]
    Rw=np.array([[E_[i].get(l,np.nan) for l in nodes] for i in it]); A=np.array([E_[i][a] for i in it]); Bx=np.array([E_[i][b] for i in it])
    Sx=series_np(Rw,A,Bx)
    ust=sorted(set(st_),key=st_.index); idx={s:[i for i,x in enumerate(st_) if x==s] for s in ust}
    for k,x in Sx.items():
        c,e=x[:40],x[40:]
        z=fstat(c,e)[1]; zs.append(z)
        # 簇稳健：study 中位数 → 差的均值 + 簇标准误
        med_st=[np.nanmedian(x[idx[s]]) if np.any(~np.isnan(x[idx[s]])) else np.nan for s in ust]
        # study 打乱：把 study 随机分到校准/评估（保持题块完整）
        cal_st={st_[i] for i in range(40)}; ncal=len(cal_st)
        pz=[]
        for _ in range(40):
            pick=set(rng.choice(ust,ncal,replace=False).tolist())
            cc=[x[i] for s in ust if s in pick for i in idx[s]]; ee=[x[i] for s in ust if s not in pick for i in idx[s]]
            pz.append(fstat(cc,ee)[1])
        zperm+=pz
        # 组内相关 ICC（ANOVA）
        v=[x[idx[s]][~np.isnan(x[idx[s]])] for s in ust]; v=[u for u in v if len(u)>=2]
        if len(v)>=3:
            allv=np.concatenate(v); gm=np.median(allv)
            # 用 MAD 稳健：组间方差 / 总方差
            bm=np.array([np.median(u) for u in v]); w=mad(np.concatenate([u-np.median(u) for u in v])); bb=mad(bm)
            nb=np.mean([len(u) for u in v])
            icc.append(max(bb**2-(np.pi/2)*w**2/nb,0)/(max(bb**2-(np.pi/2)*w**2/nb,0)+w**2))
zs=np.array(zs); zperm=np.array(zperm)
lam=lambda a:np.median(a**2)/.4549
print(f"实际 |z|>3 {np.mean(abs(zs)>T):.3f} λ {lam(zs):.2f}；study 整块打乱 |z|>3 {np.mean(abs(zperm)>T):.3f} λ {lam(zperm):.2f}；稳健 ICC 中位 {np.median(icc):.2f}")
print(f" → SLQA 的 z 过散主要来自同一 study 的题不独立（n=70 题实际只有 10 个 study，评估期 4 个）")
# ---------- BIPM 全量滚动
print("\n=== BIPM ===")
labs,G=V.groups(); gps,glo=B.gnss("gps"),B.gnss("glo")
epochs=[t for t in sorted({t for d in labs.values() for t in d}) if t>=B.SANDBOX_END]
pg=B.diffs(gps,epochs)
D={k:B.diffs(labs[k],epochs) for g in G for k in g}
ev=EV.parse_notes()
lab_ev={k:[e for e in ev.get(k,[]) if e[0]>=B.SANDBOX_END] for k in D}
print("事件数按实验室(前12)",sorted(((len(v),k) for k,v in lab_ev.items()),reverse=True)[:12])
# 17 窗复算
MB=json.load(open(W+"/map_bipm.json")); WB=windows_cached("BIPM")
rec=[]
for m,(cal,ev_) in zip(MB,WB):
    Rc,Ac,Bc=unpack(cal);Re,Ae,Be=unpack(ev_)
    Sc,Se=series_np(Rc,Ac,Bc),series_np(Re,Ae,Be)
    for key in Sc:
        if key[0] not in("peer","nodeA"): continue
        lab=m["nodes"][key[1]]
        z=fstat(Sc[key],Se[key])[1]; sh=np.nanmedian(Se[key])-np.nanmedian(Sc[key])
        has_eval=any(m["cut"]<=e[0]<=m["hi"] for e in lab_ev.get(lab,[]))
        has_any=any(m["lo"]<=e[0]<=m["hi"] for e in lab_ev.get(lab,[]))
        rec.append(dict(kind=key[0],lab=lab,g=m["g"],w=m["w"],z=z,sh=sh,mad=mad(Sc[key]),ev=has_eval,any=has_any))
for kind in("peer","nodeA"):
    r=[x for x in rec if x["kind"]==kind]
    al=[x for x in r if abs(x["z"])>T]
    print(f"{kind}: {len(r)} 序列, |z|>3 {len(al)}={len(al)/len(r):.3f}；有事件(校准+评估) {np.mean([x['any'] for x in r]):.2f}；报警里有事件 {np.mean([x['any'] for x in al]):.2f}，无事件序列的报警率 {np.mean([abs(x['z'])>T for x in r if not x['any']]):.3f}，有事件 {np.mean([abs(x['z'])>T for x in r if x['any']]):.3f}；λ(无事件) {lam(np.array([x['z'] for x in r if not x['any']])):.2f}")
    print(f"   报警的 |偏移| ns 分位 25/50/75: {np.round(np.percentile([abs(x['sh']) for x in al],[25,50,75]),2)}；小于 1.5ns(最小h) {np.mean([abs(x['sh'])<1.5 for x in al]):.2f}；小于 0.75ns {np.mean([abs(x['sh'])<.75 for x in al]):.2f}")
    c=collections.Counter(x["lab"] for x in al); tot=collections.Counter(x["lab"] for x in r)
    print("   集中度：",[(l,c[l],tot[l]) for l,_ in c.most_common(8)]," 涉及实验室",len(c),"/",len(tot))
# 窗口内同时报警数 vs 二项
for kind in("peer","nodeA"):
    cnt=collections.Counter()
    for x in rec:
        if x["kind"]==kind: cnt[(x["g"],x["w"])]+=abs(x["z"])>T
    v=np.array(list(cnt.values())); p=v.mean()/9
    print(f"{kind} 每窗同时报警数 {sorted(v.tolist())}，均值 {v.mean():.2f}，方差 {v.var(ddof=1):.2f}，二项方差 {9*p*(1-p):.2f}；任一 {np.mean(v>0):.2f}（独立二项 {1-(1-p)**9:.2f}）")
# ACF + 方差-时间 + z vs n（全部 36 实验室的 peer、nodeA，全测试段）
def acf(x,L):
    x=x[~np.isnan(x)]; x=x-np.median(x); x=np.clip(x,-5*mad(x),5*mad(x)); return [float(np.corrcoef(x[:-l],x[l:])[0,1]) for l in L]
L=[1,2,3,4,6,8,13,26,52]
Aall=[];H=[]
ser={}
for k in D:
    ts=[t for t in epochs if t in D[k] and t in pg]
    if len(ts)<200: continue
    xa=np.array([D[k][t]-pg[t] for t in ts]); ser[k]=(ts,xa)
    Aall.append(acf(xa,L))
    # 方差-时间：块中位数的 MAD vs 块长
    bs=[1,2,4,8,16,32]; 
    y=[mad([np.median(xa[i:i+b]) for i in range(0,len(xa)-b+1,b)]) for b in bs]
    H.append(1+np.polyfit(np.log(bs),np.log(np.maximum(y,1e-6)),1)[0])
print("nodeA 的稳健截尾 ACF 中位：",dict(zip(L,np.round(np.median(Aall,0),3))),"；Hurst 估计中位",round(float(np.median(H)),2),"四分位",np.round(np.percentile(H,[25,75]),2),"（iid=0.5）")
print("纯 GPS 差分 ACF：",dict(zip(L,np.round(acf(np.array([pg[t] for t in epochs if t in pg]),L),3))))
print("\n滚动窗长度对 z 的影响（不注入；同样统计量；全实验室 nodeA；每窗不重叠）：理论 iid 下与 n 无关")
for nc,ne in((13,7),(26,13),(52,26),(104,26),(104,52),(208,52)):
    zz=[];shs=[]
    for k,(ts,x) in ser.items():
        for s in range(0,len(x)-nc-ne+1,nc+ne):
            z=fstat(x[s:s+nc],x[s+nc:s+nc+ne])[1]; zz.append(z)
    zz=np.array(zz); print(f"  校准{nc:4d} 评估{ne:3d}: {len(zz):5d} 段, |z|>3 {np.mean(abs(zz)>T):.3f}, λ {lam(zz):.2f}, 稳健 SD {mad(zz):.2f}")
print("同上，时间打乱对照：")
for nc,ne in((26,13),(104,26),(208,52)):
    zz=[]
    for k,(ts,x) in ser.items():
        for s in range(0,len(x)-nc-ne+1,nc+ne):
            y=rng.permutation(x[s:s+nc+ne]); zz.append(fstat(y[:nc],y[nc:])[1])
    zz=np.array(zz); print(f"  校准{nc:4d} 评估{ne:3d}: |z|>3 {np.mean(abs(zz)>T):.3f}, λ {lam(zz):.2f}")
# ---------- 算术重构
print("\n=== E 的“实际”能否由 S0 + 算术得到 ===")
for name,deltas,kmin,kmaj in (("SLQA",(0.005,0.01,0.02,0.03),3,5),("BIPM",(1.,2.,3.,4.),4,6)):
    Wn=windows_cached(name); ok=tot=0; ok2=tot2=0
    for wi,(cal,ev_) in enumerate(Wn):
        Rc,Ac,Bc=unpack(cal);Re,Ae,Be=unpack(ev_)
        Sc,Se=series_np(Rc,Ac,Bc),series_np(Re,Ae,Be)
        for d in deltas:
            # scale：nodeA、consA 全部 + d（减号），AB + d；与 Se 直接算的结果比
            for key in Sc:
                c,e=Sc[key],Se[key]
                if key[0]=="peer" or key[0]=="consB":continue
                mu=-d if key[0] in("nodeA","consA") else d
                z0=fstat(c,e)[1]
                se=abs(d)
                zr=fstat(c,e+mu)[0]   # 真实加注入
                rec_=abs(z0+mu/ (fstat(c,e)[0]/max(abs(z0),1e-9) if False else 1)) if False else None
                m=~np.isnan(e)
                sv=1.2533*math.sqrt(mad(c[~np.isnan(c)])**2/len(c[~np.isnan(c)])+mad(e[m])**2/m.sum())
                zr2=abs(np.nanmedian(e)+mu-np.nanmedian(c))/sv
                ok+=(zr>3)==(zr2>3); tot+=1
    print(name,f"scale 场景逐序列：只用 S0 的中位差 + μ 重构检出，吻合 {ok}/{tot}")
