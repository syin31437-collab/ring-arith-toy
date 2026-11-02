"""E：对手预测器、错标 SE 敏感度、簇 bootstrap、干净窗检验、算术重构。"""
from e_eval import *
import json
MS=json.load(open(W+"/map_slqa.json")); MB=json.load(open(W+"/map_bipm.json"))
clu={"SLQA":{i:m["s"] for i,m in enumerate(MS)},"BIPM":{i:m["w"] for i,m in enumerate(MB)}}
lab={"SLQA":{i:m["g"] for i,m in enumerate(MS)},"BIPM":{i:m["g"] for i,m in enumerate(MB)}}
def rep(tag,P,A):
    P,A=np.array(P,float),np.array(A,float)
    sp=spearman(P,A); m=float(np.mean(np.abs(P-A)))
    print(f"  {tag:46s} Spearman {sp:5.2f}  MAE {m:.3f}  {'PASS' if (sp>=.8 and m<=.10) else 'fail'}")
    return sp,m
out={}
for name in ("SLQA","BIPM"):
    print(f"\n======= {name} =======")
    G=table(name); sp,m,P,A,K=score(G)
    bgrp=np.array([np.mean([x[3] for x in G[k]]) for k in K])
    Lv=[k[0] for k in K]; Sc=[k[1] for k in K]; Dl=[k[2] for k in K]; A=np.array(A); P=np.array(P)
    Ponly=np.array([np.mean([x[0] for x in G[k]]) for k in K])
    print(" 实际均值 %.3f；b 按级别 %s；实际 分布 分位25/50/75 %s"%(A.mean(),{l:round(float(bgrp[[i for i,x in enumerate(Lv) if x==l]].mean()),2) for l in set(Lv)},np.round(np.percentile(A,[25,50,75]),2)))
    rep("v2 论断：P + 基础率 b",P,A)
    rep("只用 P（不校正）",Ponly,A)
    rep("只用 b（完全不看注入）",bgrp,A)
    rep("常数=全局均值(用实际，作弊)",np.full(len(A),A.mean()),A)
    # 留一 δ：同 (L,sc) 其他 δ 的实际均值
    def loo(keyf):
        out=np.zeros(len(A))
        for i in range(len(A)):
            js=[j for j in range(len(A)) if keyf(j)==keyf(i) and j!=i]
            out[i]=A[js].mean() if js else A.mean()
        return out
    rep("留一：同 级别×场景 其他δ的实际均值",loo(lambda j:(Lv[j],Sc[j])),A)
    rep("留一：同 级别×δ 其他场景的实际均值",loo(lambda j:(Lv[j],Dl[j])),A)
    rep("留一：同 级别 的实际均值",loo(lambda j:Lv[j]),A)
    # 只看 δ 秩：同一 δ 全同的预测，给定 b
    dr=sorted(set(Dl))
    # 单一 se：领域内所有序列 se 的中位数，μ 只取 δ（无场景结构）
    rows,base=O[name]
    ses=[v[1] for r in rows for key,v in r[3].items() if v[0]!=0 and key[0]!="AB"]
    se0=np.median(ses)
    naive=[]
    for i,k in enumerate(K):
        naive.append(1-(1-pover(Dl[i]/se0))*(1-bgrp[i]))
    rep("零拟合：P(δ/单一全局se)+b（无场景/无逐序列）",naive,A)
    # 跨领域表：另一领域同 (L,sc,δ 秩) 的实际
    oth="BIPM" if name=="SLQA" else "SLQA"
    Go=table(oth); _,_,_,Ao,Ko=score(Go)
    dro=sorted({k[2] for k in Ko}); mp={(k[0],k[1],dro.index(k[2])):a for k,a in zip(Ko,Ao)}
    tr=[mp[(k[0],k[1],dr.index(k[2]))] for k in K]
    rep("跨领域抄表：另一领域同 L×场景×δ秩 的实际",tr,A)
    # 敏感度
    print(" 错标 se（预测 z 除以 c；c=1 为论断；b 不变）：")
    for c in (0.25,0.5,0.71,1,1.41,1.7,2,3,4):
        _,mm,PP,AA,_=score(table(name,c=c))
        print(f"   c={c:<5} Spearman {spearman(PP,AA):.3f} MAE {mm:.3f}")
    # P(|N(z,s²)|>3)，不用 b
    print(" 换零分布：z 的噪声 SD=s，不用 b：")
    for s in (1,1.2,1.4,1.7,2,2.5,3,4):
        _,mm,PP,AA,_=score(table(name,s=s,use_b=False))
        print(f"   s={s:<4} Spearman {spearman(PP,AA):.3f} MAE {mm:.3f}")
    # 按级别内 / δ 内 Spearman
    print(" 条件 Spearman：")
    for l in sorted(set(Lv)):
        idx=[i for i,x in enumerate(Lv) if x==l]; print(f"   级别 {l}（{len(idx)} 组）：{spearman(P[idx],A[idx]):.2f}  MAE {np.abs(P-A)[idx].mean():.3f}  只用b MAE {np.abs(bgrp-A)[idx].mean():.3f}")
    for d in dr:
        idx=[i for i,x in enumerate(Dl) if x==d]; print(f"   δ={d}（{len(idx)} 组）：{spearman(P[idx],A[idx]):.2f}  MAE {np.abs(P-A)[idx].mean():.3f}")
    for s_ in sorted(set(Sc)):
        idx=[i for i,x in enumerate(Sc) if x==s_]; print(f"   场景 {s_}（{len(idx)} 组）：{spearman(P[idx],A[idx]):.2f}  MAE {np.abs(P-A)[idx].mean():.3f}  偏差 {np.mean(P[idx]-A[idx]):+.3f}")
    # 超额检出：去掉 b 后
    ex=(A-bgrp)/(1-bgrp)
    print(f" 超额检出 (a−b)/(1−b) vs P：Spearman {spearman(Ponly,ex):.2f} MAE {np.mean(np.abs(Ponly-ex)):.3f}；负的超额组 {np.mean(ex<0):.0%}")
    # 只保留过渡区 + 只看窗口条件：S0 不报警的窗
    rows_,base_=O[name]
    clean={L:{wi for wi in base_ if not any(v[2]>3 for key,v in base_[wi].items() if key[0] in k)} for L,k in LV.items()}
    print(" S0 不报警的窗口数：",{L:len(v) for L,v in clean.items()},"/",len(base_))
    Pc,Ac,Kc=[],[],[]
    Pd,Ad=[],[]
    for k in K:
        vv=[x for x in G[k] if x[2] in clean[k[0]]]
        ww=[x for x in G[k] if x[2] not in clean[k[0]]]
        if vv:
            Pc.append(np.mean([x[0] for x in vv]));Ac.append(np.mean([x[1] for x in vv]));Kc.append(k)
        if ww: Pd.append(np.mean([x[0] for x in ww]));Ad.append(np.mean([x[1] for x in ww]))
    rep("S0 不报警的窗：纯 P，无 b",Pc,Ac)
    print(f"   平均 预测 {np.mean(Pc):.3f} 实际 {np.mean(Ac):.3f}；S0 报警窗 实际 {np.mean(Ad):.3f}（按 b 独立假设应接近 1 这部分） ")
    # 条件独立假设：P(a|S0报警) P(a|S0不报警)
    # 窗内判别：同一组内 预测高的窗是否更常报警（分层 AUC）
    aucs=[];wts=[]
    for k in K:
        v=[(x[0],x[1]) for x in G[k] if x[2] in clean[k[0]]]
        pos=[a for a,b in v if b];neg=[a for a,b in v if not b]
        if pos and neg:
            aucs.append(sum((p>q)+.5*(p==q) for p in pos for q in neg)/(len(pos)*len(neg)));wts.append(len(pos)*len(neg))
    print(f" 组内（同 级别×场景×δ、S0 不报警的窗）P 的分层 AUC：{np.average(aucs,weights=wts):.3f}（{len(aucs)} 组）")
    # 簇 bootstrap
    rg=np.random.default_rng(1); wins=sorted(base_)
    def boot(groups_of,B=2000):
        S,Mx,Pb,Bo=[],[],[],[]
        gs=sorted(set(groups_of.values()))
        for _ in range(B):
            pick=rg.choice(gs,len(gs))
            ws=[w for g in pick for w in wins if groups_of[w]==g]
            cnt={}
            bb={0:{},1:{}}
            for w in ws:
                for L,kk in LV.items():
                    bb[w%2].setdefault(L,[]).append(any(v[2]>3 for key,v in base_[w].items() if key[0] in kk))
            P_,A_,Bb=[],[],[]
            for k in K:
                d=[]
                by={x[2]:x for x in G[k]}
                pp=[];aa=[];b0=[]
                for w in ws:
                    x=by[w]; o=1-w%2
                    src=bb[o].get(k[0]) or bb[w%2].get(k[0])
                    b_=np.mean(src); pp.append(1-(1-x[0])*(1-b_)); aa.append(x[1]); b0.append(b_)
                P_.append(np.mean(pp));A_.append(np.mean(aa));Bb.append(np.mean(b0))
            S.append(spearman(P_,A_)); Mx.append(np.mean(np.abs(np.array(P_)-np.array(A_)))); Bo.append(np.mean(np.abs(np.array(Bb)-np.array(A_))))
        S,Mx,Bo=np.array(S),np.array(Mx),np.array(Bo)
        f=lambda a:(round(float(np.percentile(a,2.5)),3),round(float(np.percentile(a,97.5)),3))
        return f"Spearman 95% {f(S)} P(≥0.8)={np.mean(S>=.8):.2f}；MAE 95% {f(Mx)} P(≤0.10)={np.mean(Mx<=.10):.2f}；MAE(只b)−MAE(论断) 95% {f(Bo-Mx)}"
    print(" 簇 bootstrap：")
    print("   按窗口(视为独立)：",boot({w:w for w in wins},600))
    print("   按时段：",sorted(set(clu[name].values())),boot(clu[name],600))
    print("   按组/实验室集：",boot(lab[name],600))
    out[name]=dict(K=[list(map(str,k)) for k in K],P=P.tolist(),A=A.tolist(),b=bgrp.tolist())
json.dump(out,open(W+"/e_table.json","w"))
