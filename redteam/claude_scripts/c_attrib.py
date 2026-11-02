"""归因现实性：把恒定注入换成噪声/部分/间歇/和噪声同量级；再加第三假设。规则同 ladder2：cb>ab → 判节点。"""
from rtc import *
import collections
rng=np.random.default_rng(5)
def score(cal,ev,node_sh,a_sh,b_sh=None):
    """node_sh: T×n, a_sh: T, b_sh: T。返回 cb, ab, ca"""
    Rc,Ac,Bc=unpack(cal);Re,Ae,Be=unpack(ev)
    Re2=Re+node_sh; Ae2=Ae+a_sh; Be2=Be+(0 if b_sh is None else b_sh)
    med=lambda M:np.array([np.median(r[~np.isnan(r)]) if (~np.isnan(r)).sum()>=2 else np.nan for r in M])
    Cc,Ce=med(Rc),med(Re2)
    return dict(cb=fstat(Cc-Bc,Ce-Be2)[0],ab=fstat(Ac-Bc,Ae2-Be2)[0],ca=fstat(Cc-Ac,Ce-Ae2)[0])
def dec(y): return "node" if y["cb"]>y["ab"] else "scale"
def noise_ref(W_):
    # A−B 的 MAD、共识−B 的 MAD、以及评估期 se
    v=[];s=[];c=[]
    for cal,ev in W_:
        Rc,Ac,Bc=unpack(cal);v.append(mad(Ac-Bc));
        med=np.array([np.median(r[~np.isnan(r)]) for r in Rc]); c.append(mad(med-Bc)); s.append(1.2533*mad(Ac-Bc)*math.sqrt(1/len(cal)+1/len(ev)))
    return np.median(v),np.median(c),np.median(s)
def run(name,deltas):
    W_=windows_cached(name)
    mAB,mCB,seAB=noise_ref(W_)
    print(f"\n=== {name}: {len(W_)} 窗；A−B MAD中位 {mAB:.4g}，共识−B MAD中位 {mCB:.4g}，A−B 的 z 分母中位 {seAB:.4g}；原实验 δ={deltas}（δ/A−B MAD = {[round(d/mAB,2) for d in deltas]}）")
    def inject(kind,d,T,n):
        """返回 (node_sh, a_sh)，kind=node 故障 or scale 故障的形式"""
        if kind=="恒定":      return np.full(T,d)
        if kind=="噪声大CV=1": return d*(1+rng.normal(0,1,T))
        if kind=="噪声CV=3":  return d*(1+rng.normal(0,3,T))
        if kind=="间歇50%":   return d*(rng.random(T)<.5)
        if kind=="间歇20%":   return d*(rng.random(T)<.2)
        if kind=="零均值噪声": return d*rng.normal(0,1,T)
        if kind=="单次脉冲":  a=np.zeros(T);a[rng.integers(T)]=d*T;return a
        raise
    kinds=["恒定","噪声大CV=1","噪声CV=3","间歇50%","间歇20%","零均值噪声","单次脉冲"]
    print("形式 × δ/A−B的MAD：节点全体 vs 秤A，两类平均准确率（节点召回/秤召回）")
    for rep_ in [1]:
      for mult in (0.1,0.25,0.5,1,2,4):
        d=mult*mAB; row=[]
        for k in kinds:
            na=[];sa=[]
            for cal,ev in W_:
                T=len(ev);n=len(ev[0][2])
                for _ in range(10):
                    sh=inject(k,d,T,n); y=score(cal,ev,np.tile(sh[:,None],(1,n)),np.zeros(T)); na.append(dec(y)=="node")
                    sh=inject(k,d,T,n); y=score(cal,ev,np.zeros((T,n)),sh); sa.append(dec(y)=="scale")
            row.append(f"{k}:{(np.mean(na)+np.mean(sa))/2:.2f}({np.mean(na):.2f}/{np.mean(sa):.2f})")
        print(f"  ×{mult:<5}"," ".join(row))
    # 部分节点
    print("部分节点同偏（k/n 个节点，其余干净） vs 秤A，用论文的 δ 中档：")
    d=deltas[1]
    for frac in (1,.8,.6,.5,.4,.2):
        na=[]
        for cal,ev in W_:
            T=len(ev);n=len(ev[0][2]);k=max(1,round(frac*n))
            for _ in range(10):
                idx=rng.choice(n,k,replace=False);S=np.zeros((T,n));S[:,idx]=d
                y=score(cal,ev,S,np.zeros(T));na.append(dec(y)=="node")
        print(f"  节点比例 {frac}: 判节点 {np.mean(na):.2f}")
    # 第三假设：B 坏 / A、节点同偏 / 秤 A+B 同偏 / 不注入
    print("二选一规则对其他世界的回答（δ 中、大）：")
    for d in deltas[1:]:
        cnt={}
        for wn,fn in (("不注入 S0",lambda T,n:(np.zeros((T,n)),np.zeros(T),np.zeros(T))),("B 坏",lambda T,n:(np.zeros((T,n)),np.zeros(T),np.full(T,d))),("节点全体+A 同偏(B 干净)",lambda T,n:(np.full((T,n),d),np.full(T,d),np.zeros(T))),("A+B 同偏",lambda T,n:(np.zeros((T,n)),np.full(T,d),np.full(T,d))),("节点全体+B 同偏",lambda T,n:(np.full((T,n),d),np.zeros(T),np.full(T,d)))):
            o=[]; zz=[]
            for cal,ev in W_:
                T=len(ev);n=len(ev[0][2]);S,a,b=fn(T,n);y=score(cal,ev,S,a,b);o.append(dec(y)); zz.append((y["cb"],y["ab"],y["ca"]))
            zz=np.array(zz)
            cnt[wn]=f"判节点 {np.mean([x=='node' for x in o]):.2f} 判秤A {np.mean([x=='scale' for x in o]):.2f} (z̄ cb {np.median(zz[:,0]):.1f} ab {np.median(zz[:,1]):.1f} ca {np.median(zz[:,2]):.1f})"
        print(f"  δ={d}: ",cnt)
    # 分数差余量
    print("裕度：恒定注入下 z 的差 |cb−ab| 中位")
    for d in deltas:
        v=[]
        for cal,ev in W_:
            T=len(ev);n=len(ev[0][2]);y=score(cal,ev,np.full((T,n),d),np.zeros(T));v.append(y["cb"]-y["ab"])
        print(f"  δ={d}: 中位 {np.median(v):.1f}，最小 {np.min(v):.1f}")
    # 自然报警窗：一个真实事件落在 A−B 或 cons−B 时，归因说什么？S0 下超线的情况
    # 三方两两比较（真正的定位）：A,B,C 三个中哪个被两条差分同时指向
    print("三路定位（A,B,C 三个差分：超 3 的边的公共点；两条超线才判，否则“不判”）：")
    for d in deltas:
      res=collections.defaultdict(list)
      for wn,fn,truth in (("节点全体",lambda T,n:(np.full((T,n),d),np.zeros(T),np.zeros(T)),"C"),("秤A",lambda T,n:(np.zeros((T,n)),np.full(T,d),np.zeros(T)),"A"),("秤B",lambda T,n:(np.zeros((T,n)),np.zeros(T),np.full(T,d)),"B")):
        for cal,ev in W_:
            T=len(ev);n=len(ev[0][2]);S,a,b=fn(T,n);y=score(cal,ev,S,a,b)
            cb,ab,ca=y["cb"]>3,y["ab"]>3,y["ca"]>3
            if ca and cb and not ab: r="C"
            elif ab and ca and not cb: r="A"
            elif ab and cb and not ca: r="B"
            else: r="?"
            res[wn].append(r==truth); res[wn+"_拒"].append(r=="?")
      print(f"  δ={d}: "+", ".join(f"{k}:{np.mean(v):.2f}" for k,v in res.items()))
run("SLQA",(0.05,0.10,0.20))
run("BIPM",(6.,12.,24.))
