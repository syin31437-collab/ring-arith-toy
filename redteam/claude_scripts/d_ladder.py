"""ladder2 BIPM 未过的两条是不是假失败：负样本污染、分母被漂移撑大。"""
from rtc import *
import ladder2 as LL
W_=windows_cached("BIPM")
def auc(p,n): return sum((a>b)+.5*(a==b) for a in p for b in n)/(len(p)*len(n))
rng=random.Random("ladder2-BIPM")
X=[]
for cal,ev in W_:
    n,T=len(cal[0][2]),len(ev)
    X.append(("S0",0.,LL.scores(cal,ev,lambda t,j:0.,0.)))
    for dl in(6.,12.,24.):
        for sc,k in(("one",1),("min",4),("maj",6),("all",n)):
            idx=set(rng.sample(range(n),k)); X.append((sc,dl,LL.scores(cal,ev,lambda t,j,idx=idx,dl=dl:dl if j in idx else 0.,0.)))
        X.append(("drift",dl,LL.scores(cal,ev,lambda t,j,dl=dl,T=T:2*dl*(t+1)/T,0.)))
        X.append(("scale",dl,LL.scores(cal,ev,lambda t,j:0.,dl)))
neg={L:[x[2][L] for x in X if x[0]=="S0"] for L in("L0","L2","L3")}
print("S0 分数：",{L:np.round(np.percentile(v,[25,50,75,90]),1).tolist() for L,v in neg.items()})
for sc,dl,L in(("one",12.,"L2"),("one",24.,"L2"),("one",12.,"L0"),("drift",12.,"L2"),("drift",24.,"L2")):
    p=[x[2][L] for x in X if x[0]==sc and x[1]==dl]; n=neg[L]
    clean=[v for v in n if v<=3]
    # 配对：同窗口 注入后分数 > S0 分数
    s0=n; pw=np.mean([a>b for a,b in zip(p,s0)])
    print(f"{sc} δ={dl} {L}: AUC {auc(p,n):.3f}；只用未报警的 S0 ({len(clean)}) AUC {auc(p,clean):.3f}；同窗配对 分数上升 {pw:.2f}；报警(>3) {np.mean(np.array(p)>3):.2f}")
# drift 用校准期 MAD 当分母
def stat_cal(c,e):
    c=np.array(c,float);e=np.array(e,float);c=c[~np.isnan(c)];e=e[~np.isnan(e)]
    if len(c)<5 or len(e)<5:return 0.
    m=mad(c); return abs(np.median(e)-np.median(c))/(1.2533*m*math.sqrt(1/len(c)+1/len(e)))
ps={};ng=[]
for cal,ev in W_:
    Rc,Ac,Bc=unpack(cal);Re,Ae,Be=unpack(ev);T=len(ev)
    med=lambda M:np.array([np.median(r[~np.isnan(r)]) for r in M])
    ng.append(max(stat_cal(Rc[:,j]-Ac,Re[:,j]-Ae) for j in range(Rc.shape[1])))
    for dl in(6.,12.,24.):
        Rd=Re+(2*dl*(np.arange(1,T+1))/T)[:,None]
        s=max(max(stat_cal(Rc[:,j]-Ac,Rd[:,j]-Ae) for j in range(Rc.shape[1])),stat_cal(med(Rc)-Ac,med(Rd)-Ae))
        ps.setdefault(dl,[]).append(s)
for dl,v in ps.items(): print(f"drift δ={dl}: 分母换成校准期 MAD 后 L2 AUC {auc(v,ng):.3f}（S0 同样换）")
