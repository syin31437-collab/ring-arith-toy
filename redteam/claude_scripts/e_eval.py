from rtc import *
import pickle, math
from statistics import NormalDist
N=NormalDist(); LV={"L0":("peer",),"L2":("nodeA","consA"),"L3":("nodeA","consA","consB","AB")}
O=pickle.load(open(W+"/e_cells.pkl","rb"))
def pover(z,s=1.0,t=3.0): return 1-(N.cdf((t-z)/s)-N.cdf((-t-z)/s))
def table(name,c=1.0,s=1.0,win=None,use_b=True,bfn=None):
    """c: 给 se 乘（改预测 z）；s: P 用 N(z,s²)。返回 {(L,sc,d):[(pred,act,wi)]}, b"""
    rows,base=O[name]
    wins=sorted(base) if win is None else win
    b={0:{},1:{}}
    for wi in sorted(base):
        for L,k in LV.items():
            al=any(v[2]>3 for key,v in base[wi].items() if key[0] in k)
            b[wi%2].setdefault(L,[]).append(al)
    G={}
    for wi,sc,d,res in rows:
        for L,k in LV.items():
            if L=="L0" and sc in("all","drift","scale"): continue
            q=1.0; act=False
            for key,(mu,se,zz) in res.items():
                if key[0] in k:
                    q*=1-pover(mu/(se*c),s); act=act or zz>3
            bb=np.mean(b[1-wi%2][L]) if use_b else 0.0
            G.setdefault((L,sc,d),[]).append((1-q,act,wi,bb))
    return G
def score(G,wmask=None):
    P,A=[],[]
    for k,v in G.items():
        vv=[x for x in v if wmask is None or x[2] in wmask]
        P.append(np.mean([1-(1-x[0])*(1-x[3]) for x in vv])); A.append(np.mean([x[1] for x in vv]))
    return spearman(P,A),float(np.mean(np.abs(np.array(P)-np.array(A)))),P,A,list(G)
if __name__=="__main__":
    for name in O:
        G=table(name); sp,m,P,A,K=score(G)
        print(name,"复现 Spearman %.3f MAE %.3f 组数 %d 每组窗 %d"%(sp,m,len(K),len(next(iter(G.values())))))
