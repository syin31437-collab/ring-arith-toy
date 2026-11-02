import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/Users/sly/Desktop/agent/claude/ring-arith-toy/theory')
import distinguish as D, random, statistics as st, torch
from ladder2 import slqa_windows,bipm_windows
def v1(name,windows,deltas,kmin,kmaj):
    rg=random.Random("dist-"+name); G={}
    for cal,ev in windows:
        n=len(cal[0][2])
        def add(sc,d,res):
            for L,(p,a) in res.items(): G.setdefault((L,sc,d),[]).append((p,a))
        add("S0",0.0,D.cell(cal,ev,[0.0]*n,0.0))
        for d in deltas:
            for sc,k in (("one",1),("min",kmin),("maj",kmaj),("all",n)):
                idx=set(rg.sample(range(n),k)); add(sc,d,D.cell(cal,ev,[d if j in idx else 0.0 for j in range(n)],0.0))
            add("drift",d,D.cell(cal,ev,[d]*n,0.0,drift=True)); add("scale",d,D.cell(cal,ev,[0.0]*n,d))
    P=[st.mean(x[0] for x in v) for v in G.values()];A=[st.mean(x[1] for x in v) for v in G.values()]
    rk=lambda v:torch.tensor(v,dtype=torch.float).argsort().argsort().float()
    print(name,"v1 组",len(P),"argsort Spearman %.3f；平均秩 Spearman %.3f；MAE %.3f；实际=1.00 的组 %.0f%%"%(torch.corrcoef(torch.stack([rk(P),rk(A)]))[0,1].item(),torch.corrcoef(torch.stack([torch.tensor(D.avg_rank(P)),torch.tensor(D.avg_rank(A))]))[0,1].item(),st.mean(abs(p-a) for p,a in zip(P,A)),100*st.mean(a==1 for a in A)))
v1("SLQA",list(slqa_windows()),(0.05,0.10,0.20),3,5)
v1("BIPM",list(bipm_windows()),(6.0,12.0,24.0),4,6)
