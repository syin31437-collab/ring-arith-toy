from e_eval import *
def wilson(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
for name in O:
    rows,base=O[name]; n=len(base)
    for L,k in LV.items():
        a=sum(any(v[2]>3 for key,v in base[w].items() if key[0] in k) for w in base)
        print(name,L,f"S0 报警窗 {a}/{n} Wilson 95% {tuple(round(x,2) for x in wilson(a,n))}；odd/even 各",[sum(any(v[2]>3 for key,v in base[w].items() if key[0] in k) for w in base if w%2==p) for p in(0,1)])
    G=table(name)
    print("  固定 b 替换（所有级别同一 b）：",end="")
    for bb in (0,.1,.2,.3,.4,.5,.6,.7,.8):
        P=[];A=[]
        for kk,v in G.items():
            P.append(np.mean([1-(1-x[0])*(1-bb) for x in v]));A.append(np.mean([x[1] for x in v]))
        print(f"b={bb}:{np.mean(np.abs(np.array(P)-A)):.3f}/ρ{spearman(P,A):.2f} ",end="")
    print()
