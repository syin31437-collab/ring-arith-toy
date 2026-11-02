"""复算 distinguish v2，逐窗逐序列保留 (mu, se, z_actual)；存 pickle。先核对能复现 0.92/0.082, 0.86/0.038。"""
from rtc import *
import pickle, distinguish as D, ladder2 as LL
def cell2(cal, ev, s, zeta, drift=False):
    n = len(cal[0][2]); T_ = len(ev)
    Rc=[x[2] for x in cal]; Ac,Bc=[x[0] for x in cal],[x[1] for x in cal]
    if drift:
        Re=[[x[2][j]+2*s[j]*(t+1)/T_ for j in range(n)] for t,x in enumerate(ev)]
    else:
        Re=[[x[2][j]+s[j] for j in range(n)] for x in ev]
    Ae,Be=[x[0]+zeta for x in ev],[x[1] for x in ev]
    Sc,Se=D.series(Rc,Ac,Bc),D.series(Re,Ae,Be)
    out={}
    for key in Sc:
        c=[v for v in Sc[key] if v==v]; e=[v for v in Se[key] if v==v]
        if len(c)<5 or len(e)<5:
            out[key]=(0.0,1.0,0.0); continue
        m=D._mad(c); mu=D.shift_of(key,list(s),zeta)
        extra=(abs(mu)*1.4826/2) if drift else 0.0
        se=1.2533*math.sqrt(m**2/len(c)+(m**2+extra**2)/len(e))
        out[key]=(mu,max(se,1e-9),D.stat(c,e))
    return out
def run(name,windows,deltas,kmin,kmaj):
    rg=random.Random("dist2-"+name)
    rows=[]; base={}
    for wi,(cal,ev) in enumerate(windows):
        n=len(cal[0][2])
        base[wi]=cell2(cal,ev,[0.0]*n,0.0)
        for d in deltas:
            for sc,k in (("one",1),("min",kmin),("maj",kmaj),("all",n)):
                idx=set(rg.sample(range(n),k))
                rows.append((wi,sc,d,cell2(cal,ev,[d if j in idx else 0.0 for j in range(n)],0.0)))
            rows.append((wi,"drift",d,cell2(cal,ev,[d]*n,0.0,drift=True)))
            rows.append((wi,"scale",d,cell2(cal,ev,[0.0]*n,d)))
    return rows,base
out={}
out["SLQA"]=run("SLQA",windows_cached("SLQA"),(0.005,0.01,0.02,0.03),3,5)
out["BIPM"]=run("BIPM",windows_cached("BIPM"),(1.0,2.0,3.0,4.0),4,6)
pickle.dump(out,open(W+"/e_cells.pkl","wb"))
