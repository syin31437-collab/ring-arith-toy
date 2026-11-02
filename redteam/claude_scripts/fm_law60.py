"""v6.0 vs v6.1 在 Fashion-MNIST 测试格上：v6.1 的补丁来自 Fashion 沙盒，补丁值多少？"""
import os; os.environ["FL_LR"]="0.1"; os.environ["FL_DATA"]="data_fashion"
from rtc import *
os.chdir(REPO+"/fl")
import itertools,json,torch
import flbench as F, law_v6 as L1, law60 as L0
from analytic import sigma_c
from hetero import measure
J=json.load(open(REPO+"/validate_v61.json"))["cells"]
attacks=["ALIE0.5","ALIE1","ALIE2","ALIE4","noise","flip"]; k=0; out=[]; i=0
for n,f,attack,m in itertools.product((10,20),(0.2,0.4,0.6),attacks,(32,512)):
    k+=1
    clean,bad,_=F.run(n,f,attack,m,7000+k,dense=0.5,v2=True)
    for judge in("软","硬2"):
        s0=max(sigma_c(clean)[1],1e-6)
        p1=L1.predict(clean,bad,judge,s0); p0=L0.predict(clean,bad,judge,s0); a=measure(clean+bad,bad[0][3],judge,s0)["miss"]
        out.append(dict(attack=attack,judge=judge,m=m,n=n,f=f,pt1=p1,pt0=p0,act=a,jact=J[i]["act"],jpt=J[i]["pt"])); i+=1
    print(k,round(out[-1]["pt1"],3),round(out[-1]["jpt"],3),flush=True)
json.dump(out,open(W+"/fm_law60.json","w"),ensure_ascii=False)
a=np.array([r["act"] for r in out]);p1=np.array([r["pt1"] for r in out]);p0=np.array([r["pt0"] for r in out])
print("复现 json：act 差",np.abs(a-np.array([r['jact'] for r in out])).max(),"pt 差",np.abs(p1-np.array([r['jpt'] for r in out])).max())
print(f"v6.1 MAE {np.abs(p1-a).mean():.3f} ρ {spearman(p1,a):.3f}; v6.0 MAE {np.abs(p0-a).mean():.3f} ρ {spearman(p0,a):.3f}")
