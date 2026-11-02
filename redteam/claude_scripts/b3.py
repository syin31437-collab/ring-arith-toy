"""一个 5 行零拟合对手 B3：裁判输出相对秤的偏差分布，扣掉秤噪声后算 P(|N(μ,σ²)|>h)。不用诚实后验、不用子集。"""
import os,sys
os.environ["FL_LR"]="0.1"; os.environ["FL_DATA"]="data_fashion"
from rtc import *
import json,itertools,torch,collections
import law_v6 as L, validate_bipm as V
from analytic import Phi,sigma_c
from hetero import judge_out,measure
def b3(clean,bad,judge,s0):
    """全部只用 校准题(无真值) + 评估题(无真值)。"""
    b,_=sigma_c(clean)
    jd="软" if judge=="软" else "硬"; s=s0
    if judge=="硬2":
        dev=torch.stack([r-p for _,p,r,_ in clean]); s=max(s,L._mad(dev).min().item())
    Rc=torch.stack([r for _,_,r,_ in clean])
    cons=Rc.median(1).values; pc=torch.tensor([p for _,p,_,_ in clean])
    n=Rc.shape[1]; sig=L._loo(Rc).median().item()
    vc=max(L._mad(pc-b-cons).item()**2-(math.pi/2)*sig**2/n,(0.1*L._mad(pc-b-cons).item())**2)
    d=[judge_out(r,torch.tensor(p),s,jd)-(p-b) for _,p,r,_ in bad]
    h=bad[0][3]/2; mu=st.median(d); md=1.4826*st.median(abs(x-mu) for x in d)
    so=math.sqrt(max(md**2-vc,1e-4*md**2+1e-12))        # 扣掉秤噪声
    return 1-(Phi((h-mu)/so)-Phi((-h-mu)/so))
if __name__=="__main__":
    J=json.load(open(REPO+"/bipm/validate_bipm_test.json"))["cells"]; i=0; out=[]
    rng=random.Random("bipm-inject")
    for gi,w,cal,ev in V.windows("test"):
        for h,sh in ([(1.5,None),(3.0,None)]+[(dl/2,(k,dl)) for k,dl in itertools.product((2,4,6),(3.,6.,12.))]):
            s_=torch.zeros(9)
            if sh: idx=rng.sample(range(9),sh[0]); s_[idx]=sh[1]
            c=[(False,a,r,2*h) for _,a,_,r in cal]; e=[(True,a,r+s_,2*h) for _,a,_,r in ev]
            if sh:
                # 与原脚本的随机流保持一致：B、S 在外层消费
                pass
            for j in("软","硬2"):
                s0=max(sigma_c(c)[1],1e-6); a=measure(c+e,2*h,j,s0)["miss"]
                out.append(dict(tag="自然" if not sh else "注入",judge=j,h=h,act=a,b3=b3(c,e,j,s0),pt=L.predict(c,e,j,s0),b2_j=J[i]["b2"],jact=J[i]["act"],k=sh[0] if sh else 0)); i+=1
        # 同样消费 S 的随机
        for dl in (6.,12.): rng.sample(range(9),4)
    a=np.array([r["act"] for r in out]);p=np.array([r["pt"] for r in out]);b=np.array([r["b3"] for r in out])
    print("随机流复现：act 与 json 一致比例",np.mean(np.abs(a-np.array([r["jact"] for r in out]))<1e-9))
    def L_(t,s):
        A=np.array([r["act"] for r in s]);P=np.array([r["pt"] for r in s]);B=np.array([r["b3"] for r in s]);I=(A>=.1)&(A<=.9)
        print(f" {t:16s} {len(s)}格 v6.1 MAE {np.abs(P-A).mean():.3f} ρ {spearman(P,A):.2f} 信息格 {np.abs(P-A)[I].mean():.3f} | B3 MAE {np.abs(B-A).mean():.3f} ρ {spearman(B,A):.2f} 信息格 {np.abs(B-A)[I].mean():.3f}")
    print("BIPM"); L_("全部",out)
    for j in("软","硬2"): L_(j,[r for r in out if r["judge"]==j])
    for t in("自然","注入"): L_(t,[r for r in out if r["tag"]==t])
    json.dump(out,open(W+"/b3_bipm.json","w"),ensure_ascii=False)
