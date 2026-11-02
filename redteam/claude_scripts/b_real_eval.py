from rtc import *
import json
R=json.load(open(W+"/b_real.json"))
tags=[]
for r in R:
    if r["tag"] not in tags: tags.append(r["tag"])
def L(s,tag):
    a=np.array([r["act"] for r in s]);p=np.array([r["pt"] for r in s]);b=np.array([r["b2"] for r in s])
    info=(a>=.1)&(a<=.9)
    print(f"  {tag:28s} {len(s):4d}格 实测{a.mean():.3f} 预测{p.mean():.3f} | v6.1 MAE {np.abs(p-a).mean():.3f} ρ{spearman(p,a):+.2f} | B2 {np.abs(b-a).mean():.3f} ρ{spearman(b,a):+.2f} | 常数 {np.abs(a-a.mean()).mean():.3f} | 信息格{info.mean():.0%}: v6.1 {np.abs(p-a)[info].mean():.3f} B2 {np.abs(b-a)[info].mean():.3f}")
for t in tags:
    L([r for r in R if r["tag"]==t],t)
    for j in("软","硬2"): L([r for r in R if r["tag"]==t and r["judge"]==j],"  "+j)
print("解耦后按 δ（h=1.5）")
for t in ("解耦 h=1.5","解耦 h=3"):
  for dl in (3.,6.,12.):
    for j in("软","硬2"): L([r for r in R if r["tag"]==t and r["dl"]==dl and r["judge"]==j],f"{t} δ={dl:g} {j}")
