from rtc import *
import os,re,collections
os.chdir(REPO+"/bipm")
import bipm as B, events as E
ev=E.parse_notes(); W_,keep=E.windows()
lab_ev={k:[e for e in ev.get(k,[]) if e[0]>=B.SANDBOX_END] for k in keep}
print("vsl 在 keep?", "vsl" in keep, "za 在 keep?", "za" in keep, "dmdm", "dmdm" in keep, "bev","bev" in keep)
print("vsl 事件",len(lab_ev.get("vsl",[])),"za",len(lab_ev.get("za",[])),"az",len(lab_ev.get("az",[])),"bev",len(lab_ev.get("bev",[])))
def has(w,E_): return any(w["lo"]<=m<=w["hi"] for m,_,_ in E_.get(w["lab"],[]))
def rep(tag,E_):
    for L,i,nm in (("L2",0,"EWMA"),("L2",1,"单点")):
        a=[has(w,E_) for w in W_ if w[L][i]];b=[has(w,E_) for w in W_ if not w[L][i]]
        print(f"  {tag} {nm}: 报警窗 {np.mean(a):.2f}({len(a)}) 无报警 {np.mean(b):.2f} 提升度 {np.mean(a)/np.mean(b):.2f}; 召回(事件窗被报警) {sum(a)/(sum(a)+sum(b)):.2f} 精确率 {np.mean(a):.2f}")
rep("原解析",lab_ev)
E2=dict(lab_ev); E2["vsl"]=lab_ev.get("vsl",[])+lab_ev.get("za",[]); E2["za"]=[]
rep("za→vsl 重新归属",E2)
# 只留“非常规调频”：去掉含 frequency / microstep / correction 的日常操纵行
ev3=collections.defaultdict(list); cur=None
for line in open("data/notes.utc",errors="ignore"):
    m=re.match(r"\s{3,6}([A-Z][A-Z0-9]{1,4})\s*\.",line)
    if m: cur=m.group(1).lower()
    if cur is None: continue
    low=line.lower()
    if any(k in low for k in("frequency correction","on mjd","microstep")) and not any(k in low for k in("step of","master clock")): 
        continue
    for mj in re.findall(r"MJD[^0-9]{0,8}(\d{5})",line)[:1]:
        ev3[cur].append((int(mj),"x",None))
E3={k:[e for e in ev3.get(k,[]) if e[0]>=B.SANDBOX_END] for k in keep}
print("去日常操纵后事件数",sum(len(v) for v in E3.values()))
rep("去日常操纵",E3)
