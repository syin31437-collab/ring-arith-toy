"""无差别投放表达式的一维精确形式（2026-11-02，看过 blindrule 正式结果后推导，数学恒等式，不算预注册的检验）。

设 e = y − θ（投放中心的误差），C = [y − r, y + r]，h = 门槛。节点投影到 C：
  好节点（|x − θ| ≤ h）保住 ⇔ |e| ≤ h + r（C 和真相的 h 邻域相交；不相交则全部好节点都变错）
  坏节点在 θ 的 s 侧（s = ±1）回来 ⇔ x 在 C 外，且 −(h + r) ≤ s·e ≤ h − r
    （x 在 C 内不动；x 在 C 外落到靠近自己一侧的边界 y + s·r，新误差 = e + s·r）
即 r 让"不伤好节点"的余量变成 h + r，让"纠正坏节点"的余量变成 h − r。高维没有这个等价，只有 θ ∈ C 的充分条件。
用法：python blind_margin.py <目录>
"""
import glob
import json
import math
import statistics as st
import sys

H = math.log10(1.4) / 2
A = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b", "claude_haiku", "codex"]
CS = (0.5, 1.0, 1.5, 2.0)


def mad(v):
    m = st.median(v)
    return 1.4826 * st.median(abs(x - m) for x in v)


R = []
for fn in sorted(glob.glob(sys.argv[1] + "/*/run.json")):
    J = json.load(open(fn))
    for it in J["items"]:
        i, t = it["id"], math.log10(it["truth"])
        s = {a: math.log10(J["s1"][a][i]) - t for a in A}
        for a in A:
            o = [s[x] for x in A if x != a]
            R.append(dict(k=J["k"], x=s[a], e=st.median(o), m=mad(o)))
bad_k = bad_b = n_k = n_b = 0
print("k  c    K实测  K=P(|e|≤h+r)  B实测  B公式   宽度r中位  |e|中位")
for k in range(5):
    X = [r for r in R if r["k"] == k]
    for c in CS:
        kk, kp, bb, bp = [], [], [], []
        for r in X:
            rr = c * r["m"]
            nx = max(r["e"] - rr, min(r["e"] + rr, r["x"]))
            if abs(r["x"]) <= H:
                a, p = abs(nx) <= H, abs(r["e"]) <= H + rr
                kk.append(a); kp.append(p); bad_k += a != p; n_k += 1
            else:
                sg = 1 if r["x"] > 0 else -1
                out = abs(r["x"] - r["e"]) > rr
                a = abs(nx) <= H
                landing = r["e"] + (1 if r["x"] > r["e"] else -1) * rr
                p = out and abs(landing) <= H
                bb.append(a); bp.append(p); bad_b += a != p; n_b += 1
        print(f"{k}  {c:.1f}  {st.mean(kk):5.1%}   {st.mean(kp):5.1%}       {st.mean(bb):5.1%}  {st.mean(bp):5.1%}   {st.median(c * r['m'] for r in X):.3f}    {st.median(abs(r['e']) for r in X):.3f}")
print(f"恒等式核对：好节点不符 {bad_k}/{n_k}；坏节点不符 {bad_b}/{n_b}")
