"""无差别投放的候选通用表达式：离线检验（2026-11-02，库伯：发现有问题但定位不了时，对全部节点无差别投放，不伤好节点、纠正坏节点）。

表达式（投影到集合 C；x 在 C 内就不动）：x' = P_C(x) = clip(x, y − r, y + r)（一维），高维同理，C 是任意闭凸集。
  不伤好节点：θ ∈ C ⇒ ‖P_C(x) − θ‖² ≤ ‖x − θ‖² − ‖P_C(x) − x‖²（投影的非扩张性，老定理），所以任何节点都不变差，不需要知道谁好谁坏。
  纠正坏节点：一维且 θ ∈ C 时，新误差 = 旧误差 − d（d = x 到 C 的距离）；diam(C) ≤ h 时全部节点都落进门槛。
  θ ∉ C（离 C 的距离 g）：一维时新误差 ≤ max(旧误差, g + diam(C))，这就是伤害的上限。
  投放物和任务的关系：投放物 = 一个集合；来源可以是同伴（中位数 ± r）、证据（报告 ± 噪声）、任务约束、锚。它有用当且仅当"包含真相且足够窄"。

这里只检验同伴来源（y = 其他 5 人第①步中位数，r = c × 其他 5 人的 MAD），只用第①步的数字、不调模型；不代表 LLM 会照做。
c = 0.5 是看过旧 57 局后选的（旧数据：k ≤ 1 时 K ≥ 99.5%、B ≥ 88%；data3 没有一个 c 能同时好）。

预注册（提交时 blindrule.py 的正式局在跑，新数据的数值没看过）：
  E1 定理核对：θ ∈ C 的节点-题上，误差变大的次数 = 0（查代码）
  E2 k ≤ 1、c = 0.5：K ≥ 98%，B ≥ 80%，全员准确率 ≥ 90%
  E3 k ≤ 1、c = 0.5：K ≥ LLM 实测的 K（同局 note 组）
  E4 k = 3：c ∈ {0.5, 1, 1.5, 2} 里没有一个能同时 K ≥ 90% 且 B ≥ 80%（信息不够时，表达式应该说做不到）
  E5 每个 k：c 增大时 K 不降、B 不升（宽度的取舍）
  E6 只用 (n, k, δ, σ, h, c) 的高斯模拟（σ = k = 0 第①步节点误差 MAD）预测每个 k 的 K、B、θ ∈ C 比例，和实测差的绝对值 ≤ 0.05
不算通过的情况照实报；不改判据。这个实验的数据生成就是高斯噪声，E6 通过不能说明跨领域通用。

用法：python blind_clip.py <目录>
"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

H = math.log10(1.4) / 2
DELTA = math.log10(1.4)
A = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b", "claude_haiku", "codex"]
CS = (0.5, 1.0, 1.5, 2.0)
ok = lambda x: abs(x) <= H


def mad(v):
    m = st.median(v)
    return 1.4826 * st.median(abs(x - m) for x in v)


def clip(x, y, r):
    return max(y - r, min(y + r, x))


def load(d):
    R = []
    for fn in sorted(glob.glob(os.path.join(d, "*", "run.json"))):
        J = json.load(open(fn))
        for it in J["items"]:
            i, t = it["id"], math.log10(it["truth"])
            if not all(J["s1"][a].get(i, 0) > 0 and J["note"]["s3"][a].get(i, 0) > 0 for a in A):
                continue
            s = {a: math.log10(J["s1"][a][i]) - t for a in A}
            for a in A:
                o = [s[x] for x in A if x != a]
                R.append(dict(k=J["k"], s1=s[a], y=st.median(o), m=mad(o), s3=math.log10(J["note"]["s3"][a][i]) - t))
    return R


def stats(R, c):
    g, b = [r for r in R if ok(r["s1"])], [r for r in R if not ok(r["s1"])]
    f = lambda r: clip(r["s1"], r["y"], c * r["m"])
    return dict(K=st.mean(ok(f(r)) for r in g) if g else float("nan"), B=st.mean(ok(f(r)) for r in b) if b else float("nan"), acc=st.mean(ok(f(r)) for r in R), inside=st.mean(abs(r["y"]) <= c * r["m"] for r in R))


def simulate(k, sigma, c, n=6, trials=40000, seed=0):
    g = random.Random(seed + k)
    R = []
    for _ in range(trials // n):
        s = [g.gauss(0, sigma) + (DELTA if j < k else 0) for j in range(n)]
        for j in range(n):
            o = s[:j] + s[j + 1:]
            R.append(dict(s1=s[j], y=st.median(o), m=mad(o)))
    return stats(R, c)


def main():
    R = load(sys.argv[1])
    print(f"节点-题 {len(R)}")
    sigma = mad([r["s1"] for r in R if r["k"] == 0])
    print(f"σ（k = 0）= {sigma:.4f}")
    viol = sum(abs(clip(r["s1"], r["y"], c * r["m"])) > abs(r["s1"]) + 1e-12 for c in CS for r in R if abs(r["y"]) <= c * r["m"])
    print(f"E1 θ ∈ C 时误差变大的次数：{viol} {'✅' if viol == 0 else '❌'}")
    print("\nk  c    K      B      全员   θ∈C  ｜模拟 K     B     θ∈C ｜LLM(note) K    B    全员")
    T = {}
    for k in (0, 1, 2, 3, 4):
        X = [r for r in R if r["k"] == k]
        if not X:
            continue
        g, b = [r for r in X if ok(r["s1"])], [r for r in X if not ok(r["s1"])]
        lk, lb, la = st.mean(ok(r["s3"]) for r in g), st.mean(ok(r["s3"]) for r in b), st.mean(ok(r["s3"]) for r in X)
        for c in CS:
            m, p = stats(X, c), simulate(k, sigma, c)
            T[k, c] = (m, p, lk)
            print(f"{k}  {c:.1f}  {m['K']:5.1%} {m['B']:5.1%} {m['acc']:5.1%} {m['inside']:4.0%} ｜{p['K']:5.1%} {p['B']:5.1%} {p['inside']:4.0%} ｜{lk:5.1%} {lb:5.1%} {la:5.1%}")
    print("\n预注册")
    for k in (0, 1):
        m, _, lk = T[k, 0.5]
        print(f"E2 k={k}: K {m['K']:.1%} ≥ 98% {'✅' if m['K'] >= .98 else '❌'}；B {m['B']:.1%} ≥ 80% {'✅' if m['B'] >= .8 else '❌'}；全员 {m['acc']:.1%} ≥ 90% {'✅' if m['acc'] >= .9 else '❌'}")
        print(f"E3 k={k}: K {m['K']:.1%} ≥ LLM {lk:.1%} {'✅' if m['K'] >= lk else '❌'}")
    both = [c for c in CS if T[3, c][0]["K"] >= .9 and T[3, c][0]["B"] >= .8]
    print(f"E4 k=3 同时 K ≥ 90%、B ≥ 80% 的 c：{both or '无'} {'✅' if not both else '❌'}")
    for k in (0, 1, 2, 3, 4):
        Ks, Bs = [T[k, c][0]["K"] for c in CS], [T[k, c][0]["B"] for c in CS]
        print(f"E5 k={k}: K 不降 {'✅' if all(a <= b + 1e-9 for a, b in zip(Ks, Ks[1:])) else '❌'}；B 不升 {'✅' if all(a >= b - 1e-9 for a, b in zip(Bs, Bs[1:])) else '❌'}")
    worst = max((abs(T[k, c][0][q] - T[k, c][1][q]), k, c, q) for k in (0, 1, 2, 3, 4) for c in CS for q in ("K", "B", "inside"))
    print(f"E6 模拟与实测最大差 {worst[0]:.3f}（k={worst[1]}，c={worst[2]}，{worst[3]}）≤ 0.05 {'✅' if worst[0] <= .05 else '❌'}")


if __name__ == "__main__":
    main()
