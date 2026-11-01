"""失效地图：三比值（ρ, κ, λ）的曲线在什么条件下预测不准，以及以前失败的案例能不能被它解释。

曲线：hetero.py 的 C 组特征（δ_obs 版 ρ、κ + λ），在 异质玩具 + 时间玩具 + 代数玩具 上合并拟合漏检率，之后一律零重拟合。
一、压力测试：在异质玩具里一次只破坏一条前提，从轻到重扫，记录 预测误差（实际 − 预测，平均和 MAE）怎么随强度变：
  共享    故障题上约束预测 p 也被带偏 ζ × 平均毒量（区域内错误；ζ = 0 是原设定）
  厚尾    节点噪声换成自由度 df 的 t 分布（df = ∞ 是原设定）
  δ 不定  每道故障题的 δ 乘一个对数正态因子（σ_log = 0 是原设定）
  少数坏  7 个节点里 n_clean 个干净（原设定 1–3，即投毒过半；4–6 = 投毒是少数）
  节点数  N ∈ {3, 5, 7, 15, 31}，干净的占 1/7–3/7
二、以前的失败案例（量出三个比值，和曲线比）：
  国家密度（llm/，共享信念）、硬裁判"全剔除时用预测值"（algebra.py 旧版）、"自己当答案"（geometry.py guard=False）
判定：|实际 − 预测| > 0.15 算失效，并记下当时的条件；失效比例要和无压力基线比（基线本身就有 20–35%），不看绝对值。
防假通过：老案例用的 geo/alg 配置（种子 0–39）从拟合集里排除；稳健版 E 是看过压力测试后设计的，
  所以另加两条 E 没见过的压力（运行前写下的预测，见 NEW）。
NEW 预注册预测（11-01，运行前）：
  相关噪声 c（各节点噪声共享一个公共成分）：C 和 E 都偏乐观（平均误差 > 0），因为公共成分不出现在节点之间的分散里
  反向投毒 f（被投毒节点里 f 比例往 −δ 推）：E 的 MAE 每档都 ≤ C
  两条新压力上 E 的 MAE 都不高于 C
第二轮预注册（11-01，运行前；新种子 stress2，全部压力重扫）：
  F（λ 计入好节点数）：节点数 N = 31 时 |平均误差| < 0.05（C 是 −0.10 / −0.14），无压力基线 MAE 不比 C 高 0.02 以上
  G（再加 δ 分散度 q）：σ_log = 1.5 时 |平均误差| 比 C 小一半以上（C 是 −0.13 / −0.23），基线 MAE 不比 C 高 0.02 以上
  共享 ζ：F、G 都改变不了乐观失效（ζ 从内部看不见）
第三轮预注册（11-01，运行前；新种子 stress3；漏检改按每题自己的 δ_k/2 判）：
  拟合玩具换成 het2（N ∈ 3–31、σ_log ∈ 0–1.5 随机），曲线 C2 / F2 / G2：
    G2 在 σ_log = 1.5 和 N = 3、31 时 |平均误差| ≤ 0.05（软、硬都要）
    G2 无压力基线 MAE ≤ C2 + 0.03
    没见过的压力（厚尾、相关噪声、反向投毒）上，G2 的 MAE 不比第二轮的 C 高 0.03 以上
    投毒实验漏检 R²：G2 ≥ C（0.39 / 0.60）
  逐事件（理想版，用真实 δ_k）：C2 逐事件在硬裁判 σ_log = 1.5 时 |平均误差| ≤ 0.05
第四轮预注册（11-01，运行前；新种子 stress4）：H2 = G2 的稳健版（MAD 尺度）
  厚尾 df = 1 时 |平均误差| ≤ 0.08（软、硬）
  σ_log = 1.5、N = 3、N = 31 时 |平均误差| 仍 ≤ 0.05
  无压力基线 MAE ≤ 同种子 G2 + 0.02
  投毒实验漏检 R² ≥ 0.5（软、硬）

用法：~/gut-external/torch215-venv/bin/python failmap.py stress | stress2 | stress3 | stress4 | cases
"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

import torch

import algebra
import geometry
from geometry import measure_full
from hetero import N as N0, fit, judge_out, lu, measure, predict

HERE = os.path.dirname(os.path.abspath(__file__))
BAD = 0.15


def curve(judge, kind="C", hold=40, src="het"):
    rows = json.load(open(os.path.join(HERE, f"{src}_{judge}.json")))
    for env in ("geo", "alg"):
        for f in sorted(glob.glob(os.path.join(HERE, f"tr_{env}_{judge}_*.json"))):
            rows += [r for r in json.load(open(f)) if r["sigma_eff"] > 0 and r["sigma_node"] > 0 and "sigma_min" in r and r["seed"] >= hold]
    return fit(rows, kind, "miss")


def stress_toy(k, judge, zeta=0.0, df=None, dvar=0.0, n_clean=None, n=N0, corr=0.0, flip=0.0, tag="stress", same=False, cmult=1.0, couple=0.0, raw=False, nbias=0.0):
    rng = random.Random(f"{tag}-{k}")
    g = torch.Generator().manual_seed({"stress": 50000, "stress2": 70000, "stress3": 90000, "stress4": 110000}.get(tag, 130000) + k)
    sc, scl = lu(rng, 0.01, 1), lu(rng, 0.05, 2)
    spo, delta, tlo = scl * lu(rng, 1, 30), lu(rng, 0.03, 30), rng.uniform(0, 1)
    spo, sc = (scl if same else spo), sc * cmult              # same：投毒节点和干净节点一样精确；cmult：约束噪声放大
    nc = n_clean if n_clean is not None else max(1, round(n * rng.choice([1, 2, 3]) / 7))
    sig = torch.tensor([scl] * nc + [spo] * (n - nc)) * (0.5 + torch.rand(n, generator=g))
    tau = torch.tensor([0.0] * nc + [rng.uniform(tlo, 1) * (-1 if rng.random() < flip else 1) for _ in range(n - nc)])
    t = torch.distributions.StudentT(df) if df else None
    beta = nbias * scl * torch.randn(n, generator=g)        # nbias：每个节点自己的系统偏差（每道题都一样），单位 = 干净节点噪声
    rows = []
    for _ in range(400):
        fault = torch.rand(1, generator=g).item() < 0.3
        dk = delta * math.exp(dvar * torch.randn(1, generator=g).item()) if fault else 0.0
        noise = t.sample((n,)) if t else torch.randn(n, generator=g)
        noise = math.sqrt(1 - corr) * noise + math.sqrt(corr) * torch.randn(1, generator=g)
        r = sig * noise + tau * dk + (couple * dk if fault else 0.0) * (tau == 0).float() + beta   # couple：诚实节点也被带偏 α·δ
        p = sc * torch.randn(1, generator=g).item() + zeta * tau[nc:].abs().mean().item() * dk
        rows.append((fault, p, r, dk if fault else delta))
    if raw:                                                  # analytic.py 用：原始题目（真值 = 0）；raw="nc" 时再返回干净节点数
        return (rows, delta, sc, nc) if raw == "nc" else (rows, delta, sc)
    return measure(rows, delta, judge, s=sc)


SWEEPS = {
    "共享 ζ": ("zeta", [0.0, 0.25, 0.5, 0.75, 1.0]),
    "厚尾 df": ("df", [None, 10, 4, 2, 1]),
    "δ 不定 σ_log": ("dvar", [0.0, 0.3, 0.6, 1.0, 1.5]),
    "干净节点数 / 7": ("n_clean", [1, 2, 3, 4, 5, 6]),
    "节点数 N": ("n", [3, 5, 7, 15, 31]),
    "NEW 相关噪声 c": ("corr", [0.0, 0.25, 0.5, 0.75, 0.9]),
    "NEW 反向投毒 f": ("flip", [0.0, 0.2, 0.4, 0.5]),
}


def stress2():
    """第二轮：新种子，比较 C / F / G。"""
    torch.manual_seed(1)
    for judge in ("软", "硬"):
        ws = {k: curve(judge, k) for k in ("C", "F", "G")}
        print(f"\n== {judge}裁判：第二轮压力测试（新种子；每格 = 平均误差 / MAE） ==")
        print("前提            强度     实际   " + "".join(f"{k:>16s}" for k in ws))
        for name, (key, levels) in SWEEPS.items():
            for lv in levels:
                R = [stress_toy(k, judge, tag="stress2", **{key: lv}) for k in range(80)]
                cells = []
                for kind, w in ws.items():
                    e = [r["miss"] - p for r, p in zip(R, predict(w, R, kind).tolist())]
                    cells.append(f"{st.mean(e):+7.2f} / {st.mean(abs(x) for x in e):4.2f}")
                print(f"{name:14s} {str(lv):>6s}   {st.mean(r['miss'] for r in R):5.0%}   " + "  ".join(cells))


def per_event(w, R, kind):
    """逐事件预测：每道故障题用自己的 δ_k 查曲线再平均（d_obs 按 δ_k 等比缩放）。用的是真实 δ_k，是理想版。"""
    out = []
    for m in R:
        mk = sum(m["dks"]) / len(m["dks"])
        rows = [{**m, "delta": dk, "d_obs": m["d_obs"] * dk / mk} for dk in m["dks"]]
        out.append(predict(w, rows, kind).mean().item())
    return out


def stress3(round4=False):
    """第三轮：拟合玩具 het2，新种子，C（旧）/ C2 / F2 / G2 / C2 逐事件。第四轮（round4）：G2 / H2 / H2 逐事件，种子 stress4。"""
    from hetero import r2, real
    torch.manual_seed(3 if round4 else 2)
    tag = "stress4" if round4 else "stress3"
    for judge in ("软", "硬"):
        if round4:
            ws = {"G2": (curve(judge, "G", src="het2"), "G"), "H2": (curve(judge, "H", src="het2"), "H")}
        else:
            ws = {"C": (curve(judge), "C"), "C2": (curve(judge, src="het2"), "C"),
                  "F2": (curve(judge, "F", src="het2"), "F"), "G2": (curve(judge, "G", src="het2"), "G")}
        ev = "H2" if round4 else "C2"
        names = list(ws) + [f"{ev}逐事件"]
        print(f"\n== {judge}裁判：{'第四' if round4 else '第三'}轮压力测试（新种子；每格 = 平均误差 / MAE） ==")
        print("前提            强度     实际   " + "".join(f"{k:>16s}" for k in names))
        for name, (key, levels) in SWEEPS.items():
            for lv in levels:
                R = [stress_toy(k, judge, tag=tag, **{key: lv}) for k in range(80)]
                preds = [predict(w, R, kind).tolist() for w, kind in ws.values()] + [per_event(ws[ev][0], R, ws[ev][1])]
                cells = []
                for pr in preds:
                    e = [r["miss"] - p for r, p in zip(R, pr)]
                    cells.append(f"{st.mean(e):+7.2f} / {st.mean(abs(x) for x in e):4.2f}")
                print(f"{name:14s} {str(lv):>6s}   {st.mean(r['miss'] for r in R):5.0%}   " + "  ".join(cells))
        Rr = real(judge)
        print("投毒实验漏检 R²：" + "  ".join(f"{k} {r2(Rr, predict(w, Rr, kind), 'miss'):.2f}" for k, (w, kind) in ws.items()))


def stress():
    torch.manual_seed(0)
    for judge in ("软", "硬"):
        w, we = curve(judge), curve(judge, "E")
        print(f"\n== {judge}裁判：压力测试（每档 80 个随机配置；误差 = 实际漏检 − 预测漏检） ==")
        print("前提            强度     实际    预测    平均误差   MAE    失效比例(|误差|>0.15)  ‖ 稳健版 E：预测  平均误差  MAE  失效比例")
        for name, (key, levels) in SWEEPS.items():
            for lv in levels:
                R = [stress_toy(k, judge, **{key: lv}) for k in range(80)]
                pr = predict(w, R, "C").tolist()
                e = [r["miss"] - p for r, p in zip(R, pr)]
                pe = predict(we, R, "E").tolist()
                ee = [r["miss"] - p for r, p in zip(R, pe)]
                print(f"{name:14s} {str(lv):>6s}   {st.mean(r['miss'] for r in R):5.0%}   {st.mean(pr):5.0%}   "
                      f"{st.mean(e):+8.2f}   {st.mean(abs(x) for x in e):5.2f}   {sum(abs(x) > BAD for x in e) / len(e):6.0%}"
                      f"              ‖   {st.mean(pe):5.0%}   {st.mean(ee):+6.2f}   {st.mean(abs(x) for x in ee):5.2f}  {sum(abs(x) > BAD for x in ee) / len(ee):5.0%}")


def report(name, judge, w, R, extra=""):
    for kind in ("C", "E"):
        report1(name if kind == "C" else "", judge, w[kind], R, kind, extra if kind == "C" else "")


def report1(name, judge, w, R, kind, extra):
    pr = predict(w, R, kind).tolist()
    e = [r["miss"] - p for r, p in zip(R, pr)]
    lg = lambda key: st.median(math.log10(max(r["d_obs"], 1e-3 * r["delta"]) / r[key]) if key != "sigma_min" else math.log10(r["delta"] / r[key]) for r in R)
    print(f"{name:22s} {kind} {judge}  {len(R):4d}   ρ {10 ** lg('sigma_eff'):8.2f}  κ {10 ** lg('sigma_node'):6.2f}  λ {10 ** lg('sigma_min'):6.2f}   "
          f"实际 {st.mean(r['miss'] for r in R):4.0%}  预测 {st.mean(pr):4.0%}  失效 {sum(abs(x) > BAD for x in e) / len(e):4.0%}  {extra}")


def density_rows():
    """国家密度（llm/analyze.py 同一套数据）：每个事件单独当一个配置，δ = |共识 − 真值|。"""
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    from analyze import answers
    from collect import truth
    T, A = truth(), answers()
    items = []
    for iso, t in T.items():
        a = A.get(iso, {})
        if min(len(a.get(k, {})) for k in ("pop", "area", "dens")) < 5:
            continue
        items.append((a["dens"], st.median(a["pop"].values()) + 6 - st.median(a["area"].values()), math.log10(t["dens"])))
    items = [(torch.tensor(list(dd.values())), p, tr, dd) for dd, p, tr in items]
    ev = [x for x in items if abs(x[0].median().item() - x[2]) > 0.05]
    ok = [x for x in items if abs(x[0].median().item() - x[2]) <= 0.05]
    s = st.pstdev(p - tr for _, p, tr, _ in ok)
    sn = math.sqrt(st.mean(((d - d.median()) ** 2).mean().item() for d, _, _, _ in ok))
    nodes = {n for *_, dd in ok for n in dd}
    smin = min(math.sqrt(st.mean((dd[n] - p) ** 2 for _, p, _, dd in ok if n in dd)) for n in nodes)
    out = []
    mad = lambda xs: 1.4826 * st.median(abs(x - st.median(xs)) for x in xs)
    snr = st.median(1.4826 * (d - d.median()).abs().median().item() for d, *_ in ok)
    for judge in ("软", "硬"):
        so = mad([judge_out(d, torch.tensor(p), s, judge) - p for d, p, _, _ in ok])
        R = []
        for d, p, tr, _ in ev:
            delta = abs(d.median().item() - tr)
            o = judge_out(d, torch.tensor(p), s, judge)
            R.append({"delta": delta, "sigma_eff": s, "sigma_node": sn, "sigma_min": smin, "d_obs": abs(d.median().item() - p), "sigma_out": max(so, 1e-9), "sigma_node_r": max(snr, 1e-9),
                      "miss": float(abs(o - tr) > delta / 2), "zeta": abs(p - tr) / delta})
        out.append((judge, R))
    return out


def cases():
    torch.set_num_threads(1)
    W = {j: {k: curve(j, k) for k in ("C", "E")} for j in ("软", "硬")}
    print("案例                   裁判  配置数   三个比值（中位）                    实际 / 预测漏检 / 失效比例   条件")
    for judge, R in density_rows():
        report("国家密度（共享信念）", judge, W[judge], R, f"区域内占比 ζ 中位 {st.median(r['zeta'] for r in R):.2f}")
    cfg = [c for c in __import__("transfer").configs("alg")][:40]
    R = []
    for c in cfg:
        log = algebra.run(c["seed"], c["ab"], c["scale"], c["delta"], "硬·退回预测")
        R.append({**c, **measure_full(log, 1000, 5000, c["delta"])})
    report("硬裁判·全剔除用预测值", "硬", W["硬"], [r for r in R if r["sigma_eff"] > 0], "（对照：同 40 个配置的现行版本见下一行）")
    R2 = []
    for c in cfg:
        log = algebra.run(c["seed"], c["ab"], c["scale"], c["delta"], "硬")
        R2.append({**c, **measure_full(log, 1000, 5000, c["delta"])})
    report("  对照：退回中位数", "硬", W["硬"], [r for r in R2 if r["sigma_eff"] > 0])
    cfg = [c for c in __import__("transfer").configs("geo")][:40]
    for guard, name in ((False, "自己当答案（无守门）"), (True, "  对照：一致才学")):
        R = []
        for c in cfg:
            log = geometry.run(c["seed"], c["proc"], c["scale"], c["delta"], judge="软", guard=guard)
            R.append({**c, **measure_full(log, 1000, 6000, c["delta"])})
        report(name, "软", W["软"], [r for r in R if r["sigma_eff"] > 0])


if __name__ == "__main__":
    {"stress": stress, "stress2": stress2, "stress3": stress3, "stress4": lambda: stress3(True)}.get(sys.argv[1], cases)()
