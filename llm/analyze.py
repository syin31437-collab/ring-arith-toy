"""真实模型：玩具曲线能不能零重拟合预测"共同错误被漏检"的概率。

全部在 log10 空间（这类数量的误差是乘法式的）。对每个国家：
  密度报告 d_i（各节点），约束预测 pred = 人口报告中位数 − 面积报告中位数（密度 = 人口 / 面积）
  裁判（不看真值）：
    软  按离 pred 的远近加权取加权中位数，尺度 s = 所有国家上 (pred − 密度中位数) 的稳健标准差
    硬  离 pred 超过 2.5s 的报告剔除，剩下的取中位数，全被剔除退回中位数
  共同错误 δ = 密度报告中位数 − 真值；|δ| > 0.05（约 12%）算一次"事件"
  漏检 = 事件上 |输出 − 真值| > |δ|/2
把每个事件的误差拆成两部分（库伯指出：模型可能给出互相一致的错人口、错面积、错密度）：
  区域内 δ_in  = pred − 真值          和人口、面积一致的错误，约束原理上看不见
  区域外 δ_off = 密度中位数 − pred    违反约束的部分，才可能被发现
理论预测（零重拟合）：
  |δ_in| > |δ|/2  → 即使完美拉回约束也还错一半以上 → 必然漏检（预测概率 1）
  否则            → 用玩具的漏检率曲线，ρ = |δ_off| / σ_eff，κ = |δ_off| / σ_node
  σ_eff = 非事件国家上 (pred − 密度中位数) 的标准差；σ_node = 非事件国家上 各节点偏离密度中位数的 RMS
对照：原来的错误定义（ρ、κ 都用总误差 |δ|，不拆分）。

用法：~/gut-external/torch215-venv/bin/python analyze.py
"""
import glob
import itertools
import json
import math
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "switching"))
os.chdir(os.path.join(HERE, "..", "switching"))
from collect import NODES, truth  # noqa: E402
from decouple import fit_sigmoid, pred_sigmoid  # noqa: E402
from robust import wmedian  # noqa: E402
from transfer import as_feats, load  # noqa: E402

EVENT = float(os.environ.get("EVENT", 0.05))


def pk(T, A):
    """模型 PK：各自准不准、自己内部一不一致、彼此错误有多相关。全部按 log10 误差。"""
    nodes = sorted({n for a in A.values() for q in a.values() for n in q})
    print("== 模型 PK（log10 误差；10% 以内 ≈ |误差| < 0.041） ==")
    print("节点            人口:中位误差/10%内   面积:中位误差/10%内   密度:中位误差/10%内   自洽:密度=人口÷面积(2%内)")
    errs = {}
    for n in nodes:
        cells = []
        for q in ("pop", "area", "dens"):
            e = [A[i][q][n] - (math.log10(T[i][q]) if q != "pop" else math.log10(T[i]["pop"]))
                 for i in A if i in T and n in A[i].get(q, {})]
            if q == "dens":
                errs[n] = {i: A[i][q][n] - math.log10(T[i]["dens"]) for i in A if i in T and n in A[i].get(q, {})}
            cells.append(f"{st.median(abs(x) for x in e):.3f}/{sum(abs(x) < 0.041 for x in e) / len(e):4.0%}")
        cons = [abs(A[i]["dens"][n] - (A[i]["pop"][n] + 6 - A[i]["area"][n])) for i in A
                if all(n in A[i].get(q, {}) for q in ("pop", "area", "dens"))]
        print(f"{n:14s} " + "".join(f"{c:>21s}" for c in cells) + f"   {sum(c < 0.0086 for c in cons) / len(cons):18.0%}")
    pairs, same = [], []
    for a, b in itertools.combinations(nodes, 2):
        common = [i for i in errs[a] if i in errs[b]]
        xa, xb = [errs[a][i] for i in common], [errs[b][i] for i in common]
        ma, mb = st.mean(xa), st.mean(xb)
        pairs.append(sum((p - ma) * (q - mb) for p, q in zip(xa, xb)) /
                     math.sqrt(sum((p - ma) ** 2 for p in xa) * sum((q - mb) ** 2 for q in xb)))
        both = [(p, q) for p, q in zip(xa, xb) if abs(p) > 0.041 and abs(q) > 0.041]
        if both:
            same.append(sum((p > 0) == (q > 0) and abs(p - q) < 0.041 for p, q in both) / len(both))
    print(f"\n密度误差两两相关：平均 {st.mean(pairs):.2f}（范围 {min(pairs):.2f}–{max(pairs):.2f}）；"
          f"两个模型都错（>10%）时，错成几乎同一个数（相差 <10%）的比例平均 {st.mean(same):.0%}\n")


def answers():
    out = {}
    for f in glob.glob(os.path.join(HERE, "cache", "*.json")):
        node, qkey, _ = os.path.basename(f)[:-5].rsplit("_", 2)
        for iso, v in json.load(open(f))["answers"].items():
            if v > 0:
                out.setdefault(iso, {}).setdefault(qkey, {})[node] = math.log10(v)
    return out


def main():
    T, A = truth(), answers()
    pk(T, A)
    rows = []
    for iso, t in T.items():
        a = A.get(iso, {})
        if min(len(a.get(k, {})) for k in ("pop", "area", "dens")) < 5:
            continue
        d = torch.tensor(list(a["dens"].values()))
        pred = st.median(a["pop"].values()) + 6 - st.median(a["area"].values())   # 人口单位是百万
        rows.append({"iso": iso, "d": d, "pred": pred, "truth": math.log10(t["dens"]), "med": d.median().item()})
    evaluate(rows, "国家")


def evaluate(rows, label):
    """rows: 每项含 d（各节点报告，log10 张量）、pred（约束预测）、truth、med。打印误差拆分和零重拟合检验。"""
    if not rows:
        print(f"没有任何{label}凑齐 ≥5 个节点的回答，无法检验")
        return
    resid = [r["pred"] - r["med"] for r in rows]
    mad = st.median(abs(x - st.median(resid)) for x in resid) * 1.4826
    print(f"{len(rows)} 个{label}有 ≥5 个节点的完整回答；约束尺度 s（稳健）= {mad:.3f}")

    for judge in ("软", "硬"):
        for r in rows:
            d, pd = r["d"], torch.tensor(r["pred"])
            if judge == "软":
                r["out"] = wmedian(d, torch.exp(-((d - pd) ** 2) / (2 * mad ** 2)) + 1e-6).item()
            else:
                keep = (d - pd).abs() <= 2.5 * mad
                r["out"] = (d[keep].median() if keep.any() else d.median()).item()
        ev = [r for r in rows if abs(r["med"] - r["truth"]) > EVENT]
        ok = [r for r in rows if abs(r["med"] - r["truth"]) <= EVENT]
        if not ev:
            print(f"\n[{judge}裁判] 没有任何事件（共识偏离真值都不超过门槛），无法检验")
            continue
        sigma_eff = st.pstdev(r["pred"] - r["med"] for r in ok)
        sigma_node = math.sqrt(st.mean(((r["d"] - r["med"]) ** 2).mean().item() for r in ok))
        for r in ev:
            r["delta"] = abs(r["med"] - r["truth"])
            r["d_in"] = abs(r["pred"] - r["truth"])
            r["d_off"] = max(abs(r["med"] - r["pred"]), 1e-4)
            r["miss"] = float(abs(r["out"] - r["truth"]) > r["delta"] / 2)
        w = fit_sigmoid(as_feats(load("geo", judge) + load("alg", judge), "miss"), "ρ+κ")
        y = torch.tensor([r["miss"] for r in ev], dtype=torch.float64)
        inside = torch.tensor([r["d_in"] > r["delta"] / 2 for r in ev])
        off = [{"delta": r["d_off"], "sigma_eff": sigma_eff, "sigma_node": sigma_node, "miss": r["miss"]} for r in ev]
        p = torch.where(inside, torch.ones(len(ev), dtype=torch.float64), pred_sigmoid(w, as_feats(off, "miss"), "ρ+κ"))
        naive = [{"delta": r["delta"], "sigma_eff": sigma_eff, "sigma_node": sigma_node, "miss": r["miss"]} for r in ev]
        p_naive = pred_sigmoid(w, as_feats(naive, "miss"), "ρ+κ")
        print(f"\n[{judge}裁判] 误差拆分：区域内部分超过总误差一半的事件 {int(inside.sum())}/{len(ev)} = {inside.double().mean():.1%}；"
              f"这些事件实际漏检率 {y[inside].mean() if inside.any() else float('nan'):.1%}；"
              f"区域内占总误差比例的中位数 {st.median(min(1.0, r['d_in'] / r['delta']) for r in ev):.2f}")
        base = y.mean()
        brier, brier0 = ((p - y) ** 2).mean().item(), ((base - y) ** 2).mean().item()
        pos, neg = p[y == 1], p[y == 0]
        auc = ((pos.unsqueeze(1) > neg.unsqueeze(0)).double().mean() + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).double().mean()).item() \
            if len(pos) and len(neg) else float("nan")
        print(f"\n== {judge}裁判 ==  事件 {len(ev)} 个（非事件 {len(ok)}）；σ_eff = {sigma_eff:.3f}，σ_node = {sigma_node:.3f}")
        print(f"实际漏检率 {base:.1%}；投票（中位数）在事件上的漏检率 = 100%（按定义）")
        print(f"零重拟合预测的平均漏检概率 {p.mean():.1%}")
        print(f"Brier 分数：拆分后的理论预测 {brier:.3f}  vs  只猜平均值 {brier0:.3f}  vs  不拆分的旧定义 {((p_naive - y) ** 2).mean().item():.3f}（越低越好）；AUC {auc:.3f}")
        rest = ~inside
        if rest.any() and (y[rest] == 1).any() and (y[rest] == 0).any():
            pr, yr = p[rest], y[rest]
            auc_r = ((pr[yr == 1].unsqueeze(1) > pr[yr == 0].unsqueeze(0)).double().mean()).item()
            print(f"只看区域外部分起主导的 {int(rest.sum())} 个事件：实际漏检 {yr.mean():.1%}，预测 {pr.mean():.1%}，"
                  f"Brier {((pr - yr) ** 2).mean().item():.3f} vs 只猜平均 {((yr.mean() - yr) ** 2).mean().item():.3f}，AUC {auc_r:.3f}")
        print("按预测概率分组：预测漏检概率 → 实际漏检率（事件数）")
        qs = torch.quantile(p, torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)).tolist()
        edges = [-1.0] + qs + [2.0]
        for lo, hi in zip(edges, edges[1:]):
            m = (p > lo) & (p <= hi)
            if m.any():
                print(f"   {p[m].mean():6.1%} → {y[m].mean():6.1%}  ({int(m.sum())})")


if __name__ == "__main__":
    main()
