"""扩展玩具：节点噪声异质 + 被投毒节点只传出部分毒量。检验哪组特征能零重拟合预测真实投毒实验（llm/poison.py）。

投毒实验暴露了原玩具的两条前提：节点噪声同质、故障节点把 δ 全部传出。这里把两条都放开，题目照投毒实验搭：
  每题真值 x = 0，约束预测 p = x + N(0, σ_c)；7 个节点，前 n_clean ∈ {1,2,3} 个干净（噪声 σ_clean × U(0.5,1.5)），
  其余被投毒（噪声 σ_pois × U(0.5,1.5)；故障题上偏 τ_k·δ，τ_k ~ U(τ_lo, 1)）；30% 的题是故障题。
  裁判与 poison.py 相同：软 = 以 p 为中心、宽 s = σ_eff 的加权中位数；硬 = 剔除 |r − p| > 2.5 s，全剔除时退回中位数。
特征（全部不用真值也能测；σ_eff 例外，投毒实验里用 SIGEFF=truth 量）：
  A  ρ = δ/σ_eff，κ = δ/σ_node                  （原定义：名义 δ，σ_node = 无故障题上偏离中位数的 RMS）
  B  δ 换成 δ_obs = 故障题上（报告中位数 − p）的平均，减去无故障题上的同一量
  C  B + λ = δ/σ_min，σ_min = 无故障题上各节点 RMS(r − p) 的最小值（全选对人时剩下的噪声）
  D  A + λ（名义 δ）；λ  只用 λ
  F  C 里 λ 换成 λ_g = δ·√k_good / σ_min（k_good = RMS 不超过最准节点 2 倍的节点数：好节点互相平均）
  G  F + q（q = 故障题上 log|报告中位数 − p| 的标准差：δ 的分散程度）
  I  H 里 k_good_r 换成 k_honest（故障题上离 p 更近那团的人数，见 honest_count）
  H  G 的稳健版：κ 用 MAD 量节点噪声，λ_g 里的 σ_min、k_good 都按各节点 MAD 算
  E  C 的稳健版（failmap.py 压力测试后加）：κ 用 MAD 量节点噪声，λ' = δ / σ_out，σ_out = 裁判在无故障题上输出偏离 p 的 MAD
零重拟合：只在玩具上拟合 S 形曲线（特征取 log、二次项），原样预测投毒实验的 14 种配置 × 两种裁判。

回测（backtest）：同质玩具（时间 / 代数 / 信息网络，transfer.py 重新生成、带 σ_min 和 δ_obs）上，加了 λ 会不会变差；
  以及各来源（异质玩具 / 原玩具 / 合并）拟合的曲线，零重拟合预测其余环境。

用法：~/gut-external/torch215-venv/bin/python hetero.py
      ~/gut-external/torch215-venv/bin/python hetero.py backtest
"""
import itertools
import json
import math
import os
import random
import statistics as st
import sys

import torch

from robust import wmedian

HERE = os.path.dirname(os.path.abspath(__file__))
N, ITEMS = 7, 400


def lu(rng, lo, hi):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def judge_out(r, p, s, judge):
    if judge == "软":
        return wmedian(r, torch.exp(-((r - p) ** 2) / (2 * s ** 2)) + 1e-6).item()
    keep = (r - p).abs() <= 2.5 * s
    return (r[keep].median() if keep.any() else r.median()).item()


def feats(m):
    """m 里有 delta, d_obs, sigma_eff, sigma_node, sigma_min → 三组特征（log10）。"""
    d, do = m["delta"], max(m["d_obs"], 1e-3 * m["delta"])
    return {"A": [math.log10(d / m["sigma_eff"]), math.log10(d / m["sigma_node"])],
            "B": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node"])],
            "C": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node"]), math.log10(d / m["sigma_min"])],
            "D": [math.log10(d / m["sigma_eff"]), math.log10(d / m["sigma_node"]), math.log10(d / m["sigma_min"])],
            "λ": [math.log10(d / m["sigma_min"])],
            "E": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node_r"]), math.log10(d / m["sigma_out"])],
            "F": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node"]), math.log10(d * math.sqrt(m["k_good"]) / m["sigma_min"])],
            "G": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node"]), math.log10(d * math.sqrt(m["k_good"]) / m["sigma_min"]), m["q"]],
            "H": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node_r"]), math.log10(d * math.sqrt(m["k_good_r"]) / m["sigma_min_r"]), m["q"]],
            "I": [math.log10(do / m["sigma_eff"]), math.log10(do / m["sigma_node_r"]), math.log10(d * math.sqrt(max(m["k_honest"], 0.5)) / m["sigma_min_r"]), m["q"]]}


def toy(k, judge):
    rng = random.Random(f"het-{k}")
    g = torch.Generator().manual_seed(k)
    sc, scl = lu(rng, 0.01, 1), lu(rng, 0.05, 2)
    spo, delta, tlo, nc = scl * lu(rng, 1, 30), lu(rng, 0.03, 30), rng.uniform(0, 1), rng.choice([1, 2, 3])
    sig = torch.tensor([scl] * nc + [spo] * (N - nc)) * (0.5 + torch.rand(N, generator=g))
    tau = torch.tensor([0.0] * nc + [rng.uniform(tlo, 1) for _ in range(N - nc)])
    rows = []
    for _ in range(ITEMS):
        fault = torch.rand(1, generator=g).item() < 0.3
        p = sc * torch.randn(1, generator=g).item()
        r = sig * torch.randn(N, generator=g) + (tau * delta if fault else 0.0)
        rows.append((fault, p, r))
    return measure(rows, delta, judge, s=sc)


def toy2(k, judge):
    """拟合玩具 v2：在 toy 的基础上让节点数 N 和 δ 的分散度变化（stress2 发现曲线没见过这两个量的变化）。"""
    rng = random.Random(f"het2-{k}")
    g = torch.Generator().manual_seed(20000 + k)
    sc, scl = lu(rng, 0.01, 1), lu(rng, 0.05, 2)
    spo, delta, tlo = scl * lu(rng, 1, 30), lu(rng, 0.03, 30), rng.uniform(0, 1)
    n, dvar = rng.choice([3, 5, 7, 11, 15, 21, 31]), rng.uniform(0, 1.5)
    nc = max(1, round(n * rng.choice([1, 2, 3]) / 7))
    sig = torch.tensor([scl] * nc + [spo] * (n - nc)) * (0.5 + torch.rand(n, generator=g))
    tau = torch.tensor([0.0] * nc + [rng.uniform(tlo, 1) for _ in range(n - nc)])
    rows = []
    for _ in range(ITEMS):
        fault = torch.rand(1, generator=g).item() < 0.3
        dk = delta * math.exp(dvar * torch.randn(1, generator=g).item())
        p = sc * torch.randn(1, generator=g).item()
        rows.append((fault, p, sig * torch.randn(n, generator=g) + (tau * dk if fault else 0.0), dk))
    return measure(rows, delta, judge, s=sc)


def robust_node(dev):
    """稳健版：各节点偏离 p 的 MAD（不用 RMS，厚尾时 RMS 会被离群值撑大）→ 最准节点 σ_min_r、好节点数 k_good_r。"""
    mad = 1.4826 * (dev - dev.median(0).values).abs().median(0).values.clamp(min=1e-12)
    return {"sigma_min_r": mad.min().item(), "k_good_r": int((mad <= 2 * mad.min()).sum())}


def honest_count(devs, smin):
    """诚实团大小（不用真值）：每道故障题把各节点相对 p 的偏移排序、在最大缺口处分成两团，
    最大缺口 ≤ 3·σ_min 时算没分开（= 全部节点）；否则取离 p 更近的那团的人数。返回各故障题的中位数。
    （盲测第三版发现：δ=0 时数"准的节点"会把同样精确、只在故障时才偏的节点也算进来。）"""
    ks = []
    for d in devs:
        v, _ = torch.sort(d)
        gaps = v[1:] - v[:-1]
        k = int(torch.argmax(gaps))
        if gaps[k] <= 3 * smin:
            ks.append(len(d))
        else:
            lo, hi = v[:k + 1], v[k + 1:]
            ks.append(len(lo) if lo.median().abs() <= hi.median().abs() else len(hi))
    return st.median(ks) if ks else len(devs[0]) if devs else 1


def good_count(dev):
    """好节点数：各节点 RMS(r − p) 不超过最准节点 2 倍的个数（不用真值）。"""
    rms = (dev ** 2).mean(0).sqrt()
    return int((rms <= 2 * rms.min()).sum())


def spread_log(xs):
    """δ 的分散程度：故障题上 log10|报告中位数 − p| 的标准差（不用真值）。"""
    return st.pstdev(math.log10(max(x, 1e-12)) for x in xs) if len(xs) > 1 else 0.0


def measure(rows, delta, judge, s):
    """rows = [(是否故障, 约束预测 p, 报告张量 r[, 这道故障题的 δ_k])]，真值 = 0。
    漏检按每道题自己的 δ_k / 2 判（没给 δ_k 时 = δ）；dks 记下各故障题的 δ_k，供逐事件预测用。"""
    rows = [(x[0], x[1], x[2], x[3] if len(x) > 3 else delta) for x in rows]
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    med = lambda xs: st.mean(r.median().item() - p for _, p, r, _ in xs)
    dev = torch.stack([r - p for _, p, r, _ in clean])
    out = [judge_out(r, torch.tensor(p), s, judge) for _, p, r, _ in bad]
    base = st.mean(judge_out(r, torch.tensor(p), s, judge) for _, p, r, _ in clean)
    oc = [judge_out(r, torch.tensor(p), s, judge) - p for _, p, r, _ in clean]
    mad = lambda xs: 1.4826 * st.median(abs(x - st.median(xs)) for x in xs)
    return {"delta": delta, "sigma_eff": s, "d_obs": med(bad) - med(clean),
            "sigma_out": max(mad(oc), 1e-9),                       # 裁判在无故障题上的输出噪声（稳健）
            "sigma_node_r": max(st.median(1.4826 * (r - r.median()).abs().median().item() for _, _, r, _ in clean), 1e-9),
            "sigma_node": math.sqrt(st.mean(((r - r.median()) ** 2).mean().item() for _, _, r, _ in clean)),
            "sigma_min": (dev ** 2).mean(0).sqrt().min().item(),
            "k_good": good_count(dev), "q": spread_log([abs(r.median().item() - p) for _, p, r, _ in bad]),
            **robust_node(dev),
            "k_honest": honest_count([r - p for _, p, r, _ in bad], robust_node(dev)["sigma_min_r"]),
            "dks": [dk for *_, dk in bad],
            "vote": st.mean(abs(r.median().item()) > dk / 2 for _, _, r, dk in bad),
            "miss": st.mean(abs(o) > dk / 2 for o, (*_, dk) in zip(out, bad)), "pass": (st.mean(out) - base) / delta}


def real(judge):
    """投毒实验的 14 种配置：同样的测量，单位 log10 距离。"""
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    os.chdir(os.path.join(HERE, "..", "llm"))
    from cities import gc, load_answers
    from poison import CACHE, DELTAS, JITTERS, POISONED, chosen
    by, ps = chosen()
    mem, _ = load_answers()
    mc = {c: [st.median(v[n][0] for n in v), st.median(v[n][1] for n in v)] for c, v in mem.items()}
    rep = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        j, d, b = key.split("_")
        if node in POISONED and not b.startswith("n"):
            continue
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            if v > 0:
                rep.setdefault((float(j), float(d)), {}).setdefault(i, {})[node] = math.log10(v)
    os.chdir(cwd)
    truth = {i: math.log10(gc(by[a]["lat"], by[a]["lon"], by[b]["lat"], by[b]["lon"])) for i, a, b in ps}
    pred = {i: math.log10(gc(*mc[a], *mc[b])) for i, a, b in ps}
    s = st.pstdev(pred[i] - truth[i] for i in truth)
    nodes = sorted({n for v in rep.values() for x in v.values() for n in x})
    out = []
    for j in JITTERS:
        def rows(d, fault):           # 以真值为 0；只用 7 个节点都交回的题
            return [(fault, pred[i] - truth[i], torch.tensor([v[n] - truth[i] for n in nodes]))
                    for i, v in rep[(j, d)].items() if len(v) == len(nodes)]
        clean = rows(0.0, False)
        for d in DELTAS[1:]:
            m = measure(clean + rows(d, True), math.log10(1 + d), judge, s)
            out.append({**m, "jitter": j, "δ": d})
    return out


def quad(x):
    return [1.0] + x + [p * q for p, q in itertools.combinations_with_replacement(x, 2)]


def fit(rows, kind, target):
    X = torch.tensor([quad(feats(r)[kind]) for r in rows], dtype=torch.float64)
    y = torch.tensor([r[target] for r in rows], dtype=torch.float64).clamp(0, 1)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((torch.sigmoid(X @ w) - y) ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def predict(w, rows, kind):
    return torch.sigmoid(torch.tensor([quad(feats(r)[kind]) for r in rows], dtype=torch.float64) @ w)


def r2(rows, pred, target):
    y = torch.tensor([min(1.0, max(0.0, r[target])) for r in rows], dtype=torch.float64)
    return 1 - ((y - pred) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()


def main():
    torch.set_num_threads(4)
    for judge in ("软", "硬"):
        p2 = os.path.join(HERE, f"het2_{judge}.json")
        if not os.path.exists(p2):
            json.dump([toy2(k, judge) for k in range(1200)], open(p2, "w"))
        path = os.path.join(HERE, f"het_{judge}.json")
        if not os.path.exists(path):
            json.dump([toy(k, judge) for k in range(600)], open(path, "w"))
        T = json.load(open(path))
        R = real(judge)
        half = len(T) // 2
        print(f"\n== {judge}裁判：玩具 {len(T)} 个配置，投毒实验 {len(R)} 个配置 ==")
        print("目标     特征   玩具内交叉 R²   零重拟合→投毒实验 R²   MAE")
        for target in ("miss", "pass"):
            for kind in ("A", "B", "C", "D", "λ", "E"):
                cv = r2(T[half:], predict(fit(T[:half], kind, target), T[half:], kind), target)
                p = predict(fit(T, kind, target), R, kind)
                mae = st.mean(abs(a - min(1, max(0, r[target]))) for a, r in zip(p.tolist(), R))
                print(f"{target:6s}   {kind}     {cv:10.3f}       {r2(R, p, target):14.3f}        {mae:.3f}")
        w = fit(T, "C", "miss")
        print("逐配置（漏检，特征 C）：抖动  δ     δ_obs/δ   实际   预测")
        for r, p in zip(R, predict(w, R, "C").tolist()):
            print(f"                       {r['jitter']:4}  {r['δ']:5.1%}  {r['d_obs'] / r['delta']:6.2f}   {r['miss']:4.0%}   {p:4.0%}")


def backtest():
    import glob
    for judge in ("软", "硬"):
        old = {}
        for env in ("geo", "alg", "net"):
            rows = [r for f in sorted(glob.glob(os.path.join(HERE, f"tr_{env}_{judge}_*.json"))) for r in json.load(open(f))]
            old[env] = [r for r in rows if r["sigma_eff"] > 0 and r["sigma_node"] > 0 and not math.isnan(r["pass"]) and "sigma_min" in r]
        src = {"异质玩具": json.load(open(os.path.join(HERE, f"het_{judge}.json"))), "时间+代数": old["geo"] + old["alg"]}
        src["异质+时间+代数"] = src["异质玩具"] + src["时间+代数"]
        dst = {"信息网络": old["net"], "时间": old["geo"], "代数": old["alg"], "投毒实验": real(judge)}
        print(f"\n== {judge}裁判（R²）：时间 {len(old['geo'])} / 代数 {len(old['alg'])} / 网络 {len(old['net'])} ==")
        for target in ("miss", "pass"):
            print(f"{target}：拟合来源 → " + "  ".join(f"{d:>8s}" for d in dst) + "    （A = 原 ρ+κ，D = ρ+κ+λ，C = δ_obs 版 ρ+κ+λ）")
            for s, rows in src.items():
                for kind in ("A", "D", "C"):
                    w = fit(rows, kind, target)
                    cells = [r2(v, predict(w, v, kind), target) if not (s == "时间+代数" and d in ("时间", "代数")) and not (s == "异质+时间+代数" and d in ("时间", "代数")) else float("nan") for d, v in dst.items()]
                    print(f"  {s:12s} {kind}   " + "  ".join(f"{c:8.2f}" for c in cells))


if __name__ == "__main__":
    backtest() if sys.argv[1:] == ["backtest"] else main()
