"""冻结前的诊断（2026-11-01，库伯 + 外部审稿意见）：后验方差偏小？诚实权重 ≠ 诚实概率？两种 z 分母版本的差别？

数据（开发数据、真值已知，不是验证）：17 套真实数据（invert.sets，R0，每个 δ > 0），投毒身份 = always ∪ {卧底}；
  玩具：mincond.gen（新标签 cal，60 个配置，真诚实集已知）。
z 分母两版（derive5 内的诚实度 z = 平均偏移 / 分母）：
  A = derive5 现状：se = std(X) / √(全部题数)，X = r − (p − b)，全部题（干净 + 故障）汇总
  B = mincond / realcond 的识别：√(se² + se_b²)，se_b = 1.2533 × MAD(p − 节点中位数) / √(校准题数)
诊断：
  D1 后验方差：已知诚实集（全知）和抽样诚实子集（A / B）下，真值的标准化残差 t = (0 − m) / sd：Var(t) 应 ≈ 1；|t| > 1.96 应 ≈ 5%。
     抽样子集用混合分布的 PIT（真值处的混合 CDF）：90% 中心区间覆盖应 ≈ 90%。
  D2 权重校准：w 分档 → 档内真诚实节点的比例（若 w 是概率，应 ≈ 档中值）。
  D3 逐题被骗概率的校准：按预测概率分 5 档 → 实际被骗比例；Brier 分数。

结果（开发数据，非验证）：
  D1 后验确实过窄：全知诚实集下 |t| > 1.96 真实 25%、玩具 32%（应 5%）；90% 区间覆盖 62–79%。
     原因拆解（freezediag_varcause.py）：玩具 = b、σ_c 的估计误差（换真值后 6%）；真实 = 还没定位——b、σ_c 换真值仍 21%，
     诚实节点两两相关平均只有 0.04（排除"相关节点重复计数"这一假设），把节点方差放大 n_h 倍才到 9%。
  D2 w 不是概率：w ≥ 0.2 的各档真诚实比例基本是平的（真实 0.77–0.97、玩具 0.40–0.57），只有 w < 0.2 这一档明显低；w 的行为像一个门限检验。
  D3 逐题被骗概率的校准尚可：预测 < 0.2 → 实际 0.02–0.03，预测 ≥ 0.8 → 实际 0.81–0.92，中间档有噪声；Brier 0.05–0.06。
  分母 A vs B：B 在 w < 0.2 档真诚实比例更低（真实 0.22 → 0.10、玩具 0.17 → 0.03）、真实数据 Brier 0.060 → 0.052，玩具打平。

用法：~/gut-external/torch215-venv/bin/python freezediag.py
"""
import math
import statistics as st

import torch

from analytic import Phi, sigma_c
from hetero import judge_out


def parts(rows, judge, s, zden="A", honest=None, M=64):
    """derive5 的内部量：返回 (各节点权重 w, 每道故障题 [(裁判输出 o, 混合均值 m(M,), 混合 sd(M,), 门槛 t)])。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    jd = "软" if judge == "软" else "硬"
    b, sc = sigma_c(clean)
    n = len(rows[0][2])
    X = torch.stack([r - (p - b) for _, p, r, _ in rows])
    se = X.std(0) / math.sqrt(len(rows))
    if zden == "B":
        e = [p - r.median().item() for _, p, r, _ in clean]
        se_b = 1.2533 * 1.4826 * st.median(abs(x - b) for x in e) / math.sqrt(len(e))
        se = torch.sqrt(se ** 2 + se_b ** 2)
    w = torch.exp(-0.5 * (X.mean(0) / se.clamp(min=1e-12)) ** 2)
    if honest is None:
        g = torch.Generator().manual_seed(0)
        S = (torch.rand(M, n, generator=g) < w).float()
        S[S.sum(1) == 0, int(torch.argmax(w))] = 1.0
    else:
        S = honest.float().unsqueeze(0)
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    prec = 1 / (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread) ** 2
    out = []
    for _, p, r, dk in bad:
        P = S @ prec + 1 / sc ** 2
        out.append((judge_out(r, torch.tensor(p), s, jd), (S @ (prec * r) + (p - b) / sc ** 2) / P, 1 / torch.sqrt(P), dk / 2))
    return w, out


def real_cases():
    import os
    cwd = os.getcwd()
    from invert import sets
    S = sets()
    os.chdir(cwd)
    for name, data, nodes, always, sleeper, deltas in S:
        clean = [(f, p, r, x) for f, p, r, x, _ in data[("R0", 0.0)]]
        honest = torch.tensor([n not in always and n != sleeper for n in nodes])
        for d in deltas[1:]:
            yield name, clean + [(f, p, r, x) for f, p, r, x, _ in data[("R0", d)]], honest


def toy_cases():
    from mincond import gen
    for k in range(60):
        cal, ev, nc, n, _ = gen(k, tag="cal")
        yield f"toy{k}", cal + [x for x in ev if x[0]], torch.arange(n) < nc


def main():
    for label, cases in (("真实 17 套", list(real_cases())), ("玩具 60", list(toy_cases()))):
        print(f"\n===== {label} =====")
        acc = {}
        for name, rows, honest in cases:
            s = max(sigma_c([x for x in rows if not x[0]])[1], 1e-9)
            for tag, zd, hon in (("全知诚实集", "A", honest), ("抽样 A", "A", None), ("抽样 B", "B", None)):
                w, out = parts(rows, "软", s, zd, hon)
                a = acc.setdefault(tag, {"t": [], "pit": [], "w": [], "q": []})
                for o, m, sd, t in out:
                    if hon is not None:
                        a["t"].append((0 - m[0] / sd[0]).item())
                    a["pit"].append(st.mean(Phi(x) for x in ((0 - m) / sd).tolist()))
                    q = st.mean(1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist()))
                    a["q"].append((q, abs(o) > t))
                if hon is None:
                    a["w"] += list(zip(w.tolist(), honest.tolist()))
        for tag, a in acc.items():
            line = f"{tag:8s}："
            if a["t"]:
                tt = a["t"]
                line += f"D1 Var(t) {st.pvariance(tt):.2f}（≈ 1）、|t| > 1.96 {st.mean(abs(x) > 1.96 for x in tt):.0%}（≈ 5%）；"
            pit = a["pit"]
            line += f"90% 区间覆盖 {st.mean(0.05 <= x <= 0.95 for x in pit):.0%}；"
            qs = a["q"]
            line += f"D3 Brier {st.mean((q - y) ** 2 for q, y in qs):.3f}"
            print(line)
            bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
            print("     D3 校准（预测档 → 实际被骗）：" + "  ".join(f"[{lo:.1f},{hi:.1f}) {st.mean(y for q, y in qs if lo <= q < hi):.2f}（{sum(lo <= q < hi for q, _ in qs)}）"
                                              for lo, hi in bins if any(lo <= q < hi for q, _ in qs)))
            if a["w"]:
                print("     D2 权重档 → 真诚实比例：" + "  ".join(f"[{lo:.1f},{hi:.1f}) {st.mean(h for w_, h in a['w'] if lo <= w_ < hi):.2f}（{sum(lo <= w_ < hi for w_, _ in a['w'])}）"
                                                  for lo, hi in bins if any(lo <= w_ < hi for w_, _ in a["w"])))


if __name__ == "__main__":
    main()
