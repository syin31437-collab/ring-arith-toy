"""最小成立条件（2026-11-01，库伯）：给定一个可校准的外部信息源和多个可靠性未知的节点，
不用任务真值训练，能否预测系统的错误行为、识别异常贡献者？把候选条件一个个拿掉，看哪条拿掉就失效。

玩具（数值版，和 failmap.stress_toy 同分布）：7 个节点，前 nc 个诚实；约束 p = 真值 + σ_c 噪声；故障题上投毒节点偏 τ_k·δ_k。
  前 ncal 题是已知干净的校准期，之后的题 30% 是故障题（预测者不知道哪些是）。
  评估集 = 校准期之后的全部题（干净 + 故障混在一起），门槛 = 每题 δ_k / 2。
  预测：推导版第五版（analytic.derive5），只用校准期估秤的 b、σ_c 和节点噪声；裁判宽度也只用校准期估（不偷看）。
  识别：跨题汇总的诚实权重 w_i = exp(−½ z_i²)（z = 相对秤中心的平均偏移 / 标准误），w < 0.5 判为投毒；平衡准确率。
条件与拿掉方式（每档 60 个配置，新种子 mc）：
  A 外部源独立       ζ ∈ {0, 0.5, 1}（约束在故障题上跟着毒偏 ζ）
  B 至少一个诚实节点  α ∈ {0, 0.5, 1}（诚实节点在故障题上也被带偏 α·δ）
  C 已知干净的校准期  ncal ∈ {100, 20, 5, 0}（0 = 不知道哪段干净，只能拿全部题做估计）
  D 节点身份跨题稳定  每题随机打乱节点顺序（是 / 否）
  E 外部源够准       约束噪声 × {1, 10, 100}
  F 题目够多         校准期后的题数 ∈ {300, 100, 30, 10}
  G 错误抱团同向     每题毒的方向随机（是 / 否）

预注册（11-01，运行前）：
  0. 基线（全部条件满足）：预测 MAE ≤ 0.05，识别平衡准确率 ≥ 0.90
  判定：拿掉到最重一档时，MAE > 0.10 或识别准确率 < 0.70 → 该条件必要
  预测：A 必要（ζ = 1 乐观）；B 必要；C 必要但只要很少（ncal = 20 时仍 MAE ≤ 0.07、识别 ≥ 0.85，ncal = 0 时失效）；
       D 对识别必要（≈ 0.5），对预测不必要（MAE ≤ 0.10）；E 对识别必要（×100 时 < 0.70），预测 MAE ≤ 0.15；
       F 题数 30 时识别 ≥ 0.80、10 时 < 0.70；G 对现方法的识别必要（平均偏移抵消，< 0.70）

用法：~/gut-external/torch215-venv/bin/python mincond.py
"""
import math
import random
import statistics as st

import torch

from analytic import derive5, sigma_c
from hetero import judge_out, lu


def gen(k, nc_rng=True, ncal=100, n_eval=300, zeta=0.0, couple=0.0, cmult=1.0, permute=False, randsign=False, n=7):
    rng = random.Random(f"mc-{k}")
    g = torch.Generator().manual_seed(150000 + k)
    sc, scl = lu(rng, 0.01, 1) * cmult, lu(rng, 0.05, 2)
    spo, delta, tlo = scl * lu(rng, 1, 30), lu(rng, 0.03, 30), rng.uniform(0, 1)
    nc = rng.choice([1, 2, 3])
    sig = torch.tensor([scl] * nc + [spo] * (n - nc)) * (0.5 + torch.rand(n, generator=g))
    tau = torch.tensor([0.0] * nc + [rng.uniform(tlo, 1) for _ in range(n - nc)])
    cal, ev = [], []
    for t in range(ncal + n_eval):
        fault = t >= ncal and rng.random() < 0.3
        dk = delta * math.exp(0.5 * rng.gauss(0, 1))
        sgn = (1 if rng.random() < 0.5 else -1) if randsign else 1
        shift = (tau * dk + couple * dk * (tau == 0).float()) * sgn if fault else torch.zeros(n)
        r = sig * torch.randn(n, generator=g) + shift
        p = sc * torch.randn(1, generator=g).item() + (zeta * tau[nc:].mean().item() * dk * sgn if fault else 0.0)
        if permute:
            r = r[torch.randperm(n, generator=g)]
        (cal if t < ncal else ev).append((fault, p, r, dk))
    return cal, ev, nc, n


def evaluate(cal, ev, nc, n, judge="软"):
    est = cal if cal else ev                                   # 没有已知干净的校准期 → 只能拿全部题估（被毒污染）
    s = max(sigma_c([(False, p, r, x) for _, p, r, x in est])[1], 1e-9)
    rows = [(False, p, r, x) for _, p, r, x in est] + [(True, p, r, x) for _, p, r, x in ev]
    act = st.mean(abs(judge_out(r, torch.tensor(p), s, judge)) > x / 2 for _, p, r, x in ev)
    pred = derive5(rows, judge, s)
    b, _ = sigma_c([(False, p, r, x) for _, p, r, x in est])
    X = torch.stack([r - (p - b) for _, p, r, _ in est + ev])
    w = torch.exp(-0.5 * (X.mean(0) / (X.std(0) / math.sqrt(len(X))).clamp(min=1e-12)) ** 2)
    flag = (w < 0.5).tolist()
    truth = [i >= nc for i in range(n)]
    tpr = st.mean(f for f, t in zip(flag, truth) if t)
    tnr = st.mean(not f for f, t in zip(flag, truth) if not t)
    return act - pred, (tpr + tnr) / 2


SWEEPS = [("基线", {}),
          ("A ζ=0.5", {"zeta": 0.5}), ("A ζ=1", {"zeta": 1.0}),
          ("B α=0.5", {"couple": 0.5}), ("B α=1", {"couple": 1.0}),
          ("C 校准 20 题", {"ncal": 20}), ("C 校准 5 题", {"ncal": 5}), ("C 不知道哪段干净", {"ncal": 0}),
          ("D 身份每题打乱", {"permute": True}),
          ("E 约束 ×10", {"cmult": 10.0}), ("E 约束 ×100", {"cmult": 100.0}),
          ("F 题数 100", {"n_eval": 100}), ("F 题数 30", {"n_eval": 30}), ("F 题数 10", {"n_eval": 10}),
          ("G 毒的方向每题随机", {"randsign": True})]


def main():
    torch.manual_seed(26)
    print("条件                   预测 MAE / 平均误差      识别平衡准确率")
    for tag, kw in SWEEPS:
        E, A = [], []
        for k in range(60):
            cal, ev, nc, n = gen(k, **kw)
            e, a = evaluate(cal, ev, nc, n)
            E.append(e)
            A.append(a)
        mae = st.mean(abs(x) for x in E)
        flag = "必要" if (mae > 0.10 or st.mean(A) < 0.70) else ""
        print(f"{tag:18s}   {mae:.3f} / {st.mean(E):+.3f}          {st.mean(A):.2f}     {flag}")


if __name__ == "__main__":
    main()
