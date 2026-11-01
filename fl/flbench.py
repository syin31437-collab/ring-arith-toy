"""外部验证二：联邦学习的鲁棒聚合（2026-11-01，库伯：拿数学推导到别的地方验证，先做这个）。

映射（零重拟合，规律一个参数都不改）：
  节点 = n 个客户端上传的梯度；题 = 每轮随机抽的 K 个梯度坐标；秤 = 服务器用自己 m 张干净图算的梯度（FLTrust 式的干净参照）；
  真值 = 所有客户端"诚实梯度"的平均（没有攻击时聚合器该输出的东西）；裁判 = 我们的软 / 硬2 裁判（按和秤的距离加权的中位数）。
  单位：每个坐标除以服务器从自己的数据估的"单客户端梯度噪声"（服务器逐样本梯度的标准差 / √客户端批量）——不看客户端、不看真值。
  校准期 = 前 T0 轮没有攻击（已知干净）；之后每轮攻击。被骗 = |裁判输出 − 真值| > h。
模型：MNIST 线性 softmax（7850 个参数），客户端 IID 各 600 张图、每轮批量 64；用软裁判的聚合梯度更新（lr 0.5），共 T0 + T1 轮。
攻击（投毒客户端占比 f）：
  ALIE    报"诚实梯度 − z"（每个坐标同向挪 z 个噪声单位，投毒者完全一样 = 抱团）；δ = z，h = z / 2
  noise   报"诚实梯度 + 10 × 标准正态"（各自乱、不抱团）；δ = 1，h = 0.5
  flip    投毒客户端把标签 y 换成 9 − y（数据投毒，梯度偏移由数据决定）；δ = 1，h = 0.5
  flip-ζ  同 flip，但服务器的干净集也是同样翻转过的（秤和故障同源，ζ = 1）
网格：n ∈ {10, 20} × f ∈ {0.2, 0.4, 0.6} × 攻击 {ALIE z = 0.5 / 1 / 2 / 4, noise, flip, flip-ζ} × m ∈ {32, 512} = 84 个配置，各两种裁判。
另外每个 n × m 跑一次无攻击，作下游准确率的基线。

预注册（11-01，写代码时、运行前；判据沿用 16 个领域的大规模测试）：
  1. ζ = 0 的格子（ALIE、noise、flip）：区间覆盖 ≥ 90%；ρ̂ ≥ 1 的格子点预测误差中位 ≤ 0.08
  2. 分档：ALIE、f = 0.6、ρ̂ < 1 的格子，实测被骗 ≥ 0.5 的比例 ≥ 80%（投毒多数 + 秤分不清 → 退回人头）
  3. ζ = 1（flip-ζ）且 f ≥ 0.4 的格子：预测偏乐观，实测 − 点预测 ≥ 0.20 的比例 ≥ 80%
  4. 下游：ζ = 0 的配置里，软裁判预测被骗率 与 最终测试准确率下降（对同 n、m 的无攻击基线）的 Spearman ≥ 0.6

用法：~/gut-external/torch215-venv/bin/python flbench.py
"""
import gzip
import itertools
import json
import math
import os
import random
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "switching"))
from hetero import judge_out, measure  # noqa: E402
from intervals import interval2  # noqa: E402

T0, T1, K, B, LR = 10, 20, 50, 64, 0.5


def idx(fn):
    raw = gzip.open(os.path.join(HERE, "data", fn)).read()
    if raw[2] == 8 and raw[3] == 3:
        n, r, c = (int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(3))
        return torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).view(n, r * c).float() / 255
    n = int.from_bytes(raw[4:8], "big")
    return torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).long()[:n]


XTR, YTR = idx("train-images-idx3-ubyte.gz"), idx("train-labels-idx1-ubyte.gz")
XTE, YTE = idx("t10k-images-idx3-ubyte.gz"), idx("t10k-labels-idx1-ubyte.gz")


def grads(W, X, Y):
    """逐样本梯度 (b, 7850)：线性 softmax 的交叉熵。"""
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    P = torch.softmax(Xb @ W, 1)
    P[torch.arange(len(Y)), Y] -= 1
    return (Xb.unsqueeze(2) * P.unsqueeze(1)).reshape(len(X), -1)


def soft_all(R, p, s):
    """软裁判对所有坐标批量算（和 hetero.judge_out 的软裁判同一定义）：R (n, D)、p (D,)。"""
    w = torch.exp(-((R - p) ** 2) / (2 * s ** 2)) + 1e-6
    v, o = torch.sort(R, 0)
    cw = torch.cumsum(torch.gather(w, 0, o), 0)
    j = (cw >= cw[-1] / 2).float().argmax(0)
    return v[j, torch.arange(R.shape[1])]


def run(n, f, attack, m, seed):
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    perm = torch.randperm(len(XTR), generator=g)
    cl = [perm[600 * i:600 * (i + 1)] for i in range(n)]
    srv = perm[600 * n:600 * n + m]
    nb = int(round(f * n)) if attack != "none" else 0
    bad = set(range(n - nb, n))
    W = torch.zeros(785, 10)
    clean_rows, bad_rows = [], []
    s_run = None
    for t in range(T0 + T1):
        on = t >= T0 and attack != "none"
        ys = YTR[srv]
        if attack == "flip-ζ":
            ys = 9 - ys
        Gs = grads(W, XTR[srv], ys)
        p_full = Gs.mean(0)
        scale = (Gs.std(0) / math.sqrt(B)).clamp(min=1e-8)                 # 单客户端梯度噪声（服务器自己估）
        H, R = [], []
        for i in range(n):
            ib = cl[i][torch.randint(0, 600, (B,), generator=g)]
            hon = grads(W, XTR[ib], YTR[ib]).mean(0)
            H.append(hon)
            if on and i in bad:
                if attack.startswith("ALIE"):
                    R.append(hon - float(attack[4:]) * scale)
                elif attack == "noise":
                    R.append(hon + 10 * scale * torch.randn(785 * 10, generator=g))
                else:
                    R.append(grads(W, XTR[ib], 9 - YTR[ib]).mean(0))
            else:
                R.append(hon)
        H, R = torch.stack(H), torch.stack(R)
        truth = H.mean(0)
        live = torch.nonzero(scale > 1e-6).flatten()
        coords = live[torch.randperm(len(live), generator=g)[:K]]
        dz = float(attack[4:]) if attack.startswith("ALIE") else 1.0
        for j in coords.tolist():
            row = (on, ((p_full[j] - truth[j]) / scale[j]).item(), (R[:, j] - truth[j]) / scale[j], dz)
            (bad_rows if on else clean_rows).append(row)
        if s_run is None and t == T0 - 1 and clean_rows:
            from analytic import sigma_c
            s_run = max(sigma_c(clean_rows)[1], 1e-6)
        Z, pz = R / scale, p_full / scale
        agg = soft_all(Z, pz, s_run if s_run else 1.0) * scale
        W -= LR * agg.view(785, 10)
    acc = ((torch.cat([XTE, torch.ones(len(XTE), 1)], 1) @ W).argmax(1) == YTE).float().mean().item()
    return clean_rows, bad_rows, acc


def main():
    torch.manual_seed(0)
    attacks = ["ALIE0.5", "ALIE1", "ALIE2", "ALIE4", "noise", "flip", "flip-ζ"]
    base = {}
    for n, m in itertools.product((10, 20), (32, 512)):
        _, _, acc = run(n, 0.0, "none", m, 1000 + n + m)
        base[(n, m)] = acc
    print("无攻击基线准确率：", {k: round(v, 3) for k, v in base.items()})
    R = []
    for k, (n, f, attack, m) in enumerate(itertools.product((10, 20), (0.2, 0.4, 0.6), attacks, (32, 512))):
        clean, bad, acc = run(n, f, attack, m, 2000 + k)
        for judge in ("软", "硬2"):
            pt, lo, hi, s0, rh, bw = interval2(clean, bad, judge, B=20)
            act = measure([(False, p, r, x) for _, p, r, x in clean] + bad, bad[0][3], judge, s0)["miss"]
            R.append({"n": n, "f": f, "attack": attack, "m": m, "judge": judge, "act": act, "pt": pt, "lo": lo, "hi": hi, "rho": rh,
                      "acc": acc, "drop": base[(n, m)] - acc})
        print(f"{k + 1}/84 n={n} f={f} {attack:7s} m={m:3d}  软 实测 {R[-2]['act']:.2f} 点 {R[-2]['pt']:.2f} [{R[-2]['lo']:.2f}, {R[-2]['hi']:.2f}] ρ̂ {R[-2]['rho']:.2f}"
              f" | 硬2 实测 {R[-1]['act']:.2f} 点 {R[-1]['pt']:.2f}  准确率 {acc:.3f}", flush=True)
    json.dump(R, open(os.path.join(HERE, "flbench_results.json"), "w"), indent=1, ensure_ascii=False)
    z0 = [r for r in R if r["attack"] != "flip-ζ"]
    cov = st.mean(r["lo"] <= r["act"] <= r["hi"] for r in z0)
    pe = [abs(r["act"] - r["pt"]) for r in z0 if r["rho"] >= 1]
    print("\n预注册检查：")
    print(f"1. ζ = 0：区间覆盖 {cov:.0%}（≥ 90%）；ρ̂ ≥ 1 的 {len(pe)} 格点误差中位 {st.median(pe) if pe else float('nan'):.3f}（≤ 0.08）")
    g2 = [r for r in z0 if r["attack"].startswith("ALIE") and r["f"] == 0.6 and r["rho"] < 1]
    print(f"2. ALIE、f = 0.6、ρ̂ < 1 的 {len(g2)} 格：实测 ≥ 0.5 的比例 {st.mean(r['act'] >= 0.5 for r in g2) if g2 else float('nan'):.0%}（≥ 80%）")
    g3 = [r for r in R if r["attack"] == "flip-ζ" and r["f"] >= 0.4]
    print(f"3. ζ = 1、f ≥ 0.4 的 {len(g3)} 格：实测 − 点预测 ≥ 0.20 的比例 {st.mean(r['act'] - r['pt'] >= 0.2 for r in g3):.0%}（≥ 80%）")
    soft = [r for r in z0 if r["judge"] == "软"]
    import torch as T
    rk = lambda v: T.tensor(v).argsort().argsort().float()
    sp = T.corrcoef(T.stack([rk([r["pt"] for r in soft]), rk([r["drop"] for r in soft])]))[0, 1].item()
    print(f"4. 软裁判预测被骗率 vs 准确率下降 Spearman {sp:.2f}（≥ 0.6）")


if __name__ == "__main__":
    main()
