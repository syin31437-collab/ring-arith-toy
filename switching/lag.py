"""检测为什么比公式慢约 1 轮（2026-11-01）：invert.py 里 ≥ 2 轮才抓到的 46 个投毒节点，推出 μ / 实际 中位 0.57。
拆件：公式（μ 恒定、高斯噪声）vs 回放（用该节点真实每题传出的毒、高斯噪声）vs 实际检测。
怀疑：公式假设毒每轮一样大，真实毒每题不同（δ_k = δ·e^{0.5 z}），单轮截到 ±3σ 后，大毒被截、小毒不够，平均爬得更慢。
诊断（数据看过；不预注册门槛，只看哪一个对得上实际）。

用法：~/gut-external/torch215-venv/bin/python lag.py
"""
import math
import statistics as st

import torch

from analytic import sigma_c
from invert import loo, sets, delay_dist
from purge import C, SD_S, W, baseline


def replay(shifts, start, n=2000, horizon=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = SD_S * torch.randn(n, generator=g) if start == "stationary" else torch.zeros(n)
    hit = torch.full((n,), horizon + 1)
    for m, mu in enumerate(shifts[:horizon + 1]):
        S = (1 - W) * S + W * (mu + torch.randn(n, generator=g)).clamp(-3, 3)
        new = (S.abs() > C * SD_S) & (hit > horizon)
        hit[new] = m
    return hit.float().median().item()


def replay_emp(shifts, noise, start, skip, n=2000, horizon=60, seed=0):
    """回放 2：噪声不用高斯，而是从这个节点在干净那轮的真实 z 里有放回抽（保留厚尾）；秤坏那几轮按真实情况跳过。"""
    g = torch.Generator().manual_seed(seed)
    nz = torch.tensor(noise)
    S = SD_S * torch.randn(n, generator=g) if start == "stationary" else torch.zeros(n)
    hit = torch.full((n,), horizon + 1)
    for m, (mu, sk) in enumerate(list(zip(shifts, skip))[:horizon + 1]):
        if sk:
            continue
        e = nz[torch.randint(0, len(nz), (n,), generator=g)]
        S = (1 - W) * S + W * (mu + e).clamp(-3, 3)
        new = (S.abs() > C * SD_S) & (hit > horizon)
        hit[new] = m
    return hit.float().median().item()


def replay_boot(zs, start, n=2000, horizon=60, seed=0):
    """回放 3：直接用这个节点在投毒那段的真实 z（已经是 毒 + 噪声 + 秤误差），有放回重抽、打乱顺序。"""
    g = torch.Generator().manual_seed(seed)
    z = torch.tensor(zs)
    S = SD_S * torch.randn(n, generator=g) if start == "stationary" else torch.zeros(n)
    hit = torch.full((n,), horizon + 1)
    for m in range(horizon + 1):
        S = (1 - W) * S + W * z[torch.randint(0, len(z), (n,), generator=g)].clamp(-3, 3)
        new = (S.abs() > C * SD_S) & (hit > horizon)
        hit[new] = m
    return hit.float().median().item()


def theory_sd(mu, sd, start, n=4000, horizon=60, seed=0):
    """公式的同一个递推，只把每轮噪声的标准差从 1 换成 sd。"""
    g = torch.Generator().manual_seed(seed)
    S = SD_S * torch.randn(n, generator=g) if start == "stationary" else torch.zeros(n)
    hit = torch.full((n,), horizon + 1)
    for m in range(horizon + 1):
        S = (1 - W) * S + W * (mu + sd * torch.randn(n, generator=g)).clamp(-3, 3)
        new = (S.abs() > C * SD_S) & (hit > horizon)
        hit[new] = m
    return hit.float().median().item()


def main():
    print("数据 / 节点 / δ               实际   公式（μ 恒定）   回放（真实每题毒、高斯噪声）   每题毒的变异系数")
    rec = []
    for name, data, nodes, always, sleeper, deltas in sets():
        clean5 = data[("R0", 0.0)]
        clean = [(f, p, r, x) for f, p, r, x, _ in clean5]
        b, sc = sigma_c(clean)
        sig = loo(torch.stack([r for _, _, r, _ in clean]))
        u = baseline(clean, b)
        cl_by = {k: r for _, _, r, _, k in clean5}
        for d in deltas[1:]:
            if d < 0.1:
                continue
            rows = data[("R0", d)]
            S = torch.zeros(len(nodes))
            first = {}
            for _, p, r, _, k in rows:
                z = (r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
                if z.abs().min() > 3:
                    continue
                S = (1 - W) * S + W * z.clamp(-3, 3)
                for j in torch.nonzero(S.abs() > C * SD_S).flatten().tolist():
                    first.setdefault(nodes[j], k)
            k0 = rows[0][4]
            for n in sorted(always) + ([sleeper] if sleeper else []):
                if n not in first or (n == sleeper and first[n] < 30):
                    continue
                j = nodes.index(n)
                start_k = k0 if n in always else 30
                obs = first[n] - start_k
                if obs < 2 or name.startswith("bigtest:planet"):
                    continue
                tot = math.sqrt(sig[j].item() ** 2 + sc ** 2)
                sh = [((r[j] - cl_by[k][j]) / tot).item() for _, _, r, _, k in rows if k >= start_k and k in cl_by]
                mu_eff = abs(st.mean(sh))
                sign = 1 if st.mean(sh) > 0 else -1
                start = "zero" if n in always else "stationary"
                th = delay_dist(mu_eff, start).median().item()
                rp = replay([sign * x for x in sh], start)
                zc = [((cl_by[k][j] - u[j] - (p - b)) / tot).item() * sign for _, p, _, _, k in clean5]
                skip = [((r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)).abs().min().item() > 3 for _, p, r, _, k in rows if k >= start_k and k in cl_by]
                re2 = replay_emp([sign * x for x in sh], zc, start, skip)
                zr = [sign * ((r[j] - u[j] - (p - b)) / tot).item() for _, p, r, _, k in rows if k >= start_k
                      and ((r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)).abs().min() <= 3]
                re3 = replay_boot(zr, start)
                ztail = st.mean(abs(x) > 3 for x in zr)
                nskip = sum(skip[:obs + 1]) if obs < len(skip) else sum(skip)
                tsd = theory_sd(st.mean(zr), st.pstdev(zr), start) + nskip
                cv = st.pstdev(sh) / mu_eff
                rec.append((obs, th, rp, re2, re3, st.mean(zr), st.pstdev(zr), ztail, tsd))
                print(f"{name.split(':')[-1][:10]:10s} {n:13s} {d:4.0%}   {obs:4d}      {th:5.1f}              {rp:5.1f}                      {cv:5.2f}     回放2 {re2:5.1f}  回放3 {re3:5.1f}  真实 z 均值 {st.mean(zr):5.2f} 标准差 {st.pstdev(zr):5.2f}")
    o = [x[0] for x in rec]
    for i, nm in ((1, "公式（μ 恒定、高斯）"), (2, "回放：真实每题毒 + 高斯"), (3, "回放2：+ 干净轮噪声 + 秤坏跳过"), (4, "回放3：真实 z 打乱顺序"), (8, "公式 + 真实标准差 + 跳过的轮")):
        print(f"实际 − {nm}：平均 {st.mean(a - x[i] for a, x in zip(o, rec)):+.2f} 轮，|·| {st.mean(abs(a - x[i]) for a, x in zip(o, rec)):.2f}")
    print(f"真实 z 的标准差中位 {st.median(x[6] for x in rec):.2f}（公式假设 1）；|z| > 3 的比例中位 {st.median(x[7] for x in rec):.0%}（高斯 0.3%）")


if __name__ == "__main__":
    main()
