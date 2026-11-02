"""规律 v6.1 的独立跨领域预注册验证（2026-11-01，库伯 + 外部审稿要求）。冻结：git tag law-v6.1；结果文件记录 LAW_VERSION。冻结只为这次验证，规律随时可被推翻。

域 1 Fashion-MNIST 联邦学习（v6 开发从未用过；映射预检 12/12 通过：FL_LR = 0.1、dense = 0.5、v2 映射；新种子 7000+k）：
  前提成立：n ∈ {10, 20} × f ∈ {0.2, 0.4, 0.6} × {ALIE0.5/1/2/4, noise, flip} × m ∈ {32, 512} × 软 / 硬2 = 144 格；区间按轮整块重采样。
  B 不成立：f = 1.0（没有诚实节点）× {ALIE1/2/4, noise, flip} × n × m = 20 配置。
  S 不成立（ζ = 1）：flip-ζ（秤只在攻击期翻转）× f ∈ {0.4, 0.6} × n × m = 8 配置。
域 2 自然的前提不成立（低成本空气传感器；映射预检第 ④ 条节点独立性没过，所以不算点预测的检验，只测拒答）：
  奥斯陆 KVN：Fidas = 真值、TEOM = 秤、第 38→39、39→40、40→41 周；节点相关 0.97。
  EPA 6 站：节点 = AQY/CNO/MAX/PAR，秤 = RAM（沙盒里和节点共识相关最低）；沙盒后每 4 周一个窗口（1 周校准 + 1 周评估）；各品牌减沙盒偏移；节点相关 0.43–0.68。
基线（同一批格子）：derive5；B1 鲁棒估计插值 = 全部节点当诚实，P(|N(0, (π/2)·散布² / n)| > h)；B2 独立锚点阈值 = 裁判输出偏离 p − b 超过 h 的题比例。
  识别基线：锚点阈值检测 |x̄| / √v > 3 判故障，vs v6.1 诚实概率 < 0.5。裁判基线：中位数、20% 截尾均值的实际被骗率。

预注册（运行前；Fashion-MNIST 的攻击数据、空气传感器的测试段都没看过）：
  1. 点预测与排序（域 1 前提成立）：MAE ≤ 0.10；Spearman ≥ 0.80
  2. 后验校准（同上，逐题）：ECE（10 档）≤ 0.05；后验 90% 区间覆盖 ∈ [80%, 97%]
  3. 预测区间：覆盖 ≥ 90%，宽度中位 ≤ 0.50
  4. 拒答：a 前提成立的格误拒 ≤ 5%；b B 不成立（f = 1）拒答 ≥ 80%；c ζ = 1 内部看不见（理论预言）：拒答 ≤ 20% 且 实测 − 点预测 ≥ 0.20 的 ≥ 50%；
          d 自然前提不成立（域 2）拒答 ≥ 80%（我的预期：过不了——共模误差对秤是稳定的，自检只看挪动）
  5. 基线：v6.1 的 MAE 低于 derive5、B1、B2 三者；诚实识别平衡准确率 ≥ 锚点阈值检测；同时报告裁判 vs 中位数 / 截尾均值的被骗率差

用法：FL_LR=0.1 FL_DATA=data_fashion ~/gut-external/torch215-venv/bin/python validate_v61.py
"""
import collections, datetime as dt, itertools, json, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("switching", "fl", "senseur", "epa"):
    sys.path.insert(0, os.path.join(HERE, d))
import torch
from analytic import Phi, derive5, sigma_c
from hetero import judge_out, measure
import law_v6 as L
assert L.LAW_VERSION == "6.1"


def baselines(clean, bad, judge, s0):
    b, _ = sigma_c(clean)
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    n = len(clean[0][2])
    sm = math.sqrt(math.pi / 2) * spread / math.sqrt(n)
    s = s0
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, L._mad(dev).min().item())
    jd = "软" if judge == "软" else "硬"
    b1 = st.mean(2 * (1 - Phi((x / 2) / sm)) for *_, x in bad)
    b2 = st.mean(abs(judge_out(r, torch.tensor(p), s, jd) - (p - b)) > x / 2 for _, p, r, x in bad)
    return b1, b2


def identify(clean, bad, poisoned):
    """诚实概率 < 0.5 vs 锚点阈值 |x̄|/√v > 3 的平衡准确率（投毒身份已知时）。"""
    F = L.fit(clean, bad)
    b = F["b"]
    Rb = torch.stack([r for _, _, r, _ in bad])
    pb = torch.tensor([p for _, p, _, _ in bad])
    X = Rb - (pb - b).unsqueeze(1)
    e = [p - r.median().item() for _, p, r, _ in clean]
    se_b = 1.2533 * 1.4826 * st.median(abs(x - b) for x in e) / math.sqrt(len(e))
    z = X.median(0).values / torch.sqrt((1.2533 * L._mad(X)) ** 2 / len(bad) + se_b ** 2)
    ba = lambda flag: (st.mean(flag[i] for i in range(len(flag)) if poisoned[i]) + st.mean(not flag[i] for i in range(len(flag)) if not poisoned[i])) / 2
    return ba((F["ph"] < 0.5).tolist()), ba((z.abs() > 3).tolist())


def fl_domain():
    import flbench as F
    attacks = ["ALIE0.5", "ALIE1", "ALIE2", "ALIE4", "noise", "flip"]
    R, items, ids = [], [], []
    k = 0
    kw = {"dense": 0.5, "v2": True}
    for n, f, attack, m in itertools.product((10, 20), (0.2, 0.4, 0.6), attacks, (32, 512)):
        k += 1
        clean, bad, acc = F.run(n, f, attack, m, 7000 + k, **kw)
        bl = ([i // F.K for i in range(len(clean))], [i // F.K for i in range(len(bad))])
        refuse = L.premise_check(clean, bad, bl[0], bl[1], bad[0][3] / 2)[0]
        nb = int(round(f * n))
        pois = [i >= n - nb for i in range(n)]
        ids.append(identify(clean, bad, pois))
        for judge in ("软", "硬2"):
            pt, lo, hi, s0, rh = L.interval(clean, bad, judge, B=20, blocks=bl)
            q, parts, _ = L.item_probs(clean, bad, judge, s0)
            act = measure(clean + bad, bad[0][3], judge, s0)["miss"]
            b1, b2 = baselines(clean, bad, judge, s0)
            d5 = derive5([(False, p, r, x) for _, p, r, x in clean] + bad, judge, s0)
            vote = st.mean(abs(r.median().item()) > x / 2 for _, _, r, x in bad)
            def trim(r):
                v, _ = torch.sort(r); c = int(0.2 * len(v))
                return v[c:len(v) - c].mean().item()
            tm = st.mean(abs(trim(r)) > x / 2 for _, _, r, x in bad)
            R.append({"n": n, "f": f, "attack": attack, "m": m, "judge": judge, "act": act, "pt": pt, "lo": lo, "hi": hi, "rho": rh, "refuse": refuse, "d5": d5, "b1": b1, "b2": b2, "vote": vote, "trim": tm})
            for qi, (o, mm, sd, t) in zip(q, parts):
                items.append((qi, abs(o) > t, st.mean(Phi(x) for x in ((0 - mm) / sd).tolist())))
        print(f"{k}/72 n={n} f={f} {attack} m={m} 实测 {R[-2]['act']:.2f} 点 {R[-2]['pt']:.2f}", flush=True)
    Bv = []
    for n, m, attack in itertools.product((10, 20), (32, 512), ["ALIE1", "ALIE2", "ALIE4", "noise", "flip"]):
        k += 1
        clean, bad, _ = F.run(n, 1.0, attack, m, 7000 + k, **kw)
        bl = ([i // F.K for i in range(len(clean))], [i // F.K for i in range(len(bad))])
        ref = L.premise_check(clean, bad, bl[0], bl[1], bad[0][3] / 2)
        s0 = max(sigma_c(clean)[1], 1e-6)
        Bv.append({"attack": attack, "n": n, "m": m, "refuse": ref[0], "moved": ref[1], "act": measure(clean + bad, bad[0][3], "软", s0)["miss"], "pt": L.predict(clean, bad, "软", s0)})
    Zv = []
    for n, m, f in itertools.product((10, 20), (32, 512), (0.4, 0.6)):
        k += 1
        clean, bad, _ = F.run(n, f, "flip-ζ", m, 7000 + k, **kw)
        bl = ([i // F.K for i in range(len(clean))], [i // F.K for i in range(len(bad))])
        ref = L.premise_check(clean, bad, bl[0], bl[1], bad[0][3] / 2)
        s0 = max(sigma_c(clean)[1], 1e-6)
        Zv.append({"n": n, "m": m, "f": f, "refuse": ref[0], "act": measure(clean + bad, bad[0][3], "软", s0)["miss"], "pt": L.predict(clean, bad, "软", s0)})
    return R, items, ids, Bv, Zv


def natural():
    """域 2：每格 (拒答?, 实测, 点预测)。"""
    import oslo as O
    import epa as E
    H = math.log10(1.25)
    out = []
    D = O.load()
    us = O.present(D, [39, 40])
    for wc, we in ((38, 39), (39, 40), (40, 41)):
        rc, re = O.rows(D, [wc], us), O.rows(D, [we], us)
        if len(rc) < 48 or len(re) < 48:
            continue
        off = torch.stack([r for _, _, r in O.rows(D, [36, 37], us)]).median(0).values if O.rows(D, [36, 37], us) else 0
        cal = [(False, p, r - off, 2 * H) for _, p, r in rc]
        ev = [(True, p, r - off, 2 * H) for _, p, r in re]
        ref = L.premise_check(cal, ev, [h[:10] for h, _, _ in rc], [h[:10] for h, _, _ in re], H)
        s0 = max(sigma_c(cal)[1], 1e-6)
        out.append({"src": f"Oslo w{wc}", "refuse": ref[0], "act": measure(cal + ev, 2 * H, "软", s0)["miss"], "pt": L.predict(cal, ev, "软", s0)})
    S, Ref, _ = E.load()
    NODES, SC = ["AQY", "CNO", "MAX", "PAR"], "RAM"
    for loc in sorted(S):
        ts = sorted(S[loc])
        t0 = dt.datetime.fromisoformat(ts[0])
        rows = []
        for t in ts:
            v = S[loc][t]
            if Ref[loc].get(t, 0) >= 3 and all(m in v and st.median(v[m]) > 0 for m in NODES + [SC]):
                tru = math.log10(Ref[loc][t])
                rows.append(((dt.datetime.fromisoformat(t) - t0).days, t[:10], math.log10(st.median(v[SC])) - tru, torch.tensor([math.log10(st.median(v[m])) - tru for m in NODES])))
        sb = [x for x in rows if x[0] < 90]
        if len(sb) < 200:
            continue
        off = torch.stack([x[3] for x in sb]).median(0).values
        off_s = st.median(x[2] for x in sb)
        for w in range(90, rows[-1][0] - 14, 28):
            rc = [x for x in rows if w <= x[0] < w + 7]
            re = [x for x in rows if w + 7 <= x[0] < w + 14]
            if len(rc) < 48 or len(re) < 48:
                continue
            cal = [(False, x[2] - off_s, x[3] - off, 2 * H) for x in rc]
            ev = [(True, x[2] - off_s, x[3] - off, 2 * H) for x in re]
            ref = L.premise_check(cal, ev, [x[1] for x in rc], [x[1] for x in re], H)
            s0 = max(sigma_c(cal)[1], 1e-6)
            out.append({"src": f"EPA {loc} d{w}", "refuse": ref[0], "act": measure(cal + ev, 2 * H, "软", s0)["miss"], "pt": L.predict(cal, ev, "软", s0)})
    return out


def main():
    rk = lambda v: torch.tensor(v, dtype=torch.float).argsort().argsort().float()
    R, items, ids, Bv, Zv = fl_domain()
    Nat = natural()
    json.dump({"version": L.LAW_VERSION, "cells": R, "items": items, "ids": ids, "B": Bv, "Z": Zv, "natural": Nat}, open(os.path.join(HERE, "validate_v61.json"), "w"), ensure_ascii=False)
    act = [r["act"] for r in R]
    mae = lambda k: st.mean(abs(r[k] - r["act"]) for r in R)
    print(f"\n规律版本 {L.LAW_VERSION}；域 1：{len(R)} 格，实测被骗均值 {st.mean(act):.3f}")
    print(f"1. MAE {mae('pt'):.3f}（≤ 0.10）；点误差中位 {st.median(abs(r['pt'] - r['act']) for r in R):.3f}；Spearman {torch.corrcoef(torch.stack([rk([r['pt'] for r in R]), rk(act)]))[0, 1].item():.2f}（≥ 0.80）")
    bins = [[] for _ in range(10)]
    for q, y, _ in items:
        bins[min(int(q * 10), 9)].append((q, y))
    ece = sum(len(b) * abs(st.mean(x[0] for x in b) - st.mean(x[1] for x in b)) for b in bins if b) / len(items)
    print(f"2. ECE {ece:.3f}（≤ 0.05）；Brier {st.mean((q - y) ** 2 for q, y, _ in items):.3f}；后验 90% 区间覆盖 {st.mean(.05 <= p <= .95 for _, _, p in items):.0%}（80–97%）")
    print("   校准表：" + "  ".join(f"{st.mean(x[0] for x in b):.2f}→{st.mean(x[1] for x in b):.2f}" for b in bins if b))
    print(f"3. 区间覆盖 {st.mean(r['lo'] <= r['act'] <= r['hi'] for r in R):.0%}（≥ 90%）；宽度中位 {st.median(r['hi'] - r['lo'] for r in R):.2f}（≤ 0.50）")
    print(f"4a. 前提成立误拒 {st.mean(r['refuse'] for r in R):.0%}（≤ 5%）")
    print(f"4b. f = 1 拒答 {st.mean(r['refuse'] for r in Bv):.0%}（≥ 80%）；按攻击 " + str({a: round(st.mean(r['refuse'] for r in Bv if r['attack'] == a), 2) for a in {r['attack'] for r in Bv}}) + f"；没拒的 MAE {st.mean([abs(r['act'] - r['pt']) for r in Bv if not r['refuse']] or [float('nan')]):.3f}")
    print(f"4c. ζ = 1 拒答 {st.mean(r['refuse'] for r in Zv):.0%}（≤ 20%）；实测 − 点预测 ≥ 0.20 的 {st.mean(r['act'] - r['pt'] >= 0.2 for r in Zv):.0%}（≥ 50%）")
    print(f"4d. 自然前提不成立 {len(Nat)} 格：拒答 {st.mean(r['refuse'] for r in Nat):.0%}（≥ 80%）；没拒的 MAE {st.mean([abs(r['act'] - r['pt']) for r in Nat if not r['refuse']] or [float('nan')]):.3f}")
    print(f"5. MAE：v6.1 {mae('pt'):.3f} vs derive5 {mae('d5'):.3f}、B1 鲁棒 {mae('b1'):.3f}、B2 锚点阈值 {mae('b2'):.3f}")
    print(f"   Spearman：derive5 {torch.corrcoef(torch.stack([rk([r['d5'] for r in R]), rk(act)]))[0, 1].item():.2f}、B1 {torch.corrcoef(torch.stack([rk([r['b1'] for r in R]), rk(act)]))[0, 1].item():.2f}、B2 {torch.corrcoef(torch.stack([rk([r['b2'] for r in R]), rk(act)]))[0, 1].item():.2f}")
    print(f"   识别平衡准确率：诚实概率 {st.mean(i[0] for i in ids):.2f} vs 锚点阈值 {st.mean(i[1] for i in ids):.2f}")
    soft = [r for r in R if r["judge"] == "软"]
    print(f"   实际被骗率：软裁判 {st.mean(r['act'] for r in soft):.3f}、中位数 {st.mean(r['vote'] for r in soft):.3f}、20% 截尾均值 {st.mean(r['trim'] for r in soft):.3f}")
    for lab, sel in (("f≥0.4 & m=512", lambda r: r["f"] >= 0.4 and r["m"] == 512), ("f=0.6", lambda r: r["f"] == 0.6), ("m=32", lambda r: r["m"] == 32)):
        s = [r for r in R if sel(r)]
        print(f"   难区 {lab}（{len(s)} 格）MAE：v6.1 {st.mean(abs(r['pt'] - r['act']) for r in s):.3f}、B2 {st.mean(abs(r['b2'] - r['act']) for r in s):.3f}、derive5 {st.mean(abs(r['d5'] - r['act']) for r in s):.3f}")


if __name__ == "__main__":
    main()
