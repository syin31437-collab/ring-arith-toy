"""外部验证：BIPM 时间实验室（2026-11-02）。规律 6.1 + 前提自检 2.0 冻结（git tag law-v6.1、premise-2.0），一个字不改。子代理找到的数据；v6 开发完全没碰过。

映射：节点 = 9 个实验室（5 天差分，ns，真值 0）；秤 A = GPS、秤 B = GLONASS（只用于自检）；裁判 = 软 / 硬2。
  选节点只用沙盒（2015 年以前覆盖 ≥ 97%）：按固定种子分成不重叠的 4 组，每组 9 个。
窗口：测试段（2015 起）每 130 个历元一格，前 104 校准（约 1.4 年）、后 26 评估；缺数据的历元丢掉；校准 < 50 或评估 < 15 题的格丢掉。
条件（每个窗口 × 每组）：
  自然   不注入，h ∈ {1.5, 3} ns，看真实噪声和真实跳变下裁判被骗多少
  注入   k ∈ {2, 4, 6} 个节点在评估期频率阶跃 δ ∈ {3, 6, 12} ns / 5 天（h = δ / 2）。噪声是真的，故障是注入的
  B 不成立  k = 9，δ ∈ {6, 12}（评估期所有节点一起阶跃）
  S 不成立  秤 A 在评估期和 k = 4 个节点同向偏 δ ∈ {6, 12}（ζ = 1），秤 B 不受影响
基线同 validate_v61：derive5、B1 鲁棒插值、B2 独立锚点阈值；识别 vs 锚点阈值。

映射预检（precheck.py，沙盒）：④ 相关 0.017 ✅、① −0.2 ns ✅、② |z| 95% 2.45 ✅ 但 99% 5.05 略超门槛 5 ⚠️（真实时钟跳变造成的厚尾）。
  ③ 没有过：秤实际 MAD / 规律估的秤标准差，校准 104 题时比值中位 1.36、但离群到 4.7，只有 4/8 个沙盒窗落在 [0.7, 1.4]（precheck3b.py）。校准再短更差（26 题 6/20）。
  破例说明：之前的规矩是预检全过才预注册。这次照跑，因为 ① ④ 表明映射本身是对的（节点独立、真值独立、秤独立），不稳的是规律 6.1 的秤噪声估计（节点平均比秤还吵时相减不稳）——这正是独立验证要暴露的东西，规律不改。预期校准会因此变差。

预注册（运行前，测试段没看过）：
  1. 自然 + 注入（前提成立）：MAE ≤ 0.10；Spearman ≥ 0.80
  2. 校准：逐题 ECE ≤ 0.05；后验 90% 区间覆盖 80–97%
  3. 预测区间覆盖 ≥ 90%，宽度中位 ≤ 0.50
  4. 拒答（前提自检 2.0）：a 前提成立误拒 ≤ 5%（预期过不了）；b B 不成立拒答 ≥ 80%；c S 不成立拒答 ≥ 80%
  5. 基线：MAE 低于 derive5、B1、B2；注入格的诚实识别平衡准确率 ≥ 锚点阈值
  6. 规律的分档：ρ̂ ≥ 1 的格子比 ρ̂ < 1 的 MAE 低（秤 GPS 比一半节点准、比最好的节点粗）
  沙盒只用于调试流程（--sandbox 的数字不算结果）。

测试段结果（6179a7e 预注册；17 个窗口-组、374 格，覆盖不足的窗口按规则丢掉，比预想的 24 个少）：
  1 ✅ MAE 0.071、点误差中位 0.038、Spearman 0.85
  2 半对：ECE 0.047 ✅；后验 90% 区间覆盖 78% ❌（门槛 80%）。高端仍过自信：0.98 → 0.83
  3 ✅ 覆盖 94%、宽度中位 0.43
  4 a ❌ 误拒 7%（门槛 5%，比预期好得多）；b ❌ B 不成立拒答 59%，但这些格预测 MAE 只有 0.052（不拒也没大错）；c ✅ S 不成立拒答 91%，预测本身 MAE 0.72（实测 0.84、预测 0.18）——拒答救了它
  5 ✅ MAE v6.1 0.071 < derive5 0.089、B2 0.108、B1 0.153（按窗口-组 bootstrap 差值区间都不含 0）；识别 0.81 ≥ 锚点阈值 0.80（基本打平）
  6 无法检验：374 格 ρ̂ 全部 ≥ 1（最小 1.1）
  最弱处（复查，非预注册）：纯自然格（不注入）MAE 0.109，系统性高估：实测 0.104、预测 0.200；h = 1.5 ns 时实测 0.164、预测 0.315。原因没定位。注入格 MAE 0.063，δ = 3 时 0.122、δ = 12 时 0.015。
  局限：注入格的故障是人造的（噪声是真的）；17 个窗口-组里同一时段共用同一把 GPS 秤，不完全独立；预检 ② ③ 有违规。

定位（locate.py、locate2.py，看过结果后的诊断，修法都没验证）：
  高估主要来自硬裁判：硬裁判实测 0.104、预测 0.322（h = 1.5）；软裁判 0.224、0.308。
  ① 硬裁判比后验中心更准（误差 MAD 0.59 vs 0.90）。后验按高斯加权，宽度对这个估计量是诚实的（|m| / sd 中位 0.69），但真实时钟厚尾，稳健的裁判用信息更有效 → 公式把"后验中心的不确定"算成了"裁判的不确定"。
  ② 软裁判的窗宽 s 还在用旧的 sigma_c（中位 0.26），规律 v6.1 自己用的秤标准差是 1.74：v6.1 只修了后验、没修裁判 → 软裁判几乎只跟离 GPS 最近的节点走，自然格被骗 0.224，比不看秤的中位数 0.152 还差。
  ③ 后验覆盖 79% 主要因为秤噪声仍估低：换真值后 94%，但那会让预测更高（MAE 0.113 → 0.131）——① 和 ③ 方向相反，现在两个错互相抵消了一部分。
  否决的怀疑：只把后验中心换成加权中位数没用（MAE 0.113 → 0.124）；去掉 λ、只用校准期噪声、全员诚实都几乎不变。

用法：~/gut-external/torch215-venv/bin/python validate_bipm.py sandbox | test
"""
import itertools, json, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../switching", "..", "."):
    sys.path.insert(0, os.path.join(HERE, d))
import torch
import bipm as B
import law_v6 as L
from analytic import Phi, derive5, sigma_c
from hetero import measure
from premise2 import decide
from validate_v61 import baselines, identify
assert L.LAW_VERSION == "6.1"


def groups():
    labs = B.all_labs()
    ep = [t for t in sorted({t for d in labs.values() for t in d}) if 55564 <= t < B.SANDBOX_END]
    good = sorted(k for k, d in labs.items() if len(B.diffs(d, ep)) >= 0.97 * len(ep))
    random.Random("bipm-groups").shuffle(good)
    return labs, [good[i:i + 9] for i in range(0, len(good) - 8, 9)][:4]


def windows(mode):
    lo, hi = (55564, B.SANDBOX_END) if mode == "sandbox" else (B.SANDBOX_END, 10 ** 6)
    labs, G = groups()
    gps, glo = B.gnss("gps"), B.gnss("glo")
    epochs = [t for t in sorted({t for d in labs.values() for t in d}) if lo <= t < hi]
    pg, pl = B.diffs(gps, epochs), B.diffs(glo, epochs)
    for gi, g in enumerate(G):
        D = [B.diffs(labs[k], epochs) for k in g]
        for w in range(0, len(epochs) - 129, 130):
            cut = epochs[w:w + 130]
            rows = [(t, pg[t], pl[t], torch.tensor([d[t] for d in D])) for t in cut if t in pg and t in pl and all(t in d for d in D)]
            cal = [x for x in rows if x[0] < cut[104]]
            ev = [x for x in rows if x[0] >= cut[104]]
            if len(cal) >= 50 and len(ev) >= 15:
                yield gi, w, cal, ev


def one(tag, cal, ev, h, shift=None, zeta=0.0, judges=("软", "硬2")):
    """shift = 节点的注入向量；zeta = 秤 A 在评估期的注入。"""
    sh = shift if shift is not None else torch.zeros(len(cal[0][3]))
    c = [(False, a, r, 2 * h) for _, a, _, r in cal]
    e = [(True, a + zeta, r + sh, 2 * h) for _, a, _, r in ev]
    prem = decide(c, e, [b for _, _, b, _ in cal], [b for _, _, b, _ in ev], h)
    out, items = [], []
    for judge in judges:
        pt, lo, hi, s0, rh = L.interval(c, e, judge, B=20)
        act = measure(c + e, 2 * h, judge, s0)["miss"]
        b1, b2 = baselines(c, e, judge, s0)
        d5 = derive5(c + e, judge, s0)
        q, parts, _ = L.item_probs(c, e, judge, s0)
        for qi, (o, m, sd, t) in zip(q, parts):
            items.append((qi, abs(o) > t, st.mean(Phi(x) for x in ((0 - m) / sd).tolist())))
        out.append({"tag": tag, "judge": judge, "h": h, "act": act, "pt": pt, "lo": lo, "hi": hi, "rho": rh, "refuse": prem["refuse"], "d5": d5, "b1": b1, "b2": b2, "s1": prem["s1"], "m1": prem["m1"]})
    return out, items


def main(mode):
    R, items, ids, Bv, Sv = [], [], [], [], []
    rng = random.Random("bipm-inject")
    n_w = 0
    for gi, w, cal, ev in windows(mode):
        n_w += 1
        for h in (1.5, 3.0):
            o, it = one("自然", cal, ev, h)
            R += o; items += it
        for k, dl in itertools.product((2, 4, 6), (3.0, 6.0, 12.0)):
            idx = rng.sample(range(9), k)
            sh = torch.zeros(9); sh[idx] = dl
            o, it = one("注入", cal, ev, dl / 2, sh)
            R += o; items += it
            pois = [i in idx for i in range(9)]
            c = [(False, a, r, dl) for _, a, _, r in cal]
            e = [(True, a, r + sh, dl) for _, a, _, r in ev]
            ids.append(identify(c, e, pois))
        for dl in (6.0, 12.0):
            o, _ = one("B", cal, ev, dl / 2, torch.full((9,), dl), judges=("软",))
            Bv += o
            idx = rng.sample(range(9), 4)
            sh = torch.zeros(9); sh[idx] = dl
            o, _ = one("S", cal, ev, dl / 2, sh, zeta=dl, judges=("软",))
            Sv += o
        print(f"组 {gi} 窗 {w} 完成（校准 {len(cal)}、评估 {len(ev)}）", flush=True)
    json.dump({"law": L.LAW_VERSION, "cells": R, "B": Bv, "S": Sv, "ids": ids, "items": items}, open(os.path.join(HERE, f"validate_bipm_{mode}.json"), "w"), ensure_ascii=False)
    rk = lambda v: torch.tensor(v, dtype=torch.float).argsort().argsort().float()
    act = [r["act"] for r in R]
    mae = lambda k, s=R: st.mean(abs(r[k] - r["act"]) for r in s)
    sp = lambda k: torch.corrcoef(torch.stack([rk([r[k] for r in R]), rk(act)]))[0, 1].item()
    print(f"\n规律 {L.LAW_VERSION}；{mode}：{n_w} 个窗口-组，{len(R)} 格，实测被骗均值 {st.mean(act):.3f}")
    print(f"1. MAE {mae('pt'):.3f}（≤ 0.10）；点误差中位 {st.median(abs(r['pt'] - r['act']) for r in R):.3f}；Spearman {sp('pt'):.2f}（≥ 0.80）")
    for t in ("自然", "注入"):
        s = [r for r in R if r["tag"] == t]
        print(f"   {t}：{len(s)} 格，实测 {st.mean(r['act'] for r in s):.3f}，预测 {st.mean(r['pt'] for r in s):.3f}，MAE {mae('pt', s):.3f}，B2 {mae('b2', s):.3f}")
    bins = [[] for _ in range(10)]
    for q, y, _ in items:
        bins[min(int(q * 10), 9)].append((q, y))
    ece = sum(len(b) * abs(st.mean(x[0] for x in b) - st.mean(x[1] for x in b)) for b in bins if b) / len(items)
    print(f"2. ECE {ece:.3f}（≤ 0.05）；Brier {st.mean((q - y) ** 2 for q, y, _ in items):.3f}；后验 90% 区间覆盖 {st.mean(.05 <= p <= .95 for _, _, p in items):.0%}（80–97%）")
    print("   校准表：" + "  ".join(f"{st.mean(x[0] for x in b):.2f}→{st.mean(x[1] for x in b):.2f}" for b in bins if b))
    print(f"3. 区间覆盖 {st.mean(r['lo'] <= r['act'] <= r['hi'] for r in R):.0%}（≥ 90%）；宽度中位 {st.median(r['hi'] - r['lo'] for r in R):.2f}（≤ 0.50）")
    print(f"4a. 前提成立误拒 {st.mean(r['refuse'] for r in R):.0%}（≤ 5%）")
    print(f"4b. B 不成立拒答 {st.mean(r['refuse'] for r in Bv):.0%}（≥ 80%）；预测 MAE {mae('pt', Bv):.3f}")
    print(f"4c. S 不成立拒答 {st.mean(r['refuse'] for r in Sv):.0%}（≥ 80%）；S1 触发 {st.mean(r['s1'] for r in Sv):.0%}；预测 MAE {mae('pt', Sv):.3f}（实测 {st.mean(r['act'] for r in Sv):.2f}、预测 {st.mean(r['pt'] for r in Sv):.2f}）")
    print(f"5. MAE：v6.1 {mae('pt'):.3f} vs derive5 {mae('d5'):.3f}、B1 {mae('b1'):.3f}、B2 {mae('b2'):.3f}；Spearman：derive5 {sp('d5'):.2f}、B1 {sp('b1'):.2f}、B2 {sp('b2'):.2f}")
    print(f"   识别平衡准确率：诚实概率 {st.mean(i[0] for i in ids):.2f} vs 锚点阈值 {st.mean(i[1] for i in ids):.2f}")
    hi_, lo_ = [r for r in R if r["rho"] >= 1], [r for r in R if r["rho"] < 1]
    print(f"6. ρ̂ ≥ 1：{len(hi_)} 格 MAE {mae('pt', hi_) if hi_ else float('nan'):.3f}；ρ̂ < 1：{len(lo_)} 格 MAE {mae('pt', lo_) if lo_ else float('nan'):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
