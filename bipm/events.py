"""第 3 步：报警对不对得上别人独立记录的真实事件（BIPM notes.utc）。事件标签由 BIPM 记录，和我们的检测无关；不注入任何故障。
事件解析（规则在看对照结果前写定）：notes 里每个实验室段，凡提到 MJD 的行算一个事件，MJD 取该行第一个 5 位数；
  类型：含 "frequency" → 频率；含 "master clock" / "clock" 且无 step → 换钟；含 "step" → 时间跳变（再取 ns 大小）；其余 → 其他。
检测（常数沿用冻结值）：测试段（2015 起）每个实验室，滚动窗口：52 个历元校准 + 26 个历元评估。
  EWMA：节点 − 秤（L2，GPS）/ 节点 − 其他实验室中位数（L0），和前面一样，越线 = |S| > 3.3 × SD_S。截断使它只对持续偏移敏感。
  单点：|z| > 5（不截断，只看这一题），针对一次性跳变。
评估：节点-窗口 为单位，事件落在该评估窗内 = 正例。
预注册（运行前，notes 的事件还没和报警比过）：
  1. EWMA 报警的节点-窗口里，有记录事件的比例 ≥ 2 × 没报警的（提升度 ≥ 2）；实验室标签随机打乱 2000 次的置换 p < 0.01
  2. 单点报警：同样提升度 ≥ 2、p < 0.01
  3. 大跳变（≥ 10 ns）的事件窗，单点检出 ≥ 70%（L2）
  4. 频率事件的窗，EWMA 检出 > 全部窗的 EWMA 报警基础率（持续偏移才看得见，T6）；时间跳变的窗则不要求（设计上看不见）
  5. L0 和 L2 对单个实验室的事件差不多（T3：单点故障不需要秤）：提升度相差 ≤ 1.5 倍
用法：~/gut-external/torch215-venv/bin/python events.py
"""
import collections, random, re, statistics as st, sys
sys.path.insert(0, "../switching")
import bipm as B
from purge import C as CC, SD_S, W


def parse_notes():
    ev = collections.defaultdict(list)
    cur = None
    for line in open("data/notes.utc", errors="ignore"):
        m = re.match(r"\s{3,6}([A-Z][A-Z0-9]{1,4})\s*\.", line)
        if m:
            cur = m.group(1).lower()
        if cur is None:
            continue
        for mj in re.findall(r"MJD[^0-9]{0,8}(\d{5})", line)[:1]:
            low = line.lower()
            kind = "freq" if "frequency" in low else ("step" if "step" in low else ("clock" if "clock" in low else "other"))
            ns = re.search(r"([-+]?\s?\d+(?:\.\d+)?)\s*ns\b", line)
            ev[cur].append((int(mj), kind, abs(float(ns.group(1).replace(" ", ""))) if ns else None))
    return ev


def windows():
    labs = B.all_labs()
    gps = B.gnss("gps")
    epochs = [t for t in sorted({t for d in labs.values() for t in d}) if t >= B.SANDBOX_END]
    pg = B.diffs(gps, epochs)
    D = {k: B.diffs(v, epochs) for k, v in labs.items()}
    keep = [k for k, d in D.items() if len(d) >= 0.8 * len(epochs)]
    med = {t: sorted(D[k][t] for k in keep if t in D[k]) for t in epochs}
    out = []
    for k in keep:
        for w in range(0, len(epochs) - 77, 78):
            cut = epochs[w:w + 78]
            ts = [t for t in cut if t in D[k] and t in pg and len(med[t]) >= 5]
            nc = sum(t < cut[52] for t in ts)
            if nc < 30 or len(ts) - nc < 15:
                continue
            def stats(x):
                base = st.median(x[:nc])
                sd = max(1.4826 * st.median(abs(v - base) for v in x[:nc]), 1e-6)
                S, ew, one = 0.0, False, False
                for i, v in enumerate(x):
                    z = (v - base) / sd
                    S = (1 - W) * S + W * max(-3.0, min(3.0, z))
                    if i >= nc:
                        ew = ew or abs(S) > CC * SD_S
                        one = one or abs(z) > 5
                return ew, one
            xa = [D[k][t] - pg[t] for t in ts]
            xp = [D[k][t] - st.median(v for v in med[t]) for t in ts]
            out.append({"lab": k, "lo": cut[52], "hi": cut[-1], "L2": stats(xa), "L0": stats(xp)})
    return out, keep


def lift(W_, ev, idx, L):
    has = lambda w, E: any(w["lo"] <= m <= w["hi"] for m, _, _ in E.get(w["lab"], []))
    a = [has(w, ev) for w in W_ if w[L][idx]]
    b = [has(w, ev) for w in W_ if not w[L][idx]]
    pa, pb = (st.mean(a) if a else 0.0), (st.mean(b) if b else 0.0)
    return pa, pb, (pa / pb if pb else float("inf")), len(a), len(b)


def main():
    ev = parse_notes()
    W_, keep = windows()
    lab_ev = {k: [e for e in ev.get(k, []) if e[0] >= B.SANDBOX_END] for k in keep}
    cnt = collections.Counter(e[1] for v in lab_ev.values() for e in v)
    print(f"{len(keep)} 个实验室，{len(W_)} 个节点-窗口；测试段记录事件 {sum(cnt.values())}，类型 {dict(cnt)}")
    rg = random.Random(0)
    for L in ("L2", "L0"):
        for idx, nm in ((0, "EWMA"), (1, "单点")):
            pa, pb, lf, na, nb = lift(W_, lab_ev, idx, L)
            perm = []
            labs = list(lab_ev)
            for _ in range(2000):
                sh = labs[:]
                rg.shuffle(sh)
                E2 = dict(zip(labs, (lab_ev[s] for s in sh)))
                perm.append(lift(W_, E2, idx, L)[2])
            p = st.mean(x >= lf for x in perm)
            print(f"  {L} {nm}：报警 {na} 窗里有事件 {pa:.0%}，没报警 {nb} 窗里 {pb:.0%}，提升度 {lf:.2f}，置换 p = {p:.4f}")
    big = [w for w in W_ if any(w["lo"] <= m <= w["hi"] and k == "step" and (n or 0) >= 10 for m, k, n in lab_ev[w["lab"]])]
    fr = [w for w in W_ if any(w["lo"] <= m <= w["hi"] and k == "freq" for m, k, _ in lab_ev[w["lab"]])]
    st_ = [w for w in W_ if any(w["lo"] <= m <= w["hi"] and k == "step" for m, k, _ in lab_ev[w["lab"]])]
    print(f"3. 大跳变（≥ 10 ns）的 {len(big)} 窗：单点检出 L2 {st.mean(w['L2'][1] for w in big) if big else float('nan'):.0%}（≥ 70%）；L0 {st.mean(w['L0'][1] for w in big) if big else float('nan'):.0%}；EWMA L2 {st.mean(w['L2'][0] for w in big) if big else float('nan'):.0%}")
    print(f"4. 频率事件 {len(fr)} 窗：EWMA 检出 {st.mean(w['L2'][0] for w in fr) if fr else float('nan'):.0%}；时间跳变 {len(st_)} 窗：EWMA {st.mean(w['L2'][0] for w in st_) if st_ else float('nan'):.0%}；全部窗 EWMA 基础率 {st.mean(w['L2'][0] for w in W_):.0%}、单点基础率 {st.mean(w['L2'][1] for w in W_):.0%}")
    json_out = {"windows": len(W_)}


if __name__ == "__main__":
    main()
