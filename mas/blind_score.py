"""blindrule.py 的评分：预注册判据原样计算；真值只在这里用。用法：python blind_score.py <目录>"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

H = math.log10(1.4) / 2
A = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b", "claude_haiku", "codex"]
ok = lambda x: abs(x) <= H


def load(d):
    R, miss, tot, errs = [], 0, 0, 0
    for fn in sorted(glob.glob(os.path.join(d, "*", "run.json"))):
        J = json.load(open(fn))
        errs += len(J["errors"])
        for it in J["items"]:
            i, t = it["id"], math.log10(it["truth"])
            s1all = {a: J["s1"][a].get(i, 0) for a in A}
            for arm in ("num", "note"):
                for a in A:
                    tot += 1
                    vals = [s1all[x] for x in A] + [J[arm]["s2"][a].get(i, 0), J[arm]["s3"][a].get(i, 0)]
                    if min(vals) <= 0:
                        miss += 1
                        continue
                    s1 = math.log10(s1all[a]) - t
                    pm = st.median(math.log10(s1all[x]) - t for x in A if x != a)
                    R.append(dict(k=J["k"], rep=J["rep"], arm=arm, a=a, i=i, bad=a in J["bad"], s1=s1, s3=math.log10(J[arm]["s3"][a][i]) - t, pm=pm, rep_v=math.log10(J["reports"][a][i]) - t))
    return R, miss, tot, errs


def stats(rs):
    g = [r for r in rs if ok(r["s1"])]
    b = [r for r in rs if not ok(r["s1"])]
    return dict(K=st.mean(ok(r["s3"]) for r in g) if g else float("nan"), B=st.mean(ok(r["s3"]) for r in b) if b else float("nan"), net=st.mean(ok(r["s3"]) for r in rs) - st.mean(ok(r["s1"]) for r in rs))


def boot(rs, fn, n=1000):
    reps = sorted({r["rep"] for r in rs})
    by = {u: [r for r in rs if r["rep"] == u] for u in reps}
    g = random.Random(0)
    v = sorted(fn([r for u in g.choices(reps, k=len(reps)) for r in by[u]]) for _ in range(n))
    return v[int(.025 * n)], v[int(.975 * n)]


def frac(r):
    return (r["s3"] - r["s1"]) / (r["pm"] - r["s1"])


def main():
    R, miss, tot, errs = load(sys.argv[1])
    print(f"局数 {len({r['rep'] for r in R})}；节点-题-arm {tot}，缺失 {miss}（{miss / max(tot, 1):.1%}）；调用失败 {errs}")
    S = lambda k, arm: [r for r in R if r["k"] == k and r["arm"] == arm]
    print("\n映射预检（k = 0）")
    c = [r for r in S(0, "note")]
    if c:
        import torch
        items = sorted({(r["rep"], r["i"]) for r in c})
        M = torch.tensor([[next(r["s1"] for r in c if (r["rep"], r["i"]) == q and r["a"] == a) for a in A] for q in items])
        C = torch.corrcoef((M - M.mean(0)).T)
        print(f"  共识中位偏移 {M.median(1).values.median():+.4f}（≤ {H / 4:.4f}）；①−报告中位 {st.median(r['s1'] - r['rep_v'] for r in c):+.4f}（≤ 0.005）；两两相关 {(C.sum() - 6) / 30:.3f}（≤ 0.3）")
    print("\nk  arm   n局  K 好节点保住        B 错节点回来       net                ①准确率→③  干净报告节点 ①→③    污染报告节点 ①→③")
    table = {}
    for k in (0, 1, 2, 3, 4):
        for arm in ("num", "note"):
            rs = S(k, arm)
            if not rs:
                continue
            m = stats(rs)
            cK, cB, cN = boot(rs, lambda x: stats(x)["K"]), boot(rs, lambda x: stats(x)["B"]), boot(rs, lambda x: stats(x)["net"])
            table[k, arm] = (m, cK, cB, cN)
            good, bad = [r for r in rs if not r["bad"]], [r for r in rs if r["bad"]]
            f = lambda x, key: st.mean(ok(r[key]) for r in x) if x else float("nan")
            print(f"{k}  {arm:4s} {len({r['rep'] for r in rs}):3d}  {m['K']:5.1%}[{cK[0]:4.0%},{cK[1]:4.0%}]  {m['B']:5.1%}[{cB[0]:4.0%},{cB[1]:4.0%}]  {m['net']:+.3f}[{cN[0]:+.2f},{cN[1]:+.2f}]  {f(rs, 's1'):4.0%}→{f(rs, 's3'):4.0%}  {f(good, 's1'):4.0%}→{f(good, 's3'):4.0%}   {f(bad, 's1'):4.0%}→{f(bad, 's3'):4.0%}")
    if (0, "note") not in table:
        return
    print("\n预注册判据")
    for nm, arm in (("P1", "note"), ("P2", "num")):
        for k in (0, 1):
            if (k, arm) in table:
                m = table[k, arm][0]
                print(f"  {nm} k={k}: K {m['K']:.1%} ≥ 95% {'✅' if m['K'] >= .95 else '❌'}；B {m['B']:.1%} ≥ 85% {'✅' if m['B'] >= .85 else '❌'}；net {m['net']:+.3f} ≥ +0.15 {'✅' if m['net'] >= .15 else '❌'}")
    if (3, "note") in table:
        m = table[3, "note"][0]
        print(f"  P3 k=3: K {m['K']:.1%} ≤ 70% {'✅' if m['K'] <= .7 else '❌'}；net {m['net']:+.3f} ≤ +0.05 {'✅' if m['net'] <= .05 else '❌'}")
    if (4, "note") in table:
        m = table[4, "note"][0]
        print(f"  P4 k=4: K {m['K']:.1%} ≤ 50% {'✅' if m['K'] <= .5 else '❌'}；net {m['net']:+.3f} ≤ 0 {'✅' if m['net'] <= 0 else '❌'}")
    if (2, "note") in table:
        m, _, _, cN = table[2, "note"]
        print(f"  P5 k=2: net 下限 {cN[0]:+.3f} > 0 {'✅' if cN[0] > 0 else '❌'}；K {m['K']:.1%} ∈ [70%, 95%] {'✅' if .7 <= m['K'] <= .95 else '❌'}")
    print("  P6 配对差 note − num（按局 bootstrap）")
    for k in (0, 1, 2, 3, 4):
        if (k, "note") not in table:
            continue
        x = [r for r in R if r["k"] == k]
        fn = lambda key: (lambda y: stats([r for r in y if r["arm"] == "note"])[key] - stats([r for r in y if r["arm"] == "num"])[key])
        for key in ("K", "B"):
            lo, hi = boot(x, fn(key), 500)
            d = fn(key)(x)
            tag = ("✅" if abs(d) <= .05 else "❌") if k <= 1 else "（只报告）"
            print(f"    k={k} {key}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {tag}")
    low = [r for r in R if r["k"] <= 1 and abs(r["pm"] - r["s1"]) > .02]
    print("  P7 靠拢比例（k ≤ 1，全部节点）")
    bins = ((.02, .05), (.05, .10), (.10, 9))
    meds = []
    for lo, hi in bins:
        x = [r for r in low if lo < abs(r["pm"] - r["s1"]) <= hi]
        g, b = [r for r in x if ok(r["s1"])], [r for r in x if not ok(r["s1"])]
        m = st.median(map(frac, x))
        meds.append(m)
        print(f"    |gap| {lo:.2f}–{hi:.2f}: n={len(x):5d} 全部 {m:.2f}；好 {st.median(map(frac, g)) if g else float('nan'):.2f}（{len(g)}）；错 {st.median(map(frac, b)) if b else float('nan'):.2f}（{len(b)}）；差 {(st.median(map(frac, b)) - st.median(map(frac, g))) if g and b else float('nan'):+.2f}")
    print(f"    高 − 低 {meds[2] - meds[0]:+.2f} ≥ 0.10 {'✅' if meds[2] - meds[0] >= .10 else '❌'}；同格差绝对值 ≤ 0.05 见上")
    w = st.median(map(frac, [r for r in S(0, "note") if abs(r["pm"] - r["s1"]) > .02]))
    print(f"  P8 固定规则 w = {w:.2f}（k = 0、note、全部节点）")
    for k in (1, 2, 3, 4):
        for arm in ("num", "note"):
            rs = S(k, arm)
            if not rs:
                continue
            g, b = [r for r in rs if ok(r["s1"])], [r for r in rs if not ok(r["s1"])]
            rule = lambda r: r["s1"] + w * (r["pm"] - r["s1"])
            K_, B_ = st.mean(ok(rule(r)) for r in g), st.mean(ok(rule(r)) for r in b)
            m = table[k, arm][0]
            tag = (f"{'✅' if abs(m['K'] - K_) <= .05 and abs(m['B'] - B_) <= .05 else '❌'}") if k == 1 else "（只报告）"
            print(f"    k={k} {arm:4s}: 规则 K {K_:.1%} B {B_:.1%} net {st.mean(ok(rule(r)) for r in rs) - st.mean(ok(r['s1']) for r in rs):+.3f} ｜ LLM K {m['K']:.1%} B {m['B']:.1%} net {m['net']:+.3f} {tag}")


if __name__ == "__main__":
    main()
