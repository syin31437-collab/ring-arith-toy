"""第 4 步（theory/ladder.md T5）：任务内约束当秤。不看同伴、不看秤、不看真值，只用一个模型自己三个答案之间的矛盾，能不能知道它记错了？
数据：cache/（早期采集，7 个节点分批凭记忆答各国 2022 人口、总面积、人口密度；分开提问，不是我们注入的错误）。真值只用于评分：世界银行 SP.POP.TOTL、AG.SRF.TOTL.K2。
约束信号（单模型、无参照）：|log10 密度 − (log10 人口 − log10 面积)|，人口单位换成人。
对照：
  同伴（L0）：该节点三个量各自相对其他节点中位数的最大偏移（需要同伴，不需要约束）。
  合并：两者各自转成排名后取大。
目标：节点-国家级，三个量里至少一个相对真值错 > 20%。
T5 的盲区：错了但约束残差 < 5%（错误内部自洽）。

预注册（运行前；这个检验从没做过，cache 的答案以前只看过逐量误差）：
  1. 约束信号 AUC ≥ 0.70（无任何外部参照也能察觉）
  2. T5 盲区：错了的节点-国家里，约束残差 < 5% 的 ≥ 30%（模型常常自洽地错）
  3. 同伴信号 AUC ≥ 0.70；合并 AUC > 两者各自（两种信息互补）
  4. 错误是全体共有时（节点多数都错同一方向，同伴看不见）：约束 AUC ≥ 0.60、同伴 AUC ≤ 0.60（同伴的盲区不是约束的盲区）
结果（eafdc2c 预注册；7 节点、215 国、1430 节点-国家；错 5.8%，其中 93% 是共有错误）：
  1 ✅ 约束 AUC 0.720，但按国家 bootstrap 95% [0.586, 0.825]，下界低于门槛
  2 ✅ 自洽地错 75%（T5 的盲区很大）；残差 > 20% 时几乎全是真错（没错的里只有 0.0%），但只抓到错误的 11% → 响了很可信，大多数时候不响
  3 ❌ 同伴 AUC 0.691（差一点）；合并 0.700 不高于约束 0.720
  4 半对：共有错误上约束 0.704 ✅；同伴 0.671 没有 ≤ 0.60 ❌（难的国家大家也更分散，同伴仍有信号）

用法：~/gut-external/torch215-venv/bin/python constraint.py
"""
import json, math, os, random, statistics as st, collections

HERE = os.path.dirname(os.path.abspath(__file__))


def wb(ind):
    return {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, f"wb_{ind}.json")))[1] if r["value"]}


def load():
    A = collections.defaultdict(lambda: collections.defaultdict(dict))
    for fn in os.listdir(os.path.join(HERE, "cache")):
        for q in ("pop", "area", "dens"):
            if f"_{q}_" in fn:
                node = fn[:fn.index(f"_{q}_")]
                for i, v in json.load(open(os.path.join(HERE, "cache", fn)))["answers"].items():
                    if isinstance(v, (int, float)) and v > 0:
                        A[node][q][i] = v
    return A


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted((v, 1) for v in pos) + sorted((v, 0) for v in neg)
    allv.sort(key=lambda x: x[0])
    r, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        rk = (i + j + 1) / 2
        r += rk * sum(x[1] for x in allv[i:j])
        i = j
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    A = load()
    pop, area = wb("SP.POP.TOTL"), wb("AG.SRF.TOTL.K2")
    nodes = sorted(A)
    ids = sorted(i for i in pop if i in area and any(i in A[n]["pop"] for n in nodes))
    print(len(nodes), "个节点", nodes, len(ids), "个国家")
    T = math.log10(1.2)
    R = []
    for i in ids:
        tr = {"pop": math.log10(pop[i] / 1e6), "area": math.log10(area[i]), "dens": math.log10(pop[i] / area[i])}
        have = [n for n in nodes if all(i in A[n][q] for q in tr)]
        if len(have) < 3:
            continue
        L = {n: {q: math.log10(A[n][q][i]) for q in tr} for n in have}
        for n in have:
            err = max(abs(L[n][q] - tr[q]) for q in tr)
            res = abs(L[n]["dens"] - (L[n]["pop"] + 6 - L[n]["area"]))
            peer = max(abs(L[n][q] - st.median(L[m][q] for m in have if m != n)) for q in tr)
            shared = max(abs(st.median(L[m][q] for m in have) - tr[q]) for q in tr) > T
            R.append({"i": i, "n": n, "wrong": err > T, "res": res, "peer": peer, "shared": shared})
    print(len(R), "个节点-国家；错的比例", round(st.mean(r["wrong"] for r in R), 3), "；其中共有错误（中位数也错）占", round(st.mean(r["shared"] for r in R if r["wrong"]), 3))
    # 合并：各自排名
    def rank(key):
        o = sorted(range(len(R)), key=lambda j: R[j][key])
        out = [0.0] * len(R)
        for pos, j in enumerate(o):
            out[j] = pos / len(R)
        return out
    rr, rp = rank("res"), rank("peer")
    for j, r in enumerate(R):
        r["both"] = max(rr[j], rp[j])
    def A_(key, sel=lambda r: True):
        s = [r for r in R if sel(r)]
        return auc([r[key] for r in s if r["wrong"]], [r[key] for r in s if not r["wrong"]])
    print(f"1. 约束 AUC {A_('res'):.3f}（≥ 0.70）")
    w = [r for r in R if r["wrong"]]
    print(f"2. 错了但约束残差 < 5%（自洽地错）：{st.mean(r['res'] < math.log10(1.05) for r in w):.0%}（≥ 30%）；没错但残差 > 20% 的（约束自己的误报）{st.mean(r['res'] > T for r in R if not r['wrong']):.1%}；错了且残差 > 20% {st.mean(r['res'] > T for r in w):.0%}")
    print(f"3. 同伴 AUC {A_('peer'):.3f}（≥ 0.70）；合并 AUC {A_('both'):.3f}（> 两者）")
    sh = [r for r in R if r["shared"]]
    no = [r for r in R if not r["wrong"] and not r["shared"]]
    print(f"4. 全体共有错误（{sum(r['wrong'] for r in sh)} 例）vs 没错（{len(no)} 例）：约束 AUC {auc([r['res'] for r in sh if r['wrong']], [r['res'] for r in no]):.3f}（≥ 0.60）；同伴 AUC {auc([r['peer'] for r in sh if r['wrong']], [r['peer'] for r in no]):.3f}（≤ 0.60）")
    print("按节点：" + "  ".join(f"{n}: 约束 {A_('res', lambda r: r['n'] == n):.2f}/同伴 {A_('peer', lambda r: r['n'] == n):.2f}" for n in nodes))
    rg = random.Random(0)
    ctr = sorted({r["i"] for r in R})
    by = collections.defaultdict(list)
    for r in R:
        by[r["i"]].append(r)
    bs = []
    for _ in range(300):
        s = [r for i in (rg.choice(ctr) for _ in ctr) for r in by[i]]
        bs.append(auc([r["res"] for r in s if r["wrong"]], [r["res"] for r in s if not r["wrong"]]))
    bs.sort()
    print(f"约束 AUC 按国家 bootstrap 95%：[{bs[7]:.3f}, {bs[292]:.3f}]")


if __name__ == "__main__":
    main()
