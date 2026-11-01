"""对应四个评价指标：
  1 误差是否持续下降：同一条规律第一遍 vs 第二遍的稳态误差
  2 切换代价：切换后前 100 步的平均误差 ÷ 该段所有系统中最好的稳态误差（越小恢复越快）
  3 污染：坏节点期间，其余 5 个节点的误差比对照（无坏节点）高多少；系统输出被拉偏多少
  4 新规律：第 9 段（R5）的恢复速度和稳态误差
"""
import json
import statistics as st

from switch import BAD_FROM, BAD_NODE, BAD_TO, NEW_SEG, SEG

SYSTEMS = ["A", "B", "C", "C-trim"]
STARTS = [i * SEG for i in range(8)] + [8 * SEG]
LENS = [SEG] * 8 + [NEW_SEG]
NAMES = ["R1", "R2", "R3", "R4"] * 2 + ["R5"]


def steady(err, k):
    a, n = STARTS[k], LENS[k]
    return st.mean(err[a + n - 300:a + n])


def switch_cost(err, k, best):
    a = STARTS[k]
    return st.mean(err[a:a + 100]) / best


def med(xs):
    return st.median(xs)


def main():
    data = json.load(open("results.json"))
    n_seeds = len(data["A"])
    # 用无坏节点的对照跑来算 1/2/4，避免坏节点期间（落在第 5 段）干扰
    seg_steady = {s: [[steady(r["sys_clean"], k) for k in range(9)] for r in data[s]] for s in SYSTEMS}
    rec = {s: [] for s in SYSTEMS}
    for i in range(n_seeds):
        best = [min(seg_steady[s][i][k] for s in SYSTEMS) for k in range(9)]
        for s in SYSTEMS:
            rec[s].append([switch_cost(data[s][i]["sys_clean"], k, best[k]) for k in range(9)])

    print("== 1. 误差是否持续下降：同一规律第一遍 → 第二遍的稳态误差（中位数，20 种子） ==")
    for s in SYSTEMS:
        cells = []
        for k in range(4):
            a = med([x[k] for x in seg_steady[s]])
            b = med([x[k + 4] for x in seg_steady[s]])
            cells.append(f"{NAMES[k]} {a:.4f}→{b:.4f}")
        print(f"{s:7s} " + "   ".join(cells))

    print("\n== 2. 切换代价：切换后前 100 步平均误差 / 该段最好稳态误差（中位数） ==")
    print("        " + "  ".join(f"{NAMES[k]}#{k+1:<2d}" for k in range(1, 8)))
    for s in SYSTEMS:
        print(f"{s:7s} " + "  ".join(f"{med([x[k] for x in rec[s]]):6.1f}" for k in range(1, 8)))

    print("\n== 3. 坏节点污染（第 4500–4900 步，节点 0 输出 +3 且停止学习） ==")
    print("系统     其余5节点误差/对照   邻居(节点1,5)/对照   远端(节点2,3,4)/对照   系统输出误差/对照   坏后200步其余节点/对照")
    for s in SYSTEMS:
        def ratio(nodes, a, b):
            r = []
            for x in data[s]:
                bad = st.mean(st.mean(x["node"][t][j] for j in nodes) for t in range(a, b))
                ok = st.mean(st.mean(x["node_clean"][t][j] for j in nodes) for t in range(a, b))
                r.append(bad / ok)
            return med(r)
        others = [j for j in range(6) if j != BAD_NODE]
        sysr = med([st.mean(x["sys"][BAD_FROM:BAD_TO]) / st.mean(x["sys_clean"][BAD_FROM:BAD_TO]) for x in data[s]])
        print(f"{s:7s} {ratio(others, BAD_FROM, BAD_TO):14.2f}x {ratio([1, 5], BAD_FROM, BAD_TO):18.2f}x "
              f"{ratio([2, 3, 4], BAD_FROM, BAD_TO):20.2f}x {sysr:18.2f}x {ratio(others, BAD_TO, BAD_TO + 200):20.2f}x")

    print("\n== 4. 从未见过的新规律 R5 ==")
    for s in SYSTEMS:
        print(f"{s:7s} 切换代价 中位={med([x[8] for x in rec[s]]):5.1f}   稳态误差 中位={med([x[8] for x in seg_steady[s]]):.4f}")

    print("\n== 附：各段稳态误差（中位数，噪声下限 0.01） ==")
    for s in SYSTEMS:
        print(f"{s:7s} " + " ".join(f"{med([x[k] for x in seg_steady[s]]):.4f}" for k in range(9)))


if __name__ == "__main__":
    main()
