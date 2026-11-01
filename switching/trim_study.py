"""深挖 C-trim：和 A（独立，即 B 的节点）、C（平均值耦合）对比三件事。

  1 瞎节点：每条规律只用到某几个历史值，节点看到的占比 = 覆盖度（0 = 全瞎）。
            按覆盖度分组看各节点稳态误差，耦合有没有帮到瞎的节点。
  2 极限：坏 1 个、坏 2 个相邻、坏 2 个隔开、坏 3 个隔开、坏 3 个相邻，健康节点被污染多少。
  3 汇总：系统输出用平均 vs 用中位数，坏节点期间的误差 / 对照。

用法：python trim_study.py run SYSTEM [N_SEEDS]   →  trim_SYSTEM.json
      python trim_study.py report
"""
import json
import statistics as st
import sys

import torch

from switch import BAD_FROM, BAD_TO, N, NEW_SEG, SEG, VIEWS, run

USED = {"R1": {0, 1}, "R2": {2, 3}, "R3": {1}, "R4": {0, 4, 5}, "R5": {0, 3}}
NAMES = ["R1", "R2", "R3", "R4"] * 2 + ["R5"]
STARTS = [i * SEG for i in range(9)]
LENS = [SEG] * 8 + [NEW_SEG]
SCENARIOS = {"坏1个": (0,), "坏2个相邻": (0, 1), "坏2个隔开": (0, 3), "坏3个隔开": (0, 2, 4), "坏3个相邻": (0, 1, 2)}
SYSTEMS = ["A", "C", "C-trim"]


def coverage(node, rule):
    view = set(VIEWS[node].tolist())
    return len(view & USED[rule]) / len(USED[rule])


def window_stats(preds, bad):
    """坏节点窗口内：健康节点误差（分邻居/远端）、平均汇总误差、中位数汇总误差。"""
    healthy = [j for j in range(N) if j not in bad]
    near = [j for j in healthy if any((j - b) % N in (1, N - 1) for b in bad)]
    far = [j for j in healthy if j not in near]
    acc = {"near": [], "far": [], "mean_out": [], "median_out": []}
    for shown, y in preds[BAD_FROM:BAD_TO]:
        e = [(v - y) ** 2 for v in shown]
        if near:
            acc["near"].append(st.mean(e[j] for j in near))
        if far:
            acc["far"].append(st.mean(e[j] for j in far))
        acc["mean_out"].append((st.mean(shown) - y) ** 2)
        acc["median_out"].append((st.median(shown) - y) ** 2)
    return {k: (st.mean(v) if v else None) for k, v in acc.items()}


def do_run(system, n_seeds):
    torch.set_num_threads(1)
    out = []
    for seed in range(n_seeds):
        clean = []
        run(system, seed, corrupt=False, preds=clean)
        steady = []                      # [段][节点]
        out_steady = []                  # [段] -> (平均汇总, 中位数汇总)
        for k in range(9):
            a, n = STARTS[k], LENS[k]
            seg = clean[a + n - 300:a + n]
            steady.append([st.mean((s[j] - y) ** 2 for s, y in seg) for j in range(N)])
            out_steady.append((st.mean((st.mean(s) - y) ** 2 for s, y in seg),
                               st.mean((st.median(s) - y) ** 2 for s, y in seg)))
        rec = {"seed": seed, "steady": steady, "out_steady": out_steady, "scen": {}}
        for name, bad in SCENARIOS.items():
            dirty = []
            run(system, seed, corrupt=True, bad=bad, preds=dirty)
            rec["scen"][name] = {"dirty": window_stats(dirty, bad), "clean": window_stats(clean, bad)}
        out.append(rec)
        print(system, seed, flush=True)
    with open(f"trim_{system}.json", "w") as f:
        json.dump(out, f)


def report():
    data = {s: json.load(open(f"trim_{s}.json")) for s in SYSTEMS}
    med = st.median

    print("== 1. 按覆盖度分组的节点稳态误差（所有段、所有种子的中位数；噪声下限 0.01） ==")
    print("系统      全瞎(0)   部分看到   全看到(1)")
    for s in SYSTEMS:
        groups = {"blind": [], "part": [], "full": []}
        for r in data[s]:
            for k in range(9):
                for j in range(N):
                    c = coverage(j, NAMES[k])
                    g = "blind" if c == 0 else "full" if c == 1 else "part"
                    groups[g].append(r["steady"][k][j])
        print(f"{s:7s}  {med(groups['blind']):.4f}    {med(groups['part']):.4f}     {med(groups['full']):.4f}")

    print("\n   逐条规律：全瞎节点的稳态误差（第二遍 + R5）")
    print("系统     " + "  ".join(f"{NAMES[k]}#{k+1}" for k in range(4, 9)))
    for s in SYSTEMS:
        cells = []
        for k in range(4, 9):
            blind = [j for j in range(N) if coverage(j, NAMES[k]) == 0]
            cells.append(f"{med([st.mean(r['steady'][k][j] for j in blind) for r in data[s]]):.4f}" if blind else "  —   ")
        print(f"{s:7s}  " + "  ".join(cells))

    print("\n== 2. 极限：坏节点期间健康节点误差 / 对照（中位数） ==")
    print("场景         " + "".join(f"{s+' 邻居':>12s}{s+' 远端':>12s}" for s in SYSTEMS))
    for name in SCENARIOS:
        cells = []
        for s in SYSTEMS:
            for part in ("near", "far"):
                vals = [r["scen"][name]["dirty"][part] / r["scen"][name]["clean"][part]
                        for r in data[s] if r["scen"][name]["clean"][part]]
                cells.append(f"{med(vals):11.2f}x" if vals else "          —")
        print(f"{name:10s} " + "".join(cells))

    print("\n== 3. 汇总方式：坏节点期间系统输出误差 / 对照（中位数） ==")
    print("场景         " + "".join(f"{s+' 平均':>12s}{s+' 中位数':>13s}" for s in SYSTEMS))
    for name in SCENARIOS:
        cells = []
        for s in SYSTEMS:
            for agg in ("mean_out", "median_out"):
                vals = [r["scen"][name]["dirty"][agg] / r["scen"][name]["clean"][agg] for r in data[s]]
                cells.append(f"{med(vals):11.2f}x")
        print(f"{name:10s} " + "".join(cells))

    print("\n   没有坏节点时，两种汇总方式的稳态误差（所有段中位数）")
    for s in SYSTEMS:
        m = med([r["out_steady"][k][0] for r in data[s] for k in range(9)])
        d = med([r["out_steady"][k][1] for r in data[s] for k in range(9)])
        print(f"{s:7s}  平均 {m:.4f}   中位数 {d:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        report()
