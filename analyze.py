"""读 traj_*.json：最终分布、坏种子在哪坏、是突然还是逐步。"""
import json
import statistics as st
import sys

ARMS = ["A", "D", "A+sparse", "D+sparse", "D+gate"]
BAD = 1.0          # 泛化区相对误差 > 100% 算炸
GOOD = 0.1


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def shape(traj):
    """坏的方式：从轨迹最低点到终点之间，最大的单段(100步)上升占总上升的比例。"""
    ood = [r["ood"] for r in traj]
    i_min = min(range(len(ood)), key=ood.__getitem__)
    rise = ood[-1] - ood[i_min]
    jumps = [ood[i + 1] - ood[i] for i in range(i_min, len(ood) - 1)]
    big = max(jumps) if jumps else 0.0
    i_big = i_min + jumps.index(big) if jumps else i_min
    return {
        "min": ood[i_min], "min_step": traj[i_min]["step"], "final": ood[-1],
        "jump_frac": big / rise if rise > 0 else 0.0, "jump_step": traj[i_big + 1]["step"] if jumps else None,
    }


def main():
    data = {a: json.load(open(f"traj_{a}.json")) for a in ARMS if open_ok(a)}
    print("== 最终泛化误差分布 ==")
    for a, runs in data.items():
        f = [r["traj"][-1]["ood"] for r in runs]
        print(f"{a:9s} n={len(f)} 中位={st.median(f):.3f} 四分位=[{q(f,.25):.3f},{q(f,.75):.3f}] "
              f"好(<{GOOD})={sum(x < GOOD for x in f)} 炸(>{BAD})={sum(x > BAD for x in f)} 最坏={max(f):.2f}")

    for a in ("D", "D+sparse", "D+gate"):
        if a not in data:
            continue
        print(f"\n== {a}：按终值排序的每个种子 ==")
        print("seed   final   min@step     最大单段上升占比@step   最终分段误差(10-15/15-20/20-25/25-30)  终点分歧")
        rows = sorted(data[a], key=lambda r: r["traj"][-1]["ood"])
        for r in rows:
            s = shape(r["traj"])
            last = r["traj"][-1]
            bins = "/".join(f"{last[k]:.2f}" for k in ("ood_10_15", "ood_15_20", "ood_20_25", "ood_25_30"))
            print(f"{r['seed']:4d}  {s['final']:6.3f}  {s['min']:.3f}@{s['min_step']:<5d}  "
                  f"{s['jump_frac']:.2f}@{s['jump_step']}   {bins:30s}  {last['ood_consistency']:.4f}")

    if len(sys.argv) > 1:   # 打印指定种子的完整轨迹：python analyze.py D 2
        arm, seed = sys.argv[1], int(sys.argv[2])
        r = next(r for r in data[arm] if r["seed"] == seed)
        print(f"\n== {arm} seed {seed} 轨迹 ==")
        for row in r["traj"]:
            print(f"{row['step']:5d} id={row['id']:.3f} ood={row['ood']:.3f} cons={row.get('ood_consistency', 0):.4f} "
                  + " ".join(f"{row[k]:.2f}" for k in ("ood_10_15", "ood_15_20", "ood_20_25", "ood_25_30")))


def open_ok(a):
    try:
        open(f"traj_{a}.json").close()
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    main()
