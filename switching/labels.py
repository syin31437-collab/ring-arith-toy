"""只揭晓一部分真实答案：100% / 10% / 1% / 0.1%，三种"没答案时怎么办"，看性能怎么退化。

汇总方式固定用 PG（loop.py）。
  整体误差：故障运行全程（第 2000 步起）输出误差 ÷ "全部答案 + 无故障"的基准
  过半故障漏过率：过半节点坏的步上，输出相对"同样答案比例 + 同样做法、无故障"的对照偏移 > 1.5
                  （只看故障造成的额外伤害，不把节点本身学得差算进去）

用法：python labels.py run MODE [N_SEEDS]  →  labels_MODE.json
      python labels.py report
"""
import json
import statistics as st
import sys

import torch

from loop import BIAS, N, fault_plan, run

RATES = [1.0, 0.1, 0.01, 0.001]
MODES = {"only": "只学真答案", "self": "全靠自己判断", "nodes": "节点自学·裁判只认真答案",
         "agree": "只在大家一致时自学·裁判只认真答案",
         "judge": "诊断：节点永远有真答案，只有裁判缺答案"}


def mask(rate, seed):
    g = torch.Generator().manual_seed(70000 + seed)
    return (torch.rand(9500, generator=g) < rate).tolist()


def do_run(mode, n_seeds):
    torch.set_num_threads(1)
    out = {}
    for seed in range(n_seeds):
        plan = fault_plan(seed)
        ref, _ = run("PG", seed, None)                   # 全部答案、无故障
        for rate in RATES:
            lab = mask(rate, seed)
            clean, _ = run("PG", seed, None, lab, mode)
            dirty, _ = run("PG", seed, plan, lab, mode)
            from switch import make_series
            ys = [y for _, y in make_series(seed)[0]]
            qual = st.mean((dirty[t] - ys[t]) ** 2 for t in range(2000, 9500)) / \
                st.mean((ref[t] - ys[t]) ** 2 for t in range(2000, 9500))
            res = {"quality": qual, "n_labels": sum(lab)}
            for name, lo in (("leak_all", 2000), ("leak_late", 7500)):
                major = [t for t in range(lo, 9500) if plan[t][0].sum() * 2 > N]
                res[name] = sum(abs(dirty[t] - clean[t]) > BIAS / 2 for t in major) / len(major)
            out.setdefault(str(rate), []).append(res)
        print(mode, seed, flush=True)
    with open(f"labels_{mode}.json", "w") as f:
        json.dump(out, f)


def report():
    med = st.median
    for mode, name in MODES.items():
        data = json.load(open(f"labels_{mode}.json"))
        print(f"\n== {name} ==")
        print("答案比例  答案个数   整体误差/基准   过半故障漏过(全程)   过半故障漏过(最后2000步)")
        for rate in RATES:
            r = data[str(rate)]
            print(f"{rate:7.1%}  {int(med([x['n_labels'] for x in r])):6d}   {med([x['quality'] for x in r]):12.2f}x"
                  f"   {med([x['leak_all'] for x in r]):16.1%}   {med([x['leak_late'] for x in r]):20.1%}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        report()
