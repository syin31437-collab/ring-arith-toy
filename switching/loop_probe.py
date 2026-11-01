"""老失败"自己当答案"（labels.py）的机制检验：它是不是"错误变成共享的"（ζ → 1）？

loop.py 里没有数值约束（裁判靠学出来的信任表），ρ 没有定义，三比值曲线不能直接套。
理论给出的可证伪说法：节点拿系统输出当答案学 → 节点的错误跟着输出走 → 错误变成节点之间共享的。
预注册预测（11-01，运行前写下）：
  self / nodes（节点拿输出学）：随时间 节点间错误相关 → > 0.9；分散度 / 误差 → 趋近 0（节点一致但都错）
  only / agree（节点不学或只在一致时学）：相关保持 < 0.5，分散度 / 误差不塌
⚠️ 第一版（无故障）设计错了：爆炸只发生在有故障的运行里；而且 12 个节点共用一个 MLP（只是视野不同），
   错误相关在所有模式里都是 0.93–1.00，分辨不了。"only/agree 相关 < 0.5"落空是环境结构造成的，不是理论的问题。
第二版预注册（运行前）：有故障（loop.fault_plan）时
  self / nodes：误差随时间成数量级增长，分散度/误差 → 0（一起错、越错越一致）
  only / agree：误差不成数量级增长，分散度/误差 不塌
测量（每 500 步一格，答案比例 1%，有故障；只看节点自己的预测，不含故障偏移）：
  误差    = 各节点预测 − 真值 的 RMS
  分散度  = 各节点预测之间的标准差（RMS）
  相关    = 各节点误差两两相关的平均

用法：~/gut-external/torch215-venv/bin/python loop_probe.py run MODE SEED  /  report
"""
import glob
import json
import statistics as st
import sys

import torch

from labels import mask
from loop import fault_plan, run

MODES = ["self", "nodes", "agree", "only"]


def probe(mode, seed):
    torch.set_num_threads(1)
    tr = []
    run("PG", seed, fault_plan(seed), mask(0.01, seed), mode, trace=tr)
    out = []
    for lo in range(0, 9500, 500):
        P = torch.stack([p for p, _ in tr[lo:lo + 500]])
        y = torch.tensor([y for _, y in tr[lo:lo + 500]])
        E = P - y[:, None]
        C = torch.nan_to_num(torch.corrcoef(E.T), nan=0.0)
        off = C[~torch.eye(C.shape[0], dtype=torch.bool)]
        out.append({"lo": lo, "err": E.pow(2).mean().sqrt().item(), "spread": P.std(1).pow(2).mean().sqrt().item(),
                    "corr": off.mean().item()})
    json.dump(out, open(f"probe_{mode}_{seed}.json", "w"))


def report():
    print("模式    步数段        误差      分散度   分散度/误差   节点间错误相关   （4 个种子中位数）")
    for mode in MODES:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"probe_{mode}_*.json"))]
        for k in (0, 2, 4, 8, 12, 18):
            c = [r[k] for r in runs]
            m = lambda key: st.median(x[key] for x in c)
            print(f"{mode:6s}  {c[0]['lo']:5d}-{c[0]['lo'] + 499:<5d}  {m('err'):9.3g}  {m('spread'):9.3g}  "
                  f"{st.median(x['spread'] / x['err'] for x in c):9.3f}   {m('corr'):10.2f}")


if __name__ == "__main__":
    probe(sys.argv[2], int(sys.argv[3])) if sys.argv[1] == "run" else report()
