"""稀疏探测：每块每个节点 15% 概率出错，只用恰好一个节点在错的块来估计。
对比密集探测（所有节点同时出错）下 C-trim 的大偏差权重。"""
import statistics as st
import sys

import torch

from observer import N, ring_dist, weights

torch.set_num_threads(1)
system = sys.argv[1]
for eps in (0.1, 2.0):
    mats = torch.tensor([weights(system, seed, eps, 100, q=0.15, single=True) for seed in range(20)])
    W = mats.mean(0)
    by = {k: [] for k in range(4)}
    for i in range(N):
        for j in range(N):
            by[ring_dist(i, j)].append(W[i][j].item())
    print(system, f"eps={eps}", "  ".join(f"距离{k} {st.mean(v):+.4f}" for k, v in by.items()), flush=True)
