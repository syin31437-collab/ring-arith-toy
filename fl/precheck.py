"""映射预检（无攻击沙盒，可看真值，只调映射、不碰测试格子）：三条都过才能预注册。
  ① 诚实节点围着真值转：各题节点中位数相对真值的偏移，|中位| < h / 4
  ② 标准化后噪声接近正态：节点误差 |z| 的 95% 分位 ∈ [1.5, 3]、99% 分位 < 5
  ③ 秤的噪声估计准：秤误差的 MAD / sigma_c 的估计 ∈ [0.7, 1.4]
用法：~/gut-external/torch215-venv/bin/python precheck.py <dense>"""
import sys
import statistics as st
import torch
sys.path.insert(0, '.')
import flbench as F
from law_v6 import fit
dense = float(sys.argv[1])
v2 = len(sys.argv) > 2 and sys.argv[2] == "v2"
ok_all = True
for n, m, sd in [(n, m, sd) for sd in (0, 100, 200) for n, m in ((10, 32), (10, 512), (20, 32), (20, 512))]:
    c, _, _ = F.run(n, 0.0, "none", m, 88000 + sd + n + m, dense=dense, v2=v2)
    R = torch.stack([r for _, _, r, _ in c])
    P = torch.tensor([p for _, p, _, _ in c])
    off = R.median(1).values.median().item()
    q95, q99 = R.abs().flatten().quantile(.95).item(), R.abs().flatten().quantile(.99).item()
    mad = 1.4826 * (P - P.median()).abs().median().item()
    sc = fit(c[:300], [(True,) + x[1:] for x in c[300:]])["sig_c"]
    ok = abs(off) < 0.5 / 4 and 1.5 <= q95 <= 3 and q99 < 5 and 0.7 <= mad / sc <= 1.4
    ok_all &= ok
    print(f"n={n:2d} m={m:3d} 题 {len(c)}：①偏移 {off:+.3f}（< 0.125） ②|z| 95% {q95:.2f} 99% {q99:.2f} ③秤 MAD / 估计 {mad:.2f} / {sc:.2f} = {mad / sc:.2f}  → {'过' if ok else '不过'}")
print("映射预检", "全部通过" if ok_all else "没通过")
