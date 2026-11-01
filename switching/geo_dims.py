"""实验一补充分析：穿透率到底由几个比值决定？
比较几种"分箱方式"能解释多少方差（箱内平均当预测）：
  1 个比值：ρ = δ/σ_eff
  2 个比值：ρ 和 κ = δ/σ_s（错误把坏的那团和好的那团拉开了几倍传感器噪声）
  对照：原始变量 (δ, 噪声倍数)、(δ, 过程噪声)，以及三个原始变量全用（同配置不同种子的平均 = 噪声上限）"""
import json
import math
import statistics as st

rows = [r for p in (0.05, 0.1, 0.2) for r in json.load(open(f"geo1_{p}.json"))]
rows = [r for r in rows if r["sigma_eff"] > 0 and not math.isnan(r["pass"])]
ys = [r["pass"] for r in rows]
SST = sum((y - st.mean(ys)) ** 2 for y in ys)


def r2(keys):
    groups = {}
    for k, y in zip(keys, ys):
        groups.setdefault(k, []).append(y)
    mean = {k: st.mean(v) for k, v in groups.items()}
    return 1 - sum((y - mean[k]) ** 2 for k, y in zip(keys, ys)) / SST, len(groups)


def b(x, w):
    return round(x / w)


lr = [math.log10(r["delta"] / r["sigma_eff"]) for r in rows]
lk = [math.log10(r["delta"] / (0.175 * r["scale"])) for r in rows]
ld = [math.log10(r["delta"]) for r in rows]
for name, keys in [
    ("ρ 一个比值（0.1 宽的箱）", [b(x, 0.1) for x in lr]),
    ("ρ + κ 两个比值（0.25×0.25 的箱）", [(b(x, 0.25), b(y, 0.25)) for x, y in zip(lr, lk)]),
    ("δ + 噪声倍数（原始变量）", [(r["delta"], r["scale"]) for r in rows]),
    ("δ + 过程噪声（原始变量）", [(r["delta"], r["proc"]) for r in rows]),
    ("三个原始变量全用（噪声上限）", [(r["delta"], r["scale"], r["proc"]) for r in rows]),
]:
    v, n = r2(keys)
    print(f"{name:28s} R² = {v:.3f}   （{n} 个箱）")
