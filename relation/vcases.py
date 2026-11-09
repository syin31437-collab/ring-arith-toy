"""
v* 的分段情况表 + 留一节点敏感性（探索性，不是预注册检验；给论文附录"v* 的定义与边界"用）

对 297 题轮的每个节点 × 形式算 n_t、n_f，归到四格：
  A 正常阈值      n_f < 0 < n_t   v* = −n_f/(n_t−n_f) ∈ (0,1)
  B 恒可接受      n_f ≥ 0, n_t ≥ 0   任何 v 下期望增益 ≥ 0；截断记 0，原定义无定义/≤0
  C 恒不可接受    n_f ≤ 0, n_t ≤ 0（非全零）  任何 v 下期望增益 ≤ 0；截断记 1，原定义无定义/≥1
  D 反向          n_f > 0 > n_t   假版本帮忙、真版本伤人，阈值方向反转；截断记 0
留一节点：S1 v*(L3)−v*(cond2)（带回显 297、无回显 297）与 E1c v*(L3)−v*(one1)，每次去掉一个节点重算（截断与原定义）。

用法：~/gut-external/torch215-venv/bin/python vcases.py > analysis_vcases.txt
"""
import json
import os

import scale as X
import wording as W

HERE = os.path.dirname(os.path.abspath(__file__))
m = W.m
OLD6 = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns", "qwen14", "llama8"]
NEW = ["gemma12", "mistral7"]


def ntnf(R, a, s, p):
    X_ = [r for r in R if r["a"] == a and r["id"] in s]
    bl = [r for r in X_ if r["p"] == "blank"]
    e = m(not r["right"] for r in bl)
    K0, B0 = m(r["ok"] for r in bl if r["right"]), m(r["ok"] for r in bl if not r["right"])
    g = lambda q, right: m(r["ok"] for r in X_ if r["p"] == q and r["right"] == right)
    nt = e * (g(p + "t", False) - B0) - (1 - e) * (K0 - g(p + "t", True))
    nf = e * (g(p + "f", False) - B0) - (1 - e) * (K0 - g(p + "f", True))
    return nt, nf


def case(nt, nf):
    if nf < 0 < nt:
        return "A"
    if nf >= 0 and nt >= 0:
        return "B"
    if nf <= 0 and nt <= 0:
        return "C"
    return "D"


def table(R, nodes, forms, lab):
    ids = {r["id"] for r in R if r["p"] == "blank"}
    print(f"\n[{lab}] 题 {len(ids)}；每格 n_t / n_f / 格 / 截断 v* / 原定义 v*")
    cnt = {}
    for a in nodes:
        row = []
        for p in forms:
            nt, nf = ntnf(R, a, ids, p)
            if nt != nt or nf != nf:
                row.append(f"{p}: —")
                continue
            c = case(nt, nf)
            cnt[c] = cnt.get(c, 0) + 1
            u = -nf / (nt - nf) if nt != nf else float("nan")
            row.append(f"{p}: {nt:+.3f}/{nf:+.3f}/{c}/{X.v0(R, a, ids, p):.2f}/{u:.2f}")
        print(f"  {a:11s} " + "  ".join(row))
    print("  四格计数 " + " ".join(f"{k} {cnt.get(k, 0)}" for k in "ABCD"))


def loo(R, nodes, p1, p2, lab):
    ids = sorted({r["id"] for r in R if r["p"] == "blank"})
    print(f"\n[留一节点] {lab} v*({p1}) − v*({p2})")
    for drop in [None] + nodes:
        ns = [a for a in nodes if a != drop]
        for name, vf in (("截断", X.v0), ("原定义", W.vstar)):
            vs = {a: (vf(R, a, set(ids), p1), vf(R, a, set(ids), p2)) for a in ns}
            ok = [a for a in ns if all(x == x for x in vs[a])]
            fn = lambda s: m(x for x in (vf(R, a, set(s), p1) - vf(R, a, set(s), p2) for a in ok) if x == x)
            v = m(vs[a][0] - vs[a][1] for a in ok)
            lo, hi = W.boot(fn, ids, n=1000)
            print(f"  去掉 {drop or '—':11s} {name:4s} {v:+.2f} [{lo:+.2f},{hi:+.2f}] {len(ok)} 节点")


if __name__ == "__main__":
    Rs = [r for r in json.load(open(os.path.join(HERE, "rows_s.json"))) if r["a"] in OLD6]
    Rm = json.load(open(os.path.join(HERE, "rows_sm.json")))
    Rf = [r for r in json.load(open(os.path.join(HERE, "rows_fu.json"))) if r["p"] in ("L3nt", "L3nf")]
    Rp = json.load(open(os.path.join(HERE, "rows_rp.json")))
    R6 = Rs + [r for r in Rm if r["p"] in ("one1t", "one1f")] + Rf
    table(R6, OLD6, ["set2", "cond2", "L3", "L3c", "one1", "L3n"], "297 题轮 + 控制轮，带回显，六节点")
    Rne = [dict(r, p=r["p"][:-3]) for r in Rm if r["p"].endswith("_ne")]
    table(Rne, OLD6, ["set2", "cond2", "L3"], "无回显，六节点")
    table(Rp, NEW, ["set2", "cond2", "L3", "L3c", "one1", "L3n"], "复现轮，两新节点")
    loo(Rs, OLD6, "L3", "cond2", "S1 带回显 297")
    loo(Rne, OLD6, "L3", "cond2", "E2d 无回显 297")
    loo(R6, OLD6, "L3", "one1", "E1c 带回显 297")
