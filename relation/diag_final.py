"""探索性诊断（事后，不进判决）：v* 在假版本零杀伤时算不出；把 net_f ≥ 0 记 v*=0（永远划算）再看三候选。"""
import json, sys
sys.path.insert(0, ".")
import wording as W
R = json.load(open("rows_final.json"))["j"]
m = W.m
def v0(R, a, p):
    X = [r for r in R if r["a"] == a]
    bl = [r for r in X if r["p"] == "blank"]
    e = m(not r["right"] for r in bl)
    K0, B0 = m(r["ok"] for r in bl if r["right"]), m(r["ok"] for r in bl if not r["right"])
    g = lambda q, right: m(r["ok"] for r in X if r["p"] == q and r["right"] == right)
    nt = e * (g(p + "t", False) - B0) - (1 - e) * (K0 - g(p + "t", True))
    nf = e * (g(p + "f", False) - B0) - (1 - e) * (K0 - g(p + "f", True))
    if nf >= 0: return 0.0
    if nt <= 0: return 1.0
    return -nf / (nt - nf)
nodes = ["kimi", "grok", "ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns"]
print("v*（零杀伤记 0，零收益记 1）：")
for p in ["set2", "cond2", "L3", "L3c"]:
    vs = {a: v0(R, a, p) for a in nodes}
    print(f"  {p:6s} 均值 {m(vs.values()):.2f}  " + " ".join(f"{a} {v:.2f}" for a, v in vs.items()))
print("假前提 vs 假集合+出口，好节点守住率 K：")
for a in nodes:
    X = [r for r in R if r["a"] == a and r["right"]]
    print(f"  {a:12s} " + " ".join(f"{p} {m(r['ok'] for r in X if r['p']==p):.0%}" for p in ["set2f", "cond2f", "L3f", "L3cf"]))
