"""题型对照的诊断（看过结果后的探索，不改 analysis_f.txt 的判决）。

T 预注册要报的"去掉带工具的回复"，analyze 没做：这里直接去掉有工具记录的节点（Grok、Codex）。
F 集合 vs 前提：每个条件 × 层的 K(set2f)、K(L3f) 节点均值。
X 上一轮开放题的 K(set2f) 按任务类分开：Sonnet 94% 那个数字来自哪。
"""
import json

import formtest as F
import stage2 as S2

R = json.load(open("rows_f.json"))
m = F.m
F.verdicts(R, "去掉 Grok、Codex", [a for a in F.NODES if a not in ("grok", "codex_ns")])
print("\nF 节点均值 K(set2f) / K(L3f) / 差")
for c in "JMO":
    for s in "HN":
        a_ = m(F.K(R, cond=c, stratum=s, a=a, p="set2f") for a in F.NODES)
        b_ = m(F.K(R, cond=c, stratum=s, a=a, p="L3f") for a in F.NODES)
        print(f"  {c}×{s} {a_:.2f} {b_:.2f} {b_ - a_:+.2f}")
print("\nX 上一轮开放题 K(set2f)、K(L3f)（第一次答对的行数）")
X = json.load(open("rows_xa.json"))
for fam in ("code", "qa", "plan", "case"):
    print(" ", fam, " ".join(f"{a} {S2.K([r for r in X if r['a'] == a and r['family'] == fam], 'set2f'):.0%}/{S2.K([r for r in X if r['a'] == a and r['family'] == fam], 'L3f'):.0%}({sum(r['right'] for r in X if r['a'] == a and r['family'] == fam and r['p'] == 'set2f')})" for a in F.NODES))
