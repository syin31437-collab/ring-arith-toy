"""第四阶段事后诊断（未预注册）：① 去掉推理对同一模型的影响；② 只看读得懂集合的节点，ε 和盈亏点；③ 智力下降时趋近投影的程度。"""
import json

import analyze as AN
import stage2 as S2
import stage4 as S4

R4 = json.load(open("rows_s4.json"))
RS = json.load(open("rows_s2.json"))
K, B = S2.K, S2.B
R = R4 + [r for r in RS if r["a"] in S4.TOP2]
print("一、同一个 Flash：有理由和置信度 vs 只回一个字母")
for a, X in (("ds_flash_a（有理由）", [r for r in R if r["a"] == "ds_flash_a"]), ("flash_short（只回字母）", [r for r in R if r["a"] == "flash_short"])):
    print(f"  {a}: K set2f {K(X, 'set2f'):.0%} set2f_r {K(X, 'set2f_r'):.0%} L3f {K(X, 'L3f'):.0%} L4f {K(X, 'L4f'):.0%} ｜ B set2t {B(X, 'set2t'):.0%} L3t {B(X, 'L3t'):.0%} L4t {B(X, 'L4t'):.0%}")
print("\n二、读得懂字母集合的节点（K(set2f) ≤ 50% 或第二阶段 LLM）：ε × v*，自然 + 困难分开")
for a in ("codex_ns", "claude_haiku", "ds_flash_a", "ds_pro_a", "flash_short", "flash_hot", "flash_m50", "flash_m80", "lex_proj", "rand_proj"):
    X = [r for r in R if r["a"] == a]
    n, h = [r for r in X if not r["hard"]], [r for r in X if r["hard"]]
    e = lambda y: 1 - AN.mean(r["right"] for r in y if r["p"] == "blank")
    print(f"  {a:13s} 自然 ε {e(n):.0%} v* {S4.vs(n):.2f} ｜ 困难 ε {e(h):.0%} v* {S4.vs(h):.2f}")
print("\n三、离投影多远（集合外的节点跳进集合的比例；对错节点合并）")
for a in ("codex_ns", "claude_haiku", "ds_flash_a", "ds_pro_a", "flash_short", "flash_m50", "flash_m80", "rand_proj", "lex", "emb"):
    X = [r for r in R if r["a"] == a]
    print(f"  {a:13s} 真集合 {AN.mean(r['adopt'] for r in X if r['p'] == 'set2t' and r['conflict']):4.0%} 假集合 {AN.mean(r['adopt'] for r in X if r['p'] == 'set2f' and r['conflict']):4.0%} ｜ 集合内仍不动 真 {AN.mean(not r['changed'] for r in X if r['p'] == 'set2t' and r['conflict'] is False):4.0%} 假 {AN.mean(not r['changed'] for r in X if r['p'] == 'set2f' and r['conflict'] is False):4.0%}")
