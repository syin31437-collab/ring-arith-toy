"""第三阶段事后诊断（未预注册）。"""
import difflib
import json
import os

import analyze as AN
import stage2 as S2
import stage3 as S3

R = json.load(open("rows_s3.json"))
items = {x["id"]: x for x in json.load(open(S3.ITEMS))}
m, K, B = AN.mean, S2.K, S2.B


def real_sim(r):
    f = json.load(open(S3.first_path(r["id"], r["a"])))["text"]
    s = json.load(open(S3.second_path(items[r["id"]], r["a"], r["p"])))["text"]
    return difflib.SequenceMatcher(None, S3.strip_conf(f)[:3000], S3.strip_conf(s)[:3000]).ratio()


for r in R:
    r["rsim"] = real_sim(r)
print("一、O5 复查：quick_ratio 平均 vs ratio 平均；真相似度 < 0.7 的比例（判分翻转 = 对错变了）")
forms = {"blank": ("blank", "blank"), "L0": ("L0t", "L0t"), "L2": ("L2t", "L2f"), "L3": ("L3t", "L3f"), "L4": ("L4t", "L4f"), "set2": ("set2t", "set2f"), "set2_r": ("set2t_r", "set2f_r")}
vals, truth = {}, []
for nm, (pt, pf) in forms.items():
    f = lambda p, s: m(r["rsim"] < .7 for r in R if r["p"] == p and r["right"] == s)
    t, fa = (f(pt, True) + f(pt, False)) / 2, (f(pf, True) + f(pf, False)) / 2
    q = m(r["sim"] for r in R if r["p"] in (pt, pf))
    rs = m(r["rsim"] for r in R if r["p"] in (pt, pf))
    vals[nm] = (t + fa) / 2
    if nm not in ("blank", "L0"):
        truth.append(abs(t - fa))
    print(f"  {nm:7s} quick {q:.2f} ratio {rs:.2f} ｜ 真句动 {t:.0%} 假句动 {fa:.0%}")
core = {k: v for k, v in vals.items() if k not in ("blank", "L0")}
print(f"  O5 用真相似度重算：极差 {max(core.values()) - min(core.values()):.2f}，真假差均值 {sum(truth) / len(truth):.2f}")
print("\n二、按第一次对错分开的真相似度 < 0.7（有没有动）：")
for nm, (pt, pf) in forms.items():
    if nm in ("blank", "L0"):
        continue
    print(f"  {nm:7s} 对节点 真 {m(r['rsim'] < .7 for r in R if r['p'] == pt and r['right']):.0%} 假 {m(r['rsim'] < .7 for r in R if r['p'] == pf and r['right']):.0%} ｜ 错节点 真 {m(r['rsim'] < .7 for r in R if r['p'] == pt and not r['right']):.0%} 假 {m(r['rsim'] < .7 for r in R if r['p'] == pf and not r['right']):.0%}")
print("\n三、O4 精确值：")
Cp = [r for r in R if r["a"] in S3.CAPABLE]
print(f"  K(set2f_r) {K(Cp, 'set2f_r'):.4f}（n={sum(r['p'] == 'set2f_r' and r['right'] for r in Cp)}）；区间 {AN.boot(Cp, lambda y: K(y, 'set2f_r'), 500)}；B(set2t_r) {B(Cp, 'set2t_r'):.3f} 区间 {AN.boot(Cp, lambda y: B(y, 'set2t_r'), 500)}；K(blank) {K(Cp, 'blank'):.3f}")
print("\n四、不相关投放 L0t 是否伤对节点（K(blank) − K(L0t)）：")
for lab, X in (("全部", R), ("自然", [r for r in R if not r["hard"]]), ("困难", [r for r in R if r["hard"]])):
    print(f"  {lab}: {K(X, 'blank') - K(X, 'L0t'):+.3f} 区间 {tuple(round(v, 3) for v in AN.boot(X, lambda y: K(y, 'blank') - K(y, 'L0t'), 500))}（n={sum(r['p'] == 'L0t' and r['right'] for r in X)}）")
for f_ in S3.FAM:
    X = [r for r in R if r["family"] == f_]
    print(f"    {f_:5s} K blank {K(X, 'blank'):.0%} L0t {K(X, 'L0t'):.0%}（n={sum(r['p'] == 'L0t' and r['right'] for r in X)}）")
print("\n五、任务能否靠自身核对：对节点 K（排班题可以逐条验约束；MBPP 看不到隐藏测试；问答只能凭记忆）")
for s in ("plan9", "humaneval", "mbpp", "simpleqa", "truthfulqa", "professional_law", "clinical"):
    X = [r for r in R if r["source"] == s]
    print(f"  {s:18s} n对 {sum(r['p'] == 'blank' and r['right'] for r in X):3d} n错 {sum(r['p'] == 'blank' and not r['right'] for r in X):3d} ｜ L3f {K(X, 'L3f'):4.0%} L4f {K(X, 'L4f'):4.0%} set2f {K(X, 'set2f'):4.0%} set2f_r {K(X, 'set2f_r'):4.0%} ｜ B L3t {B(X, 'L3t'):4.0%} L4t {B(X, 'L4t'):4.0%} set2t_r {B(X, 'set2t_r'):4.0%}")
print("\n六、接收者 × 不同投放的 K / B（对比第二阶段）")
S = json.load(open("rows_s2.json"))
for a in S3.AGENTS:
    x2, x3 = [r for r in S if r["a"] == a], [r for r in R if r["a"] == a]
    print(f"  {a:13s} 选择题 K(L3f) {K(x2, 'L3f'):.0%} K(L4f) {K(x2, 'L4f'):.0%} K(set2f) {K(x2, 'set2f'):.0%} K(set2f_r) {K(x2, 'set2f_r'):.0%} B(L3t) {B(x2, 'L3t'):.0%} B(L4t) {B(x2, 'L4t'):.0%} ｜ 开放 K(L3f) {K(x3, 'L3f'):.0%} K(L4f) {K(x3, 'L4f'):.0%} K(set2f) {K(x3, 'set2f'):.0%} K(set2f_r) {K(x3, 'set2f_r'):.0%} B(L3t) {B(x3, 'L3t'):.0%} B(L4t) {B(x3, 'L4t'):.0%}")
