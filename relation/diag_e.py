"""外部依据对照的诊断（看过结果后的探索，不改 analysis_e.txt 的判决）。

Q2 不过：E 条件丢答案的行 94% 引用是真原文。那它们引的是什么？
S 丢答案行里引用句和第一次引用句相同的比例（同一句真原文，换了答案）；引用句里含不含新选的错选项的字。
X 丢答案的例子。
"""
import json
import os

import evid as V

R = json.load(open("rows_e.json"))
items = {x["id"]: x for x in json.load(open(V.ITEMS))}
pay = json.load(open(V.PAY))


def quote(c, p, i, a):
    f = os.path.join(V.OUT, f"{c}_{p}_{i}_{a}.json" if c else f"s1_{i}_{a}.json")
    u = json.load(open(f))["answers"].get(i, {})
    return u.get("quote", "") if isinstance(u, dict) else ""


lost = [r for r in R if r["cond"] == "E" and r["p"] == "set2f" and r["right"] and not r["ok"]]
kept = [r for r in R if r["cond"] == "E" and r["p"] == "set2f" and r["right"] and r["ok"]]
print("S 丢答案行", len(lost), "保住行", len(kept))
for lab, rs in (("丢答案", lost), ("保住", kept)):
    same = [V.norm(quote("E", "set2f", r["id"], r["a"])).strip() == V.norm(quote(None, None, r["id"], r["a"])).strip() for r in rs]
    print(f"  {lab}：引用句和第一次完全相同 {V.m(same):.0%}")
for a in V.NODES:
    rs = [r for r in lost if r["a"] == a]
    same = [V.norm(quote("E", "set2f", r["id"], r["a"])).strip() == V.norm(quote(None, None, r["id"], r["a"])).strip() for r in rs]
    print(f"    {a:12s} 丢 {len(rs):2d} 同句 {V.m(same):.0%}" if rs else f"    {a:12s} 丢 0")
print("\nX 例子（丢答案、引用为真）")
for r in [r for r in lost if r["q"] == "real"][:4]:
    it = items[r["id"]]
    print(f"  {r['id']} {r['a']} 正确 {V.RUN.LET[it['gold']]} → 改成 {V.RUN.LET[r['y']]}；题：{it['q'][:90]}")
    print(f"    选项：{' / '.join(f'{l}. {o[:40]}' for l, o in zip(V.RUN.LET, it['opts']))}")
    print(f"    第一次引用：{quote(None, None, r['id'], r['a'])[:160]}")
    print(f"    改后引用：{quote('E', 'set2f', r['id'], r['a'])[:160]}")
