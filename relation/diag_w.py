"""探索性诊断（事后，不进判决）：好节点不守时去了哪、坏节点没救回时去了哪。"""
import json, collections, sys
sys.path.insert(0, ".")
import wording as W
import stage2 as S2
import run as RUN

items = {x["id"]: x for x in json.load(open(W.ITEMS))}
R, _ = W.load()
def members(it, p):
    w, g = S2.wrong(it), it["gold"]
    return {i for i in {"2t": [g, w[0]], "2f": w[:2], "3t": [g, w[0], w[1]], "3f": w}[p[-2:]]}

print("好节点（第一次对）在假集合下的去向：守 / 进集合 / 第三个选项")
for p in ["set2f", "soft2f", "cond2f", "set3f"]:
    c = collections.Counter()
    for r in R:
        if r["p"] == p and r["right"]:
            c["守" if r["y"] == r["x"] else ("进集合" if r["y"] in members(items[r["id"]], p) else "第三个")] += 1
    n = sum(c.values()); print(f"  {p:7s} n={n}  " + "  ".join(f"{k} {c[k]/n:.0%}" for k in ["守", "进集合", "第三个"]))

print("坏节点（第一次错）在真集合下的去向：救回 / 守原错 / 进集合里的错 / 第三个")
for p in ["set2t", "soft2t", "cond2t", "set3t"]:
    c = collections.Counter()
    for r in R:
        if r["p"] == p and not r["right"]:
            g = items[r["id"]]["gold"]
            c["救回" if r["y"] == g else "守原错" if r["y"] == r["x"] else "进集合错" if r["y"] in members(items[r["id"]], p) else "第三个"] += 1
    n = sum(c.values()); print(f"  {p:7s} n={n}  " + "  ".join(f"{k} {c[k]/n:.0%}" for k in ["救回", "守原错", "进集合错", "第三个"]))

print("坏节点第一次的错是否本来就在真集合里（set2t 的 w[0]）：")
c = collections.Counter()
for r in R:
    if r["p"] == "set2t" and not r["right"]:
        c[r["x"] in members(items[r["id"]], "set2t")] += 1
print("  ", dict(c))

print("跳出按题分布：cond2f 下每题守住的节点数（0–6）")
per = collections.defaultdict(lambda: [0, 0])
for r in R:
    if r["p"] == "cond2f" and r["right"]:
        per[r["id"]][0] += r["y"] == r["x"]; per[r["id"]][1] += 1
h = collections.Counter(round(a / b, 1) for a, b in per.values())
print("  ", sorted(h.items()))
