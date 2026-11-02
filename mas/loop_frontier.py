"""闭环第①步上的宽度取舍（事后诊断，非预注册；不调模型，P = clip(x, E−c·m, E+c·m)）。用法：python loop_frontier.py runs_loop"""
import math
import statistics as st
import sys

import llmloop_score as S

R, _, _, _ = S.load(sys.argv[1])
ok = S.ok
base = {(q["k"], q["rep"], q["a"], round(q["x"], 8)): (q["x"], q["E"], q["r"] / q["c"]) for q in R if q["arm"] == "rule" and q["c"] == 1.0}
rows = [dict(k=key[0], x=v[0], E=v[1], m=v[2]) for key, v in base.items()]
print("c      " + "".join(f"k={k}: K / B / net      " for k in range(5)) + "最差 net")
for c in (0, .1, .25, .5, .75, 1, 1.5, 2):
    s, worst = "", 9
    for k in range(5):
        X = [r for r in rows if r["k"] == k]
        P = lambda r: max(r["E"] - c * r["m"], min(r["E"] + c * r["m"], r["x"]))
        g, b = [r for r in X if ok(r["x"])], [r for r in X if not ok(r["x"])]
        net = st.mean(ok(P(r)) for r in X) - st.mean(ok(r["x"]) for r in X)
        worst = min(worst, net)
        s += f"{st.mean(ok(P(r)) for r in g):4.0%}/{st.mean(ok(P(r)) for r in b):4.0%}/{net:+.2f}   "
    print(f"{c:4.2f}   {s}{worst:+.2f}")
