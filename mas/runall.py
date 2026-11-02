"""跑 A、B 两组共 64 局（8 局并行；已有的 run.json 跳过）。"""
import concurrent.futures as cf, os, sys, traceback
import team2 as T
CONDS = {"clean": 8, "data1": 6, "data3": 6, "persuader": 6, "inject": 6}
jobs = [(arm, c, rep) for rep_i in range(8) for c, n in CONDS.items() if rep_i < n for rep in [1000 + rep_i] for arm in ("A", "B")]
def go(j):
    arm, c, rep = j
    out = os.path.join(T.HERE, f"runs_{arm}")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, f"{c}_{rep}", "run.json")):
        return f"跳过 {j}"
    try:
        T.run(c, rep, out)
        return f"完成 {j}"
    except Exception:
        return f"失败 {j}: {traceback.format_exc()[-200:]}"
print(len(jobs), "局", flush=True)
with cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(go, jobs):
        print(r, flush=True)
