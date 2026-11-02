"""validate_v61 结果复查（非预注册，看过结果之后）：配对 bootstrap、校准偏差来源、区间信息量。"""
import json, random, statistics as st, collections
J = json.load(open("validate_v61.json"))
R, I = J["cells"], J["items"]
print("逐题数", len(I), "格", len(R), "每格题数", len(I) / len(R))
# 配对 bootstrap（按配置，两种裁判算一个簇）
cfg = collections.defaultdict(list)
for r in R:
    cfg[(r["n"], r["f"], r["attack"], r["m"])].append(r)
keys = list(cfg)
for k2 in ("d5", "b2", "b1"):
    obs = st.mean(abs(r[k2] - r["act"]) - abs(r["pt"] - r["act"]) for r in R)
    D = []
    rg = random.Random(0)
    for _ in range(2000):
        s = [r for k in (rg.choice(keys) for _ in keys) for r in cfg[k]]
        D.append(st.mean(abs(r[k2] - r["act"]) - abs(r["pt"] - r["act"]) for r in s))
    D.sort()
    print(f"MAE({k2}) − MAE(v6.1) = {obs:+.3f}，95% 区间 [{D[50]:+.3f}, {D[1949]:+.3f}]")
# 分攻击
print("攻击     m   实测  点   |误差|  B2误差  区间覆盖 宽度")
for a in ["ALIE0.5", "ALIE1", "ALIE2", "ALIE4", "noise", "flip"]:
    for m in (32, 512):
        s = [r for r in R if r["attack"] == a and r["m"] == m]
        print(f"{a:8s}{m:4d} {st.mean(r['act'] for r in s):.2f} {st.mean(r['pt'] for r in s):.2f}  {st.mean(abs(r['pt'] - r['act']) for r in s):.3f}  {st.mean(abs(r['b2'] - r['act']) for r in s):.3f}   {st.mean(r['lo'] <= r['act'] <= r['hi'] for r in s):.0%}   {st.median(r['hi'] - r['lo'] for r in s):.2f}")
# 高端校准来源
pos = 0
hi = collections.defaultdict(list)
for r in R:
    n = int(len(I) / len(R))
    for q, y, _ in I[pos:pos + n]:
        if q >= 0.8:
            hi[(r["attack"], r["m"])].append((q, y))
    pos += n
print("预测 ≥ 0.8 的题：来源 → 平均预测 / 实际 / 题数")
for k, v in sorted(hi.items(), key=lambda x: -len(x[1])):
    print(f"  {k}: {st.mean(x[0] for x in v):.2f} / {st.mean(x[1] for x in v):.2f} / {len(v)}")
# 区间信息量：把区间换成"点预测 ± 全部格点误差的 90% 分位"这个最简单的固定宽度，覆盖多少、多宽
E = sorted(abs(r["pt"] - r["act"]) for r in R)
q90 = E[int(.9 * len(E))]
print(f"点误差 90% 分位 {q90:.3f} → 固定宽度 {2 * q90:.2f}（事后最优）；v6.1 区间宽度中位 {st.median(r['hi'] - r['lo'] for r in R):.2f}")
# 自然域
Nat = J["natural"]
print("自然域：Oslo", [(x["src"], round(x["act"], 2), round(x["pt"], 2)) for x in Nat if x["src"].startswith("Oslo")])
epa = [x for x in Nat if x["src"].startswith("EPA")]
print(f"EPA {len(epa)} 格：实测均值 {st.mean(x['act'] for x in epa):.2f}、预测均值 {st.mean(x['pt'] for x in epa):.2f}（低估 = 实测 > 预测）")
