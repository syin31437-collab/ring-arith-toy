"""测试段复查（非预注册，看过结果之后）：配对 bootstrap（按窗口-组成簇）、自然格 vs 注入格、校准分层。"""
import json, random, statistics as st, collections
J = json.load(open("validate_bipm_test.json"))
R = J["cells"]
per = len(R) // 22 if len(R) % 22 == 0 else None     # 每窗口-组：(2 自然 + 9 注入) × 2 裁判 = 22 格
print("格", len(R), "窗口-组", per)
cl = [R[i:i + 22] for i in range(0, len(R), 22)]
for k in ("d5", "b1", "b2"):
    obs = st.mean(abs(r[k] - r["act"]) - abs(r["pt"] - r["act"]) for r in R)
    rg = random.Random(0); D = []
    for _ in range(2000):
        s = [r for c in (rg.choice(cl) for _ in cl) for r in c]
        D.append(st.mean(abs(r[k] - r["act"]) - abs(r["pt"] - r["act"]) for r in s))
    D.sort(); print(f"MAE({k}) − MAE(v6.1) = {obs:+.3f}，95% [{D[50]:+.3f}, {D[1949]:+.3f}]")
print("按 h 分（自然格）：")
for h in (1.5, 3.0):
    s = [r for r in R if r["tag"] == "自然" and r["h"] == h]
    print(f"  h={h}：实测 {st.mean(r['act'] for r in s):.3f} 预测 {st.mean(r['pt'] for r in s):.3f} MAE {st.mean(abs(r['pt'] - r['act']) for r in s):.3f} B2 {st.mean(abs(r['b2'] - r['act']) for r in s):.3f} 中位数基线（全信节点）B1 {st.mean(abs(r['b1'] - r['act']) for r in s):.3f}")
print("按 δ、k 分（注入格，看预测 / 实测）：")
inj = [r for r in R if r["tag"] == "注入"]
for h in (1.5, 3.0, 6.0):
    s = [r for r in inj if r["h"] == h]
    print(f"  δ={2 * h}：实测 {st.mean(r['act'] for r in s):.3f} 预测 {st.mean(r['pt'] for r in s):.3f} MAE {st.mean(abs(r['pt'] - r['act']) for r in s):.3f} 覆盖 {st.mean(r['lo'] <= r['act'] <= r['hi'] for r in s):.0%}")
print("ρ̂ 分布：", sorted(round(r["rho"], 1) for r in R)[::len(R) // 8])
