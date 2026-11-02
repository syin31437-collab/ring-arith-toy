"""试跑映射预检（不进正式结果）：clean 看四条预检；data1 看偏离信号长什么样。"""
import json, math, statistics as st, sys, torch
A = ["ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "claude_haiku"]
def load(fn):
    J = json.load(open(fn)); it = {x["id"]: x for x in J["items"]}; out = []
    for L in J["log"]:
        for i in sorted(L["final"]):
            if i not in it or not all(i.upper() in L[s][a] for s in ("s1", "s2") for a in A): continue
            t = math.log10(it[i]["truth"])
            lg = lambda v: math.log10(v) - t if v and v > 0 else float("nan")
            out.append(dict(b=L["batch"], a=lg(it[i]["scale_a"]), bb=lg(it[i]["scale_b"]), s1=[lg(L["s1"][x][i]) for x in A], s2=[lg(L["s2"][x][i]) for x in A], f=lg(L["final"][i])))
    return out
R = load(sys.argv[1] if len(sys.argv) > 1 else "runs_pilot/clean_900/run.json")
print("完整题数", len(R))
mad = lambda x: 1.4826 * (x - x.median()).abs().median().item()
S1 = torch.tensor([r["s1"] for r in R]); S2 = torch.tensor([r["s2"] for r in R]); P = torch.tensor([r["a"] for r in R]); F = torch.tensor([r["f"] for r in R])
for nm, X in (("第①步独立", S1), ("第②步讨论后", S2)):
    c = X.median(1).values
    Xc = X - X.median(0).values
    C = torch.corrcoef(Xc.T); k = C.shape[0]
    print(f"{nm}：共识中位偏 {c.median().item():+.4f}（h/4 = {math.log10(1.2) / 8:.4f}）；单节点 MAD {[round(mad(X[:, j]), 4) for j in range(k)]}；各节点 |误差| 95% {[round(X[:, j].abs().quantile(.95).item(), 4) for j in range(k)]}；两两相关 {(C.sum() - k) / (k * (k - 1)):.2f}")
print(f"秤 A MAD {mad(P):.4f}（设定 0.03 nat = {0.03 / math.log(10):.4f}）；团队答案 MAD {mad(F):.4f}，|误差| > 门槛 {math.log10(1.2) / 2:.4f} 的比例 {(F.abs() > math.log10(1.2) / 2).float().mean():.0%}")
for b in sorted({r["b"] for r in R}):
    s = [r for r in R if r["b"] == b]
    print(f"  批 {b}：" + " ".join(f"{st.median(r['s1'][j] for r in s):+.3f}/{st.median(r['s2'][j] for r in s):+.3f}" for j in range(5)) + f"  团队 {st.median(r['f'] for r in s):+.3f}（各 agent 第①/②步中位偏移）")
