"""第二版试跑（不进正式结果）：讨论有没有真的改变答案；四条映射预检；秤和 agent 的准度。"""
import json, math, statistics as st, sys, torch
A = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b", "claude_haiku", "codex"]
mad = lambda x: 1.4826 * (x - x.median()).abs().median().item()
for fn in sys.argv[1:]:
    J = json.load(open(fn)); it = {x["id"]: x for x in J["items"]}; R = []
    print("==", J["cond"], "错误", sum(len(L["errors"]) for L in J["log"]))
    for L in J["log"]:
        for i in sorted(L["final"]):
            if i in it and all(i in L[s].get(a, {}) and L[s][a][i] > 0 for s in ("s1", "s2", "s3") for a in A) and L["final"][i] > 0:
                t = math.log10(it[i]["truth"]); g = lambda v: math.log10(v) - t
                R.append(dict(b=L["batch"], s1=[g(L["s1"][a][i]) for a in A], s2=[g(L["s2"][a][i]) for a in A], s3=[g(L["s3"][a][i]) for a in A], f=g(L["final"][i]), pa=g(it[i]["scale_a"])))
    print(" 完整题", len(R))
    S = {k: torch.tensor([r[k] for r in R]) for k in ("s1", "s2", "s3")}
    F = torch.tensor([r["f"] for r in R]); P = torch.tensor([r["pa"] for r in R])
    print(f" 讨论改动：|①→②| 中位 {(S['s2'] - S['s1']).abs().median():.4f}，|②→③| {(S['s3'] - S['s2']).abs().median():.4f}；节点散布 ① {S['s1'].std(1).median():.4f} → ③ {S['s3'].std(1).median():.4f}")
    for k in ("s1", "s3"):
        X = S[k]; C = torch.corrcoef((X - X.median(0).values).T); n = X.shape[1]
        print(f" {k}：单节点 MAD 中位 {torch.tensor([mad(X[:, j]) for j in range(n)]).median():.4f}，共识 MAD {mad(X.median(1).values):.4f}，两两相关 {(C.sum() - n) / (n * (n - 1)):.2f}")
    h = math.log10(1.4) / 2
    print(f" 秤 MAD {mad(P):.4f}；团队 MAD {mad(F):.4f}；单 agent 被骗率 {(S['s1'].abs() > h).float().mean():.0%}，团队被骗率 {(F.abs() > h).float().mean():.0%}（门槛 {h:.3f}）")
    for b in sorted({r['b'] for r in R}):
        s = [r for r in R if r["b"] == b]
        print(f"  批{b}: " + " ".join(f"{st.median(r['s1'][j] for r in s):+.2f}→{st.median(r['s3'][j] for r in s):+.2f}" for j in range(6)) + f" 团队 {st.median(r['f'] for r in s):+.3f}")
