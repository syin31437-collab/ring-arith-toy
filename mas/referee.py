"""隐身裁判（2026-11-02，库伯：不问"这里错了吗"，问"系统知不知道自己偏离了平时的样子"）。
离线只读 team2.py 的记录，输出不回到 agent。判断全程不用真值；真值只在 score.py 里评分时用。基准 = 系统在校准期（第 0–2 批）自己的样子。
常数全部沿用冻结的检测层：W = 0.3、C = 3.3、截断 ±3σ；预测 = law_v6（6.1）、前提自检 = premise2.decide（2.0）。
三层：
  agent：每个 agent 第①步（独立）相对秤 A 的偏移，闸门版 EWMA（commonmode.run_lines cm2）；评估期越线 = 该 agent 偏离。
  对话：逐题 c3 − c1（讨论把共识挪了多少，不用秤）和 log 散布（讨论后还剩多少分歧），各减校准中位、除校准 MAD 后 EWMA；任一越线 = 对话偏离。
  团队：团队答案 − (秤 A + 秤 B) / 2，同样 EWMA；越线 = 团队偏离。
另外：对照基线"只靠同伴"：每个 agent 第①步相对其他人中位数的偏移做同样的 EWMA（没有秤）。预测：规律 6.1 给出团队答案的被骗率（裁判输出 = 团队答案）。
"""
import json
import math
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "switching"))
from analytic import Phi, sigma_c  # noqa: E402
from commonmode import run_lines  # noqa: E402
from law_v6 import LAW_VERSION, _mad, fit, subsets  # noqa: E402
from premise2 import PREMISE_VERSION, decide  # noqa: E402
from purge import C as CC, SD_S, W  # noqa: E402

AGENTS = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b", "claude_haiku", "codex"]
START, DELTA = 3, 0.40
H = math.log10(1 + DELTA) / 2


def load(fn):
    J = json.load(open(fn))
    it = {x["id"]: x for x in J["items"]}
    R = []
    for L in J["log"]:
        for i in sorted(L["final"]):
            if i not in it or L["final"][i] <= 0:
                continue
            if not all(L[s].get(a, {}).get(i, 0) > 0 for s in ("s1", "s2", "s3") for a in AGENTS):
                continue
            t = math.log10(it[i]["truth"])
            g = lambda v: math.log10(v) - t                 # 下面的判断都只用差值，真值被抵消；评分才用绝对值
            R.append({"b": L["batch"], "a": g(it[i]["scale_a"]), "bb": g(it[i]["scale_b"]), "f": g(L["final"][i]),
                      "s1": torch.tensor([g(L["s1"][a][i]) for a in AGENTS]), "s3": torch.tensor([g(L["s3"][a][i]) for a in AGENTS])})
    return J, R


def ewma_first(x, ncal):
    """x 全序列；前 ncal 个算基线。返回评估期第一次越线的下标（相对评估期），没越线 = None。"""
    base = st.median(x[:ncal])
    sd = max(1.4826 * st.median(abs(v - base) for v in x[:ncal]), 1e-6)
    S = 0.0
    for i, v in enumerate(x):
        S = (1 - W) * S + W * max(-3.0, min(3.0, (v - base) / sd))
        if i >= ncal and abs(S) > CC * SD_S:
            return i - ncal
    return None


def analyze(R):
    cal, ev = [r for r in R if r["b"] < START], [r for r in R if r["b"] >= START]
    n = len(AGENTS)
    rc = [(False, r["a"], r["s1"], 2 * H) for r in cal]
    re = [(True, r["a"], r["s1"], 2 * H) for r in ev]
    rows5 = [(f, p, x, d, k) for k, (f, p, x, d) in enumerate(rc + re)]
    first, _ = run_lines(rc, rows5, AGENTS, "cm2")
    agent = {a: (first[a] - len(cal) if a in first and first[a] >= len(cal) else None) for a in AGENTS}
    both = cal + ev
    peer = {}
    for j, a in enumerate(AGENTS):
        x = [(r["s1"][j] - torch.cat([r["s1"][:j], r["s1"][j + 1:]]).median()).item() for r in both]
        peer[a] = ewma_first(x, len(cal))
    d = [(r["s3"].median() - r["s1"].median()).item() for r in both]
    sp = [math.log10(max(r["s3"].std().item(), 1e-4)) for r in both]
    conv = {"shift": ewma_first(d, len(cal)), "spread": ewma_first(sp, len(cal))}
    team = ewma_first([r["f"] - (r["a"] + r["bb"]) / 2 for r in both], len(cal))
    F = fit(rc, re)
    S = subsets(F["ph"])
    P = S @ F["prec"] + 1 / F["vc"]
    sd = 1 / torch.sqrt(P)
    q = []
    for r in ev:
        m = (S @ (F["prec"] * r["s1"]) + (r["a"] - F["b"]) / F["vc"]) / P
        q.append(st.mean(1 - (Phi((H - (r["f"] - mi)) / si) - Phi((-H - (r["f"] - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist())))
    prem = decide(rc, re, [r["bb"] for r in cal], [r["bb"] for r in ev], H, [r["b"] for r in cal], [r["b"] for r in ev])
    return {"law": LAW_VERSION, "premise": PREMISE_VERSION, "agent": agent, "peer": peer, "conv": conv, "team": team, "ph": dict(zip(AGENTS, F["ph"].tolist())), "pred": st.mean(q), "refuse": prem["refuse"], "prem": {k: (bool(v) if isinstance(v, bool) else v) for k, v in prem.items()}, "ncal": len(cal), "nev": len(ev)}


def truth_view(R):
    """只用于评分：团队实际被骗率、真实偏离（评估期 − 校准期的团队中位偏移）。"""
    cal, ev = [r for r in R if r["b"] < START], [r for r in R if r["b"] >= START]
    return {"wrong": st.mean(abs(r["f"]) > H for r in ev), "wrong_cal": st.mean(abs(r["f"]) > H for r in cal), "mae": st.mean(abs(r["f"]) for r in ev),
            "shift": st.median(r["f"] for r in ev) - st.median(r["f"] for r in cal), "agent_wrong": st.mean((r["s1"].abs() > H).float().mean().item() for r in ev),
            "scale_wrong": st.mean(abs(r["a"]) > H for r in ev)}


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        J, R = load(fn)
        A, T = analyze(R), truth_view(R)
        print(f"== {J['cond']}：agent 越线 { {k[:10]: v for k, v in A['agent'].items() if v is not None} } | 同伴基线 { {k[:10]: v for k, v in A['peer'].items() if v is not None} }")
        print(f"   对话 {A['conv']} | 团队 {A['team']} | 拒答 {A['refuse']} | 诚实概率 { {k[:7]: round(v, 2) for k, v in A['ph'].items()} }")
        print(f"   预测被骗 {A['pred']:.3f} 实际 {T['wrong']:.3f}（校准期 {T['wrong_cal']:.3f}）；团队真实偏移 {T['shift']:+.3f}；单 agent 被骗 {T['agent_wrong']:.2f}，秤被骗 {T['scale_wrong']:.2f}")
