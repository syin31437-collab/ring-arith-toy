"""真实模型当节点 · 第四题：检索资料被投毒，用模型自己的记忆当约束。

前三题说明：强 LLM 内部高度自洽，剩下的错误是共享信念，约束看不见。所以这里人为制造信息来源的独立性：
  资料：每个节点拿到一份参考资料，给出城市对的坐标，让它算大圆距离（它们算得很准，答案基本由资料决定）。
  投毒：5 个 DeepSeek 节点拿到的资料里，每对的第二个城市被沿大圆方向挪开，使距离变长 δ（过半的共模故障）；
        Codex、Claude Haiku 拿到干净资料（少数派）。
  抖动：每个节点的资料坐标再加各自独立的随机抖动（两档：0.01° / 0.2°），调节节点噪声。
  约束：模型自己的记忆——cities.py 采到的 7 个模型凭记忆报的坐标中位数 → 预测距离（和资料是两个来源）。
配置：抖动 2 档 × δ ∈ {0, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%}，每种配置 26 对城市（距离 > 500 km），每次请求 13 对。
测量（log10 距离）：δ=0 的配置给出 σ_eff（约束预测与报告中位数之差的标准差）和 σ_node（各节点偏离中位数的 RMS）；
  其余配置算漏检率（|输出 − 真值| > δ/2 的比例）和穿透率。
零重拟合：transfer.py 里玩具（时间 + 代数）拟合的曲线，原样预测这里的漏检率和穿透率。

用法：python3 poison.py collect
      ~/gut-external/torch215-venv/bin/python poison.py analyze
"""
import concurrent.futures as cf
import json
import math
import os
import random
import sys
import time

from cities import cities, gc, load_answers as load_city, pairs
from collect import NODES, ask, parse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_poison")
DELTAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
JITTERS = [0.01, 0.2]
POISONED = {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"}
BATCH = 13
BATCH_DS, TOKENS_DS = 13, 4000              # DeepSeek 关掉推理：开着推理会手算球面距离，16000 token 也写不完


def chosen():
    cs = cities()
    by = {c["id"]: c for c in cs}
    out = [(i, a, b) for i, a, b in pairs(cs) if gc(by[a]["lat"], by[a]["lon"], by[b]["lat"], by[b]["lon"]) > 500][:26]
    return by, out


def destination(lat, lon, bearing, dist):
    p1, l1, br, ang = map(math.radians, (lat, lon, bearing, dist / 6371.0088 * 180 / math.pi))
    p2 = math.asin(math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(br))
    l2 = l1 + math.atan2(math.sin(br) * math.sin(ang) * math.cos(p1), math.cos(ang) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    return math.degrees(math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))


def sheet(node, delta, jitter, by, ps):
    rng = random.Random(f"{node}-{delta}-{jitter}")
    lines = []
    for i, a, b in ps:
        A, B = by[a], by[b]
        blat, blon = B["lat"], B["lon"]
        if node in POISONED and delta > 0:
            d = gc(A["lat"], A["lon"], blat, blon)
            blat, blon = destination(A["lat"], A["lon"], bearing(A["lat"], A["lon"], blat, blon), d * (1 + delta))
        j = lambda x: x + rng.gauss(0, jitter)
        lines.append(f"{i}: {A['name']}, {A['country']} ({j(A['lat']):.4f}, {j(A['lon']):.4f}) — "
                     f"{B['name']}, {B['country']} ({j(blat):.4f}, {j(blon):.4f})")
    return lines


def job(node, key, text, max_tokens=8000, think=True):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask(node, text, max_tokens, think)
            json.dump({"answers": parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(10)                        # 429 限流时退避
    return node, key, f"FAILED {err}"


def collect():
    os.makedirs(CACHE, exist_ok=True)
    by, ps = chosen()
    jobs = []
    for node in NODES:
        for jitter in JITTERS:
            for delta in DELTAS:
                lines = sheet(node, delta, jitter, by, ps)
                ds = NODES[node][0] == "deepseek"
                size = BATCH_DS if ds else BATCH
                for b in range(0, len(lines), size):
                    text = ("Here is a reference sheet with the coordinates (latitude, longitude) of city pairs, from a document you retrieved. "
                            "Using these coordinates, give the great-circle distance in kilometres for each pair.\n"
                            "Reply with ONLY a JSON object mapping the pair ID to a number. No prose.\n\n" + "\n".join(lines[b:b + size]))
                    jobs.append((node, f"{jitter}_{delta}_{'n' if ds else ''}{b // size}", text, TOKENS_DS if ds else 8000, not ds))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def analyze():
    import statistics as st
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    os.chdir(os.path.join(HERE, "..", "switching"))
    from decouple import fit_sigmoid, pred_sigmoid
    from robust import wmedian
    from transfer import as_feats, load

    by, ps = chosen()
    mem, _ = load_city()
    med_c = {c: [st.median(v[n][0] for n in v), st.median(v[n][1] for n in v)] for c, v in mem.items()}
    rep = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        jitter, delta, b = key.split("_")
        if node in POISONED and not b.startswith("n"):        # 只用关掉推理后的 DeepSeek 回答
            continue
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            if v > 0:
                rep.setdefault((float(jitter), float(delta)), {}).setdefault(i, {})[node] = math.log10(v)
    truth = {i: math.log10(gc(by[a]["lat"], by[a]["lon"], by[b]["lat"], by[b]["lon"])) for i, a, b in ps}
    pred = {i: math.log10(gc(*med_c[a], *med_c[b])) for i, a, b in ps}

    curves = {j: {t: fit_sigmoid(as_feats(load("geo", j) + load("alg", j), t), "ρ+κ") for t in ("miss", "pass")} for j in ("软", "硬")}
    for jitter in JITTERS:
        base = rep.get((jitter, 0.0), {})
        base = {i: v for i, v in base.items() if len(v) >= 5}
        sig_eff = st.pstdev(pred[i] - st.median(v.values()) for i, v in base.items())
        sig_node = math.sqrt(st.mean(st.mean((x - st.median(v.values())) ** 2 for x in v.values()) for v in base.values()))
        if os.environ.get("SIGEFF") == "truth":         # 约束自身误差：预测距离 vs 真值（报告中位数被嘈杂节点拉偏时用）
            sig_eff = st.pstdev(pred[i] - truth[i] for i in truth)
        s = sig_eff
        print(f"\n== 抖动 {jitter}°：σ_eff = {sig_eff:.5f}，σ_node = {sig_node:.5f}（log10） ==")
        print("δ       ρ       κ     |  软：实际漏检 / 预测   穿透 / 预测   |  硬：实际漏检 / 预测   穿透 / 预测   | 投票漏检")
        for delta in DELTAS[1:]:
            ld = math.log10(1 + delta)
            items = {i: v for i, v in rep.get((jitter, delta), {}).items() if len(v) >= 5}
            cells = []
            for judge in ("软", "硬"):
                miss, pas = [], []
                for i, v in items.items():
                    d = torch.tensor(list(v.values()))
                    pd = torch.tensor(pred[i])
                    if judge == "软":
                        out = wmedian(d, torch.exp(-((d - pd) ** 2) / (2 * s ** 2)) + 1e-6).item()
                    else:
                        keep = (d - pd).abs() <= 2.5 * s
                        out = (d[keep].median() if keep.any() else d.median()).item()
                    miss.append(abs(out - truth[i]) > ld / 2)
                    pas.append((out - truth[i]) / ld)
                feat = [{"delta": ld, "sigma_eff": sig_eff, "sigma_node": sig_node, "miss": 0.0, "pass": 0.0}]
                pm = pred_sigmoid(curves[judge]["miss"], as_feats(feat, "miss"), "ρ+κ").item()
                pp = pred_sigmoid(curves[judge]["pass"], as_feats(feat, "pass"), "ρ+κ").item()
                cells.append(f"{sum(miss) / len(miss):9.0%} / {pm:4.0%}   {st.mean(pas):5.2f} / {pp:4.2f}")
            vote = sum(abs(st.median(v.values()) - truth[i]) > ld / 2 for i, v in items.items()) / len(items)
            print(f"{delta:5.1%}  {ld / sig_eff:6.2f} {ld / sig_node:6.2f}  | " + "   | ".join(cells) + f"   | {vote:6.0%}")


if __name__ == "__main__":
    collect() if sys.argv[1] == "collect" else analyze()
