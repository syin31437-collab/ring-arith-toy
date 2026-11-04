"""非 LLM 分布式节点：真实 4 类数据（Vehicle）上只看部分特征的分类器 + 一条收到投放后的固定规则。

学习器：lr（softmax 回归，训练时随机遮特征）、knn（k = 9，只算看得到的特征）、nb（高斯朴素贝叶斯）。
视野：n6、n12（每个节点固定看 6 / 12 个特征，其余用训练均值）。
规则（统一作用于所有投放）：候选 c = 把投放用上之后的答案；
  proj  无条件接受 c；bayes 目标集合的概率乘 e²，不是硬投影；chk1、chk3 只有自己第一次给 c 的概率 ≥ 0.1 / 0.3 才接受，否则不动。
投放（同题全员相同）：
  L3t 公开贪心选出的 k 个特征的真值；L3f 公开这些特征在错误目标类里最近一行的值。
  L4t / L4f 直接说类别；set2t {真, 错} / set2f {错, 错}；set2t_r / set2f_r = 集合 + 对应的 L3。
"""
import random

import numpy as np

import stage2 as S2

LET = "ABCD"
LEARN = ("lr", "knn", "nb")
VIEWS = (6, 12)
RULES = ("proj", "bayes", "chk1", "chk3")
FORMS = ["blank", "L3t", "L3f", "L4t", "L4f", "set2t", "set2f", "set2t_r", "set2f_r"]


def load(path):
    X, y = [], []
    names = ["opel", "saab", "bus", "van"]
    for line in open(path):
        line = line.strip()
        if not line or line[0] in "%@":
            continue
        p = line.split(",")
        X.append([float(v) for v in p[:-1]])
        y.append(names.index(p[-1].strip().strip("'")))
    return np.array(X), np.array(y)


class Model:
    def __init__(self, kind, X, y):
        self.kind, self.X, self.y = kind, X, y
        if kind == "lr":
            W = np.zeros((X.shape[1] + 1, 4))
            A = np.c_[X, np.ones(len(X))]
            Y = np.eye(4)[y]
            rng = np.random.default_rng(0)
            for _ in range(1500):
                M = np.c_[rng.random(X.shape) < .6, np.ones(len(X))]
                B = A * M
                P = self.soft(B @ W)
                W -= .3 * (B.T @ (P - Y) / len(X) + 1e-3 * W)
            self.W = W
        if kind == "nb":
            self.mu = np.array([X[y == c].mean(0) for c in range(4)])
            self.s = np.array([X[y == c].std(0) + .1 for c in range(4)])
            self.pr = np.array([(y == c).mean() for c in range(4)])

    @staticmethod
    def soft(s):
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(-1, keepdims=True)

    def p(self, x, seen):
        if self.kind == "lr":
            return self.soft(np.r_[np.where(seen, x, 0.), 1.] @ self.W)
        if self.kind == "knn":
            d = (((self.X - x) * seen) ** 2).sum(1)
            nn = np.argsort(d, kind="stable")[:9]
            return (np.bincount(self.y[nn], minlength=4) + .5) / 11
        ll = np.log(self.pr) - (((x - self.mu) / self.s) ** 2 / 2 + np.log(self.s))[:, seen].sum(1)
        return self.soft(ll)


def info_order(X, y, k=8):
    """贪心前向选特征（knn 的留一准确率），避开共线。"""
    top = []
    for _ in range(k):
        best, bs = None, -1
        for f in range(X.shape[1]):
            if f in top:
                continue
            Z = X[:, top + [f]]
            d = ((Z[:, None] - Z[None]) ** 2).sum(-1)
            np.fill_diagonal(d, 1e9)
            nn = np.argsort(d, 1)[:, :9]
            acc = np.mean([np.bincount(y[r], minlength=4).argmax() == t for r, t in zip(nn, y)])
            if acc > bs:
                best, bs = f, acc
        top.append(best)
    return top


def wrong_ids(i, gold):
    return S2.wrong(dict(id=i, gold=gold))


def payload(i, x, gold, Xtr, ytr, top):
    w = wrong_ids(i, gold)
    d = ((Xtr - x) ** 2).sum(1)
    d[ytr != w[0]] = 1e9
    near = Xtr[int(np.argmin(d))]
    return dict(true=x[top], false=near[top], w=w, gold=gold)


def answer(rule, model, x, seen, top, pay, form):
    """返回 第一次答案、置信度、第二次答案。"""
    p0 = model.p(x, seen)
    x0 = int(np.argmax(p0))
    if form == "blank":
        return x0, 100 * p0[x0], x0
    g, w = pay["gold"], pay["w"]
    T = {"L4t": {g}, "L4f": {w[0]}, "set2t": {g, w[0]}, "set2f": {w[0], w[1]}, "set2t_r": {g, w[0]}, "set2f_r": {w[0], w[1]}}.get(form)
    p1 = p0
    if form in ("L3t", "L3f", "set2t_r", "set2f_r"):
        v = pay["true"] if form in ("L3t", "set2t_r") else pay["false"]
        x2 = x.copy()
        x2[top] = v
        s2 = seen.copy()
        s2[top] = True
        p1 = model.p(x2, s2)
    if rule == "bayes":
        q = p1 * np.array([np.e ** 2 if T and c in T else 1. for c in range(4)])
        c = int(np.argmax(q))
    elif T:
        c = max(T, key=lambda k: p1[k])
    else:
        c = int(np.argmax(p1))
    if rule == "chk1" and p0[c] < .1:
        c = x0
    if rule == "chk3" and p0[c] < .3:
        c = x0
    return x0, 100 * p0[x0], c


def build_nodes(Xtr, ytr, seed="views"):
    nodes = {}
    models = {k: Model(k, Xtr, ytr) for k in LEARN}
    for k in LEARN:
        for n in VIEWS:
            r = random.Random(f"{seed}-{k}-{n}")
            ix = r.sample(range(Xtr.shape[1]), n)
            seen = np.zeros(Xtr.shape[1], dtype=bool)
            seen[ix] = True
            for rl in RULES:
                nodes[f"{k}{n}_{rl}"] = (rl, models[k], seen)
    return nodes
