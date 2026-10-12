import numpy as np
import torch

import fast
from model import DEVICE

ROLLS = np.array([(a, b) for a in range(1, 7) for b in range(a, 7)], dtype=np.uint8)
WEIGHTS = np.array([1 / 36 if a == b else 2 / 36 for a, b in ROLLS], dtype=np.float32)


@torch.no_grad()
def eq_of(net, states, chunk=1 << 18):
    out = []
    for i in range(0, len(states), chunk):
        x = torch.from_numpy(states[i:i + chunk]).to(DEVICE).float()
        p = torch.sigmoid(net(x))
        out.append(2 * p[:, 0] - 1 + p[:, 1] - p[:, 2])
    return torch.cat(out) if out else torch.zeros(0, device=DEVICE)


def seg_max(vals, counts):
    n = len(counts)
    seg = torch.repeat_interleave(torch.arange(n, device=DEVICE), torch.from_numpy(counts).to(DEVICE))
    return torch.full((n,), -1e9, device=DEVICE).scatter_reduce(0, seg, vals, "amax")


@torch.no_grad()
def ply2_values(net, gen, X, chunk=3000):
    k = len(X)
    val = np.zeros(k, dtype=np.float32)
    won = X[:, 26] == 15
    val[won] = np.where(X[won, 27] == 0, 2.0, 1.0)
    idx = np.nonzero(~won)[0]
    for i in range(0, len(idx), chunk):
        part = idx[i:i + chunk]
        opp = fast.flip(X[part])
        S = np.repeat(opp, 21, axis=0)
        D = np.tile(ROLLS, (len(part), 1))
        c2, cnt = gen(S, D)
        best_opp = seg_max(eq_of(net, c2), cnt).view(len(part), 21)
        val[part] = (-best_opp * torch.from_numpy(WEIGHTS).to(DEVICE)).sum(1).cpu().numpy()
    return val


@torch.no_grad()
def pick2(net, gen, cands, counts, topk=8):
    eq1 = eq_of(net, cands).cpu().numpy()
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sel, owner = [], []
    for g, (o, c) in enumerate(zip(offs, counts)):
        top = np.argsort(-eq1[o:o + c])[:topk] + o
        sel.append(top)
        owner.append(np.full(len(top), g))
    sel, owner = np.concatenate(sel), np.concatenate(owner)
    v2 = ply2_values(net, gen, cands[sel])
    best = np.full(len(counts), -1e9, dtype=np.float32)
    choice = np.zeros(len(counts), dtype=np.int64)
    for s_i, g, v in zip(sel, owner, v2):
        if v > best[g]:
            best[g], choice[g] = v, s_i
    return choice, best
