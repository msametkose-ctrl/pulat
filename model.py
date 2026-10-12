import random
import numpy as np
import torch
import torch.nn as nn

N_FEAT = 196
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(x):
    pts = x[:, :24]
    feats = []
    for side in (pts.clamp(min=0), (-pts).clamp(min=0)):
        feats += [(side >= 1).float(), (side >= 2).float(), (side >= 3).float(), (side - 3).clamp(min=0) / 2]
    feats += [x[:, 24:26] / 2, x[:, 26:28] / 15]
    return torch.cat(feats, dim=1)


class Net(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        mods, d = [], N_FEAT
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*mods, nn.Linear(d, 3))

    def forward(self, x):
        return self.body(encode(x))


def equity(p):
    return 2 * p[..., 0] - 1 + p[..., 1] - p[..., 2]


class NetPolicy:
    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def choose(self, cand_lists):
        sizes = [len(c) for c in cand_lists]
        flat = [s for c in cand_lists for s in c]
        x = torch.tensor(flat, dtype=torch.float32, device=DEVICE)
        p = torch.sigmoid(self.net(x)).cpu().numpy()
        eq = 2 * p[:, 0] - 1 + p[:, 1] - p[:, 2]
        idx, vals, off = [], [], 0
        for n in sizes:
            j = int(np.argmax(eq[off:off + n]))
            idx.append(j)
            vals.append(p[off + j])
            off += n
        return idx, vals


class RandomPolicy:
    def choose(self, cand_lists):
        return [random.randrange(len(c)) for c in cand_lists], [None] * len(cand_lists)


def load(path):
    ck = torch.load(path, map_location=DEVICE)
    net = Net(ck.get("hidden", 256), ck.get("layers", 2)).to(DEVICE)
    net.load_state_dict(ck["model"])
    net.eval()
    return net
