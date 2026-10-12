import sys
import time

import numpy as np

import fast
from model import load
from search import pick2
from train_fast import pick

sys.stdout.reconfigure(encoding="utf-8")


def match(net, gen, n, rng, topk):
    s = fast.initial(n)
    turn = (np.arange(n) % 2).astype(np.int8)
    alive = np.arange(n)
    pts = np.zeros(n)
    while len(alive):
        new_s = np.empty_like(s)
        for who in (0, 1):
            m = turn == who
            if not m.any():
                continue
            c, cnt = gen(s[m], fast.roll(int(m.sum()), rng))
            j = pick2(net, gen, c, cnt, topk)[0] if who == 0 else pick(net, c, cnt)[0]
            new_s[m] = c[j]
        r = np.where(new_s[:, 26] == 15, np.where(new_s[:, 27] == 0, 2, 1), 0)
        fin = r > 0
        pts[alive[fin]] = np.where(turn[fin] == 0, r[fin], -r[fin])
        keep = ~fin
        alive, s, turn = alive[keep], fast.flip(new_s[keep]), 1 - turn[keep]
    return float((pts > 0).mean()), float(pts.mean()), float(pts.std() / np.sqrt(n))


if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    net, gen, rng = load(path), fast.Gen(), np.random.default_rng(0)
    t = time.time()
    w, p, se = match(net, gen, n, rng, topk=8)
    print(f"{path}: 2-ply, 1-ply'a karşı {n} oyunda %{w*100:.1f} kazandı, {p:+.3f} ± {se:.3f} puan/oyun ({time.time()-t:.0f} sn)")
