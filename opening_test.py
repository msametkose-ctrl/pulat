import sys
import torch
from engine import initial, legal_moves, fmt_move
from model import load, equity, DEVICE

sys.stdout.reconfigure(encoding="utf-8")
BEST = {
    (2, 1): ["13/11 24/23", "13/11 6/5"], (3, 1): ["8/5 6/5"], (4, 1): ["24/23 13/9"], (5, 1): ["13/8 24/23"],
    (6, 1): ["13/7 8/7"], (3, 2): ["24/21 13/11", "13/10 13/11"], (4, 2): ["8/4 6/4"], (5, 2): ["13/8 13/11"],
    (6, 2): ["24/18 13/11"], (4, 3): ["24/20 13/10", "13/10 13/9"], (5, 3): ["8/3 6/3"], (6, 3): ["24/18 13/10", "24/18 18/15"],
    (5, 4): ["24/20 13/8", "13/8 13/9"], (6, 4): ["24/18 18/14", "8/2 6/2", "24/18 13/9"], (6, 5): ["24/18 18/13"],
}
norm = lambda t: tuple(sorted(t.split()))
if __name__ == "__main__":
    for path in sys.argv[1:]:
        net = load(path)
        ok = 0
        print(f"\n== {path}")
        for (a, b), best in BEST.items():
            opts = legal_moves(initial(), a, b)
            cands = list(opts)
            with torch.no_grad():
                eq = equity(torch.sigmoid(net(torch.tensor(cands, dtype=torch.float32, device=DEVICE)))).cpu()
            order = eq.argsort(descending=True)
            txt = lambda i: " ".join(fmt_move(m) for m in opts[cands[i]])
            pick = txt(order[0])
            hit = norm(pick) in {norm(x) for x in best}
            ok += hit
            print(f"  {a}-{b}: AI {pick:<14} {'✓' if hit else '✗ uzman: ' + ' / '.join(best)}")
        print(f"  Doğru: {ok}/{len(BEST)}")
