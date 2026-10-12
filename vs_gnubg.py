import argparse
import sys
import time

import numpy as np

import fast
from gnubg_client import GnubgPool
from model import load
from search import pick2
from train_fast import pick


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/pulat_best.pt")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=250)
    ap.add_argument("--depth", type=int, default=2, help="Pulat: 1 = sezgi, 2 = ileriye bakma")
    ap.add_argument("--gnubg-plies", type=int, default=2, help="rakip gnubg gücü: 0 (hızlı) .. 2 (en güçlü)")
    ap.add_argument("--analysis-plies", type=int, default=2)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--sanity", action="store_true", help="Pulat yerine gnubg 0-ply oynar (ölçüm testi)")
    args = ap.parse_args()

    net, gen, rng = load(args.model), fast.Gen(), np.random.default_rng(1)
    pool = GnubgPool(args.procs)
    t0 = time.time()
    pts_all, losses, decisions = [], [], 0
    big = 0
    done_games = 0
    while done_games < args.games:
        n = min(args.parallel, args.games - done_games)
        s = fast.initial(n)
        turn = (np.arange(n) % 2).astype(np.int8)
        alive = np.arange(n)
        pts = np.zeros(n)
        while len(alive):
            new_s = np.empty_like(s)
            m = turn == 0
            if m.any():
                c, cnt = gen(s[m], fast.roll(int(m.sum()), rng))
                if args.sanity:
                    e0 = pool.equity(c, 0)
                    o0 = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                    j = np.array([o + int(np.argmax(e0[o:o + k])) for o, k in zip(o0, cnt)])
                else:
                    j = pick2(net, gen, c, cnt)[0] if args.depth == 2 else pick(net, c, cnt)[0]
                new_s[m] = c[j]
                offs = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                multi = np.nonzero(cnt > 1)[0]
                if len(multi):
                    idx = np.concatenate([np.arange(offs[g], offs[g] + cnt[g]) for g in multi])
                    eq = pool.equity(c[idx], args.analysis_plies)
                    o = 0
                    for g in multi:
                        e = eq[o:o + cnt[g]]
                        loss = float(e.max() - e[j[g] - offs[g]])
                        losses.append(loss)
                        big += loss > 0.08
                        o += cnt[g]
                    decisions += len(multi)
            m = turn == 1
            if m.any():
                c, cnt = gen(s[m], fast.roll(int(m.sum()), rng))
                eq = pool.equity(c, args.gnubg_plies)
                offs = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                j = np.array([o + int(np.argmax(eq[o:o + k])) for o, k in zip(offs, cnt)])
                new_s[m] = c[j]
            r = np.where(new_s[:, 26] == 15, np.where(new_s[:, 27] == 0, 2, 1), 0)
            fin = r > 0
            pts[alive[fin]] = np.where(turn[fin] == 0, r[fin], -r[fin])
            keep = ~fin
            alive, s, turn = alive[keep], fast.flip(new_s[keep]), 1 - turn[keep]
        pts_all.extend(pts.tolist())
        done_games += n
        P = np.array(pts_all)
        pr = 500 * np.mean(losses) if losses else float("nan")
        print(f"[{(time.time()-t0)/60:5.1f} dk] {done_games} oyun | Pulat %{(P > 0).mean()*100:.1f} kazandı, "
              f"{P.mean():+.3f} ± {P.std()/np.sqrt(len(P)):.3f} puan/oyun | PR {pr:.1f} | "
              f"büyük hata (>0.08): {big}/{decisions} hamle", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
