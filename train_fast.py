import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import fast
from model import Net, DEVICE, load

CK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_fast")
LMAX = 600


@torch.no_grad()
def pick(net, cands, counts, rng=None):
    n = len(counts)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    if net is None:
        return offs + (rng.random(n) * counts).astype(np.int64), None
    x = torch.from_numpy(cands).to(DEVICE, non_blocking=True).float()
    p = torch.sigmoid(net(x))
    eq = 2 * p[:, 0] - 1 + p[:, 1] - p[:, 2]
    seg = torch.repeat_interleave(torch.arange(n, device=DEVICE), torch.from_numpy(counts).to(DEVICE))
    mx = torch.full((n,), -1e9, device=DEVICE).scatter_reduce(0, seg, eq, "amax")
    pos = torch.arange(len(eq), device=DEVICE)
    pos = torch.where(eq >= mx[seg], pos, len(eq))
    first = torch.full((n,), len(eq), device=DEVICE, dtype=torch.long).scatter_reduce(0, seg, pos, "amin")
    return first.cpu().numpy(), p[first].cpu().numpy()


def match(gen, net_a, net_b, n, rng):
    s = fast.initial(n)
    turn = (np.arange(n) % 2).astype(np.int8)
    alive = np.arange(n)
    pts = np.zeros(n)
    while len(alive):
        new_s = np.empty_like(s)
        for who, net in ((0, net_a), (1, net_b)):
            m = turn == who
            if not m.any():
                continue
            c, cnt = gen(s[m], fast.roll(int(m.sum()), rng))
            j, _ = pick(net, c, cnt, rng)
            new_s[m] = c[j]
        r = np.where(new_s[:, 26] == 15, np.where(new_s[:, 27] == 0, 2, 1), 0)
        fin = r > 0
        pts[alive[fin]] = np.where(turn[fin] == 0, r[fin], -r[fin])
        keep = ~fin
        alive, s, turn = alive[keep], fast.flip(new_s[keep]), 1 - turn[keep]
    return float((pts > 0).mean()), float(pts.mean())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--parallel", type=int, default=8192, help="aynı anda oynanan oyun")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr-end", type=float, default=None, help="süre sonunda ulaşılacak öğrenme hızı (doğrusal azalır)")
    ap.add_argument("--out", default=None, help="kayıt klasörü (varsayılan checkpoints_fast)")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--updates", type=int, default=1, help="her adımda kaç öğrenme güncellemesi")
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--eval-min", type=float, default=3.0, help="kaç dakikada bir test")
    ap.add_argument("--eval-games", type=int, default=4000)
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "pulat_best.pt"),
                    help="karşılaştırma için sabit rakip (eski Python eğitimi)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    global CK_DIR
    if args.out:
        CK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(CK_DIR, exist_ok=True)
    rng = np.random.default_rng()
    gen = fast.Gen()
    net = Net(args.hidden, args.layers).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    games_played = 0
    latest = os.path.join(CK_DIR, "latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        games_played = ck["games"]
        print(f"devam: {games_played:,} oyundan")

    def save(path):
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "games": games_played,
                    "hidden": args.hidden, "layers": args.layers}, path)

    ref = load(args.ref) if os.path.exists(args.ref) else None
    prev = copy.deepcopy(net).eval()

    buf_s = torch.zeros((args.buffer, 28), dtype=torch.int8, device=DEVICE)
    buf_y = torch.zeros((args.buffer, 3), dtype=torch.float32, device=DEVICE)
    buf_n = buf_i = 0

    log_path = os.path.join(CK_DIR, "log.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["games", "minutes", "games_per_s", "vs_random_win", "vs_random_ppg",
                      "vs_prev_win", "vs_prev_ppg", "vs_ref_win", "vs_ref_ppg", "avg_len"])

    P = args.parallel
    S = fast.initial(P)
    traj_s = np.zeros((P, LMAX, 28), dtype=np.int8)
    traj_v = np.zeros((P, LMAX, 3), dtype=np.float32)
    tlen = np.zeros(P, dtype=np.int64)
    rows = np.arange(P)
    lens = []
    t0 = last_eval = time.time()
    games_at_eval = games_played

    while time.time() - t0 < args.hours * 3600:
        cands, counts = gen(S, fast.roll(P, rng))
        net.eval()
        j, v = pick(net, cands, counts)
        net.train()
        A = cands[j]
        traj_s[rows, tlen] = A
        traj_v[rows, tlen] = v
        tlen += 1

        done = A[:, 26] == 15
        over = (tlen >= LMAX) & ~done
        if done.any():
            Fi = np.nonzero(done)[0]
            L = tlen[Fi]
            nF = len(Fi)
            G = np.zeros((nF, 3), dtype=np.float32)
            G[:, 0] = 1
            G[:, 1] = A[Fi, 27] == 0
            targ = np.zeros((nF, int(L.max()), 3), dtype=np.float32)
            targ[np.arange(nF), L - 1] = G
            for t in range(int(L.max()) - 2, -1, -1):
                act = t <= L - 2
                nxt = np.where((t + 1 == L - 1)[:, None], G, (1 - args.lam) * traj_v[Fi, t + 1] + args.lam * G)
                Gn = np.stack([1 - nxt[:, 0], nxt[:, 2], nxt[:, 1]], axis=1)
                G = np.where(act[:, None], Gn, G)
                targ[act, t] = Gn[act]
            mask = np.arange(targ.shape[1])[None, :] < L[:, None]
            xs = torch.from_numpy(traj_s[Fi, :targ.shape[1]][mask]).to(DEVICE)
            ys = torch.from_numpy(targ[mask]).to(DEVICE)
            k = len(xs)
            idx = (torch.arange(k, device=DEVICE) + buf_i) % args.buffer
            buf_s[idx], buf_y[idx] = xs, ys
            buf_i = (buf_i + k) % args.buffer
            buf_n = min(buf_n + k, args.buffer)
            games_played += nF
            lens.extend(L.tolist())
        reset = done | over
        S = fast.flip(A)
        if reset.any():
            S[reset] = fast.initial(1)
            tlen[reset] = 0

        if args.lr_end is not None:
            frac = min((time.time() - t0) / (args.hours * 3600), 1.0)
            for gr in opt.param_groups:
                gr["lr"] = args.lr + (args.lr_end - args.lr) * frac
        if buf_n >= 100_000:
            for _ in range(args.updates):
                k = torch.randint(0, buf_n, (args.batch,), device=DEVICE)
                loss = F.binary_cross_entropy_with_logits(net(buf_s[k].float()), buf_y[k])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if time.time() - last_eval >= args.eval_min * 60:
            gps = (games_played - games_at_eval) / (time.time() - last_eval)
            net.eval()
            rw, rp = match(gen, net, None, args.eval_games, rng)
            pw, pp = match(gen, net, prev, args.eval_games, rng)
            fw, fp = match(gen, net, ref, args.eval_games, rng) if ref is not None else (float("nan"),) * 2
            net.train()
            mins = (time.time() - t0) / 60
            avg_len = sum(lens) / max(len(lens), 1)
            lens = []
            print(f"[{mins:6.1f} dk] {games_played:>11,} oyun ({gps:,.0f}/s) | rastgele %{rw*100:5.1f} ({rp:+.2f}) "
                  f"| önceki %{pw*100:5.1f} ({pp:+.2f}) | referans %{fw*100:5.1f} ({fp:+.2f}) | ort. hamle {avg_len:.0f}",
                  flush=True)
            log.writerow([games_played, round(mins, 1), round(gps), rw, rp, pw, pp, fw, fp, round(avg_len, 1)])
            logf.flush()
            save(os.path.join(CK_DIR, f"gen_{games_played:010d}.pt"))
            save(latest)
            prev = copy.deepcopy(net).eval()
            last_eval = time.time()
            games_at_eval = games_played

    save(latest)
    print(f"bitti: {games_played:,} oyun")


if __name__ == "__main__":
    main()
