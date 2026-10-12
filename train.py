import argparse
import copy
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import initial, flip, afterstates, roll, result
from model import Net, NetPolicy, RandomPolicy, DEVICE

CK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def lambda_returns(traj, gammon, lam):
    T = len(traj) - 1
    G = np.array([1.0, 1.0 if gammon else 0.0, 0.0], dtype=np.float32)
    out = [None] * (T + 1)
    out[T] = G
    for t in range(T - 1, -1, -1):
        nxt = G if t + 1 == T else (1 - lam) * traj[t + 1][1] + lam * G
        G = np.array([1 - nxt[0], nxt[2], nxt[1]], dtype=np.float32)
        out[t] = G
    return out


def new_start():
    s = initial()
    return s if random.random() < 0.5 else flip(s)


def match(pol_a, pol_b, n_games, parallel=200):
    wins = pts = done = started = 0
    games = []

    def start():
        nonlocal started
        started += 1
        return {"s": initial(), "turn": started % 2}

    games = [start() for _ in range(min(parallel, n_games))]
    while games:
        cands = [afterstates(g["s"], *roll()) for g in games]
        choice = [None] * len(games)
        for who, pol in ((0, pol_a), (1, pol_b)):
            ids = [i for i, g in enumerate(games) if g["turn"] == who]
            if ids:
                idx, _ = pol.choose([cands[i] for i in ids])
                for i, j in zip(ids, idx):
                    choice[i] = j
        keep = []
        for g, c, j in zip(games, cands, choice):
            a = c[j]
            r = result(a)
            if r:
                done += 1
                sign = 1 if g["turn"] == 0 else -1
                pts += sign * r
                wins += g["turn"] == 0
                if started < n_games:
                    keep.append(start())
            else:
                g["s"] = flip(a)
                g["turn"] ^= 1
                keep.append(g)
        games = keep
    return wins / done, pts / done


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--parallel", type=int, default=256, help="aynı anda oynanan oyun sayısı")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--eval-every", type=int, default=10_000, help="kaç oyunda bir test")
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(CK_DIR, exist_ok=True)
    net = Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    games_played = 0
    latest = os.path.join(CK_DIR, "latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        games_played = ck["games"]
        print(f"devam: {games_played} oyundan")

    def save(path):
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "games": games_played, "hidden": 256}, path)
    if not args.resume:
        save(os.path.join(CK_DIR, "gen_000000.pt"))
    prev = copy.deepcopy(net).eval()

    buf_s = np.zeros((args.buffer, 28), dtype=np.int8)
    buf_y = np.zeros((args.buffer, 3), dtype=np.float32)
    buf_n = buf_i = 0

    log_path = os.path.join(CK_DIR, "log.csv")
    new_log = not os.path.exists(log_path) or not args.resume
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["games", "minutes", "vs_random_win", "vs_random_ppg", "vs_prev_win", "vs_prev_ppg", "vs_anchor_win", "vs_anchor_ppg", "avg_len"])
    anchor_path = os.path.join(CK_DIR, "anchor_10k.pt")
    anchor = None
    if os.path.exists(anchor_path):
        from model import load
        anchor = load(anchor_path)

    policy = NetPolicy(net)
    games = [{"s": new_start(), "traj": []} for _ in range(args.parallel)]
    t0 = time.time()
    next_eval = (games_played // args.eval_every + 1) * args.eval_every
    lens = []
    net.train()

    while time.time() - t0 < args.hours * 3600:
        cands = [afterstates(g["s"], *roll()) for g in games]
        net.eval()
        idx, vals = policy.choose(cands)
        net.train()
        for g, c, j, v in zip(games, cands, idx, vals):
            a = c[j]
            g["traj"].append((a, v))
            r = result(a)
            if r:
                ys = lambda_returns(g["traj"], r == 2, args.lam)
                for (st, _), y in zip(g["traj"], ys):
                    buf_s[buf_i] = st
                    buf_y[buf_i] = y
                    buf_i = (buf_i + 1) % args.buffer
                    buf_n = min(buf_n + 1, args.buffer)
                lens.append(len(g["traj"]))
                games_played += 1
                g["s"], g["traj"] = new_start(), []
            else:
                g["s"] = flip(a)

        if buf_n >= 20_000:
            k = np.random.randint(0, buf_n, size=args.batch)
            x = torch.from_numpy(buf_s[k]).to(DEVICE, torch.float32)
            y = torch.from_numpy(buf_y[k]).to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if games_played >= next_eval:
            next_eval += args.eval_every
            net.eval()
            rw, rp = match(NetPolicy(net), RandomPolicy(), args.eval_games)
            pw, pp = match(NetPolicy(net), NetPolicy(prev), args.eval_games)
            if anchor is None and games_played >= 10_000:
                save(anchor_path)
                anchor = copy.deepcopy(net).eval()
            aw, ap_ = match(NetPolicy(net), NetPolicy(anchor), args.eval_games) if anchor is not None else (float("nan"), float("nan"))
            net.train()
            mins = (time.time() - t0) / 60
            avg_len = sum(lens) / max(len(lens), 1)
            lens = []
            print(f"[{mins:6.1f} dk] {games_played:>8} oyun | rastgeleye karşı %{rw*100:5.1f} ({rp:+.2f} puan/oyun) "
                  f"| önceki sürüm %{pw*100:5.1f} ({pp:+.2f}) | 10k sürümü %{aw*100:5.1f} ({ap_:+.2f}) | ort. hamle {avg_len:.0f}", flush=True)
            log.writerow([games_played, round(mins, 1), rw, rp, pw, pp, aw, ap_, round(avg_len, 1)])
            logf.flush()
            save(os.path.join(CK_DIR, f"gen_{games_played:06d}.pt"))
            save(latest)
            prev = copy.deepcopy(net).eval()

    save(latest)
    print("bitti:", games_played, "oyun")


if __name__ == "__main__":
    main()
