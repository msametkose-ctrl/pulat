import json
import os
import random
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch

import fast
from engine import initial, flip, legal_moves, _single, result, BAR
from model import load, DEVICE
from search import pick2, eq_of

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [os.path.join(HERE, "checkpoints", "pulat_best.pt"), os.path.join(HERE, "checkpoints_fast", "latest.pt")]
GEN = fast.Gen()
PORT = int(os.environ.get("PORT", 8765))

G = {"s": initial(), "score": [0, 0], "seqs": [], "dice": None, "over": True, "ai": None, "model": "", "games": 0}


def load_ai():
    for p in MODELS:
        if os.path.exists(p):
            try:
                ck = torch.load(p, map_location="cpu")
                G["ai"], G["model"], G["games"] = load(p), os.path.relpath(p, HERE), ck.get("games", 0)
                return
            except Exception as e:
                print("model yüklenemedi, eskisi kullanılıyor:", e)
                if G["ai"]:
                    return


def apply(s, m):
    b = list(s)
    f, t = m
    b[f] -= 1
    if t == -1:
        b[26] += 1
    else:
        if b[t] == -1:
            b[t] = 0
            b[25] += 1
        b[t] += 1
    return tuple(b)


def sequences(s, d1, d2):
    legal = legal_moves(s, d1, d2)
    best = len(next(iter(legal.values())))
    if best == 0:
        return []
    finals = set(legal)
    orders = [(d1,) * 4] if d1 == d2 else [(d1, d2), (d2, d1)]
    out = set()

    def rec(st, rem, mv):
        if len(mv) == best:
            if st in finals:
                out.add(tuple(mv))
            return
        if not rem:
            return
        for n, m in _single(list(st), rem[0]):
            rec(tuple(n), rem[1:], mv + [m])

    for o in orders:
        rec(s, o, [])
    return [list(map(list, x)) for x in out]


def view(**extra):
    return {"state": list(G["s"]), "score": G["score"], "model": G["model"], "games": G["games"], **extra}


def finish(winner, r):
    G["over"] = True
    G["score"][winner] += r
    return {"winner": ["human", "ai"][winner], "points": r}


def api_new(_):
    load_ai()
    G.update(s=initial(), over=False, seqs=[], dice=None)
    return view(first=random.choice(["human", "ai"]))


def api_roll(_):
    d = (random.randint(1, 6), random.randint(1, 6))
    G["dice"], G["seqs"] = d, sequences(G["s"], *d)
    return view(dice=d, seqs=G["seqs"])


def api_play(req):
    mv = [list(m) for m in req.get("moves", [])]
    if G["over"] or (G["seqs"] and mv not in G["seqs"]) or (not G["seqs"] and mv):
        return view(error="geçersiz hamle")
    s = G["s"]
    for m in mv:
        s = apply(s, m)
    G["s"], G["seqs"] = s, []
    out = view()
    r = result(s)
    if r:
        out.update(finish(0, r), state=list(s), score=G["score"])
    return out


def api_ai(req):
    if G["over"]:
        return view(error="oyun bitti")
    d = (random.randint(1, 6), random.randint(1, 6))
    s_ai = flip(G["s"])
    legal = legal_moves(s_ai, *d)
    cands = list(legal)
    arr = np.array(cands, dtype=np.int8)
    cnt = np.array([len(cands)])
    if req.get("depth", 2) == 2 and len(cands) > 1:
        j = int(pick2(G["ai"], GEN, arr, cnt)[0][0])
    else:
        j = int(eq_of(G["ai"], arr).argmax())
    a = cands[j]
    with torch.no_grad():
        p = torch.sigmoid(G["ai"](torch.tensor([a], dtype=torch.float32, device=DEVICE)))[0].cpu().numpy()
    frames, st = [], s_ai
    for m in legal[a]:
        st = apply(st, m)
        frames.append(list(flip(st)))
    moves = [[("bar" if f == BAR else 24 - f), ("off" if t == -1 else 24 - t)] for f, t in legal[a]]
    G["s"] = flip(a)
    out = view(dice=d, frames=frames, moves=moves, win_prob=float(p[0]), gammon_prob=float(p[1]))
    r = result(a)
    if r:
        out.update(finish(1, r), score=G["score"])
    return out


ROUTES = {"/api/new": api_new, "/api/roll": api_roll, "/api/play": api_play, "/api/ai": api_ai}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self._send(404, b"yok", "text/plain")

    def do_POST(self):
        fn = ROUTES.get(self.path)
        if not fn:
            return self._send(404, b"{}", "application/json")
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        self._send(200, json.dumps(fn(req)).encode(), "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    load_ai()
    print(f"Tavla: http://localhost:{PORT}  (model: {G['model']})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
