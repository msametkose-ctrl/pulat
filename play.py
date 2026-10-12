import os
import random
import sys

from engine import initial, flip, legal_moves, roll, result, fmt_move, BAR, OBAR, OFF, OOFF
from model import load, NetPolicy, equity

sys.stdout.reconfigure(encoding="utf-8")


def cell(v):
    if v > 0:
        return f"X{v:<2}"
    if v < 0:
        return f"O{-v:<2}"
    return " · "


def show(s):
    top = list(range(12, 24))
    bot = list(range(11, -1, -1))
    print()
    print(" " + " ".join(f"{i+1:>3}" for i in top[:6]) + "  |" + " ".join(f"{i+1:>3}" for i in top[6:]))
    print(" " + " ".join(cell(s[i]) for i in top[:6]) + "  |" + " ".join(cell(s[i]) for i in top[6:]))
    print(f"   kırık: X={s[BAR]}  O={s[OBAR]}      toplanan: X={s[OFF]}  O={s[OOFF]}")
    print(" " + " ".join(cell(s[i]) for i in bot[:6]) + "  |" + " ".join(cell(s[i]) for i in bot[6:]))
    print(" " + " ".join(f"{i+1:>3}" for i in bot[:6]) + "  |" + " ".join(f"{i+1:>3}" for i in bot[6:]))
    print()


def ai_fmt(m):
    f, t = m
    fs = "bar" if f == BAR else str(24 - f)
    ts = "off" if t == -1 else str(24 - t)
    return f"{fs}/{ts}"


def norm(txt):
    return sorted(txt.lower().replace(",", " ").split())


def human_turn(s, d):
    opts = legal_moves(s, *d)
    items = list(opts.items())
    if items[0][1] == []:
        print(f"Zar {d[0]}-{d[1]}: oynayacak hamlen yok.")
        input("(enter) ")
        return items[0][0]
    for k, (_, mv) in enumerate(items, 1):
        print(f"  {k:>2}) {' '.join(fmt_move(m) for m in mv)}")
    while True:
        t = input(f"Zar {d[0]}-{d[1]} → hamlen: ").strip()
        if t.isdigit() and 1 <= int(t) <= len(items):
            return items[int(t) - 1][0]
        for st, mv in items:
            if norm(" ".join(fmt_move(m) for m in mv)) == norm(t):
                return st
        print("Geçersiz. Listeden numara seç.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "pulat_best.pt")
    ai = NetPolicy(load(path))
    print(f"Model: {path}")
    score = [0, 0]
    while True:
        s = initial()
        human = random.random() < 0.5
        print("\n=== Yeni oyun ===", "Sen başlıyorsun." if human else "AI başlıyor.")
        while True:
            d = roll()
            if human:
                show(s)
                a = human_turn(s, d)
            else:
                opts = legal_moves(s, *d)
                cands = list(opts)
                (j,), (p,) = ai.choose([cands])
                a = cands[j]
                mv = opts[a]
                txt = " ".join(ai_fmt(m) for m in mv) if mv else "(oynayamadı)"
                print(f"AI zar {d[0]}-{d[1]}: {txt}   [AI'nın kazanma tahmini: %{p[0]*100:.0f}]")
            r = result(a)
            if r:
                who = 0 if human else 1
                score[who] += r
                show(a if human else flip(a))
                print(("Kazandın" if human else "AI kazandı") + (" — MARS! (2 puan)" if r == 2 else " (1 puan)"))
                print(f"Skor → Sen {score[0]} - {score[1]} AI")
                break
            s = flip(a)
            human = not human
        if input("Bir oyun daha? (e/h) ").strip().lower() != "e":
            break


if __name__ == "__main__":
    main()
