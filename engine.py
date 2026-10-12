import random

BAR, OBAR, OFF, OOFF = 24, 25, 26, 27


def initial():
    b = [0] * 28
    b[23], b[12], b[7], b[5] = 2, 5, 3, 5
    b[0], b[11], b[16], b[18] = -2, -5, -3, -5
    return tuple(b)


def flip(s):
    return tuple([-s[23 - i] for i in range(24)] + [s[OBAR], s[BAR], s[OOFF], s[OFF]])


def _single(b, d):
    res = []
    if b[BAR] > 0:
        t = 24 - d
        if b[t] >= -1:
            n = b[:]
            n[BAR] -= 1
            if n[t] == -1:
                n[t] = 0
                n[OBAR] += 1
            n[t] += 1
            res.append((n, (BAR, t)))
        return res
    top = max((i for i in range(24) if b[i] > 0), default=-1)
    can_bear = top < 6
    for i in range(top, -1, -1):
        if b[i] <= 0:
            continue
        t = i - d
        if t >= 0:
            if b[t] >= -1:
                n = b[:]
                n[i] -= 1
                if n[t] == -1:
                    n[t] = 0
                    n[OBAR] += 1
                n[t] += 1
                res.append((n, (i, t)))
        elif can_bear and (t == -1 or i == top):
            n = b[:]
            n[i] -= 1
            n[OFF] += 1
            res.append((n, (i, -1)))
    return res


def legal_moves(s, d1, d2):
    if d1 == d2:
        orders = [(d1,) * 4]
    else:
        hi, lo = max(d1, d2), min(d1, d2)
        orders = [(hi, lo), (lo, hi)]
    best, out = 0, {}
    for order in orders:
        level, depth = {s: []}, 0
        for d in order:
            nxt = {}
            for st, mv in level.items():
                for n, m in _single(list(st), d):
                    t = tuple(n)
                    if t not in nxt:
                        nxt[t] = mv + [m]
            if not nxt:
                break
            level, depth = nxt, depth + 1
        if depth > best:
            best, out = depth, dict(level)
        elif depth == best and depth > 0 and not (depth == 1 and d1 != d2):
            for k, v in level.items():
                out.setdefault(k, v)
    return out if best else {s: []}


def afterstates(s, d1, d2):
    return list(legal_moves(s, d1, d2))


def roll():
    return random.randint(1, 6), random.randint(1, 6)


def result(after):
    if after[OFF] == 15:
        return 2 if after[OOFF] == 0 else 1
    return 0


def fmt_move(m):
    f, t = m
    fs = "bar" if f == BAR else str(f + 1)
    ts = "off" if t == -1 else str(t + 1)
    return f"{fs}/{ts}"
