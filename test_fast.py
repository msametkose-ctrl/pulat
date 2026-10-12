import random, time
import numpy as np
import engine, fast

gen = fast.Gen()
random.seed(1)
states, dice = [], []
for g in range(3000):
    s = engine.initial()
    while True:
        d = engine.roll()
        states.append(s); dice.append(d)
        a = random.choice(engine.afterstates(s, *d))
        if engine.result(a): break
        s = engine.flip(a)
S = np.array(states, dtype=np.int8); D = np.array(dice, dtype=np.uint8)
cands, counts = gen(S, D)
off = np.concatenate([[0], np.cumsum(counts)])
bad = 0
for i, (s, d) in enumerate(zip(states, dice)):
    py = set(engine.afterstates(s, *d))
    rs = set(map(tuple, cands[off[i]:off[i+1]].tolist()))
    bad += py != rs
print(f"{len(states)} pozisyon karşılaştırıldı, farklı: {bad}")
assert (fast.flip(S) == np.array([engine.flip(s) for s in states])).all()

big = np.repeat(S, 20, axis=0)[:500_000]; bd = np.repeat(D, 20, axis=0)[:500_000]
t = time.time(); c, _ = gen(big, bd); dt = time.time() - t
print(f"Rust: {len(big)/dt:,.0f} pozisyon/s ({len(c):,} aday)")
t = time.time()
for s, d in zip(states[:20000], dice[:20000]): engine.afterstates(s, *d)
print(f"Python: {min(20000,len(states))/(time.time()-t):,.0f} pozisyon/s")
