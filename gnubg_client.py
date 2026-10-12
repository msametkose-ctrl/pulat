import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

GNUBG = os.environ.get("GNUBG", r"C:\Program Files (x86)\gnubg\gnubg-cli.exe")
BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnubg_bridge", "bridge.py")


class GnubgPool:
    def __init__(self, n=32, base_port=5800, plies=2):
        self.plies = plies
        self.procs, self.socks = [], []
        for i in range(n):
            env = dict(os.environ, GNUBG_BRIDGE_PORT=str(base_port + i))
            self.procs.append(subprocess.Popen([GNUBG, "-q", "-t", "-p", BRIDGE], env=env, cwd=os.path.dirname(GNUBG),
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               stdin=subprocess.DEVNULL))
        for i in range(n):
            for _ in range(200):
                try:
                    s = socket.create_connection(("127.0.0.1", base_port + i), timeout=600)
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("gnubg köprüsü başlamadı")
            self.socks.append(s.makefile("rw", encoding="utf-8", newline="\n"))
        self.ex = ThreadPoolExecutor(n)

    def _ask(self, k, boards, plies):
        f = self.socks[k]
        f.write(json.dumps({"b": boards, "p": plies}) + "\n")
        f.flush()
        return json.loads(f.readline())

    def probs(self, states, plies=None):
        states = np.asarray(states).tolist()
        if not states:
            return np.zeros((0, 3), dtype=np.float32)
        n = len(self.socks)
        parts = [states[i::n] for i in range(n)]
        res = list(self.ex.map(self._ask, range(n), parts, [self.plies if plies is None else plies] * n))
        out = np.zeros((len(states), 3), dtype=np.float32)
        for i in range(n):
            if res[i]:
                out[i::n] = res[i]
        return out

    def equity(self, states, plies=None):
        p = self.probs(states, plies)
        return 2 * p[:, 0] - 1 + p[:, 1] - p[:, 2]

    def close(self):
        for f in self.socks:
            try:
                f.write(json.dumps({"quit": 1}) + "\n")
                f.flush()
            except OSError:
                pass
        time.sleep(0.5)
        for p in self.procs:
            p.kill()
