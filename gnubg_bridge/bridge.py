import json
import os
import socket

import gnubg

gnubg.command("set threads 1")
CI = gnubg.cubeinfo(1, -1, 0, 0, (0, 0), 0, 0)


def to_gnubg(a):
    mover = [0] * 25
    onroll = [0] * 25
    for k in range(24):
        if a[k] > 0:
            mover[k] = a[k]
        elif a[k] < 0:
            onroll[23 - k] = -a[k]
    mover[24] = a[24]
    onroll[24] = a[25]
    return (mover, onroll)


def evaluate(a, ec):
    if a[26] == 15:
        return [1.0, 1.0 if a[27] == 0 else 0.0, 0.0]
    w, wg, wbg, lg, lbg, eq = gnubg.evaluate(to_gnubg(a), CI, ec)
    return [1.0 - w, lg, wg]


def main():
    port = int(os.environ.get("GNUBG_BRIDGE_PORT", "5800"))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print("BRIDGE_READY", port, flush=True)
    while True:
        conn, _ = srv.accept()
        f = conn.makefile("rw", encoding="utf-8", newline="\n")
        for line in f:
            req = json.loads(line)
            if req.get("quit"):
                return
            ec = {"cubeful": 0, "plies": int(req.get("p", 2)), "deterministic": 1, "prune": 1, "noise": 0.0}
            f.write(json.dumps([evaluate(a, ec) for a in req["b"]]) + "\n")
            f.flush()
        conn.close()


main()
