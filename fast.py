import ctypes
import os
import sys

import numpy as np

_LIB = {"win32": "tavla_fast.dll", "darwin": "libtavla_fast.dylib"}.get(sys.platform, "libtavla_fast.so")
_dll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastengine", "target", "release", _LIB))
_dll.gen_batch.restype = ctypes.c_int64
_dll.gen_batch.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
THREADS = os.cpu_count() or 8
_FLIP = list(range(23, -1, -1))


class Gen:
    def __init__(self, cap=1 << 20):
        self._alloc(cap)

    def _alloc(self, cap):
        self.cap = cap
        self.out = np.empty((cap, 28), dtype=np.int8)

    def __call__(self, states, dice):
        states = np.ascontiguousarray(states, dtype=np.int8)
        dice = np.ascontiguousarray(dice, dtype=np.uint8)
        n = len(states)
        counts = np.empty(n, dtype=np.uint32)
        while True:
            t = _dll.gen_batch(n, states.ctypes.data, dice.ctypes.data, self.out.ctypes.data, self.cap,
                               counts.ctypes.data, THREADS)
            if t >= 0:
                return self.out[:t].copy(), counts.astype(np.int64)
            self._alloc(int(-t * 1.5))


def flip(s):
    return np.concatenate([-s[:, _FLIP], s[:, [25, 24, 27, 26]]], axis=1)


def initial(n):
    b = np.zeros((n, 28), dtype=np.int8)
    b[:, 23], b[:, 12], b[:, 7], b[:, 5] = 2, 5, 3, 5
    b[:, 0], b[:, 11], b[:, 16], b[:, 18] = -2, -5, -3, -5
    return b


def roll(n, rng):
    return rng.integers(1, 7, size=(n, 2), dtype=np.uint8)
