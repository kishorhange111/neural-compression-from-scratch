"""Adaptive count models: the classical baselines a neural compressor has to beat.

``OrderK(k)`` predicts the next byte from counts of what followed the last k bytes so
far in *this* stream. It starts knowing nothing and learns as it codes, so it needs no
stored model. The decoder rebuilds the same counts from the bytes it has decoded.

Unseen contexts are the hard part. Each order's counts are mixed with the next lower
order's prediction, using the escape weight of PPM method C: a context seen n times
with u distinct successors gives u / (n + u) to the lower order. At the bottom is a
uniform distribution, so no byte ever has probability 0. All arithmetic is in float64
on integer counts, which makes encoder and decoder bit-identical.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

V = 256


class OrderK:
    """``prime``: bytes to count before coding starts. That is a model learned from a
    training set and shipped to both sides, like the transformer's weights."""

    def __init__(self, k, prime=b""):
        self.k = k
        self.primed = None
        if prime:
            self.start(1)
            for s in prime:
                self.feed([s])
            self.primed = self.tables

    def start(self, B):
        self.hist = [b"" for _ in range(B)]
        # tables[j][context] -> counts over the next byte, for every order j ≤ k
        self.tables = [defaultdict(lambda: np.zeros(V)) for _ in range(self.k + 1)]
        if self.primed is not None:
            for t, p in zip(self.tables, self.primed, strict=True):
                t.update({c: v.copy() for c, v in p.items()})

    def _predict(self, h):
        p = np.full(V, 1.0 / V)
        for j in range(min(self.k, len(h)) + 1):
            ctx = h[len(h) - j :] if j else b""
            c = self.tables[j].get(ctx)
            if c is None:
                continue
            n, u = c.sum(), np.count_nonzero(c)
            p = (c + u * p) / (n + u)
        return p

    def probs(self):
        return np.stack([self._predict(h) for h in self.hist])

    def feed(self, syms):
        for b, s in enumerate(syms):
            h = self.hist[b]
            for j in range(min(self.k, len(h)) + 1):
                self.tables[j][h[len(h) - j :] if j else b""][int(s)] += 1
            self.hist[b] = (h + bytes([int(s)]))[-self.k :] if self.k else b""
