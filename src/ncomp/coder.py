"""Arithmetic coding with 32-bit integers, and compression with any predictive model.

A model that assigns probability p to the next symbol lets an arithmetic coder spend
−log₂ p bits on it, to within 2 bits for the whole message. So a model's cross-entropy
*is* a compressed size, and a better language model is a better compressor.

The coder is the classic one of Witten, Neal & Cleary (1987):
- [low, high] narrows to the symbol's sub-interval of the cumulative frequency table;
- leading bits that low and high agree on are emitted;
- a near-straddle of the midpoint is deferred as a pending bit, which stops the
  interval from collapsing.

Frequencies are integers summing to 2¹⁶, so every symbol keeps a non-empty sub-interval.

**Encoder and decoder must see bit-identical probabilities.** The decoder recomputes
each table from what it has decoded so far. If one quantised frequency differs, every
later symbol is decoded wrongly, and nothing detects it. :func:`quantize` is
deterministic given the float probabilities, so the models must produce the *same*
floats on both sides; see :mod:`ncomp.transformer` for why that is not automatic.
"""

from __future__ import annotations

import struct

import numpy as np

PRECISION = 32
FULL = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
TOTAL_BITS = 16
TOTAL = 1 << TOTAL_BITS


def quantize(p, total=TOTAL):
    """Integer frequencies, each ≥ 1, summing exactly to ``total``; deterministic in p.

    Returns cumulative counts of shape (..., V + 1)."""
    p = np.asarray(p, dtype=np.float64)
    V = p.shape[-1]
    f = np.floor(p * (total - V)).astype(np.int64) + 1
    short = total - f.sum(-1)  # ≥ 0: give the rounding leftover to the likeliest symbol
    idx = np.argmax(p, -1)
    np.put_along_axis(
        f, idx[..., None], np.take_along_axis(f, idx[..., None], -1) + short[..., None], -1
    )
    cum = np.zeros(p.shape[:-1] + (V + 1,), dtype=np.int64)
    np.cumsum(f, -1, out=cum[..., 1:])
    return cum


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending = 0, FULL, 0
        self.bits = []

    def _emit(self, b):
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def encode(self, lo, hi, total=TOTAL):
        if not 0 <= lo < hi <= total:
            raise ValueError(f"empty or invalid interval [{lo}, {hi}) of {total}")
        r = self.high - self.low + 1
        self.high = self.low + r * hi // total - 1
        self.low = self.low + r * lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return np.packbits(np.array(self.bits, dtype=np.uint8)).tobytes()


class Decoder:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = 0
        self.low, self.high = 0, FULL
        self.value = 0
        for _ in range(PRECISION):
            self.value = 2 * self.value + self._bit()

    def _bit(self):
        b = int(self.bits[self.pos]) if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def decode(self, cum, total=TOTAL):
        """The symbol whose cumulative range holds the current value; narrows past it."""
        r = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // r
        s = int(np.searchsorted(cum, target, side="right")) - 1
        lo, hi = int(cum[s]), int(cum[s + 1])
        self.high = self.low + r * hi // total - 1
        self.low = self.low + r * lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1
            self.value = 2 * self.value + self._bit()
        return s


# -- compression with a model ------------------------------------------------------------
# A model codes a batch of independent streams in lock-step:
#   model.start(B)            reset to B empty contexts
#   model.probs() -> (B, V)   next-symbol distributions
#   model.feed(symbols)       append one symbol to every stream
# The data is split into chunks of at most ``ctx`` symbols (None: one chunk), and
# ``group`` chunks are coded in parallel. Symbols are coded in order (group, step,
# chunk), so there is a single arithmetic-coded stream.


def _chunks(n, ctx, group):
    ctx = max(n, 1) if ctx is None else ctx
    starts = list(range(0, n, ctx)) or [0]
    return [starts[i : i + group] for i in range(0, len(starts), group)], ctx


def compress(data: bytes, model, ctx=None, group=1, return_bits=False):
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)
    enc = Encoder()
    ideal = 0.0  # Σ −log₂ p(symbol) under the quantised tables: the coder's target
    groups, ctx = _chunks(n, ctx, group)
    for starts in groups:
        lens = [min(ctx, n - s) for s in starts]
        model.start(len(starts))
        for t in range(max(lens, default=0)):
            cum = quantize(model.probs())
            # int64: as uint8, syms + 1 wraps to 0 for byte 255
            syms = np.array(
                [arr[s + t] if t < L else 0 for s, L in zip(starts, lens, strict=True)], np.int64
            )
            for b, L in enumerate(lens):
                if t < L:
                    lo, hi = int(cum[b, syms[b]]), int(cum[b, syms[b] + 1])
                    enc.encode(lo, hi)
                    ideal += TOTAL_BITS - np.log2(hi - lo)
            model.feed(syms)
    out = struct.pack(">I", n) + enc.finish()
    return (out, ideal) if return_bits else out


def decompress(blob: bytes, model, ctx=None, group=1) -> bytes:
    (n,) = struct.unpack(">I", blob[:4])
    dec = Decoder(blob[4:])
    out = np.zeros(n, dtype=np.uint8)
    groups, ctx = _chunks(n, ctx, group)
    for starts in groups:
        lens = [min(ctx, n - s) for s in starts]
        model.start(len(starts))
        for t in range(max(lens, default=0)):
            cum = quantize(model.probs())
            syms = np.zeros(len(starts), dtype=np.int64)
            for b, L in enumerate(lens):
                if t < L:
                    syms[b] = dec.decode(cum[b])
                    out[starts[b] + t] = syms[b]
            model.feed(syms)
    return out.tobytes()
