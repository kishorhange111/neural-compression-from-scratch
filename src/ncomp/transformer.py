"""A byte-level transformer, and how to use it for compression without corrupting data.

Training uses the usual parallel forward over a whole window. Compression needs one
byte at a time, because the decoder only knows byte t after decoding it. So
:class:`NeuralModel` steps the model incrementally with a KV cache.

**The trap.** The full forward and the cached step compute the same function, but
not the same floats. Their matmuls differ in shape, so they reduce in a different order
and differ in the last few bits. Quantised to integer frequencies, a difference of
1e-7 occasionally flips a frequency by one, and the arithmetic decoder then follows
a different interval from that point on. The encoder and decoder must therefore run
*the same computation*: the same path, the same batch composition and the same
threads. :func:`path_mismatch` measures how often the two paths disagree.

The same applies across machines: a file compressed on one CPU or GPU kernel is not
guaranteed to decode on another. Practical neural compressors either pin the kernels
or quantise the model's arithmetic to integers.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BOS = 256  # input-only token that starts every chunk


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _split(self, x):  # (B, T, d) -> (B, h, T, d/h)
        B, T, d = x.shape
        return x.view(B, T, self.h, d // self.h).transpose(1, 2)

    def forward(self, x, cache=None):
        """Full causal forward when ``cache`` is None. Otherwise x is (B, 1, d), and
        ``cache`` is a dict holding this layer's keys and values so far, updated in place."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if cache is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # cache: preallocated "k", "v" of shape (B, h, ctx, d/h), and position "t"
            t = cache["t"]
            cache["k"][:, :, t] = k[:, :, 0]
            cache["v"][:, :, t] = v[:, :, 0]
            k, v = cache["k"][:, :, : t + 1], cache["v"][:, :, : t + 1]
            a = F.scaled_dot_product_attention(q, k, v)  # one query sees every key so far
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    def __init__(self, d=192, layers=4, heads=6, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(257, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """Logits (B, T, 256) for inputs (B, T), with T ≤ ctx."""
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))

    def new_caches(self, B):
        d = self.pos.weight.shape[1]
        shape = (B, self.blocks[0].h, self.ctx, d // self.blocks[0].h)
        return [{"k": torch.zeros(shape), "v": torch.zeros(shape), "t": 0} for _ in self.blocks]

    def step(self, x, t, caches):
        """Logits (B, 256) for one new input token per stream at position t."""
        h = self.tok(x[:, None]) + self.pos.weight[t]
        for b, c in zip(self.blocks, caches, strict=True):
            c["t"] = t
            h = b(h, c)
        return self.head(self.ln(h))[:, 0]


def inputs_targets(chunks):
    """Bytes (B, T) -> inputs [BOS, b0 … b_{T−2}] and targets [b0 … b_{T−1}]."""
    bos = torch.full((chunks.shape[0], 1), BOS, dtype=torch.long)
    return torch.cat([bos, chunks[:, :-1]], 1), chunks


@torch.no_grad()
def bits_per_byte(model, chunks, batch=64):
    """Cross-entropy, in bits per byte, of full-window prediction."""
    total = 0.0
    for i in range(0, len(chunks), batch):
        x, y = inputs_targets(chunks[i : i + batch])
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum").item()
    return total / chunks.numel() / math.log(2)


def coding_group(model, budget=1.5e9):
    """How many chunks to code in parallel, so the KV cache fits in ``budget`` bytes."""
    per_chunk = 2 * len(model.blocks) * model.ctx * model.pos.weight.shape[1] * 4
    return max(1, int(budget // per_chunk))


class NeuralModel:
    """The coder's model interface (start / probs / feed) for a :class:`ByteGPT`.

    ``path="step"`` uses the KV cache, one token at a time. ``path="full"`` re-runs the
    full forward over the whole prefix at every step: the same function, computed
    differently. It is quadratic, and it exists only to demonstrate the mismatch."""

    def __init__(self, model, path="step"):
        self.m, self.path = model.eval(), path

    def start(self, B):
        self.t = 0
        self.seq = torch.full((B, 1), BOS, dtype=torch.long)
        self.caches = self.m.new_caches(B) if self.path == "step" else None
        self._p = None

    @torch.no_grad()
    def probs(self):
        if self._p is None:
            if self.path == "step":
                logits = self.m.step(self.seq[:, -1], self.t, self.caches)
            else:
                logits = self.m(self.seq)[:, -1]
            self._p = torch.softmax(logits.double(), -1).numpy()
        return self._p

    def feed(self, syms):
        self.seq = torch.cat(
            [self.seq, torch.as_tensor(np.asarray(syms), dtype=torch.long)[:, None]], 1
        )
        self.t += 1
        self._p = None


@torch.no_grad()
def path_mismatch(model, chunks):
    """Compare full-forward and cached-step probabilities on the same chunks.

    Returns the max |Δp|, and the fraction of positions whose quantised frequency table
    differs between the two paths."""
    from .coder import quantize

    x, _ = inputs_targets(chunks)
    full = torch.softmax(model(x).double(), -1).numpy()
    caches = model.new_caches(x.shape[0])
    step = np.stack(
        [
            torch.softmax(model.step(x[:, t], t, caches).double(), -1).numpy()
            for t in range(x.shape[1])
        ],
        1,
    )
    diff = float(np.abs(full - step).max())
    flips = float((quantize(full) != quantize(step)).any(-1).mean())
    return diff, flips
