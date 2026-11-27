import math
import os

import numpy as np
import pytest
import torch

from ncomp.coder import TOTAL, compress, decompress, quantize
from ncomp.counts import OrderK
from ncomp.data import split
from ncomp.transformer import (
    ByteGPT,
    NeuralModel,
    bits_per_byte,
    coding_group,
    inputs_targets,
    path_mismatch,
)


class Static:
    """A fixed distribution for every position: lets the coder be tested on its own."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)

    def start(self, B):
        self.B = B

    def probs(self):
        return np.broadcast_to(self.p, (self.B, len(self.p)))

    def feed(self, syms):
        pass


def text(n, seed=0):
    rng = np.random.default_rng(seed)
    words = [b"def ", b"return ", b"self", b".", b"(", b")", b":\n", b"    ", b"x", b" = ", b"if "]
    out = b""
    while len(out) < n:
        out += words[rng.integers(len(words))]
    return out[:n]


# -- quantisation and the coder --------------------------------------------------------
def test_quantize_sums_to_total_and_keeps_every_symbol():
    rng = np.random.default_rng(0)
    for p in [np.full(256, 1 / 256), np.eye(256)[3], rng.dirichlet(np.full(256, 0.01), 50)]:
        cum = quantize(p)
        f = np.diff(cum, axis=-1)
        assert (cum[..., -1] == TOTAL).all() and (f >= 1).all()
        assert (quantize(p) == cum).all()


@pytest.mark.parametrize(
    "data",
    [b"", b"a", bytes(range(256)) * 3, bytes(5000), text(4000)],
    ids=["empty", "one", "every-byte", "zeros", "text"],
)
def test_roundtrip_static_and_adaptive(data):
    for model in [Static(np.full(256, 1 / 256)), OrderK(0), OrderK(3)]:
        assert decompress(compress(data, model), model) == data


def test_size_matches_ideal_code_length():
    """The coder spends −Σ log₂ p, plus at most a couple of bits and the 4-byte header."""
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(256))
    data = bytes(rng.choice(256, 20000, p=p).astype(np.uint8))
    blob, ideal = compress(data, Static(p), return_bits=True)
    assert 0 <= 8 * (len(blob) - 4) - ideal < 10
    entropy = -(p * np.log2(p)).sum() * len(data)
    assert abs(ideal - entropy) / entropy < 0.02


def test_skewed_source_costs_almost_nothing():
    blob = compress(bytes(10000), OrderK(0))
    assert len(blob) < 40


@pytest.mark.parametrize("ctx,group", [(7, 1), (7, 3), (100, 2), (None, 1)])
def test_chunked_and_grouped_coding(ctx, group):
    data = text(997)
    blob = compress(data, OrderK(2), ctx=ctx, group=group)
    assert decompress(blob, OrderK(2), ctx=ctx, group=group) == data


def test_wrong_probabilities_corrupt_silently():
    """Decoding with a slightly different model gives wrong bytes, not an error."""
    data = text(2000)
    blob = compress(data, OrderK(2))
    out = decompress(blob, OrderK(1))
    assert len(out) == len(data) and out != data


# -- count models ---------------------------------------------------------------------
def test_order_k_is_a_distribution_and_learns():
    m = OrderK(2)
    m.start(1)
    seq = b"abcabcabcabc"
    costs = []
    for s in seq:
        p = m.probs()[0]
        assert abs(p.sum() - 1) < 1e-12 and (p > 0).all()
        costs.append(-math.log2(p[s]))
        m.feed([s])
    assert costs[-1] < 0.5 < costs[0]


def test_priming_equals_feeding_the_prefix():
    prefix, data = text(500, 1), text(300, 2)
    a = OrderK(2, prime=prefix)
    a.start(1)
    b = OrderK(2)
    b.start(1)
    for s in prefix:
        b.feed([s])
    b.hist = [b""]  # priming learns the counts but starts a fresh context
    for s in data:
        assert np.array_equal(a.probs(), b.probs())
        a.feed([s])
        b.feed([s])
    primed = OrderK(2, prime=prefix)
    assert decompress(compress(data, primed), primed) == data


# -- the transformer ------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return ByteGPT(d=32, layers=2, heads=2, ctx=24).eval()


def test_cached_step_matches_full_forward(tiny):
    chunks = torch.randint(0, 256, (5, 24))
    diff, flips = path_mismatch(tiny, chunks)
    assert diff < 1e-5 and 0 <= flips <= 1


def test_neural_roundtrip_with_ragged_last_chunk(tiny):
    data = text(24 * 7 + 5)
    for group in (1, 3, 8):
        blob = compress(data, NeuralModel(tiny), ctx=24, group=group)
        assert decompress(blob, NeuralModel(tiny), ctx=24, group=group) == data


def test_sliding_window_roundtrip_past_the_context(tiny):
    data = text(24 * 9 + 7)  # streams of 56 bytes, more than twice the context of 24
    for keep in (0.25, 0.5):
        model = lambda k=keep: NeuralModel(tiny, slide=True, keep=k)
        blob = compress(data, model(), ctx=56, group=4)
        assert decompress(blob, model(), ctx=56, group=4) == data
    with pytest.raises(ValueError, match="slide"):
        compress(data, NeuralModel(tiny), ctx=56, group=4)


def test_ideal_code_length_is_the_cross_entropy(tiny):
    data = text(24 * 10)
    _, ideal = compress(data, NeuralModel(tiny), ctx=24, group=10, return_bits=True)
    chunks = torch.tensor(list(data)).view(10, 24)
    xent = bits_per_byte(tiny, chunks) * len(data)
    assert abs(ideal - xent) / xent < 0.01  # quantisation costs well under 1%


def test_inputs_targets_shift():
    x, y = inputs_targets(torch.tensor([[5, 6, 7]]))
    assert x.tolist() == [[256, 5, 6]] and y.tolist() == [[5, 6, 7]]


def test_coding_group_fits_budget(tiny):
    g = coding_group(tiny, budget=1e6)
    per = 2 * 2 * 24 * 32 * 4
    assert g == int(1e6 // per)


def test_training_reduces_loss():
    from ncomp.train import windows

    torch.manual_seed(0)
    m = ByteGPT(d=32, layers=1, heads=2, ctx=32)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    data = np.frombuffer(text(5000), np.uint8)
    rng = np.random.default_rng(0)
    val = windows(data, 32, 16, np.random.default_rng(1))
    before = bits_per_byte(m, val)
    for _ in range(60):
        x, y = inputs_targets(windows(data, 32, 16, rng))
        loss = torch.nn.functional.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert bits_per_byte(m, val) < before - 2


def test_split_is_by_file_and_deterministic(tmp_path):
    for i in range(60):
        d = tmp_path / f"pkg{i % 3}"
        d.mkdir(exist_ok=True)
        (d / f"m{i}.py").write_bytes(f"x = {i}\n".encode())
    (tmp_path / "notes.txt").write_bytes(b"skip me")
    tr, te, names = split(str(tmp_path))
    tr2, te2, names2 = split(str(tmp_path))
    assert (tr, te, names) == (tr2, te2, names2)
    assert 0 < len(names) < 60 and b"skip" not in tr + te
    assert len(tr) + len(te) == sum(len(f"x = {i}\n") for i in range(60))
    assert all(os.sep not in n for n in names)
