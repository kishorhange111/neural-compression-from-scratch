"""Benchmark: neural compression against gzip, bz2, lzma and count models.

    python -m ncomp.bench --models results/small results/large --out results

Every compressed size here is real: the bytes were arithmetic-coded, decoded back and
compared with the input. The bench also reports:
- the coder's overhead over the ideal −Σ log₂ p;
- the model's cross-entropy (from the parallel forward);
- the cost of shipping the model;
- why encoder and decoder must use the same computation path.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import math
import os
import sys
import time

import numpy as np
import torch

from .coder import compress, decompress, quantize
from .counts import OrderK
from .data import split
from .train import test_chunks
from .transformer import (
    ByteGPT,
    NeuralModel,
    bits_per_byte,
    coding_group,
    inputs_targets,
    path_mismatch,
)


def load(path):
    s = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)
    m = ByteGPT(**s["cfg"])
    m.load_state_dict(s["model"])
    return m.eval(), s["cfg"]


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t


def run_coder(name, data, make, ctx=None, group=1, extra=None):
    (blob, ideal), te = timed(compress, data, make(), ctx=ctx, group=group, return_bits=True)
    out, td = timed(decompress, blob, make(), ctx=ctx, group=group)
    assert out == data, f"{name}: round trip failed"
    n = len(data)
    return {"name": name, "bytes": len(blob), "bpb": 8 * len(blob) / n, "ideal_bpb": ideal / n,
            "encode_s": te, "decode_s": td, "roundtrip": True, **(extra or {})}  # fmt: skip


@torch.no_grad()
def position_bpb(model, chunks, batch=64):
    """Mean bits for the byte at each position of a chunk: the cost of a cold start."""
    total = 0
    for i in range(0, len(chunks), batch):
        x, y = inputs_targets(chunks[i : i + batch])
        lp = torch.log_softmax(model(x).double(), -1).gather(-1, y[..., None])[..., 0]
        total = total + (-lp / math.log(2)).sum(0)
    return (total / len(chunks)).numpy()


@torch.no_grad()
def batch_mismatch(model, chunks):
    """The same chunk stepped alone vs inside a batch: do the floats agree?"""

    def run(x):
        caches = model.new_caches(x.shape[0])
        return np.stack([torch.softmax(model.step(x[:, t], t, caches).double(), -1).numpy()
                         for t in range(x.shape[1])], 1)  # fmt: skip

    x, _ = inputs_targets(chunks)
    together = run(x)[0]
    alone = run(x[:1])[0]
    return float(np.abs(together - alone).max()), float(
        (quantize(together) != quantize(alone)).any(-1).mean()
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["results/small"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-counts", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    tr, te, names = split()
    n = len(te)
    rows = []
    print(f"test: {n:,} bytes in {len(names)} files; train: {len(tr):,} bytes", flush=True)

    def add(r):
        rows.append(r)
        print(json.dumps(r), flush=True)

    # 1. general-purpose compressors, at their strongest settings
    for name, f in [("gzip -9", lambda d: gzip.compress(d, 9, mtime=0)),
                    ("bz2 -9", lambda d: bz2.compress(d, 9)),
                    ("lzma -9e", lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME))]:  # fmt: skip
        blob, t = timed(f, te)
        add({"name": name, "bytes": len(blob), "bpb": 8 * len(blob) / n, "encode_s": t})

    # 2. adaptive count models (one stream, unbounded history), then a primed one
    if not a.skip_counts:
        for k in (0, 2, 4):
            add(run_coder(f"order-{k} counts, adaptive", te, lambda k=k: OrderK(k)))
        primed, tp = timed(OrderK, 2, tr)
        add(run_coder("order-2 counts, primed on train", te, lambda: primed,
                      extra={"prime_s": tp}))  # fmt: skip

    # 3. the transformers
    for path in a.models:
        if not os.path.exists(os.path.join(path, "model.pt")):
            print(f"skip {path}: no model.pt", flush=True)
            continue
        model, cfg = load(path)
        ctx = cfg["ctx"]
        chunks = test_chunks(np.frombuffer(te, np.uint8), ctx)
        params = sum(p.numel() for p in model.parameters())
        fp16 = b"".join(p.detach().half().numpy().tobytes() for p in model.parameters())
        model_bytes = len(lzma.compress(fp16, preset=9))
        steps = json.loads(open(os.path.join(path, "train.jsonl")).read().splitlines()[-1])["step"]
        xent = bits_per_byte(model, chunks)
        group = coding_group(model)
        r = run_coder(f"ByteGPT {cfg['d']}×{cfg['layers']}, ctx {ctx}", te,
                      lambda m=model: NeuralModel(m), ctx=ctx, group=group,
                      extra={"params": params, "model_bytes_lzma_fp16": model_bytes, "group": group,
                             "train_steps": steps, "xent_bpb_full_chunks": xent, "ctx": ctx})  # fmt: skip
        add(r)
        pos = position_bpb(model, chunks)
        add({"name": "position", "model": path, "bpb_by_position": pos.round(4).tolist()})
        d, flips = path_mismatch(model, chunks[:64])
        bd, bflips = batch_mismatch(model, chunks[:64])
        # encode with one path, decode with the other: how far does decoding get?
        seg = te[: 8 * ctx]
        blob = compress(seg, NeuralModel(model, "full"), ctx=ctx, group=8)
        bad = decompress(blob, NeuralModel(model, "step"), ctx=ctx, group=8)
        wrong = np.frombuffer(bad, np.uint8) != np.frombuffer(seg, np.uint8)
        first = int(np.argmax(wrong)) if wrong.any() else None
        add({"name": "determinism", "model": path, "full_vs_step_max_dp": d,
             "full_vs_step_table_flips": flips, "batch_vs_alone_max_dp": bd,
             "batch_vs_alone_table_flips": bflips, "crosspath_bytes": len(seg),
             "crosspath_wrong_fraction": float(wrong.mean()), "crosspath_first_error": first})  # fmt: skip

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "bench.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
