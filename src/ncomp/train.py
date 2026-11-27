"""Train a :class:`ByteGPT` on the stdlib split. Resumable, with a wall-clock limit.

    python -m ncomp.train --out results/small --d 128 --layers 3 --heads 4 --steps 1500
    python -m ncomp.train --out /content/out/large --d 384 --layers 6 --heads 6 --ctx 512 \
        --batch 64 --steps 20000 --device cuda --minutes 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import split
from .transformer import ByteGPT, bits_per_byte, inputs_targets


def windows(data, ctx, n, rng):
    idx = rng.integers(0, len(data) - ctx, n)
    return torch.from_numpy(np.stack([data[i : i + ctx] for i in idx]).astype(np.int64))


def test_chunks(data, ctx):
    n = len(data) // ctx
    return torch.from_numpy(data[: n * ctx].reshape(n, ctx).astype(np.int64))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--minutes", type=float, default=1e9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    tr, te, _ = split()
    tr, te = np.frombuffer(tr, np.uint8), np.frombuffer(te, np.uint8)
    val = test_chunks(te, a.ctx)[:256].to(a.device)  # a fixed slice of test, for curves

    cfg = {"d": a.d, "layers": a.layers, "heads": a.heads, "ctx": a.ctx}
    model = ByteGPT(**cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    ck = os.path.join(a.out, "ckpt.pt")
    step, rng = 0, np.random.default_rng(a.seed)
    if os.path.exists(ck):
        s = torch.load(ck, map_location=a.device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, rng = s["step"], s["rng"]
        print(f"resumed at step {step}", flush=True)
    amp = a.device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    warm = min(200, a.steps // 10)

    def lr_at(i):  # linear warm-up, cosine to 10%
        if i < warm:
            return a.lr * (i + 1) / warm
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / (a.steps - warm))))

    def save():
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "rng": rng}, ck
        )
        torch.save({"cfg": cfg, "model": {k: v.cpu() for k, v in model.state_dict().items()}},
                   os.path.join(a.out, "model.pt"))  # fmt: skip

    t0, log = time.time(), open(os.path.join(a.out, "train.jsonl"), "a")
    print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = inputs_targets(windows(tr, a.ctx, a.batch, rng))
        x, y = x.to(a.device), y.to(a.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            v = bits_per_byte(model, val)
            model.train()
            rec = {"step": step, "train_bpb": loss.item() / math.log(2), "val_bpb": v,
                   "minutes": (time.time() - t0) / 60}  # fmt: skip
            log.write(json.dumps(rec) + "\n")
            log.flush()
            tr_bpb = rec["train_bpb"]
            print(f"[{rec['minutes']:.1f}m] step {step} train {tr_bpb:.3f} val {v:.3f}", flush=True)
            save()
            if (time.time() - t0) / 60 > a.minutes:
                print("time limit", flush=True)
                break
    save()
    if step >= a.steps:
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
