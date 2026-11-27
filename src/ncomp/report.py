"""Collect ``results/bench_*.jsonl`` into ``results/bench.md`` and ``results/position.png``.

python -m ncomp.report --out results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "bench_*.jsonl"))):
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows = load(a.out)
    sizes = [r for r in rows if "bpb" in r]
    sizes.sort(key=lambda r: -r["bpb"])
    lzma = next(r["bpb"] for r in rows if r["name"] == "lzma -9e")
    md = ["## Compressed size of the held-out stdlib files (593,029 bytes)", "",
          "Every coded row was decoded back and compared with the input.", "",
          "| compressor | bytes | bits / byte | vs lzma | encode s | decode s |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for r in sizes:
        dec = f"{r['decode_s']:.1f}" if "decode_s" in r else "—"
        md.append(f"| {r['name']} | {r['bytes']:,} | {r['bpb']:.4f} | "
                  f"{100 * (r['bpb'] / lzma - 1):+.1f}% | {r['encode_s']:.1f} | {dec} |")  # fmt: skip

    md += ["", "## Where the bits go (transformer, independent chunks)", "",
           "| model | params | train steps | cross-entropy | ideal (quantised) | actual | "
           "model, lzma'd fp16 |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        if "xent_bpb_full_chunks" in r:
            md.append(f"| {r['name']} | {r['params']:,} | {r['train_steps']:,} | "
                      f"{r['xent_bpb_full_chunks']:.4f} | {r['ideal_bpb']:.4f} | {r['bpb']:.4f} | "
                      f"{r['model_bytes_lzma_fp16']:,} B |")  # fmt: skip

    md += ["", "## Encoder and decoder must compute identically", "",
           "| model | full vs cached: max abs Δp | tables that differ | batch-of-64 vs alone: max "
           "abs Δp | tables that differ | cross-path decode: first wrong byte | bytes wrong |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        if r["name"] == "determinism":
            md.append(f"| {r['model']} | {r['full_vs_step_max_dp']:.1e} | "
                      f"{100 * r['full_vs_step_table_flips']:.1f}% | {r['batch_vs_alone_max_dp']:.1e} | "
                      f"{100 * r['batch_vs_alone_table_flips']:.1f}% | {r['crosspath_first_error']} | "
                      f"{100 * r['crosspath_wrong_fraction']:.1f}% of {r['crosspath_bytes']:,} |")  # fmt: skip
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in rows:
        if r["name"] == "position":
            ax.plot(r["bpb_by_position"], label=os.path.basename(r["model"]), lw=1.2)
    for name, ls in [("gzip -9", ":"), ("lzma -9e", "--")]:
        v = next(x["bpb"] for x in rows if x["name"] == name)
        ax.axhline(v, color="gray", ls=ls, lw=1, label=f"{name} (whole file)")
    ax.set_xlabel("byte position within a chunk")
    ax.set_ylabel("bits / byte")
    ax.set_ylim(0, 5)
    ax.set_title("Cold start: the first bytes of every chunk are expensive")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "position.png"), dpi=130)


if __name__ == "__main__":
    main()
