## Compressed size of the held-out stdlib files (593,029 bytes)

Every coded row was decoded back and compared with the input.

| compressor | bytes | bits / byte | vs lzma | encode s | decode s |
|---|---|---|---|---|---|
| order-0 counts, adaptive | 347,690 | 4.6904 | +180.8% | 116.7 | 101.3 |
| order-2 counts, primed on train | 189,139 | 2.5515 | +52.8% | 54.4 | 49.8 |
| order-2 counts, adaptive | 186,376 | 2.5142 | +50.5% | 141.5 | 128.6 |
| gzip -9 | 145,413 | 1.9616 | +17.4% | 0.2 | — |
| ByteGPT 128×3, ctx 256 | 139,456 | 1.8813 | +12.6% | 49.3 | 41.5 |
| ByteGPT 128×3, sliding, keep 0.25 | 135,950 | 1.8340 | +9.8% | 75.5 | 102.1 |
| ByteGPT 128×3, sliding, keep 0.5 | 135,851 | 1.8326 | +9.7% | 84.4 | 84.0 |
| order-4 counts, adaptive | 135,014 | 1.8213 | +9.0% | 102.8 | 85.2 |
| lzma -9e | 123,816 | 1.6703 | +0.0% | 0.5 | — |
| bz2 -9 | 123,739 | 1.6692 | -0.1% | 0.1 | — |

## Where the bits go (transformer, independent chunks)

| model | params | train steps | cross-entropy | ideal (quantised) | actual | model, lzma'd fp16 |
|---|---|---|---|---|---|---|
| ByteGPT 128×3, ctx 256 | 693,760 | 2,000 | 1.8772 | 1.8812 | 1.8813 | 1,248,236 B |

## Encoder and decoder must compute identically

| model | full vs cached: max abs Δp | tables that differ | batch-of-64 vs alone: max abs Δp | tables that differ | cross-path decode: first wrong byte | bytes wrong |
|---|---|---|---|---|---|---|
| results/small | 1.5e-05 | 5.2% | 3.1e-06 | 5.5% | 5 | 90.6% of 2,048 |
