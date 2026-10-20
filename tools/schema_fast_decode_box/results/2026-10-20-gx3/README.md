# GX.3, headless: the GPU backend's per-submit fixed cost, 2026-10-20

Plan §7.1 step GX.3 asks whether a GPU slice that runs M token rows costs clearly less than M one-row slices. The
existing GPU path cannot run M token rows of one sequence through 4 layers as one slice:

- the prompt and schema-content bridges always run to full depth, and split a call into sub-chunks of at most
  `kT2169TdrSafeMaxChunkTokens` = 4 tokens, one submission each;
- inside a sub-chunk each token's own full-depth dispatch body is recorded in turn (token-outer, layer-inner), so every
  GEMM still runs one row per token, and the GPU work grows linearly with M;
- `sslm_decode_step_batch_gpu` slices M different sequences, and submits and drains each one on its own.

So the only saving a multi-token submission can show is the per-submit fixed cost (recording, submission, fence and
readback), and these readings measure that, with the existing API and no engine change
(`tools/schema_fast_decode_gx3_bench.cpp`).

- **Box:** AMD Ryzen 9 3950X, NVIDIA RTX 2080 SUPER, Windows 10. Headless (no rendering on the GPU).
- **Source:** SuperSLM `fd86ba222eaf311bad9058d1fb66362ba8997ab8`; bench built with build.bat's `t2113_b7` source list and
  shaders compiled beside it (`run-gx3.ps1`), bench sha256 `92e40b3c…`.
- **Model:** `qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm` (24 layers), sha256 `9ccf7e378bbe47aa0ffd3810aa82875926cbe51af1bdfa08393c3fae657463da`.
- **Method:** 15 interleaved pairs per M after one warm-up, the first arm alternating per pair and the sizes rotated per
  round. Wall time is the whole call. `gpu_busy_ms` is `superslm_gpu::LastCallTiming()` after each submission (the GPU's
  own timestamp window), summed over the one-token arm. Ratios are per pair, as median and the n = 15 sign interval.

## Chunk: M tokens in one prompt-bridge call against M one-token calls (full depth, 24 layers)

Each sample starts from a reset sequence with the same 41-token prefix (untimed). M ≤ 4 is one submission.

| M | chunk wall / gpu_busy ms | M one-token calls wall / gpu_busy ms | wall ratio [interval] | gpu_busy ratio [interval] | wall − gpu_busy: chunk / singles ms |
|---|---|---|---|---|---|
| 1 | 10.83 / 6.22 | 10.80 / 6.23 | 1.006 [0.994, 1.018] | 0.999 [0.993, 1.003] | 4.64 / 4.58 |
| 2 | 18.05 / 12.50 | 21.74 / 12.50 | 0.832 [0.822, 0.840] | 1.000 [0.997, 1.002] | 5.55 / 9.16 |
| 3 | 25.20 / 18.77 | 32.60 / 18.72 | 0.772 [0.769, 0.782] | 1.001 [0.999, 1.003] | 6.54 / 13.88 |
| 4 | 32.48 / 25.09 | 43.64 / 24.97 | 0.746 [0.742, 0.750] | 1.005 [1.002, 1.008] | 7.36 / 18.60 |

The GPU time is the same either way: 6.2 ms of GPU per token at full depth, linear in M. Only the time outside the GPU's
window falls. It is about 4.6 ms per one-token call, and in a chunk about 4.6 ms plus 0.9 ms per further token, so each
submission avoided saves about 3.7 ms.

## Batch: sslm_decode_step_batch_gpu over M sequences against M one-row slices (4 layers each)

Every sequence is parked at layer 20 (untimed), so a batch-wide budget of M × 4 layers gives each sequence exactly
layers [20, 24), the same layers the one-row arm times.

| M | batch wall ms | M one-row slices wall / gpu_busy ms | wall ratio [interval] |
|---|---|---|---|
| 1 | 4.72 | 4.72 / 1.15 | 1.006 [0.954, 1.025] |
| 2 | 9.37 | 9.36 / 2.29 | 0.997 [0.986, 1.015] |
| 3 | 13.98 | 13.81 / 3.40 | 1.009 [0.999, 1.022] |
| 4 | 18.57 | 18.67 / 4.49 | 0.996 [0.988, 1.014] |
| 5 | 23.18 | 23.13 / 5.60 | 1.000 [0.983, 1.021] |
| 6 | 27.82 | 27.77 / 6.72 | 1.010 [0.996, 1.024] |

The batch call saves nothing, as its implementation predicts (one submission and drain per sequence). It gives the
same fixed cost from the other side: a 4-layer one-row slice takes 4.72 ms of wall time for 1.15 ms of GPU time,
about 3.6 ms outside the GPU's window per slice. The batch arm reports no gpu_busy, because `LastCallTiming` keeps
only the last of its M submissions.

## What this says

Headless, a GPU submission costs about 3.6-3.7 ms beyond its GPU work on this box: about three times the GPU work of a
4-layer slice of this model. Batching token rows into one submission would recover that fixed cost and nothing else,
until the GPU path gains M-row dispatches; the GPU work per token does not change with M today. These are headless readings: in-game the slice's GPU window also
holds the scene's own rendering (the README's 3.4 ms at 4 layers), and the per-submit cost inside a running game was
not measured here.

## Slice depth: one one-row slice at 4, 8, 12 and 24 layers (`gx3-slices.*`)

The README's GPU columns run 4, 8, 12 and 24 layers per frame. The bench's `--slices-only` mode times one
`sslm_decode_step_gpu` slice of L layers (layers [24 − L, 24), after an untimed run to 24 − L) with its blocking
`sslm_gpu_ready`, against that slice's own `gpu_busy_ms`; L rotates per round, 15 rounds after one warm-up. Built from
`tools/schema_fast_decode_gx3_bench.cpp` as committed with these results (bench sha256 `c20af5d4…`, `run-gx3-slices.ps1`).

| L (layers per slice) | wall ms [interval] | gpu_busy ms [interval] | wall − gpu_busy ms [interval] | slices per token | wall per token ms | GPU share of wall |
|---|---|---|---|---|---|---|
| 4 | 4.70 [4.58, 4.78] | 1.11 [1.10, 1.13] | 3.60 [3.47, 3.65] | 6 | 28.2 | 24 % |
| 8 | 5.87 [5.80, 5.94] | 2.19 [2.18, 2.22] | 3.66 [3.58, 3.77] | 3 | 17.6 | 37 % |
| 12 | 7.14 [7.05, 7.30] | 3.32 [3.29, 3.37] | 3.85 [3.80, 3.89] | 2 | 14.3 | 46 % |
| 24 | 10.93 [10.84, 11.08] | 6.57 [6.49, 6.75] | 4.34 [4.27, 4.42] | 1 | 10.9 | 60 % |

(Wall per token is slices per token × the slice's median wall time; GPU share is gpu_busy / wall.)

The fixed cost outside the GPU's window is nearly flat across depth (3.6 to 4.3 ms per slice; it grows a little with
L, in recording and readback), while the GPU work is about 0.27 ms per layer. So the slice count per token, not the
layers, sets most of a headless token's cost: a whole token as one 24-layer slice takes 10.9 ms of wall time against
28.2 ms as six 4-layer slices, for the same 6.6-6.7 ms of GPU work. In a game at a frame cap, one slice per frame means
the slice count also sets the frames per token; the in-game frame-time check measures that separately.
