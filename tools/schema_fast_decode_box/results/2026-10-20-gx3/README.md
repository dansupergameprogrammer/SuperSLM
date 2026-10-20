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

## Jump-forward off against on, on the GPU, whole-token depth (`gpujf.*`)

`tools/schema_fast_decode_gpu_jf_bench.cpp`: the example's extraction (potion_shop_order, the scene's default utterance,
41 prompt tokens) through the GPU G5 bridges, 15 interleaved pairs after one warm-up, the first arm alternating.

- **off:** `SslmGpuSeqDecodeStepForG5Bridge` once per token, the whole token in one slice, until the walk accepts.
- **on:** the §5.6 host loop on the GPU. Whenever the walk state has a forced chain, the chain goes to the device in one
  `SslmGpuSeqPrefillSchemaContentForG5Bridge` call (sub-chunks of at most 4 tokens), and the next decode step only
  finishes. A produced token is walked but not yet run; when a chain follows it, it runs first through
  `SslmGpuSeqPrefillPromptForG5Bridge`, because the schema-content bridge checks every token against the walk state and
  takes no pre-walked prefix (the CPU verb's `prewalked_prefix` has no GPU counterpart), so the GPU cannot fuse a produced
  token with the chain after it into one submission as the CPU verb does.
- **The forced chain** is the library's own: the CPU model's forced-next table, read through the §11.0 seam
  `superslm::test::sfd::ForcedTableEntry`, followed through the SCM1 transitions from the GPU sequence's walk state. The
  CPU model is only read; no CPU decode runs. The seam build is a test build of the CPU library (its logging hooks run
  only in CPU decode, which this bench never calls).
- **Build:** the engine from `36fe4b37` (src and include are unchanged from there to the branch tip; the tip's test seam
  header is ahead of the library and does not compile with the seams on) with CMake's core source list, the GPU sources
  and this bench file; bench sha256 `4ce98f6b…`; shaders as above (`run-gpujf.ps1`).
- **Timed:** generation only, from the first call after the prompt prefill to acceptance. gpu_busy is the sum of
  `LastCallTiming().gpu_busy_ms` over the run's layer-loop submissions; the finish's logits are not in it. The model is
  mapped with the default residency, so each finish computes the full-vocabulary logits row on the host.

| | off | on |
|---|---|---|
| tokens, stop | 28, accepting | 21, accepting |
| layer submissions | 27 | 10 |
| query wall ms [interval] | 645.4 [644.4, 646.6] | 242.8 [242.3, 243.9] |
| query gpu_busy ms | 176.6 | 110.2 |
| per token: wall / gpu_busy ms | 23.05 / 6.31 | 11.56 / 5.25 |
| digest (all 15 runs equal) | `7e46330608d11704` | `8ad5ea7961656cef` |

off / on wall = 2.660 [2.646, 2.665].

- **Tokens:** both digests equal the CPU arms' (off `7e463306…`, on `8ad5ea79…`, `../2026-10-20-q6c/`): the GPU path
  produces the same tokens as the CPU in each mode, and on gives the same 21-token stream (jump-forward's standard
  tokenization of the fixed text) as on the CPU.
- **Where the time goes:** off spends about 23 ms per token for 6.3 ms of GPU work. The rest is the per-submit cost
  (about 3.7 ms, GX.3 above) and the finish, which computes all 151,936 logit rows on the host at this residency (about
  11 ms per token, B1's `finishrows` at V). On removes 17 submissions, and with them the finishes of the forced tokens.
- **The GPU finish:** `SslmGpuSeqFinishTokenForG5Bridge` computes the full-vocabulary logits row (on the device with a
  device head, otherwise on the host through `LogitsSiteParallel`) and only then applies the schema mask
  (`ApplyMaskAndArgmax`). It does not take the masked finish's legal-rows-only path, so the GPU gets none of M1's saving
  at a finish; with a device head (as the packaged example maps it) the host part of a finish is about 1.35 ms.

### The same comparison with the logits head on the device (`gpujf-dev.*`)

The packaged example maps the model with `SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE`, so each finish computes the logits row on
the device. The bench's `--device-head` flag maps it the same way; everything else is as above (bench sha256 `bf995c70…`,
`run-gpujf-dev.ps1`).

| | off, head on host | on, head on host | off, head on device | on, head on device |
|---|---|---|---|---|
| query wall ms [interval] | 645.4 [644.4, 646.6] | 242.8 [242.3, 243.9] | 342.2 [340.8, 344.3] | 201.0 [200.6, 201.7] |
| query gpu_busy ms | 176.6 | 110.2 | 174.5 | 110.0 |
| layer submissions | 27 | 10 | 27 | 10 |
| tokens | 28 | 21 | 28 | 21 |
| per token wall ms | 23.05 | 11.56 | 12.22 | 9.57 |
| digest | `7e46330608d11704` | `8ad5ea7961656cef` | `7e46330608d11704` | `8ad5ea7961656cef` |

off / on wall: 2.660 [2.646, 2.665] with the head on the host; 1.707 [1.693, 1.711] with the head on the device.

With the head on the device, a token's finish no longer pays the host's full-vocabulary logits row (about 11 ms), so the
off arm drops from 645 to 342 ms and jump-forward's lead narrows to 1.71x. What it still saves at this residency is 17
layer submissions (about 3.7 ms each, GX.3 above), the forced tokens' finishes on the device, and 7 tokens of the stream.
The tokens are the same at both residencies.
