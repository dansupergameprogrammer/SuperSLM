# S5 bench: what the Q31 score row saves

These are reports, not gates (plan §8 10.1). The host is the shared 4-vCPU cloud Xeon (AVX2, AVX-512BW) with GCC 13.3 -O3.
No real Qwen3 artifact is available here: a QK-norm artifact is still refused at map time (plan R4), so the real-artifact
run stays the box's B1. Two synthetic readings stand in for it.

1. **Kernel**, the plan's own §0 method. `tools/sslm_sites_bench.cpp` gains a `q31` mode. It uses one query head at
   head_dim 128 (Qwen3-0.6B's) over one KV head's key rows, with ratios in [2^29, 2^31] (inside the loader's range). In one
   process it times the layer loops' v1.9.0 per-key `QkQ31Score` loop and `QkQ31ScoreRow`, alternating. Both paths run
   the same tier, so their difference is S5's own effect. It checks equality first (0 mismatches in all 18 runs).
   Per token at Qwen3-0.6B depth, 28 layers × 16 query heads make one call per head per token. Prefill of T tokens sums
   the calls at widths 1..T and divides by T. Decode at context C is one call at width C + 1.
2. **Forward, one layer at Qwen3-0.6B width.** This is a scratch probe (`probe_q31_forward.cpp`, not built by CMake). It is
   the in-tree QK-norm fixture (`tests/support/qk_attention_fixture.h`) re-parameterised to hidden 1,024, 16 query
   heads over 8 KV heads, head_dim 128, intermediate 3,072 and context_cap 1,024. It uses the fixture's own constants,
   except hidden gain 4,096 and q/k-norm gains in [2,048, 4,096]; with larger gains a later funnel is out of domain at
   T = 1,024. It runs one `RunLayerLoopChunkBatched` over T positions and returns Ok at every T measured. It was linked
   four ways: the S4 head (v1.9.0's per-key loop) and S5, each with auto dispatch (AVX-512) and forced AVX2. **All four
   emit the same output codes and scales at T = 128, 512 and 1,024**, which is an extra bit-identity check at Qwen3 width.

Each kernel reading is a best-of-30. The builds ran alternately for 9 rounds, and the saving is the median of the paired
differences, with the min–max range in brackets. Each forward reading is a best-of-5 (best-of-3 at T = 1,024) over 5
alternating rounds. The raw output is in `bench-q31.txt` and `bench-forward.txt`.

## Kernel: per (head × key), and per token at Qwen3-0.6B depth

| Reading | AVX2: per-key | AVX2: row | Saved (range) | × | AVX-512: per-key | AVX-512: row | Saved | × | Plan §0 (AVX2) |
|---|---|---|---|---|---|---|---|---|---|
| ns per (head × key), width 1 | 415 | 278 | 139 (98–163) | 1.5 | 326 | 388 | −61 | 0.84 | |
| width 8 | 419 | 32.2 | 387 | 13 | 329 | 46.8 | 283 | 7.0 | |
| width 128 | 420 | 14.4 | 406 | 29 | 332 | 13.6 | 317 | 24 | 376–385 → 16.5–17.6 (spike, head_dim 128) |
| width 1,024 | 408 | 13.5 | 395 | 30 | 332 | 12.8 | 321 | 26 | |
| **Prefill T = 128**, ms/token | 11.48 | 0.47 | **11.02** (10.5–11.7) | 24 | 9.47 | 0.47 | **9.00** (7.6–9.2) | 20 | **10.9 → 0.5** |
| **Prefill T = 512** | 47.40 | 1.66 | **45.74** (40.5–47.7) | 28 | 38.06 | 1.60 | **36.51** | 24 | (43.6 → 2.0 by the same arithmetic) |
| **Prefill T = 1,024** | 97.58 | 3.23 | **94.28** (77.7–99.4) | 30 | 76.14 | 3.11 | **73.45** (69.4–74.0) | 25 | **87 → 3.9** |
| Decode, context 300, ms/token | 57.05 | 1.88 | 55.10 (44.2–55.6) | 30 | 44.58 | 1.78 | 42.81 | 25 | |
| Decode, context 600 | 109.72 | 3.72 | 105.99 | 30 | 89.01 | 3.54 | 85.48 | 25 | |

## Forward: one layer at Qwen3-0.6B width, ms per prompt token

| T | AVX2: base | AVX2: S5 | Saved (range) | × 28 layers | AVX-512: base | AVX-512: S5 | Saved (range) | × 28 layers |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.955 | 0.533 | 0.421 (0.39–0.47) | 11.8 | 0.782 | 0.437 | 0.353 (0.33–0.36) | 9.9 |
| 512 | 2.373 | 0.636 | 1.663 (1.64–1.87) | 46.6 | 1.906 | 0.515 | 1.378 (1.35–1.51) | 38.6 |
| 1,024 | 4.530 | 0.875 | 3.657 (3.03–3.94) | 102 | 3.649 | 0.630 | 3.019 (2.91–3.26) | 84.5 |

The "× 28 layers" column scales the one-layer saving to Qwen3-0.6B's depth. It is an extrapolation, not a 28-layer
reading.

## Against the estimate

- **The estimate holds on AVX2 and is exceeded at long prompts.** §0 has 10.9 → 0.5 at T = 128 and 87 → 3.9 at T = 1,024.
  Measured on AVX2 here: 11.5 → 0.47 and 97.6 → 3.23. The row kernel costs 13.5–14.4 ns per (head × key) on long rows,
  a little under the spike's 16.5–17.6. The per-key loop costs 408–420 ns, a little over the spike's 376–385. So the
  saving (11.0 and 94.3 ms per token) is at or above §0's implied 10.4 and 83. The one-layer forward agrees with the
  kernel: 0.42 and 3.66 ms per layer, which is 11.8 and 102 ms at 28 layers.
- **On AVX-512 the base is cheaper, so less is saved.** The shipped per-key AVX-512 tier costs about 330 ns per key.
  The row kernel is barely faster than on AVX2 (12.8 vs 13.5 ns). It saves 9.0 / 36.5 / 73.4 ms per token.
- **A one-key row on AVX-512 is 0.06 µs slower than the per-key call.** The body packs a whole 16-key block (15 zero
  rows) and builds the limbs for 128 channels, whatever the width. On AVX2 (8-key blocks) the one-key row is still
  faster than the per-key AVX2 call. The cost is paid once per head for the first prompt token only, and it is included
  in the prefill figures.
- **§6's whole-prefill figure (≈ 140 → ≈ 57 ms per token at T = 1,024) is not checked here.** It needs a 28-layer
  forward on real weights, which is B1/B2 on the box.

These are engine figures on synthetic weights. They say nothing about any consumer's end-to-end speed, and they were not
measured on the project's reference hardware (the box's B2, §8 10.2, is box-only).
