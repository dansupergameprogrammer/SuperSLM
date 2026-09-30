# S3 bench: what the requant row leaf saves

These are reports, not gates (plan §8 10.1). The host is the shared 4-vCPU cloud Xeon (AVX2, AVX-512BW) with GCC 13.3 -O3.
`tools/sslm_sites_bench.cpp` gains a `requant` mode. One object was linked three ways:

- against the base library, the S2 head, which runs v1.9.0's element loop in the funnel;
- against S3's library with auto dispatch, which runs the AVX-512 row leaf here;
- against S3's `libsuperslm_avx2_forced.a`, which runs the AVX2 row leaf.

Each reading is a best-of-R inside one process. The builds run alternately, and the medians are taken over rounds. The raw
output is in `bench-requant.txt` and `bench-forward.txt`.

## Method 1 (the plan's): the funnel call, scaled to per-token call counts

`requant` mode times one whole `RequantChainChecked` call: max-abs, preflight, the element loop that S3 replaces, and the
scale fold. It uses the 0.5B's two funnel widths, 896 and 4,864, over 16 rows whose max-abs spans 2^16 to 2^31. The per-token
figure is 24 layers × (8 calls at 896 + 3 at 4,864), plus the embed's one call at 896 (G26). Prefill and decode make the same
calls per token. Only the element loop differs between builds, so the difference is the loop's saving.

The runs used best of 50 and 9 interleaved rounds. The saving is the median of the paired differences; the range is the minimum
and maximum over the rounds.

| Reading | Base (v1.9.0 loop) | S3 AVX-512 | Saved (range) | × | S3 AVX2 | Saved (range) | × | Plan §0 / §4.3 estimate |
|---|---|---|---|---|---|---|---|---|
| µs per call, width 896 | 5.219 | 1.009 | 4.15 (3.84–4.28) | 5.2 | 1.234 | 3.90 (3.46–4.08) | 4.2 | loop 2.4 → 1.1 (AVX2), 0.5 (AVX-512) |
| µs per call, width 4,864 | 31.459 | 5.608 | 25.8 (23.6–26.1) | 5.6 | 6.700 | 24.8 (22.5–24.9) | 4.7 | loop 19.8 → 6.1 (AVX2), 2.4 (AVX-512) |
| **ms per token** (prefill and decode) | 3.270 | 0.599 | **2.66** (2.52–2.69) | 5.5 | 0.720 | **2.53** (2.30–2.57) | 4.5 | **1.22** (AVX2), about 1.6 (AVX-512) |

**Against the estimate.** The saving is about twice the estimate: 2.53 ms/token on AVX2 against 1.22 (207%), and 2.66 on
AVX-512 against about 1.6. The lanes are not faster than the spike found. The difference is the base: at width 4,864 the
v1.9.0 loop's saving alone is 24.8 µs on AVX2, where the spike measured the whole loop at 19.8 µs. The plan's per-token
arithmetic reproduces the estimate from the spike's figures, 24 × (3 × 13.7 + 8 × 1.3) µs = 1.24 ms. So the error is in the
spike's base per-element cost, not in the lanes.

- **Part of the gap is a per-element call, but only a small part.** In the base, the funnel reaches `RequantTokenCodeWide`
  through a relocated out-of-line call per element (it lives in `intmath.cpp`, and the build has no LTO). The same loop built
  inside `intmath.cpp` (the forced-SSE2 library, where `RequantRowWide` runs it) measures 27.5 µs at 4,864 against the base's
  31.3, and 2.86 against 3.24 ms/token (supplementary rows in `bench-requant.txt`). That is about 12%.
- **The rest is the element code's own cost on this host**, which is higher than the spike's reading.
- **AVX2 is within 10–20% of AVX-512** (1.23 against 1.01 µs at 896; 6.70 against 5.61 at 4,864).

## Method 2: the whole forward on the reduced-layer artifacts, scaled to 24 layers

This is `prefill` and `decode` mode on the 0.5B-width synthetics, with p05_l1 at sha256 f0fd4886…6ed3. It used best of 3 and
7 interleaved rounds, and compares base (auto) with S3 (auto).

| Reading | Base | S3 | Paired saving, median (Q1, Q3) | Per layer | × 24 | Method 1 |
|---|---|---|---|---|---|---|
| Prefill, p05_l1, T = 128, ms per prompt token | 0.518 | 0.394 | 0.121 (0.115, 0.138) | 0.121 | 2.9 | 2.66 |
| Prefill, p05_l1, T = 512 | 0.583 | 0.454 | 0.123 (0.106, 0.155) | 0.123 | 3.0 | 2.66 |
| Decode, p05_l1, context 300, 32 steps, ms per step | 1.460 | 1.270 | 0.219 (0.111, 0.405) | 0.219 | 5.3 | 2.66 |
| Decode, p05_l2, context 300, 32 steps | 5.561 | 5.357 | 0.517 (−0.098, 0.753) | 0.259 | 6.2 | 2.66 |

The two prefill readings agree with method 1: 2.9 and 3.0 against 2.66, where the per-layer figure also carries the embed's
call once per token rather than once per 24 layers. The decode readings are too wide to use: their quartiles span 3–4× on the
shared host. Method 1's figures are the ones reported.

These are per-layer engine figures on synthetic weights. They say nothing about any consumer's end-to-end speed, and they were
not measured on the project's reference hardware (the box's B2, §8 10.2, is box-only).
