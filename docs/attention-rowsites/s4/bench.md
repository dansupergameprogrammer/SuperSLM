# S4 bench: what the guarded softmax saves

These are reports, not gates (plan §8 10.1). The host is the shared 4-vCPU cloud Xeon (AVX2, AVX-512BW) with GCC 13.3 -O3.
`tools/sslm_sites_bench.cpp` gains a `softmax` mode. One object was linked three ways:

- against the base library, the S3 head, which runs v1.9.0's softmax body;
- against S4's library with auto dispatch, which runs the AVX-512 body here;
- against S4's `libsuperslm_avx2_forced.a`, which runs the AVX2 body. That library also carries the test counters (one relaxed
  atomic increment per row), as S3's AVX2 reading did.

Each reading is a best-of-30 inside one process. The builds ran alternately for 9 rounds. The saving is the median of the paired
differences; the range is the minimum and maximum over the rounds. The raw output is in `bench-softmax.txt`.

## Method: the softmax call, at the forward's widths and constants

`softmax` mode times `SoftmaxRowQ15` on rows with the forward's own constants: `IExpScaleConstants` with the format-30
coefficients the forward passes, kept when q_ln2 falls in [347, 944]. That is the range cell 11.1(d)'s data term measured on the
0.5B-width synthetic (`softmax-data-terms.txt`). The scores are spread over about 16·q_ln2. Every row is inside the guard, as all
2,240 rows in 11.1(d)'s windows are. The per-token figure is §6's: 24 layers × 14 query heads, one call per head per token.
Prefill of T tokens sums the calls at widths 1..T and divides by T. Decode at context C is one call at width C + 1.

| Reading | Base (v1.9.0) | S4 AVX-512 | Saved (range) | × | S4 AVX2 | Saved (range) | × | Plan §0 (AVX2) |
|---|---|---|---|---|---|---|---|---|
| µs per call, width 1 | 0.026 | 0.086 | −0.059 | 0.31 | 0.059 | −0.032 | 0.44 | |
| µs per call, width 128 | 0.921 | 0.237 | 0.683 (0.63–0.71) | 3.9 | 0.277 | 0.644 (0.60–0.66) | 3.3 | |
| µs per call, width 512 | 3.798 | 0.900 | 2.894 (2.66–2.96) | 4.2 | 1.062 | 2.733 (2.48–2.82) | 3.6 | |
| µs per call, width 1,024 | 7.554 | 1.856 | 5.317 (5.07–5.81) | 4.1 | 2.113 | 5.247 (4.95–5.56) | 3.6 | |
| **Prefill T = 128**, ms/token | 0.160 | 0.060 | **0.097** (0.088–0.120) | 2.7 | 0.058 | **0.097** (0.085–0.108) | 2.7 | **0.15** |
| **Prefill T = 512** | 0.651 | 0.175 | **0.476** (0.39–0.58) | 3.7 | 0.188 | **0.464** (0.41–0.56) | 3.5 | **0.54** |
| **Prefill T = 1,024** | 1.307 | 0.363 | **0.909** (0.81–1.29) | 3.6 | 0.390 | **0.912** (0.79–1.26) | 3.4 | **1.15** |
| **Decode, context 300**, ms/token | 0.747 | 0.202 | **0.544** (0.45–0.68) | 3.7 | 0.219 | **0.528** (0.46–0.66) | 3.4 | **0.53** |
| Decode, context 600 | 1.491 | 0.384 | 1.096 (0.95–1.39) | 3.9 | 0.423 | 1.068 (0.91–1.35) | 3.5 | 1.07 (§6) |

**Against the estimate.** Decode matches: 0.53 ms/token on AVX2 at context 300, as estimated, and 1.07 at 600, as §6 estimates.
Prefill falls short: 0.10 / 0.46 / 0.91 against 0.15 / 0.54 / 1.15, that is 65%, 86% and 79%.

- **§0's prefill and decode figures do not use the same per-element saving.** Per token at full depth, prefill at T touches
  336·(T + 1)/2 elements and decode at context 300 touches 336·301. The estimates then imply 6.9 ns saved per element at T = 128,
  6.3 at 512 and 6.7 at 1,024, but 5.2 in decode. Measured here, the AVX2 body saves about 5.1–5.3 ns per element on long rows
  (base 7.4 ns, S4 2.1 ns). That agrees with the decode estimate and sits below the prefill ones. The spike's 7.2–9.0 → 2.7 ns
  per element gives 4.5–6.3, so the prefill figures look taken from the top of that range or above it.
- **Short rows pay a fixed cost per row.** It is the guard's pass over the scores and two 64-bit divides (the z reciprocal and
  the p reciprocal). A width-1 row is 0.03 µs slower than v1.9.0 on AVX2 and 0.06 µs on AVX-512. Prefill at T = 128 averages
  width 64, so the fixed cost takes a larger share there. This is why T = 128 shows the largest shortfall. The total cost of the
  short-row slowdown is included in the prefill figures above; no prefill length measured here is slower than the base.
- **AVX-512 is barely ahead of AVX2** (at most 10–15% per call; equal in prefill). The per-row work outside the vector loops
  (the guard's scalar max pass and the two divides) does not get wider.

No whole-forward reading is reported for S4. On the reduced-layer artifacts the per-layer saving is about 0.02 ms per token (0.46
/ 24). That is below the forward bench's run-to-run spread on this host (S3's `bench.md` method 2: quartile widths of 0.03–0.3
ms).

These are engine figures on synthetic constants from the forward's measured range. They say nothing about any consumer's
end-to-end speed, and they were not measured on the project's reference hardware (the box's B2, §8 10.2, is box-only).
