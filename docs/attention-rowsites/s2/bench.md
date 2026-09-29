# S2 bench: what prob·V on int16 multiply-add saves

These are reports, not gates (plan §8 10.1). The host is the shared 4-vCPU cloud Xeon (AVX2, AVX-512BW), with GCC 13.3 -O3.
`tools/sslm_sites_bench.cpp` was built three times from the same source:

- against the base library (the S1 head, which runs the v1.9.0 loop);
- against S2's library (auto dispatch, which runs the AVX-512 body here);
- against S2's library built with `SUPERSLM_FORCE_AVX2_MATMUL` (the AVX2 body).

Each reading is a best-of-R inside one process. The builds run alternately, and the medians are taken over rounds. The raw
output is in `bench-pv.txt` and `bench-forward.txt`.

## Method 1 (the plan's): the kernel, scaled to per-token call counts

`pv` mode calls `GemmProbQ15Accumulate` at head_dim 64 (the 0.5B's) over realistic rows: every p is formed as the softmax
forms it, with weights 2^(0..12) plus jitter. It uses one KV head's value rows. The mode reports three things:

- The per-call cost at single widths.
- **Prefill** of T tokens: the calls at widths 1..T, summed, times 24 layers × 14 heads, divided by T.
- **Decode** at context C: one call at width C + 1, times 24 × 14.

The runs used best of 50 and 9 interleaved rounds. The saving is the median of the paired differences; the range is the minimum
and maximum over the rounds.

| Reading | Base (v1.9.0 loop) | S2 AVX-512 | Saved (range) | × | S2 AVX2 | Saved (range) | × | Plan §0 estimate |
|---|---|---|---|---|---|---|---|---|
| µs per call, width 128 | 5.241 | 0.266 | 4.97 | 19.7 | 0.300 | 4.94 | 17.5 | |
| µs per call, width 301 | 12.405 | 0.600 | 11.80 | 20.7 | 0.674 | 11.70 | 18.4 | |
| µs per call, width 601 | 24.788 | 1.182 | 23.59 | 21.0 | 1.323 | 23.47 | 18.7 | |
| µs per call, width 1,024 | 42.265 | 2.293 | 39.98 | 18.4 | 2.393 | 39.81 | 17.7 | |
| **Prefill T = 128**, ms/token | 0.878 | 0.053 | **0.825** (0.80–0.83) | 16.7 | 0.054 | 0.824 (0.81–0.82) | 16.2 | **0.95** |
| **Prefill T = 512** | 3.558 | 0.175 | **3.38** (3.18–3.42) | 20.3 | 0.196 | 3.36 (3.16–3.39) | 18.2 | **4.38** |
| **Prefill T = 1,024** | 7.183 | 0.434 | **6.74** (6.57–6.78) | 16.6 | 0.444 | 6.75 (6.54–6.77) | 16.2 | **8.47** |
| **Decode, context 300**, ms/token | 4.156 | 0.202 | **3.95** (3.94–4.02) | 20.6 | 0.226 | 3.92 (3.88–4.00) | 18.4 | **3.83** |
| **Decode, context 600** | 8.369 | 0.397 | **7.97** (7.84–8.02) | 21.1 | 0.444 | 7.91 (7.75–7.99) | 18.8 | **7.72** |

At width 1 there is no saving (0.056 µs on every build). A realistic width-1 row is one-hot (p = 2^15), so it fails the
int16 condition and takes the shipped loop by design.

**Against the estimate.**

- **Decode matches it:** 3.95 and 7.97 ms/token against 3.83 and 7.72, which is 103%.
- **Prefill comes in at 77–87% of it:** 0.825 / 3.38 / 6.74 against 0.95 / 4.38 / 8.47.

The kernel ratio (16–21× AVX-512, 16–19× AVX2) is above the spike's 10.7–14.2×, so the shortfall is not in the kernel. It is in
the estimate's base. The plan's T = 512 and T = 1,024 savings (4.38, 8.47) exceed this host's *entire* measured v1.9.0 prob·V
cost at those lengths (3.56, 7.18 ms/token). The estimate therefore assumed a per-key base cost about 20–25% higher than the
0.041 µs per key per head measured here: 5.24 µs at width 128.

The AVX2 body is within 2–13% of the AVX-512 body. The AVX-512 body's 32-dimension in-lane units are what put it ahead; see the
progress file's deviations.

## Method 2: the whole forward on the reduced-layer artifacts, scaled to 24 layers

This is `prefill` and `decode` mode on the 0.5B-width synthetics, with p05_l1 at sha256 f0fd4886…6ed3. It used best of 3 and
7 interleaved rounds. It compares base (auto) with S2 (auto). The forced-AVX2 build also forces AVX2 GEMMs, so it is not
comparable with the auto base at this level. Its rows are in the raw file but are not used here.

| Reading | Base | S2 | Paired saving, median (Q1, Q3) | Per layer | × 24 | Method 1 |
|---|---|---|---|---|---|---|
| Prefill, p05_l1, T = 128, ms per prompt token | 0.537 | 0.497 | 0.044 (0.028, 0.064) | 0.044 | 1.07 | 0.825 |
| Prefill, p05_l1, T = 512 | 0.710 | 0.561 | 0.141 (0.108, 0.194) | 0.141 | 3.39 | 3.38 |
| Prefill, p05_l2, T = 512 | 1.396 | 1.061 | 0.334 (0.306, 0.363) | 0.167 | 4.01 | 3.38 |
| Decode, p05_l1, context 300, 32 steps, ms per step | 1.517 | 1.356 | 0.165 (0.115, 0.187) | 0.165 | 3.97 | 3.95 |
| Decode, p05_l2, context 300, 32 steps | 3.438 | 2.654 | 0.472 (0.327, 0.932) | 0.236 | 5.7 | 3.95 |

The 1-layer readings agree with method 1:

- prefill at T = 512: 3.39 against 3.38;
- decode at context 300: 3.97 against 3.95;
- prefill at T = 128: 1.07 against 0.83, within its quartiles' noise.

The 2-layer readings run higher and wider, which is the shared host's noise. Method 1's figures are the ones reported.

These are per-layer engine figures on synthetic weights. They say nothing about any consumer's end-to-end speed, and they were
not measured on the project's reference hardware (the box's B2, §8 10.2, is box-only).
