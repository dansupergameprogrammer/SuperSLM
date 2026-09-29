# S1 bench: what the per-row tables save

Reports, not gates (plan §8 10.1). Host: the shared 4-vCPU cloud Xeon (AVX-512 here; the tables are not SIMD, so the tier does
not matter to them), GCC 13.3 -O3. `tools/sslm_sites_bench.cpp` built twice from the same source: against the base library
(`fb56397`) and against the S1 implementation commit's. Every reading is a best-of-R inside one process; the two builds are
run alternately (interleaved pairs) and the medians are over pairs. Raw output: `bench-kernel.txt`, `bench-forward.txt`.

## Method 1 (the plan's): per-site savings times per-token site counts

Kernel mode calls each site through its public entry point at the Qwen2.5-0.5B widths with realistic constants (16 distinct
rows, best of 15 batches, 5 interleaved pairs). The site call includes the funnel and the site's own allocation, which S1 leaves
alone, so the difference is the loop the table replaces.

| Site (width) | Base µs/call | S1 µs/call | Saved µs/call | Calls per token (24 layers) | Saved ms/token |
|---|---|---|---|---|---|
| `MlpActSite` (4,864) | 58.10 | 38.80 | 19.30 | 24 | 0.463 |
| `ResidualReconcileSite` (896) | 19.05 | 13.74 | 5.31 | 48 | 0.255 |
| `RmsNormSite` (896) | 8.70 | 6.95 | 1.75 | 48 | 0.084 |
| **Total** | | | | | **0.80** |

The plan estimates **0.87 ms/token** (§0, §4.1: the spike's loop-only figures 25.2 → 4.2 µs, 8.5 → 2.6 µs, 3.0 → 1.2 µs over the
same counts). Measured: **0.80 ms/token**, 92% of the estimate. The saved microseconds per call (19.3, 5.3, 1.75) are close to
the spike's (21.0, 5.9, 1.8); the site-level ratios are 1.50×, 1.39× and 1.25×, lower than the spike's loop-only 6.0×, 3.2× and
2.6× because the funnel and allocation stay.

## Method 2: whole forward on reduced-layer artifacts, scaled to 24 layers

| Reading | Base | S1 | Paired saving, median (Q1, Q3) | Per layer | × 24 |
|---|---|---|---|---|---|
| Prefill, p05_l1, T = 128, ms per prompt token (7 pairs) | 0.574 | 0.532 | 0.041 (0.021, 0.073) | 0.041 | 0.99 |
| Prefill, p05_l2, T = 128 (7 pairs) | 1.108 | 1.037 | 0.080 (0.036, 0.122) | 0.040 | 0.96 |
| Decode, p05_l1, context 300, 32 steps (7 pairs) | 1.614 | 1.481 | 0.121 (0.062, 0.157) | — | not resolved |
| Decode, p05_l1, context 300, 64 steps (7 pairs) | 1.629 | 1.567 | 0.091 (−0.003, 0.114) | — | not resolved |
| Decode, p05_l2, context 300, 64 steps (7 pairs) | 3.304 | 3.335 | −0.122 (−0.335, 0.151) | — | not resolved |

Prefill agrees with method 1 within its noise (0.96–0.99 against 0.80; the forward's per-element loops run on colder caches than
the kernel bench's 16 hot rows, which plausibly widens the saving). **Decode is not resolved on this host**: S1 saves about
0.033 ms per layer per decode token (method 1), about 2% of a 1-layer decode step, and the pair-to-pair spread here is ±10%; the
2-layer decode median even goes the wrong way. The decode saving is the same site work per token as prefill's (the plan's G9: the
same functions), so method 1's 0.80 ms/token stands for decode too; a quiet host (or the box's B2) is where a decode reading can
resolve it.
