# whole prefill, p05_l1.sslm, 1 layer(s): v1.9.0 base vs this branch, batched (chunk_budget=N); best of 7 interleaved runs, pins checked on every run
| build | tokens | v1.9.0 ms | branch ms | speedup | per layer, branch |
|---|---|---|---|---|---|
| auto (AVX-512 here) | 8 | 7.32 | 6.10 | 1.20 | 6.10 ms |
| auto (AVX-512 here) | 32 | 26.68 | 18.18 | 1.47 | 18.18 ms |
| auto (AVX-512 here) | 128 | 117.77 | 68.32 | 1.72 | 68.32 ms |
| forced AVX2 | 8 | 9.18 | 6.70 | 1.37 | 6.70 ms |
| forced AVX2 | 32 | 34.90 | 21.43 | 1.63 | 21.43 ms |
| forced AVX2 | 128 | 161.25 | 82.28 | 1.96 | 82.28 ms |

Note (2026-09-29): the forced-AVX2 128-token row (1.96) did not reproduce; its v1.9.0 base read slow for the whole session. A rerun gives 129.09 → 83.95 ms, 1.54×. See `prefill-remeasure-2026-09-29.md`; this file is kept as recorded.
