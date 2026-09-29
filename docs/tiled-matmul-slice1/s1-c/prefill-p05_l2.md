# whole prefill, p05_l2.sslm, 2 layer(s): v1.9.0 base vs this branch, batched (chunk_budget=N); best of 7 interleaved runs, pins checked on every run
| build | tokens | v1.9.0 ms | branch ms | speedup | per layer, branch |
|---|---|---|---|---|---|
| auto (AVX-512 here) | 8 | 16.27 | 12.42 | 1.31 | 6.21 ms |
| auto (AVX-512 here) | 32 | 53.70 | 34.32 | 1.56 | 17.16 ms |
| auto (AVX-512 here) | 128 | 229.86 | 140.91 | 1.63 | 70.46 ms |
| forced AVX2 | 8 | 18.06 | 13.11 | 1.38 | 6.56 ms |
| forced AVX2 | 32 | 65.54 | 41.12 | 1.59 | 20.56 ms |
| forced AVX2 | 128 | 268.82 | 166.94 | 1.61 | 83.47 ms |
