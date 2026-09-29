# whole prefill, p06_l1.sslm, 1 layer(s): v1.9.0 base vs this branch, batched (chunk_budget=N); best of 7 interleaved runs, pins checked on every run
| build | tokens | v1.9.0 ms | branch ms | speedup | per layer, branch |
|---|---|---|---|---|---|
| auto (AVX-512 here) | 8 | 8.43 | 6.40 | 1.32 | 6.40 ms |
| auto (AVX-512 here) | 32 | 28.65 | 19.37 | 1.48 | 19.37 ms |
| auto (AVX-512 here) | 128 | 127.33 | 79.95 | 1.59 | 79.95 ms |
| forced AVX2 | 8 | 9.51 | 7.26 | 1.31 | 7.26 ms |
| forced AVX2 | 32 | 35.51 | 22.59 | 1.57 | 22.59 ms |
| forced AVX2 | 128 | 148.42 | 91.83 | 1.62 | 91.83 ms |
