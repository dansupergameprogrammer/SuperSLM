# whole prefill, 128 tokens: independent re-measurement, 2026-09-29

Same tool (`tools/t2147_chunk_batched_pins.cpp`) built against v1.9.0's libraries and against this
branch at c3e0004, batched (chunk_budget = N), same synthetic artifacts, 7 interleaved runs per point,
pins PASS on every run. GCC 13.3.0, `-O3 -DNDEBUG`; 4-vCPU cloud Xeon @ 2.10 GHz, shared (1-minute load
1.0-2.7 during the runs). This supersedes the 128-token speedups in `prefill-*.md` as the figures to quote.

| artifact, build | v1.9.0 ms (best) | branch ms (best) | x best | x median | first reading x |
|---|---|---|---|---|---|
| p05_l1 auto (AVX-512) | 112.41 | 72.83 | 1.54 | 1.49 | 1.72 |
| p05_l1 forced AVX2 | 129.09 | 83.95 | 1.54 | 1.61 | 1.96 (v1.9.0 base 161.25 ms) |
| p05_l2 auto (AVX-512) | 225.84 | 137.95 | 1.64 | 1.57 | 1.63 |
| p05_l2 forced AVX2 | 267.84 | 163.39 | 1.64 | 1.64 | 1.61 |
| p06_l1 auto (AVX-512) | 128.44 | 79.70 | 1.61 | 1.67 | 1.59 |
| p06_l1 forced AVX2 | 145.38 | 94.07 | 1.55 | 1.48 | 1.62 |

Range at 128 tokens: 1.54-1.64x best of 7, 1.48-1.67x by median. At 8 tokens 1.25-1.42x and at 32 tokens
1.35-1.63x, in agreement with the first readings within noise.

The first reading's 1.96x does not reproduce. The branch-side time agrees (83.95 ms against 82.28 ms); the
difference is entirely in that session's v1.9.0 base, 161.25 ms per layer against 129.09 ms here and against
134 ms per layer in the 2-layer run of the same geometry. The per-GEMM table predicts a saving of about
52 ms per layer for that cell; the first reading measured 79 ms. A base arm that stays slow for a whole
session is not caught by best-of-7 within that session.

Not measured: any AMD host, MSVC or clang-cl builds, real weights, full layer depth.
