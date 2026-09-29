# Decode threading D1: the bench, run here as indicative only

`tools/decode_threading_bench.cpp` (§7 B0/B1), GCC 13 Release, on the 4-vCPU cloud Xeon (2.1 GHz, AVX-512; shared with
other agents), after every build had finished. The hook is `docs/parallel_for_reference.hpp`'s std::thread pool with the
calling thread included. Timings are n = 15 interleaved off/on pairs, median and IQR, as §7 asks. **None of these is a
plan reading:** the gate figures are the box's (B0–B2, Zen 2, the real Qwen2.5-0.5B artifact), and no real 0.5B artifact
exists on this host. The artifact here is the S1–S5 synthetic one at 0.5B width (hidden 896, 14 query and 2 KV heads of
64, intermediate 4,864) with **2 layers and vocabulary 256**, so per-token figures are two layers' worth, not 24.

The production build reads the library constant (`kMinRowBytesPerTask` = 256 KiB); the `min_row_bytes 0` in its lines
means "the library's own". `--min-row-bytes` needs `sslm_decode_threading_bench_seams`.

## groups (no artifact): one decode layer's five groups at 0.5B shapes, 24 layers of synthetic weights

```
groups max_tasks 2 layers 24 n 15 serial_ms_per_layer_median 2.6247 iqr 2.4802..2.7776 split_ms_per_layer_median 1.7072 iqr 1.6430..1.7821 run_calls_per_layer 4.00 rows_equal 1 projected_ms_saved_per_token_24_layers 22.02
groups max_tasks 4 layers 24 n 15 serial_ms_per_layer_median 2.4859 iqr 2.4212..2.5865 split_ms_per_layer_median 1.2919 iqr 1.1069..1.4536 run_calls_per_layer 4.00 rows_equal 1 projected_ms_saved_per_token_24_layers 28.66
```

At 256 KiB only k + v is serial at this width (K·N = 229,376 bytes, below 2 × 256 KiB); q, o, gate + up and down thread:
4 `run` calls per layer, which is §3.1's 4L + 1 = 97 per token at 24 layers. Rows are equal to the serial run. At 4 tasks one layer's
matvecs go from 2.49 to 1.29 ms here; scaled by 24 layers that is about 29 ms per token, above §5's 9.3–14.8 ms box
projection, because this host's memory bandwidth per core and its core count differ from the box's.

## verify (B0's shape): 300-token prompt, 64 greedy tokens, no hook against bit 1 at max_tasks 1, 2, 4, 8

```
verify max_tasks 1 tokens_equal 1 blob_equal 1 run_calls 0 run_calls_per_decode_token 0.00
verify max_tasks 2 tokens_equal 1 blob_equal 1 run_calls 577 run_calls_per_decode_token 9.00
verify max_tasks 4 tokens_equal 1 blob_equal 1 run_calls 577 run_calls_per_decode_token 9.00
verify max_tasks 8 tokens_equal 1 blob_equal 1 run_calls 577 run_calls_per_decode_token 9.00
verify complete 0 mismatches
```

Tokens and the final SSB5 blob equal the no-hook run at every `max_tasks`. 9 `run` calls per token = 4L + 1 at L = 2
(§3.1's 97 at L = 24). `max_tasks` 1 makes no call: below two tasks nothing threads, the finish included.

## time (B1's shape): decode tokens/s, bit 1 off against on, hook installed both ways

```
time min_row_bytes 0 context 16 max_tasks 2 n 15 off_tok_s_median 393.695 off_iqr 384.385..415.518 on_tok_s_median 499.739 on_iqr 472.907..540.630 ratio 1.269
time min_row_bytes 0 context 16 max_tasks 4 n 15 off_tok_s_median 312.678 off_iqr 256.749..353.882 on_tok_s_median 491.031 on_iqr 462.789..524.567 ratio 1.570
time min_row_bytes 0 context 300 max_tasks 2 n 15 off_tok_s_median 334.147 off_iqr 314.825..369.296 on_tok_s_median 443.910 on_iqr 422.573..478.260 ratio 1.328
time min_row_bytes 0 context 300 max_tasks 4 n 15 off_tok_s_median 353.046 off_iqr 328.290..385.382 on_tok_s_median 505.767 on_iqr 478.315..542.741 ratio 1.433
```

Through the seam build, max_tasks 4, context 16 (B1's threshold sweep):

```
time min_row_bytes 65536 context 16 max_tasks 4 n 15 off_tok_s_median 336.820 off_iqr 315.483..360.451 on_tok_s_median 458.048 on_iqr 376.088..570.348 ratio 1.360
time min_row_bytes 262144 context 16 max_tasks 4 n 15 off_tok_s_median 380.908 off_iqr 369.704..402.787 on_tok_s_median 535.570 on_iqr 508.971..616.932 ratio 1.406
time min_row_bytes 1048576 context 16 max_tasks 4 n 15 off_tok_s_median 363.670 off_iqr 348.911..379.080 on_tok_s_median 513.607 on_iqr 489.396..531.716 ratio 1.412
```

## onetoken: one-token prompt calls, 64-token prompt, ms per token

```
onetoken max_tasks 2 n 15 off_ms_per_token_median 2.484 off_iqr 2.363..2.638 on_ms_per_token_median 1.873 on_iqr 1.833..2.094
onetoken max_tasks 4 n 15 off_ms_per_token_median 2.411 off_iqr 2.179..2.577 on_ms_per_token_median 1.574 on_iqr 1.519..1.831
```

## Reading

- Every threaded run is byte-equal to the serial one; the mechanism's `run` count matches §3.1.
- Here, bit 1 on is 1.3–1.6x the decode rate off (2 layers, vocabulary 256, so the fixed per-token cost outside the
  layers weighs far more than at 24 layers), and one-token prompt calls go from about 2.4 to 1.6 ms at 4 tasks.
- The threshold sweep does not separate 64 KiB, 256 KiB and 1 MiB on this host: the IQRs overlap. Choosing the
  constant is D2's job on the box (B1).
