# One-thread MSVC A/B: the J1-X build against the Q6 C fix, 2026-10-21

- **A** = `2af41659` (the Q6 C fix: the unsplit path is the pre-Q6 C single batched call). Its bench is the fresh build
  from the 2026-10-20 ab-fix run (`../2026-10-20-q6c/ab/ab-fix/`), sha256 `b1260b8f…`.
- **B** = `f1ca2e33` (TF and J1-X), built fresh for this run (MSVC 19.44, Release, Ninja). Its bench is
  `tools/schema_fast_decode_bench.cpp` at f1ca2e33 plus the `on:exact` arm (committed with these results), sha256
  `654bb91e…`. The patch adds the arm only; the off, on and prefill rows run unchanged code.
- **Box and model:** AMD Ryzen 9 3950X, Windows 10; `qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm`, sha256 `9ccf7e37…`, passed
  explicitly to every process.
- **Method** (`run-abj.ps1`): one thread (no hook), one process per arm per row per round, 15 rounds after one discarded
  warm-up, the arm order and the row order rotated per round. Rows: B2's extraction with jump-forward off and on
  (canonical), a plain `sslm_prefill` of 7 and of 16 tokens at the example prompt's depth (`pass --sizes 7,16
  --no-one-row`), and on B only the extraction with jump-forward on in exact mode (`on:exact`: `sslm_jump_options` with
  `SSLM_JUMP_EXACT` and the canonical chain as the draft). Ratios are B / A per round, median and the n = 15 sign interval.
- **Load:** the box was clear (no editor, build, build lock or U3 driver). The load sample taken at the start, before
  B's build, read 77 %; afterwards the box read 2 % with only desktop applications running. The rounds are interleaved,
  and every interval below is narrow.

| Row | A ms | B ms | B / A [interval] | Digests |
|---|---|---|---|---|
| jump-forward off | 895.9 | 934.7 | 1.044 [1.042, 1.045] | A and B `7e46330608d11704` |
| jump-forward on | 452.2 | 565.4 | 1.250 [1.247, 1.251] | A and B `8ad5ea7961656cef` |
| prefill, 7 tokens | 144.3 | 186.7 | 1.296 [1.290, 1.297] | (no tokens produced) |
| prefill, 16 tokens | 233.9 | 234.8 | 1.004 [1.000, 1.006] | (no tokens produced) |
| jump-forward on, exact (B only) | — | 1386.6 | — | B `7e46330608d11704` |

- **Digests:** off and on match across the arms, and B's exact-on digest equals its off digest, as exact mode promises.
- **The slowdown is back on B under MSVC at one thread.** A plain 7-token batched prefill costs 29.6 % more, while a
  16-token one is unchanged; the jump-forward-on extraction, whose spans are all 2 to 6 tokens, costs 25.0 % more; the
  off extraction, whose only batched call is the 41-token prompt, costs 4.4 % more. The pattern follows short batched
  passes, the same shape as the d7ec81b slowdown (`../2026-10-20-q6c/`), now on `sslm_prefill` too.
- **Exact mode** at one thread took 1386.6 ms against B's 934.7 ms for off, 1.48x slower. Its passes (pending token plus
  draft, then a finish at each row) are short batched passes, the shape that slowed above, so this reading includes that
  slowdown and does not price exact mode on its own.
