# Exact skip-ahead box steps (plan rev 1.3.4, B7), 2026-10-21

**Build.** SuperSLM `f03a535836aee178d9897df4f1ea282bb9c4f441` (TF, J1-X with the anti-LM fix, the kRowCounts template and
the MSVC GEMM pin) plus `tools/schema_fast_decode_bench.cpp`'s draft sources (committed with these results), built fresh
(MSVC 19.44, Release, Ninja; `run-exact-build.ps1`). Timed bench sha256 `fcba88db…`. Model aex-pr `9ccf7e37…`;
population v1 `d846b9e3…`. AMD Ryzen 9 3950X; the box was clear (no editor, build, build lock or U3 driver).

**Step 1, the pin.** A fresh build of f03a5358 and the timed build both place the GEMM kernels as expected (address mod
64): `GemmInt8AccumulateCols` 0, `DotRowAvx2` 16, `TiledGemmAvx2` 32, `ActiveGemmTier` 32.

## The learned draft (D2) and the bench's draft sources

- **The table** (`d2-table.txt`, sha256 `4cc216d2fe691d060887d90b12b9267afe17d1c7b97a47e575137abf7fb2b853`, frozen before
  any timing): built offline by the bench's `drafttable` mode from B4a's 20 on-file off streams
  (`../2026-10-20/B4a.json`, sha256 `8b93415b…`), each walked through `potion_shop_order` from state 0 with the model's
  token counted at every state: 513 tokens, 64 of the schema's 148 states seen. The table also carries every state's
  forced flag from the library's own forced-next table (142 of 148 forced), read through the §11.0 seam
  `ForcedTableEntry`; `drafttable` is therefore compiled only into a seam build (`superslm_scalar_forced_seams`), which
  times nothing. The timed bench is a production build and reads the table file.
- **The draft** (`on:exact:learned`): from the walk state after the pending token (the bench tracks the walk itself with
  the SCM1 transitions; the ABI has no walk-state query), the table's most frequent token at each state while the
  running product of those tokens' shares stays at or above 0.5; it stops at a state the table never saw and at a state
  the library marks unforced (a choice state), and is clipped to the span budget less the pending token. Nothing stored
  for the state: an empty draft, and the verb is not called (the decode step embeds the pending token).
- **The NULL draft** (`on:exact`): the canonical chain, the library's own default.
- **The always-wrong draft** (`on:exact:wrong`): at each forced state with two or more legal tokens, a legal token other
  than off's next token (off's stream comes from one unmeasured off run in the same process); it stops at a state with
  one legal token or an unforced state. After its first token the draft is off off's path, so later tokens are any legal
  token at the draft's own state.

## B7b, exactness (step 2)

All 40 population prompts and B2's prompt, one process per prompt, arms off, exact with the learned draft, exact with
the NULL draft and exact with the always-wrong draft; each exact arm's token stream and text compared with off's, byte
for byte (`b7b-t1/`, `b7b-t4/`).

| | prompts | exact arms compared | not identical to off |
|---|---|---|---|
| one thread | 41 | 123 | **0** |
| 4 tasks | 41 | 123 | **0** |

## B7c and B7d, B2's prompt (steps 3 and 4)

One process per row per round, 15 rounds after a warm-up, the row order rotated. Ratios are off / row per round (above
1: the row is faster).

| Row | one thread ms | off / row [interval] | 4 tasks ms | off / row [interval] | Digest |
|---|---|---|---|---|---|
| off | 896.6 | 1 | 548.5 | 1 | `7e46330608d11704` |
| on (canonical, not exact) | 453.0 | 1.980 [1.978, 1.981] | 411.2 | 1.341 [1.315, 1.365] | `8ad5ea7961656cef` |
| exact, NULL draft | 1112.0 | 0.806 [0.806, 0.809] | 571.6 | 0.963 [0.954, 0.978] | `7e46330608d11704` |
| **exact, learned draft** | **604.3** | **1.483 [1.481, 1.487]** | 557.2 | **0.996 [0.964, 1.003]** | `7e46330608d11704` |

Predictions (made before the pin): B7c ≈ 1.476x (about 626 against 925 ms); B7d ≈ 1.00x.

## B7e, B4b's 20 prompts, one thread (step 5)

Off against exact with the learned draft, 5 interleaved pairs per prompt after a warm-up pair, one process per prompt
(`b7e/`). Every prompt's exact digest equals its off digest.

- **Total over the 20 prompts:** 16,676 ms off against 12,878 ms exact: **1.295x** (prediction ≈ 1.30x).
- **Per prompt:** median 1.418 [1.275, 1.474] (interquartile range; n = 20). 17 prompts gain 1.25-1.49x; B4b-07 1.045x and
  B4b-14 1.005x break even; **B4b-09 is slower, 0.824x** (1,231.6 against 1,015.2 ms), the one prompt where drafting
  from the B4a table loses at one thread.

## Notes

- The exact NULL-draft arm is slower than off at one thread (0.806x): the canonical chain is rarely the model's own
  tokenization, so most drafted rows are rolled back. The learned draft is the model's own tokens.
- At 4 tasks exact mode with either draft is about even with off: the one-token split already threads each decode, and
  exact mode's checks run as one-token passes under the split (docs/api.md, exact mode).
- Scripts: `run-exact-build.ps1` (builds and the table), `run-exact-steps.ps1` (steps 2-5, resumable by step).
