# One-thread MSVC A/B: the J1-X fix (f4f69efb) against the Q6 C fix, 2026-10-21

The same run as `../2026-10-21-j1x-ab/` (`run-abj-fix.ps1`), with B moved to the new tip:

- **A** = `2af41659`, the bench from the 2026-10-20 ab-fix run (sha256 `b1260b8f…`).
- **B** = `f4f69efb` (the batched kernel a template, so non-exact calls take the original per-token loop; the anti-LM
  install fixed), built fresh (MSVC 19.44, Release, Ninja); its bench is the tip's own, which carries the `on:exact` arm
  (sha256 `e2592961…`).
- Same box and model (aex-pr, `9ccf7e37…`, passed explicitly), one thread, 15 interleaved rounds after a warm-up, arm
  and row order rotated per round. The box was clear: no editor, build, build lock or U3 driver, load 0 % at the start.

| Row | A ms | B ms | B / A [interval] | Digests |
|---|---|---|---|---|
| jump-forward off | 894.9 | 928.4 | 1.036 [1.032, 1.039] | A and B `7e46330608d11704` |
| jump-forward on | 452.1 | 566.7 | 1.253 [1.250, 1.255] | A and B `8ad5ea7961656cef` |
| prefill, 7 tokens | 144.1 | 186.2 | 1.292 [1.290, 1.295] | (no tokens produced) |
| prefill, 16 tokens | 232.9 | 236.4 | 1.013 [1.011, 1.019] | (no tokens produced) |
| jump-forward on, exact (B only) | — | 1387.9 | — | B `7e46330608d11704` |

- **Digests:** off and on match across the arms, and B's exact-on digest equals its off digest.
- **The slowdown is unchanged at f4f69efb.** Against f1ca2e33 (`../2026-10-21-j1x-ab/`): on 1.253 (was 1.250), a 7-token
  prefill 1.292 (was 1.296), off 1.036 (was 1.044), a 16-token prefill 1.013 (was 1.004). The per-row counter test in the
  batched kernel, which this tip moves out of the non-exact path, is not what costs the short passes under MSVC at one
  thread, or the change did not remove it.
- **Where to look next.** Between A and f1ca2e33 the commits that touch `src` or `include` include the TF slice
  (`bfdec2a6`, the decode step's finish as a transaction), the precached-prefixes merge (`a448d93e`) and J1-X itself
  (`a4692482`). A fresh-build bisect at those three points, one thread and the 7-token prefill row alone (its ratio is
  the largest and it produces no tokens), would place the cost; each arm is about 15 seconds of measurement per round.
