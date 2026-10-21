# One-thread MSVC bisect of the 7-token prefill, 2026-10-21

The row that showed the J1-X-era slowdown most clearly (`../2026-10-21-j1x-ab/`, `../2026-10-21-j1x-ab-fix/`): a plain
`sslm_prefill` of 7 tokens at the example prompt's depth (`pass --example --sizes 7 --no-one-row`), one thread, the
aex-pr model (`9ccf7e37…`, passed explicitly; every process logged it). `run-abp.ps1`: one process per arm per round,
the arm order rotated per round, 15 rounds after a discarded warm-up; ratios are arm / 2af41659 per round, median and
the n = 15 sign interval. MSVC 19.44, Release, Ninja; AMD Ryzen 9 3950X; the box was clear (no editor, build, build lock
or U3 driver). Every arm is a fresh build: 2af41659 and f4f69efb reuse their fresh builds from the two A/B runs above;
the others were built for this run, each in its own worktree. v1.13.1 and the precache tip have no bench of their own:
their libraries were built fresh with CMake, and the bench source from 2af41659 was compiled against each decode-only
(`SFD_BENCH_DECODE_ONLY`), as the box driver's B0 builds main's bench. The prefill row times the library either way.

| Commit | What it is | 7-token prefill, ms [interval] | arm / 2af41659 [interval] |
|---|---|---|---|
| `d697927` | v1.13.1 | 178.5 [178.1, 178.9] | 1.233 [1.230, 1.235] |
| `83a82c20` | the precache tip (wt2dpx), a448d93e's second parent | 184.9 [184.5, 185.4] | 1.280 [1.269, 1.283] |
| `2af41659` | the Q6 C fix (baseline) | 145.1 [144.1, 145.9] (phase 1); 144.6 [144.4, 145.0] (phase 2) | 1 |
| `fc63ca84` | a448d93e's first parent (library as 2af41659) | 144.7 [144.3, 144.9] | 1.001 [0.997, 1.003] |
| `bfdec2a6` | TF (on the fast-decode line, after fc63ca84) | 186.1 [185.7, 187.5] | 1.284 [1.282, 1.289] |
| `a448d93e` | the precache merge (fc63ca84 + 83a82c20; no TF) | 185.4 [184.6, 186.9] (phase 1); 184.9 [184.6, 185.8] (phase 2) | 1.278 [1.276, 1.282]; 1.280 [1.272, 1.285] |
| `a4692482` | J1-X (contains TF, not the precache merge) | 144.8 [144.7, 145.9] | 0.999 [0.997, 1.004] |
| `f4f69efb` | the tip (contains both lines) | 185.5 [184.0, 187.0] | 1.276 [1.267, 1.289] |

Phase 1 is the first five rows' run (2af41659, bfdec2a6, a448d93e, a4692482, f4f69efb); phase 2 the second (2af41659,
fc63ca84, a448d93e, 83a82c20, d697927). a448d93e and 2af41659 appear in both and agree within 0.5 ms.

**What it shows.** The 7-token batched prefill takes about 145 ms or about 180-186 ms, with nothing in between:

- **The fast state is not the norm the later commits lost.** v1.13.1 itself takes 178.5 ms, and the precache line
  (83a82c20) 184.9 ms. The fast-decode line reached about 145 ms by 2af41659 (and fc63ca84, its library unchanged).
- **It flips on commits that do not change this path in any obvious way.** TF (bfdec2a6) takes it back to 186 ms; J1-X
  (a4692482), which contains TF, returns it to 145 ms; the precache merge (a448d93e), which contains neither TF nor
  J1-X, takes it to 185 ms; the tip, which merges both lines, is at 185 ms.
- So a step at a448d93e is real, but the merge brings in the precache line's state, which is v1.13.1's within about 3.5
  %; and TF and J1-X move the same row by the same amount in opposite directions. That pattern fits MSVC code generation
  in the short batched pass (inlining, layout or vectorisation decisions that change with unrelated edits in the same
  translation unit) better than a cost any one feature adds. The 16-token pass, by contrast, moved by at most 1.3 % at
  f1ca2e33 and f4f69efb.

**Not yet examined:** the generated code of the batched pass (`RunLayerLoopChunkBatched` and what it inlines) in a
fast build (2af41659 or a4692482) against a slow one (bfdec2a6 or f4f69efb), which is where a code-generation cause would
show, and whether pinning the relevant inlining (or moving the per-row kernel into its own translation unit) makes the
fast state stable across these commits.
