# Attention and per-row sites, slice S5: build progress

The plan of record is the attention and per-row sites plan, rev 3.1, approved by the owner. This series builds its
slice S5 only (§4.5, §5.5): `QkQ31ScoreRow`, every key's Q31 score for one query head (the Qwen3 QK-norm path), in
three 16-bit pieces on the AVX2 and AVX-512BW tiers, bit-identical to the per-key `QkQ31Score` on every tier, and
called by both layer loops in place of their per-key loops.

**Base.** The S1, S2, S3 and S4 series (three patches each) on tiled-matmul slice 1's branch at `fb56397`. Delivered as
a patch series (`git am` after S4's), not pushed.

**Host.** The same 4-vCPU cloud Xeon as S1–S4 (AVX2, AVX-512F/BW/DQ, AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3), shared
with other agents, so every timing is best-of-N and noisy.

States: **done**, **CI-only**, **box-only**, **not done** (with why).

## Resume here

1. Clone SuperSLM, `git checkout -b attn-s5 fb56397`, `git am` the S1, S2, S3 and S4 series, then this one.
2. Cell 11.1(d) needs the 0.5B-width 1-layer synthetic artifact (sha256 `f0fd4886…6ed3`), as in S1–S4:
   `SUPERSLM_ATTN_ROWSITES_ARTIFACT=<path>`, suites run from the repository root, one `TMPDIR` per binary when they run
   at once. Cell 11.1(c) needs nothing: its fixture is in-tree (`tests/support/qk_attention_fixture.h`).
3. The fp-free scan on a GCC build needs main's 90e48de (a `TiledGemmAvx512` fix) until this series is rebased onto it.

## S5 items

| Item | State | Evidence |
|---|---|---|
| Red-first: cells (4.S5 grid, 7.S5b margin corners, 7.S5c ties, 7.S5d inside corners, width 0, 2.S5 hostile rows, 6.1, 6.3, 11.1(c) on the widened QK-norm fixture, 11.1(d)'s q31_row rows), the test-side guard copy, per-tier q31_row counters (declared, not incremented), `QkQ31ScoreRow` declared with a stub that runs the per-key loop, the S5 rows appended to `c32_attention`, the S5 golden hash and the fixture hash | done (commit 1) | `docs/attention-rowsites/s5/red-suites.txt`: auto and forced AVX2/AVX-512 fail 358 assertions each, every one a q31_row path assertion; forced SSE2 0. Every value assertion passes on the base. Digests: every section but `c32_attention` byte-identical to S4's on all five legs; `c32_attention` `ddbdb76e…` on all five |
| The widened QK-norm fixture (11.1(c)): its premise on the base | done (commit 1) | `fixture-premise.txt`: every step Ok in all three runs, which hash alike; 96 softmax rows, all inside §5.4's guard; 4 prob-V rows fail the int16 condition (position 0's width-1 rows) |
| Golden pin (6.3) from the v1.9.0 tag | done (commit 1) | `golden.txt`: S5 Q31-row `daea9a39…` over 33,618 values; fixture `336b8d41…` over 14,384; S1–S4 unchanged |
| `QkQ31ScoreRow` on the AVX2 and AVX-512BW tiers, both layer loops calling it | done (commit 2) | GCC 13.3 and Clang 18.1, auto (AVX-512 here) and forced SSE2/AVX2/AVX-512: 0 failures on all eight binaries; S5 and fixture hashes equal the pins; digests equal the red run's on all five legs, both compilers (GLOBAL `f740f833…`) |
| 11.3 linkage checker: the S5 bodies and the forward_sites.cpp objects join; vitality plant | done (commit 2) | `linkage-plant.txt`: a planted external `Q31RoundAvx2Planted` turns it red on all three forward_sites objects; restored, OK. The CI job passes the three forward_sites.cpp objects |
| Isolation checker prose names the S5 bodies | done (commit 2) | `check_matmul_avx_isolation.py` exits 0; its population test passes (31) |

## Deviations from the plan as written

1. **The golden pin carries one hash per slice**, as in S2–S4, and S5 has two: the Q31 set's and the fixture's (§3.3
   names both). The Q31 set is driven through the v1.9.0 per-key `QkQ31Score` in the generator (v1.9.0 has no row
   entry) and through `QkQ31ScoreRow` in the suite and the digest.
2. **The fixture's constants are tuned, not canonical.** `QkNormWiringFixture` uses the canonical site constant
   everywhere; at hidden 256 with random weights that drives three sites out of domain (the post-norm funnel preflight,
   the SiLU gate scale, the kernel's softmax constants) and clamps every V code. `fixture-premise.txt` records each
   sweep. The RoPE table is built from Pythagorean triples, so no platform's libm enters the fixture.
3. **4.S5's widths add 15, 16 and 17** to the plan's {1, 7, 8, 9, 1,024}: the AVX-512 body packs keys in blocks of 16,
   and those three are its full block and both partials.

4. **The linkage checker reports a stale name on a Clang build, before and after this slice.** On Clang 18 objects
   `TiledWidenActivations` resolves to no symbol and no inlined signature (the S4 tree gives the same FAIL); the CI
   job builds with GCC 13, where the check is OK. Every S5 name resolves on both compilers. Not changed here.

## What the plan got wrong

(filled in at the evidence commit)

## ID-PENDING list

Nothing minted here.
