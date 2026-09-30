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
| 4.S5 channel-tail rows (head_dim not a multiple of 4 or 16, inside the guard), from the coverage replica | done (commit 3) | `coverage.txt`: lines 654 and 699 were untaken; 90 rows added outside the golden set; `x_pads_nonzero` dies on them and nowhere else |
| Suites and digests, GCC and Clang, every tier | done (commit 3) | `suites.txt`, `suites-clang.txt`: 0 failures on all eight binaries; S5 `daea9a39…` and fixture `336b8d41…` everywhere; digests GLOBAL `f740f833…` on all ten legs, equal to red |
| fp-free scan, both compilers | done (commit 3) | `fp-scan.txt`: Clang PASS on auto and both forced AVX libraries; GCC fails only on the base's `TiledGemmAvx512` in matmul.cpp.o and PASSes on all three with main's 90e48de applied temporarily; forward_sites.cpp.o clean everywhere; allow-lists unchanged |
| §9 mutants | done (commit 3) | `mutants.txt`: the 11 killable §9 rows (12 scripts, ties per tier) and 11 extras killed on every binary that runs their code; "a₂ by logical shift" equivalent (see below) |
| Sanitizers | done (commit 3) | `sanitizers.txt`: ASan+UBSan on auto, forced AVX2 and forced AVX-512 (both suites) and TSan on auto: 0 reports |
| Save-blob protocol (6.4) and the QK-norm equality | done (commit 3) | `blob-protocol.txt`: 66 of 66 rows EQUAL (auto and forced AVX2), row for row S4's hashes; the QK-norm fixture pinned to v1.9.0 in both loops on every binary; the Qwen3-width forward probe's outputs equal across base and S5, auto and AVX2, at T = 128, 512 and 1,024 |
| Coverage replica (11.6), projected without AVX-512 | done (commit 3), indicative | `coverage.txt`: intmath.cpp and matmul.cpp unchanged from S4 (S5 edits neither; forward_sites.cpp is outside the leg's glob and has no floor); every S5 side covered with five profiles; without AVX-512 only QkQ31RowAvx512 and the dispatcher's AVX-512 arms are lost. Floors not re-pinned |
| Bench against §0 (10.1) | done (commit 3) | `bench.md`: AVX2 11.5 → 0.47 ms/token at T = 128 and 97.6 → 3.23 at T = 1,024, against §0's 10.9 → 0.5 and 87 → 3.9; AVX-512 9.5 → 0.47 and 76.1 → 3.1; a one-layer forward at Qwen3-0.6B width agrees (0.42 / 3.66 ms per layer and token on AVX2) |
| CHANGELOG and platform-support entries | done (commit 3) | `CHANGELOG.md` [Unreleased]; `docs/platform-support.md` "Q31 attention score rows" |
| Box runs B0–B2 (the real Qwen3 artifact, S5-cut timing) | box-only | §7; the only real-artifact run of the Q31 kernel |

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

5. **The saving is measured on synthetic Qwen3-width data, not a real artifact.** A QK-norm artifact is still refused at map
   time on this host (R4), so the kernel bench (the plan's own §0 method, at head_dim 128 and 28 × 16 heads) and a
   one-layer forward probe at Qwen3-0.6B width stand in. The probe is the in-tree fixture re-parameterised; at that width it
   needs hidden gain 4,096 and q/k-norm gains in [2,048, 4,096] to stay Ok through T = 1,024 (`probe_q31_forward.cpp`).
6. **A coverage-found cell is added at the evidence commit, after green.** It is not red-first: the kernel already handled
   the channel pads, and no cell reached them. The pad mutant (`x_pads_nonzero`) shows that the cell is load-bearing. It is
   outside the golden set, so the pin is unchanged.

## What the plan got wrong

1. **§9's "a₂ by logical shift" mutant is equivalent, for any kernel that stores a₂ as an int16 lane.** The low 16 bits of
   w >> 30 are bits 30–45 of w, whichever shift forms it. The two shifts differ only in bits 34–63 of the int64 result, and
   the int16 store (vpmaddwd's operand) drops them. Inside the guard |w| < 2³⁹, so the int16 value is exactly a₂ both ways.
   Negative q cannot kill it, and no input can. Executed: it survives on all three binaries (`mutants.txt`). The row should
   be withdrawn as equivalent, as rev 3 withdrew two S3 rows.
2. **4.S5's head_dim set has no in-guard value that is not a multiple of 4.** {4, 8, 60, 64, 128, 132, 256, 512} leaves the
   limb and key-pack channel pads unexercised, so a pad bug survives the whole plan-specified suite. The coverage replica
   found it; the channel-tail rows close it (`coverage.txt`, `mutants.txt`).
3. **§0's S5 figures hold for AVX2 but not for AVX-512.** §0 is AVX2 (per-key 376–385 ns). Here the shipped per-key AVX-512
   tier costs about 330 ns, while the row kernel is barely faster than on AVX2 (12.8 against 13.5 ns). So AVX-512 saves
   about 20% less: 9.0 / 73.4 against AVX2's 11.0 / 94.3 ms per token at T = 128 / 1,024. On AVX2 the estimate holds and is
   slightly exceeded.
4. **A one-key row is slower on AVX-512, which the plan does not anticipate.** The body packs a whole 16-key block and builds
   128 channels of limbs whatever the width, so width 1 costs 388 ns against the per-key 326 ns. This happens once per head,
   for the first prompt token. On AVX2 (8-key blocks) the one-key row is still faster.
5. **§3.4 / 11.6 name only intmath.cpp and matmul.cpp, and src/forward/ is outside the coverage leg's glob.** S5's kernel
   lives in forward_sites.cpp, so no hosted measurement of it exists or can exist until the leg's export includes
   src/forward/. G27 notes the missing floor, but not that the file is never measured. The replica measures it separately.
6. **§6's whole-prefill figure for S5 (0.6B at T = 1,024, ≈ 140 → ≈ 57 ms per token) cannot be checked on the cloud host.**
   It needs a 28-layer forward on real weights (B1/B2). The one-layer probe's saving, scaled by 28, is 102 ms per token on
   AVX2 and 84.5 on AVX-512, against the ≈ 83 implied by 140 → 57.

## ID-PENDING list

Nothing minted here.
