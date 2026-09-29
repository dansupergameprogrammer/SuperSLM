# Attention and per-row sites, slice S4: build progress

The plan of record is the attention and per-row sites plan, rev 3.1, approved by the owner. This series builds its
slice S4 only (§4.4, §5.4): a guarded fast path inside `SoftmaxRowQ15` on the AVX2 and AVX-512BW tiers, with the shipped
body as the fallback, bit-identical (bool and every probability) to the v1.9.0 body on every tier.

**Base.** The S1, S2 and S3 series (three patches each) on tiled-matmul slice 1's branch at `fb56397`. Delivered as a
patch series (`git am` after S3's), not pushed.

**Host.** The same 4-vCPU cloud Xeon as S1–S3 (AVX2, AVX-512F/BW/DQ, AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3), shared with
other agents, so every timing is best-of-N and noisy.

States: **done**, **CI-only**, **box-only**, **not done** (with why).

## Resume here

1. Clone SuperSLM, `git checkout -b attn-s4 fb56397`, `git am` the S1, S2 and S3 series, then this one.
2. Cell 11.1(d) needs the 0.5B-width 1-layer synthetic artifact (sha256 `f0fd4886…6ed3`), as in S1–S3:
   `SUPERSLM_ATTN_ROWSITES_ARTIFACT=<path>`, suites run from the repository root, one `TMPDIR` per binary when they run
   at once.
3. The fp-free scan on a GCC build needs main's 90e48de (a `TiledGemmAvx512` fix) until this series is rebased onto it.

## S4 items

| Item | State | Evidence |
|---|---|---|
| Red-first: cells (2.S4, 4.S4 grid, inside corners, correction rows, aliased rows, width 0, 6.1, 6.3, 11.1(d) softmax rows), the test-side guard copy and estimate replica, per-tier softmax counters (declared, not incremented), the S4 rows appended to `c32_attention`, the S4 golden hash | done (commit 1) | `docs/attention-rowsites/s4/red-suites.txt`: auto and forced AVX2/AVX-512 fail 3,125 assertions each, every one a path assertion; forced SSE2 0. Every value assertion passes on the base. Digests: every section but `c32_attention` byte-identical to S3's on all five legs; `c32_attention` `6bb5971d…` on all five |
| Golden pin (6.3) from the v1.9.0 tag | done | `golden.txt`: S4 `2e47ea3c…` over 268,078 values; S1's, S2's and S3's unchanged |
| 11.1(d) data term re-derived on the base | done | `softmax-data-terms.txt`: 0 rows outside the guard in both windows (1,792 prefill, 448 decode rows), as the plan measured |
| Implementation: the row guard, the AVX2 and AVX-512BW bodies (integer estimates, exact corrections), the dispatcher and its fallback counter | done (commit 2) | GCC 13.3 and Clang 18.1: auto (AVX-512 here) and forced SSE2/AVX2/AVX-512 suites 0 failures (116,129 / 116,071 / 116,087 / 116,087 checks); S4 golden `2e47ea3c…` on every binary; digests equal the red run's on all five legs of both compilers |
| 11.3 linkage checker: the S4 bodies join the population; vitality plant | done | `linkage-plant.txt`: a planted external `SoftmaxExpAvx2Planted` turns it red; restored, OK |
| 6.2 digests: every section equal to the red run's on all five legs, both compilers | done | `suites.txt`, `suites-clang.txt`, `sslm_axis_digest*.txt`: GLOBAL `bc7b1cbe…` on all ten; `c32_attention` `6bb5971d…`; every other section equal to S3's record |
| 11.1(d) exact counter counts: `softmax` fast L·H·N (1,792 prefill, 448 decode on the artifact), fallback 0, on the running tier's counters only | done | asserted in the suite on every binary; the always-fallback mutant reads fast +0 against both windows |
| fp-free scan (`scan_build_output.py --target superslm --isa x86-64`), allow-lists unchanged | done (Clang); GCC with 90e48de applied | `fp-scan.txt`. Clang: PASS (490 ACCEPT). GCC: only the base's `TiledGemmAvx512` rejects, as in S3; with 90e48de applied temporarily, PASS (561 ACCEPT). `intmath.cpp.o` is clean on both |
| §9 mutants, each tier's body separately, on auto, forced AVX2 and forced AVX-512 | done, all killed | `mutants.txt`, `mutation-scripts/` (30): the 22 §9 S4 mutants (14 guard, 6 correction, 2 dispatcher), the 3 all-slice rows and 5 extras die on every binary that runs the code they mutate. Three guard-dropped mutants die by SIGFPE in the 4.S4 grid, before the 2.S4 rows run |
| Sanitizers: ASan+UBSan (auto, forced AVX2, forced AVX-512), TSan (auto) | done, 0 reports | `sanitizers.txt` |
| Save-blob protocol (6.4) against the S1 head's build | done, 46 of 46 EQUAL | `blob-protocol.txt`: auto and forced-AVX2 candidates; every hash equals S3's record |
| 11.6 branch coverage (clang-18 replica) | done locally; floors not re-pinned | `coverage.txt`: full union OK (intmath.cpp 91.32%, all 54 new branches covered). Projected for a runner without AVX-512: intmath.cpp 83.47%, **below its 87.93% floor** (S3 already 86.70%); allowlist notes added |
| CHANGELOG and `docs/platform-support.md` entries | done (commit 3) | the Unreleased entry and a section beside S3's |
| 10.1 bench against §0's 0.15 / 0.54 / 1.15 prefill and 0.53 decode ms/token | done | `bench.md`: AVX2 **0.10 / 0.46 / 0.91 prefill, 0.53 decode** (AVX-512 0.10 / 0.48 / 0.91, 0.54); 3.4–3.6× on rows of 128–1,024 keys; a one-key row 0.03 µs slower |

## Deviations from the plan as written

1. **The estimates are integer, not IEEE double.** §4.4 and §5.4 estimate z = floor(a / q_ln2) and p = floor(e·2¹⁵ / denom)
   in IEEE double. The library is floating-point-free: `tests/ci/scan_build_output.py` gates `libsuperslm.a` on "no
   floating-point arithmetic", and its allow-list is not to be widened. §5.4's own floating-point bullet names what carries
   over: the estimates only need to be within one of the floor, because exactness comes from the exact integer corrections.
   The integer estimates keep the plan's structure (estimate, then one exact correction each way):
   - z: `(a·inv_z) >> kz` with `inv_z = floor(2^kz / q_ln2)`, `kz = 30 + bit_width(q_ln2)`. A floored reciprocal never
     overestimates, so, as §5.4 step 3 argues for the double, **only the upward z correction can fire**; the downward one
     stays as defensive code with no mutant owed.
   - p: `(e·R) >> 47` with `R = round(2^62 / denom)`. |error| ≤ e / 2^48 ≤ 1/2, so **both p corrections are live**, as
     §5.4 step 4 has them.
   - The plan's p-downward steering constants (q_b = 11,863,283, M within 2⁻²⁴ of 2⁴⁷) were tuned to the double. With the
     integer estimate, M·R / 2⁴⁷ sits just below the integer R when M ≈ 2⁴⁷, so the estimate for e = M never overshoots
     there. The steered generator keeps the plan's shape (denominator first, then a row summing to it; the plan's two
     constant shapes) with M near ¾ and 7⁄10 of 2⁴⁷ (`tests/support/attention_cases.h`, `SmSteeredPDownRows`).
2. **The golden pin carries one hash per slice**, as in S2 and S3.
3. **The off-ratio witness fails three conjuncts, not two.** §8 2.S4 says `kSoftmaxRowOffRatioWitness` fails q_c ≥ 0 and
   M ≥ 1. Its q_ln2 is 3,000,000,001 against q_b = 10, so it fails q_ln2 ≤ 2·q_b + 1 too. It stays in the set as a
   realistic hostile row; the cell asserts "more than one".

4. **The fp scan on GCC was passed with main's 90e48de applied temporarily.** It is not part of this series. S4 adds nothing
   the GCC scan rejects.
5. **Floors are not re-pinned (11.6).** Step 3 needs the hosted leg's measurement.
6. **The 11.3 vitality plant is an added external function, not a moved body.** S3's plant (a body given external linkage)
   does not work for S4: every S4 body takes `SoftmaxFastRow`, an anonymous-namespace type, so the moved function stays
   local and the checker rightly stays green. A planted external target-attributed function turns it red
   (`linkage-plant.txt`).

## What the plan got wrong

- **Its estimates are floating point, which the library forbids** (deviation 1). §4.4 and §5.4 estimate both divides in IEEE
  double; `scan_build_output.py` gates `libsuperslm.a` on no floating-point arithmetic, with an allow-list not to be widened. The
  argument carries over to integer estimates unchanged, because it only ever needed an estimate within one.
- **The p-downward steering constants are tied to the double** (deviation 1). At q_b = 11,863,283 the integer estimate never
  overshoots, so the plan's generator finds no row. With M near ¾ and 7⁄10 of 2⁴⁷ it finds them at once. The integer estimate
  also fires the p-downward correction on ordinary grid rows (44 of the grid's rows die under its skip mutant), where the plan
  measured about 2⁻³⁸ per row for the double.
- **The off-ratio witness fails three conjuncts, not two** (deviation 3).
- **§0's prefill figures assume a larger per-element saving than its decode figure.** Prefill 0.15 / 0.54 / 1.15 implies 6.9 /
  6.3 / 6.7 ns saved per element, decode 0.53 implies 5.2. Measured, the AVX2 body saves about 5.2 ns per element on long rows,
  so decode lands on the estimate and prefill at 65% / 86% / 79% of it. A fixed per-row cost (the guard pass and two
  divides) makes short rows cost more, which takes a further share at T = 128 (`bench.md`).
- **11.6 cannot pass on a runner without AVX-512**, now by 4.5 points on `intmath.cpp` (S3 1.2). The owner's floor call (G27)
  covers it.
- **§9's killing cell for three guard-dropped mutants is reached late.** q_ln2 ≥ 1, q_c ≥ 0 and M ≥ 1 dropped each trap
  (integer divide by zero) in the 4.S4 grid, whose realistic triples include constants outside the guard, before the 2.S4 rows
  §9 names run. The mutants are killed; the named signal (bool and counter) is not the one observed.

## ID-PENDING list

Nothing minted here.
