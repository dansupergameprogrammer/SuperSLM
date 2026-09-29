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

## ID-PENDING list

Nothing minted here.
