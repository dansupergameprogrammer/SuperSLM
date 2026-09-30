# Attention and per-row sites, slice S2: build progress

The plan of record is the attention and per-row sites plan, rev 3.1, approved by the owner. This series builds
its slice S2 only (§4.2, §5.2): `GemmProbQ15Accumulate` on int16 multiply-add, per head, on the AVX2 and
AVX-512BW tiers, bit-identical to the v1.9.0 loop on every tier.

**Base.** The S1 series (three patches) on tiled-matmul slice 1's branch at `fb56397`. Delivered as a patch series
(`git am` after S1's), not pushed: that branch carries slice 1's draft PR.

**Host.** The same 4-vCPU cloud Xeon as S1 (AVX2, AVX-512F/BW/DQ, AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3), shared
with other agents, so every timing is best-of-N and noisy.

States: **done**, **CI-only**, **box-only**, **not done** (with why).

## Resume here

1. Clone SuperSLM, `git checkout -b attn-s2 origin/claude/project-thread-c8iecr`, `git am` the S1 series, then
   this one.
2. Cell 11.1(d) needs the 0.5B-width 1-layer synthetic artifact (sha256 `f0fd4886…6ed3`), as in S1:
   `SUPERSLM_ATTN_ROWSITES_ARTIFACT=<path>`, suites run from the repository root.

## S2 items

| Item | State | Evidence |
|---|---|---|
| Red-first: cells (11.2, 4.S2, 2.S2, 7.S2, 6.1, 6.3, 11.1(d) prob·V rows), per-tier prob·V counters (declared, not incremented), the 11.2 selector as a stub that selects v1.9.0 code everywhere, digest section `c32_attention`, S2 golden hash | done (commit 1) | `docs/attention-rowsites/s2/red-suites.txt`: auto and forced AVX2/AVX-512 269 failures each, forced SSE2 3 (the selector); every value assertion passes on the base. Digests: sections 1–10 byte-identical to S1's on all five legs; `c32_attention` `b0d1a6cd…` on all five |
| 11.1(d) data terms re-derived on the base | done | `pv-data-terms.txt`: prefill 15 rows fail the int16 condition (14 width-1 rows, plus position 2 head 13's {0, 32,768, 0}), decode 0, as the plan measured |
| Golden pin (6.3) from the v1.9.0 tag | done | `golden.txt`: S2 `b0d1a6cd…` over 30,100 values; S1's hash unchanged |
| `GemmProbQ15Accumulate` tiered: the int16 condition and `head_dim % 16` guard, AVX2 and AVX-512BW `vpmaddwd` bodies over register-formed probability pairs, an accumulate-into core shared by the zeroing entry, per-tier counters in each body (fast) and in the dispatcher (fallback) | done (commit 2) | `src/matmul.cpp` |
| The 11.2 selector and its wiring; `SUPERSLM_SITES_AVX512_MSVC`, default 0; both forced AVX-512 Windows legs build with it on | done (commit 2) | `src/matmul.cpp`, `.github/workflows/tests.yml` |
| Linkage (11.3) and isolation checkers name the new attributed functions | done (commit 2) | `check_tiled_matmul_linkage.py` OK on the auto and both forced objects |
| GCC 13.3 suites: auto (AVX-512 here), forced SSE2, AVX2, AVX-512, at the final code plus commit 3's blocking rows | done, 0 failures | `suites.txt`: 27,927 / 27,869 / 27,885 / 27,885 checks (attn-rowsites 2,396 each) |
| Digests (6.2): auto and scalar/SSE2/AVX2/AVX-512 forced | done | all five equal and equal to the red run's: GLOBAL `ec7016a0…`, `c32_attention` `b0d1a6cd…` = the v1.9.0 pin |
| 4.S2 kernel-blocking rows: head_dim {32, 48, 80, 96, 144, 160, 208, 224, 240} x width {1, 2, 3, 64, 65}, every AVX2 and AVX-512 block count and the AVX-512 16-dimension tail behind a full block (not in the golden set) | done (commit 3) | `tests/test_attn_rowsites.cpp`, 90 checks |
| Clang 18.1.3 suites (four binaries) and digests (five legs) | done, 0 failures | `suites-clang.txt`: same counts; digests equal GCC's (GLOBAL `ec7016a0…`) |
| Mutants (§9): every §9 S2 mutant plus four extras; per-tier arithmetic mutants per body (AVX2 body, AVX-512 32-dimension unit, AVX-512 16-dimension tail) | done: 20 of 20 killed on every binary where they execute | `mutants.txt`, `mutation-scripts/`. Run at the final implementation commit before commit 3's 90 rows existed, which only add kills |
| 11.3 linkage vitality (the plan's plant) | done | `mutants.txt` tail: the checker goes red on an external-linkage AVX2 prob·V function; OK at the final code on all three objects |
| ASan+UBSan (auto, forced AVX2, forced AVX-512), TSan (auto) | done, 0 failures, 0 reports | `sanitizers.txt` |
| Save blobs (6.4), as S1 ran them | done, 45 of 45 EQUAL | `blob-protocol.txt`: 30 rows of base vs auto candidate over wide_l8 / p05_l1 / p05_l2 / in-tree at 8, 128 and 512 tokens and three chunk budgets, then 15 rows with the in-tree fixture at ids:8 and the synthetics against the forced-AVX2 candidate; the blob hashes equal S1's |
| Golden generator consistency | done | the generator built against this series prints the v1.9.0 hashes (`golden.txt`) |
| 11.1(d) with the prob·V rows | done | calls = L·H·N per window, prefill `pv_fallback` 15 and decode 0 on the AVX2 and AVX-512 binaries (no counter moves on SSE2), in every suite run above |
| Bench (10.1): the saving against §0 | done | `bench.md`: prefill 0.825 / 3.38 / 6.74 ms/token at T = 128 / 512 / 1,024 (plan 0.95 / 4.38 / 8.47), decode 3.95 at context 300 and 7.97 at 600 (plan 3.83 / 7.72); forced AVX2 within 2% of those savings. Forward-level check agrees at T = 512 and in decode |
| Branch-coverage floors (11.6) | CI-only (local replica recorded) | `coverage.txt`: a local five-binary clang-18 replica passes every floor on this AVX-512 host (matmul.cpp 89.19 → 91.67). Without AVX-512 profiles it would drop to about 70.98 against the 72.22 floor. Allowlist lines added for the AVX-512 bodies and the two implicit switch defaults. The floors are not re-pinned, because re-pinning copies the hosted leg's value and the series is not pushed |
| MSVC / clang-cl: switch compiled in (default 0), selector truth table, both forced AVX-512 Windows legs build with `/DSUPERSLM_SITES_AVX512_MSVC=1` | CI-only | nothing MSVC can be compiled here. The 11.2 truth table covers the (MSVC, switch 0/1) rows on every binary. The `s2_msvc_switch_ignored` mutant dies |
| MSVC AVX-512 execution (the switch's flip to 1 by default) | box-only | §3.2: needs an MSVC AVX-512 build to execute the path |
| CHANGELOG `[Unreleased]` and `docs/platform-support.md` section | done (commit 3) | |

## Deviations from the plan as written

1. **The AVX-512 body uses 32-dimension in-lane units, not a 16-dimension one.** A straight port of the AVX2 unit to 512-bit
   registers measured slower than AVX2 (0.42 against 0.30 µs per call at width 128). `_mm512_cvtepi8_epi16` of a 256-bit
   unpack keeps pairs in 128-bit lanes, so each unit covers 32 dimensions whose two halves are stored 16 apart. A
   16-dimension tail handles head_dim % 32 == 16. It measures 0.266 µs. The arithmetic is unchanged: the same pairs, the
   same `vpmaddwd`, and the same lane bound. The half-order store and the tail have their own mutants, and both die.
2. **The golden pin carries one hash per slice.** The S1 hash is unchanged, and S2's is added beside it, generated from the
   v1.9.0 tag's library.
3. **The blocking rows are suite rows, not golden-set rows.** Adding them to `attention_cases.h` would change S2's pin.
   They test the kernel's own blocking against the test-side v1.9.0 loop, under the path rule.
4. **The build-configuration record does not gain `SUPERSLM_SITES_AVX512_MSVC`.** The record states the tier and the force
   macros. The switch acts only on MSVC builds, where the 11.2 truth table and the Windows legs cover it.
5. **Clang's first build segfaulted once.** It was a compiler crash under parallel load. A rebuild at -j3 succeeded, and every
   Clang run above is on that rebuild.
6. **Floors are not re-pinned (11.6).** See the table: the plan's step 3 needs the hosted leg's measurement.

## What the plan got wrong

- **The prefill saving estimate is 15–25% high.** Measured 0.825 / 3.38 / 6.74 against 0.95 / 4.38 / 8.47. The estimate at
  T = 512 and 1,024 exceeds this host's entire measured v1.9.0 prob·V cost (3.56 and 7.18 ms/token). The kernel ratio
  (16–21×) beats the spike's 10.7–14.2×, so the error is in the base per-key cost the arithmetic assumed. Decode matches
  (103%).
- **§4.2's AVX-512 port is not "the same body in wider registers".** R2's simplest reading is slower than AVX2. The unit
  layout has to follow the in-lane widening; see deviation 1.
- **11.6 cannot be satisfied inside a slice delivered as patches.** On a runner without AVX-512, S2 lowers matmul.cpp's
  measured branch coverage by about 4.5 points, which is below today's floor. That makes the owner's floor call (G27) likely
  for S2, not merely possible.

