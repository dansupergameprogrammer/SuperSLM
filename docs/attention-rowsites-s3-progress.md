# Attention and per-row sites, slice S3: build progress

The plan of record is the attention and per-row sites plan, rev 3.1, approved by the owner. This series builds its
slice S3 only (§4.3, §5.3): the requant funnel's element loop as one row leaf, `RequantRowWide`, in 4 (AVX2) or 8
(AVX-512BW) unsigned 64-bit lanes, bit-identical to the per-element `RequantTokenCodeWide` on every tier.

**Base.** The S1 and S2 series (three patches each) on tiled-matmul slice 1's branch at `fb56397`. Delivered as a
patch series (`git am` after S2's), not pushed.

**Host.** The same 4-vCPU cloud Xeon as S1 and S2 (AVX2, AVX-512F/BW/DQ, AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3),
shared with other agents, so every timing is best-of-N and noisy.

States: **done**, **CI-only**, **box-only**, **not done** (with why).

## Resume here

1. Clone SuperSLM, `git checkout -b attn-s3 fb56397`, `git am` the S1 series, the S2 series, then this one.
2. Cell 11.1(d) needs the 0.5B-width 1-layer synthetic artifact (sha256 `f0fd4886…6ed3`), as in S1 and S2:
   `SUPERSLM_ATTN_ROWSITES_ARTIFACT=<path>`, suites run from the repository root. Run the four suite binaries with
   separate `TMPDIR`s if they run at once: an older cell writes a fixed temp-file name.
3. The fp-free scan on a GCC build needs main's 90e48de (a `TiledGemmAvx512` fix) until this series is rebased onto it.
4. Open for the owner: the branch-coverage floors on a runner without AVX-512 (`coverage.txt`, G27).

## S3 items

| Item | State | Evidence |
|---|---|---|
| Red-first: cells (4.S3 both fences, 7.S3 corner premises, 6.1, funnel call site, 6.3, 11.1(d) requant rows), per-tier `requant_row` counters (declared, not incremented), `RequantRowWide` declared with a stub that runs the element loop, the S3 rows appended to `c_rowsites`, the S3 golden hash | done (commit 1) | `docs/attention-rowsites/s3/red-suites.txt`: auto and forced AVX2/AVX-512 fail 10 assertions each, every one a path assertion; forced SSE2 0. Every value assertion passes on the base. Digests: every section but `c_rowsites` byte-identical to S2's on all five legs; `c_rowsites` `d0272180…` on all five |
| Golden pin (6.3) from the v1.9.0 tag | done | `golden.txt`: S3 `3e3abed7…` over 3,567,018 values; S1's and S2's unchanged |
| `RequantRowWide` in `src/intmath.cpp`: AVX2 (4 lanes) and AVX-512BW (8 lanes, F and BW only, no mask register) bodies of the §5.3 identity in unsigned 64-bit lanes with a logical H shift, P formed from r's 32-bit halves, clamp and sign restore as the element code; the tail and the scalar/SSE2 tiers run `RequantTokenCodeWide`; per-tier `requant_row` counter inside each body; dispatch through S2's `DispatchSitesKernel` (so `SUPERSLM_SITES_AVX512_MSVC` covers it) | done (commit 2) | `src/intmath.cpp` |
| The funnel's element loop becomes one `RequantRowWide` call | done (commit 2) | `src/forward/checked_chain_funnel.cpp` |
| GCC 13.3 suites at the implementation commit: auto (AVX-512 here), forced SSE2, AVX2, AVX-512 | done, 0 failures | `suites.txt`: 109,744 / 109,686 / 109,702 / 109,702 checks (attn-rowsites 84,213 each); digests equal the red run's on all five legs (GLOBAL `4892b5be…`) |
| 6.2 digests: `c_rowsites` (now S1 and S3 entries, as §3.3 evidence 2 specifies) equal on every leg; slice 1's sections byte-identical to `docs/tiled-matmul-slice1/s1-b/` | done | every section slice 1 recorded is unchanged on all five legs; `c_rowsites` `d0272180…` on all five and on both compilers |
| Clang 18.1 suites, same four binaries and five digest legs | done, 0 failures | `suites-clang.txt`: the same counts; every digest equals GCC's apart from its compiler line |
| 11.1(d) exact counter counts: `requant_row` 1,536 prefill and 384 decode on the artifact, on the running tier's counter only | done | asserted in the suite on every binary; the never-entered mutant reads +0 against both |
| fp-free scan (`scan_build_output.py --target superslm --isa x86-64`), allow-lists unchanged | done (Clang); GCC blocked by the base | `fp-scan.txt`. Clang: PASS. GCC: one REJECT, `TiledGemmAvx512`, which the S2 head's own build also rejects and main fixes in 90e48de; with that fix applied temporarily, PASS. `intmath.cpp.o` is clean on both compilers |
| §9 mutants, each tier's body separately, on auto, forced AVX2 and forced AVX-512 | done, all killed | `mutants.txt`, `mutation-scripts/` (18): the 8 §9 mutants plus the dispatch mutant and 6 extras die on every binary that runs the tier they mutate; the 3 withdrawn-mutant confirmations survive everywhere, as rev 3 predicts |
| Sanitizers: ASan+UBSan (auto, forced AVX2, forced AVX-512), TSan (auto) | done, 0 reports | `sanitizers.txt` |
| Save-blob protocol (6.4) against the S1 head's build | done, 46 of 46 EQUAL | `blob-protocol.txt`: auto and forced-AVX2 candidates |
| 11.6 branch coverage (clang-18 replica) | done locally; floors not re-pinned | `coverage.txt`: full union OK (intmath.cpp 88.83%). Projected for a runner without AVX-512: intmath.cpp 86.70%, **below its 87.93% floor**; allowlist lines added |
| CHANGELOG and `docs/platform-support.md` entries | done (commit 3) | the Unreleased entry and a section beside S2's |
| 10.1 bench against §0's 1.22 ms/token | done | `bench.md`: **2.53 ms/token saved on AVX2, 2.66 on AVX-512** (4.5× and 5.5× on the funnel call) |
| 11.4 forward-leaf check lists `RequantRowWide`; planted call from `forward_sites.cpp` turns it red | done (commit 2) | `check_no_forward_leaf_calls.py` (+ its 83-cell pytest, green); plant in `leaf-plant.txt` |
| 11.3 linkage checker gains the intmath.cpp objects (population `RequantRowAvx`, the record only in matmul objects); CI job passes all six objects; isolation checker's prose names the two bodies | done (commit 2) | OK on all six objects; plant (RequantRowAvx2 given external linkage) red, `linkage-plant.txt` |
| 11.5 the four recipes of G21 gain `src\matmul.cpp`; GCC link check | done (commit 2) | `recipe-link.txt`: without matmul.cpp all eight link units fail on `detail::ActiveGemmTier` / `DispatchSitesKernel`; with it all link. The five t2296 cells include `<windows.h>`, so their engine source sets are linked as closed sets (`-shared -Wl,--no-undefined`) |

## Deviations from the plan as written

1. **The AVX2 body does |x|, the clamp and the sign in 32-bit lanes.** The plan's reading (clamp and sign on the 64-bit lane,
   as the element code does) compiles on Clang to `vblendvpd` and `vxorpd`, FP-domain encodings the fp-free scan rejects. That
   happened for the `blendv` spelling and for the and/andnot spelling alike. The body now uses `vpabsd` (the low dword of |x|,
   exact since |x| ≤ 2^31), the unsigned 32-bit minimum with the magnitude's high dword folded into bit 7, and `vpsignd` with
   ±1 from x's high dword. The identity and every 64-bit intermediate are unchanged. The AVX-512 body keeps 64-bit lanes
   (`vpabsq`, `vpminuq`, `vpsraq`), which Clang does not lower to FP encodings.
2. **The AVX2 byte-pick constant has distinct halves.** Identical halves let Clang load it with `vbroadcasti128`, which is not on
   the scan's vetted move list. The upper half's never-read bytes are now 1 instead of −1.
3. **11.5's link check covers the five t2296 cells as closed-set shared links, not as executables.** They include
   `<windows.h>`, so they cannot link as programs here. Their engine source sets are linked with `-shared
   -Wl,--no-undefined`, which fails without `src\matmul.cpp` and succeeds with it; `build_cert.bat` and both
   `build_inspect.bat` lines link as executables (`recipe-link.txt`).
4. **The golden pin carries one hash per slice**, as in S2: S1's and S2's are unchanged, and S3's is added from the v1.9.0 tag's
   library.
5. **Floors are not re-pinned (11.6).** Step 3 needs the hosted leg's measurement.
6. **The fp scan on GCC was passed with main's 90e48de applied temporarily.** It is not part of this series. S3 adds nothing
   the GCC scan rejects.

## What the plan got wrong

- **The saving is about twice the estimate.** 2.53 ms/token on AVX2 against 1.22, and 2.66 on AVX-512 against about 1.6. The
  lanes perform as the spike found. The estimate took the spike's base loop cost (19.8 µs at 4,864); here replacing that loop
  saves 24.8 µs on AVX2, more than the spike's whole loop. A per-element out-of-line call explains only about 12% of the
  base's cost (`bench.md`).
- **The t2296 cells are Windows-only** (deviation 3). "Link all five executables" cannot be run off Windows.
- **§4.3's lane description is not portable to Clang under the fp-free scan** (deviations 1 and 2). The plan anticipated no
  compiler-specific lowering, and the scan's allow-list is correctly not widened for it.
- **11.6 cannot pass on a runner without AVX-512.** S3 takes `intmath.cpp` about 1.2 points below its floor there, on top of
  S2's `matmul.cpp` shortfall. The owner's floor call (G27) now covers two files.
- **The base's GCC fp scan was already failing** (`TiledGemmAvx512`, fixed on main by 90e48de). The plan assumed a clean base.
- **§9's loop-bound mutant is killed first by an older cell.** In the suite's own order, a stack-row `test_main` cell trips
  glibc's stack protector before 4.S3 runs. Run first, 4.S3's sentinel cell kills it as the plan intends (`mutants.txt`).
