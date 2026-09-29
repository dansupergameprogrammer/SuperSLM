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

## S3 items

| Item | State | Evidence |
|---|---|---|
| Red-first: cells (4.S3 both fences, 7.S3 corner premises, 6.1, funnel call site, 6.3, 11.1(d) requant rows), per-tier `requant_row` counters (declared, not incremented), `RequantRowWide` declared with a stub that runs the element loop, the S3 rows appended to `c_rowsites`, the S3 golden hash | done (commit 1) | `docs/attention-rowsites/s3/red-suites.txt`: auto and forced AVX2/AVX-512 fail 10 assertions each, every one a path assertion; forced SSE2 0. Every value assertion passes on the base. Digests: every section but `c_rowsites` byte-identical to S2's on all five legs; `c_rowsites` `d0272180…` on all five |
| Golden pin (6.3) from the v1.9.0 tag | done | `golden.txt`: S3 `3e3abed7…` over 3,567,018 values; S1's and S2's unchanged |
| `RequantRowWide` in `src/intmath.cpp`: AVX2 (4 lanes) and AVX-512BW (8 lanes, F and BW only, no mask register) bodies of the §5.3 identity in unsigned 64-bit lanes with a logical H shift, P formed from r's 32-bit halves, clamp and sign restore as the element code; the tail and the scalar/SSE2 tiers run `RequantTokenCodeWide`; per-tier `requant_row` counter inside each body; dispatch through S2's `DispatchSitesKernel` (so `SUPERSLM_SITES_AVX512_MSVC` covers it) | done (commit 2) | `src/intmath.cpp` |
| The funnel's element loop becomes one `RequantRowWide` call | done (commit 2) | `src/forward/checked_chain_funnel.cpp` |
| GCC 13.3 suites at the implementation commit: auto (AVX-512 here), forced SSE2, AVX2, AVX-512 | done, 0 failures | 109,744 / 109,686 / 109,702 / 109,702 checks (attn-rowsites 84,213 each); digests equal the red run's on all five legs (GLOBAL `4892b5be…`) |
| 11.4 forward-leaf check lists `RequantRowWide`; planted call from `forward_sites.cpp` turns it red | done (commit 2) | `tests/ci/check_no_forward_leaf_calls.py` (+ its 83-cell pytest, green); plant in `leaf-plant.txt` |
| 11.3 linkage checker gains the intmath.cpp objects (population `RequantRowAvx`, the record only in matmul objects); CI job passes all six objects; isolation checker's prose names the two bodies | done (commit 2) | OK on all six objects; plant (RequantRowAvx2 given external linkage) red, `linkage-plant.txt` |
| 11.5 the four recipes of G21 gain `src\matmul.cpp`; GCC link check | done (commit 2) | `recipe-link.txt`: without matmul.cpp all eight link units fail on `detail::ActiveGemmTier` / `DispatchSitesKernel`; with it all link. The five t2296 cells include `<windows.h>`, so their engine source sets are linked as closed sets (`-shared -Wl,--no-undefined`) |
