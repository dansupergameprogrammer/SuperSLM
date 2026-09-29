# Attention and per-row sites, slice S1: build progress

The plan of record is the attention and per-row sites plan, rev 3.1, approved by the owner. This series builds
its slice S1 only (§4.1): 256-entry per-row tables for the SwiGLU sigmoid (`MlpActSite`), the residual landing
(`ResidualReconcileSite`) and the RMSNorm divide (`RmsNormSite`). Same pure function, same arguments; every tier
bit-identical to the v1.9.0 per-element code.

**Base.** Tiled-matmul slice 1's branch at `fb56397` (the plan names `c3e0004`; `fb56397` is that plus one
documentation commit). Delivered as a patch series, not pushed: that branch carries slice 1's draft PR.

**Host.** A 4-vCPU cloud Xeon with AVX2, AVX-512F/BW/DQ and AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3. The host is
shared with other agents, so every timing is best-of-N and noisy.

States: **done**, **red** (written and failing, as red-first requires), **box-only**, **CI-only** (runs on the
hosted legs, not runnable here; stated why), **pending**.

## Resume here

1. Clone SuperSLM, `git checkout -b attn-s1 origin/claude/project-thread-c8iecr`, `git am` the series.
2. The 11.1(d) cell needs the 0.5B-width 1-layer synthetic artifact (sha256 `f0fd4886…6ed3`, about 9 minutes and
   7.4 GiB with `tools/consumer_reach/synth_artifact.py`); point `SUPERSLM_ATTN_ROWSITES_ARTIFACT` at it and run each
   suite binary from the repository root. Without it the cell prints SKIPPED.
3. The golden pin is regenerated with `tools/gen_attn_rowsite_golden.cpp` built against the v1.9.0 tag (recipe in
   its header).

## S1 items

| Item | State | Evidence |
|---|---|---|
| Row-table path counters in the instrument seam (§3.6), outside the x64 block | red commit: declared, not incremented | `tests/support/matmul_dispatch_instrument.h` |
| Cells 4.S1, 1.S1, 5.S1, 7.S1a–d, 11.1(a) row tables, 6.3, 11.1(d) row-table and trace rows | red: every counter assertion fails on the base, every value assertion passes | `docs/attention-rowsites/s1/red-suites.txt` (GCC auto, forced SSE2/AVX2/AVX-512: 1,756 checks, 258 failures each, all counter assertions) |
| Digest section `c_rowsites` (6.2) | done | `8836d5eb…` on all five GCC legs; sections 1–9 byte-identical to slice 1's |
| Golden pin (6.3) from the v1.9.0 tag | done | `tests/attn_rowsite_golden_pin.h`, `8836d5eb…` over 634,120 values |

## Deviations from the plan, and why

- **11.1(c) (the QK-norm attention fixture) is not built in S1.** Its table's first column is "S5 landed"; the
  fixture exists to drive the Q31 path, and its row-table rows (all `…_skipped`, widths 64 and 256) add nothing
  4.S1 does not already assert at widths 1 to 511. S5 builds it and adds its forward hash to the pin header.
- **The golden pin is one hash per slice**, not one hash over every section's inputs. S1's hash covers
  `c_rowsites`' S1 entries; S3 and S2/S4–S6 add their own constants beside it, so no later slice regenerates S1's.
- **11.1(d) is an environment-gated cell in the suite binaries**, since the plan does not commit the artifact.
  S1 asserts its structural rows that exist in S1: the three `rowtable_*_taken` closed forms, every `…_skipped` 0,
  and the chain trace records ((11·L + 1)·N, N per site name). The `requant_row`, `softmax`, `pv` and `grouped`
  rows arrive with their slices.

## ID-PENDING list

Nothing minted here.
