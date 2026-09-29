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
