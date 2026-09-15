# Fused-K numeric specification

This is the engine-side numeric companion to [the artifact format](sslm_format.md).
It is the slice-2 specification for T-2693. The capture it defines is a slice-6
deliverable: slice 2 supplies its oracle, float factoring, snapshot/clone boundary, and
timing predictor; it does not add a capture seam or change C++ forward arithmetic.

## Index

1. Scope and authority
2. Normative arithmetic
3. Calibration and prefix flow
4. Operand registry
5. Multi-image census and retirement
6. Format and validation
7. Bounds and acceptance

## 1. Scope and authority

For a QK-norm layer, raw K landing is unchanged, then K RMSNorm produces wide integers,
wide K receives Q30 RoPE, one channel landing clamps to int8, and the result enters the
existing cache address. `KNormSiteScale[L] = m_kn * 2^e_kn` is existing KVC1 authority and
the only wide source is:

```
KWideSourceScale[L] = canonical_scale(127 * value(KNormSiteScale[L]))
```

`canonical_scale` is exact-rational, half-even normalized to `2^30 <= m < 2^31`.
The pair is derived once after the QK composition-domain validation; CPU, both GPU entry
points, converter capture, and the Python reference consume that same pair. A bare
`KNormSiteScale` is never a substitute.

## 2. Normative arithmetic

The loader-admitted gain domain is `g in [-128,127]`; bounds use `G=128`, not the
converter's narrower symmetric output. For every nonzero RMSNorm row, signed arithmetic is
greatest-integer division over the entire one-sparse class, including `code=-1`:

```
wide[d] = floor_div(h[d] * g[d] * normalization_numerator, root)
rot0 = round_nearest_away((wide0*cos_q30 - wide1*sin_q30) / 2^30)
rot1 = round_nearest_away((wide0*sin_q30 + wide1*cos_q30) / 2^30)
```

The source `S[L,H,D]` is exact binary64 bits and is the sole serialized authority:

```
S = max(float_post_RoPE_peak, integer_reference_post_RoPE_peak) / 127
B[L,H] = max_D S[L,H,D]
R[L,H,D] = round_nearest_away(S/B * 2^31)
B_carried[L,H] = canonical_scale(B/sqrt(head_dim))
```

`integer_reference_post_RoPE_peak` is `abs(rotated) * value(KWideSourceScale[L])`, in
the same real unit as the float peak. `R` is Int64, must satisfy `1 <= R <= 2^31`, and a
post-rounding zero is a construction refusal. The score uses fixed ascending channel order:

```
A = sum_d(int64(q[d]) * int64(k[d]) * R[L,H,d])
z = round_nearest_away(A / 2^31)
```

## 3. Calibration and prefix flow

The fixed independent population is `tools/reference_pipeline/data/shopkeeper_corpus_v1.jsonl`;
it is disjoint from corpus-239. The production rule is the max of float and integer-reference
post-RoPE peaks above, never T-2624's float-only shortcut. Slice 6 loads a provisional
table-bearing artifact, captures pass A, builds `max(float, A)`, captures pass B, builds
`max(float, A, B)`, then captures pass C against that final table. Pass C refuses only when
clipped direct-K landings exceed one per million observations. It alone owns
`ChannelScaleDidNotConverge`, capture equivalence obligations (1)/(4), and the observation mutant.

The exact common token-ID prefix is 453 tokens (not the 454-token standalone text rendering).
Slice 2's `t2693_prefix_clone` uses the ABI's SSB4 state serialization: residual/sequence
state and the complete KV block are copied into bytes, restored into a fresh pool block, and
suffixes run through existing `sslm_prefill`/`RunLayerLoop`. It proves restored bytes equal the
parent snapshot before continuation and re-saves the parent after every clone.

Float calibration factors the same exact token prefix. Its full and factored calibration
walks use the same per-query-row attention and projection reduction shape so BLAS may not
choose different reduction trees merely because a suffix has fewer rows. Maxima and derived
`StaticScales` must be byte-identical.

## 4. Operand registry

Every row has definition, domain, enforcement, and authority/derivation. This condensed table
is normative; names match the loader and forward vocabulary.

| # | Operand family | Domain and enforcement | Authority / derivation |
|---:|---|---|---|
| 1 | `L,H,D,d,p,i,token,position` | CFG bounds and position check | CFG geometry |
| 2 | geometry and `N=head_dim` | nonzero, GQA join, even; fused K `N=128` | CFG1 |
| 3 | converter containers/aliases | required unique keys, finite values | converter model |
| 4 | `F=NORM_FRAC_BITS=16` | CPU/GPU/reference equality | compile constants |
| 5 | `KNormSiteScale,m_kn,e_kn` | QK canonical-positive; generic KVC floor | KVC1 |
| 6 | `KWideSourceScale,m_wide,e_wide` | canonical, `m` Q30, `e [-74,14]` | factor-127 derivation |
| 7 | `value(m,e)`, half-even shift | positive input and canonical output | exact rational math |
| 8 | raw `h,k_raw,c` | signed landed code range | existing raw landing |
| 9 | `g,k_norm_gain,G` | WGT1 Int8 `[-128,127]` | WGT1 |
| 10 | `T,sumsq,root` | nonnegative; root positive | RMSNorm primitive |
| 11 | `wide,w,W` | abs <= 94,916,480 | signed full-domain proof |
| 12 | RoPE Q30 and `rotated` | table/domain/shape and int64 guard | ROP1 / RoPE primitive |
| 13 | float peak | finite nonnegative dense `(L,H,D)` | float calibration |
| 14 | integer-reference peak | finite, same dense shape/unit | slice-6 capture + wide pair |
| 15 | `S,k_channel_scale_bits` | finite positive binary64, zero sentinel non-QK | serialized source bits |
| 16 | `B` | finite positive same-head maximum | exact `S` derivation |
| 17 | `R` | Int64 `[1,2^31]` | away-rounded `S/B` |
| 18 | landing `r_t,e_t,m_t` | canonical target / reciprocal domain | exact `S` derivation |
| 19 | `kacc,k_row,clamp` | checked int64/U128 then `[-127,127]` | landing primitive |
| 20 | `q,k,S_q` | landed code range, positive carried scale | existing Q/QK path |
| 21 | `A,term,z` | fixed d order, signed int64 | Q31 score equation |
| 22 | `B_carried,softmax_khead` | canonical QK scale | `B/sqrt(N)` |
| 23 | retired QK scale chain | prohibited; reserved regions zero | no authority |
| 24 | i-exp constants | existing i-exp/width gates | existing contract |
| 25 | GPU `P,off,stride,Layout` | checked bytes, 66 words, old ranges zero | GPU layout |
| 26 | dispatch counts | 24 non-QK / 25 QK | GPU dispatch contract |
| 27 | channel table fields | Int64 shape and exact relation | source bits then derived fields |
| 28 | QK flag/mask | known mask `0x7`, bit 2 iff paired QK/table | WGT1 predicate + join |
| 29 | version/section | version 2; free table type only | format registry |
| 30 | rejected alternatives | analysis only, never runtime operand | fold comparison |
| 31 | paths/hashes | sorted POSIX bytes + SHA-256 | manifest |
| 32 | retrieval inputs | 239, float64, diagonal excluded | pinned analyzer |
| 33 | `t_A,t_B,t_C,t_float,t_total` | A/B/C <=30 min each; full flow about 80 min | monotonic timers |
| 34 | 453 prefix/snapshots/hashes | immutable parent, full suffix population | token arrays / SSB4 |
| 35 | saturation/out-of-domain counts | pass C clips at most one direct-K observation per million | capture telemetry |

## 5. Multi-image census and retirement

Ten families are dispositioned: nine live and one eliminated. Geometry, QK capability,
Q-gain, K-gain, Q site scale, K site scale/wide pair, RoPE cosine, RoPE sine, and the
channel `S/B/(r_t,e_t)/R/B_carried` family each have one authority. The channel family is
authoritative only at `k_channel_scale_bits`; every other image is recomputed and checked.

The tenth family is eliminated: tensor-wide `k_normed` maxima/scales, QK
`k_norm.requant`, `k_normed_head` KLR1, the second landing, QK `softmax.input`, old WSC1
gain duplicates, and old telemetry. At the pinned pre-build census its 28-layer artifact had
224 nonlinear rows, 224 KLR1 tuples, 28 K-side requant rows, 28 QK softmax-input rows, and
56 WSC1 candidates; final target counts are zero. Raw K/V KLR1, WGT1 gains, KVC1 q/k-norm,
and no-QK `softmax.input` remain live. GPU `off[62]`, `off[63]`, and scratch words 28/29 are
allocated zero-reserved, never reused.

## 6. Format and validation

The base schema remains v2. Bit 2 (`0x4`) is content-derived: paired QK WGT1 gains imply
bit 2 and a table; no-QK implies neither. The known mask is `0x7`; unknown bits reject.
The table has four dense Int64 fields: binary64 source bits, target reciprocal, target
exponent, and ratio. A non-QK row is canonical all zero.

Validation order is structural/header and generic KVC checks; QK bit/tensor/table/layer
join; source binary64 and row domains; retired-key refusal; QK composition domain; exact
source-to-derived relations; then marshal. `QkChannelScaleSourceOutOfDomain`,
`QkChannelScaleRelationMismatch`, `QkChannelRatioUnderflow`,
`QkChannelRatioOutOfDomain`, `LegacyFusedKMetadataPresent`, and
`LegacyFusedKKeyPresent` are loud before marshal. `k_norm.m=0` never reaches a canonicalizer.

## 7. Bounds and acceptance

At `N=128`, `sumsq <= 2,064,512`; the shifted RMS numerator is
`8,867,011,522,199,552`; full-Int8 one-sparse execution gives
`abs(wide) <= 94,916,480`; the two-product Q30 RoPE sum is
`203,831,588,725,719,040`; and the maximum landing pre-exponent product is
`1,750,900,014,122,044,826,866,155,520` (91 bits). The Q31 score has
`abs(A) <= 4,433,505,761,099,776` and `abs(z) <= 2,064,512`.

Slice 2's timing projection is only a predictor on the current compiled forward. Slice 6's
600-record cell is binding: float+A+B+C, including provisional/final table and artifact merges,
is about 80 minutes end to end, and each compiled pass stays within the 30-minute bar. Pass C
refuses only when clipped direct-K landings exceed one per million observations. A miss is a design
failure, never a scalar fallback.
