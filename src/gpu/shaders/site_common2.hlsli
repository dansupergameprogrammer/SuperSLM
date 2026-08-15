// T-2039 (GPU-serial port: real-capacity shader geometry). Primitives
// site_common.hlsli does not carry -- RoPE's own rotation (RopeApplyPair),
// the i-exp softmax construction (IExpConstruct/IExpEvaluate/
// IExpScaleConstants/CheckSoftmaxRowWidthDomain), a buffer-resident
// SoftmaxRowQ15 (no per-call local array -- the row lives in a caller-
// supplied buffer window, sized at the real per-call `context_cap`, never a
// compile-time capacity -- T-2052 M4, Claude/Poirot/36b9327-gpu-serial-
// port-reconfirmation-review.md: the caller is LayerScratch's own persistent
// `scores` region since T-2045's own C3 split attention into four separate
// dispatches, never a WorkScratch window -- see SoftmaxRowQ15BufGpu's own
// header comment below), and the SiLU LUT lookup (SiluSigmoidQ15/
// CheckSiluCompositionScaleDomain, already array-free). Every function has a
// named C++ sibling; see src/intmath.cpp, src/matmul.cpp, src/silu_lut.cpp,
// src/forward/checked_chain_funnel.cpp.
#ifndef SSLM_SITE_COMMON2_HLSLI
#define SSLM_SITE_COMMON2_HLSLI

#include "site_common.hlsli"

// wide128.hlsli-family helpers this file needs but B1's own family doesn't
// carry: a zeroed U128 and a full-width (non-narrowing) right shift on
// intmath.cpp's OWN U128 (distinct struct instance from landing_rescale.hlsli's
// forward_sites.cpp family, per that file's own header comment on why the two
// are not shared). Declared FIRST -- IExpScaleConstantsGpu below uses both.
U128 UZeroGpu() { U128 z; z.lo = 0ULL; z.hi = 0ULL; return z; }
U128 UShrFullGpu(U128 v, int k)
{
    if (k <= 0) return v;
    if (k >= 128) return UZeroGpu();
    if (k >= 64) { U128 r; r.lo = v.hi >> (uint)(k - 64); r.hi = 0ULL; return r; }
    U128 r;
    r.lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
    r.hi = v.hi >> (uint)k;
    return r;
}

// intmath.cpp RoundingDivideByPOTImpl<int64_t> -- ties away from zero, int64 width
// (RopeApplyPair's own rounding rule, and SiluSigmoidQ15's right branch).
int64_t RoundingDivideByPotI64Gpu(int64_t x, int exponent)
{
    uint64_t mask = (exponent <= 0) ? 0ULL : ((1ULL << (uint)exponent) - 1ULL);
    uint64_t remainder = (uint64_t)x & mask;
    uint64_t threshold = (mask >> 1) + ((x < 0) ? 1ULL : 0ULL);
    int64_t shifted = x >> exponent;  // arithmetic (signed) shift
    return shifted + ((remainder > threshold) ? (int64_t)1 : (int64_t)0);
}

static const int kRopeFracBitsGpu = 30;

// intmath.cpp RopeApplyPair: (x*cos - y*sin, x*sin + y*cos), int64-exact, rounded
// once at ROPE_FRAC_BITS. Unclamped -- the caller clamps (ClampRopeCodeGpu).
void RopeApplyPairGpu(int x, int y, int cos_q30, int sin_q30, out int64_t out_x, out int64_t out_y)
{
    int64_t xr = (int64_t)x * (int64_t)cos_q30 - (int64_t)y * (int64_t)sin_q30;
    int64_t yr = (int64_t)x * (int64_t)sin_q30 + (int64_t)y * (int64_t)cos_q30;
    out_x = RoundingDivideByPotI64Gpu(xr, kRopeFracBitsGpu);
    out_y = RoundingDivideByPotI64Gpu(yr, kRopeFracBitsGpu);
}

// --- intmath.cpp i-exp construction (C7/C30), bit-exact -----------------------

static const int64_t kIExpClipNGpu = 30;  // I_EXP_CLIP_N

// IExpDomain tags: 0=kOk, 1=kNotRepresentable, 2=kBadQ, 3=kBadQLn2, 4=kBadQB.
// Returns the domain tag; on kOk (0) writes out_z/out_base/out_qc (the
// IExpConstruction the caller passes to IExpEvaluateGpu).
int IExpConstructGpu(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c, out int64_t out_z,
                      out int64_t out_base, out int64_t out_qc)
{
    out_z = 0; out_base = 0; out_qc = 0;
    if (q > 0) return 2;  // kBadQ

    int64_t kIExpMaxQLn2 = 0x7FFFFFFFFFFFFFFFLL / kIExpClipNGpu;
    if (q_ln2 < 1 || q_ln2 > kIExpMaxQLn2) return 3;  // kBadQLn2

    int64_t clip_lo = -kIExpClipNGpu * q_ln2;
    int64_t clipped = (q < clip_lo) ? clip_lo : q;
    int64_t z = (-clipped) / q_ln2;
    int64_t q_p = clipped + z * q_ln2;

    // q_b < INT64_MIN - q_p, computed without UB: q_p <= 0 so INT64_MIN - q_p
    // is representable except at q_p == INT64_MIN, which cannot occur here
    // (q_p in (-q_ln2, 0], q_ln2 >= 1, so q_p > INT64_MIN always).
    int64_t bad_qb_threshold = (int64_t)0x8000000000000000ULL - q_p;
    if (q_b < bad_qb_threshold) return 4;  // kBadQB
    int64_t base = q_p + q_b;

    out_z = z;
    out_base = base;
    out_qc = q_c;

    S128 v = SAdd(SMul(base, base), SFromI64(q_c));
    S128 limit = STwice(SFromI64((int64_t)1 << 62));
    for (int64_t i = 0; i < z; ++i) limit = STwice(limit);
    S128 upper = SSub(limit, SFromI64(1));
    S128 lower = SSub(SFromI64(0), limit);
    bool ok = SGe(upper, v) && SGe(v, lower);
    return ok ? 0 : 1;  // kOk : kNotRepresentable
}

// intmath.cpp IExpEvaluate: (base^2 + q_c) >> z.
int64_t IExpEvaluateGpu(int64_t z, int64_t base, int64_t qc)
{
    S128 v = SAdd(SMul(base, base), SFromI64(qc));
    return SShrToI64(v, (int)z);
}

// intmath.cpp IExpScaleConstants (C30's per-token derivation). Returns the
// IExpScaleDomain tag (0=kOk, 1=kBadCoefficient, 2=kNegativeShift,
// 3=kNotRepresentable); writes out_q_ln2/out_q_b/out_q_c only on kOk.
int IExpScaleConstantsGpu(int64_t m, int64_t e, int64_t ln2_q, int ln2_fmt, int64_t b_q, int b_fmt,
                           int64_t ca_q, int ca_fmt, out int64_t out_q_ln2, out int64_t out_q_b,
                           out int64_t out_q_c)
{
    out_q_ln2 = 0; out_q_b = 0; out_q_c = 0;
    if (ln2_q <= 0 || b_q <= 0 || ca_q <= 0) return 1;  // kBadCoefficient

    // funnel_guards.hlsli's own SaturatingAdd64Gpu_ (identical overflow-safe add
    // this derivation's three shift terms need, checked_chain_funnel.cpp's own
    // CombineCarriedScale-shared technique) -- reused rather than reinvented.
    int64_t shift_ln2 = SaturatingAdd64Gpu_(SaturatingAdd64Gpu_((int64_t)ln2_fmt, 62), e);
    int64_t shift_b = SaturatingAdd64Gpu_(SaturatingAdd64Gpu_((int64_t)b_fmt, 62), e);
    int64_t two_e = SaturatingAdd64Gpu_(e, e);
    int64_t shift_c = SaturatingAdd64Gpu_(SaturatingAdd64Gpu_((int64_t)ca_fmt, 124), two_e);

    int64_t min_shift = shift_ln2;
    if (shift_b < min_shift) min_shift = shift_b;
    if (shift_c < min_shift) min_shift = shift_c;
    if (min_shift < 0) return 2;  // kNegativeShift

    int64_t r_m = DynamicScaleReciprocalSharedGpu(m);
    uint64_t ur_m = (uint64_t)r_m;

    U128 num_ln2 = UMul((uint64_t)ln2_q, ur_m);
    U128 num_b = UMul((uint64_t)b_q, ur_m);
    U128 rm2 = UMul(ur_m, ur_m);
    U128 num_c = UMulWide(rm2, (uint64_t)ca_q);

    uint64_t u_q_ln2 = (shift_ln2 >= 128) ? 0 : UShrToU64(num_ln2, (int)shift_ln2);
    uint64_t u_q_b = (shift_b >= 128) ? 0 : UShrToU64(num_b, (int)shift_b);
    uint64_t u_q_c = (shift_c >= 128) ? 0 : UShrToU64(num_c, (int)shift_c);

    // HLSL's ?: does not support struct-typed results (U128) -- if/else in
    // place of the C++ ternary this mirrors.
    U128 full_ln2, full_b, full_c;
    if (shift_ln2 >= 128) full_ln2 = UZeroGpu(); else full_ln2 = UShrFullGpu(num_ln2, (int)shift_ln2);
    if (shift_b >= 128) full_b = UZeroGpu(); else full_b = UShrFullGpu(num_b, (int)shift_b);
    if (shift_c >= 128) full_c = UZeroGpu(); else full_c = UShrFullGpu(num_c, (int)shift_c);
    if (!UFitsI64(full_ln2) || !UFitsI64(full_b) || !UFitsI64(full_c)) return 3;  // kNotRepresentable

    out_q_ln2 = (int64_t)u_q_ln2;
    out_q_b = (int64_t)u_q_b;
    out_q_c = (int64_t)u_q_c;
    return 0;  // kOk
}

// checked_chain_funnel.cpp CheckSoftmaxRowWidthDomain.
bool CheckSoftmaxRowWidthDomainGpu(int64_t q_b, int64_t q_c, int width)
{
    if (q_c < 0) return false;
    if (width == 0) return false;
    S128 m128 = SAdd(SMul(q_b, q_b), SFromI64(q_c));
    static const int64_t kSoftmaxRowMaxSafeExponent = ((int64_t)1) << 47;
    bool m_usable = SGe(SFromI64(0x7FFFFFFFFFFFFFFFLL), m128) && SGe(m128, SFromI64(1)) &&
                     SGe(SFromI64(kSoftmaxRowMaxSafeExponent), m128);
    return m_usable;
}

// T-2113 (B10 lever 2, D-SLM3400): the group-cooperative row primitive `softmax_site.hlsl` calls
// today. The single-thread-per-head sequential predecessor this replaced (`SoftmaxRowQ15BufGpu`,
// ported bit-exact from `intmath.cpp`'s own `SoftmaxRowQ15`) is retired -- its last caller was
// converted to this function in the same change, so no unreferenced scaffolding is left in HEAD
// (StandardsDocument.md §6.6); the sequential algorithm's own contract (bit-exact to
// `intmath.cpp:799`'s `width == 0` short-circuit, the m-bound re-check pass, the CPU sibling's
// own `!m_usable` all-zero disposition) is preserved unchanged inside the phases below, only the
// STRIPING and the two row-wide reductions changed shape.
// `softmax_site.hlsl`'s own per-head loop assigns ONE thread per head (`for h = t; h <
// num_attention_heads; h += 256`), which at the real 1.5B tier's 12 heads leaves 244 of 256
// threads idle while the 12 active ones each run the sequential algorithm above alone, at a cost
// that grows with context width (T-2105 §6/§8: 0.397 ms at ctx<=16 to 3.169 ms at ctx<=256) --
// the identical low-occupancy shape D-SLM3398 found and D-SLM3399 fixed for the adapter-delta
// stage-1 reduction, now applied to softmax's own row.
//
// `lanes` threads (a power of two, computed by the caller from 256/num_attention_heads, the same
// `Stage1LanesForRank`-shaped derivation) cooperate on ONE head's own `width`-wide row, striped so
// thread `lane` (0 <= lane < lanes) owns indices {lane, lane+lanes, lane+2*lanes, ...} in EVERY
// phase below -- every element's own per-index computation (the i-exp construction/evaluation,
// the m-bound check, the Q15 divide) is a pure function of that element's own score and the
// row-wide broadcast values (`peak`, `denom`), IDENTICAL to what the sequential primitive above
// computes for that index in any order, and because every phase re-uses the SAME striping, a
// lane's own UAV writes are always read back by that SAME lane's own later phase -- ordinary
// same-thread program order, no barrier needed for that. Only the two ROW-WIDE reductions (max,
// sum) and the well-formed flag's OR-across-lanes need cross-thread coordination, each realized
// as a fixed groupshared binary tree over, respectively, integer max, integer addition, and
// integer min-as-AND -- all three exactly associative/commutative (the same argument
// `GemmCoalescedGpuAt`'s and `ApplyFusedAdapterDeltaGpu`'s own header comments make, D-SLM3334),
// so the tree returns the IDENTICAL result the sequential primitive's own left-to-right reduction
// returns, for any width and lane count.
//
// `gSoftmaxReduceAcc` (below) is REUSED, barrier-separated, across this function's own three
// sequential tree phases (max, well-formed, sum) and across every concurrently active head-group
// in the same wave (each head-group touches only its own disjoint `[group_base, group_base+lanes)`
// window of `t`, exactly `GemmCoalescedGpuAt`'s own multi-channel-per-dispatch precedent). `active`
// is uniform across a head-group's own `lanes` threads (a function of the head index alone) and
// MUST be reached unconditionally by every thread on every wave iteration regardless of its own
// value -- a thread whose row is inactive (past `num_attention_heads`, or its head failed the
// domain check the caller already ran) contributes a neutral identity value at every reduction
// and performs no buffer I/O, but still executes every barrier below: skipping a barrier for an
// inactive thread while an active sibling reaches it is undefined behavior (the identical
// discipline `GemmCoalescedGpuAt`'s own header comment states for its own out-of-range channels).
groupshared int64_t gSoftmaxReduceAcc[256];

bool SoftmaxRowQ15BufGpuParallel(bool active, uint t, uint lane, uint lanes, RWByteAddressBuffer buf,
                                    uint base, int width, int64_t q_ln2, int64_t q_b, int64_t q_c)
{
    // Phase 1: max-reduce, striped.
    int64_t local_max = (int64_t)0x8000000000000000ULL;  // INT64_MIN -- the additive identity for
                                                            // max, safe both for an inactive thread
                                                            // and for a lane whose own stripe is
                                                            // empty (width < lanes).
    if (active)
    {
        for (int i = (int)lane; i < width; i += (int)lanes)
        {
            int64_t v = buf.Load<int64_t>(base + (uint)i * 8u);
            if (v > local_max) local_max = v;
        }
    }
    gSoftmaxReduceAcc[t] = local_max;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = lanes >> 1; s > 0u; s >>= 1)
    {
        if (lane < s && gSoftmaxReduceAcc[t + s] > gSoftmaxReduceAcc[t])
            gSoftmaxReduceAcc[t] = gSoftmaxReduceAcc[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    int64_t peak = gSoftmaxReduceAcc[t - lane];
    GroupMemoryBarrierWithGroupSync();  // every lane has its own copy of `peak` before
                                          // `gSoftmaxReduceAcc` is reused below.

    // Phase 2: per-element i-exp compute + store, striped -- a pure map, no cross-thread
    // coordination needed (each lane's own writes are visible to its own subsequent reads
    // without a barrier).
    bool well_formed_local = true;
    if (active)
    {
        for (int k = (int)lane; k < width; k += (int)lanes)
        {
            int64_t shifted = buf.Load<int64_t>(base + (uint)k * 8u) - peak;
            int64_t z, ebase, qc;
            int d = IExpConstructGpu(shifted, q_ln2, q_b, q_c, z, ebase, qc);
            int64_t value = 0;
            if (d == 2 || d == 3 || d == 4) well_formed_local = false;
            else value = IExpEvaluateGpu(z, ebase, qc);
            buf.Store<int64_t>(base + (uint)k * 8u, value);
        }
    }

    // Phase 3: the m-bound re-check + sum-reduce, striped -- SAME indices per lane as phase 2.
    // `m_usable` is a pure function of q_b/q_c (call-level constants), computed redundantly and
    // identically by every lane -- matching the sequential primitive's own unconditional per-call
    // re-derivation rather than trusting the caller.
    S128 m128 = SAdd(SMul(q_b, q_b), SFromI64(q_c));
    static const int64_t kSoftmaxRowMaxSafeExponentP = ((int64_t)1) << 47;
    bool m_usable = SGe(SFromI64(0x7FFFFFFFFFFFFFFFLL), m128) && SGe(m128, SFromI64(1)) &&
                     SGe(SFromI64(kSoftmaxRowMaxSafeExponentP), m128);
    int64_t m = m_usable ? SLow64(m128) : 0;
    int64_t local_total = 0;
    if (active)
    {
        for (int k2 = (int)lane; k2 < width; k2 += (int)lanes)
        {
            int64_t value = 0;
            if (m_usable)
            {
                value = buf.Load<int64_t>(base + (uint)k2 * 8u);
                if (value < 0 || value > m) { well_formed_local = false; value = 0; }
            }
            else
            {
                well_formed_local = false;
            }
            buf.Store<int64_t>(base + (uint)k2 * 8u, value);
            local_total += value;
        }
    }

    // well-formed AND-reduce (0/1, min = AND), reusing `gSoftmaxReduceAcc`.
    gSoftmaxReduceAcc[t] = well_formed_local ? 1 : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint s2 = lanes >> 1; s2 > 0u; s2 >>= 1)
    {
        if (lane < s2 && gSoftmaxReduceAcc[t + s2] < gSoftmaxReduceAcc[t])
            gSoftmaxReduceAcc[t] = gSoftmaxReduceAcc[t + s2];
        GroupMemoryBarrierWithGroupSync();
    }
    bool all_well_formed = gSoftmaxReduceAcc[t - lane] != 0;
    GroupMemoryBarrierWithGroupSync();  // every lane has read the group's own well-formed flag
                                          // before `gSoftmaxReduceAcc` is reused for the sum-reduce.

    // sum-reduce, reusing `gSoftmaxReduceAcc`.
    gSoftmaxReduceAcc[t] = local_total;
    GroupMemoryBarrierWithGroupSync();
    for (uint s3 = lanes >> 1; s3 > 0u; s3 >>= 1)
    {
        if (lane < s3) gSoftmaxReduceAcc[t] += gSoftmaxReduceAcc[t + s3];
        GroupMemoryBarrierWithGroupSync();
    }
    int64_t total = gSoftmaxReduceAcc[t - lane];
    GroupMemoryBarrierWithGroupSync();  // every lane has read `total` before `gSoftmaxReduceAcc`
                                          // is free for the NEXT head-group's own wave iteration.

    // Phase 4: normalize, striped -- SAME indices per lane as phases 2/3.
    if (active)
    {
        int64_t denom = (total > 1) ? total : 1;
        static const int kProbFracBitsGpuP = 15;
        for (int j = (int)lane; j < width; j += (int)lanes)
        {
            int64_t e = buf.Load<int64_t>(base + (uint)j * 8u);
            buf.Store<int64_t>(base + (uint)j * 8u, (e << kProbFracBitsGpuP) / denom);
        }
    }
    return all_well_formed;
}

// checked_chain_funnel.cpp CheckSiluCompositionScaleDomain.
static const int64_t kCompositionScaleMaxAbsMGpu = ((int64_t)1 << 31) - 1;
static const int kSiluCompositionRuntimeMinEGpu = -80;
static const int kSiluCompositionRuntimeMaxEGpu = 8;

bool CheckSiluCompositionScaleDomainGpu(int64_t m, int e)
{
    if (m < -kCompositionScaleMaxAbsMGpu || m > kCompositionScaleMaxAbsMGpu) return false;
    if (e < kSiluCompositionRuntimeMinEGpu || e > kSiluCompositionRuntimeMaxEGpu) return false;
    return true;
}

static const int kSiluLutNGpu = 1024;
static const int kSiluLutLog2KGpu = 5;
static const int kSiluLutQIdxGpu = 12;

// silu_lut.cpp SiluSigmoidQ15 -- `table` is the SIL1 SRV (kSiluLutN+1 = 1025 int32
// entries), `LayerWeights`-style ByteAddressBuffer read.
int SiluSigmoidQ15Gpu(ByteAddressBuffer table, int code, int64_t m, int e)
{
    int64_t term = (int64_t)code * m;
    int shift = e + kSiluLutLog2KGpu + kSiluLutQIdxGpu;
    int64_t pos_fixed = (shift >= 0) ? (term << shift) : RoundingDivideByPotI64Gpu(term, -shift);

    int64_t domain_hi = ((int64_t)kSiluLutNGpu) << kSiluLutQIdxGpu;
    pos_fixed += domain_hi / 2;
    if (pos_fixed < 0) pos_fixed = 0;
    else if (pos_fixed > domain_hi) pos_fixed = domain_hi;

    int64_t i0_64 = pos_fixed >> kSiluLutQIdxGpu;
    if (i0_64 > kSiluLutNGpu - 1) i0_64 = kSiluLutNGpu - 1;
    int i0 = (int)i0_64;
    int64_t frac = pos_fixed - ((int64_t)i0 << kSiluLutQIdxGpu);

    int64_t lo = (int64_t)(int)table.Load((uint)i0 * 4u);
    int64_t hi = (int64_t)(int)table.Load((uint)(i0 + 1) * 4u);
    int64_t diff = hi - lo;
    int64_t product = frac * diff;
    int64_t delta = RoundingDivideByPotI64Gpu(product, kSiluLutQIdxGpu);
    return (int)(lo + delta);
}

#endif  // SSLM_SITE_COMMON2_HLSLI
