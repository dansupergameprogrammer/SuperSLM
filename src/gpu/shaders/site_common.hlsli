// T-2032 (Sec5.6/Sec11 B4 well-scoped next checkpoint): shared primitives for
// the composed pipeline's first four REAL sites -- attn_norm, q_proj, kv_proj
// (fused), and RoPE's own guard. Every function here has a named C++ sibling;
// see src/forward/forward_sites.cpp, src/intmath.cpp, and
// src/forward/checked_chain_funnel.cpp. B1/B2's own funnel_guards.hlsli is
// reused rather than duplicated (CombineCarriedScale, NormalizeScale,
// CarriedScaleMantissaFitsInt32, the composed-exponent domain check,
// SaturatingRoundingDoublingHighMul).
#ifndef SSLM_SITE_COMMON_HLSLI
#define SSLM_SITE_COMMON_HLSLI

#include "wide128.hlsli"
#include "funnel_guards.hlsli"
#include "landing_rescale.hlsli"

static const int kMaxSiteN = 8;  // generous headroom over this suite's own hidden_size/kv_hidden_size (2)

// T-2032's own local status-tag encoding (mirrors check_rdp_exponent.hlsl /
// requant_chain_checked.hlsl's already-established convention) -- mapped to
// the real superslm::SslmForwardStatus by name on the host side.
static const int64_t kTagOk = 0;
static const int64_t kTagCarriedScaleMantissaOutOfDomain = 1;
static const int64_t kTagChainInputOutOfDomain = 2;
static const int64_t kTagRoundingDivideByPotExponentOutOfDomain = 3;
static const int64_t kTagBiasReconcileProductOutOfDomain = 4;
static const int64_t kTagRopeTableTensorMissing = 5;
static const int64_t kTagRopeTableExtentExceeded = 6;
static const int64_t kTagPositionOverCap = 7;
static const int64_t kTagNotYetImplemented = 8;
static const int64_t kTagSoftmaxRowWidthOutOfDomain = 9;
static const int64_t kTagIExpScaleDerivationOutOfDomain = 10;
static const int64_t kTagSoftmaxKernelRefusedAfterGateAccepted = 11;
static const int64_t kTagResidualReconciliationMagnitudeOutOfDomain = 12;
static const int64_t kTagSiluCompositionScaleOutOfDomain = 13;

// forward_sites.cpp FloorDivI64.
int64_t FloorDivI64Gpu(int64_t a, int64_t b)
{
    int64_t q = a / b;
    int64_t r = a % b;
    return (r != 0 && r < 0) ? (q - 1) : q;
}

// intmath.cpp ISqrt (ISqrtIterates): 32-iteration restoring shift-and-
// subtract, unconditional op count (Sec14: no data-dependent early exit).
int64_t ISqrtGpu(int64_t n)
{
    int64_t remainder = n;
    int64_t root = 0;
    int64_t bit = (int64_t)1 << 62;
    [unroll]
    for (int i = 0; i < 32; ++i)
    {
        int64_t trial = root + bit;
        if (remainder >= trial)
        {
            remainder -= trial;
            root = (root >> 1) + bit;
        }
        else
        {
            root = root >> 1;
        }
        bit = bit >> 2;
    }
    return root;
}

// intmath.cpp RoundingDivideByPOTImpl<int32_t> -- ties away from zero.
int RoundingDivideByPotI32Gpu(int x, int exponent)
{
    uint mask = (exponent <= 0) ? 0u : ((1u << (uint)exponent) - 1u);
    uint remainder = (uint)x & mask;
    uint threshold = (mask >> 1) + ((x < 0) ? 1u : 0u);
    int shifted = x >> exponent;  // arithmetic (signed) shift
    return shifted + ((remainder > threshold) ? 1 : 0);
}

// forward_sites.cpp ApplyWeightScaleFold.
int64_t ApplyWeightScaleFoldGpu(int64_t acc, int identity, int mult, int shift)
{
    if (identity != 0) return acc;
    int hm = SaturatingRoundingDoublingHighMulGpu_((int)acc, mult);
    return (int64_t)RoundingDivideByPotI32Gpu(hm, shift);
}

// Sign-extended single-byte load from a ByteAddressBuffer -- HLSL has no int8
// scalar type at this compile target (cs_6_2), so every int8 weight/code read
// goes through this word-load-then-extract idiom, matching B3's own already-
// proven descriptor_table_readback.hlsl::LoadSignedByte exactly (Sec5.2's
// corrected shift-left-then-arithmetic-right-shift sign extension).
int LoadSignedByteGpu(ByteAddressBuffer buf, uint byteOff)
{
    uint w = buf.Load((byteOff / 4u) * 4u);
    uint shift = (byteOff % 4u) * 8u;
    uint b = (w >> shift) & 0xFFu;
    return int(b << 24) >> 24;
}

// Overload for a UAV source (the KV cache, read back through the same
// sign-extend idiom as the read-only SRV overload above -- HLSL resolves by
// buffer type, not by const-ness, so a genuinely separate overload is needed
// rather than an implicit conversion).
int LoadSignedByteGpu(RWByteAddressBuffer buf, uint byteOff)
{
    uint w = buf.Load((byteOff / 4u) * 4u);
    uint shift = (byteOff % 4u) * 8u;
    uint b = (w >> shift) & 0xFFu;
    return int(b << 24) >> 24;
}

// Read-modify-write single-byte store into a ByteAddressBuffer, the write-
// side sibling of LoadSignedByteGpu above -- used for the K/V cache's own
// int8-per-element layout (KeyRow/ValueRow's own addressing, matched exactly
// so the CPU-side accessors read the identical bytes this dispatch writes).
// Safe under this design's own execution model: every dispatch is fully
// serialized by a global UAV barrier (Sec5.6), and within one dispatch these
// stores run on a single thread (numthreads(1,1,1), Sec11.2's own stated
// simplification for this suite's small fixture geometry) in program order.
void StoreSignedByteGpu(RWByteAddressBuffer buf, uint byteOff, int val)
{
    uint wordOff = (byteOff / 4u) * 4u;
    uint shift = (byteOff % 4u) * 8u;
    uint word = buf.Load(wordOff);
    uint mask = ~(0xFFu << shift);
    uint newWord = (word & mask) | (((uint)val & 0xFFu) << shift);
    buf.Store(wordOff, newWord);
}

// intmath.cpp GemmInt8AccumulateRow, one output channel -- plain int64 dot
// product over already-loaded local int arrays (design Sec4: "proven
// bit-identical to the SSE2 production path by associativity", so summation
// order is not load-bearing here).
int64_t GemmDotGpu(int in_codes[kMaxSiteN], int weight_row[kMaxSiteN], int in_channels)
{
    int64_t acc = 0;
    for (int i = 0; i < in_channels; ++i)
    {
        acc += (int64_t)in_codes[i] * (int64_t)weight_row[i];
    }
    return acc;
}

// intmath.cpp DynamicScaleReciprocal -- 3 Newton iterations + 2 branch-free
// correction steps (bit-exact copy of dyn_recip.hlsl's own already-GPU-
// verified DynamicScaleReciprocalGpuImpl, factored here for reuse by
// RequantChainCheckedFullGpu's step 6 below).
int64_t DynamicScaleReciprocalSharedGpu(int64_t dn)
{
    const int64_t kC32 = (2 * ((int64_t)48 << 31) + 17) / 34;
    const int64_t kC32_2 = (2 * ((int64_t)32 << 31) + 17) / 34;
    int64_t y = kC32 - ((kC32_2 * dn) >> 31);
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        int64_t dn_y = SShrToI64(SMul(dn, y), 31);
        int64_t delta = ((int64_t)1 << 32) - dn_y;
        y = SShrToI64(SMul(y, delta), 31);
    }
    [unroll]
    for (int j = 0; j < 2; ++j)
    {
        S128 residual_2x = STwice(SSub(SFromI64((int64_t)1 << 62), SMul(y, dn)));
        if (SGe(residual_2x, SFromI64(dn))) { y += 1; }
        else if (SLt(residual_2x, SFromI64(-dn))) { y -= 1; }
    }
    return y;
}

// intmath.cpp RequantTokenCodeWide (C22's formula), bit-exact copy of
// requant_wide.hlsl's own already-GPU-verified body, factored for reuse.
int64_t RequantTokenCodeWideSharedGpu(int64_t x_i, int64_t r, int s)
{
    int exponent = 62 - s;
    uint64_t abs_x = (x_i < 0) ? (~(uint64_t)x_i + 1ULL) : (uint64_t)x_i;
    U128 prod = UMulWide(UMul(abs_x, (uint64_t)r), 127ULL);
    U128 numerator = UAdd64(UTwice(prod), ((uint64_t)1) << exponent);
    uint64_t magnitude = UShrToU64(numerator, exponent + 1);
    if (magnitude > 127ULL) magnitude = 127ULL;
    int q = (int)magnitude;
    int code = (x_i < 0) ? -q : q;
    return (int64_t)code;
}

// checked_chain_funnel.cpp RequantChainChecked, IN FULL (steps 0-6):
// combines B2's already-proven guard tier (requant_chain_checked.hlsl) with
// step 6's per-element code write (requant_wide.hlsl's own primitive, shared
// above) -- the "RequantChainCheckedFull" the T-2031 composed-pipeline
// analysis names as the new primitive every real site above B2's own
// guard-only tier actually needs (Claude/Brunel/t2025-gpu-serial-build-2026-
// 08-13.md Sec11.2). `n_incoming` is 0 or 1 for this checkpoint's own real
// sites (attn_norm: 0; q_proj: 1, the normed carried scale) -- headroom to
// kMaxSiteN is carried for a future site that needs more, not exercised here.
void RequantChainCheckedFullGpu(int64_t wide[kMaxSiteN], int n, int64_t incoming_m[kMaxSiteN],
                                 int64_t incoming_e[kMaxSiteN], int n_incoming, int64_t site_m,
                                 int64_t site_e, out int64_t out_codes[kMaxSiteN],
                                 out int64_t out_scale_m, out int64_t out_scale_e,
                                 out int64_t status_tag)
{
    out_scale_m = 0;
    out_scale_e = 0;
    [unroll]
    for (int z = 0; z < kMaxSiteN; ++z) out_codes[z] = 0;

    bool rejected = false;
    int64_t tag = kTagOk;

    // Step 0.
    for (int i0 = 0; i0 < n_incoming && !rejected; ++i0)
    {
        if (!CarriedScaleMantissaFitsInt32Gpu(incoming_m[i0]))
        {
            tag = kTagCarriedScaleMantissaOutOfDomain;
            rejected = true;
        }
    }
    if (!rejected && !CarriedScaleMantissaFitsInt32Gpu(site_m))
    {
        tag = kTagCarriedScaleMantissaOutOfDomain;
        rejected = true;
    }

    // Steps 1-2: MaxAbsReduceWide (single-thread, guard-tier shape -- this
    // checkpoint's own real sites never exceed kMaxSiteN elements).
    int64_t d_prime = 0;
    if (!rejected)
    {
        uint64_t d = 0;
        for (int i1 = 0; i1 < n; ++i1)
        {
            int64_t xi = wide[i1];
            uint64_t a = (xi < 0) ? (~(uint64_t)xi + 1ULL) : (uint64_t)xi;
            if (a > d) d = a;
        }
        if (d < 1ULL) d = 1ULL;
        d_prime = (d > 0x7FFFFFFFFFFFFFFFULL) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)d;
        if (d_prime > ((int64_t)1 << 31))
        {
            tag = kTagChainInputOutOfDomain;
            rejected = true;
        }
    }

    // Step 4: NormalizeScale -> DynamicScaleReciprocal.
    int64_t ns_dn = 0;
    int ns_s = 0;
    int64_t r = 0;
    if (!rejected)
    {
        NormalizeScaleGpu(d_prime, ns_dn, ns_s);
        r = DynamicScaleReciprocalSharedGpu(ns_dn);
    }

    // Step 5: left-associated fold, incoming[...] then site_constant then the
    // d_prime factor, checked after every fold step.
    int64_t run_m = 0, run_e = 0;
    if (!rejected)
    {
        bool have_running = false;
        for (int i2 = 0; i2 < n_incoming && !rejected; ++i2)
        {
            if (have_running)
            {
                int64_t new_m, new_e;
                CombineCarriedScaleGpu_(run_m, run_e, incoming_m[i2], incoming_e[i2], new_m, new_e);
                run_m = new_m; run_e = new_e;
            }
            else
            {
                run_m = incoming_m[i2]; run_e = incoming_e[i2]; have_running = true;
            }
            if (!CarriedScaleMantissaFitsInt32Gpu(run_m)) { tag = kTagCarriedScaleMantissaOutOfDomain; rejected = true; }
        }
        if (!rejected)
        {
            if (have_running)
            {
                int64_t new_m, new_e;
                CombineCarriedScaleGpu_(run_m, run_e, site_m, site_e, new_m, new_e);
                run_m = new_m; run_e = new_e;
            }
            else
            {
                run_m = site_m; run_e = site_e; have_running = true;
            }
            if (!CarriedScaleMantissaFitsInt32Gpu(run_m)) { tag = kTagCarriedScaleMantissaOutOfDomain; rejected = true; }
        }
        if (!rejected)
        {
            int64_t dpf_m = ns_dn;
            int64_t dpf_e = -(int64_t)ns_s;
            int64_t new_m, new_e;
            CombineCarriedScaleGpu_(run_m, run_e, dpf_m, dpf_e, new_m, new_e);
            run_m = new_m; run_e = new_e;
            if (!CarriedScaleMantissaFitsInt32Gpu(run_m)) { tag = kTagCarriedScaleMantissaOutOfDomain; rejected = true; }
        }
    }

    if (rejected)
    {
        status_tag = tag;
        return;
    }

    // Step 6.
    for (int i3 = 0; i3 < n; ++i3)
    {
        out_codes[i3] = RequantTokenCodeWideSharedGpu(wide[i3], r, ns_s);
    }
    out_scale_m = run_m;
    out_scale_e = run_e;
    status_tag = kTagOk;
}

// checked_chain_funnel.cpp BiasReconcileWide (bit-exact copy of
// bias_wide.hlsl's own already-GPU-verified body, factored for inline reuse
// by ApplyBiasReconcileRowGpu below rather than a separate dispatch/readback
// round trip).
bool BiasReconcileWideSharedGpu(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, out int64_t out_val)
{
    int64_t exponent = 0;
    if (!RoundingDivideByPotComposedExponentInDomainGpu(q_b, e_a, exponent))
    {
        out_val = 0;
        return false;
    }
    S128 wide = SMul(b, r_a);
    S128 rounded = RoundingDivideByPOTWideGpu(wide, (int)exponent);
    out_val = SLow64(rounded);
    return SFitsI64(rounded);
}

// checked_chain_funnel.cpp CheckBiasAccumulateMagnitudeDomain.
bool CheckBiasAccumulateMagnitudeDomainGpu_(int64_t acc_i, int64_t b, int64_t q_b, int64_t r_a,
                                             int64_t e_a, out int64_t out_term)
{
    int64_t term = 0;
    if (!BiasReconcileWideSharedGpu(b, q_b, r_a, e_a, term))
    {
        out_term = 0;
        return false;
    }
    out_term = term;
    uint64_t ua = (uint64_t)acc_i;
    uint64_t ub = (uint64_t)term;
    uint64_t sum = ua + ub;
    bool same_sign_operands = ((ua ^ ub) >> 63) == 0ULL;
    bool sum_sign_differs = ((ua ^ sum) >> 63) != 0ULL;
    if (same_sign_operands && sum_sign_differs) return false;
    return true;
}

static const int64_t kBiasQFormatGpu = 30;

// forward_sites.cpp ApplyBiasReconcileRow, in full: the composed-exponent
// gate (shared with q_proj's own domain family), then a per-channel
// check-all-then-commit-all sweep (CPU's own `:997-1001` then `:1006-1007`
// two-loop shape, matched exactly rather than folded into one pass, so a
// rejection never applies a partial row -- forward_sites.cpp:991-1012).
bool ApplyBiasReconcileRowGpu(inout int64_t acc[kMaxSiteN], int out_channels, int64_t bias[kMaxSiteN],
                               int64_t in_scale_m, int64_t in_scale_e, out int64_t status_tag)
{
    int64_t unused_exp = 0;
    if (!RoundingDivideByPotComposedExponentInDomainGpu(kBiasQFormatGpu, in_scale_e, unused_exp))
    {
        status_tag = kTagRoundingDivideByPotExponentOutOfDomain;
        return false;
    }
    int64_t r_a = DynamicScaleReciprocalSharedGpu(in_scale_m);
    bool any_out_of_domain = false;
    for (int i = 0; i < out_channels; ++i)
    {
        int64_t term_unused = 0;
        if (!CheckBiasAccumulateMagnitudeDomainGpu_(acc[i], bias[i], kBiasQFormatGpu, r_a, in_scale_e,
                                                      term_unused))
        {
            any_out_of_domain = true;
        }
    }
    if (any_out_of_domain)
    {
        status_tag = kTagBiasReconcileProductOutOfDomain;
        return false;
    }
    for (int j = 0; j < out_channels; ++j)
    {
        int64_t term = 0;
        BiasReconcileWideSharedGpu(bias[j], kBiasQFormatGpu, r_a, in_scale_e, term);
        acc[j] += term;
    }
    status_tag = kTagOk;
    return true;
}

// forward_sites.cpp ClampRopeCode -- the pinned CODE range [-127,127], NOT
// int8's storage range [-128,127].
int64_t ClampRopeCodeGpu(int64_t raw)
{
    if (raw > 127) return 127;
    if (raw < -127) return -127;
    return raw;
}

// checked_chain_funnel.cpp CheckPositionOverCap.
bool CheckPositionOverCapGpu(int64_t position, int64_t context_cap)
{
    return !(position < 0 || position >= context_cap);
}

// forward_sites.cpp's S3.7 KV addressing (KvHalfOffset / KvRowOffsetWithinHalf),
// bit-exact -- so KeyRowGpu/ValueRowGpu (superslm_gpu.cpp, CPU-side pointer
// arithmetic reading the same `workspace`) address the identical byte this
// dispatch writes.
uint KvHalfOffsetGpu(uint layer, uint context_cap, uint num_kv_heads, uint head_dim)
{
    return (layer * context_cap * num_kv_heads * head_dim) * 2u;
}
uint KvRowOffsetWithinHalfGpu(uint context_cap, uint head_dim, uint kv_head, uint position)
{
    return kv_head * context_cap * head_dim + position * head_dim;
}

#endif  // SSLM_SITE_COMMON_HLSLI
