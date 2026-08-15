// T-2039 (GPU-serial port: real-capacity shader geometry). Shared primitives
// for the composed pipeline's site shaders, rebuilt to the design's own
// production dispatch geometry (Claude/Vitruvius/t1986-gpu-serial-port-
// design-2026-08-13.md Sec5.4/Sec5.5): 256-thread groups, per-output-channel
// parallelization with N-per-thread striding (non-uniform where a width does
// not divide 256 evenly -- each thread simply iterates `for (i = t; i < K; i
// += 256)`, which naturally gives threads 0..(K mod 256 - 1) one more
// iteration than the rest), and the already-proven SCHEME 0/1 groupshared
// tree reduction (B5, reduce_max_abs_i64_scheme1.hlsl) for every cross-thread
// combine this design needs. Every wide row this file's own primitives
// operate on lives in a caller-supplied RWByteAddressBuffer window (WorkScratch,
// superslm_gpu.cpp's own per-call, per-real-dims allocation) -- NEVER a
// compile-time-capacity local array, so nothing here is sized to any one
// tier: the T-2019 suite's own hidden_size=2 fixture and the real 1.5B/0.5B
// artifacts' hidden_size=1536/896, intermediate_size=8960/4864 all run the
// identical shader, driven entirely by the real per-call root constants and
// the ScratchLayout/Layout SRVs superslm_gpu.cpp uploads.
//
// Every function here has a named C++ sibling; see src/forward/
// forward_sites.cpp, src/intmath.cpp, and src/forward/checked_chain_funnel.cpp.
// B1/B2's own funnel_guards.hlsli is reused rather than duplicated
// (CombineCarriedScale, NormalizeScale, CarriedScaleMantissaFitsInt32, the
// composed-exponent domain check, SaturatingRoundingDoublingHighMul).
#ifndef SSLM_SITE_COMMON_HLSLI
#define SSLM_SITE_COMMON_HLSLI

#include "wide128.hlsli"
#include "funnel_guards.hlsli"
#include "landing_rescale.hlsli"

// The funnel's own fold arity -- how many CarriedScale terms combine at any
// one site (observed maximum across every site this design specifies: 2,
// mlp_act's gate_scale+up_scale) -- a property of the ALGORITHM, not of any
// model tier; this headroom is never indexed by hidden_size/intermediate_size/
// head_dim/context_cap and so is not a "fixed-capacity array sized to a
// tier" in the sense this ticket's commission forbids.
static const int kMaxIncoming = 4;

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

// intmath.cpp SaturatingLeftShift32 -- T-2113 (B6b): the amplification-side sibling
// RoundingDivideByPotI32Gpu never needed (WSC1's own shift is always non-negative);
// ApplyAmplifyingWeightScaleFoldGpu's own exponent<0 branch below is the first GPU
// caller.
int SaturatingLeftShift32Gpu(int x, int shift)
{
    if (shift <= 0) return x;
    int64_t wide = ((int64_t)x) << shift;
    if (wide > (int64_t)2147483647) return 2147483647;
    if (wide < -(int64_t)2147483647 - 1) return -2147483647 - 1;
    return (int)wide;
}

// forward_sites.cpp ApplyAmplifyingWeightScaleFold -- T-2113 (B6b, design Sec8): the
// GEMM-site adapter delta-application dispatch's own fold primitive, for both the
// u-fold (rank-wide) and delta-fold (out_channels-wide) stages of
// AddAmplifyingLoraDelta. identity==1 is the exact rho==1 pass-through; exponent>=0 is
// bit-identical to ApplyWeightScaleFoldGpu's own non-identity branch (the SAME two
// gemmlowp-style primitives, called exactly the same way); only exponent<0
// (amplification) is new arithmetic, via SaturatingLeftShift32Gpu above.
int64_t ApplyAmplifyingWeightScaleFoldGpu(int64_t acc, int identity, int mult, int exponent)
{
    if (identity != 0) return acc;
    int hi = SaturatingRoundingDoublingHighMulGpu_((int)acc, mult);
    if (exponent >= 0) return (int64_t)RoundingDivideByPotI32Gpu(hi, exponent);
    return (int64_t)SaturatingLeftShift32Gpu(hi, -exponent);
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

// T-2039 correction: single-byte store into a ByteAddressBuffer, the write-
// side sibling of LoadSignedByteGpu above -- used for the K/V cache's own
// int8-per-element layout (KeyRow/ValueRow's own addressing, matched exactly
// so the CPU-side accessors read the identical bytes this dispatch writes).
// A plain Load-modify-Store (T-2032's own original body) is NOT safe under
// this design's own cooperative geometry: two threads owning ADJACENT bytes
// in the SAME 4-byte-aligned word (e.g. kv_proj's own (kv_head, dim) landing
// pairs, or RoPE's own per-query-head K write-back under a GQA group > 1)
// can both Load the same pre-image, each apply their own byte, and Store --
// the second Store silently discards the first thread's byte, a lost-update
// race a global UAV barrier (Sec5.6) does NOT prevent, since that barrier
// serializes DISPATCHES, not the 256 threads cooperating WITHIN one.
// Confirmed by execution this ticket's own build: kv_proj's landing pass
// wrote channel 0 correctly and silently lost channel 1 (the two landing
// bytes share a word at this suite's own head_dim=2), a real-hardware defect
// only a genuinely atomic byte update closes. Realized as an
// InterlockedCompareExchange retry loop -- the standard byte-in-a-word CAS
// idiom, since D3D12 has no native sub-32-bit atomic.
void StoreSignedByteGpu(RWByteAddressBuffer buf, uint byteOff, int val)
{
    uint wordOff = (byteOff / 4u) * 4u;
    uint shift = (byteOff % 4u) * 8u;
    uint mask = ~(0xFFu << shift);
    uint newByte = ((uint)val & 0xFFu) << shift;
    uint original;
    buf.InterlockedAdd(wordOff, 0u, original);
    [allow_uav_condition]
    for (;;)
    {
        uint newWord = (original & mask) | newByte;
        uint prev;
        buf.InterlockedCompareExchange(wordOff, original, newWord, prev);
        if (prev == original) break;
        original = prev;
    }
}

// intmath.cpp DynamicScaleReciprocal -- 3 Newton iterations + 2 branch-free
// correction steps (bit-exact copy of dyn_recip.hlsl's own already-GPU-
// verified DynamicScaleReciprocalGpuImpl, factored here for reuse).
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

// checked_chain_funnel.cpp BiasReconcileWide (bit-exact copy of
// bias_wide.hlsl's own already-GPU-verified body, factored for inline reuse).
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

// T-2113 (B10 lever 1, design Sec10 "B10 -- Lever Detail" #1): the fused-dispatch form of
// AddAmplifyingLoraDelta -- bit-exact to adapter_delta_site.hlsl's own two-stage body (stage 1:
// u_acc[k] over in_channels, folded+narrowed into WorkScratch's ADAPTER_U region; stage 2:
// delta_raw[j] over rank, folded and added into the base GEMM's own WorkScratch accumulator
// row), called from INSIDE each covered tail shader's own dispatch instead of from a dispatch
// of its own. `rank == 0` is the "no adapter covers this (layer, projection) slot" sentinel --
// a real LoRA adapter's own rank is always >= 1 -- and is a true no-op: every thread in the
// group evaluates the same root-constant-derived `rank`, so the early return is uniform control
// flow, not per-thread divergence, and issues neither barrier when taken (matching the
// standalone dispatch's own "never issued at all" contract for an uncovered slot exactly).
// Callers invoke this AFTER the sticky-word check and STRICTLY BEFORE any bias-reconcile/requant
// code in the same dispatch reads WorkScratch at `wide_base` -- the identical insertion point
// the standalone dispatch occupied between the base GEMM's own WSC1 fold and the site's own
// bias/requant dispatch, now collapsed into the SAME dispatch. The trailing
// DeviceMemoryBarrierWithGroupSync() (only issued when rank > 0) publishes stage 2's WorkScratch
// write to the whole group before the caller's own subsequent read, and also protects a SECOND
// call to this function later in the SAME dispatch (kv_proj_site.hlsl's own K-then-V fusion) --
// the shared ADAPTER_U scratch region is reused serially across the two calls, never
// concurrently, exactly as it already is across separate dispatches on the standalone path.
// T-2113 (B10 lever 1b): stage 1's own groupshared accumulator, one int64 slot per thread --
// the identical shape `gGemmAcc` (above) uses for the base GEMM's own coalesced reduction,
// declared separately so the two never alias across a dispatch that (in principle) could use
// both. Unused, and therefore free, in any compiled shader that never calls this function.
groupshared int64_t gAdapterStage1Acc[256];

void ApplyFusedAdapterDeltaGpu(uint t, RWByteAddressBuffer WorkScratch, RWByteAddressBuffer LayerScratch,
                                ByteAddressBuffer LoraAB, ByteAddressBuffer Fold, int in_channels,
                                int out_channels, int rank, uint a_offset, uint b_offset, uint fold_offset,
                                uint adapter_u_off, uint in_base, uint wide_base, int stage1_lanes)
{
    if (rank == 0) return;  // no adapter covers this slot -- the standalone path's own "dispatch
                             // never issued" contract, realized here as a true no-op instead.

    uint dfs_base = fold_offset;                              // out_channels rows, 12 bytes each
    uint ufs_base = fold_offset + (uint)out_channels * 12u;    // rank rows, 12 bytes each

    // Stage 1 -- T-2113 (B10 lever 1b, replacing lever 1's own one-thread-per-k form, which left
    // 256-rank of the group's 256 threads idle while the active `rank` threads each ran a full
    // in_channels-wide SERIAL reduction alone -- named as the likely dominant adapter-bound cost
    // driver at §16.5 (Claude/Brunel/t2113-1p0-core-build-2026-08-15.md) and confirmed by this
    // section's own per-site execution decomposition (down_proj's stage-1, in_channels=8960,
    // measured as the single largest adapter-bound delta of any site). `stage1_lanes` threads
    // now cooperate on EACH k, via the identical fixed groupshared binary tree
    // `GemmCoalescedGpuAt` (above) already uses for the base GEMM's own coalesced reduction --
    // integer addition is exactly associative and commutative (that function's own header
    // comment, D-SLM3334), so this returns the identical int64 the old one-thread serial loop
    // returned, for any rank and in_channels this adapter format admits. `stage1_lanes` is a
    // host-computed power of two (`stage1_lanes_for_rank`, superslm_gpu.cpp) sized so
    // `256 / stage1_lanes` k-values are covered per wave; the outer `k0` loop below covers the
    // (never expected in practice, but not excluded) case `rank > 256 / stage1_lanes` by running
    // additional waves, each barrier-separated from the last so `gAdapterStage1Acc` is never
    // read by one wave before the previous wave's own writers have retired it.
    const uint lane1 = t % (uint)stage1_lanes;
    const int channels_per_wave = 256 / stage1_lanes;
    for (int k0 = 0; k0 < rank; k0 += channels_per_wave)
    {
        const int k = k0 + (int)(t / (uint)stage1_lanes);
        int64_t acc = 0;
        if (k < rank)
        {
            for (int i = (int)lane1; i < in_channels; i += stage1_lanes)
            {
                int a = LayerScratch.Load<int>(in_base + (uint)i * 4u);
                int w = LoadSignedByteGpu(LoraAB, a_offset + (uint)(k * in_channels + i));
                acc += (int64_t)a * (int64_t)w;
            }
        }
        gAdapterStage1Acc[t] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = (uint)stage1_lanes >> 1; s > 0u; s >>= 1)
        {
            if (lane1 < s) gAdapterStage1Acc[t] += gAdapterStage1Acc[t + s];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane1 == 0u && k < rank)
        {
            int64_t total = gAdapterStage1Acc[t];
            int u_identity = (int)Fold.Load<int>(ufs_base + (uint)k * 12u + 0u);
            int u_mult     = (int)Fold.Load<int>(ufs_base + (uint)k * 12u + 4u);
            int u_exponent = (int)Fold.Load<int>(ufs_base + (uint)k * 12u + 8u);
            int64_t u_wide = ApplyAmplifyingWeightScaleFoldGpu(total, u_identity, u_mult, u_exponent);
            int64_t u_clamped = ClampRopeCodeGpu(u_wide);
            StoreSignedByteGpu(WorkScratch, adapter_u_off + (uint)k, (int)u_clamped);
        }
        GroupMemoryBarrierWithGroupSync();  // gAdapterStage1Acc is safe for the next wave (or,
                                             // on the last wave, this is redundant with the
                                             // DeviceMemoryBarrierWithGroupSync() below -- kept
                                             // unconditional so every thread's own control flow
                                             // is uniform across every iteration of this loop).
    }
    DeviceMemoryBarrierWithGroupSync();  // publish u_i8 (a UAV write) before stage 2 reads it back

    // Stage 2 -- one thread per j < out_channels, reduced over the (small) rank dimension,
    // added directly into the base GEMM's own WorkScratch accumulator row.
    for (int j = (int)t; j < out_channels; j += 256)
    {
        int64_t acc2 = 0;
        for (int k2 = 0; k2 < rank; ++k2)
        {
            int u = LoadSignedByteGpu(WorkScratch, adapter_u_off + (uint)k2);
            int w2 = LoadSignedByteGpu(LoraAB, b_offset + (uint)(j * rank + k2));
            acc2 += (int64_t)u * (int64_t)w2;
        }
        int d_identity = (int)Fold.Load<int>(dfs_base + (uint)j * 12u + 0u);
        int d_mult     = (int)Fold.Load<int>(dfs_base + (uint)j * 12u + 4u);
        int d_exponent = (int)Fold.Load<int>(dfs_base + (uint)j * 12u + 8u);
        int64_t delta_wide = ApplyAmplifyingWeightScaleFoldGpu(acc2, d_identity, d_mult, d_exponent);
        int64_t base_val = WorkScratch.Load<int64_t>(wide_base + (uint)j * 8u);
        WorkScratch.Store<int64_t>(wide_base + (uint)j * 8u, base_val + delta_wide);
    }
    DeviceMemoryBarrierWithGroupSync();  // publish the accumulator write before the caller's own
                                          // bias-reconcile/requant code reads it, and before any
                                          // second call to this function in the same dispatch
                                          // (kv_proj_site.hlsl's own K-then-V fusion) reuses
                                          // adapter_u_off.
}

// T-2039: SeqState's own per-call, per-real-hidden_size byte layout.
// hidden_codes lives first (Align8(hidden_size*4) bytes, one int32 per code
// -- T-2035's own fixed 8-slot/32-byte assumption is exactly the same UB
// class this ticket's commission names for LayerScratch, and SeqState has
// the identical defect), then hidden_scale (16 bytes), layer_index (8, the
// extra 4 kept as alignment padding rather than packed tighter -- this is a
// trivial, single-field-order layout, not a multi-field packing table like
// GpuLayerLayout/GpuScratchLayout, so it is safe and simpler for both the
// host packer (superslm_gpu.cpp) and every shader to compute independently
// from the SAME real g_hidden_size root constant rather than round-trip
// through another SRV table -- the formula has no room to drift because it
// is one line, unlike LayerWeights' 56-field layout.
uint SeqScaleOffGpu(int hidden_size) { return (uint)(((hidden_size * 4) + 7) & ~7); }
uint SeqLayerIdxOffGpu(int hidden_size) { return SeqScaleOffGpu(hidden_size) + 16u; }
uint SeqSatLoOffGpu(int hidden_size) { return SeqLayerIdxOffGpu(hidden_size) + 8u; }
uint SeqSatHiOffGpu(int hidden_size) { return SeqSatLoOffGpu(hidden_size) + 4u; }
uint SeqCtxLenOffGpu(int hidden_size) { return SeqSatHiOffGpu(hidden_size) + 4u; }
uint SeqStickyOffGpu(int hidden_size) { return SeqCtxLenOffGpu(hidden_size) + 8u; }

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

// ===========================================================================
// T-2039: the cooperative (256-thread) production primitives. Every one takes
// `t` (SV_GroupThreadID.x) and operates on caller-supplied RWByteAddressBuffer
// windows -- WorkScratch (superslm_gpu.cpp's own per-call, per-real-dims
// scratch UAV) for a transient wide row, LayerScratch/SeqState for the
// persistent per-site codes+scale slots ScratchLayout names. No local array
// here is ever indexed by hidden_size/intermediate_size/head_dim/context_cap;
// the only per-thread storage is scalars, and the only cross-thread
// coordination is the already-proven SCHEME-1 groupshared tree (B5).
// ===========================================================================

groupshared uint64_t gFunnelMax[256];

// Cooperative parallel GEMM: `out_channels` output channels spread across
// `t`'s own domain via N-per-thread striding at the caller-supplied `stride`
// (non-uniform where out_channels does not divide the stride evenly --
// threads whose `t < out_channels mod stride` simply get one more loop
// iteration than the rest, the T-2014 correction's own split). Each output
// channel's own full `in_channels`-wide int64 dot product
// (GemmInt8AccumulateRow's own accumulation order, proven bit-identical to
// the SSE2 production path by associativity, design Sec4) is computed by
// exactly one thread, in full -- no cross-thread accumulation, so no
// reduction and no barrier is needed here; the folded int64 result lands in
// `scratch` at `wide_base + j*8`, one thread's own write, never re-read by
// another thread in the SAME phase.
//
// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313):
// `stride` used to be the literal 256 unconditionally -- every SINGLE-GROUP
// caller below still passes 256 (`t` = `SV_GroupThreadID.x`, unchanged
// behavior, bit-for-bit). The NEW multi-group GEMM-only dispatches
// (`<site>_gemm_site.hlsl`) pass `t` = `SV_DispatchThreadID.x` (globally
// unique across every group the dispatch issues) and `stride` = the total
// thread count the dispatch launched (256 * group count, sized so
// `stride >= out_channels`) -- which output channel(s) a physical thread
// computes changes; HOW each channel is computed (same weight row, same
// input codes, same left-to-right reduction over `in_channels`, same
// ApplyWeightScaleFoldGpu call) does not, so which of the two calling shapes
// wrote the wide row is not observable in the row's own contents.
void GemmParallelGpu(uint t, RWByteAddressBuffer in_buf, uint in_base, ByteAddressBuffer w_buf,
                      uint w_base, ByteAddressBuffer id_buf, uint id_base, ByteAddressBuffer mult_buf,
                      uint mult_base, ByteAddressBuffer shift_buf, uint shift_base, int in_channels,
                      int out_channels, RWByteAddressBuffer scratch, uint wide_base, int stride)
{
    for (int j = (int)t; j < out_channels; j += stride)
    {
        int64_t acc = 0;
        for (int i = 0; i < in_channels; ++i)
        {
            int a = in_buf.Load<int>(in_base + (uint)i * 4u);
            int w = LoadSignedByteGpu(w_buf, w_base + (uint)(j * in_channels + i));
            acc += (int64_t)a * (int64_t)w;
        }
        int identity = (int)id_buf.Load<int>(id_base + (uint)j * 4u);
        int mult = (int)mult_buf.Load<int>(mult_base + (uint)j * 4u);
        int shift = (int)shift_buf.Load<int>(shift_base + (uint)j * 4u);
        int64_t folded = ApplyWeightScaleFoldGpu(acc, identity, mult, shift);
        scratch.Store<int64_t>(wide_base + (uint)j * 8u, folded);
    }
}

// T-2113 (B4, design Sec3/Sec6.1; re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 3/4, T-2106 debunk's confirmed endpoint): the transposed GEMM
// partition. `GemmParallelGpu` above gives ONE thread an entire output channel, so a
// thread's own weight reads walk `w_base + j*in_channels + i` with `i` increasing --
// contiguous WITHIN a thread, but adjacent THREADS are `in_channels` bytes apart (1536 or
// 8960 at the real 1.5B tier). A warp's 32 loads therefore spread across 32*in_channels
// bytes, one memory transaction each, almost every fetched byte discarded.
//
// This transposes the partition: `lanes` threads cooperate on ONE output channel, thread
// `lane` taking `i = lane, lane+lanes, ...` -- adjacent threads now read adjacent bytes, so
// a warp's 32 reads fall inside one or two cache lines. The per-lane partial sums combine
// via a fixed groupshared binary tree over INTEGER addition -- the identical reduction
// shape `RmsSumSqParallelGpu` already ships in this file, whose own header comment states
// the determinism law this construction relies on: integer addition is exactly associative
// and commutative, so a tree reduction returns the identical int64 the sequential
// left-to-right loop returns, regardless of evaluation order. No float, no atomic appears
// anywhere on this path -- the reduction order is fixed and identical on every device and
// every run, which is exactly what T-2105's own per-step CPU-oracle bit-equality (reproduced
// by this section's own gate) verifies rather than merely argues.
//
// Geometry: numthreads is fixed at 256; `lanes` (a power of two, 1..256) splits those 256
// threads into `channels_per_group = 256/lanes` output channels per group, `lanes` threads
// each. A thread whose resolved channel `j` runs past `out_channels` (or is negative)
// contributes zero and still executes every barrier below -- no divergent early return
// around a GroupMemoryBarrierWithGroupSync, which is undefined behavior.
groupshared int64_t gGemmAcc[256];

// The caller-supplies-the-channel form: `j` is this thread's own output channel, already
// resolved by the caller (a plain group_id*channels_per_group + local mapping for most
// sites; `kv_proj_gemm_site.hlsl` packs K's and V's channels into one dispatch and needs
// its own mapping). Every thread of the group must call this with the SAME lane/lanes
// derivation and must reach it unconditionally -- it contains group-wide barriers.
void GemmCoalescedGpuAt(uint t, int j, RWByteAddressBuffer in_buf, uint in_base,
                         ByteAddressBuffer w_buf, uint w_base, ByteAddressBuffer id_buf, uint id_base,
                         ByteAddressBuffer mult_buf, uint mult_base, ByteAddressBuffer shift_buf,
                         uint shift_base, int in_channels, int out_channels,
                         RWByteAddressBuffer scratch, uint wide_base, uint lanes)
{
    const uint lane = t % lanes;
    int64_t acc = 0;
    if (j < out_channels && j >= 0)
    {
        const uint row = w_base + (uint)j * (uint)in_channels;
        // Four weights + four input codes per load, taken only when the row is
        // dword-aligned and in_channels divides by four -- one packed dword per lane gives
        // a 32-lane warp 128 CONTIGUOUS bytes of weight (one cache line) with no duplicate
        // fetch. Otherwise the byte path below runs unchanged, so no tier or layout is
        // required to satisfy the fast path's own preconditions. The summation is
        // REASSOCIATED, not changed -- the same set of int64 products summed in a
        // different order, which is bit-identical for the same associativity/commutativity
        // reason the tree reduction below is.
        if ((row & 3u) == 0u && (in_channels & 3) == 0)
        {
            const int quads = in_channels >> 2;
            for (int q = (int)lane; q < quads; q += (int)lanes)
            {
                uint wp = w_buf.Load(row + (uint)q * 4u);
                uint4 a4 = in_buf.Load4(in_base + (uint)q * 16u);
                acc += (int64_t)(int)a4.x * (int64_t)(int(wp << 24) >> 24);
                acc += (int64_t)(int)a4.y * (int64_t)(int(wp << 16) >> 24);
                acc += (int64_t)(int)a4.z * (int64_t)(int(wp << 8) >> 24);
                acc += (int64_t)(int)a4.w * (int64_t)(int(wp) >> 24);
            }
        }
        else
        {
            for (int i = (int)lane; i < in_channels; i += (int)lanes)
            {
                int a = in_buf.Load<int>(in_base + (uint)i * 4u);
                int w = LoadSignedByteGpu(w_buf, row + (uint)i);
                acc += (int64_t)a * (int64_t)w;
            }
        }
    }
    gGemmAcc[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = lanes >> 1; s > 0u; s >>= 1)
    {
        if (lane < s) gGemmAcc[t] += gGemmAcc[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && j < out_channels && j >= 0)
    {
        int64_t total = gGemmAcc[t];
        int identity = (int)id_buf.Load<int>(id_base + (uint)j * 4u);
        int mult = (int)mult_buf.Load<int>(mult_base + (uint)j * 4u);
        int shift = (int)shift_buf.Load<int>(shift_base + (uint)j * 4u);
        int64_t folded = ApplyWeightScaleFoldGpu(total, identity, mult, shift);
        scratch.Store<int64_t>(wide_base + (uint)j * 8u, folded);
    }
}

// The plain mapping: group `group_id` owns channels [group_id*C, group_id*C + C) where
// C = 256/lanes. One body, shared with the caller-supplies-the-channel form above, so the
// two cannot drift apart.
void GemmCoalescedGpu(uint t, uint group_id, RWByteAddressBuffer in_buf, uint in_base,
                       ByteAddressBuffer w_buf, uint w_base, ByteAddressBuffer id_buf, uint id_base,
                       ByteAddressBuffer mult_buf, uint mult_base, ByteAddressBuffer shift_buf,
                       uint shift_base, int in_channels, int out_channels, RWByteAddressBuffer scratch,
                       uint wide_base, uint lanes)
{
    const uint channels_per_group = 256u / lanes;
    const int j = (int)(group_id * channels_per_group + t / lanes);
    GemmCoalescedGpuAt(t, j, in_buf, in_base, w_buf, w_base, id_buf, id_base, mult_buf, mult_base,
                        shift_buf, shift_base, in_channels, out_channels, scratch, wide_base, lanes);
}

groupshared uint gBiasAnyOutOfDomain;

// Cooperative ApplyBiasReconcileRow: the accumulator row already lives in
// `scratch` at `wide_base` (out_channels int64 elements, thread t owns
// elements t, t+256, ...). Two-phase, matching CPU's own check-all-then-
// commit-all shape exactly (forward_sites.cpp:991-1012's own two-loop
// construction, so a rejection never applies a partial row): phase A, every
// thread checks its own owned channels and ORs a shared failure flag; phase
// B (only reached if the whole group agrees Ok) commits every owned channel.
bool ApplyBiasReconcileRowGpuP(uint t, RWByteAddressBuffer scratch, uint wide_base, int out_channels,
                                ByteAddressBuffer bias_buf, uint bias_base, int64_t in_scale_m,
                                int64_t in_scale_e, out int64_t status_tag)
{
    int64_t unused_exp = 0;
    if (!RoundingDivideByPotComposedExponentInDomainGpu(kBiasQFormatGpu, in_scale_e, unused_exp))
    {
        status_tag = kTagRoundingDivideByPotExponentOutOfDomain;
        return false;
    }
    int64_t r_a = DynamicScaleReciprocalSharedGpu(in_scale_m);

    if (t == 0) gBiasAnyOutOfDomain = 0;
    GroupMemoryBarrierWithGroupSync();
    for (int i = (int)t; i < out_channels; i += 256)
    {
        int64_t acc_i = scratch.Load<int64_t>(wide_base + (uint)i * 8u);
        int64_t bias_i = bias_buf.Load<int64_t>(bias_base + (uint)i * 8u);
        int64_t term_unused = 0;
        if (!CheckBiasAccumulateMagnitudeDomainGpu_(acc_i, bias_i, kBiasQFormatGpu, r_a, in_scale_e,
                                                      term_unused))
        {
            InterlockedOr(gBiasAnyOutOfDomain, 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gBiasAnyOutOfDomain != 0)
    {
        status_tag = kTagBiasReconcileProductOutOfDomain;
        return false;
    }
    for (int j = (int)t; j < out_channels; j += 256)
    {
        int64_t acc_j = scratch.Load<int64_t>(wide_base + (uint)j * 8u);
        int64_t bias_j = bias_buf.Load<int64_t>(bias_base + (uint)j * 8u);
        int64_t term = 0;
        BiasReconcileWideSharedGpu(bias_j, kBiasQFormatGpu, r_a, in_scale_e, term);
        scratch.Store<int64_t>(wide_base + (uint)j * 8u, acc_j + term);
    }
    status_tag = kTagOk;
    return true;
}

// Cooperative checked_chain_funnel.cpp RequantChainChecked, IN FULL (steps
// 0-6): the wide row lives in `scratch` at `wide_base` (n int64 elements,
// thread t owns elements t, t+256, ...) rather than a per-call local array.
// Steps 1-2 (MaxAbsReduceWide) are the design's own SCHEME-1 groupshared
// binary tree (B5, reduce_max_abs_i64_scheme1.hlsl -- body unchanged, only
// the per-thread partial-max source changed from an input buffer row to this
// site's own wide-row window). Steps 3-5 are pure scalar derivations from
// values already uniform across the group after the reduction's own barrier
// (d_prime, then r/ns_s, then the incoming-fold), so every thread simply
// repeats the identical scalar computation -- cheaper than a broadcast and
// bit-identical by construction, since every thread reads the identical
// groupshared result. Step 6 writes the n output codes cooperatively (same
// striding, same per-thread ownership as the read in steps 1-2 -- each
// thread only ever re-reads its own prior write, no cross-thread dependency,
// no additional barrier needed for that data flow) directly into `out_buf`
// at `out_codes_base`; the scale pair is written once, by thread 0, to
// `out_buf` at `out_scale_base`.
void RequantChainCheckedFullGpuP(uint t, RWByteAddressBuffer scratch, uint wide_base, int n,
                                  int64_t incoming_m[kMaxIncoming], int64_t incoming_e[kMaxIncoming],
                                  int n_incoming, int64_t site_m, int64_t site_e,
                                  RWByteAddressBuffer out_buf, uint out_codes_base, uint out_scale_base,
                                  out int64_t status_tag)
{
    bool rejected = false;
    int64_t tag = kTagOk;

    // Step 0.
    for (int i0 = 0; i0 < n_incoming; ++i0)
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

    // Steps 1-2: cooperative MaxAbsReduceWide over the buffer-resident row
    // (B5's own SCHEME-1 tree, unchanged body).
    int64_t d_prime = 0;
    if (!rejected)
    {
        uint64_t local = 0;
        for (int i1 = (int)t; i1 < n; i1 += 256)
        {
            int64_t xi = scratch.Load<int64_t>(wide_base + (uint)i1 * 8u);
            uint64_t a = (xi < 0) ? (~(uint64_t)xi + 1ULL) : (uint64_t)xi;
            if (a > local) local = a;
        }
        gFunnelMax[t] = local;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 128; s > 0; s >>= 1)
        {
            if (t < s) gFunnelMax[t] = max(gFunnelMax[t], gFunnelMax[t + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        uint64_t d = gFunnelMax[0];
        if (d < 1ULL) d = 1ULL;
        d_prime = (d > 0x7FFFFFFFFFFFFFFFULL) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)d;
        if (d_prime > ((int64_t)1 << 31))
        {
            tag = kTagChainInputOutOfDomain;
            rejected = true;
        }
    }

    // Step 4: NormalizeScale -> DynamicScaleReciprocal (uniform scalar work,
    // every thread repeats it identically once d_prime is known).
    int64_t ns_dn = 0;
    int ns_s = 0;
    int64_t r = 0;
    if (!rejected)
    {
        NormalizeScaleGpu(d_prime, ns_dn, ns_s);
        r = DynamicScaleReciprocalSharedGpu(ns_dn);
    }

    // Step 5: left-associated fold, incoming[...] then site_constant then the
    // d_prime factor, checked after every fold step (uniform scalar work).
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

    // Step 6: cooperative write-out -- same striding as steps 1-2, so every
    // thread only ever re-reads its own prior write.
    for (int i3 = (int)t; i3 < n; i3 += 256)
    {
        int64_t xi = scratch.Load<int64_t>(wide_base + (uint)i3 * 8u);
        int64_t code = RequantTokenCodeWideSharedGpu(xi, r, ns_s);
        out_buf.Store<int>(out_codes_base + (uint)i3 * 4u, (int)code);
    }
    if (t == 0)
    {
        out_buf.Store<int64_t>(out_scale_base + 0, run_m);
        out_buf.Store<int64_t>(out_scale_base + 8, run_e);
    }
    status_tag = kTagOk;
}

// T-2113 (B10 lever 3, D-SLM3401): a 1024-thread-wide copy of
// RequantChainCheckedFullGpuP above, dedicated to mlp_act_site.hlsl -- the one
// caller whose own row width (intermediate_size, 8960 at the real 1.5B tier)
// is wide enough that quadrupling thread count measurably cuts the per-thread
// serial iteration count (~35 -> ~9 per pass). Bit-exact to the 256-wide
// function: every step is unchanged (Step 0 domain checks, steps 1-2 the same
// SCHEME-1 groupshared binary-tree max-reduce over a striped read, steps 3-5
// the same uniform scalar derivation every thread repeats identically, step 6
// the same cooperative striped store) -- only the stride (1024, not 256), the
// tree's own starting halving width (512, not 128), and the dedicated
// `gFunnelMax1024` groupshared array (not the shared 256-wide `gFunnelMax`,
// which the other ten RequantChainCheckedFullGpuP callers still use unchanged)
// differ. Integer max is exactly associative and commutative regardless of
// tree width (the same argument D-SLM3334/D-SLM3399/D-SLM3400 already
// commissioned) -- a 1024-wide tree returns the IDENTICAL d_prime a 256-wide
// tree or a fully sequential scan would, for any row. This is a duplicate
// function, not a parameterized generalization of the shared one, because
// generalizing the shared function would touch all eleven of its callers
// (attn_norm/attn_residual/ctx_fold/mlp_norm/mlp_residual and all six
// _proj_site tail shaders) for a change only mlp_act needs.
groupshared uint64_t gFunnelMax1024[1024];

void RequantChainCheckedFullGpuP1024(uint t, RWByteAddressBuffer scratch, uint wide_base, int n,
                                      int64_t incoming_m[kMaxIncoming], int64_t incoming_e[kMaxIncoming],
                                      int n_incoming, int64_t site_m, int64_t site_e,
                                      RWByteAddressBuffer out_buf, uint out_codes_base, uint out_scale_base,
                                      out int64_t status_tag)
{
    bool rejected = false;
    int64_t tag = kTagOk;

    // Step 0.
    for (int i0 = 0; i0 < n_incoming; ++i0)
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

    // Steps 1-2: cooperative MaxAbsReduceWide, 1024-wide stride and tree.
    int64_t d_prime = 0;
    if (!rejected)
    {
        uint64_t local = 0;
        for (int i1 = (int)t; i1 < n; i1 += 1024)
        {
            int64_t xi = scratch.Load<int64_t>(wide_base + (uint)i1 * 8u);
            uint64_t a = (xi < 0) ? (~(uint64_t)xi + 1ULL) : (uint64_t)xi;
            if (a > local) local = a;
        }
        gFunnelMax1024[t] = local;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 512; s > 0; s >>= 1)
        {
            if (t < s) gFunnelMax1024[t] = max(gFunnelMax1024[t], gFunnelMax1024[t + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        uint64_t d = gFunnelMax1024[0];
        if (d < 1ULL) d = 1ULL;
        d_prime = (d > 0x7FFFFFFFFFFFFFFFULL) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)d;
        if (d_prime > ((int64_t)1 << 31))
        {
            tag = kTagChainInputOutOfDomain;
            rejected = true;
        }
    }

    // Step 4: NormalizeScale -> DynamicScaleReciprocal (uniform scalar work,
    // every thread repeats it identically once d_prime is known).
    int64_t ns_dn = 0;
    int ns_s = 0;
    int64_t r = 0;
    if (!rejected)
    {
        NormalizeScaleGpu(d_prime, ns_dn, ns_s);
        r = DynamicScaleReciprocalSharedGpu(ns_dn);
    }

    // Step 5: left-associated fold, incoming[...] then site_constant then the
    // d_prime factor, checked after every fold step (uniform scalar work).
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

    // Step 6: cooperative write-out -- same striding as steps 1-2, so every
    // thread only ever re-reads its own prior write.
    for (int i3 = (int)t; i3 < n; i3 += 1024)
    {
        int64_t xi = scratch.Load<int64_t>(wide_base + (uint)i3 * 8u);
        int64_t code = RequantTokenCodeWideSharedGpu(xi, r, ns_s);
        out_buf.Store<int>(out_codes_base + (uint)i3 * 4u, (int)code);
    }
    if (t == 0)
    {
        out_buf.Store<int64_t>(out_scale_base + 0, run_m);
        out_buf.Store<int64_t>(out_scale_base + 8, run_e);
    }
    status_tag = kTagOk;
}

// Cooperative RmsNormSite's own sumsq reduction (sumsq = sum(h[i]^2),
// forward_sites.cpp:410-446). Integer addition is exactly associative and
// commutative with no intermediate-overflow risk at this design's own int8-
// code domain (h[i] in [-127,127], h[i]^2 <= 16129, hidden_size <=
// intermediate_size's own real ceiling ~9000 -> sumsq well under 2^28), so a
// groupshared tree-sum (B5's own SCHEME-1 shape, combine changed from max to
// add) reproduces the sequential CPU sum bit-for-bit regardless of
// evaluation order.
groupshared int64_t gSumSq[256];

int64_t RmsSumSqParallelGpu(uint t, RWByteAddressBuffer buf, uint base, int hidden_size)
{
    int64_t local = 0;
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int64_t hv = (int64_t)(int)buf.Load<int>(base + (uint)i * 4u);
        local += hv * hv;
    }
    gSumSq[t] = local;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (t < s) gSumSq[t] += gSumSq[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    return gSumSq[0];
}

#endif  // SSLM_SITE_COMMON_HLSLI
