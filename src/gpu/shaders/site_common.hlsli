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

// Cooperative parallel GEMM: `out_channels` output channels spread across the
// group via N-per-thread striding (non-uniform where out_channels does not
// divide 256 evenly -- threads 0..(out_channels mod 256 - 1) simply get one
// more loop iteration than the rest, the T-2014 correction's own split).
// Each output channel's own full `in_channels`-wide int64 dot product
// (GemmInt8AccumulateRow's own accumulation order, proven bit-identical to
// the SSE2 production path by associativity, design Sec4) is computed by
// exactly one thread, in full -- no cross-thread accumulation, so no
// reduction and no barrier is needed here; the folded int64 result lands in
// `scratch` at `wide_base + j*8`, one thread's own write, never re-read by
// another thread in the SAME phase.
void GemmParallelGpu(uint t, RWByteAddressBuffer in_buf, uint in_base, ByteAddressBuffer w_buf,
                      uint w_base, ByteAddressBuffer id_buf, uint id_base, ByteAddressBuffer mult_buf,
                      uint mult_base, ByteAddressBuffer shift_buf, uint shift_base, int in_channels,
                      int out_channels, RWByteAddressBuffer scratch, uint wide_base)
{
    for (int j = (int)t; j < out_channels; j += 256)
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
