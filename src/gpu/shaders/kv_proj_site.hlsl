// T-2039 (real-capacity shader geometry): kv_proj -- real GPU dispatch, site
// 3 of 16, FUSED (Sec5.6, D-SLM2992: k_proj and v_proj are not two CPU
// sites, they are one guard/commit block, forward_sites.cpp:1410-1531). Both
// k_bias/v_bias guards are evaluated UNCONDITIONALLY before either K or V
// row commits (T-2003's remedy), combined with CPU's own precedence
// (k_bias's failure status if k_bias failed, else v_bias's). On Ok,
// LandingRescale+ClampRopeCode lands every (kv_head, dim) pair immediately
// into the KV cache (matching CPU's own mid-layer, unstaged K/V write
// timing, gated only by the sequence-level sticky word) and folds this
// dispatch's own saturation-count delta into the split (kv_sat_lo, kv_sat_hi)
// accumulator (T-2008's remedy).
//
// Rebuilt to the design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), both GEMMs' own kv_hidden_size output channels spread
// across the group via N-per-thread striding, kacc/vacc streamed through
// WorkScratch's own WIDE_A/WIDE_B regions (never a per-call local array),
// the landing-rescale pass parallelized over (kv_head, dim) pairs, and the
// clamp counter accumulated via a groupshared atomic before the single
// thread-0 commit into SeqState's split accumulator.
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index;
    uint g_hidden_size;
    uint g_head_dim;
    uint g_num_kv_heads;
    uint g_context_cap;
    uint g_position;
};

ByteAddressBuffer   LayerWeights  : register(t0);
ByteAddressBuffer   Layout        : register(t1);
ByteAddressBuffer   RopeInfo      : register(t2);
ByteAddressBuffer   ScratchLayout : register(t7);
RWByteAddressBuffer SeqState      : register(u0);
RWByteAddressBuffer LayerScratch  : register(u1);
RWByteAddressBuffer KvCache       : register(u2);
RWByteAddressBuffer WorkScratch   : register(u3);

groupshared uint gKFailed;
groupshared uint gVFailed;
groupshared uint gTotalClamps;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    int head_dim = (int)g_head_dim;
    int num_kv_heads = (int)g_num_kv_heads;
    int kv_hidden_size = num_kv_heads * head_dim;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t normed_scale_m = LayerScratch.Load<int64_t>(normed_scale_off + 0);
    int64_t normed_scale_e = LayerScratch.Load<int64_t>(normed_scale_off + 8);

    uint off_kw = layer_base + Layout.Load<uint>(9 * 4);
    uint off_vw = layer_base + Layout.Load<uint>(10 * 4);
    uint off_kid = layer_base + Layout.Load<uint>(11 * 4);
    uint off_kmult = layer_base + Layout.Load<uint>(12 * 4);
    uint off_kshift = layer_base + Layout.Load<uint>(13 * 4);
    uint off_vid = layer_base + Layout.Load<uint>(14 * 4);
    uint off_vmult = layer_base + Layout.Load<uint>(15 * 4);
    uint off_vshift = layer_base + Layout.Load<uint>(16 * 4);
    uint off_kbp = layer_base + Layout.Load<uint>(17 * 4);
    uint off_kb = layer_base + Layout.Load<uint>(18 * 4);
    uint off_vbp = layer_base + Layout.Load<uint>(19 * 4);
    uint off_vb = layer_base + Layout.Load<uint>(20 * 4);
    uint off_rtk = layer_base + Layout.Load<uint>(21 * 4);
    uint off_etk = layer_base + Layout.Load<uint>(22 * 4);
    uint off_rtv = layer_base + Layout.Load<uint>(23 * 4);
    uint off_etv = layer_base + Layout.Load<uint>(24 * 4);

    uint wide_b_off = ScratchLayout.Load<uint>(23 * 4);

    // T-2101 (follow-up to D-SLM3312/D-SLM3313): kv_proj stays single-group. Splitting the GEMM
    // step alone across multiple groups is mechanical (same shape as q/o/gate/up/down_proj below),
    // but the fused K+V bias-precedence check (gKFailed/gVFailed) and the landing-rescale clamp
    // counter (gTotalClamps, an InterlockedAdd across every (kv_head, dim) pair this dispatch
    // commits) are genuine cross-thread cooperative reductions over the WHOLE kv_hidden_size row,
    // scoped to one group's own groupshared memory -- splitting those across groups without either
    // atomics or a second dispatch would be exactly the reordered-accumulation/cross-group-
    // reduction shape this round's own commission forbids. kv_proj is 1.7% of GPU-busy time
    // (D-SLM3313's own table); not worth the restructuring this round's two-cycle budget affords.
    GemmParallelGpu(t, LayerScratch, normed_off, LayerWeights, off_kw, LayerWeights, off_kid,
                     LayerWeights, off_kmult, LayerWeights, off_kshift, hidden_size, kv_hidden_size,
                     WorkScratch, 0u, 256);
    GemmParallelGpu(t, LayerScratch, normed_off, LayerWeights, off_vw, LayerWeights, off_vid,
                     LayerWeights, off_vmult, LayerWeights, off_vshift, hidden_size, kv_hidden_size,
                     WorkScratch, wide_b_off, 256);
    DeviceMemoryBarrierWithGroupSync();

    // Both guards evaluated unconditionally, combined with CPU's own
    // precedence: k_bias's status wins if k_bias failed, else v_bias's.
    int64_t k_tag = kTagOk, v_tag = kTagOk;
    bool k_ran = false, v_ran = false;
    if (t == 0) { gKFailed = 0; gVFailed = 0; }
    GroupMemoryBarrierWithGroupSync();

    int64_t kbias_present = LayerWeights.Load<int64_t>(off_kbp);
    if (kbias_present != 0)
    {
        k_ran = true;
        if (!ApplyBiasReconcileRowGpuP(t, WorkScratch, 0u, kv_hidden_size, LayerWeights, off_kb,
                                        normed_scale_m, normed_scale_e, k_tag))
        {
            if (t == 0) gKFailed = 1;
        }
    }
    int64_t vbias_present = LayerWeights.Load<int64_t>(off_vbp);
    if (vbias_present != 0)
    {
        v_ran = true;
        if (!ApplyBiasReconcileRowGpuP(t, WorkScratch, wide_b_off, kv_hidden_size, LayerWeights, off_vb,
                                        normed_scale_m, normed_scale_e, v_tag))
        {
            if (t == 0) gVFailed = 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();

    if (gKFailed != 0 || gVFailed != 0)
    {
        if (t == 0)
        {
            int64_t combined_tag = (gKFailed != 0) ? k_tag : v_tag;
            SeqState.Store<int64_t>(sticky_off, combined_tag);
        }
        return;  // neither K nor V lands -- fused dispatch, no partial commit
    }

    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, (uint)num_kv_heads, (uint)head_dim);
    uint v_half_off = kv_half_off + g_context_cap * (uint)num_kv_heads * (uint)head_dim;

    if (t == 0) gTotalClamps = 0;
    GroupMemoryBarrierWithGroupSync();

    for (int i = (int)t; i < kv_hidden_size; i += 256)
    {
        int h = i / head_dim;
        int d = i % head_dim;
        int64_t r_t_k = LayerWeights.Load<int64_t>(off_rtk + (uint)h * 8u);
        int64_t e_t_k = LayerWeights.Load<int64_t>(off_etk + (uint)h * 8u);
        int64_t r_t_v = LayerWeights.Load<int64_t>(off_rtv + (uint)h * 8u);
        int64_t e_t_v = LayerWeights.Load<int64_t>(off_etv + (uint)h * 8u);
        uint row_off = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)head_dim, (uint)h, g_position);

        int64_t kacc_i = WorkScratch.Load<int64_t>(0u + (uint)i * 8u);
        int64_t vacc_i = WorkScratch.Load<int64_t>(wide_b_off + (uint)i * 8u);

        bool k_clamp, k_mag_exceeded;
        int64_t k_raw = LandingRescaleGpu(kacc_i, normed_scale_m, r_t_k, normed_scale_e, e_t_k, k_clamp,
                                           k_mag_exceeded);
        if (k_clamp) InterlockedAdd(gTotalClamps, 1u);
        int64_t k_val = ClampRopeCodeGpu(k_raw);
        StoreSignedByteGpu(KvCache, kv_half_off + row_off + (uint)d, (int)k_val);

        bool v_clamp, v_mag_exceeded;
        int64_t v_raw = LandingRescaleGpu(vacc_i, normed_scale_m, r_t_v, normed_scale_e, e_t_v, v_clamp,
                                           v_mag_exceeded);
        if (v_clamp) InterlockedAdd(gTotalClamps, 1u);
        int64_t v_val = ClampRopeCodeGpu(v_raw);
        StoreSignedByteGpu(KvCache, v_half_off + row_off + (uint)d, (int)v_val);
    }
    GroupMemoryBarrierWithGroupSync();

    if (t == 0 && gTotalClamps != 0)
    {
        uint sat_lo_off = SeqSatLoOffGpu(hidden_size);
        uint sat_hi_off = SeqSatHiOffGpu(hidden_size);
        uint old_lo = SeqState.Load(sat_lo_off);
        uint new_lo = old_lo + gTotalClamps;
        SeqState.Store(sat_lo_off, new_lo);
        if (new_lo < old_lo)
        {
            uint old_hi = SeqState.Load(sat_hi_off);
            SeqState.Store(sat_hi_off, old_hi + 1u);
        }
    }
}
