// T-2032: kv_proj -- real GPU dispatch, site 3 of 16, FUSED (Sec5.6, D-SLM2992:
// k_proj and v_proj are not two CPU sites, they are one guard/commit block,
// forward_sites.cpp:1410-1531). Both k_bias/v_bias guards are evaluated
// UNCONDITIONALLY before either K or V row commits (T-2003's remedy), combined
// with CPU's own precedence (k_bias's failure status if k_bias failed, else
// v_bias's). On Ok, LandingRescale+ClampRopeCode lands every (kv_head, dim)
// pair immediately into the KV cache (matching CPU's own mid-layer, unstaged
// K/V write timing, gated only by the sequence-level sticky word -- never by
// an end-of-layer commit) and folds this dispatch's own saturation-count delta
// into the split (kv_sat_lo, kv_sat_hi) accumulator (T-2008's remedy,
// Claude/Vitruvius/t2010-remedy-probe/sat_count_split.hlsl's own carry-add
// shape, single-threaded here since this checkpoint's own fixture geometry
// needs no cross-thread reduction -- Sec11.2's own "stated, honest
// simplification" precedent for this suite's fixed small scale).
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

ByteAddressBuffer   LayerWeights : register(t0);
ByteAddressBuffer   Layout       : register(t1);
ByteAddressBuffer   RopeInfo     : register(t2);
RWByteAddressBuffer SeqState     : register(u0);
RWByteAddressBuffer LayerScratch : register(u1);
RWByteAddressBuffer KvCache      : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    int hidden_size = (int)g_hidden_size;
    int head_dim = (int)g_head_dim;
    int num_kv_heads = (int)g_num_kv_heads;
    int kv_hidden_size = num_kv_heads * head_dim;
    uint layer_base = g_layer_index * Layout.Load<uint>(25 * 4);

    int normed_codes[kMaxSiteN];
    [unroll]
    for (int z = 0; z < kMaxSiteN; ++z) normed_codes[z] = 0;
    for (int ni = 0; ni < hidden_size; ++ni) normed_codes[ni] = (int)LayerScratch.Load<int>(0 + ni * 4);
    int64_t normed_scale_m = LayerScratch.Load<int64_t>(32);
    int64_t normed_scale_e = LayerScratch.Load<int64_t>(40);

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

    int64_t kacc[kMaxSiteN];
    int64_t vacc[kMaxSiteN];
    [unroll]
    for (int z2 = 0; z2 < kMaxSiteN; ++z2) { kacc[z2] = 0; vacc[z2] = 0; }

    for (int j = 0; j < kv_hidden_size; ++j)
    {
        int krow[kMaxSiteN];
        int vrow[kMaxSiteN];
        [unroll]
        for (int z3 = 0; z3 < kMaxSiteN; ++z3) { krow[z3] = 0; vrow[z3] = 0; }
        for (int i2 = 0; i2 < hidden_size; ++i2)
        {
            krow[i2] = LoadSignedByteGpu(LayerWeights, off_kw + (uint)(j * hidden_size + i2));
            vrow[i2] = LoadSignedByteGpu(LayerWeights, off_vw + (uint)(j * hidden_size + i2));
        }
        int64_t kdot = GemmDotGpu(normed_codes, krow, hidden_size);
        int64_t vdot = GemmDotGpu(normed_codes, vrow, hidden_size);
        int kid = (int)LayerWeights.Load<int>(off_kid + (uint)j * 4u);
        int kmult = (int)LayerWeights.Load<int>(off_kmult + (uint)j * 4u);
        int kshift = (int)LayerWeights.Load<int>(off_kshift + (uint)j * 4u);
        int vid = (int)LayerWeights.Load<int>(off_vid + (uint)j * 4u);
        int vmult = (int)LayerWeights.Load<int>(off_vmult + (uint)j * 4u);
        int vshift = (int)LayerWeights.Load<int>(off_vshift + (uint)j * 4u);
        kacc[j] = ApplyWeightScaleFoldGpu(kdot, kid, kmult, kshift);
        vacc[j] = ApplyWeightScaleFoldGpu(vdot, vid, vmult, vshift);
    }

    // Both guards evaluated unconditionally (pure functions, no side effect on
    // failure), combined with CPU's own precedence: k_bias's status wins if
    // k_bias failed, else v_bias's.
    bool k_failed = false, v_failed = false;
    int64_t k_tag = kTagOk, v_tag = kTagOk;
    int64_t kbias_present = LayerWeights.Load<int64_t>(off_kbp);
    if (kbias_present != 0)
    {
        int64_t bias[kMaxSiteN];
        [unroll]
        for (int z4 = 0; z4 < kMaxSiteN; ++z4) bias[z4] = 0;
        for (int i3 = 0; i3 < kv_hidden_size; ++i3) bias[i3] = LayerWeights.Load<int64_t>(off_kb + (uint)i3 * 8u);
        int64_t kacc_copy[kMaxSiteN];
        [unroll]
        for (int z5 = 0; z5 < kMaxSiteN; ++z5) kacc_copy[z5] = kacc[z5];
        if (!ApplyBiasReconcileRowGpu(kacc_copy, kv_hidden_size, bias, normed_scale_m, normed_scale_e, k_tag))
        {
            k_failed = true;
        }
        else
        {
            [unroll]
            for (int z6 = 0; z6 < kMaxSiteN; ++z6) kacc[z6] = kacc_copy[z6];
        }
    }
    int64_t vbias_present = LayerWeights.Load<int64_t>(off_vbp);
    if (vbias_present != 0)
    {
        int64_t bias2[kMaxSiteN];
        [unroll]
        for (int z7 = 0; z7 < kMaxSiteN; ++z7) bias2[z7] = 0;
        for (int i4 = 0; i4 < kv_hidden_size; ++i4) bias2[i4] = LayerWeights.Load<int64_t>(off_vb + (uint)i4 * 8u);
        int64_t vacc_copy[kMaxSiteN];
        [unroll]
        for (int z8 = 0; z8 < kMaxSiteN; ++z8) vacc_copy[z8] = vacc[z8];
        if (!ApplyBiasReconcileRowGpu(vacc_copy, kv_hidden_size, bias2, normed_scale_m, normed_scale_e, v_tag))
        {
            v_failed = true;
        }
        else
        {
            [unroll]
            for (int z9 = 0; z9 < kMaxSiteN; ++z9) vacc[z9] = vacc_copy[z9];
        }
    }

    if (k_failed || v_failed)
    {
        int64_t combined_tag = k_failed ? k_tag : v_tag;
        SeqState.Store<int64_t>(72, combined_tag);
        return;  // neither K nor V lands -- fused dispatch, no partial commit
    }

    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, (uint)num_kv_heads, (uint)head_dim);
    uint v_half_off = kv_half_off + g_context_cap * (uint)num_kv_heads * (uint)head_dim;

    uint total_clamps = 0;
    for (int h = 0; h < num_kv_heads; ++h)
    {
        int64_t r_t_k = LayerWeights.Load<int64_t>(off_rtk + (uint)h * 8u);
        int64_t e_t_k = LayerWeights.Load<int64_t>(off_etk + (uint)h * 8u);
        int64_t r_t_v = LayerWeights.Load<int64_t>(off_rtv + (uint)h * 8u);
        int64_t e_t_v = LayerWeights.Load<int64_t>(off_etv + (uint)h * 8u);
        uint row_off = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)head_dim, (uint)h, g_position);
        for (int d = 0; d < head_dim; ++d)
        {
            int i = h * head_dim + d;
            bool k_clamp;
            int64_t k_raw = LandingRescaleGpu(kacc[i], normed_scale_m, r_t_k, normed_scale_e, e_t_k, k_clamp);
            if (k_clamp) total_clamps += 1;
            int64_t k_val = ClampRopeCodeGpu(k_raw);
            StoreSignedByteGpu(KvCache, kv_half_off + row_off + (uint)d, (int)k_val);

            bool v_clamp;
            int64_t v_raw = LandingRescaleGpu(vacc[i], normed_scale_m, r_t_v, normed_scale_e, e_t_v, v_clamp);
            if (v_clamp) total_clamps += 1;
            int64_t v_val = ClampRopeCodeGpu(v_raw);
            StoreSignedByteGpu(KvCache, v_half_off + row_off + (uint)d, (int)v_val);
        }
    }

    if (total_clamps != 0)
    {
        uint old_lo = SeqState.Load(56);
        uint new_lo = old_lo + total_clamps;
        SeqState.Store(56, new_lo);
        if (new_lo < old_lo)
        {
            uint old_hi = SeqState.Load(60);
            SeqState.Store(60, old_hi + 1u);
        }
    }
}
