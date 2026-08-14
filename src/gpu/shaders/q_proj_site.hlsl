// T-2032: q_proj -- real GPU dispatch, site 2 of 16. Bit-exact port of
// forward_sites.cpp's ProjectAndFunnel (forward_sites.cpp:1020-1048) at the
// q_proj call site (forward_sites.cpp:1395-1400): GemmInt8AccumulateRow ->
// per-channel ApplyWeightScaleFold -> optional ApplyBiasReconcileRow (q_bias)
// -> RequantChainCheckedFull with incoming={normed_scale}.
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
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    int normed_codes[kMaxSiteN];
    [unroll]
    for (int z = 0; z < kMaxSiteN; ++z) normed_codes[z] = 0;
    for (int i = 0; i < hidden_size; ++i)
    {
        normed_codes[i] = (int)LayerScratch.Load<int>(0 + i * 4);
    }
    int64_t normed_scale_m = LayerScratch.Load<int64_t>(32);
    int64_t normed_scale_e = LayerScratch.Load<int64_t>(40);

    uint off_weight = layer_base + Layout.Load<uint>(2 * 4);
    uint off_id = layer_base + Layout.Load<uint>(3 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(4 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(5 * 4);
    uint off_site = layer_base + Layout.Load<uint>(6 * 4);
    uint off_bias_present = layer_base + Layout.Load<uint>(7 * 4);
    uint off_bias = layer_base + Layout.Load<uint>(8 * 4);

    int64_t acc[kMaxSiteN];
    [unroll]
    for (int z2 = 0; z2 < kMaxSiteN; ++z2) acc[z2] = 0;
    for (int j = 0; j < hidden_size; ++j)
    {
        int weight_row[kMaxSiteN];
        [unroll]
        for (int z3 = 0; z3 < kMaxSiteN; ++z3) weight_row[z3] = 0;
        for (int i2 = 0; i2 < hidden_size; ++i2)
        {
            weight_row[i2] = LoadSignedByteGpu(LayerWeights, off_weight + (uint)(j * hidden_size + i2));
        }
        int64_t dot = GemmDotGpu(normed_codes, weight_row, hidden_size);
        int identity = (int)LayerWeights.Load<int>(off_id + (uint)j * 4u);
        int mult = (int)LayerWeights.Load<int>(off_mult + (uint)j * 4u);
        int shift = (int)LayerWeights.Load<int>(off_shift + (uint)j * 4u);
        acc[j] = ApplyWeightScaleFoldGpu(dot, identity, mult, shift);
    }

    int64_t bias_present = LayerWeights.Load<int64_t>(off_bias_present);
    if (bias_present != 0)
    {
        int64_t bias[kMaxSiteN];
        [unroll]
        for (int z4 = 0; z4 < kMaxSiteN; ++z4) bias[z4] = 0;
        for (int i3 = 0; i3 < hidden_size; ++i3)
        {
            bias[i3] = LayerWeights.Load<int64_t>(off_bias + (uint)i3 * 8u);
        }
        int64_t bias_tag;
        if (!ApplyBiasReconcileRowGpu(acc, hidden_size, bias, normed_scale_m, normed_scale_e, bias_tag))
        {
            SeqState.Store<int64_t>(72, bias_tag);
            return;
        }
    }

    int64_t incoming_m[kMaxSiteN];
    int64_t incoming_e[kMaxSiteN];
    [unroll]
    for (int z5 = 0; z5 < kMaxSiteN; ++z5) { incoming_m[z5] = 0; incoming_e[z5] = 0; }
    incoming_m[0] = normed_scale_m;
    incoming_e[0] = normed_scale_e;

    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(acc, hidden_size, incoming_m, incoming_e, /*n_incoming=*/1, site_m,
                                site_e, out_codes, out_scale_m, out_scale_e, status_tag);

    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    // T-2035: q_codes/q_scale feed RoPE (site 4) and, through RoPE's own
    // rotated output, attention -- LayerScratch offsets 48 (q_codes[8] i32)
    // and 80 (q_scale, 16 bytes).
    for (int i6 = 0; i6 < hidden_size; ++i6)
    {
        LayerScratch.Store<int>(48 + i6 * 4, (int)out_codes[i6]);
    }
    LayerScratch.Store<int64_t>(80, out_scale_m);
    LayerScratch.Store<int64_t>(88, out_scale_e);
}
