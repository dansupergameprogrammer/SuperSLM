// T-2035: o_proj -- real GPU dispatch, site 9 of 16. ProjectAndFunnel, no
// bias (forward_sites.cpp:1705-1710): GemmInt8AccumulateRow(ctx_codes,
// o_weight) -> ApplyWeightScaleFold -> RequantChainCheckedFull(incoming={ctx_scale}).
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
};

ByteAddressBuffer   LayerWeights   : register(t0);
ByteAddressBuffer   Layout         : register(t1);
ByteAddressBuffer   RopeInfo       : register(t2);
ByteAddressBuffer   ModelConstants : register(t3);
ByteAddressBuffer   SiluLut        : register(t4);
ByteAddressBuffer   RopeCosTable   : register(t5);
ByteAddressBuffer   RopeSinTable   : register(t6);
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    int hidden_size = (int)g_hidden_size;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    int in_codes[kMaxSiteN];
    [unroll] for (int z = 0; z < kMaxSiteN; ++z) in_codes[z] = 0;
    for (int i = 0; i < hidden_size; ++i) in_codes[i] = (int)LayerScratch.Load<int>(192 + i * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(224);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(232);

    uint off_weight = layer_base + Layout.Load<uint>(25 * 4);
    uint off_id = layer_base + Layout.Load<uint>(26 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(27 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(28 * 4);
    uint off_site = layer_base + Layout.Load<uint>(29 * 4);

    int64_t acc[kMaxSiteN];
    [unroll] for (int z2 = 0; z2 < kMaxSiteN; ++z2) acc[z2] = 0;
    for (int j = 0; j < hidden_size; ++j)
    {
        int weight_row[kMaxSiteN];
        [unroll] for (int z3 = 0; z3 < kMaxSiteN; ++z3) weight_row[z3] = 0;
        for (int i2 = 0; i2 < hidden_size; ++i2)
            weight_row[i2] = LoadSignedByteGpu(LayerWeights, off_weight + (uint)(j * hidden_size + i2));
        int64_t dot = GemmDotGpu(in_codes, weight_row, hidden_size);
        int identity = (int)LayerWeights.Load<int>(off_id + (uint)j * 4u);
        int mult = (int)LayerWeights.Load<int>(off_mult + (uint)j * 4u);
        int shift = (int)LayerWeights.Load<int>(off_shift + (uint)j * 4u);
        acc[j] = ApplyWeightScaleFoldGpu(dot, identity, mult, shift);
    }

    int64_t incoming_m[kMaxSiteN], incoming_e[kMaxSiteN];
    [unroll] for (int z5 = 0; z5 < kMaxSiteN; ++z5) { incoming_m[z5] = 0; incoming_e[z5] = 0; }
    incoming_m[0] = in_scale_m; incoming_e[0] = in_scale_e;
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(acc, hidden_size, incoming_m, incoming_e, 1, site_m, site_e, out_codes,
                                out_scale_m, out_scale_e, status_tag);
    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    for (int i5 = 0; i5 < hidden_size; ++i5) LayerScratch.Store<int>(240 + i5 * 4, (int)out_codes[i5]);
    LayerScratch.Store<int64_t>(272, out_scale_m);
    LayerScratch.Store<int64_t>(280, out_scale_e);
}
