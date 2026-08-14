// T-2035: down_proj -- real GPU dispatch, site 15 of 16. ProjectAndFunnel, no
// bias, input=act_codes (mlp_act's own output, scratch 432/464),
// in_channels=intermediate_size, out_channels=hidden_size, weight=
// down_weight (Layout[48]).
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

    int in_channels = (int)g_intermediate_size;
    int out_channels = (int)g_hidden_size;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    int in_codes[kMaxSiteN];
    [unroll] for (int z = 0; z < kMaxSiteN; ++z) in_codes[z] = 0;
    for (int i = 0; i < in_channels; ++i) in_codes[i] = (int)LayerScratch.Load<int>(432 + i * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(464);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(472);

    uint off_weight = layer_base + Layout.Load<uint>(48 * 4);
    uint off_id = layer_base + Layout.Load<uint>(49 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(50 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(51 * 4);
    uint off_site = layer_base + Layout.Load<uint>(52 * 4);

    int64_t acc[kMaxSiteN];
    [unroll] for (int z2 = 0; z2 < kMaxSiteN; ++z2) acc[z2] = 0;
    for (int j = 0; j < out_channels; ++j)
    {
        int weight_row[kMaxSiteN];
        [unroll] for (int z3 = 0; z3 < kMaxSiteN; ++z3) weight_row[z3] = 0;
        for (int i2 = 0; i2 < in_channels; ++i2)
            weight_row[i2] = LoadSignedByteGpu(LayerWeights, off_weight + (uint)(j * in_channels + i2));
        int64_t dot = GemmDotGpu(in_codes, weight_row, in_channels);
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
    RequantChainCheckedFullGpu(acc, out_channels, incoming_m, incoming_e, 1, site_m, site_e, out_codes,
                                out_scale_m, out_scale_e, status_tag);
    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    for (int i5 = 0; i5 < out_channels; ++i5) LayerScratch.Store<int>(480 + i5 * 4, (int)out_codes[i5]);
    LayerScratch.Store<int64_t>(512, out_scale_m);
    LayerScratch.Store<int64_t>(520, out_scale_e);
}
