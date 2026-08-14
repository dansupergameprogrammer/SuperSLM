// T-2035: mlp_residual -- real GPU dispatch, site 16 of 16. Same construction
// as attn_residual_site.hlsl (ResidualReconcileSite), reconciling against the
// STAGED attn_residual output (LayerScratch attn_stream, 288/320) rather than
// SeqState -- forward_sites.cpp:1772-1777's own "reconciles against the
// staged output, not seq" reasoning. branch=down_codes/down_scale (480/512),
// out=stream_next (528/560), the layer's own final residual this checkpoint's
// commit dispatch commits into persistent SeqState.
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

    int64_t stream_m = LayerScratch.Load<int64_t>(320);
    int64_t stream_e = LayerScratch.Load<int64_t>(328);
    if (stream_m < -2147483648LL || stream_m > 2147483647LL)
    {
        SeqState.Store<int64_t>(72, kTagCarriedScaleMantissaOutOfDomain);
        return;
    }
    int64_t r_h = DynamicScaleReciprocalSharedGpu(stream_m);

    int64_t wide[kMaxSiteN];
    [unroll] for (int z = 0; z < kMaxSiteN; ++z) wide[z] = 0;
    int64_t branch_m = LayerScratch.Load<int64_t>(512);
    int64_t branch_e = LayerScratch.Load<int64_t>(520);
    bool any_magnitude_out_of_domain = false;
    for (int i = 0; i < hidden_size; ++i)
    {
        int branch_code = LayerScratch.Load<int>(480 + i * 4);
        bool would_clamp, magnitude_exceeded;
        int64_t reconciled = LandingRescaleGpu((int64_t)branch_code, branch_m, r_h, branch_e, stream_e,
                                                 would_clamp, magnitude_exceeded);
        if (magnitude_exceeded) { any_magnitude_out_of_domain = true; continue; }
        int stream_code = LayerScratch.Load<int>(288 + i * 4);
        wide[i] = reconciled + (int64_t)stream_code;
    }
    if (any_magnitude_out_of_domain)
    {
        SeqState.Store<int64_t>(72, kTagResidualReconciliationMagnitudeOutOfDomain);
        return;
    }

    uint off_site = layer_base + Layout.Load<uint>(53 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxSiteN], incoming_e[kMaxSiteN];
    [unroll] for (int z2 = 0; z2 < kMaxSiteN; ++z2) { incoming_m[z2] = 0; incoming_e[z2] = 0; }
    incoming_m[0] = stream_m; incoming_e[0] = stream_e;

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(wide, hidden_size, incoming_m, incoming_e, 1, site_m, site_e, out_codes,
                                out_scale_m, out_scale_e, status_tag);
    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    for (int i2 = 0; i2 < hidden_size; ++i2) LayerScratch.Store<int>(528 + i2 * 4, (int)out_codes[i2]);
    LayerScratch.Store<int64_t>(560, out_scale_m);
    LayerScratch.Store<int64_t>(568, out_scale_e);
}
