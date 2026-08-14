// T-2035: mlp_act -- real GPU dispatch, site 14 of 16. Bit-exact port of
// forward_sites.cpp's MlpActSite (forward_sites.cpp): CheckSiluCompositionScaleDomain
// (gate_scale) FIRST -- SiluSigmoidQ15 never runs on rejection -- then per-
// element sig[i]=SiluSigmoidQ15(LUT, gate_code[i], gate_scale), wide[i] =
// gate_code[i]*sig[i]*up_code[i], RequantChainCheckedFull(incoming=
// {gate_scale, up_scale}).
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

    int n = (int)g_intermediate_size;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    int64_t gate_m = LayerScratch.Load<int64_t>(368);
    int64_t gate_e = LayerScratch.Load<int64_t>(376);
    if (!CheckSiluCompositionScaleDomainGpu(gate_m, (int)gate_e))
    {
        SeqState.Store<int64_t>(72, kTagSiluCompositionScaleOutOfDomain);
        return;
    }
    int64_t up_m = LayerScratch.Load<int64_t>(416);
    int64_t up_e = LayerScratch.Load<int64_t>(424);

    int64_t wide[kMaxSiteN];
    [unroll] for (int z = 0; z < kMaxSiteN; ++z) wide[z] = 0;
    for (int i = 0; i < n; ++i)
    {
        int gate_code = (int)LayerScratch.Load<int>(336 + i * 4);
        int up_code = (int)LayerScratch.Load<int>(384 + i * 4);
        int sig = SiluSigmoidQ15Gpu(SiluLut, gate_code, gate_m, (int)gate_e);
        wide[i] = (int64_t)gate_code * (int64_t)sig * (int64_t)up_code;
    }

    uint off_site = layer_base + Layout.Load<uint>(47 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxSiteN], incoming_e[kMaxSiteN];
    [unroll] for (int z2 = 0; z2 < kMaxSiteN; ++z2) { incoming_m[z2] = 0; incoming_e[z2] = 0; }
    incoming_m[0] = gate_m; incoming_e[0] = gate_e;
    incoming_m[1] = up_m; incoming_e[1] = up_e;

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(wide, n, incoming_m, incoming_e, 2, site_m, site_e, out_codes, out_scale_m,
                                out_scale_e, status_tag);
    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    for (int i2 = 0; i2 < n; ++i2) LayerScratch.Store<int>(432 + i2 * 4, (int)out_codes[i2]);
    LayerScratch.Store<int64_t>(464, out_scale_m);
    LayerScratch.Store<int64_t>(472, out_scale_e);
}
