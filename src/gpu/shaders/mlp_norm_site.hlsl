// T-2035: mlp_norm -- real GPU dispatch, site 11 of 16. Same construction as
// attn_norm_site.hlsl (RmsNormSite), reading the STAGED attn_residual output
// (LayerScratch attn_stream, offset 288/320) rather than seq.hidden_codes,
// and writing the "normed" scratch slot (0/32) -- safe to reuse: q_proj/
// kv_proj already consumed attn_norm's own value earlier this layer.
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

static const int kNormFracBitsGpu2 = 16;

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    int hidden_size = (int)g_hidden_size;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    int64_t h[kMaxSiteN];
    [unroll] for (int z = 0; z < kMaxSiteN; ++z) h[z] = 0;
    for (int i = 0; i < hidden_size; ++i) h[i] = (int64_t)(int)LayerScratch.Load<int>(288 + i * 4);

    uint off_gain = layer_base + Layout.Load<uint>(35 * 4);
    int g[kMaxSiteN];
    [unroll] for (int z2 = 0; z2 < kMaxSiteN; ++z2) g[z2] = 0;
    for (int i2 = 0; i2 < hidden_size; ++i2) g[i2] = (int)LayerWeights.Load<int>(off_gain + (uint)i2 * 4u);

    int64_t sumsq = 0;
    for (int i3 = 0; i3 < hidden_size; ++i3) sumsq += h[i3] * h[i3];
    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu2), (int64_t)hidden_size));
    root = (root > 1) ? root : 1;

    int64_t wide[kMaxSiteN];
    [unroll] for (int z3 = 0; z3 < kMaxSiteN; ++z3) wide[z3] = 0;
    for (int i4 = 0; i4 < hidden_size; ++i4) wide[i4] = FloorDivI64Gpu(h[i4] << (2 * kNormFracBitsGpu2), root) * (int64_t)g[i4];

    uint off_site = layer_base + Layout.Load<uint>(36 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxSiteN], incoming_e[kMaxSiteN];
    [unroll] for (int z4 = 0; z4 < kMaxSiteN; ++z4) { incoming_m[z4] = 0; incoming_e[z4] = 0; }

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(wide, hidden_size, incoming_m, incoming_e, 0, site_m, site_e, out_codes,
                                out_scale_m, out_scale_e, status_tag);
    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    for (int i5 = 0; i5 < hidden_size; ++i5) LayerScratch.Store<int>(0 + i5 * 4, (int)out_codes[i5]);
    LayerScratch.Store<int64_t>(32, out_scale_m);
    LayerScratch.Store<int64_t>(40, out_scale_e);
}
