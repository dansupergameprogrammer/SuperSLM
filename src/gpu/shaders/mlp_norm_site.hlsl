// T-2039 (real-capacity shader geometry): mlp_norm -- real GPU dispatch,
// site 11 of 16. Same construction as attn_norm_site.hlsl (RmsNormSite),
// reading the STAGED attn_residual output (LayerScratch's own attn_stream
// slot) rather than SeqState, and writing the "normed" scratch slot --safe
// to reuse: q_proj/kv_proj already consumed attn_norm's own value earlier
// this layer. Rebuilt to the design's own production dispatch geometry
// (Sec5.4/Sec5.5): numthreads(256,1,1), the sumsq reduction via the SCHEME-1
// groupshared tree, the wide row streamed through WorkScratch.
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
ByteAddressBuffer   ScratchLayout  : register(t7);
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);
RWByteAddressBuffer WorkScratch    : register(u3);

static const int kNormFracBitsGpu2 = 16;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint attn_stream_off = ScratchLayout.Load<uint>(9 * 4);

    int64_t sumsq = RmsSumSqParallelGpu(t, LayerScratch, attn_stream_off, hidden_size);
    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu2), (int64_t)hidden_size));
    root = (root > 1) ? root : 1;

    uint off_gain = layer_base + Layout.Load<uint>(35 * 4);
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int64_t hv = (int64_t)(int)LayerScratch.Load<int>(attn_stream_off + (uint)i * 4u);
        int g = (int)LayerWeights.Load<int>(off_gain + (uint)i * 4u);
        int64_t wv = FloorDivI64Gpu(hv << (2 * kNormFracBitsGpu2), root) * (int64_t)g;
        WorkScratch.Store<int64_t>(0u + (uint)i * 8u, wv);
    }
    DeviceMemoryBarrierWithGroupSync();

    uint off_site = layer_base + Layout.Load<uint>(36 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, 0, site_m, site_e,
                                 LayerScratch, normed_off, normed_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
