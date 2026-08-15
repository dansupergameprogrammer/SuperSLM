// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): gate_proj's own GEMM step,
// split out of gate_proj_site.hlsl (now requant-only) into its own multi-group dispatch. See
// `down_proj_gemm_site.hlsl`'s own header comment for the full correctness account (identical
// shape: SV_DispatchThreadID replaces SV_GroupThreadID as GemmParallelGpu's own thread index,
// `stride` becomes the dispatch's own total thread count instead of the literal 256 -- no change
// to any output element's computation or its summation order).
//
// Dispatched Dispatch(ceil(intermediate_size/256), 1, 1) (superslm_gpu.cpp).
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

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int hidden_size = (int)g_hidden_size;
    int out_channels = (int)g_intermediate_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);

    uint off_weight = layer_base + Layout.Load<uint>(37 * 4);
    uint off_id = layer_base + Layout.Load<uint>(38 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(39 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(40 * 4);

    int stride = ((out_channels + 255) / 256) * 256;
    GemmParallelGpu(dtid.x, LayerScratch, normed_off, LayerWeights, off_weight, LayerWeights, off_id,
                     LayerWeights, off_mult, LayerWeights, off_shift, hidden_size, out_channels,
                     WorkScratch, 0u, stride);
}
