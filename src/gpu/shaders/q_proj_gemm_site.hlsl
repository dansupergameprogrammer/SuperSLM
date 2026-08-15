// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): q_proj's own GEMM step,
// split out of q_proj_site.hlsl (now bias+requant-only) into its own multi-group dispatch. See
// `down_proj_gemm_site.hlsl`'s own header comment for the full correctness account, including its
// own "second turn" note -- this file gets the identical 64-thread-group treatment for the
// identical reason (hidden_size=1536 output at 256 threads/group is only 6 groups).
//
// Dispatched Dispatch(ceil(hidden_size/64), 1, 1) (superslm_gpu.cpp).
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

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int hidden_size = (int)g_hidden_size;
    int out_channels = hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);

    uint off_weight = layer_base + Layout.Load<uint>(2 * 4);
    uint off_id = layer_base + Layout.Load<uint>(3 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(4 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(5 * 4);

    int stride = ((out_channels + 63) / 64) * 64;
    GemmParallelGpu(dtid.x, LayerScratch, normed_off, LayerWeights, off_weight, LayerWeights, off_id,
                     LayerWeights, off_mult, LayerWeights, off_shift, hidden_size, out_channels,
                     WorkScratch, 0u, stride);
}
