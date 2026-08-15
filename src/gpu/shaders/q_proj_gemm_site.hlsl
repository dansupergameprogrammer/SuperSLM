// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): q_proj's own GEMM step,
// split out of q_proj_site.hlsl (now bias+requant-only) into its own multi-group dispatch.
// T-2113 (B4, design Sec3/Sec6.1, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 3/4): the partition is TRANSPOSED -- `g_gemm_lanes` threads
// cooperate on one output channel via `GemmCoalescedGpu` (site_common.hlsli) instead of one
// thread owning a whole channel serially. See that function's own header comment for the
// full correctness/determinism account.
//
// Dispatched Dispatch(ceil(hidden_size/(256/g_gemm_lanes)), 1, 1) (superslm_gpu.cpp, the
// SAME `ComputeGpuGemmSiteGroupPlan` this dispatch's own group count comes from).
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    // T-2113 (B4): the 10th and 11th root constants. `g_gemm_lanes` is the ONE source of
    // this dispatch's own lane split -- the host computes the group count from the SAME
    // value, so the two cannot drift.
    uint g_num_hidden_layers; uint g_gemm_lanes;
};

ByteAddressBuffer   LayerWeights  : register(t0);
ByteAddressBuffer   Layout        : register(t1);
ByteAddressBuffer   RopeInfo      : register(t2);
ByteAddressBuffer   ScratchLayout : register(t7);
RWByteAddressBuffer SeqState      : register(u0);
RWByteAddressBuffer LayerScratch  : register(u1);
RWByteAddressBuffer KvCache       : register(u2);
RWByteAddressBuffer WorkScratch   : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
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

    GemmCoalescedGpu(gtid.x, gid.x, LayerScratch, normed_off, LayerWeights, off_weight, LayerWeights, off_id,
                      LayerWeights, off_mult, LayerWeights, off_shift, hidden_size, out_channels,
                      WorkScratch, 0u, g_gemm_lanes);
}
