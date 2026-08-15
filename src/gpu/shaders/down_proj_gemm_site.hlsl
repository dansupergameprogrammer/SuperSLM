// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): down_proj's own GEMM step,
// split out of down_proj_site.hlsl (now requant-only -- see that file's own header comment) into
// its own multi-group dispatch. D-SLM3313's own per-site table found this site's achieved
// bandwidth flat at ~1.7 GB/s regardless of matrix size, the signature of ONE 256-thread group
// covering the whole 8960-wide reduction row for every one of hidden_size=1536 output channels --
// under 10% of the RTX 2080 SUPER's own 3072 CUDA cores active per dispatch, at any model tier.
//
// Correctness constraint (the whole product claim, per this round's own commission): no change to
// any output element's computation or its summation order. Each output channel `j` is still
// computed by exactly ONE thread, reading the SAME weight row and the SAME input codes, reducing
// over `in_channels` in the SAME left-to-right order GemmParallelGpu's own body always has --
// site_common.hlsli's own body is reused unchanged, only WHICH thread (now identified by
// SV_DispatchThreadID, global across every group this dispatch launches, not
// SV_GroupThreadID, local to one group) computes which `j` changes. No cross-group reduction, no
// atomic, no reordered accumulation.
//
// T-2101 (second turn, same round): 1536 output channels at 256 threads/group is only 6 groups --
// this site's own first-round measurement (D-SLM3313's own follow-up build log) found it the
// single largest remaining consumer at 36.3% of GPU-busy time, achieving ~10.3 GB/s against the
// 35-group gate/up_proj_gemm siblings' own ~32.7 GB/s at the SAME 256-thread-per-group width --
// the gap tracks group count, not matrix size, so fewer threads per group and proportionally more
// groups (64 threads/group, 24 groups instead of 6) is the same screw turned once more: the
// per-thread computation and its accumulation order are UNCHANGED (GemmParallelGpu's own body,
// unmodified), only the group/thread partition of the identical work.
//
// T-2113 (B4, design Sec3/Sec6.1, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 3/4): the partition is TRANSPOSED via `GemmCoalescedGpu`
// (site_common.hlsli) -- `g_gemm_lanes` threads cooperate on one output channel over an
// int64 groupshared tree, replacing the single-thread-per-channel `GemmParallelGpu` this
// site used through T-2101. See that function's own header comment for the correctness and
// determinism account. Down's own measured-optimal lane count is 64 (T-2105 Sec2's own
// dispatch-geometry table), twice the other five GEMM sites' 32 -- the access pattern this
// site's own in_channels=8960 reduction walks favors more cooperating lanes per channel.
//
// Dispatched Dispatch(ceil(hidden_size/(256/g_gemm_lanes)), 1, 1) (superslm_gpu.cpp, the
// SAME `ComputeGpuGemmSiteGroupPlan` this dispatch's own group count comes from).
#include "site_common2.hlsli"

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
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    int hidden_size = (int)g_hidden_size;
    int in_channels = (int)g_intermediate_size;
    int out_channels = hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint act_codes_off = ScratchLayout.Load<uint>(15 * 4);

    uint off_weight = layer_base + Layout.Load<uint>(48 * 4);
    uint off_id = layer_base + Layout.Load<uint>(49 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(50 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(51 * 4);

    // T-2113 (B4): the group count this dispatch launched is computed HERE from
    // `out_channels`/`g_gemm_lanes` alone (never from a root constant `g_width`-style value,
    // which every OTHER site sharing this signature already uses for a different meaning, the
    // attention sites' own context width) using the IDENTICAL ceiling formula
    // `superslm_gpu.cpp`'s own host-side group-count computation uses for this dispatch's grid
    // size -- both sides derive the same number from the same `out_channels`/`lanes`, so they
    // cannot drift independently of each other.
    GemmCoalescedGpu(gtid.x, gid.x, LayerScratch, act_codes_off, LayerWeights, off_weight, LayerWeights,
                      off_id, LayerWeights, off_mult, LayerWeights, off_shift, in_channels,
                      out_channels, WorkScratch, 0u, g_gemm_lanes);
}
