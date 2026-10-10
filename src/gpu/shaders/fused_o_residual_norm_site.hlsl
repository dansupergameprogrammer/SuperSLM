// GPU per-token overhead plan rev 3.1 §3.2 (S2), bracket mode only: o tail · attn_residual ·
// mlp_norm in ONE single-group dispatch, replacing three consecutive Dispatch(1,1,1) calls and the
// two global UAV barriers between them. Per-site mode keeps the three separate dispatches
// (o_proj_site.hlsl, attn_residual_site.hlsl, mlp_norm_site.hlsl), which run the SAME body text
// from fused_site_bodies.hlsli.
//
// WHY IT IS THE SAME ARITHMETIC (plan §3.2). Each body runs in one 256-thread group, exactly as it
// does as its own dispatch, and reads no SV_GroupID. Between consecutive bodies,
// DeviceMemoryBarrierWithGroupSync() makes every thread of the group wait for the others and
// orders the group's UAV writes before its own later reads -- the pattern the bodies already use
// internally (mlp_norm writes WorkScratch, syncs, then requants it). Nothing crosses groups, so
// the global barrier this replaces ordered nothing the in-group barrier does not.
//   - Every `return` in a body is uniform across the group (the S2/S3 shader record argues each
//     one), so all 256 threads reach every barrier below.
//   - Each body keeps its own entry sticky check: a rejection stored by one body (thread 0, then
//     a uniform return) is visible to the next body's sticky load after the barrier, so the later
//     bodies return before any write, exactly as the next dispatch does today (plan §7 cells 5.2,
//     7.6, 11.1).
//   - tests/ci/check_fused_site_barriers.py (cell 7.7) fails the build if a barrier between two
//     body calls in this `main` is removed.
//
// ROOT CONSTANTS. The tail lambda's layout (`bind_and_dispatch_tail`, superslm_gpu.cpp): 0-10 the
// shared fields, 11-18 o's adapter slot (slot_p 1, in_base = ctx_codes, wide_base = WIDE_A) --
// the same constants, in the same positions, that o_proj_site.hlsl reads today. attn_residual and
// mlp_norm read only 0-8, which the tail lambda writes with the same values bind_and_dispatch does.
// Index 9 is named g_num_hidden_layers (commit_site.hlsl's name; both lambdas write N there), index
// 10 is the lambda's lanes slot (written 1, unread). Bindings: t0-t9 and u0-u3, the union of the
// three sites' own (all three declare the same t0-t7/u0-u3; o adds t8/t9).
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    uint g_num_hidden_layers; uint g_lanes_unused10;
    // o's own adapter-delta slot (slot_p 1, positions 11-18), o_proj_site.hlsl's field names.
    uint g_adapter_rank;
    uint g_adapter_a_offset;
    uint g_adapter_b_offset;
    uint g_adapter_fold_offset;
    uint g_adapter_u_off;
    uint g_adapter_in_base;
    uint g_adapter_wide_base;
    uint g_adapter_stage1_lanes;
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
ByteAddressBuffer LoraAB : register(t8);
ByteAddressBuffer Fold   : register(t9);

#define SSLM_FUSED_BODY_O_PROJ_SITE
#define SSLM_FUSED_BODY_ATTN_RESIDUAL_SITE
#define SSLM_FUSED_BODY_MLP_NORM_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    OTailSiteBody(t);
    DeviceMemoryBarrierWithGroupSync();
    AttnResidualSiteBody(t);
    DeviceMemoryBarrierWithGroupSync();
    MlpNormSiteBody(t);
}
