// GPU per-token overhead plan rev 3.1 §3.3 (S3), bracket mode only: gate tail · up tail in ONE
// single-group dispatch. Per-site mode keeps gate_proj_site.hlsl and up_proj_site.hlsl, which run
// the SAME body text from fused_site_bodies.hlsli. mlp_act (one 1024-thread group) stays its own
// dispatch after this one.
//
// WHERE UP'S ACCUMULATOR LIVES. Per-site, the gate tail requants gate's wide row at WorkScratch
// base 0 and is done with it before the up GEMM overwrites base 0 with up's row. Here both rows
// must be live at once, so up's row is at WIDE_B (ScratchLayout[23] = work_wide_b_off, sized
// max(H, I, q_width) x 8 bytes, superslm_gpu.cpp), written there by either gate_up_gemm_site.hlsl
// (the merged GEMM) or up_proj_gemm_wide_b_site.hlsl (the tail-only variant). This is the kv
// tail's V-half arrangement exactly (kv_proj_site.hlsl): the up body is called with
//   - accumulator base WIDE_B, read from ScratchLayout[23] as kv_proj_site.hlsl reads `wide_b_off`
//     (per-site: the literal 0u), and
//   - adapter slot 1 (constants 19-26), whose wide base the host writes as work_wide_b_off, as it
//     writes the V slot's (per-site: up's slot sits at 11-18 with wide base work_wide_a_off).
// Gate keeps WorkScratch base 0 and slot 0 (constants 11-18), exactly its per-site values.
//
// ORDER AND SYNCHRONIZATION: fused_o_residual_norm_site.hlsl's header comment applies unchanged
// (one 256-thread group per body, no SV_GroupID, DeviceMemoryBarrierWithGroupSync() between the
// two body calls, every body `return` uniform, the up body's own sticky check live after a gate
// rejection -- plan §7 cell 7.6 -- and tests/ci/check_fused_site_barriers.py, cell 7.7). Both
// adapter deltas use the shared ADAPTER_U scratch region; ApplyFusedAdapterDeltaGpu's own
// trailing DeviceMemoryBarrierWithGroupSync() plus the barrier between the bodies order gate's use
// before up's, as kv_proj_site.hlsl's K-then-V calls are ordered.
//
// ROOT CONSTANTS: the tail lambda's 28-value layout with two slots, as for the kv tail --
// 0-10 shared, 11-18 gate's slot (slot_p 2, in_base = normed, wide_base = work_wide_a_off),
// 19-26 up's slot (slot_p 3, in_base = normed, wide_base = work_wide_b_off), 27 Q_WIDTH (unread
// here). Bindings: t0-t9 and u0-u3.
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    uint g_num_hidden_layers; uint g_lanes_unused10;
    // Slot 0: gate's own adapter-delta slot (slot_p 2, positions 11-18), gate_proj_site.hlsl's
    // field names, so the gate body reads them unchanged.
    uint g_adapter_rank;
    uint g_adapter_a_offset;
    uint g_adapter_b_offset;
    uint g_adapter_fold_offset;
    uint g_adapter_u_off;
    uint g_adapter_in_base;
    uint g_adapter_wide_base;
    uint g_adapter_stage1_lanes;
    // Slot 1: up's own adapter-delta slot (slot_p 3, positions 19-26), read by the up body
    // through SSLM_UP_BODY_ADAPTER below.
    uint g_adapter_up_rank;
    uint g_adapter_up_a_offset;
    uint g_adapter_up_b_offset;
    uint g_adapter_up_fold_offset;
    uint g_adapter_up_u_off;
    uint g_adapter_up_in_base;
    uint g_adapter_up_wide_base;
    uint g_adapter_up_stage1_lanes;
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

#define SSLM_UP_BODY_ADAPTER(field) g_adapter_up_##field
#define SSLM_FUSED_BODY_GATE_PROJ_SITE
#define SSLM_FUSED_BODY_UP_PROJ_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    GateTailSiteBody(t);
    DeviceMemoryBarrierWithGroupSync();
    UpTailSiteBody(t, /*acc_base=WIDE_B=*/ScratchLayout.Load<uint>(23 * 4));
}
