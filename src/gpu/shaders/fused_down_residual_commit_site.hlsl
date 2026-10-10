// GPU per-token overhead plan rev 3.1 §3.2 (S2), bracket mode only: down tail · mlp_residual ·
// commit in ONE single-group dispatch, replacing three consecutive Dispatch(1,1,1) calls and the
// two global UAV barriers between them. Per-site mode keeps the three separate dispatches
// (down_proj_site.hlsl, mlp_residual_site.hlsl, commit_site.hlsl), which run the SAME body text
// from fused_site_bodies.hlsli.
//
// WHY IT IS THE SAME ARITHMETIC: fused_o_residual_norm_site.hlsl's own header comment, which
// applies here unchanged -- one 256-thread group per body as today, no SV_GroupID, a
// DeviceMemoryBarrierWithGroupSync() between consecutive bodies, every `return` in a body uniform
// across the group, and each body's own entry sticky check live (a down-tail or mlp_residual
// rejection leaves commit's writes undone, as the separate commit dispatch does today: commit's
// sticky check IS the layer's "every checked call returned Ok" condition). The commit body's last
// statement is the layer commit itself; nothing follows it in this dispatch.
//
// ROOT CONSTANTS. The tail lambda's layout: 0-10 the shared fields, 11-18 down's adapter slot
// (slot_p 4, in_base = act_codes, wide_base = WIDE_A), the constants down_proj_site.hlsl reads
// today. mlp_residual reads 0-8; commit reads 0-9, and index 9 is g_num_hidden_layers, which the
// tail lambda writes as N exactly as bind_and_dispatch does (plan §2 item 9). Index 10 is the
// lambda's lanes slot (written 1, unread). Bindings: t0-t9 and u0-u3.
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    uint g_num_hidden_layers; uint g_lanes_unused10;
    // down's own adapter-delta slot (slot_p 4, positions 11-18), down_proj_site.hlsl's field names.
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

#define SSLM_FUSED_BODY_DOWN_PROJ_SITE
#define SSLM_FUSED_BODY_MLP_RESIDUAL_SITE
#define SSLM_FUSED_BODY_COMMIT_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    DownTailSiteBody(t);
    DeviceMemoryBarrierWithGroupSync();
    MlpResidualSiteBody(t);
    DeviceMemoryBarrierWithGroupSync();
    CommitSiteBody(t);
}
