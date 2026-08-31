// T-2045 (C3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): the
// ratified ctx_fold site (site 8 of 16), split out of T-2039's own four-site
// attention fusion (attention_site.hlsl, now retired) -- see
// attention_score_site.hlsl's own header comment for the full rationale.
//
// ONE funnel call (empty incoming) over the whole ctx_wide row
// context_accumulate_site.hlsl just wrote into WorkScratch's own WIDE_A
// region -- ctx_fold's own site constant, matching the composed pipeline's
// every other RequantChainCheckedFull call site.
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index;
    uint g_hidden_size;
    uint g_head_dim;
    uint g_num_kv_heads;
    uint g_context_cap;
    uint g_position;
    uint g_num_attention_heads;
    uint g_width;
    uint g_intermediate_size;
    // T-2432 (Track A step 9, new site found by this build -- not named in the design's own
    // §2.5 census, SSLM-GEOMETRY-SITE: GS-19): this dispatch has no `g_num_hidden_layers`/
    // `g_gemm_lanes` fields (it is not a GEMM-split site), so g_q_width lands at position 9,
    // not 11 -- matching this shader's own 9-field cbuffer, one field earlier than
    // q_proj_gemm_site.hlsl/o_proj_gemm_site.hlsl's own 12th-of-12 position. The host's shared
    // `bind_and_dispatch` lambda writes Q_WIDTH at consts[11] for EVERY plain dispatch
    // regardless of a given shader's own field count (D3D12 does not require a PSO to consume
    // every root parameter its shared root signature declares) -- but this shader's own 9-field
    // struct would read consts[9] as g_q_width if declared naively at the end, which is
    // g_intermediate_size's own slot, not Q_WIDTH's. Two padding fields close that gap.
    uint g_unused9; uint g_unused10;
    uint g_q_width;
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
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint off_ctx_site = layer_base + Layout.Load<uint>(33 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_ctx_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_ctx_site + 8);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int zz = 0; zz < kMaxIncoming; ++zz) { incoming_m[zz] = 0; incoming_e[zz] = 0; }

    uint ctx_codes_off = ScratchLayout.Load<uint>(5 * 4);
    uint ctx_scale_off = ScratchLayout.Load<uint>(6 * 4);
    int64_t status_tag;
    // SSLM-GEOMETRY-SITE: GS-19
    // T-2432 (Track A step 9): the ctx_wide row's own width is q_width (Q's real output
    // width), not hidden_size -- the CPU-side analog of this exact fix, forward_sites.cpp GS-12.
    int q_width = (int)g_q_width;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, q_width, incoming_m, incoming_e, 0, site_m, site_e,
                                 LayerScratch, ctx_codes_off, ctx_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
