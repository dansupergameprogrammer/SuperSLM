// GPU per-token overhead plan rev 3.1 §3.3 (S3, the merged-GEMM variant), bracket mode only:
// gate_proj's and up_proj's GEMM steps in ONE multi-group dispatch, kv_proj_gemm_site.hlsl's own
// construction (plan §2 item 8) applied to the MLP's two input projections. Per-site mode keeps
// gate_proj_gemm_site.hlsl and up_proj_gemm_site.hlsl unchanged.
//
// GATE AND UP IN ONE DISPATCH. The dispatch covers 2*intermediate_size logical output channels:
// channel c < I is gate's own channel c (weights Layout[37..40], wide row at WorkScratch base 0,
// exactly gate_proj_gemm_site.hlsl's), and c >= I is up's own channel c - I (weights
// Layout[42..45], wide row at WIDE_B = ScratchLayout[23], where fused_gate_up_site.hlsl's up body
// reads it -- per-site, up_proj_gemm_site.hlsl writes base 0 instead, after the gate tail has
// consumed gate's row there). Both GEMMs read the same input (mlp_norm's `normed` row,
// ScratchLayout[0]) with the same in_channels (hidden_size).
//
// WHY EACH CHANNEL'S VALUE IS UNCHANGED. Each channel is reduced by its own `lanes` threads with
// GemmCoalescedGpuAt's int64 groupshared tree, as in its own GEMM dispatch; the host asserts both
// plans use the same lanes (32 for both, ComputeGpuGemmSiteGroupPlan), so `g_gemm_lanes` covers
// both halves. Integer addition is associative, so the tree returns the same int64 as the serial
// sum (site_common.hlsli's header comment on GemmCoalescedGpuAt, D-SLM3334). Which group a channel
// lands in changes; what it computes does not.
//
// WHAT IT COSTS (plan §2 item 16): both calls are made by EVERY thread of every group, because each
// contains group-wide barriers and so may not be predicated. A thread whose channel belongs to the
// other half passes an out-of-range channel index, contributes zero and stores nothing (the
// past-the-end case GemmCoalescedGpuAt already handles for a ragged final group), but still runs
// that call's reduction tree. B0.7's s2gt arm prices this; the tail-only variant
// (up_proj_gemm_wide_b_site.hlsl) is the fallback.
//
// HOST CONTRACT: bind_and_dispatch's 12 constants (index 8 = intermediate_size, index 10 =
// g_gemm_lanes = the gate plan's lanes, equal to the up plan's) and
// Dispatch(ceil(2*I / (256 / lanes)), 1, 1). The sticky early return below is uniform across the
// group (one buffer word, read before any write in this dispatch, and GEMM sites never write it).
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    // The 10th and 11th root constants, as in every *_gemm_site.hlsl: `g_gemm_lanes` is the ONE
    // source of this dispatch's lane split -- the host computes the group count from the SAME
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
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    // Each half's own output width: gate's and up's are both intermediate_size.
    int half_channels = (int)g_intermediate_size;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint wide_b_off = ScratchLayout.Load<uint>(23 * 4);

    uint off_gw = layer_base + Layout.Load<uint>(37 * 4);
    uint off_gid = layer_base + Layout.Load<uint>(38 * 4);
    uint off_gmult = layer_base + Layout.Load<uint>(39 * 4);
    uint off_gshift = layer_base + Layout.Load<uint>(40 * 4);
    uint off_uw = layer_base + Layout.Load<uint>(42 * 4);
    uint off_uid = layer_base + Layout.Load<uint>(43 * 4);
    uint off_umult = layer_base + Layout.Load<uint>(44 * 4);
    uint off_ushift = layer_base + Layout.Load<uint>(45 * 4);

    uint lanes = g_gemm_lanes;
    uint channels_per_group = 256u / lanes;
    uint c_local = t / lanes;
    int c = (int)(gid.x * channels_per_group + c_local);

    // Both calls are made by EVERY thread of the group (each contains group-wide barriers, so
    // neither may be predicated); kv_proj_gemm_site.hlsl's own construction.
    int j_gate = c;
    int j_up = c - half_channels;
    GemmCoalescedGpuAt(t, j_gate, LayerScratch, normed_off, LayerWeights, off_gw, LayerWeights, off_gid,
                        LayerWeights, off_gmult, LayerWeights, off_gshift, hidden_size,
                        half_channels, WorkScratch, 0u, lanes);
    GemmCoalescedGpuAt(t, j_up, LayerScratch, normed_off, LayerWeights, off_uw, LayerWeights, off_uid,
                        LayerWeights, off_umult, LayerWeights, off_ushift, hidden_size,
                        half_channels, WorkScratch, wide_b_off, lanes);
}
