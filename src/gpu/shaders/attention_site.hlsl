// T-2039 (real-capacity shader geometry): attention (sites 5-8: score,
// softmax, context-accumulate, ctx_fold) -- one combined real dispatch. CPU
// carries no separate named site for this composition either
// (forward_sites.cpp:1587-1703). Per distinct kv_head: C30's IExpScaleConstants
// derivation, then CheckSoftmaxRowWidthDomain. Per query head: score = q_rot
// . K_row (GemmInt8AccumulateRow shape) -> SoftmaxRowQ15 -> context-accumulate
// (GemmProbQ15Accumulate) -> the per-head ctx_fold dispatch
// (ApplyWeightScaleFold). Then ONE funnel call (empty incoming) over the
// whole ctx_wide row -- site 8, ctx_fold's own site constant.
//
// Rebuilt to the design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), attention heads (an already-independent production
// unit) partitioned across the group via N-per-thread striding -- each
// owning thread runs its own head's score/softmax/context-accumulate to
// completion using a PRIVATE per-thread slice of WorkScratch's own
// ATTN_SCORES region (sized to the real per-call context_cap, never a
// compile-time capacity), then writes its owned ctx_wide entries into
// WorkScratch's WIDE_A region. Per-kv_head scale-constant derivation is
// recomputed by every owning thread rather than cached in a per-call local
// array (cheap fixed-point arithmetic, no cross-thread cost). Rejections
// across heads are combined deterministically via a groupshared
// InterlockedMin keyed on head index, so the reported status always matches
// the FIRST rejecting head in program order -- the same status CPU's own
// sequential per-head loop would report.
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

static const uint kAttnPackedNone = 0xFFFFFFFFu;
groupshared uint gAttnRejectPacked;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    int head_dim = (int)g_head_dim;
    int width = (int)g_width;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;

    uint q_rot_off = ScratchLayout.Load<uint>(4 * 4);
    uint attn_scores_base_all = ScratchLayout.Load<uint>(24 * 4);

    int64_t q_scale_m = LayerScratch.Load<int64_t>(ScratchLayout.Load<uint>(3 * 4) + 0);
    int64_t q_scale_e = LayerScratch.Load<int64_t>(ScratchLayout.Load<uint>(3 * 4) + 8);

    uint off_khead_m = layer_base + Layout.Load<uint>(54 * 4);
    uint off_khead_e = layer_base + Layout.Load<uint>(55 * 4);
    int64_t ln2_q = ModelConstants.Load<int64_t>(0);
    int64_t b_q = ModelConstants.Load<int64_t>(8);
    int64_t ca_q = ModelConstants.Load<int64_t>(16);

    uint off_ctx_id = layer_base + Layout.Load<uint>(30 * 4);
    uint off_ctx_mult = layer_base + Layout.Load<uint>(31 * 4);
    uint off_ctx_shift = layer_base + Layout.Load<uint>(32 * 4);
    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, (uint)g_head_dim);
    uint v_half_off = kv_half_off + g_context_cap * g_num_kv_heads * (uint)g_head_dim;

    if (t == 0) gAttnRejectPacked = kAttnPackedNone;
    GroupMemoryBarrierWithGroupSync();

    for (uint h = t; h < g_num_attention_heads; h += 256u)
    {
        uint kv_head = h / max(group, 1u);

        // Per-distinct-kv_head derivation (recomputed per owning head rather
        // than cached -- cheap, and every thread only ever owns a disjoint
        // subset of heads).
        int64_t sm_m = LayerWeights.Load<int64_t>(off_khead_m + kv_head * 8u);
        int64_t sm_e = LayerWeights.Load<int64_t>(off_khead_e + kv_head * 8u);
        bool q_in = q_scale_m >= -2147483648LL && q_scale_m <= 2147483647LL;
        bool k_in = sm_m >= -2147483648LL && sm_m <= 2147483647LL;
        if (!q_in || !k_in)
        {
            uint packed = h * 16u + (uint)kTagCarriedScaleMantissaOutOfDomain;
            InterlockedMin(gAttnRejectPacked, packed);
            continue;
        }
        int64_t sm_comb_m, sm_comb_e;
        CombineCarriedScaleGpu_(q_scale_m, q_scale_e, sm_m, sm_e, sm_comb_m, sm_comb_e);
        if (sm_comb_m < -2147483648LL || sm_comb_m > 2147483647LL)
        {
            uint packed = h * 16u + (uint)kTagCarriedScaleMantissaOutOfDomain;
            InterlockedMin(gAttnRejectPacked, packed);
            continue;
        }
        int64_t d_q_ln2, d_q_b, d_q_c;
        int scale_domain = IExpScaleConstantsGpu(sm_comb_m, sm_comb_e, ln2_q, 30, b_q, 30, ca_q, 30,
                                                   d_q_ln2, d_q_b, d_q_c);
        if (scale_domain != 0)
        {
            uint packed = h * 16u + (uint)kTagIExpScaleDerivationOutOfDomain;
            InterlockedMin(gAttnRejectPacked, packed);
            continue;
        }

        if (!CheckSoftmaxRowWidthDomainGpu(d_q_b, d_q_c, width))
        {
            uint packed = h * 16u + (uint)kTagSoftmaxRowWidthOutOfDomain;
            InterlockedMin(gAttnRejectPacked, packed);
            continue;
        }

        // score[k] = q_rot[h] . K_row(kv_head, position=k), k in [0, width).
        uint my_scores_base = attn_scores_base_all + t * g_context_cap * 8u;
        for (int k = 0; k < width; ++k)
        {
            uint k_row_off = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)g_head_dim, kv_head, (uint)k);
            int64_t dot = 0;
            for (int d = 0; d < head_dim; ++d)
            {
                int qv = (int)LayerScratch.Load<int>(q_rot_off + (h * (uint)g_head_dim + (uint)d) * 4u);
                int kv = LoadSignedByteGpu(KvCache, kv_half_off + k_row_off + (uint)d);
                dot += (int64_t)qv * (int64_t)kv;
            }
            WorkScratch.Store<int64_t>(my_scores_base + (uint)k * 8u, dot);
        }

        bool well_formed = SoftmaxRowQ15BufGpu(WorkScratch, my_scores_base, width, d_q_ln2, d_q_b, d_q_c);
        if (!well_formed)
        {
            uint packed = h * 16u + (uint)kTagSoftmaxKernelRefusedAfterGateAccepted;
            InterlockedMin(gAttnRejectPacked, packed);
            continue;
        }

        int cid = (int)LayerWeights.Load<int>(off_ctx_id + h * 4u);
        int cmult = (int)LayerWeights.Load<int>(off_ctx_mult + h * 4u);
        int cshift = (int)LayerWeights.Load<int>(off_ctx_shift + h * 4u);
        for (int d3 = 0; d3 < head_dim; ++d3)
        {
            int64_t ctx_acc_d = 0;
            for (int k2 = 0; k2 < width; ++k2)
            {
                uint v_row_off = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)g_head_dim, kv_head, (uint)k2);
                int64_t p = WorkScratch.Load<int64_t>(my_scores_base + (uint)k2 * 8u);
                int vv = LoadSignedByteGpu(KvCache, v_half_off + v_row_off + (uint)d3);
                ctx_acc_d += p * (int64_t)vv;
            }
            int64_t folded = ApplyWeightScaleFoldGpu(ctx_acc_d, cid, cmult, cshift);
            WorkScratch.Store<int64_t>(0u + (h * (uint)g_head_dim + (uint)d3) * 8u, folded);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();

    uint packed_result = gAttnRejectPacked;
    if (packed_result != kAttnPackedNone)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, (int64_t)(packed_result & 15u));
        return;
    }

    uint off_ctx_site = layer_base + Layout.Load<uint>(33 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_ctx_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_ctx_site + 8);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int zz = 0; zz < kMaxIncoming; ++zz) { incoming_m[zz] = 0; incoming_e[zz] = 0; }

    uint ctx_codes_off = ScratchLayout.Load<uint>(5 * 4);
    uint ctx_scale_off = ScratchLayout.Load<uint>(6 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, 0, site_m, site_e,
                                 LayerScratch, ctx_codes_off, ctx_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
