// T-2035: attention (sites 5-8: score, softmax, context-accumulate, ctx_fold)
// -- one combined real dispatch. CPU carries no separate named site for this
// composition either (forward_sites.cpp:1587-1703, "No named site for this
// composition exists anywhere in this tree; this is where it is first
// composed") -- combining the four sub-steps into one dispatch changes no
// observable result (no persistent K/V write happens mid-attention, so no
// cross-dispatch visibility hazard exists to barrier against) and matches
// this project's own kv_proj fusion precedent for a CPU-atomic unit.
//
// Per distinct kv_head: §4.5's C30 derivation (IExpScaleConstants over
// CombineCarriedScale(q_scale, softmax_khead)), then CheckSoftmaxRowWidthDomain.
// Per query head: score = q_rot . K_row (GemmInt8AccumulateRow shape) ->
// SoftmaxRowQ15 -> context-accumulate (GemmProbQ15Accumulate) -> the per-head
// ctx_fold dispatch (ApplyWeightScaleFold). Then ONE funnel call (empty
// incoming) over the whole ctx_wide row -- site 8, ctx_fold's own site
// constant.
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
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    int hidden_size = (int)g_hidden_size;
    int head_dim = (int)g_head_dim;
    int width = (int)g_width;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;

    int64_t q_scale_m = LayerScratch.Load<int64_t>(80);
    int64_t q_scale_e = LayerScratch.Load<int64_t>(88);

    uint off_khead_m = layer_base + Layout.Load<uint>(54 * 4);
    uint off_khead_e = layer_base + Layout.Load<uint>(55 * 4);

    // Derive (q_ln2, q_b, q_c) per distinct kv_head, on first sight (matching
    // CPU's own khead_derived[] cache), then reuse for every query head that
    // shares it.
    int64_t khead_q_ln2[kMaxSiteN], khead_q_b[kMaxSiteN], khead_q_c[kMaxSiteN];
    bool khead_derived[kMaxSiteN];
    [unroll]
    for (int zi = 0; zi < kMaxSiteN; ++zi) { khead_q_ln2[zi] = 0; khead_q_b[zi] = 0; khead_q_c[zi] = 0; khead_derived[zi] = false; }

    int64_t ln2_q = ModelConstants.Load<int64_t>(0);
    int64_t b_q = ModelConstants.Load<int64_t>(8);
    int64_t ca_q = ModelConstants.Load<int64_t>(16);

    int64_t ctx_wide[kMaxSiteN];
    [unroll]
    for (int zw = 0; zw < kMaxSiteN; ++zw) ctx_wide[zw] = 0;

    for (uint h = 0; h < g_num_attention_heads; ++h)
    {
        uint kv_head = h / max(group, 1u);
        if (!khead_derived[kv_head])
        {
            int64_t sm_m = LayerWeights.Load<int64_t>(off_khead_m + kv_head * 8u);
            int64_t sm_e = LayerWeights.Load<int64_t>(off_khead_e + kv_head * 8u);
            bool q_in = q_scale_m >= -2147483648LL && q_scale_m <= 2147483647LL;
            bool k_in = sm_m >= -2147483648LL && sm_m <= 2147483647LL;
            if (!q_in || !k_in)
            {
                SeqState.Store<int64_t>(72, kTagCarriedScaleMantissaOutOfDomain);
                return;
            }
            int64_t sm_comb_m, sm_comb_e;
            CombineCarriedScaleGpu_(q_scale_m, q_scale_e, sm_m, sm_e, sm_comb_m, sm_comb_e);
            if (sm_comb_m < -2147483648LL || sm_comb_m > 2147483647LL)
            {
                SeqState.Store<int64_t>(72, kTagCarriedScaleMantissaOutOfDomain);
                return;
            }
            int64_t d_q_ln2, d_q_b, d_q_c;
            int scale_domain = IExpScaleConstantsGpu(sm_comb_m, sm_comb_e, ln2_q, 30, b_q, 30, ca_q, 30,
                                                       d_q_ln2, d_q_b, d_q_c);
            if (scale_domain != 0)
            {
                SeqState.Store<int64_t>(72, kTagIExpScaleDerivationOutOfDomain);
                return;
            }
            khead_q_ln2[kv_head] = d_q_ln2;
            khead_q_b[kv_head] = d_q_b;
            khead_q_c[kv_head] = d_q_c;
            khead_derived[kv_head] = true;
        }

        if (!CheckSoftmaxRowWidthDomainGpu(khead_q_b[kv_head], khead_q_c[kv_head], width))
        {
            SeqState.Store<int64_t>(72, kTagSoftmaxRowWidthOutOfDomain);
            return;
        }

        // score[k] = q_rot[h] . K_row(kv_head, position=k), k in [0, width).
        uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, g_head_dim);
        int64_t scores[kMaxWidthGpu];
        [unroll]
        for (int zs = 0; zs < kMaxWidthGpu; ++zs) scores[zs] = 0;
        int values_flat[kMaxWidthGpu * kMaxSiteN];
        [unroll]
        for (int zv = 0; zv < kMaxWidthGpu * kMaxSiteN; ++zv) values_flat[zv] = 0;
        for (int k = 0; k < width; ++k)
        {
            uint k_row_off = KvRowOffsetWithinHalfGpu(g_context_cap, g_head_dim, kv_head, (uint)k);
            int64_t dot = 0;
            for (int d = 0; d < head_dim; ++d)
            {
                int qv = (int)LayerScratch.Load<int>(96 + (h * g_head_dim + (uint)d) * 4u);
                int kv = LoadSignedByteGpu(KvCache, kv_half_off + k_row_off + (uint)d);
                dot += (int64_t)qv * (int64_t)kv;
            }
            scores[k] = dot;

            uint v_half_off = kv_half_off + g_context_cap * g_num_kv_heads * g_head_dim;
            uint v_row_off = KvRowOffsetWithinHalfGpu(g_context_cap, g_head_dim, kv_head, (uint)k);
            for (int d2 = 0; d2 < head_dim; ++d2)
            {
                values_flat[k * kMaxSiteN + d2] = LoadSignedByteGpu(KvCache, v_half_off + v_row_off + (uint)d2);
            }
        }

        int64_t probs[kMaxWidthGpu];
        bool well_formed = SoftmaxRowQ15Gpu(scores, width, khead_q_ln2[kv_head], khead_q_b[kv_head],
                                             khead_q_c[kv_head], probs);
        if (!well_formed)
        {
            SeqState.Store<int64_t>(72, kTagSoftmaxKernelRefusedAfterGateAccepted);
            return;
        }

        int64_t ctx_acc[kMaxSiteN];
        GemmProbQ15AccumulateGpu(probs, values_flat, width, head_dim, ctx_acc);

        uint off_ctx_id = layer_base + Layout.Load<uint>(30 * 4);
        uint off_ctx_mult = layer_base + Layout.Load<uint>(31 * 4);
        uint off_ctx_shift = layer_base + Layout.Load<uint>(32 * 4);
        int cid = (int)LayerWeights.Load<int>(off_ctx_id + h * 4u);
        int cmult = (int)LayerWeights.Load<int>(off_ctx_mult + h * 4u);
        int cshift = (int)LayerWeights.Load<int>(off_ctx_shift + h * 4u);
        for (int d3 = 0; d3 < head_dim; ++d3)
        {
            ctx_wide[h * (uint)head_dim + (uint)d3] = ApplyWeightScaleFoldGpu(ctx_acc[d3], cid, cmult, cshift);
        }
    }

    uint off_ctx_site = layer_base + Layout.Load<uint>(33 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_ctx_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_ctx_site + 8);
    int64_t incoming_m[kMaxSiteN], incoming_e[kMaxSiteN];
    [unroll]
    for (int zz = 0; zz < kMaxSiteN; ++zz) { incoming_m[zz] = 0; incoming_e[zz] = 0; }

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(ctx_wide, hidden_size, incoming_m, incoming_e, 0, site_m, site_e, out_codes,
                                out_scale_m, out_scale_e, status_tag);
    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }
    for (int ci = 0; ci < hidden_size; ++ci) LayerScratch.Store<int>(192 + ci * 4, (int)out_codes[ci]);
    LayerScratch.Store<int64_t>(224, out_scale_m);
    LayerScratch.Store<int64_t>(232, out_scale_e);
}
