// T-2045 (C3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): the
// ratified softmax site (site 6 of 16), split out of T-2039's own four-site
// attention fusion (attention_site.hlsl, now retired) -- see
// attention_score_site.hlsl's own header comment for the full rationale.
//
// Per distinct kv_head (recomputed per owning query head rather than cached,
// matching this build's own established "recompute rather than cache"
// choice at every other site with a per-kv_head derivation): §4.5's C30
// derivation (IExpScaleConstants over CombineCarriedScale(q_scale,
// khead_scale)), then CheckSoftmaxRowWidthDomain. Then SoftmaxRowQ15BufGpu
// runs IN PLACE over LayerScratch's own persistent `scores` region --
// attention_score_site.hlsl's own scores become this dispatch's own probs,
// read by context_accumulate_site.hlsl next.
//
// Rejections across heads are combined deterministically via a groupshared
// InterlockedMin keyed on head index (matching T-2039's own fused
// disposition), so the reported status always matches the FIRST rejecting
// head in program order -- the same status CPU's own sequential per-head
// loop would report.
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

static const uint kSoftmaxPackedNone = 0xFFFFFFFFu;
groupshared uint gSoftmaxRejectPacked;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    int width = (int)g_width;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;

    uint q_scale_off = ScratchLayout.Load<uint>(3 * 4);
    int64_t q_scale_m = LayerScratch.Load<int64_t>(q_scale_off + 0);
    int64_t q_scale_e = LayerScratch.Load<int64_t>(q_scale_off + 8);

    uint off_khead_m = layer_base + Layout.Load<uint>(54 * 4);
    uint off_khead_e = layer_base + Layout.Load<uint>(55 * 4);
    int64_t ln2_q = ModelConstants.Load<int64_t>(0);
    int64_t b_q = ModelConstants.Load<int64_t>(8);
    int64_t ca_q = ModelConstants.Load<int64_t>(16);

    uint scores_base_all = ScratchLayout.Load<uint>(25 * 4);

    if (t == 0) gSoftmaxRejectPacked = kSoftmaxPackedNone;
    GroupMemoryBarrierWithGroupSync();

    for (uint h = t; h < g_num_attention_heads; h += 256u)
    {
        uint kv_head = h / max(group, 1u);

        int64_t sm_m = LayerWeights.Load<int64_t>(off_khead_m + kv_head * 8u);
        int64_t sm_e = LayerWeights.Load<int64_t>(off_khead_e + kv_head * 8u);
        bool q_in = q_scale_m >= -2147483648LL && q_scale_m <= 2147483647LL;
        bool k_in = sm_m >= -2147483648LL && sm_m <= 2147483647LL;
        if (!q_in || !k_in)
        {
            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagCarriedScaleMantissaOutOfDomain);
            continue;
        }
        int64_t sm_comb_m, sm_comb_e;
        CombineCarriedScaleGpu_(q_scale_m, q_scale_e, sm_m, sm_e, sm_comb_m, sm_comb_e);
        if (sm_comb_m < -2147483648LL || sm_comb_m > 2147483647LL)
        {
            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagCarriedScaleMantissaOutOfDomain);
            continue;
        }
        int64_t d_q_ln2, d_q_b, d_q_c;
        int scale_domain = IExpScaleConstantsGpu(sm_comb_m, sm_comb_e, ln2_q, 30, b_q, 30, ca_q, 30,
                                                   d_q_ln2, d_q_b, d_q_c);
        if (scale_domain != 0)
        {
            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagIExpScaleDerivationOutOfDomain);
            continue;
        }
        if (!CheckSoftmaxRowWidthDomainGpu(d_q_b, d_q_c, width))
        {
            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagSoftmaxRowWidthOutOfDomain);
            continue;
        }

        uint my_scores_base = scores_base_all + h * g_context_cap * 8u;
        bool well_formed = SoftmaxRowQ15BufGpu(LayerScratch, my_scores_base, width, d_q_ln2, d_q_b, d_q_c);
        if (!well_formed)
        {
            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagSoftmaxKernelRefusedAfterGateAccepted);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gSoftmaxRejectPacked != kSoftmaxPackedNone)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, (int64_t)(gSoftmaxRejectPacked & 15u));
    }
}
