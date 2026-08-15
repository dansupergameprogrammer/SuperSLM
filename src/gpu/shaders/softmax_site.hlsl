// T-2045 (C3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): the
// ratified softmax site (site 6 of 16), split out of T-2039's own four-site
// attention fusion (attention_site.hlsl, now retired) -- see
// attention_score_site.hlsl's own header comment for the full rationale.
//
// Per distinct kv_head (recomputed per owning query head rather than cached,
// matching this build's own established "recompute rather than cache"
// choice at every other site with a per-kv_head derivation): §4.5's C30
// derivation (IExpScaleConstants over CombineCarriedScale(q_scale,
// khead_scale)), then CheckSoftmaxRowWidthDomain. Then SoftmaxRowQ15BufGpuParallel
// (T-2113 B10 lever 2) runs IN PLACE over LayerScratch's own persistent `scores` region --
// attention_score_site.hlsl's own scores become this dispatch's own probs,
// read by context_accumulate_site.hlsl next.
//
// Rejections across heads are combined deterministically via a groupshared
// InterlockedMin keyed on head index (matching T-2039's own fused
// disposition), so the reported status always matches the FIRST rejecting
// head in program order -- the same status CPU's own sequential per-head
// loop would report.
//
// T-2113 (B10 lever 2, D-SLM3400 -- softmax split): per-head row processing (the max-reduce,
// the i-exp compute, the m-bound-check sum-reduce, and the Q15 normalize) now runs through
// `SoftmaxRowQ15BufGpuParallel` (site_common2.hlsli), `lanes` threads cooperating on each head's
// own row instead of one thread owning it start-to-finish -- see that function's own header
// comment for the full construction and the exactness argument. The domain-check ladder below
// (scale-mantissa, IExpScaleConstants, row-width) is UNCHANGED in shape and still runs once per
// head (redundantly across a head-group's own `lanes` threads, a pure function of the head index
// so every lane derives the identical result); only the InterlockedMin call on a rejection is
// gated to `lane == 0` to avoid redundant atomics with an identical value.
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

    // T-2113 (B10 lever 2): `lanes` threads cooperate on each head's own row instead of one
    // thread owning it start-to-finish -- the same low-occupancy shape D-SLM3398/D-SLM3399 fixed
    // for the adapter-delta stage-1 reduction. `lanes` is a power of two (the identical
    // derivation `Stage1LanesForRank`, superslm_gpu.cpp, uses host-side, computed here in-shader
    // since `g_num_attention_heads` is already a root constant); `heads_per_wave` head-groups are
    // processed per outer loop iteration, uniformly across every thread -- the loop's own trip
    // count depends only on `g_num_attention_heads`/`lanes`, never on `t`, so every thread
    // reaches the SAME number of `SoftmaxRowQ15BufGpuParallel` calls, required since that
    // function's own internal barriers must be reached by every thread on every call regardless
    // of whether that thread's own head-group is active this iteration.
    uint num_heads = g_num_attention_heads;
    uint want = (num_heads > 0u) ? (256u / num_heads) : 256u;
    if (want == 0u) want = 1u;
    uint lanes = 1u;
    while (lanes * 2u <= want) lanes *= 2u;
    uint heads_per_wave = 256u / lanes;
    uint lane = t % lanes;
    uint local_slot = t / lanes;

    for (uint h0 = 0u; h0 < num_heads; h0 += heads_per_wave)
    {
        uint h = h0 + local_slot;
        bool in_range = h < num_heads;
        uint kv_head = in_range ? (h / max(group, 1u)) : 0u;

        bool active = false;
        int64_t d_q_ln2 = 0, d_q_b = 0, d_q_c = 0;
        if (in_range)
        {
            int64_t sm_m = LayerWeights.Load<int64_t>(off_khead_m + kv_head * 8u);
            int64_t sm_e = LayerWeights.Load<int64_t>(off_khead_e + kv_head * 8u);
            bool q_in = q_scale_m >= -2147483648LL && q_scale_m <= 2147483647LL;
            bool k_in = sm_m >= -2147483648LL && sm_m <= 2147483647LL;
            if (!q_in || !k_in)
            {
                if (lane == 0u)
                    InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagCarriedScaleMantissaOutOfDomain);
            }
            else
            {
                int64_t sm_comb_m, sm_comb_e;
                CombineCarriedScaleGpu_(q_scale_m, q_scale_e, sm_m, sm_e, sm_comb_m, sm_comb_e);
                if (sm_comb_m < -2147483648LL || sm_comb_m > 2147483647LL)
                {
                    if (lane == 0u)
                        InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagCarriedScaleMantissaOutOfDomain);
                }
                else
                {
                    int scale_domain = IExpScaleConstantsGpu(sm_comb_m, sm_comb_e, ln2_q, 30, b_q, 30, ca_q, 30,
                                                               d_q_ln2, d_q_b, d_q_c);
                    if (scale_domain != 0)
                    {
                        if (lane == 0u)
                            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagIExpScaleDerivationOutOfDomain);
                    }
                    else if (!CheckSoftmaxRowWidthDomainGpu(d_q_b, d_q_c, width))
                    {
                        if (lane == 0u)
                            InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagSoftmaxRowWidthOutOfDomain);
                    }
                    else
                    {
                        active = true;
                    }
                }
            }
        }

        uint my_scores_base = scores_base_all + h * g_context_cap * 8u;
        bool well_formed = SoftmaxRowQ15BufGpuParallel(active, t, lane, lanes, LayerScratch,
                                                          my_scores_base, width, d_q_ln2, d_q_b, d_q_c);
        if (active && !well_formed)
        {
            if (lane == 0u)
                InterlockedMin(gSoftmaxRejectPacked, h * 16u + (uint)kTagSoftmaxKernelRefusedAfterGateAccepted);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gSoftmaxRejectPacked != kSoftmaxPackedNone)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, (int64_t)(gSoftmaxRejectPacked & 15u));
    }
}
