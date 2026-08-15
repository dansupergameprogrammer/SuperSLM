// T-2045 (C3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): the
// ratified context-accumulate site (site 7 of 16), split out of T-2039's own
// four-site attention fusion (attention_site.hlsl, now retired) -- see
// attention_score_site.hlsl's own header comment for the full rationale.
//
// Per owned head: ctx_acc[d] = sum_k probs[h][k] * V(kv_head, k, d)
// (GemmProbQ15Accumulate's own shape), folded per-head via
// ApplyWeightScaleFoldGpu, written into WorkScratch's own WIDE_A region at
// h*head_dim+d -- read next by ctx_fold_site.hlsl's own funnel over the
// whole hidden_size row. A pure weighted sum plus a fold; neither can
// reject, so this site carries no guard and no sticky write of its own
// beyond the standard top-of-dispatch check.
//
// T-2113 (B4, design Sec3/Sec6.1, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 6): FLATTENED and GRIDDED, the identical class of change as
// attention_score_site.hlsl's own header comment describes. The real independent work item
// is one (head, head_dim element) pair -- num_attention_heads * head_dim of them, each an
// independent width-long reduction over its own V column. The host dispatches
// ceil(num_attention_heads * head_dim / 256) groups. There is no cross-thread state and no
// barrier, so a thread holding no item simply returns.
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

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint t = dtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    int head_dim = (int)g_head_dim;
    int width = (int)g_width;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;

    uint scores_base_all = ScratchLayout.Load<uint>(25 * 4);
    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, (uint)g_head_dim);
    uint v_half_off = kv_half_off + g_context_cap * g_num_kv_heads * (uint)g_head_dim;

    uint off_ctx_id = layer_base + Layout.Load<uint>(30 * 4);
    uint off_ctx_mult = layer_base + Layout.Load<uint>(31 * 4);
    uint off_ctx_shift = layer_base + Layout.Load<uint>(32 * 4);

    // One work item per (head, head_dim element): num_attention_heads * head_dim of them,
    // each an independent width-long reduction over its own V column. The host dispatches
    // ceil(num_attention_heads * head_dim / 256) groups.
    uint items = g_num_attention_heads * (uint)head_dim;
    if (t >= items) return;
    {
        uint h = t / (uint)head_dim;
        int d = (int)(t % (uint)head_dim);
        uint kv_head = h / max(group, 1u);
        uint my_scores_base = scores_base_all + h * g_context_cap * 8u;

        int cid = (int)LayerWeights.Load<int>(off_ctx_id + h * 4u);
        int cmult = (int)LayerWeights.Load<int>(off_ctx_mult + h * 4u);
        int cshift = (int)LayerWeights.Load<int>(off_ctx_shift + h * 4u);

        {
            int64_t ctx_acc_d = 0;
            for (int k = 0; k < width; ++k)
            {
                uint v_row_off = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)g_head_dim, kv_head, (uint)k);
                int64_t p = LayerScratch.Load<int64_t>(my_scores_base + (uint)k * 8u);
                int vv = LoadSignedByteGpu(KvCache, v_half_off + v_row_off + (uint)d);
                ctx_acc_d += p * (int64_t)vv;
            }
            int64_t folded = ApplyWeightScaleFoldGpu(ctx_acc_d, cid, cmult, cshift);
            WorkScratch.Store<int64_t>(0u + (h * (uint)g_head_dim + (uint)d) * 8u, folded);
        }
    }
}
