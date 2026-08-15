// T-2039 (real-capacity shader geometry): q_proj -- real GPU dispatch, site 2
// of 16. Bit-exact port of forward_sites.cpp's ProjectAndFunnel
// (forward_sites.cpp:1020-1048) at the q_proj call site
// (forward_sites.cpp:1395-1400): GemmInt8AccumulateRow ->
// per-channel ApplyWeightScaleFold -> optional ApplyBiasReconcileRow (q_bias)
// -> RequantChainCheckedFull with incoming={normed_scale}.
//
// Rebuilt to the design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), the GEMM's own out_channels (hidden_size) spread
// across the group via N-per-thread striding, the wide row streamed through
// WorkScratch -- driven entirely by the real g_hidden_size root constant.
//
// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): the GEMM step (D-SLM3313's
// own fourth-largest consumer, 4.9% of GPU-busy time) now runs in its OWN multi-group dispatch,
// `q_proj_gemm_site.hlsl`, issued immediately before this one. This shader is bias+requant-only
// now -- see `down_proj_site.hlsl`'s own header comment for the full account of the split; the
// bias-reconcile step below (`ApplyBiasReconcileRowGpuP`) is unaffected -- it was already, and
// remains, a single-group cooperative reduction over the whole row, run in the SAME dispatch as
// the requant step it always ran alongside.
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index;
    uint g_hidden_size;
    uint g_head_dim;
    uint g_num_kv_heads;
    uint g_context_cap;
    uint g_position;
};

ByteAddressBuffer   LayerWeights  : register(t0);
ByteAddressBuffer   Layout        : register(t1);
ByteAddressBuffer   RopeInfo      : register(t2);
ByteAddressBuffer   ScratchLayout : register(t7);
RWByteAddressBuffer SeqState      : register(u0);
RWByteAddressBuffer LayerScratch  : register(u1);
RWByteAddressBuffer KvCache       : register(u2);
RWByteAddressBuffer WorkScratch   : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t normed_scale_m = LayerScratch.Load<int64_t>(normed_scale_off + 0);
    int64_t normed_scale_e = LayerScratch.Load<int64_t>(normed_scale_off + 8);

    uint off_site = layer_base + Layout.Load<uint>(6 * 4);
    uint off_bias_present = layer_base + Layout.Load<uint>(7 * 4);
    uint off_bias = layer_base + Layout.Load<uint>(8 * 4);

    int64_t bias_present = LayerWeights.Load<int64_t>(off_bias_present);
    if (bias_present != 0)
    {
        int64_t bias_tag;
        if (!ApplyBiasReconcileRowGpuP(t, WorkScratch, 0u, hidden_size, LayerWeights, off_bias,
                                        normed_scale_m, normed_scale_e, bias_tag))
        {
            if (t == 0) SeqState.Store<int64_t>(sticky_off, bias_tag);
            return;
        }
        DeviceMemoryBarrierWithGroupSync();
    }

    int64_t incoming_m[kMaxIncoming];
    int64_t incoming_e[kMaxIncoming];
    [unroll]
    for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = normed_scale_m;
    incoming_e[0] = normed_scale_e;

    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    // T-2035: q_codes/q_scale feed RoPE (site 4) and, through RoPE's own
    // rotated output, attention.
    uint q_codes_off = ScratchLayout.Load<uint>(2 * 4);
    uint q_scale_off = ScratchLayout.Load<uint>(3 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, /*n_incoming=*/1,
                                 site_m, site_e, LayerScratch, q_codes_off, q_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
