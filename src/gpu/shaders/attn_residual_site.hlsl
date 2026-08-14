// T-2039 (real-capacity shader geometry): attn_residual -- real GPU
// dispatch, site 10 of 16. Bit-exact port of forward_sites.cpp's
// ResidualReconcileSite (forward_sites.cpp:847-965) at the attn_residual
// call site (forward_sites.cpp:1719-1723): branch=o_codes/o_scale,
// stream=seq.hidden_codes/hidden_scale (SeqState -- the pre-layer residual,
// since this dispatch runs before the layer's own commit). Rebuilt to the
// design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), N-per-thread striding over hidden_size, the wide row
// streamed through WorkScratch, the per-channel domain check cooperatively
// ORed into a shared flag (matching ApplyBiasReconcileRowGpuP's own
// check-all-then-commit shape).
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
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

groupshared uint gResidualMagOutOfDomain;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint stream_scale_off = SeqScaleOffGpu(hidden_size);

    int64_t stream_m = SeqState.Load<int64_t>(stream_scale_off + 0);
    int64_t stream_e = SeqState.Load<int64_t>(stream_scale_off + 8);
    if (stream_m < -2147483648LL || stream_m > 2147483647LL)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagCarriedScaleMantissaOutOfDomain);
        return;
    }
    int64_t r_h = DynamicScaleReciprocalSharedGpu(stream_m);

    uint o_codes_off = ScratchLayout.Load<uint>(7 * 4);
    uint o_scale_off = ScratchLayout.Load<uint>(8 * 4);
    int64_t branch_m = LayerScratch.Load<int64_t>(o_scale_off + 0);
    int64_t branch_e = LayerScratch.Load<int64_t>(o_scale_off + 8);

    if (t == 0) gResidualMagOutOfDomain = 0;
    GroupMemoryBarrierWithGroupSync();
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int branch_code = LayerScratch.Load<int>(o_codes_off + (uint)i * 4u);
        bool would_clamp, magnitude_exceeded;
        int64_t reconciled = LandingRescaleGpu((int64_t)branch_code, branch_m, r_h, branch_e, stream_e,
                                                 would_clamp, magnitude_exceeded);
        if (magnitude_exceeded)
        {
            InterlockedOr(gResidualMagOutOfDomain, 1u);
        }
        else
        {
            int stream_code = SeqState.Load<int>((uint)i * 4u);
            WorkScratch.Store<int64_t>(0u + (uint)i * 8u, reconciled + (int64_t)stream_code);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gResidualMagOutOfDomain != 0)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagResidualReconciliationMagnitudeOutOfDomain);
        return;
    }

    uint off_site = layer_base + Layout.Load<uint>(34 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = stream_m; incoming_e[0] = stream_e;

    uint attn_stream_off = ScratchLayout.Load<uint>(9 * 4);
    uint attn_stream_scale_off = ScratchLayout.Load<uint>(10 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, 1, site_m,
                                 site_e, LayerScratch, attn_stream_off, attn_stream_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
