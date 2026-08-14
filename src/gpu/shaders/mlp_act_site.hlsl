// T-2039 (real-capacity shader geometry): mlp_act -- real GPU dispatch, site
// 14 of 16. Bit-exact port of forward_sites.cpp's MlpActSite:
// CheckSiluCompositionScaleDomain (gate_scale) FIRST -- SiluSigmoidQ15 never
// runs on rejection -- then per-element sig[i]=SiluSigmoidQ15(LUT,
// gate_code[i], gate_scale), wide[i] = gate_code[i]*sig[i]*up_code[i],
// RequantChainCheckedFull(incoming={gate_scale, up_scale}). Rebuilt to the
// design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), N-per-thread striding over intermediate_size, the
// wide row streamed through WorkScratch.
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

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    int n = (int)g_intermediate_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint gate_codes_off = ScratchLayout.Load<uint>(11 * 4);
    uint gate_scale_off = ScratchLayout.Load<uint>(12 * 4);
    uint up_codes_off = ScratchLayout.Load<uint>(13 * 4);
    uint up_scale_off = ScratchLayout.Load<uint>(14 * 4);

    int64_t gate_m = LayerScratch.Load<int64_t>(gate_scale_off + 0);
    int64_t gate_e = LayerScratch.Load<int64_t>(gate_scale_off + 8);
    if (!CheckSiluCompositionScaleDomainGpu(gate_m, (int)gate_e))
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagSiluCompositionScaleOutOfDomain);
        return;
    }
    int64_t up_m = LayerScratch.Load<int64_t>(up_scale_off + 0);
    int64_t up_e = LayerScratch.Load<int64_t>(up_scale_off + 8);

    for (int i = (int)t; i < n; i += 256)
    {
        int gate_code = (int)LayerScratch.Load<int>(gate_codes_off + (uint)i * 4u);
        int up_code = (int)LayerScratch.Load<int>(up_codes_off + (uint)i * 4u);
        int sig = SiluSigmoidQ15Gpu(SiluLut, gate_code, gate_m, (int)gate_e);
        int64_t wv = (int64_t)gate_code * (int64_t)sig * (int64_t)up_code;
        WorkScratch.Store<int64_t>(0u + (uint)i * 8u, wv);
    }
    DeviceMemoryBarrierWithGroupSync();

    uint off_site = layer_base + Layout.Load<uint>(47 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = gate_m; incoming_e[0] = gate_e;
    incoming_m[1] = up_m; incoming_e[1] = up_e;

    uint act_codes_off = ScratchLayout.Load<uint>(15 * 4);
    uint act_scale_off = ScratchLayout.Load<uint>(16 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, n, incoming_m, incoming_e, 2, site_m, site_e,
                                 LayerScratch, act_codes_off, act_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
