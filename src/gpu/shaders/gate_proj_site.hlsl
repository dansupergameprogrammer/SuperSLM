// T-2039 (real-capacity shader geometry): gate_proj -- real GPU dispatch,
// site 12 of 16. ProjectAndFunnel, no bias, input=normed (mlp_norm's own
// output), out_channels=intermediate_size, weight=gate_weight (Layout[37]).
// Rebuilt to the design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), N-per-thread striding over intermediate_size output
// channels (non-uniform where it does not divide 256 evenly, the T-2014
// correction's own split), the wide row streamed through WorkScratch.
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
    int out_channels = (int)g_intermediate_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(normed_scale_off + 0);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(normed_scale_off + 8);

    uint off_weight = layer_base + Layout.Load<uint>(37 * 4);
    uint off_id = layer_base + Layout.Load<uint>(38 * 4);
    uint off_mult = layer_base + Layout.Load<uint>(39 * 4);
    uint off_shift = layer_base + Layout.Load<uint>(40 * 4);
    uint off_site = layer_base + Layout.Load<uint>(41 * 4);

    GemmParallelGpu(t, LayerScratch, normed_off, LayerWeights, off_weight, LayerWeights, off_id,
                     LayerWeights, off_mult, LayerWeights, off_shift, hidden_size, out_channels,
                     WorkScratch, 0u);
    DeviceMemoryBarrierWithGroupSync();

    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = in_scale_m; incoming_e[0] = in_scale_e;
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    uint gate_codes_off = ScratchLayout.Load<uint>(11 * 4);
    uint gate_scale_off = ScratchLayout.Load<uint>(12 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, out_channels, incoming_m, incoming_e, 1, site_m,
                                 site_e, LayerScratch, gate_codes_off, gate_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
