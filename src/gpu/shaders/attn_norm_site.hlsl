// T-2039 (real-capacity shader geometry): attn_norm -- real GPU dispatch,
// site 1 of 16. Bit-exact port of forward_sites.cpp's RmsNormSite
// (forward_sites.cpp:410-446): sumsq -> ISqrt(FloorDivI64(sumsq<<32,
// hidden_size)) -> root=max(root,1) -> per-element FloorDivI64(h[i]<<32,
// root)*g[i] -> RequantChainCheckedFull with the incoming span EMPTY.
//
// Rebuilt to the design's own production dispatch geometry (Claude/
// Vitruvius/t1986-gpu-serial-port-design-2026-08-13.md Sec5.4/Sec5.5):
// numthreads(256,1,1), sumsq via the SCHEME-1 groupshared tree
// (RmsSumSqParallelGpu), the wide row streamed through WorkScratch (never a
// per-call local array), driven entirely by the real g_hidden_size root
// constant -- the identical shader dispatches correctly at the T-2019
// suite's own hidden_size=2 fixture and at the real 1.5B/0.5B artifacts'
// hidden_size=1536/896.
//
// Every dispatch this design issues begins by reading the sequence-level
// sticky word (Sec5.6) -- a dispatch that reads Reject performs no
// arithmetic and writes nothing. This is the composed pipeline's FIRST
// dispatch each layer, so it is also where a fresh layer's own arithmetic
// begins (a prior layer's rejection already latched the sticky word, and
// this dispatch's own guard below is what may latch it for THIS layer).
#include "site_common.hlsli"

// b0: [layer_index, hidden_size, head_dim, num_kv_heads, context_cap, position]
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

static const int kNormFracBitsGpu = 16;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;  // a prior dispatch already rejected -- no arithmetic, no writes

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);  // Layout[56] = kLayerStride

    int64_t sumsq = RmsSumSqParallelGpu(t, SeqState, 0u, hidden_size);
    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu), (int64_t)hidden_size));
    root = (root > 1) ? root : 1;

    uint off_gain = layer_base + Layout.Load<uint>(0 * 4);
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int64_t hv = (int64_t)(int)SeqState.Load<int>((uint)i * 4u);
        int g = (int)LayerWeights.Load<int>(off_gain + (uint)i * 4u);
        int64_t wv = FloorDivI64Gpu(hv << (2 * kNormFracBitsGpu), root) * (int64_t)g;
        WorkScratch.Store<int64_t>(0u + (uint)i * 8u, wv);
    }
    DeviceMemoryBarrierWithGroupSync();

    uint off_site = layer_base + Layout.Load<uint>(1 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxIncoming];
    int64_t incoming_e[kMaxIncoming];
    [unroll]
    for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, /*n_incoming=*/0,
                                 site_m, site_e, LayerScratch, normed_off, normed_scale_off, status_tag);

    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
