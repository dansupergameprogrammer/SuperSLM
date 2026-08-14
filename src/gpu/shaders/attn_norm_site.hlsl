// T-2032 (Sec5.6/Sec11 B4 well-scoped next checkpoint): attn_norm -- real GPU
// dispatch, site 1 of 16. Bit-exact port of forward_sites.cpp's RmsNormSite
// (forward_sites.cpp:410-446): sumsq -> ISqrt(FloorDivI64(sumsq<<32,
// hidden_size)) -> root=max(root,1) -> per-element FloorDivI64(h[i]<<32,
// root)*g[i] -> RequantChainCheckedFull with the incoming span EMPTY.
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

ByteAddressBuffer   LayerWeights : register(t0);
ByteAddressBuffer   Layout       : register(t1);
ByteAddressBuffer   RopeInfo     : register(t2);
RWByteAddressBuffer SeqState     : register(u0);
RWByteAddressBuffer LayerScratch : register(u1);
RWByteAddressBuffer KvCache      : register(u2);

static const int kNormFracBitsGpu = 16;

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;  // a prior dispatch already rejected -- no arithmetic, no writes

    int hidden_size = (int)g_hidden_size;
    uint layer_base = g_layer_index * Layout.Load<uint>(25 * 4);  // Layout[25] = kLayerStride

    int64_t h[kMaxSiteN];
    [unroll]
    for (int z = 0; z < kMaxSiteN; ++z) h[z] = 0;
    for (int i = 0; i < hidden_size; ++i)
    {
        h[i] = (int64_t)(int)SeqState.Load<int>(0 + i * 4);  // hidden_codes[i]
    }

    uint off_gain = layer_base + Layout.Load<uint>(0 * 4);
    int g[kMaxSiteN];
    [unroll]
    for (int z2 = 0; z2 < kMaxSiteN; ++z2) g[z2] = 0;
    for (int i2 = 0; i2 < hidden_size; ++i2)
    {
        g[i2] = (int)LayerWeights.Load<int>(off_gain + (uint)i2 * 4u);
    }

    int64_t sumsq = 0;
    for (int i3 = 0; i3 < hidden_size; ++i3)
    {
        sumsq += h[i3] * h[i3];
    }
    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu), (int64_t)hidden_size));
    root = (root > 1) ? root : 1;

    int64_t wide[kMaxSiteN];
    [unroll]
    for (int z3 = 0; z3 < kMaxSiteN; ++z3) wide[z3] = 0;
    for (int i4 = 0; i4 < hidden_size; ++i4)
    {
        wide[i4] = FloorDivI64Gpu(h[i4] << (2 * kNormFracBitsGpu), root) * (int64_t)g[i4];
    }

    uint off_site = layer_base + Layout.Load<uint>(1 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    int64_t incoming_m[kMaxSiteN];
    int64_t incoming_e[kMaxSiteN];
    [unroll]
    for (int z4 = 0; z4 < kMaxSiteN; ++z4) { incoming_m[z4] = 0; incoming_e[z4] = 0; }

    int64_t out_codes[kMaxSiteN];
    int64_t out_scale_m, out_scale_e, status_tag;
    RequantChainCheckedFullGpu(wide, hidden_size, incoming_m, incoming_e, /*n_incoming=*/0, site_m,
                                site_e, out_codes, out_scale_m, out_scale_e, status_tag);

    if (status_tag != kTagOk)
    {
        SeqState.Store<int64_t>(72, status_tag);
        return;
    }

    for (int i5 = 0; i5 < hidden_size; ++i5)
    {
        LayerScratch.Store<int>(0 + i5 * 4, (int)out_codes[i5]);
    }
    LayerScratch.Store<int64_t>(32, out_scale_m);
    LayerScratch.Store<int64_t>(40, out_scale_e);
}
