// QK norm: Q retains its carried-scale funnel. Fused-QK K remains wide through
// RMSNorm and RoPE, then lands once using QKC1's per-channel image.
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0) { uint g_layer_index, g_hidden_size, g_head_dim, g_num_kv_heads; uint g_context_cap, g_position, g_num_attention_heads, g_width, g_intermediate_size; };
ByteAddressBuffer LayerWeights : register(t0); ByteAddressBuffer Layout : register(t1); ByteAddressBuffer RopeInfo : register(t2); ByteAddressBuffer ModelConstants : register(t3); ByteAddressBuffer SiluLut : register(t4); ByteAddressBuffer RopeCosTable : register(t5); ByteAddressBuffer RopeSinTable : register(t6); ByteAddressBuffer ScratchLayout : register(t7);
RWByteAddressBuffer SeqState : register(u0); RWByteAddressBuffer LayerScratch : register(u1); RWByteAddressBuffer KvCache : register(u2); RWByteAddressBuffer WorkScratch : register(u3);
static const int kNormFracBitsGpu = 16;
groupshared uint gQkNormClamps;

bool RopeApplyPairWideGpu(int64_t x, int64_t y, int cos_q30, int sin_q30, out int64_t out_x, out int64_t out_y) {
    S128 xr = SSub(SMul(x, (int64_t)cos_q30), SMul(y, (int64_t)sin_q30));
    S128 yr = SAdd(SMul(x, (int64_t)sin_q30), SMul(y, (int64_t)cos_q30));
    S128 rx = RoundingDivideByPOTWideGpu(xr, kRopeFracBitsGpu), ry = RoundingDivideByPOTWideGpu(yr, kRopeFracBitsGpu);
    if (!SFitsI64(rx) || !SFitsI64(ry)) { out_x = 0; out_y = 0; return false; }
    out_x = SLow64(rx); out_y = SLow64(ry); return true;
}

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint t = gtid.x; int hidden_size = (int)g_hidden_size; uint sticky_off = SeqStickyOffGpu(hidden_size);
    if (SeqState.Load<int64_t>(sticky_off) != kTagOk) return;
    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4), group = gid.x;
    bool is_q = group < g_num_attention_heads; uint head = is_q ? group : group - g_num_attention_heads;
    if (!is_q && head >= g_num_kv_heads) return;
    int head_dim = (int)g_head_dim; uint present_off = is_q ? Layout.Load<uint>(57 * 4) : Layout.Load<uint>(60 * 4);
    if (LayerWeights.Load<int64_t>(layer_base + present_off) == 0) return;
    uint gain_off = layer_base + (is_q ? Layout.Load<uint>(58 * 4) : Layout.Load<uint>(61 * 4));
    uint q_codes_off = ScratchLayout.Load<uint>(2 * 4), q_scale_off = ScratchLayout.Load<uint>(3 * 4);
    uint wide_base = ScratchLayout.Load<uint>(27 * 4) + group * (uint)head_dim * 8u;
    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, (uint)head_dim);
    uint k_row_off = kv_half_off + KvRowOffsetWithinHalfGpu(g_context_cap, (uint)head_dim, head, g_position), q_row_off = q_codes_off + head * (uint)head_dim * 4u;
    int64_t sumsq;
    if (is_q) sumsq = RmsSumSqParallelGpu(t, LayerScratch, q_row_off, head_dim);
    else { int64_t local = 0; for (int i = (int)t; i < head_dim; i += 256) { int64_t v = LoadSignedByteGpu(KvCache, k_row_off + (uint)i); local += v * v; } gSumSq[t] = local; GroupMemoryBarrierWithGroupSync(); for (uint a = 128; a > 0; a >>= 1) { if (t < a) gSumSq[t] += gSumSq[t + a]; GroupMemoryBarrierWithGroupSync(); } sumsq = gSumSq[0]; }
    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu), (int64_t)head_dim)); root = root > 1 ? root : 1;
    for (int i = (int)t; i < head_dim; i += 256) { int64_t hv = is_q ? (int64_t)LayerScratch.Load<int>(q_row_off + (uint)i * 4u) : (int64_t)LoadSignedByteGpu(KvCache, k_row_off + (uint)i); int gain = (int)LayerWeights.Load<int>(gain_off + (uint)i * 4u); WorkScratch.Store<int64_t>(wide_base + (uint)i * 8u, FloorDivI64Gpu(hv << (2 * kNormFracBitsGpu), root) * (int64_t)gain); }
    DeviceMemoryBarrierWithGroupSync();
    if (is_q) {
        int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming]; [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
        int64_t tag; uint site_off = layer_base + Layout.Load<uint>(59 * 4);
        RequantChainCheckedFullGpuP(t, WorkScratch, wide_base, head_dim, incoming_m, incoming_e, 0, LayerWeights.Load<int64_t>(site_off), LayerWeights.Load<int64_t>(site_off + 8), LayerScratch, q_row_off, q_scale_off + head * 16u, tag);
        if (tag != kTagOk && t == 0) SeqState.Store<int64_t>(sticky_off, tag); return;
    }
    if (!CheckPositionOverCapGpu((int64_t)g_position, (int64_t)g_context_cap)) { if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagPositionOverCap); return; }
    uint cos_present = RopeInfo.Load(0), sin_present = RopeInfo.Load(4), pairs = g_head_dim / 2u; uint64_t cos_count = RopeInfo.Load<uint64_t>(8), sin_count = RopeInfo.Load<uint64_t>(16);
    if (cos_present == 0u || sin_present == 0u) { if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagRopeTableTensorMissing); return; }
    if (pairs == 0u || pairs > cos_count || pairs > sin_count || g_position >= cos_count / pairs || g_position >= sin_count / pairs) { if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagRopeTableExtentExceeded); return; }
    if (t == 0) gQkNormClamps = 0; AllMemoryBarrierWithGroupSync();
    uint qkc_base = layer_base + Layout.Load<uint>(65 * 4) + head * (uint)head_dim * 8u, qkc_block = g_num_kv_heads * (uint)head_dim * 8u;
    // The compact shader layout places GpuLayerLayout::off[62] in slot 63.
    uint source_off = layer_base + Layout.Load<uint>(63 * 4); int64_t wide_m = LayerWeights.Load<int64_t>(source_off), wide_e = LayerWeights.Load<int64_t>(source_off + 8u) - kRopeFracBitsGpu;
    uint row_offset = g_position * pairs;
    for (uint p = t; p < pairs; p += 256) {
        int64_t rx, ry; int c = (int)RopeCosTable.Load<int64_t>((row_offset + p) * 8u), s = (int)RopeSinTable.Load<int64_t>((row_offset + p) * 8u);
        if (!RopeApplyPairWideGpu(WorkScratch.Load<int64_t>(wide_base + 2u * p * 8u), WorkScratch.Load<int64_t>(wide_base + (2u * p + 1u) * 8u), c, s, rx, ry)) { if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagChainInputOutOfDomain); return; }
        bool clamp0, mag0, clamp1, mag1;
        int64_t raw0 = LandingRescaleGpu(rx, wide_m, LayerWeights.Load<int64_t>(qkc_base + 2u * p * 8u), wide_e, LayerWeights.Load<int64_t>(qkc_base + qkc_block + 2u * p * 8u), clamp0, mag0);
        int64_t raw1 = LandingRescaleGpu(ry, wide_m, LayerWeights.Load<int64_t>(qkc_base + (2u * p + 1u) * 8u), wide_e, LayerWeights.Load<int64_t>(qkc_base + qkc_block + (2u * p + 1u) * 8u), clamp1, mag1);
        if (clamp0) InterlockedAdd(gQkNormClamps, 1u); if (clamp1) InterlockedAdd(gQkNormClamps, 1u);
        StoreSignedByteGpu(KvCache, k_row_off + 2u * p, (int)ClampRopeCodeGpu(raw0)); StoreSignedByteGpu(KvCache, k_row_off + 2u * p + 1u, (int)ClampRopeCodeGpu(raw1));
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0 && gQkNormClamps != 0) { uint old_lo; SeqState.InterlockedAdd(SeqSatLoOffGpu(hidden_size), gQkNormClamps, old_lo); if (old_lo + gQkNormClamps < old_lo) { uint old_hi; SeqState.InterlockedAdd(SeqSatHiOffGpu(hidden_size), 1u, old_hi); } uint old_site_lo; SeqState.InterlockedAdd(SeqKNormedLandingSatLoOffGpu(hidden_size), gQkNormClamps, old_site_lo); if (old_site_lo + gQkNormClamps < old_site_lo) { uint old_site_hi; SeqState.InterlockedAdd(SeqKNormedLandingSatHiOffGpu(hidden_size), 1u, old_site_hi); } }
}
