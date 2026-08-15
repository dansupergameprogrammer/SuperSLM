// T-2113 (B4, design Sec3/Sec6.1, Sec10 B4): RoPE's own COMMIT phase, split out of
// rope_guard_site.hlsl into its own dispatch -- see that file's own header comment for the
// full split rationale. Re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-2026-08-14.md
// Sec2 change 6.
//
// K, phase 2 (COMMIT): rotate from the STAGED bytes (WorkScratch's own ROPE_STAGE region,
// written by rope_guard_site.hlsl's own phase 1, immediately before this dispatch in the
// per-layer chain) -- never re-reading KvCache -- and write back. The global UAV barrier
// `bind_and_dispatch` issues between the two dispatches guarantees every phase-1 write is
// visible to every phase-2 thread before this dispatch's first instruction runs -- strictly
// stronger than the DeviceMemoryBarrierWithGroupSync the fused version relied on (that
// barrier only ordered the 256 threads of ONE group; this construction is multi-group safe).
// Heads sharing a kv_head each redundantly recompute the identical rotation from the
// identical staged input and converge on the identical written bytes (CPU's own "redundant
// but sound" construction, forward_sites.cpp), and StoreSignedByteGpu's CAS is what makes
// two threads owning adjacent bytes of one word safe -- unchanged from the fused version.
//
// FLATTENED and GRIDDED over the real work item space: num_attention_heads * head_dim/2
// (head, rotation pair) items, identical to rope_guard_site.hlsl's own flatten.
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

    // T-2113 (B4): the stage dispatch (rope_guard_site.hlsl) already ran this identical
    // guard ladder and sets the sticky tag on rejection -- this dispatch's own sticky check
    // above already returns on that tag for every position after a rejecting layer, so these
    // re-checks are structurally unreachable in the fixed per-layer dispatch order. Re-run
    // and re-written anyway (idempotent -- the same tag, written twice, is not a distinct
    // observable outcome) rather than assumed unreachable by construction: this dispatch
    // reads RopeInfo/g_position independently and must not silently diverge from the stage
    // dispatch's own verdict if a future change ever lets the two disagree.
    if (!CheckPositionOverCapGpu((int64_t)g_position, (int64_t)g_context_cap))
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagPositionOverCap);
        return;
    }

    uint cos_present = RopeInfo.Load(0);
    uint sin_present = RopeInfo.Load(4);
    if (cos_present == 0u || sin_present == 0u)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagRopeTableTensorMissing);
        return;
    }

    uint64_t cos_elem_count = RopeInfo.Load<uint64_t>(8);
    uint64_t sin_elem_count = RopeInfo.Load<uint64_t>(16);
    uint pairs = g_head_dim / 2u;
    if (pairs == 0u || (uint64_t)pairs > cos_elem_count || (uint64_t)pairs > sin_elem_count)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagRopeTableExtentExceeded);
        return;
    }
    uint64_t cos_rows = cos_elem_count / (uint64_t)pairs;
    uint64_t sin_rows = sin_elem_count / (uint64_t)pairs;
    if ((uint64_t)g_position >= cos_rows || (uint64_t)g_position >= sin_rows)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagRopeTableExtentExceeded);
        return;
    }

    uint row_offset = g_position * pairs;
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;
    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, (uint)g_head_dim);
    uint rope_stage_base = ScratchLayout.Load<uint>(26 * 4);

    uint items = g_num_attention_heads * pairs;
    if (t >= items) return;
    {
        uint h2 = t / pairs;
        uint kv_head2 = h2 / max(group, 1u);
        uint row_off2 = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)g_head_dim, kv_head2, g_position);
        uint stage2 = rope_stage_base + h2 * (uint)g_head_dim * 4u;
        {
            uint p2 = t % pairs;
            int kx = WorkScratch.Load<int>(stage2 + 2u * p2 * 4u);
            int ky = WorkScratch.Load<int>(stage2 + (2u * p2 + 1u) * 4u);
            int cos_q30b = (int)RopeCosTable.Load<int64_t>((row_offset + p2) * 8u);
            int sin_q30b = (int)RopeSinTable.Load<int64_t>((row_offset + p2) * 8u);
            int64_t rkx, rky;
            RopeApplyPairGpu(kx, ky, cos_q30b, sin_q30b, rkx, rky);
            StoreSignedByteGpu(KvCache, kv_half_off + row_off2 + 2u * p2, (int)ClampRopeCodeGpu(rkx));
            StoreSignedByteGpu(KvCache, kv_half_off + row_off2 + 2u * p2 + 1u, (int)ClampRopeCodeGpu(rky));
        }
    }
}
