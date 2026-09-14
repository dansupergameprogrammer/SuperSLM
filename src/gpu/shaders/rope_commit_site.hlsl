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

// (D-SLM6263, external review Significant 1): this group's own accumulated count of K's
// RopeApplyPairGpu rotations RopeApplySite's own [-127,127] clamp actually clamped -- flushed
// into SeqState's split (sat_lo, sat_hi) accumulator below, the SAME host-facing
// SslmDecodeStepStatus::saturation_count every other saturating site already feeds. A
// distinct groupshared variable per compiled shader file -- rope_guard_site.hlsl's own
// gRopeGuardClamps (Q's rotation) lives in a different compiled program.
groupshared uint gRopeCommitClamps;

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
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

    // (D-SLM6263): the zero-init and its barrier run for EVERY thread of every group,
    // uniformly, before the `t < items` branch below -- the identical reasoning
    // rope_guard_site.hlsl's own comment states: `items` is uniform but `t` is not, so a
    // barrier placed after a per-thread `t >= items` return would not be reached by every
    // thread of a partially-filled last group.
    if (gtid.x == 0) gRopeCommitClamps = 0;
    GroupMemoryBarrierWithGroupSync();

    if (LayerWeights.Load<int64_t>(g_layer_index * Layout.Load<uint>(56 * 4) + Layout.Load<uint>(60 * 4)) == 0 && t < items) {
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
            // (D-SLM6263, external review Significant 1): RopeApplySite's own post-rotation
            // clamp, counted -- the identical predicated-increment CPU's LandingRescale
            // convention already uses, applied here to K's own committed rotation, which
            // previously reported no saturation signal at all (the exact defect Significant 1
            // names: "the sequence saturation counter is updated by LandingRescale, but the
            // subsequent RoPE clamp has no counter").
            //
            // CORRECTED 2026-09-03 (T-2577, D-SLM6281, external review `Claude/Poirot/
            // 5fafd98-t2573-trackb-external-fold-review.md` Observation 1): `h2` ranges over
            // QUERY heads (`items = g_num_attention_heads * pairs`, above), but this dispatch's
            // own K row is addressed by `kv_head2 = h2 / group` -- every query head sharing one
            // KV head redundantly recomputes the IDENTICAL rotation from the IDENTICAL staged
            // input (this file's own header comment: "redundant but sound") and converges on the
            // identical written bytes. The write is harmless to repeat; the COUNT is not -- a
            // saturating component was counted once per query head, `group` times too many
            // (twice, on the real candidate's 2:1 grouping). `only_representative_head` is true
            // for exactly the FIRST query head of each KV-head's own group -- gates the count,
            // never the rotation or the write, so every thread still does its own identical,
            // redundant-but-sound work; only the double bookkeeping is removed.
            bool only_representative_head = (h2 % max(group, 1u)) == 0u;
            if (only_representative_head && (rkx < -127 || rkx > 127)) InterlockedAdd(gRopeCommitClamps, 1u);
            if (only_representative_head && (rky < -127 || rky > 127)) InterlockedAdd(gRopeCommitClamps, 1u);
            StoreSignedByteGpu(KvCache, kv_half_off + row_off2 + 2u * p2, (int)ClampRopeCodeGpu(rkx));
            StoreSignedByteGpu(KvCache, kv_half_off + row_off2 + 2u * p2 + 1u, (int)ClampRopeCodeGpu(rky));
        }
    }

    // (D-SLM6263): flush this group's own accumulated clamp count into SeqState's SAME split
    // (sat_lo, sat_hi) accumulator every other saturating site already feeds. Reached by every
    // thread uniformly, outside the `t < items` branch, matching rope_guard_site.hlsl's own
    // identical discipline.
    GroupMemoryBarrierWithGroupSync();
    if (gtid.x == 0 && gRopeCommitClamps != 0) {
        uint sat_lo_off = SeqSatLoOffGpu(hidden_size);
        uint sat_hi_off = SeqSatHiOffGpu(hidden_size);
        uint old_lo;
        SeqState.InterlockedAdd(sat_lo_off, gRopeCommitClamps, old_lo);
        if (old_lo + gRopeCommitClamps < old_lo) {
            uint old_hi;
            SeqState.InterlockedAdd(sat_hi_off, 1u, old_hi);
        }
        // (T-2577, D-SLM6280): the identical flush, a second time, into this site's OWN
        // per-site slot -- "rope_k" (RoPE's K rotation, this shader) -- alongside the aggregate
        // flush immediately above, never in place of it. This is the reading Significant 1's own
        // enclosure proof watches: `rope_k` clamps must stay zero on the recalibrated candidate
        // at width > 1.
        uint rk_sat_lo_off = SeqRopeKSatLoOffGpu(hidden_size);
        uint rk_sat_hi_off = SeqRopeKSatHiOffGpu(hidden_size);
        uint rk_old_lo;
        SeqState.InterlockedAdd(rk_sat_lo_off, gRopeCommitClamps, rk_old_lo);
        if (rk_old_lo + gRopeCommitClamps < rk_old_lo) {
            uint rk_old_hi;
            SeqState.InterlockedAdd(rk_sat_hi_off, 1u, rk_old_hi);
        }
    }
}
