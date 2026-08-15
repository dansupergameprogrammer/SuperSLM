// T-2039 (real-capacity shader geometry), T-2045 C1 fix (Claude/Poirot/
// 82cfca7-gpu-serial-port-build-review.md): RoPE -- real GPU dispatch, site 4
// of 16. Bit-exact port of forward_sites.cpp's RopeApplySite
// (forward_sites.cpp:639-737) in full, plus the per-head Q/K composition
// RunLayerLoopImpl drives it with (forward_sites.cpp:1546-1585) -- Q rotated
// into LayerScratch's own q_rot slot (never in-place -- Sec4: Q's own output
// is never landed), K read from the just-landed KV cache row, rotated, and
// written back to the SAME row (option_g_fused_k_landing is always false at
// this design's build target, D-SLM2994).
//
// C1: K's own write-back is now the design's own ruled STAGED shape
// (Sec5.6, D-SLM2993) -- "the RoPE dispatch computes every head's rotation
// (Q and K) into per-thread-group scratch first, synchronizes... and commits
// the rotated K rows into the persistent K store" -- not the prior
// in-loop read-rotate-write, which raced across every query head sharing one
// K row under grouped query attention (`group = num_attention_heads /
// num_key_value_heads`, e.g. 6 at the real 1.5B-Instruct tier): a thread
// whose read landed after another thread's write would rotate an
// already-rotated row. CPU's own two-loop separation
// (forward_sites.cpp:1546-1585) is the model, and its own comment states the
// property this shape restores: "every read above happens before any write
// here, so no partially-rotated store is ever observed." Phase 1 below reads
// every OWNED head's pre-rotation K bytes into a per-thread WorkScratch
// slice (ROPE_STAGE, `superslm_gpu.cpp`'s own `work_rope_stage_off`); a
// `DeviceMemoryBarrierWithGroupSync()` then forces every thread's phase-1
// KvCache reads to complete before any thread's phase-2 KvCache write
// becomes possible; phase 2 rotates from the STAGED bytes (never re-reading
// KvCache) and writes back. Multiple heads sharing a kv_head still stage,
// rotate, and write redundantly (matching CPU's own "redundant but sound"),
// but every one computes the identical rotation from the identical
// pre-rotation input, so the redundant writes -- now safely ordered after
// every read -- converge on the same bytes regardless of which thread's
// commit lands last.
//
// Rebuilt to the design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), attention heads partitioned across the group via
// N-per-thread striding (each thread owns a disjoint subset of heads and
// runs that head's own head_dim/2-pair rotation loop to completion), driven
// entirely by the real g_head_dim/g_num_attention_heads root constants.
//
// T-2113 (B4, design Sec3/Sec6.1, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 6): two changes, both of the same class as the GEMM sites' own
// multi-group split.
//
// (1) FLATTENED. Every phase below iterated `for (h = t; h < g_num_attention_heads; h +=
//     256)` -- twelve of 256 threads doing all the work at the real 1.5B tier, in one group.
//     The real independent work item is a (head, rotation pair): num_attention_heads *
//     head_dim/2 of them, each rotating one disjoint (x,y) pair. Flattened over that and
//     gridded.
//
// (2) The K COMMIT phase moved to its OWN dispatch (rope_commit_site.hlsl). The
//     DeviceMemoryBarrierWithGroupSync that used to separate stage from commit is
//     GROUP-scoped, so it orders only the 256 threads of one group -- going multi-group
//     under it would be unsound, which is precisely why the split is a second dispatch
//     rather than a wider one. The global UAV barrier `bind_and_dispatch` issues between the
//     two dispatches is strictly stronger than the group barrier it replaces: it orders
//     EVERY thread of the stage dispatch before EVERY thread of the commit dispatch, which
//     is the read-before-write property D-SLM2993's own shape exists to establish. It is
//     also what makes this site time-slice invariant -- the two dispatches may land in
//     different command lists submitted in different slices with a fence between them and
//     the bytes do not change, which was NOT true of a group barrier.
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

    uint q_codes_off = ScratchLayout.Load<uint>(2 * 4);
    uint q_rot_off = ScratchLayout.Load<uint>(4 * 4);

    // T-2049 (N2/N6, Claude/Poirot/34ef30f-gpu-serial-port-confirmation-review.md):
    // ROPE_STAGE -- host-computed and read directly from ScratchLayout index
    // 26 (`work_rope_stage_off`), never re-derived in HLSL (N6 -- the prior
    // shape recomputed it from a retired ATTN_SCORES offset, a two-place
    // invariant held only by comment). One head_dim-sized (int32-per-element)
    // slice per HEAD, not per thread (N2): the confirmation review reproduced
    // the prior per-thread indexing silently corrupting a K row once a single
    // thread owns more than one head (`num_attention_heads > 256`) -- every
    // owned head after the first overwrote the one shared slice, so only the
    // LAST owned head's bytes survived to phase 2. Indexing by the head `h`
    // itself removes the dependency on thread ownership entirely: every
    // head's own slice is independent of which thread stages/commits it, at
    // any head count.
    uint rope_stage_base = ScratchLayout.Load<uint>(26 * 4);

    // Q: read this head's codes from scratch, rotate, write the rotated
    // result to q_rot -- never committed to persistent state (Sec4: Q's own
    // output is never landed), so Q needs no staging: every head's own
    // q_codes/q_rot slots are disjoint, and no thread ever reads another
    // thread's q_rot.
    uint items = g_num_attention_heads * pairs;
    if (t >= items) return;
    {
        uint h = t / pairs;
        {
            uint p = t % pairs;
            int x = (int)LayerScratch.Load<int>(q_codes_off + (h * (uint)g_head_dim + 2u * p) * 4u);
            int y = (int)LayerScratch.Load<int>(q_codes_off + (h * (uint)g_head_dim + 2u * p + 1u) * 4u);
            int cos_q30 = (int)RopeCosTable.Load<int64_t>((row_offset + p) * 8u);
            int sin_q30 = (int)RopeSinTable.Load<int64_t>((row_offset + p) * 8u);
            int64_t rx, ry;
            RopeApplyPairGpu(x, y, cos_q30, sin_q30, rx, ry);
            LayerScratch.Store<int>(q_rot_off + (h * (uint)g_head_dim + 2u * p) * 4u, (int)ClampRopeCodeGpu(rx));
            LayerScratch.Store<int>(q_rot_off + (h * (uint)g_head_dim + 2u * p + 1u) * 4u, (int)ClampRopeCodeGpu(ry));
        }
    }

    // K, phase 1 (STAGE): every owned head reads its own kv_head's CURRENT
    // (pre-rotation) row into that HEAD's own ROPE_STAGE slice (N2). No
    // thread writes to KvCache anywhere in this phase. T-2113 (B4): the COMMIT
    // phase that used to sit below a DeviceMemoryBarrierWithGroupSync here is its
    // own dispatch now (rope_commit_site.hlsl) -- the read-before-write ordering
    // comes from the global UAV barrier between the two dispatches instead of a
    // group-scoped barrier, strictly stronger and multi-group safe.
    {
        uint h1 = t / pairs;
        uint kv_head1 = h1 / max(group, 1u);
        uint row_off1 = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)g_head_dim, kv_head1, g_position);
        uint stage1 = rope_stage_base + h1 * (uint)g_head_dim * 4u;
        {
            uint p1 = t % pairs;
            int kx = LoadSignedByteGpu(KvCache, kv_half_off + row_off1 + 2u * p1);
            int ky = LoadSignedByteGpu(KvCache, kv_half_off + row_off1 + 2u * p1 + 1u);
            WorkScratch.Store<int>(stage1 + 2u * p1 * 4u, kx);
            WorkScratch.Store<int>(stage1 + (2u * p1 + 1u) * 4u, ky);
        }
    }
}
