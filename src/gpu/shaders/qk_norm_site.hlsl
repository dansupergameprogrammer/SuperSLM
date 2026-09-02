// T-2551 (design §2.2/§3/§4/§6 Track B step 3): QK-norm -- real GPU dispatch, inserted between
// kv_proj_site.hlsl (K/V landing) and rope_guard_site.hlsl (RoPE) in the per-layer chain. A
// bit-exact port of forward_sites.cpp's ApplyQkNormSite -- itself RmsNormSite
// (forward_sites.cpp:410-458) applied per head at width head_dim, strictly between the K/V
// landing block and the RoPE loop's first RopeApplySite call (§3's own ordering resolution: the
// runtime call site sits downstream of q_proj/k_proj's GEMM output, already in the engine's
// permuted order, so no permutation crosses this call).
//
// ONE Dispatch call issues (g_num_attention_heads + g_num_kv_heads) thread GROUPS: groups
// [0, g_num_attention_heads) each normalize ONE query head's just-produced q_codes row
// (LayerScratch, in place); groups [g_num_attention_heads, g_num_attention_heads+g_num_kv_heads)
// each normalize ONE KV head's just-landed K row (KvCache, in place) -- matching the CPU site's
// own "for each query head..., for each KV head ONCE..." construction. Every group's own 256
// threads cooperatively reduce that ONE head's own head_dim elements
// (RmsSumSqParallelGpu's own groupshared tree -- scoped per-group by GPU hardware definition, so
// the multiple groups this one Dispatch call issues never interfere with each other's own
// reduction).
//
// Gated PER LAYER on tensor presence (design §4): when this layer's own q_norm_present /
// k_norm_present flag (LayerWeights buffer, Layout indices 57/60) is 0, the corresponding half
// of every group's work is a no-op -- the host issues this dispatch unconditionally for every
// layer (superslm_gpu.cpp's own RecordOneTokenFullDepthDispatchBody), so a model with no
// q_norm/k_norm tensors at any layer pays only the cost of near-empty groups, matching every
// existing Qwen2.5 artifact's forward OUTPUT byte-for-byte (§4's own promise; dispatch count is
// not part of that promise).
//
// (carried-scale delta, `Claude/Vitruvius/t2557-trackb-carried-scale-delta-2026-09-02.md` §3/§4/
// §5, D-SLM6116/D-SLM6117/D-SLM6118 -- this contract supersedes the "converges on the identical
// q_scale value" / "K's own carried scale is discarded" text this comment carried before that
// delta): a per-head RMSNorm produces a genuinely distinct output scale per head, and this
// dispatch's own per-group addressing is what makes carrying it, per head, race-free.
//
// Q's own carried scale is written back to LayerScratch's q_scale table (ScratchLayout index 3,
// now `num_attention_heads * 16` bytes -- one 16-byte slot per query head) at THIS group's own
// disjoint slot, `q_scale_off + head*16u` -- never the single shared slot the pre-delta contract
// used. softmax_site.hlsl's own C30 derivation reads `q_scale_off + h*16u` per query head. Every
// Q group in this one Dispatch call writes only its own head's slot, which is what removes the
// C3 race identified against the pre-delta single-slot contract: no two groups ever write the
// same address, by construction of the addressing, not by the values converging.
//
// K's own carried scale is READ BACK, not discarded: after the funnel, K's post-norm codes are
// requantized a second time (LandingRescaleGpu, below) onto a new static per-(layer, KV head)
// landing scale (`k_norm_landing_r_t`/`e_t`, Layout positions 63-64) -- the identical primitive
// kv_proj_site.hlsl already uses for K's own raw pre-norm landing. This is what makes the K/V
// store's post-norm codes readable at a fixed static scale (attention_score_site.hlsl's own K
// read is still a plain packed-int8 dot product with no carried-scale operand; the scale is
// consumed at landing time, not at read time, matching every other K/V row).
//
// K's own destination (KvCache) is PACKED int8 (StoreSignedByteGpu), unlike LayerScratch's
// int32-per-code convention RequantChainCheckedFullGpuP natively writes via Store<int> -- K's own
// funnel call below targets a disjoint WorkScratch int32 staging slice (QK_NORM_K_STAGE,
// ScratchLayout index 28), never aliasing its own QK_NORM_WIDE input slice (index 27), then every
// owning thread re-reads its own staged int32 code, relands it onto the new static scale, and
// stores it packed, mirroring rope_guard_site.hlsl/rope_commit_site.hlsl's own stage-then-commit
// precedent for the identical packed-KvCache constraint.
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index;
    uint g_hidden_size;
    uint g_head_dim;
    uint g_num_kv_heads;
    uint g_context_cap;
    uint g_position;
    uint g_num_attention_heads;
};

ByteAddressBuffer   LayerWeights  : register(t0);
ByteAddressBuffer   Layout        : register(t1);
ByteAddressBuffer   ScratchLayout : register(t7);
RWByteAddressBuffer SeqState      : register(u0);
RWByteAddressBuffer LayerScratch  : register(u1);
RWByteAddressBuffer KvCache       : register(u2);
RWByteAddressBuffer WorkScratch   : register(u3);

static const int kNormFracBitsGpu = 16;

// (T-2564, S3 -- Claude/Poirot/36185a3-t2563-trackb-rebuild-review.md): the second K
// landing's own clamp count, per group -- mirrors kv_proj_site.hlsl's own gTotalClamps,
// declared separately here because each .hlsl file compiles to its own shader and does
// not share groupshared storage with another file's declaration of the same name.
groupshared uint gQkNormClamps;

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;  // a prior dispatch already rejected -- no arithmetic, no writes

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);  // Layout[56] = kLayerStride (unchanged)

    uint group = gid.x;
    bool is_q = group < g_num_attention_heads;
    uint head = is_q ? group : (group - g_num_attention_heads);
    if (!is_q && head >= g_num_kv_heads) return;  // should not occur -- host dispatches exactly the right count

    int head_dim = (int)g_head_dim;

    // Per-layer presence gate (design §4): a layer with no q_norm/k_norm tensors at all leaves
    // this group a no-op -- matches the CPU site's own lw.q_norm_gain != nullptr /
    // lw.k_norm_gain != nullptr gates exactly.
    uint present_off = is_q ? Layout.Load<uint>(57 * 4) : Layout.Load<uint>(60 * 4);
    int64_t present = LayerWeights.Load<int64_t>(layer_base + present_off);
    if (present == 0) return;

    uint gain_off = layer_base + (is_q ? Layout.Load<uint>(58 * 4) : Layout.Load<uint>(61 * 4));
    uint site_off = layer_base + (is_q ? Layout.Load<uint>(59 * 4) : Layout.Load<uint>(62 * 4));

    uint q_codes_off = ScratchLayout.Load<uint>(2 * 4);
    uint q_scale_off = ScratchLayout.Load<uint>(3 * 4);
    uint qk_norm_wide_base = ScratchLayout.Load<uint>(27 * 4);
    uint qk_norm_k_stage_base = ScratchLayout.Load<uint>(28 * 4);
    uint qk_norm_k_scale_base = ScratchLayout.Load<uint>(29 * 4);

    // This group's own disjoint QK_NORM_WIDE slice -- indexed by `group` (0..NQH+NH-1), one
    // Align8(head_dim)*8-byte slice each (work_qk_norm_wide_off's own header comment,
    // superslm_gpu.cpp).
    uint wide_base = qk_norm_wide_base + group * (uint)head_dim * 8u;

    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, (uint)head_dim);
    uint k_row_off = kv_half_off + KvRowOffsetWithinHalfGpu(g_context_cap, (uint)head_dim, head, g_position);
    uint q_row_off = q_codes_off + head * (uint)head_dim * 4u;

    // sumsq, cooperatively -- Q reads LayerScratch's int32-per-code row (RmsSumSqParallelGpu's
    // own native Load<int> contract); K reads KvCache's packed int8 row (LoadSignedByteGpu, the
    // same packed-read primitive rope_guard_site.hlsl already uses for this buffer), so K's own
    // sumsq is accumulated by hand rather than through that helper.
    int64_t sumsq;
    if (is_q) {
        sumsq = RmsSumSqParallelGpu(t, LayerScratch, q_row_off, head_dim);
    } else {
        int64_t local = 0;
        for (int i = (int)t; i < head_dim; i += 256) {
            int64_t hv = (int64_t)LoadSignedByteGpu(KvCache, k_row_off + (uint)i);
            local += hv * hv;
        }
        gSumSq[t] = local;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 128; s > 0; s >>= 1) {
            if (t < s) gSumSq[t] += gSumSq[t + s];
            GroupMemoryBarrierWithGroupSync();
        }
        sumsq = gSumSq[0];
    }

    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu), (int64_t)head_dim));
    root = (root > 1) ? root : 1;

    for (int i = (int)t; i < head_dim; i += 256) {
        int64_t hv = is_q ? (int64_t)LayerScratch.Load<int>(q_row_off + (uint)i * 4u)
                           : (int64_t)LoadSignedByteGpu(KvCache, k_row_off + (uint)i);
        int g = (int)LayerWeights.Load<int>(gain_off + (uint)i * 4u);
        int64_t wv = FloorDivI64Gpu(hv << (2 * kNormFracBitsGpu), root) * (int64_t)g;
        WorkScratch.Store<int64_t>(wide_base + (uint)i * 8u, wv);
    }
    DeviceMemoryBarrierWithGroupSync();

    int64_t site_m = LayerWeights.Load<int64_t>(site_off + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(site_off + 8);
    int64_t incoming_m[kMaxIncoming];
    int64_t incoming_e[kMaxIncoming];
    [unroll]
    for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }

    int64_t status_tag;
    if (is_q) {
        // (carried-scale delta §5, D-SLM6118): this head's own DISJOINT slot -- q_scale_off is
        // now num_attention_heads*16 bytes wide (ComputeScratchLayout, superslm_gpu.cpp), one
        // slot per query head, so this write can never race another Q group's write. Q's own
        // destination (LayerScratch) matches RequantChainCheckedFullGpuP's native int32-per-code
        // write exactly -- written in place, at the SAME q_row_off this group already fully
        // consumed above.
        uint q_scale_slot_off = q_scale_off + head * 16u;
        RequantChainCheckedFullGpuP(t, WorkScratch, wide_base, head_dim, incoming_m, incoming_e,
                                     /*n_incoming=*/0, site_m, site_e, LayerScratch, q_row_off,
                                     q_scale_slot_off, status_tag);
    } else {
        // K's own destination (KvCache) is packed int8 -- stage the funnel's native
        // int32-per-code output into this KV head's own disjoint QK_NORM_K_STAGE slice (never
        // wide_base itself, which the funnel's own internal steps still read from concurrently
        // within this same call), and its own disjoint QK_NORM_K_SCALE slot -- no longer a
        // discard target (carried-scale delta §4): read back below for the second landing.
        uint k_stage_off = qk_norm_k_stage_base + head * (uint)head_dim * 4u;
        uint k_scale_off = qk_norm_k_scale_base + head * 16u;
        RequantChainCheckedFullGpuP(t, WorkScratch, wide_base, head_dim, incoming_m, incoming_e,
                                     /*n_incoming=*/0, site_m, site_e, WorkScratch, k_stage_off,
                                     k_scale_off, status_tag);
    }

    if (status_tag != kTagOk) {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }

    if (!is_q) {
        // (carried-scale delta §4, D-SLM6117, §7 Cell 3): requantize the staged post-norm codes
        // a SECOND time, onto the new static per-(layer, KV head) landing scale -- the funnel's
        // own just-produced (m, e), read back from its own disjoint scale slot, and
        // LandingRescaleGpu, the identical primitive kv_proj_site.hlsl already uses for K's own
        // raw pre-norm landing (:199). Every owning thread re-reads its own staged int32 code,
        // relands it, and stores it packed into KvCache, in place.
        uint k_stage_off = qk_norm_k_stage_base + head * (uint)head_dim * 4u;
        uint k_scale_off = qk_norm_k_scale_base + head * 16u;
        // (T-2564, S2 -- Claude/Poirot/36185a3-t2563-trackb-rebuild-review.md): the scale
        // word at k_scale_off is written by thread 0 alone, inside
        // RequantChainCheckedFullGpuP (site_common.hlsli's own store under `if (t == 0)`),
        // and every thread of this group reads it back below. Without a barrier between
        // that store and this load, a thread reaching this point before thread 0 reaches
        // its own store reads a stale/uninitialized word -- probe-commissioned: masked on
        // this GPU/driver (0/100), reproduced 100/100 with a must-reject control. The same
        // primitive appears 46 lines above this call for the identical reason.
        // AllMemoryBarrierWithGroupSync (not DeviceMemoryBarrierWithGroupSync alone): this
        // barrier also has to make gQkNormClamps's own zero-init below (S3) visible to
        // every thread before any InterlockedAdd against it, which is a groupshared-memory
        // ordering guarantee DeviceMemoryBarrierWithGroupSync does not make.
        if (t == 0) gQkNormClamps = 0;
        AllMemoryBarrierWithGroupSync();
        int64_t normed_m = WorkScratch.Load<int64_t>(k_scale_off + 0);
        int64_t normed_e = WorkScratch.Load<int64_t>(k_scale_off + 8);
        uint off_k_norm_r_t = layer_base + Layout.Load<uint>(63 * 4);
        uint off_k_norm_e_t = layer_base + Layout.Load<uint>(64 * 4);
        int64_t r_t = LayerWeights.Load<int64_t>(off_k_norm_r_t + (uint)head * 8u);
        int64_t e_t = LayerWeights.Load<int64_t>(off_k_norm_e_t + (uint)head * 8u);
        for (int i = (int)t; i < head_dim; i += 256) {
            int64_t code = (int64_t)WorkScratch.Load<int>(k_stage_off + (uint)i * 4u);
            bool would_clamp, mag_exceeded;
            int64_t raw = LandingRescaleGpu(code, normed_m, r_t, normed_e, e_t, would_clamp, mag_exceeded);
            // (T-2564, S3 -- Claude/Poirot/36185a3-t2563-trackb-rebuild-review.md): the
            // second K landing's own clamp signal, computed and discarded before this fix
            // -- now counted the way kv_proj_site.hlsl counts its own K/V landing clamps,
            // four lines away.
            if (would_clamp) InterlockedAdd(gQkNormClamps, 1u);
            int64_t landed = ClampRopeCodeGpu(raw);
            StoreSignedByteGpu(KvCache, k_row_off + (uint)i, (int)landed);
        }
        // This dispatch issues one thread group PER KV head (unlike kv_proj_site.hlsl's
        // single group), so multiple groups can reach this flush concurrently -- a plain
        // Load-then-Store here would be the identical cross-group lost-update race
        // StoreSignedByteGpu's own header comment already documents and closes for the
        // K/V cache byte writes. InterlockedAdd on the RWByteAddressBuffer itself (the
        // established primitive, site_common.hlsli's own StoreSignedByteGpu) keeps the
        // lo-word accumulation atomic across groups; the hi-word carry is detected from
        // this group's own InterlockedAdd's returned pre-image, which is correct
        // regardless of what any other concurrent group's own add does to the same word.
        GroupMemoryBarrierWithGroupSync();
        if (t == 0 && gQkNormClamps != 0) {
            uint sat_lo_off = SeqSatLoOffGpu(hidden_size);
            uint sat_hi_off = SeqSatHiOffGpu(hidden_size);
            uint old_lo;
            SeqState.InterlockedAdd(sat_lo_off, gQkNormClamps, old_lo);
            if (old_lo + gQkNormClamps < old_lo) {
                uint old_hi;
                SeqState.InterlockedAdd(sat_hi_off, 1u, old_hi);
            }
        }
    }
}
