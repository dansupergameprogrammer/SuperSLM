// T-2113 (B4, design Sec3/Sec6.1, Sec10 B4): kv_proj's own two GEMM steps (K and V), split
// out of kv_proj_site.hlsl into their own multi-group coalesced dispatch. Re-derived from
// Claude/Laplace/t2105-gpu-speed-ceiling-2026-08-14.md Sec2 change 5 -- `kv_proj_gemm` is
// NEW production work (design Sec3's own correction, D-SLM3341): the pre-1.0 substrate's
// `GpuGemmSplitSite` enum never carried a `KvProj` entry, and `kv_proj_site.hlsl`'s own
// header comment states the GEMM step could not be split "without either atomics or a
// second dispatch" -- this file is that second dispatch.
//
// WHY THIS SPLIT IS SOUND, against kv_proj_site.hlsl's own header comment naming the real
// obstacle. That comment is correct: the fused K+V bias-precedence check (gKFailed/gVFailed)
// and the landing-rescale clamp counter (gTotalClamps) ARE genuine cross-thread cooperative
// reductions over the whole kv_hidden_size row, and splitting THEM across groups would need
// atomics or reordered accumulation. But the GEMM step alone has no cross-thread dependency
// at all (each output channel's own reduction is independent of every other), so it moves
// out to a multi-group dispatch and the guard/landing pass stays behind, single-group, over
// the 256-element kv_hidden_size row it was always cheap on. The global UAV barrier
// `bind_and_dispatch` issues after every dispatch (superslm_gpu.cpp) is what publishes the
// wide rows from this dispatch to the next -- strictly stronger synchronization than the
// DeviceMemoryBarrierWithGroupSync the fused version relied on, and the reason this split is
// also time-slice invariant: the two dispatches may be submitted in different command lists,
// in different execution slices, with a fence between them, and the bytes do not change.
//
// K AND V IN ONE DISPATCH. The dispatch covers 2*kv_hidden_size logical output channels:
// channel c < kv_hidden_size is K's own channel c (weights off_kw, wide row at WorkScratch
// base 0), and c >= kv_hidden_size is V's own channel c - kv_hidden_size (weights off_vw,
// wide row at WorkScratch's own WIDE_B base). Both halves land in exactly the byte ranges the
// fused version's own two GemmParallelGpu calls used to write, so kv_proj_site.hlsl's own
// downstream code reads the identical values from the identical addresses.
//
// The accumulation is GemmCoalescedGpuAt's own int64 groupshared tree -- see site_common.hlsli's
// own header comment for why a tree over integer addition returns the identical int64 the
// serial loop does, and why that is the reduction shape this substrate already ships
// (RmsSumSqParallelGpu).
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    // T-2113 (B4): the 10th and 11th root constants. `g_gemm_lanes` is the ONE source of
    // this dispatch's own lane split -- the host computes the group count from the SAME
    // value, so the two cannot drift.
    uint g_num_hidden_layers; uint g_gemm_lanes;
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
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    // Uniform across the group (one buffer word, same value for every thread), so this early
    // return does not skip a barrier for some threads and not others.
    if (sticky != kTagOk) return;

    int head_dim = (int)g_head_dim;
    int num_kv_heads = (int)g_num_kv_heads;
    int kv_hidden_size = num_kv_heads * head_dim;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint wide_b_off = ScratchLayout.Load<uint>(23 * 4);

    uint off_kw = layer_base + Layout.Load<uint>(9 * 4);
    uint off_vw = layer_base + Layout.Load<uint>(10 * 4);
    uint off_kid = layer_base + Layout.Load<uint>(11 * 4);
    uint off_kmult = layer_base + Layout.Load<uint>(12 * 4);
    uint off_kshift = layer_base + Layout.Load<uint>(13 * 4);
    uint off_vid = layer_base + Layout.Load<uint>(14 * 4);
    uint off_vmult = layer_base + Layout.Load<uint>(15 * 4);
    uint off_vshift = layer_base + Layout.Load<uint>(16 * 4);

    uint lanes = g_gemm_lanes;
    uint channels_per_group = 256u / lanes;
    uint c_local = t / lanes;
    int c = (int)(gid.x * channels_per_group + c_local);

    // Both calls are made by EVERY thread of the group (each contains group-wide barriers, so
    // neither may be predicated). A thread whose `c` belongs to the other half passes an
    // out-of-range channel index, contributes zero, and stores nothing -- exactly the
    // past-the-end case GemmCoalescedGpuAt already handles for a ragged final group.
    int j_k = c;
    int j_v = c - kv_hidden_size;
    GemmCoalescedGpuAt(t, j_k, LayerScratch, normed_off, LayerWeights, off_kw, LayerWeights, off_kid,
                        LayerWeights, off_kmult, LayerWeights, off_kshift, hidden_size,
                        kv_hidden_size, WorkScratch, 0u, lanes);
    GemmCoalescedGpuAt(t, j_v, LayerScratch, normed_off, LayerWeights, off_vw, LayerWeights, off_vid,
                        LayerWeights, off_vmult, LayerWeights, off_vshift, hidden_size,
                        kv_hidden_size, WorkScratch, wide_b_off, lanes);
}
