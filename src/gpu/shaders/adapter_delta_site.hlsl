// T-2113 (B6b, design Sec8/Sec10 B6): the GEMM-site adapter delta-application dispatch.
// Bit-exact port of forward_sites.cpp's AddAmplifyingLoraDelta -- the runtime-additive
// LoRA delta this design adds is a chained pair of small GEMMs (x*A -> u_acc[rank], then
// u_i8*B -> delta_raw[out_channels]), each folded through ApplyAmplifyingWeightScaleFoldGpu
// (site_common.hlsli), with a narrow-and-clamp (ClampRopeCodeGpu) between the two stages --
// added into the SAME WorkScratch wide-row accumulator the corresponding `<site>_gemm_site.hlsl`
// dispatch just wrote (WSC1-folded, per-channel), exactly where AddAmplifyingLoraDelta's own
// insertion point sits on the CPU path: after the base GEMM's own WSC1 fold, before the
// site's own bias/requant dispatch.
//
// Dispatched Dispatch(1,1,1) -- a single 256-thread group. Rank is small (a LoRA adapter's
// own rank-sized matrices, never the base GEMM's own out_channels/in_channels scale), so
// this shader never needs more than one group: stage 1 assigns one thread per u_acc[k]
// (k < rank), each computing its OWN full in_channels-wide reduction; stage 2 assigns one
// thread per delta_raw[j] (j < out_channels), each reducing over the (small) rank
// dimension. Integer addition is exactly associative (no float, no atomic, no reordering
// risk within this design's own proven int8-code/int64-accumulator domain -- the same
// determinism law GemmCoalescedGpuAt/GemmParallelGpu's own header comments already state),
// so a thread-per-output partition reproduces the CPU path's own left-to-right per-row
// reduction bit-for-bit regardless of which thread computed which output.
//
// Never dispatched at all when no adapter is bound or a (layer, projection) slot is
// uncovered (superslm_gpu.cpp's own `bind_and_dispatch_adapter_delta`, which checks
// `AdapterProjSlot::present` before recording this dispatch) -- design Sec8's own
// "records the identical dispatch chain a base-only sequence always did" contract for the
// unadapted case, realized as an absent dispatch, matching AddAmplifyingLoraDelta's own
// gated-no-op contract exactly (byte-identical WorkScratch row when this dispatch never
// runs).
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    // T-2113 (B6b): this shader's own root-constant layout -- distinct meaning from every
    // other composed-pipeline shader's shared 11-value cbuffer (superslm_gpu.cpp's
    // `bind_and_dispatch_adapter_delta` packs this set fresh before each dispatch, never
    // reusing `bind_and_dispatch`'s own {layer_index,H,HD,NH,...} packing).
    uint g_hidden_size;    // for SeqStickyOffGpu's own sticky-word offset only
    uint g_in_channels;    // the base GEMM's own in_channels this delta's x*A stage reduces over
    uint g_out_channels;   // the base GEMM's own out_channels -- delta_raw's width, DFS row count
    uint g_rank;           // this (layer, projection) slot's own rank -- u_acc's width, UFS row count
    uint g_a_offset;       // byte offset into LoraAB (t8): lora_A, [rank, in_channels] row-major int8
    uint g_b_offset;       // byte offset into LoraAB (t8): lora_B, [out_channels, rank] row-major int8
    uint g_fold_offset;    // byte offset into Fold (t9): DeltaFoldScales rows (out_channels*12 bytes)
                           // immediately followed by UFoldScales rows (rank*12 bytes)
    uint g_wide_base;      // WorkScratch byte offset of the base GEMM's own WSC1-folded acc row
    uint g_adapter_u_off;  // WorkScratch byte offset of this call's own ADAPTER_U scratch region
    uint g_in_base;        // LayerScratch byte offset of the activation row (in_codes) this
                           // dispatch's x*A stage reads -- the SAME row the base GEMM read
    uint g_unused;
};

ByteAddressBuffer   LoraAB       : register(t8);
ByteAddressBuffer   Fold         : register(t9);
RWByteAddressBuffer SeqState     : register(u0);
RWByteAddressBuffer LayerScratch : register(u1);
RWByteAddressBuffer WorkScratch  : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    int in_channels = (int)g_in_channels;
    int out_channels = (int)g_out_channels;
    int rank = (int)g_rank;
    uint dfs_base = g_fold_offset;                              // out_channels rows, 12 bytes each
    uint ufs_base = g_fold_offset + (uint)out_channels * 12u;    // rank rows, 12 bytes each

    // Stage 1 (forward_sites.cpp AddAmplifyingLoraDelta's own first half): u_acc[k] =
    // GemmInt8AccumulateRow(x, A)[k], then u_wide[k] = ApplyAmplifyingWeightScaleFold(u_acc[k],
    // UFoldScales[k]), then u_i8[k] = int8(ClampRopeCode(u_wide[k])) -- ONE thread per k,
    // rank is always small relative to 256 (a real LoRA adapter's own rank), so no
    // cross-thread reduction is needed here, only cross-thread PARTITION.
    for (int k = (int)t; k < rank; k += 256)
    {
        int64_t acc = 0;
        for (int i = 0; i < in_channels; ++i)
        {
            int a = LayerScratch.Load<int>(g_in_base + (uint)i * 4u);
            int w = LoadSignedByteGpu(LoraAB, g_a_offset + (uint)(k * in_channels + i));
            acc += (int64_t)a * (int64_t)w;
        }
        int u_identity = (int)Fold.Load<int>(ufs_base + (uint)k * 12u + 0u);
        int u_mult     = (int)Fold.Load<int>(ufs_base + (uint)k * 12u + 4u);
        int u_exponent = (int)Fold.Load<int>(ufs_base + (uint)k * 12u + 8u);
        int64_t u_wide = ApplyAmplifyingWeightScaleFoldGpu(acc, u_identity, u_mult, u_exponent);
        int64_t u_clamped = ClampRopeCodeGpu(u_wide);
        StoreSignedByteGpu(WorkScratch, g_adapter_u_off + (uint)k, (int)u_clamped);
    }
    // u_i8 lives in WorkScratch (a UAV), written above by this SAME group -- a device-memory
    // barrier (not merely groupshared) is required before stage 2 reads it back, mirroring
    // q_proj_site.hlsl's own DeviceMemoryBarrierWithGroupSync() after ApplyBiasReconcileRowGpuP's
    // own cross-phase UAV write.
    DeviceMemoryBarrierWithGroupSync();

    // Stage 2 (AddAmplifyingLoraDelta's own second half): delta_raw[j] =
    // GemmInt8AccumulateRow(u_i8, B)[j], delta_wide[j] = ApplyAmplifyingWeightScaleFold(
    // delta_raw[j], DeltaFoldScales[j]), acc[j] += delta_wide[j] -- ONE thread per j, added
    // directly into the base GEMM's own WorkScratch row (design Sec3's own exact insertion
    // point, reproduced here as a read-modify-write of the SAME bytes the just-completed
    // `<site>_gemm_site.hlsl` dispatch wrote, published across dispatches by the global UAV
    // barrier `bind_and_dispatch`/`bind_and_dispatch_adapter_delta` already issue after every
    // dispatch).
    for (int j = (int)t; j < out_channels; j += 256)
    {
        int64_t acc2 = 0;
        for (int k2 = 0; k2 < rank; ++k2)
        {
            int u = LoadSignedByteGpu(WorkScratch, g_adapter_u_off + (uint)k2);
            int w2 = LoadSignedByteGpu(LoraAB, g_b_offset + (uint)(j * rank + k2));
            acc2 += (int64_t)u * (int64_t)w2;
        }
        int d_identity = (int)Fold.Load<int>(dfs_base + (uint)j * 12u + 0u);
        int d_mult     = (int)Fold.Load<int>(dfs_base + (uint)j * 12u + 4u);
        int d_exponent = (int)Fold.Load<int>(dfs_base + (uint)j * 12u + 8u);
        int64_t delta_wide = ApplyAmplifyingWeightScaleFoldGpu(acc2, d_identity, d_mult, d_exponent);
        int64_t base_val = WorkScratch.Load<int64_t>(g_wide_base + (uint)j * 8u);
        WorkScratch.Store<int64_t>(g_wide_base + (uint)j * 8u, base_val + delta_wide);
    }
}
