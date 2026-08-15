// T-2039 (real-capacity shader geometry): down_proj -- real GPU dispatch,
// site 15 of 16. ProjectAndFunnel, no bias, input=act_codes (mlp_act's own
// output), in_channels=intermediate_size, out_channels=hidden_size,
// weight=down_weight (Layout[48]). Rebuilt to the design's own production
// dispatch geometry (Sec5.4/Sec5.5): numthreads(256,1,1), N-per-thread
// striding over hidden_size output channels, the wide row streamed through
// WorkScratch.
//
// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): the GEMM step (D-SLM3313's
// own top consumer, 28.9% of GPU-busy time, achieved bandwidth flat at ~1.7 GB/s regardless of
// matrix size -- the signature of one 256-thread group covering the whole output row, not a
// per-dispatch fixed cost) now runs in its OWN multi-group dispatch, `down_proj_gemm_site.hlsl`,
// issued immediately before this one (`superslm_gpu.cpp`'s own per-layer dispatch sequence). This
// shader is requant-only now: the wide row it reads from `WorkScratch` was written by that prior
// dispatch, already visible here via the host's own UAV barrier between every dispatch this
// design issues (unchanged, `bind_and_dispatch`'s own `ResourceBarrier` call) -- no new
// synchronization was added or removed, only WHICH dispatch performs the GEMM step changed.
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size;
    // T-2113 (B10 lever 1): positions 9-10 -- unread padding, see q_proj_site.hlsl's own
    // identical comment.
    uint g_unused9; uint g_unused10;
    // T-2113 (B10 lever 1): down's own adapter-delta coverage (slot=4, positions 11-17).
    uint g_adapter_rank;
    uint g_adapter_a_offset;
    uint g_adapter_b_offset;
    uint g_adapter_fold_offset;
    uint g_adapter_u_off;
    uint g_adapter_in_base;
    uint g_adapter_wide_base;
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
// T-2113 (B10 lever 1): see q_proj_site.hlsl's own identical comment.
ByteAddressBuffer LoraAB : register(t8);
ByteAddressBuffer Fold   : register(t9);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    int out_channels = hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    // T-2113 (B10 lever 1): the fused adapter-delta -- down's own slot (down=4), inserted before
    // this shader's own requant read of WorkScratch at g_adapter_wide_base. No-op when rank == 0.
    // in_channels is g_intermediate_size (down_proj's own in_channels, already delivered to this
    // shader today at position 8, unused until now) -- matching bind_and_dispatch_adapter_delta's
    // own standalone-path argument (I, H) exactly.
    ApplyFusedAdapterDeltaGpu(t, WorkScratch, LayerScratch, LoraAB, Fold, (int)g_intermediate_size,
                               out_channels, (int)g_adapter_rank, g_adapter_a_offset,
                               g_adapter_b_offset, g_adapter_fold_offset, g_adapter_u_off,
                               g_adapter_in_base, g_adapter_wide_base);

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint act_scale_off = ScratchLayout.Load<uint>(16 * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(act_scale_off + 0);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(act_scale_off + 8);

    uint off_site = layer_base + Layout.Load<uint>(52 * 4);

    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = in_scale_m; incoming_e[0] = in_scale_e;
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    uint down_codes_off = ScratchLayout.Load<uint>(17 * 4);
    uint down_scale_off = ScratchLayout.Load<uint>(18 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, out_channels, incoming_m, incoming_e, 1, site_m,
                                 site_e, LayerScratch, down_codes_off, down_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
