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
//
// GPU per-token overhead plan rev 3.1 §3.2: this file is now a thin wrapper. Its cbuffer and
// bindings are unchanged; `main`'s body moved verbatim into `fused_site_bodies.hlsli` as
// `DownTailSiteBody`, so the per-site dispatch (entry point, numthreads, groups, root constants) is
// v1.11.0's -- the compiled container is byte-identical to v1.11.0's -- and the fused shaders run
// the identical body text. The body reads this file's cbuffer fields and bindings by name, which
// is why they are declared before the include.
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
    // T-2113 (B10 lever 1b): see q_proj_site.hlsl's own identical comment.
    uint g_adapter_stage1_lanes;
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

#define SSLM_FUSED_BODY_DOWN_PROJ_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    DownTailSiteBody(t);
}
