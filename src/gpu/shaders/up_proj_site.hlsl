// T-2039 (real-capacity shader geometry): up_proj -- real GPU dispatch, site
// 13 of 16. ProjectAndFunnel, no bias, input=normed (mlp_norm's own output),
// out_channels=intermediate_size, weight=up_weight (Layout[42]). Same
// construction as gate_proj_site.hlsl.
//
// T-2101 (per-dispatch parallelism, follow-up to D-SLM3312/D-SLM3313): the GEMM step (D-SLM3313's
// own third-largest consumer, 28.7% of GPU-busy time) now runs in its OWN multi-group dispatch,
// `up_proj_gemm_site.hlsl`, issued immediately before this one. This shader is requant-only now --
// see `down_proj_site.hlsl`'s own header comment for the full account.
//
// GPU per-token overhead plan rev 3.1 §3.2: this file is now a thin wrapper. Its cbuffer and
// bindings are unchanged; `main`'s body moved verbatim into `fused_site_bodies.hlsli` as
// `UpTailSiteBody`, so the per-site dispatch (entry point, numthreads, groups, root constants) is
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
    // T-2113 (B10 lever 1): up's own adapter-delta coverage (slot=3, positions 11-17).
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

// The up body reads its adapter slot through the SSLM_UP_ADAPTER_* macros (fused_site_bodies.hlsli's
// header comment says why they are macros): per-site, that is this file's own `g_adapter_*` fields,
// and the accumulator base is the literal 0u, both exactly v1.11.0's.
#define SSLM_UP_ADAPTER_RANK g_adapter_rank
#define SSLM_UP_ADAPTER_A_OFFSET g_adapter_a_offset
#define SSLM_UP_ADAPTER_B_OFFSET g_adapter_b_offset
#define SSLM_UP_ADAPTER_FOLD_OFFSET g_adapter_fold_offset
#define SSLM_UP_ADAPTER_U_OFF g_adapter_u_off
#define SSLM_UP_ADAPTER_IN_BASE g_adapter_in_base
#define SSLM_UP_ADAPTER_WIDE_BASE g_adapter_wide_base
#define SSLM_UP_ADAPTER_STAGE1_LANES g_adapter_stage1_lanes
#define SSLM_FUSED_BODY_UP_PROJ_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    UpTailSiteBody(t, /*acc_base=*/0u);
}
