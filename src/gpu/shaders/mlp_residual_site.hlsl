// T-2039 (real-capacity shader geometry): mlp_residual -- real GPU dispatch,
// site 16 of 16. Same construction as attn_residual_site.hlsl
// (ResidualReconcileSite), reconciling against the STAGED attn_residual
// output (LayerScratch's own attn_stream slot) rather than SeqState --
// forward_sites.cpp:1772-1777's own "reconciles against the staged output,
// not seq" reasoning. branch=down_codes/down_scale, out=stream_next, the
// layer's own final residual this checkpoint's commit dispatch commits into
// persistent SeqState.
//
// GPU per-token overhead plan rev 3.1 §3.2: this file is now a thin wrapper. Its cbuffer and
// bindings are unchanged; `main`'s body moved verbatim into `fused_site_bodies.hlsli` as
// `MlpResidualSiteBody`, so the per-site dispatch (entry point, numthreads, groups, root constants) is
// v1.11.0's -- the compiled container is byte-identical to v1.11.0's -- and the fused shaders run
// the identical body text. The body reads this file's cbuffer fields and bindings by name, which
// is why they are declared before the include.
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
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

#define SSLM_FUSED_BODY_MLP_RESIDUAL_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    MlpResidualSiteBody(t);
}
