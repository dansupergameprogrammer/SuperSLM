// T-2039 (real-capacity shader geometry): attn_residual -- real GPU
// dispatch, site 10 of 16. Bit-exact port of forward_sites.cpp's
// ResidualReconcileSite (forward_sites.cpp:847-965) at the attn_residual
// call site (forward_sites.cpp:1719-1723): branch=o_codes/o_scale,
// stream=seq.hidden_codes/hidden_scale (SeqState -- the pre-layer residual,
// since this dispatch runs before the layer's own commit). Rebuilt to the
// design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), N-per-thread striding over hidden_size, the wide row
// streamed through WorkScratch, the per-channel domain check cooperatively
// ORed into a shared flag (matching ApplyBiasReconcileRowGpuP's own
// check-all-then-commit shape).
//
// GPU per-token overhead plan rev 3.1 §3.2: this file is now a thin wrapper. Its cbuffer and
// bindings are unchanged; `main`'s body moved verbatim into `fused_site_bodies.hlsli` as
// `AttnResidualSiteBody`, so the per-site dispatch (entry point, numthreads, groups, root constants) is
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

#define SSLM_FUSED_BODY_ATTN_RESIDUAL_SITE
#include "fused_site_bodies.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    AttnResidualSiteBody(t);
}
