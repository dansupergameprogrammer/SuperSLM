// T-2032/T-2035: the per-layer commit dispatch. T-2032 built this as the
// shared, always-no-op placeholder (never reached with sticky==Ok in that
// checkpoint's own scope). T-2035 makes it real: forward_sites.cpp:1779-1799's
// "ONE commit point for the whole layer" -- hidden_codes/hidden_scale/
// layer_index move together, only when every checked call in the layer
// returned Ok (this dispatch's own sticky check IS that condition, since
// every real site latches Reject on its own first rejection and nothing
// downstream can un-latch it -- Sec5.6). context_length advances by exactly
// one, at this SAME commit point, only when the layer that just committed is
// the token's last (forward_sites.cpp:1797-1799).
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size; uint g_num_hidden_layers;
};

ByteAddressBuffer   LayerWeights   : register(t0);
ByteAddressBuffer   Layout         : register(t1);
ByteAddressBuffer   RopeInfo       : register(t2);
ByteAddressBuffer   ModelConstants : register(t3);
ByteAddressBuffer   SiluLut        : register(t4);
ByteAddressBuffer   RopeCosTable   : register(t5);
ByteAddressBuffer   RopeSinTable   : register(t6);
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    int hidden_size = (int)g_hidden_size;
    for (int i = 0; i < hidden_size; ++i)
    {
        int v = (int)LayerScratch.Load<int>(528 + i * 4);
        SeqState.Store<int>(i * 4, v);
    }
    SeqState.Store<int64_t>(32, LayerScratch.Load<int64_t>(560));
    SeqState.Store<int64_t>(40, LayerScratch.Load<int64_t>(568));

    uint new_layer_index = g_layer_index + 1u;
    SeqState.Store(48, new_layer_index);
    if (new_layer_index == g_num_hidden_layers)
    {
        int64_t ctxlen = SeqState.Load<int64_t>(64);
        SeqState.Store<int64_t>(64, ctxlen + 1);
    }
}
