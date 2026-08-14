// T-2039 (real-capacity shader geometry): the per-layer commit dispatch.
// forward_sites.cpp:1779-1799's "ONE commit point for the whole layer" --
// hidden_codes/hidden_scale/layer_index move together, only when every
// checked call in the layer returned Ok (this dispatch's own sticky check IS
// that condition). context_length advances by exactly one, at this SAME
// commit point, only when the layer that just committed is the token's last
// (forward_sites.cpp:1797-1799).
//
// Rebuilt to the design's own production dispatch geometry (Sec5.4/Sec5.5):
// numthreads(256,1,1), N-per-thread striding over hidden_size for the
// residual-codes copy, driven entirely by the real g_hidden_size root
// constant (superseding T-2035's own fixed 640-byte LayerScratch/hidden_codes
// assumption).
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
ByteAddressBuffer   ScratchLayout  : register(t7);
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);
RWByteAddressBuffer WorkScratch    : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint stream_next_off = ScratchLayout.Load<uint>(19 * 4);
    uint stream_next_scale_off = ScratchLayout.Load<uint>(20 * 4);
    uint seq_scale_off = SeqScaleOffGpu(hidden_size);

    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int v = (int)LayerScratch.Load<int>(stream_next_off + (uint)i * 4u);
        SeqState.Store<int>((uint)i * 4u, v);
    }
    if (t == 0)
    {
        SeqState.Store<int64_t>(seq_scale_off + 0, LayerScratch.Load<int64_t>(stream_next_scale_off + 0));
        SeqState.Store<int64_t>(seq_scale_off + 8, LayerScratch.Load<int64_t>(stream_next_scale_off + 8));

        uint new_layer_index = g_layer_index + 1u;
        SeqState.Store(SeqLayerIdxOffGpu(hidden_size), new_layer_index);
        if (new_layer_index == g_num_hidden_layers)
        {
            uint ctx_len_off = SeqCtxLenOffGpu(hidden_size);
            int64_t ctxlen = SeqState.Load<int64_t>(ctx_len_off);
            SeqState.Store<int64_t>(ctx_len_off, ctxlen + 1);
        }
    }
}
