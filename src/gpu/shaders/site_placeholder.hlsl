// T-2032 (Sec5.6/Sec11 B4 well-scoped next checkpoint): the shared
// placeholder for sites 5-16 (attention through mlp_residual) and the
// per-layer commit dispatch -- 13 dispatches of this ONE shared .cso per
// layer (the build log's own Sec11.2 analysis names "12" for sites 5-16
// alone; the commit dispatch is dispatched a 13th time with the identical
// shared shader here, since its own required behavior at this checkpoint --
// respect the sticky word, do nothing real -- is the same shape, see the
// T-2032 build-log section for this documented count correction).
//
// Every dispatch checks the sequence-level sticky word first (Sec5.6): if
// already Reject, this dispatch performs no arithmetic and writes nothing
// (matching every real site's own discipline). If still Ok, this site's own
// real arithmetic is not yet built -- the design's own NotYetImplemented
// sentinel (Sec11.2) is written rather than silently fabricating a wrong
// commit, so a fixture whose own CPU oracle needs any of these 13 dispatches
// to do real work fails this suite's own assertion honestly rather than by
// accident.
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index;
    uint g_hidden_size;
    uint g_head_dim;
    uint g_num_kv_heads;
    uint g_context_cap;
    uint g_position;
};

ByteAddressBuffer   LayerWeights : register(t0);
ByteAddressBuffer   Layout       : register(t1);
ByteAddressBuffer   RopeInfo     : register(t2);
RWByteAddressBuffer SeqState     : register(u0);
RWByteAddressBuffer LayerScratch : register(u1);
RWByteAddressBuffer KvCache      : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;
    SeqState.Store<int64_t>(72, kTagNotYetImplemented);
}
