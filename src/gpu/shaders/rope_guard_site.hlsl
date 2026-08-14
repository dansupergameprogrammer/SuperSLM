// T-2032: RoPE's own GUARD -- real GPU dispatch, site 4 of 16. This
// checkpoint's own scope is the guard only (Claude/Brunel/t2025-gpu-serial-
// build-2026-08-13.md Sec11.2's "well-scoped next checkpoint": "sites 1-4
// (attn_norm, q_proj, kv_proj, RoPE's guard only) real"), not the rotation
// itself (RopeApplyPair, ClampRopeCode-on-rotated-output, the stage-then-
// commit K write-back) -- bit-exact port of forward_sites.cpp's RopeApplySite
// (forward_sites.cpp:639-737) THROUGH its own step 3 (the extent bound), never
// reaching step 4/5 (the table row read + rotation).
//
// The table's own presence/extent is resolved HOST-SIDE, once per call
// (rope_tables is model-wide, shared across every layer, Sec5.6), from the
// SAME SslmTensorManifest::Tensor("cos")/Tensor("sin") lookup RopeApplySite
// itself performs on CPU -- this dispatch reads the resolved flags/extents a
// host-side lookup already performed, exactly as this design's own weight
// buffers are host-resolved-then-uploaded rather than looked-up-by-name on
// the device.
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
ByteAddressBuffer   RopeInfo     : register(t2);  // cos_present u32, sin_present u32, cos_elem_count u64, sin_elem_count u64
RWByteAddressBuffer SeqState     : register(u0);
RWByteAddressBuffer LayerScratch : register(u1);
RWByteAddressBuffer KvCache      : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    // Step 1 (D-SLM376's own gate line): CheckPositionOverCap is the FIRST
    // act -- no table read of any kind before it returns.
    if (!CheckPositionOverCapGpu((int64_t)g_position, (int64_t)g_context_cap))
    {
        SeqState.Store<int64_t>(72, kTagPositionOverCap);
        return;
    }

    // Step 2: resolve "cos"/"sin" -- absent either returns RopeTableTensorMissing.
    uint cos_present = RopeInfo.Load(0);
    uint sin_present = RopeInfo.Load(4);
    if (cos_present == 0u || sin_present == 0u)
    {
        SeqState.Store<int64_t>(72, kTagRopeTableTensorMissing);
        return;
    }

    // Step 3: bound position and head_dim/2 against cos/sin's own validated elem_count.
    uint64_t cos_elem_count = RopeInfo.Load<uint64_t>(8);
    uint64_t sin_elem_count = RopeInfo.Load<uint64_t>(16);
    uint pairs = g_head_dim / 2u;
    if (pairs == 0u || (uint64_t)pairs > cos_elem_count || (uint64_t)pairs > sin_elem_count)
    {
        SeqState.Store<int64_t>(72, kTagRopeTableExtentExceeded);
        return;
    }
    uint64_t cos_rows = cos_elem_count / (uint64_t)pairs;
    uint64_t sin_rows = sin_elem_count / (uint64_t)pairs;
    if ((uint64_t)g_position >= cos_rows || (uint64_t)g_position >= sin_rows)
    {
        SeqState.Store<int64_t>(72, kTagRopeTableExtentExceeded);
        return;
    }

    // Ok: this checkpoint's own scope stops here (guard only) -- the rotation
    // itself (steps 4-5, plus the stage-then-commit K write-back, Sec5.6's
    // fifth-fracture remedy) is a placeholder-site's own obligation until the
    // next checkpoint lands it as a real dispatch.
}
