// GPU per-token overhead plan rev 3.1 §3.2/§3.3 (S2, S3): the `main` bodies of the eight non-K/V
// sites that the fused dispatches combine, moved VERBATIM out of their own `.hlsl` files into
// functions, so one body text serves both recording modes:
//   - per-site mode: each site's own `.hlsl` (now a thin wrapper -- its cbuffer, its bindings, and
//     a `main` that calls the body with today's arguments) is today's dispatch, unchanged;
//   - bracket mode: `fused_o_residual_norm_site.hlsl`, `fused_down_residual_commit_site.hlsl` and
//     `fused_gate_up_site.hlsl` call several bodies in site order, one
//     `DeviceMemoryBarrierWithGroupSync()` between consecutive calls (tests/ci/
//     check_fused_site_barriers.py asserts that barrier is present, plan §7 cell 7.7).
//
// HOW A BODY SEES ITS CONSTANTS AND BINDINGS. A body refers to cbuffer fields (`g_hidden_size`,
// `g_adapter_rank`, ...) and resources (`SeqState`, `WorkScratch`, ...) by their v1.11.0 names. It
// does not declare them: every shader that includes this file declares its cbuffer and bindings
// with those exact names BEFORE the include, then defines one `SSLM_FUSED_BODY_<SITE>` macro per
// body it calls. A body whose macro is not defined is not compiled, so a wrapper that never
// declared the adapter fields (attn_residual, mlp_norm, mlp_residual, commit) never sees a body
// that reads them. Each body keeps its own entry sticky check and its own rejection stores, so a
// rejection in one body stops the next exactly as it stops the next dispatch today (plan §3.2).
//
// THE ONLY EDITS TO MOVED TEXT. Everything below is byte-for-byte the v1.11.0 `main` body (minus
// its first line `uint t = gtid.x;`, which became the parameter `t`) plus the site-local
// declarations that sat between that file's bindings and its `main`, except:
//   - The up body (S3) takes what the fused gate+up tail must change (plan §3.3, the kv tail's
//     V-half arrangement): `acc_base`, its requant's accumulator base (v1.11.0's hard-coded `0u`),
//     as a parameter; and its adapter slot through the macro `SSLM_UP_BODY_ADAPTER(field)`, which
//     the includer defines -- `g_adapter_##field` in up_proj_site.hlsl (per-site, unchanged), slot
//     1's `g_adapter_up_##field` (constants 19-26, wide base = WIDE_B) in fused_gate_up_site.hlsl.
//     The slot is a macro, not eight more parameters, because parameters are evaluated at the call:
//     dxc then hoists the eight cbuffer loads above the body's sticky check and the wrapper's DXIL
//     stops being v1.11.0's (measured; the S2/S3 shader record has the diff). With the macro the
//     wrapper's DXIL is byte-identical to v1.11.0's.
//   - attn_residual and mlp_residual each define a file-local helper named
//     `BuildResidualCandidateGpu`, with DIFFERENT bodies (attn_residual reads the stream from
//     SeqState, mlp_residual from LayerScratch). They keep that name -- renaming changes the DXIL's
//     alias-scope metadata strings and so the shader hash -- and the two guards are refused
//     together below, so they can never meet in one translation unit. Their groupshared flags were
//     already distinct (`gResidualMagOutOfDomain`, `gResidualMagOutOfDomain2`).
//
// Inlined, each wrapper compiles to a DXIL container byte-identical to its v1.11.0 shader's (the
// S2/S3 shader record gives the comparison). None of these is a K/V shader (paged-KV G2's six are
// attention_score, context_accumulate, kv_proj, qk_norm, rope_commit, rope_guard), and
// `mlp_act_site.hlsl` is deliberately not here: it is a 1024-thread group (plan §1).
#ifndef SSLM_FUSED_SITE_BODIES_HLSLI
#define SSLM_FUSED_SITE_BODIES_HLSLI

#include "site_common2.hlsli"

#if defined(SSLM_FUSED_BODY_ATTN_RESIDUAL_SITE) && defined(SSLM_FUSED_BODY_MLP_RESIDUAL_SITE)
#error "attn_residual's and mlp_residual's bodies each carry their own BuildResidualCandidateGpu"
#endif
#if defined(SSLM_FUSED_BODY_UP_PROJ_SITE) && !defined(SSLM_UP_BODY_ADAPTER)
#error "the up body reads its adapter slot through SSLM_UP_BODY_ADAPTER(field); define it first"
#endif

#if defined(SSLM_FUSED_BODY_O_PROJ_SITE)
// ---- o_proj_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2).
void OTailSiteBody(uint t)
{
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    // T-2113 (B10 lever 1): the fused adapter-delta -- o's own slot (o=1), inserted before this
    // shader's own requant read of WorkScratch at g_adapter_wide_base. No-op when rank == 0.
    ApplyFusedAdapterDeltaGpu(t, WorkScratch, LayerScratch, LoraAB, Fold, (int)g_hidden_size,
                               (int)g_hidden_size, (int)g_adapter_rank, g_adapter_a_offset,
                               g_adapter_b_offset, g_adapter_fold_offset, g_adapter_u_off,
                               g_adapter_in_base, g_adapter_wide_base, (int)g_adapter_stage1_lanes);

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint ctx_scale_off = ScratchLayout.Load<uint>(6 * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(ctx_scale_off + 0);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(ctx_scale_off + 8);

    uint off_site = layer_base + Layout.Load<uint>(29 * 4);

    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = in_scale_m; incoming_e[0] = in_scale_e;
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    uint o_codes_off = ScratchLayout.Load<uint>(7 * 4);
    uint o_scale_off = ScratchLayout.Load<uint>(8 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, /*n_incoming=*/1,
                                 site_m, site_e, LayerScratch, o_codes_off, o_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
#endif  // SSLM_FUSED_BODY_O_PROJ_SITE

#if defined(SSLM_FUSED_BODY_ATTN_RESIDUAL_SITE)
// ---- attn_residual_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2),
// with the site-local declarations that stood between that file's bindings and its `main`.

groupshared uint gResidualMagOutOfDomain;

bool BuildResidualCandidateGpu(uint t, int hidden_size, uint branch_codes_off, uint stream_codes_off,
                               bool select_branch, int64_t branch_m, int64_t branch_e,
                               int64_t stream_m, int64_t stream_e, out int64_t selected_m,
                               out int64_t selected_e, out int64_t status_tag)
{
    selected_m = select_branch ? branch_m : stream_m;
    selected_e = select_branch ? branch_e : stream_e;
    int64_t other_m = select_branch ? stream_m : branch_m;
    int64_t other_e = select_branch ? stream_e : branch_e;
    int64_t reciprocal, normalization_shift;
    CarriedScaleNormalizedReciprocalSharedGpu(AbsUnsignedI64Gpu(selected_m), reciprocal, normalization_shift);

    if (t == 0) gResidualMagOutOfDomain = 0;
    GroupMemoryBarrierWithGroupSync();
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int direct_code = select_branch ? LayerScratch.Load<int>(branch_codes_off + (uint)i * 4u)
                                        : SeqState.Load<int>(stream_codes_off + (uint)i * 4u);
        int other_code = select_branch ? SeqState.Load<int>(stream_codes_off + (uint)i * 4u)
                                       : LayerScratch.Load<int>(branch_codes_off + (uint)i * 4u);
        bool would_clamp, magnitude_exceeded;
        int64_t landed = LandingRescaleGpu((int64_t)other_code, other_m, reciprocal, other_e,
                                            selected_e, normalization_shift, would_clamp,
                                            magnitude_exceeded);
        int64_t wide_sum;
        if (magnitude_exceeded || (selected_m < 0 && landed == (int64_t)0x8000000000000000ULL))
            InterlockedOr(gResidualMagOutOfDomain, 1u);
        else
        {
            if (selected_m < 0) landed = -landed;
            if (!CheckedAddI64Gpu((int64_t)direct_code, landed, wide_sum))
                InterlockedOr(gResidualMagOutOfDomain, 1u);
            else
                WorkScratch.Store<int64_t>((uint)i * 8u, wide_sum);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    status_tag = gResidualMagOutOfDomain != 0 ? kTagResidualReconciliationMagnitudeOutOfDomain : kTagOk;
    return status_tag == kTagOk;
}

void AttnResidualSiteBody(uint t)
{
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint stream_scale_off = SeqScaleOffGpu(hidden_size);

    int64_t stream_m = SeqState.Load<int64_t>(stream_scale_off + 0);
    int64_t stream_e = SeqState.Load<int64_t>(stream_scale_off + 8);
    uint o_codes_off = ScratchLayout.Load<uint>(7 * 4);
    uint o_scale_off = ScratchLayout.Load<uint>(8 * 4);
    int64_t branch_m = LayerScratch.Load<int64_t>(o_scale_off + 0);
    int64_t branch_e = LayerScratch.Load<int64_t>(o_scale_off + 8);
    if (branch_m == 0 || stream_m == 0)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagResidualReconciliationScaleOutOfDomain);
        return;
    }
    if (branch_m < -2147483648LL || branch_m > 2147483647LL ||
        stream_m < -2147483648LL || stream_m > 2147483647LL)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagCarriedScaleMantissaOutOfDomain);
        return;
    }
    uint off_site = layer_base + Layout.Load<uint>(34 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    bool branch_selected = ResidualBranchGridIsFinerGpu(branch_m, branch_e, stream_m, stream_e);
    int64_t selected_m, selected_e, candidate_status;
    bool candidate_ok = BuildResidualCandidateGpu(t, hidden_size, o_codes_off, 0u, branch_selected,
                                                   branch_m, branch_e, stream_m, stream_e, selected_m,
                                                   selected_e, candidate_status);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = selected_m; incoming_e[0] = selected_e;
    if (candidate_ok)
        candidate_ok = PreflightRequantChainFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m,
                                                       incoming_e, 1, site_m, site_e, candidate_status);
    if (!candidate_ok)
    {
        candidate_ok = BuildResidualCandidateGpu(t, hidden_size, o_codes_off, 0u, !branch_selected,
                                                  branch_m, branch_e, stream_m, stream_e, selected_m,
                                                  selected_e, candidate_status);
        incoming_m[0] = selected_m; incoming_e[0] = selected_e;
        if (candidate_ok)
            candidate_ok = PreflightRequantChainFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m,
                                                           incoming_e, 1, site_m, site_e, candidate_status);
    }
    if (!candidate_ok)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, candidate_status);
        return;
    }

    uint attn_stream_off = ScratchLayout.Load<uint>(9 * 4);
    uint attn_stream_scale_off = ScratchLayout.Load<uint>(10 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, 1, site_m,
                                 site_e, LayerScratch, attn_stream_off, attn_stream_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
#endif  // SSLM_FUSED_BODY_ATTN_RESIDUAL_SITE

#if defined(SSLM_FUSED_BODY_MLP_NORM_SITE)
// ---- mlp_norm_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2),
// with the site-local declarations that stood between that file's bindings and its `main`.

static const int kNormFracBitsGpu2 = 16;

void MlpNormSiteBody(uint t)
{
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint attn_stream_off = ScratchLayout.Load<uint>(9 * 4);

    int64_t sumsq = RmsSumSqParallelGpu(t, LayerScratch, attn_stream_off, hidden_size);
    int64_t root = ISqrtGpu(FloorDivI64Gpu(sumsq << (2 * kNormFracBitsGpu2), (int64_t)hidden_size));
    root = (root > 1) ? root : 1;

    uint off_gain = layer_base + Layout.Load<uint>(35 * 4);
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int64_t hv = (int64_t)(int)LayerScratch.Load<int>(attn_stream_off + (uint)i * 4u);
        int g = (int)LayerWeights.Load<int>(off_gain + (uint)i * 4u);
        int64_t wv = FloorDivI64Gpu(hv << (2 * kNormFracBitsGpu2), root) * (int64_t)g;
        WorkScratch.Store<int64_t>(0u + (uint)i * 8u, wv);
    }
    DeviceMemoryBarrierWithGroupSync();

    uint off_site = layer_base + Layout.Load<uint>(36 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }

    uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, 0, site_m, site_e,
                                 LayerScratch, normed_off, normed_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
#endif  // SSLM_FUSED_BODY_MLP_NORM_SITE

#if defined(SSLM_FUSED_BODY_GATE_PROJ_SITE)
// ---- gate_proj_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2).
void GateTailSiteBody(uint t)
{
    // SSLM-GEOMETRY-SITE: GS-30
    // T-2509 (widening byproduct, geometry_site_census.py's own multi-line window sweep):
    // `hidden_size` here and `out_channels` two lines below are TWO INDEPENDENT, both-correct
    // local variables that happen to share this shader's own generic prelude, not one standing
    // in for the other -- `hidden_size` feeds only SeqStickyOffGpu's per-sequence tag-offset
    // lookup (unrelated to any projection's width), and `out_channels` correctly equals
    // g_intermediate_size, gate_proj's own real, unaffected output width (matching
    // gate_proj_gemm_site.hlsl's own confirmed-correct GS-23). Registered, not fixed: no code
    // here assumes hidden_size == num_attention_heads * head_dim.
    int hidden_size = (int)g_hidden_size;
    int out_channels = (int)g_intermediate_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    // T-2113 (B10 lever 1): the fused adapter-delta -- gate's own slot (gate=2), inserted before
    // this shader's own requant read of WorkScratch at g_adapter_wide_base. No-op when rank == 0.
    ApplyFusedAdapterDeltaGpu(t, WorkScratch, LayerScratch, LoraAB, Fold, (int)g_hidden_size,
                               out_channels, (int)g_adapter_rank, g_adapter_a_offset,
                               g_adapter_b_offset, g_adapter_fold_offset, g_adapter_u_off,
                               g_adapter_in_base, g_adapter_wide_base, (int)g_adapter_stage1_lanes);

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(normed_scale_off + 0);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(normed_scale_off + 8);

    uint off_site = layer_base + Layout.Load<uint>(41 * 4);

    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = in_scale_m; incoming_e[0] = in_scale_e;
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    uint gate_codes_off = ScratchLayout.Load<uint>(11 * 4);
    uint gate_scale_off = ScratchLayout.Load<uint>(12 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, out_channels, incoming_m, incoming_e, 1, site_m,
                                 site_e, LayerScratch, gate_codes_off, gate_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
#endif  // SSLM_FUSED_BODY_GATE_PROJ_SITE

#if defined(SSLM_FUSED_BODY_UP_PROJ_SITE)
// ---- up_proj_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2).
void UpTailSiteBody(uint t, uint acc_base)
{
    // SSLM-GEOMETRY-SITE: GS-31
    // T-2509 (widening byproduct, geometry_site_census.py's own multi-line window sweep):
    // `hidden_size` here and `out_channels` two lines below are TWO INDEPENDENT, both-correct
    // local variables that happen to share this shader's own generic prelude, not one standing
    // in for the other -- `hidden_size` feeds only SeqStickyOffGpu's per-sequence tag-offset
    // lookup (unrelated to any projection's width), and `out_channels` correctly equals
    // g_intermediate_size, up_proj's own real, unaffected output width (matching
    // up_proj_gemm_site.hlsl's own confirmed-correct GS-24). Registered, not fixed: no code here
    // assumes hidden_size == num_attention_heads * head_dim.
    int hidden_size = (int)g_hidden_size;
    int out_channels = (int)g_intermediate_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    // T-2113 (B10 lever 1): the fused adapter-delta -- up's own slot (up=3), inserted before
    // this shader's own requant read of WorkScratch at g_adapter_wide_base. No-op when rank == 0.
    ApplyFusedAdapterDeltaGpu(t, WorkScratch, LayerScratch, LoraAB, Fold, (int)g_hidden_size,
                               out_channels, (int)SSLM_UP_BODY_ADAPTER(rank),
                               SSLM_UP_BODY_ADAPTER(a_offset), SSLM_UP_BODY_ADAPTER(b_offset),
                               SSLM_UP_BODY_ADAPTER(fold_offset), SSLM_UP_BODY_ADAPTER(u_off),
                               SSLM_UP_BODY_ADAPTER(in_base), SSLM_UP_BODY_ADAPTER(wide_base),
                               (int)SSLM_UP_BODY_ADAPTER(stage1_lanes));

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);

    uint normed_scale_off = ScratchLayout.Load<uint>(1 * 4);
    int64_t in_scale_m = LayerScratch.Load<int64_t>(normed_scale_off + 0);
    int64_t in_scale_e = LayerScratch.Load<int64_t>(normed_scale_off + 8);

    uint off_site = layer_base + Layout.Load<uint>(46 * 4);

    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = in_scale_m; incoming_e[0] = in_scale_e;
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);

    uint up_codes_off = ScratchLayout.Load<uint>(13 * 4);
    uint up_scale_off = ScratchLayout.Load<uint>(14 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, acc_base, out_channels, incoming_m, incoming_e, 1,
                                 site_m, site_e, LayerScratch, up_codes_off, up_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
#endif  // SSLM_FUSED_BODY_UP_PROJ_SITE

#if defined(SSLM_FUSED_BODY_DOWN_PROJ_SITE)
// ---- down_proj_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2).
void DownTailSiteBody(uint t)
{
    int hidden_size = (int)g_hidden_size;
    // SSLM-GEOMETRY-SITE: GS-22 -- confirmed-correct-and-marked (T-2432): down_proj's own
    // output width is genuinely hidden_size, unaffected by Q/O's own width decoupling.
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
                               g_adapter_in_base, g_adapter_wide_base, (int)g_adapter_stage1_lanes);

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
#endif  // SSLM_FUSED_BODY_DOWN_PROJ_SITE

#if defined(SSLM_FUSED_BODY_MLP_RESIDUAL_SITE)
// ---- mlp_residual_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2),
// with the site-local declarations that stood between that file's bindings and its `main`.

groupshared uint gResidualMagOutOfDomain2;

bool BuildResidualCandidateGpu(uint t, int hidden_size, uint branch_codes_off, uint stream_codes_off,
                               bool select_branch, int64_t branch_m, int64_t branch_e,
                               int64_t stream_m, int64_t stream_e, out int64_t selected_m,
                               out int64_t selected_e, out int64_t status_tag)
{
    selected_m = select_branch ? branch_m : stream_m;
    selected_e = select_branch ? branch_e : stream_e;
    int64_t other_m = select_branch ? stream_m : branch_m;
    int64_t other_e = select_branch ? stream_e : branch_e;
    int64_t reciprocal, normalization_shift;
    CarriedScaleNormalizedReciprocalSharedGpu(AbsUnsignedI64Gpu(selected_m), reciprocal, normalization_shift);

    if (t == 0) gResidualMagOutOfDomain2 = 0;
    GroupMemoryBarrierWithGroupSync();
    for (int i = (int)t; i < hidden_size; i += 256)
    {
        int direct_code = select_branch ? LayerScratch.Load<int>(branch_codes_off + (uint)i * 4u)
                                        : LayerScratch.Load<int>(stream_codes_off + (uint)i * 4u);
        int other_code = select_branch ? LayerScratch.Load<int>(stream_codes_off + (uint)i * 4u)
                                       : LayerScratch.Load<int>(branch_codes_off + (uint)i * 4u);
        bool would_clamp, magnitude_exceeded;
        int64_t landed = LandingRescaleGpu((int64_t)other_code, other_m, reciprocal, other_e,
                                            selected_e, normalization_shift, would_clamp,
                                            magnitude_exceeded);
        int64_t wide_sum;
        if (magnitude_exceeded || (selected_m < 0 && landed == (int64_t)0x8000000000000000ULL))
            InterlockedOr(gResidualMagOutOfDomain2, 1u);
        else
        {
            if (selected_m < 0) landed = -landed;
            if (!CheckedAddI64Gpu((int64_t)direct_code, landed, wide_sum))
                InterlockedOr(gResidualMagOutOfDomain2, 1u);
            else
                WorkScratch.Store<int64_t>((uint)i * 8u, wide_sum);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    status_tag = gResidualMagOutOfDomain2 != 0 ? kTagResidualReconciliationMagnitudeOutOfDomain : kTagOk;
    return status_tag == kTagOk;
}

void MlpResidualSiteBody(uint t)
{
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    uint attn_stream_off = ScratchLayout.Load<uint>(9 * 4);
    uint attn_stream_scale_off = ScratchLayout.Load<uint>(10 * 4);

    int64_t stream_m = LayerScratch.Load<int64_t>(attn_stream_scale_off + 0);
    int64_t stream_e = LayerScratch.Load<int64_t>(attn_stream_scale_off + 8);
    uint down_codes_off = ScratchLayout.Load<uint>(17 * 4);
    uint down_scale_off = ScratchLayout.Load<uint>(18 * 4);
    int64_t branch_m = LayerScratch.Load<int64_t>(down_scale_off + 0);
    int64_t branch_e = LayerScratch.Load<int64_t>(down_scale_off + 8);
    if (branch_m == 0 || stream_m == 0)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagResidualReconciliationScaleOutOfDomain);
        return;
    }
    if (branch_m < -2147483648LL || branch_m > 2147483647LL ||
        stream_m < -2147483648LL || stream_m > 2147483647LL)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, kTagCarriedScaleMantissaOutOfDomain);
        return;
    }
    uint off_site = layer_base + Layout.Load<uint>(53 * 4);
    int64_t site_m = LayerWeights.Load<int64_t>(off_site + 0);
    int64_t site_e = LayerWeights.Load<int64_t>(off_site + 8);
    bool branch_selected = ResidualBranchGridIsFinerGpu(branch_m, branch_e, stream_m, stream_e);
    int64_t selected_m, selected_e, candidate_status;
    bool candidate_ok = BuildResidualCandidateGpu(t, hidden_size, down_codes_off, attn_stream_off,
                                                   branch_selected, branch_m, branch_e, stream_m,
                                                   stream_e, selected_m, selected_e, candidate_status);
    int64_t incoming_m[kMaxIncoming], incoming_e[kMaxIncoming];
    [unroll] for (int z = 0; z < kMaxIncoming; ++z) { incoming_m[z] = 0; incoming_e[z] = 0; }
    incoming_m[0] = selected_m; incoming_e[0] = selected_e;
    if (candidate_ok)
        candidate_ok = PreflightRequantChainFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m,
                                                       incoming_e, 1, site_m, site_e, candidate_status);
    if (!candidate_ok)
    {
        candidate_ok = BuildResidualCandidateGpu(t, hidden_size, down_codes_off, attn_stream_off,
                                                  !branch_selected, branch_m, branch_e, stream_m,
                                                  stream_e, selected_m, selected_e, candidate_status);
        incoming_m[0] = selected_m; incoming_e[0] = selected_e;
        if (candidate_ok)
            candidate_ok = PreflightRequantChainFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m,
                                                           incoming_e, 1, site_m, site_e, candidate_status);
    }
    if (!candidate_ok)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, candidate_status);
        return;
    }

    uint stream_next_off = ScratchLayout.Load<uint>(19 * 4);
    uint stream_next_scale_off = ScratchLayout.Load<uint>(20 * 4);
    int64_t status_tag;
    RequantChainCheckedFullGpuP(t, WorkScratch, 0u, hidden_size, incoming_m, incoming_e, 1, site_m,
                                 site_e, LayerScratch, stream_next_off, stream_next_scale_off, status_tag);
    if (status_tag != kTagOk)
    {
        if (t == 0) SeqState.Store<int64_t>(sticky_off, status_tag);
        return;
    }
}
#endif  // SSLM_FUSED_BODY_MLP_RESIDUAL_SITE

#if defined(SSLM_FUSED_BODY_COMMIT_SITE)
// ---- commit_site.hlsl's `main` body, moved verbatim (GPU per-token overhead plan rev 3.1 §3.2).
void CommitSiteBody(uint t)
{
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
#endif  // SSLM_FUSED_BODY_COMMIT_SITE

#endif  // SSLM_FUSED_SITE_BODIES_HLSLI
