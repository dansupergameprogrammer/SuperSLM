"""T-1691 build-sequencing session, site kind 2/4 (the attention interior,
build log Claude/Brunel/superslm-t1691-site-comparison-build2-2026-08-03.md):
red-first proof for the six new fixed-point primitives and the four composite
site functions `shadow_layer_recompute.py` gains for q_proj.requant/
o_proj.requant (ProjectAndFunnel), attn_ctx (the softmax/RoPE composition),
and attn_residual (ResidualReconcileSite).

Three tiers, extending this campaign's own established two-tier convention
(design Sec7 red-first proof parts 1-2) with the primitive-probe tier T-1691's
own gating step already established for RMSNorm's primitives:

  1. Primitive-level: each of the six new primitives (RopeApplyPair,
     LandingRescale, BiasReconcileWide, CombineCarriedScale,
     IExpScaleConstants, SoftmaxRowQ15) checked against the REAL COMPILED
     function via tests/t1691_primitive_probe.cpp's own six new subcommands
     -- at hand-chosen AND randomized values -- before any composite trusts
     it (design Sec7 red-first proof part 2's own standard).
  2. Composite-level guard vitality: hand-controlled, non-fixture cases
     proving each of the four new site functions has genuine discriminating
     power (a wrong input changes the output) -- this campaign's own
     established fallback whenever the smallest available real-engine
     fixture turns out to blind a mechanism (RMSNorm's own saturating
     fixture, design Sec7 step 3's width==1 corner).
  3. Composite-level plumbing, against the REAL compiled engine at fixture
     scale (`kv_context_trailer_fixture.py`): q_proj/o_proj/gate/up_proj are
     all-zero by this fixture's own construction (D-SLM721's inherited
     constraint, module docstring there) -- this proves the composition's
     shapes, indices, and status codes agree with the real engine, but
     q_proj is all-zero so RoPE'd Q is identically zero and every attention
     score is uniformly 0 regardless of K's correctness: this tier has NO
     discriminating power over the score/softmax path (tier 2 supplies
     that), matching how this campaign has always named a fixture's own
     blind spot rather than silently relying on it for more than it proves.

Skipped (not failed) if the relevant binary has not been built locally,
matching this tool family's own established convention.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_context_trailer_fixture as FIX  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAYER_TRACE_DRIVER = os.path.join(_REPO_ROOT, "out", "sslm_layer_trace.exe")
_PROBE = os.path.join(_REPO_ROOT, "out", "t1691_primitive_probe.exe")

_probe_skip = pytest.mark.skipif(
    not os.path.isfile(_PROBE),
    reason="out/t1691_primitive_probe.exe not built -- run tests\\build_t1691_primitive_probe.bat")
_driver_skip = pytest.mark.skipif(
    not os.path.isfile(_LAYER_TRACE_DRIVER),
    reason="out/sslm_layer_trace.exe not built -- run tools\\build_layer_trace.bat")


def _probe(*args: str) -> dict[str, str]:
    proc = subprocess.run([_PROBE, *args], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"probe {args} failed: {proc.stdout}\n{proc.stderr}"
    out: dict[str, str] = {}
    for tok in proc.stdout.strip().split():
        k, _, v = tok.partition("=")
        out[k] = v
    return out


_RNG = random.Random(20260803)


# =============================================================================
# Tier 1: each new primitive against the REAL compiled function.
# =============================================================================


@_probe_skip
@pytest.mark.parametrize("x,y,cos_q30,sin_q30", [
    (0, 0, 1 << 30, 0),
    (100, -50, 1 << 30, 0),
    (127, -127, 759250125, 613571060),  # cos/sin of an arbitrary Q2.30 angle, cos^2+sin^2 ~= 2^30
    (-128, 127, 0, 1 << 30),
] + [
    (_RNG.randint(-128, 127), _RNG.randint(-128, 127), _RNG.randint(-(1 << 30), 1 << 30),
     _RNG.randint(-(1 << 30), 1 << 30))
    for _ in range(200)
])
def test_rope_apply_pair_reference_matches_the_real_compiled_function(x, y, cos_q30, sin_q30):
    got = S.rope_apply_pair_reference(x, y, cos_q30, sin_q30)
    want = _probe("rope_apply_pair", str(x), str(y), str(cos_q30), str(sin_q30))
    assert got == (int(want["x"]), int(want["y"]))


@_probe_skip
@pytest.mark.parametrize("branch_code,m_a,r_t,e_a,e_t", [
    (0, 1 << 30, 1 << 32, -14, -14),
    (127, 1 << 30, 1 << 32, -14, -14),
    (-127, 1 << 30, 1 << 32, -14, -14),
    (100, 2_000_000_000, 3_000_000_000, 3, 1),
    (100, -2_000_000_000, 3_000_000_000, 3, 1),
] + [
    (_RNG.randint(-127, 127), _RNG.randint(-(1 << 30), 1 << 30), _RNG.randint(1, (1 << 32) - 1),
     _RNG.randint(-20, 20), _RNG.randint(-20, 20))
    for _ in range(200)
])
def test_landing_rescale_reference_matches_the_real_compiled_function(branch_code, m_a, r_t, e_a, e_t):
    raw, exceeded = S.landing_rescale_reference(branch_code, m_a, r_t, e_a, e_t)
    want = _probe("landing_rescale", str(branch_code), str(m_a), str(r_t), str(e_a), str(e_t))
    assert raw == int(want["raw"])
    assert exceeded == bool(int(want["magnitude_exceeded"]))


@_probe_skip
@pytest.mark.parametrize("b,q_b,r_a,e_a", [
    (0, 30, 1 << 31, 0),
    (1000, 30, 2_000_000_000, -50),
    (-1000, 30, 2_000_000_000, -50),
    (1000, 30, 2_000_000_000, 100),  # exponent = 30+62+100 = 192, out of [0,63] -> 0
] + [
    (_RNG.randint(-(1 << 40), 1 << 40), 30, _RNG.randint(1, (1 << 32) - 1), _RNG.randint(-40, 5))
    for _ in range(200)
])
def test_bias_reconcile_reference_matches_the_real_compiled_function(b, q_b, r_a, e_a):
    got = S.bias_reconcile_reference(b, q_b, r_a, e_a)
    want = _probe("bias_reconcile_wide", str(b), str(q_b), str(r_a), str(e_a))
    # BiasReconcile forwards BiasReconcileWide's own written `out` regardless of `fits`.
    assert got == int(want["out"])


@_probe_skip
@pytest.mark.parametrize("a_m,a_e,b_m,b_e", [
    (1 << 30, -30, 1 << 30, -30),
    (1073741824, 5, 1200000000, -3),
    (-2147483647, 0, 2147483647, 0),
] + [
    (_RNG.randint(-(2**31 - 1), 2**31 - 1), _RNG.randint(-40, 40),
     _RNG.randint(-(2**31 - 1), 2**31 - 1), _RNG.randint(-40, 40))
    for _ in range(200)
])
def test_combine_carried_scale_reference_matches_the_real_compiled_function(a_m, a_e, b_m, b_e):
    got = S.combine_carried_scale_reference(a_m, a_e, b_m, b_e)
    want = _probe("combine_carried_scale", str(a_m), str(a_e), str(b_m), str(b_e))
    assert got == (int(want["m"]), int(want["e"]))


@_probe_skip
@pytest.mark.parametrize("m,e", [
    (1 << 30, -14),
    (2 ** 30, 0),
    (2 ** 31 - 1, -100),
] + [
    (_RNG.randint(1 << 30, (1 << 31) - 1), _RNG.randint(-80, 20))
    for _ in range(100)
])
def test_iexp_scale_constants_reference_matches_the_real_compiled_function(m, e):
    domain, triple = S.iexp_scale_constants_reference(m, e, S.kIExpLn2Q, S.kIExpBQ, S.kIExpCaQ)
    want = _probe("iexp_scale_constants", str(m), str(e), str(S.kIExpLn2Q), str(S.kIExpBQ), str(S.kIExpCaQ))
    assert domain == want["domain"]
    if domain == "kOk":
        assert triple == (int(want["q_ln2"]), int(want["q_b"]), int(want["q_c"]))


@_probe_skip
def test_softmax_row_q15_reference_matches_the_real_compiled_function_uniform_row():
    """A real, non-degenerate (q_ln2, q_b, q_c) triple (derived from a
    canonical carried scale via IExpScaleConstants, already validated above)
    at width 5, all-equal scores -- the uniform-softmax corner width==1
    fixtures cannot reach at all (D-SLM503)."""
    domain, triple = S.iexp_scale_constants_reference(1 << 30, -14, S.kIExpLn2Q, S.kIExpBQ, S.kIExpCaQ)
    assert domain == "kOk"
    q_ln2, q_b, q_c = triple
    scores = [0, 0, 0, 0, 0]
    well_formed, probs = S.softmax_row_q15_reference(scores, 5, q_ln2, q_b, q_c)
    want = _probe("softmax_row_q15", str(q_ln2), str(q_b), str(q_c), "5", *[str(s) for s in scores])
    assert well_formed == bool(int(want["well_formed"]))
    assert probs == [int(x) for x in want["probs"].split(",")]


@_probe_skip
def test_softmax_row_q15_reference_matches_the_real_compiled_function_varying_row():
    domain, triple = S.iexp_scale_constants_reference(1 << 30, -14, S.kIExpLn2Q, S.kIExpBQ, S.kIExpCaQ)
    assert domain == "kOk"
    q_ln2, q_b, q_c = triple
    for _ in range(50):
        width = _RNG.randint(1, 12)
        scores = [_RNG.randint(-500, 500) for _ in range(width)]
        well_formed, probs = S.softmax_row_q15_reference(scores, width, q_ln2, q_b, q_c)
        want = _probe("softmax_row_q15", str(q_ln2), str(q_b), str(q_c), str(width),
                       *[str(s) for s in scores])
        assert well_formed == bool(int(want["well_formed"]))
        assert probs == [int(x) for x in want["probs"].split(",")]


# =============================================================================
# Tier 2: composite-level guard vitality (hand-controlled, non-fixture).
# =============================================================================


def test_project_and_funnel_shadow_codes_guard_vitality_wrong_weight_changes_output():
    """identity=1 (pass-through WSC1 fold) so the requantized output tracks
    the GEMM accumulate directly, with no shift free to crush both rows to
    the same near-zero code regardless of which weight matrix was used."""
    in_codes = np.array([10, -20, 30, -5], dtype=np.int8)
    w_correct = np.array([[1, 2, -3, 4], [5, -6, 7, -8]], dtype=np.int8)
    w_wrong = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int8)
    identity = np.array([1, 1], dtype=np.int32)
    mult = np.array([0, 0], dtype=np.int32)
    shift = np.array([0, 0], dtype=np.int32)
    status_c, codes_c = S.project_and_funnel_shadow_codes(in_codes, 1 << 30, -30, w_correct, identity, mult, shift, None)
    status_w, codes_w = S.project_and_funnel_shadow_codes(in_codes, 1 << 30, -30, w_wrong, identity, mult, shift, None)
    assert status_c == "Ok" and status_w == "Ok"
    assert not np.array_equal(codes_c, codes_w), "a wrong weight matrix must change ProjectAndFunnel's output"


def test_project_and_funnel_shadow_codes_guard_vitality_wrong_bias_changes_output():
    """in_scale_e=-70 clears CheckRoundingDivideByPotExponentDomain's own
    gate (exponent = kBiasQFormat(30) + 62 + e_a must land in [0,63]; -70
    gives 22) -- ApplyBiasReconcileRow's own real gate, reproduced by
    apply_bias_reconcile_row_reference, rejects the entire row before this
    test's own bias value is ever inserted at a scale that does not clear
    it (the -14 scale RMSNorm's own toy fixture derives, e.g., does not)."""
    in_codes = np.array([10, -20, 30, -5], dtype=np.int8)
    w = np.array([[1, 2, -3, 4], [5, -6, 7, -8]], dtype=np.int8)
    identity = np.array([1, 1], dtype=np.int32)
    mult = np.array([0, 0], dtype=np.int32)
    shift = np.array([0, 0], dtype=np.int32)
    bias_zero = np.array([0, 0], dtype=np.int64)
    bias_nonzero = np.array([500_000, -300_000], dtype=np.int64)
    status_z, codes_z = S.project_and_funnel_shadow_codes(in_codes, 1 << 30, -70, w, identity, mult, shift, bias_zero)
    status_n, codes_n = S.project_and_funnel_shadow_codes(in_codes, 1 << 30, -70, w, identity, mult, shift, bias_nonzero)
    assert status_z == "Ok" and status_n == "Ok"
    assert not np.array_equal(codes_z, codes_n), "a nonzero bias must change ProjectAndFunnel's output"


def test_rope_apply_site_reference_guard_vitality_nonzero_rotation_changes_row():
    """head_dim=4, a real (non-identity) Q2.30 rotation angle -- a nonzero
    row must differ from its own unrotated codes once cos != 1/sin != 0."""
    row = np.array([100, -50, 30, -80], dtype=np.int8)
    context_cap = 2
    pairs = 2
    cos_table = np.array([[1 << 30, 1 << 30], [759250125, 700000000]], dtype=np.int64)
    sin_table = np.array([[0, 0], [613571060, 500000000]], dtype=np.int64)
    identity_rot = S.rope_apply_site_reference(row, 0, cos_table, sin_table)  # cos=1,sin=0 -> identity
    rotated = S.rope_apply_site_reference(row, 1, cos_table, sin_table)
    assert np.array_equal(identity_rot, row), "cos=1,sin=0 must be an exact identity rotation"
    assert not np.array_equal(rotated, row), "a real rotation angle must change the row"


def test_kv_landing_shadow_codes_guard_vitality_wrong_reciprocal_changes_output():
    """fold_identity=1 (pass-through WSC1 fold) so the landing composite's
    own magnitude tracks `LandingRescale`'s formula directly -- a large
    versus a small landing reciprocal produces a materially different
    magnitude before ClampRopeCode, not both crushed to the same code by an
    unrelated fold shift."""
    in_codes = np.array([50, -30], dtype=np.int8)
    weight = np.array([[1, 0], [0, 1]], dtype=np.int8)  # identity, 1 kv_head x head_dim=2
    fold_identity = np.array([1, 1], dtype=np.int32)
    fold_mult = np.array([0, 0], dtype=np.int32)
    fold_shift = np.array([0, 0], dtype=np.int32)
    r_t_correct = [1 << 32]
    r_t_wrong = [1 << 20]
    e_t = [-14]
    out_correct = S.kv_landing_shadow_codes(in_codes, 1 << 30, -14, weight, fold_identity, fold_mult,
                                             fold_shift, None, r_t_correct, e_t, 1, 2)
    out_wrong = S.kv_landing_shadow_codes(in_codes, 1 << 30, -14, weight, fold_identity, fold_mult,
                                           fold_shift, None, r_t_wrong, e_t, 1, 2)
    assert not np.array_equal(out_correct, out_wrong), "a wrong landing reciprocal must change the landed codes"


def test_residual_reconcile_shadow_codes_guard_vitality_wrong_branch_changes_output():
    stream = np.array([10, 20, 30, 40], dtype=np.int8)
    branch_zero = np.zeros(4, dtype=np.int8)
    branch_nonzero = np.array([50, -50, 60, -60], dtype=np.int8)
    status_z, codes_z = S.residual_reconcile_shadow_codes(branch_zero, 1 << 30, -14, stream, 1 << 30, -14, 4)
    status_n, codes_n = S.residual_reconcile_shadow_codes(branch_nonzero, 1 << 30, -14, stream, 1 << 30, -14, 4)
    assert status_z == "Ok" and status_n == "Ok"
    assert not np.array_equal(codes_z, codes_n), "a nonzero branch must change ResidualReconcileSite's output"


def test_attn_ctx_shadow_codes_guard_vitality_wrong_prior_key_changes_output():
    """num_heads=num_kv_heads=1, head_dim=2, width=2 (one prior position, one
    current). A nonzero Q (so RoPE'd Q is nonzero and the score genuinely
    varies with K) -- proving this composite's own score/softmax path has
    discriminating power the fixture-scale (all-zero-Q) plumbing check
    below cannot supply."""
    class _Layer:
        pass

    layer = _Layer()
    layer.k_weight = np.array([[1, 0], [0, 1]], dtype=np.int8)
    layer.v_weight = np.array([[1, 0], [0, 1]], dtype=np.int8)
    # identity=1 (pass-through WSC1 fold) throughout -- avoids the same
    # shift-crushes-everything-to-zero trap the ProjectAndFunnel/kv_landing
    # guard-vitality tests above hit with a naively-chosen (mult, shift).
    layer.k_fold = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int64)
    layer.v_fold = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int64)
    layer.k_bias = None
    layer.v_bias = None
    layer.kv_landing_r_t_k = [1 << 32]
    layer.kv_landing_e_t_k = [-14]
    layer.kv_landing_r_t_v = [1 << 32]
    layer.kv_landing_e_t_v = [-14]
    layer.iexp_softmax_khead_m = [1 << 30]
    layer.iexp_softmax_khead_e = [-60]
    layer.ctx_fold = np.array([[1, 0, 0]], dtype=np.int64)

    q_codes_real = np.array([80, -40], dtype=np.int8)
    attn_norm_codes_real = np.array([50, -30], dtype=np.int8)
    cos_table = np.ones((2, 1), dtype=np.int64) * (1 << 30)  # identity rotation (cos=1, sin=0)
    sin_table = np.zeros((2, 1), dtype=np.int64)

    real_v_prior_shared = np.array([[[10, 20]]], dtype=np.int8)  # [context_length=1, kv_heads=1, head_dim=2]

    real_k_prior_a = np.array([[[5, 5]]], dtype=np.int8)
    real_k_prior_b = np.array([[[100, -100]]], dtype=np.int8)

    status_a, codes_a = S.attn_ctx_shadow_codes(
        q_codes_real, 1 << 30, -14, attn_norm_codes_real, 1 << 30, -14, 1, 2, layer, cos_table, sin_table,
        real_k_prior_a, real_v_prior_shared, 1, 1, 2)
    status_b, codes_b = S.attn_ctx_shadow_codes(
        q_codes_real, 1 << 30, -14, attn_norm_codes_real, 1 << 30, -14, 1, 2, layer, cos_table, sin_table,
        real_k_prior_b, real_v_prior_shared, 1, 1, 2)
    assert status_a == "Ok" and status_b == "Ok"
    assert not np.array_equal(codes_a, codes_b), (
        "a wrong prior-position K row must change attn_ctx's output once Q is nonzero")


# =============================================================================
# Tier 3: composite-level plumbing against the REAL compiled engine, fixture
# scale (all-zero q/o/gate/up_proj by this fixture's own construction -- see
# module docstring for what this tier does and does not prove).
# =============================================================================


@_driver_skip
def test_attention_interior_shadow_matches_the_real_engine_at_the_fixture(tmp_path):
    tok_path, model_path = FIX.write_fixture(tmp_path)
    dump_path = str(tmp_path / "attn.dump")
    site_dump_path = str(tmp_path / "attn.sitedump")
    proc = subprocess.run(
        [_LAYER_TRACE_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path,
         "--site-dump", site_dump_path],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"driver failed: {proc.stdout}\n{proc.stderr}"

    base = load_int8_layer_dump(dump_path)
    site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}
    artifact = read_artifact(str(model_path))
    trailer = S.load_kv_context_trailer(dump_path, after_offset=_saturation_trailer_end(dump_path, base))

    layer_index = 4  # the only design-band row (row 5) this 6-layer fixture reaches
    layer = artifact.layers[layer_index]
    num_heads = artifact.config["num_attention_heads"]
    num_kv_heads = artifact.config["num_key_value_heads"]
    head_dim = artifact.config["head_dim"]
    position = trailer.context_length  # the current (last-prompt-token) position
    width = position + 1

    attn_norm_real = site_records[f"layer{layer_index}.attn_norm"].codes

    # q_proj (all-zero weight by this fixture's own construction -- trivial,
    # plumbing-only coverage; see this file's own module docstring).
    q_status, q_codes = S.project_and_funnel_shadow_codes(
        attn_norm_real, site_records[f"layer{layer_index}.attn_norm"].m_out,
        site_records[f"layer{layer_index}.attn_norm"].e_out, layer.q_weight,
        layer.q_fold[:, 0], layer.q_fold[:, 1], layer.q_fold[:, 2], layer.q_bias)
    assert q_status == "Ok"
    assert np.array_equal(q_codes, site_records[f"layer{layer_index}.q_proj.requant"].codes)

    q_rec = site_records[f"layer{layer_index}.q_proj.requant"]
    q_scale_m, q_scale_e = q_rec.m_out, q_rec.e_out
    norm_rec = site_records[f"layer{layer_index}.attn_norm"]
    attn_norm_scale_m, attn_norm_scale_e = norm_rec.m_out, norm_rec.e_out

    real_k_prior = trailer.k_codes[0]  # this is the only target layer this geometry reaches
    real_v_prior = trailer.v_codes[0]

    ctx_status, ctx_codes = S.attn_ctx_shadow_codes(
        q_codes, q_scale_m, q_scale_e, attn_norm_real, attn_norm_scale_m, attn_norm_scale_e,
        position, width, layer, artifact.rope_cos, artifact.rope_sin, real_k_prior, real_v_prior,
        num_heads, num_kv_heads, head_dim)
    assert ctx_status == "Ok"
    assert np.array_equal(ctx_codes, site_records[f"layer{layer_index}.attn_ctx"].codes), (
        f"attn_ctx mismatch at the fixture: shadow={ctx_codes.tolist() if ctx_codes is not None else None} "
        f"real={site_records[f'layer{layer_index}.attn_ctx'].codes.tolist()}")

    ctx_rec = site_records[f"layer{layer_index}.attn_ctx"]
    ctx_scale_m, ctx_scale_e = ctx_rec.m_out, ctx_rec.e_out
    o_status, o_codes = S.project_and_funnel_shadow_codes(
        ctx_codes, ctx_scale_m, ctx_scale_e, layer.o_weight, layer.o_fold[:, 0], layer.o_fold[:, 1],
        layer.o_fold[:, 2], None)
    assert o_status == "Ok"
    assert np.array_equal(o_codes, site_records[f"layer{layer_index}.o_proj.requant"].codes)

    o_rec = site_records[f"layer{layer_index}.o_proj.requant"]
    o_scale_m, o_scale_e = o_rec.m_out, o_rec.e_out
    stream_m, stream_e = base.scales[layer_index]
    res_status, res_codes = S.residual_reconcile_shadow_codes(
        o_codes, o_scale_m, o_scale_e, base.codes[layer_index], int(stream_m), int(stream_e),
        artifact.config["hidden_size"])
    assert res_status == "Ok"
    assert np.array_equal(res_codes, site_records[f"layer{layer_index}.attn_residual"].codes)


def _saturation_trailer_end(dump_path: str, base) -> int:
    """The K/V-context trailer is appended after the 29-row body AND T-1689's
    own 28-uint64 saturation-delta trailer (design Sec7 step 4). Computes
    that byte offset directly from the dump's own documented layout, the
    same arithmetic `t1691_rmsnorm_drive.py`'s own sibling tools already use
    -- transcribed here rather than imported, so a defect in one reader
    cannot silently hide a defect in the other (this module's own "no shared
    apparatus" construction, applied to trailer-offset arithmetic)."""
    import struct
    with open(dump_path, "rb") as f:
        data = f.read()
    rows, hidden_size, _fp, _cm = struct.unpack_from("<QQQQ", data, 0)
    body_end = 32 + rows * (16 + hidden_size)
    (num_deltas,) = struct.unpack_from("<Q", data, body_end)
    return body_end + 8 + num_deltas * 8
