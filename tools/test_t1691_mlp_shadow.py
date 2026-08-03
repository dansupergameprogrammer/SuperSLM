"""T-1691 build-sequencing session, site kind 3/4 (the MLP, build log
Claude/Brunel/superslm-t1691-mlp-site-comparison-build-2026-08-03.md):
red-first proof for `SiluSigmoidQ15`'s own independent reproduction and the
`mlp_act_shadow_codes` composite `shadow_layer_recompute.py` gains.
gate_proj.requant/up_proj.requant/down_proj.requant reuse
`project_and_funnel_shadow_codes` unchanged; mlp_residual reuses
`residual_reconcile_shadow_codes` unchanged -- both already red-first proved
and real-population driven at site kind 2/4 (the attention interior,
D-SLM750/751) -- so this file's own new coverage is scoped to `mlp_act`, this
project's most-scrutinised arithmetic (T-183/D-SLM312's own SwiGLU finding).

Three tiers, this campaign's own established convention:

  1. Primitive-level: `SiluSigmoidQ15` (the whole composite the real engine
     calls, both internal shift branches) against the REAL COMPILED function
     via `tests/t1691_primitive_probe.cpp`'s own new `silu_sigmoid_q15`
     subcommand -- at hand-chosen AND randomized values -- before
     `mlp_act_shadow_codes` trusts it. `CheckSiluCompositionScaleDomain` is a
     bound-comparison-only predicate with no internal state to probe,
     reproduced from source and checked at its own hand-computed boundary
     values, the same convention `check_softmax_row_width_domain_reference`
     (site kind 2/4) already used.
  2. Composite-level guard vitality: hand-controlled, non-fixture cases
     proving `mlp_act_shadow_codes` has genuine discriminating power (a wrong
     gate code, a wrong up code, and a rejecting domain each change the
     result) -- this campaign's own established fallback whenever the
     smallest available real-engine fixture blinds a mechanism.
  3. Composite-level plumbing, against the REAL compiled engine at fixture
     scale (`kv_context_trailer_fixture.py`): gate_proj/up_proj are all-zero
     by this fixture's own construction (D-SLM721's inherited constraint,
     the same construction the attention interior pass's own q/o_proj
     already named) -- `mlp_act`'s own SwiGLU product is therefore
     identically zero at this fixture regardless of `SiluSigmoidQ15`'s
     result (sig(0) folds into a 0*sig*0 product), so this tier proves the
     composition's shapes, indices, and status codes agree with the real
     engine, but has NO discriminating power over the sigmoid-LUT
     arithmetic itself -- tier 1 supplies that, on the real compiled
     function directly, not through this fixture.

Skipped (not failed) if the relevant binary has not been built locally,
matching this tool family's own established convention.
"""

from __future__ import annotations

import os
import random
import struct
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


_RNG = random.Random(20260804)
_CANONICAL_TABLE = None  # populated lazily by _canonical_table(), below


def _canonical_table():
    """The real SIL1 canonical table, read from the real Qwen2.5-1.5B-Instruct
    artifact (`sslm_artifact_reader.py`'s own `sigmoid_lut` field) when it is
    available locally, else the fixture artifact's own SIL1 section (written
    identically by `kv_context_trailer_fixture.py`'s own `CANONICAL`
    reference) -- both are the SAME 1025-node content by construction
    (`SslmModel::Load` pins every loaded SIL1 section against this one
    table, module docstring above), so either source is a legitimate shared
    INPUT for this tier's own probe comparisons."""
    global _CANONICAL_TABLE
    if _CANONICAL_TABLE is None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            _tok_path, model_path = FIX.write_fixture(Path(td))
            _CANONICAL_TABLE = read_artifact(str(model_path)).sigmoid_lut
    return _CANONICAL_TABLE


# =============================================================================
# Tier 1: SiluSigmoidQ15 against the REAL compiled function; the domain check
# against its own hand-computed boundary (no probe -- no internal state).
# =============================================================================


@_probe_skip
@pytest.mark.parametrize("code,m,e", [
    (0, 0, 0),                       # code==0 -> term==0 regardless of m/e -> the midpoint node
    (0, 1 << 30, -14),
    (127, 1 << 30, -30),             # a real calibrated-shape (m,e): sig should saturate high
    (-127, 1 << 30, -30),            # sig should saturate low
    (127, (1 << 31) - 1, 8),         # max |m|, max e (kSiluCompositionRuntimeMaxE) -- left-shift branch
    (-128, (1 << 31) - 1, 8),
    (5, 100, 8),                     # shift = 8+5+12 = 25 >= 0 -- left-shift branch, small term
    (5, 100, -80),                   # shift = -80+5+12 = -63 -- right-divide branch, near its own ceiling
    (1, 1, -17),                     # shift = -17+5+12 = 0 -- the branch boundary itself
    (1, 1, -18),                     # shift = -18+5+12 = -1 -- one step into the divide branch
] + [
    (_RNG.randint(-128, 127), _RNG.randint(-((1 << 31) - 1), (1 << 31) - 1), _RNG.randint(-80, 8))
    for _ in range(200)
])
def test_silu_sigmoid_q15_reference_matches_the_real_compiled_function(code, m, e):
    table = _canonical_table()
    got = S.silu_sigmoid_q15_reference(table, code, m, e)
    want = _probe("silu_sigmoid_q15", str(code), str(m), str(e))
    assert got == int(want["sig"])


@pytest.mark.parametrize("m,e,want", [
    (0, 0, "Ok"),
    (S._SILU_COMPOSITION_MAX_ABS_M, 0, "Ok"),           # |m| at the boundary -- accepted
    (-S._SILU_COMPOSITION_MAX_ABS_M, 0, "Ok"),
    (S._SILU_COMPOSITION_MAX_ABS_M + 1, 0, "SiluCompositionScaleOutOfDomain"),   # one past -- rejected
    (-(S._SILU_COMPOSITION_MAX_ABS_M + 1), 0, "SiluCompositionScaleOutOfDomain"),
    (0, S._SILU_COMPOSITION_RUNTIME_MAX_E, "Ok"),        # e at the upper boundary -- accepted
    (0, S._SILU_COMPOSITION_RUNTIME_MAX_E + 1, "SiluCompositionScaleOutOfDomain"),
    (0, S._SILU_COMPOSITION_RUNTIME_MIN_E, "Ok"),        # e at the lower boundary -- accepted
    (0, S._SILU_COMPOSITION_RUNTIME_MIN_E - 1, "SiluCompositionScaleOutOfDomain"),
])
def test_check_silu_composition_scale_domain_reference_boundary(m, e, want):
    assert S.check_silu_composition_scale_domain_reference(m, e) == want


# =============================================================================
# Tier 2: composite-level guard vitality (hand-controlled, non-fixture).
# =============================================================================


def test_mlp_act_shadow_codes_guard_vitality_wrong_gate_code_changes_output():
    table = _canonical_table()
    gate_correct = np.array([10, -20, 30], dtype=np.int8)
    gate_wrong = np.array([10, -20, 100], dtype=np.int8)
    up = np.array([5, 5, 5], dtype=np.int8)
    status_c, codes_c = S.mlp_act_shadow_codes(gate_correct, 1 << 30, -14, up, table)
    status_w, codes_w = S.mlp_act_shadow_codes(gate_wrong, 1 << 30, -14, up, table)
    assert status_c == "Ok" and status_w == "Ok"
    assert not np.array_equal(codes_c, codes_w), "a wrong gate code must change MlpActSite's output"


def test_mlp_act_shadow_codes_guard_vitality_wrong_up_code_changes_output():
    table = _canonical_table()
    gate = np.array([10, -20, 30], dtype=np.int8)
    up_correct = np.array([5, 5, 5], dtype=np.int8)
    up_wrong = np.array([5, 5, -100], dtype=np.int8)
    status_c, codes_c = S.mlp_act_shadow_codes(gate, 1 << 30, -14, up_correct, table)
    status_w, codes_w = S.mlp_act_shadow_codes(gate, 1 << 30, -14, up_wrong, table)
    assert status_c == "Ok" and status_w == "Ok"
    assert not np.array_equal(codes_c, codes_w), "a wrong up code must change MlpActSite's output"


def test_mlp_act_shadow_codes_guard_vitality_wrong_gate_scale_e_rejects():
    """gate_scale.e outside [kSiluCompositionRuntimeMinE, kSiluCompositionRuntimeMaxE]
    is step 1's own FIRST-ACT rejection -- the whole site returns before
    SiluSigmoidQ15 ever evaluates, distinct from any Ok result the same
    codes could otherwise produce."""
    table = _canonical_table()
    gate = np.array([10, -20, 30], dtype=np.int8)
    up = np.array([5, 5, 5], dtype=np.int8)
    status_ok, _ = S.mlp_act_shadow_codes(gate, 1 << 30, -14, up, table)
    status_bad, codes_bad = S.mlp_act_shadow_codes(gate, 1 << 30, 9, up, table)  # e=9 > max e=8
    assert status_ok == "Ok"
    assert status_bad == "SiluCompositionScaleOutOfDomain"
    assert codes_bad is None


def test_mlp_act_shadow_codes_mismatched_lengths_raise():
    table = _canonical_table()
    gate = np.array([10, -20, 30], dtype=np.int8)
    up = np.array([5, 5], dtype=np.int8)
    with pytest.raises(ValueError):
        S.mlp_act_shadow_codes(gate, 1 << 30, -14, up, table)


# =============================================================================
# Tier 3: composite-level plumbing against the REAL compiled engine, fixture
# scale (all-zero gate/up_proj by this fixture's own construction -- see
# module docstring for what this tier does and does not prove).
# =============================================================================


@_driver_skip
def test_mlp_shadow_matches_the_real_engine_at_the_fixture(tmp_path):
    tok_path, model_path = FIX.write_fixture(tmp_path)
    dump_path = str(tmp_path / "mlp.dump")
    site_dump_path = str(tmp_path / "mlp.sitedump")
    proc = subprocess.run(
        [_LAYER_TRACE_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path,
         "--site-dump", site_dump_path],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"driver failed: {proc.stdout}\n{proc.stderr}"

    base = load_int8_layer_dump(dump_path)
    site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}
    artifact = read_artifact(str(model_path))

    layer_index = 4  # the only design-band row (row 5) this 6-layer fixture reaches
    layer = artifact.layers[layer_index]

    def rec(name: str):
        return site_records[f"layer{layer_index}.{name}"]

    mlp_norm_rec = rec("mlp_norm")

    gate_status, gate_codes = S.project_and_funnel_shadow_codes(
        mlp_norm_rec.codes, mlp_norm_rec.m_out, mlp_norm_rec.e_out, layer.gate_weight,
        layer.gate_fold[:, 0], layer.gate_fold[:, 1], layer.gate_fold[:, 2], None)
    assert gate_status == "Ok"
    assert np.array_equal(gate_codes, rec("gate_proj.requant").codes)

    up_status, up_codes = S.project_and_funnel_shadow_codes(
        mlp_norm_rec.codes, mlp_norm_rec.m_out, mlp_norm_rec.e_out, layer.up_weight,
        layer.up_fold[:, 0], layer.up_fold[:, 1], layer.up_fold[:, 2], None)
    assert up_status == "Ok"
    assert np.array_equal(up_codes, rec("up_proj.requant").codes)

    gate_rec = rec("gate_proj.requant")
    act_status, act_codes = S.mlp_act_shadow_codes(
        gate_codes, gate_rec.m_out, gate_rec.e_out, up_codes, artifact.sigmoid_lut)
    assert act_status == "Ok"
    assert np.array_equal(act_codes, rec("mlp_act").codes), (
        f"mlp_act mismatch at the fixture: shadow={act_codes.tolist() if act_codes is not None else None} "
        f"real={rec('mlp_act').codes.tolist()}")

    act_rec = rec("mlp_act")
    down_status, down_codes = S.project_and_funnel_shadow_codes(
        act_codes, act_rec.m_out, act_rec.e_out, layer.down_weight,
        layer.down_fold[:, 0], layer.down_fold[:, 1], layer.down_fold[:, 2], None)
    assert down_status == "Ok"
    assert np.array_equal(down_codes, rec("down_proj.requant").codes)

    down_rec = rec("down_proj.requant")
    attn_residual_rec = rec("attn_residual")
    res_status, res_codes = S.residual_reconcile_shadow_codes(
        down_codes, down_rec.m_out, down_rec.e_out, attn_residual_rec.codes,
        attn_residual_rec.m_out, attn_residual_rec.e_out, artifact.config["hidden_size"])
    assert res_status == "Ok"
    assert np.array_equal(res_codes, rec("mlp_residual").codes)


def _saturation_trailer_end(dump_path: str, base) -> int:
    """Transcribed independently, matching this tool family's own established
    convention (site kind 2/4's identically-named helper) -- not imported, so
    a defect in one reader cannot silently hide a defect in the other."""
    with open(dump_path, "rb") as f:
        data = f.read()
    rows, hidden_size, _fp, _cm = struct.unpack_from("<QQQQ", data, 0)
    body_end = 32 + rows * (16 + hidden_size)
    (num_deltas,) = struct.unpack_from("<Q", data, body_end)
    return body_end + 8 + num_deltas * 8
