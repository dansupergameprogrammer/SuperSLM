"""T-1691 build-sequencing session (2026-08-03, Brunel): red-first proof for
the FIRST site kind ported under this session's own per-site-kind
decomposition -- RMSNorm (`rmsnorm_shadow_codes`, shadow_layer_recompute.py).

Two tiers, per this campaign's own established convention (design Sec7 red-
first proof parts 1-2): a hand-computable primitive check (this file's own
`test_isqrt_and_floor_div_*`, using a SEPARATE, dumber reference computed
directly in this file -- not reusing shadow_layer_recompute.py's own code),
then an exact-equality check against the REAL, compiled engine at fixture
scale (`test_rmsnorm_shadow_matches_the_real_engine_at_the_fixture`), driving
the real `out/sslm_layer_trace.exe` with `--site-dump` and comparing this
shadow's own independently-computed codes against the real captured codes,
bit for bit.

Skipped (not failed) if the binary has not been built locally, matching this
tool family's own established convention.
"""

from __future__ import annotations

import os
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
_DRIVER = os.path.join(_REPO_ROOT, "out", "sslm_layer_trace.exe")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_DRIVER),
    reason="out/sslm_layer_trace.exe not built -- run tools\\build_layer_trace.bat")


# --- Tier 1: hand-computable primitives, an INDEPENDENT dumber reference ----


def _dumb_isqrt(n: int) -> int:
    """A third, deliberately naive floor-sqrt: linear search upward from 0.
    Slow, never reused by production code or by shadow_layer_recompute.py --
    exists only to cross-check isqrt_reference's own math.isqrt call at small
    values, by a structurally unrelated method."""
    if n < 0:
        raise ValueError
    r = 0
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


def _dumb_floor_div(a: int, b: int) -> int:
    """A second, naive floor division: Python's own `divmod`, taking only
    the quotient -- a different call than the `//` operator floor_div_i64_
    reference itself uses, though both route through the same interpreter
    primitive; the genuine independent check is `_dumb_isqrt` above and the
    fixture-scale exact-equality test below."""
    q, _r = divmod(a, b)
    return q


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 15, 16, 17, 1_000_000])
def test_isqrt_reference_matches_a_naive_linear_search(n):
    """Small values: cross-checked against an independent, dumber
    linear-search implementation."""
    assert S.isqrt_reference(n) == _dumb_isqrt(n)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 15, 16, 17, 1_000_000, 2**62, 2**62 - 1])
def test_isqrt_reference_satisfies_its_own_defining_property(n):
    """Large values: the linear search is too slow to run in a unit test, so
    this asserts the property that DEFINES "floor square root" directly --
    floor(sqrt(n))^2 <= n < (floor(sqrt(n))+1)^2 -- independent of how
    isqrt_reference itself is implemented."""
    root = S.isqrt_reference(n)
    assert root * root <= n
    assert (root + 1) * (root + 1) > n


@pytest.mark.parametrize("a,b,expected", [
    (7, 2, 3),
    (-7, 2, -4),
    (8, 2, 4),
    (-8, 2, -4),
    (0, 5, 0),
    (-1, 3, -1),
])
def test_floor_div_i64_reference_matches_hand_derived_values(a, b, expected):
    assert S.floor_div_i64_reference(a, b) == expected
    assert S.floor_div_i64_reference(a, b) == _dumb_floor_div(a, b)


def test_rmsnorm_shadow_codes_all_zero_row_hits_the_root_floor():
    """h all-zero: sumsq=0, root would compute to 0 and is floored to 1
    (RmsNormSite's own `root = root > 1 ? root : 1`), so wide is all-zero
    regardless of g -- a hand-derivable degenerate case requiring no
    root-finding at all."""
    h = np.array([0, 0, 0, 0], dtype=np.int8)
    g = np.array([100, -200, 300, 1], dtype=np.int32)
    status, codes = S.rmsnorm_shadow_codes(h, g)
    assert status == "Ok"
    assert np.all(codes == 0), "an all-zero normalized row must requantize to all-zero codes"


# --- Tier 2: exact equality against the REAL compiled engine, fixture scale -


def test_rmsnorm_shadow_matches_the_real_engine_at_the_fixture(tmp_path):
    """The deciding check for this site kind, at the smallest scale the
    real compiled engine can supply: row 5 of kv_context_trailer_fixture.py's
    6-layer geometry (0-indexed layer 4) is the only design-band row this
    fixture reaches. attn_norm's real input is the base dump's own row 4
    (state entering layer 4); mlp_norm's real input is THIS SAME layer's own
    captured attn_residual output (the site-dump's own record for
    "layer4.attn_residual") -- neither is available from the base 29-row
    dump alone, which is exactly the per-site capture gap this session's
    own gating step closed."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    dump_path = str(tmp_path / "rmsnorm.dump")
    site_dump_path = str(tmp_path / "rmsnorm.sitedump")
    proc = subprocess.run(
        [_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path,
         "--site-dump", site_dump_path],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"driver failed: {proc.stdout}\n{proc.stderr}"

    base = load_int8_layer_dump(dump_path)
    site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}
    artifact = read_artifact(str(model_path))

    layer_index = 4  # row 5 == state after committing 0-indexed layer 4
    attn_norm_gain = artifact.layers[layer_index].attn_norm_gain.astype(np.int32)
    mlp_norm_gain = artifact.layers[layer_index].mlp_norm_gain.astype(np.int32)

    # attn_norm: real input is the base dump's row `layer_index` (state
    # entering this layer, before any of its own sites have run).
    h_attn = base.codes[layer_index]
    status, shadow_codes = S.rmsnorm_shadow_codes(h_attn, attn_norm_gain)
    assert status == "Ok"
    real_codes = site_records["layer4.attn_norm"].codes
    assert np.array_equal(shadow_codes, real_codes), (
        f"attn_norm mismatch: shadow={shadow_codes.tolist()} real={real_codes.tolist()}")

    # mlp_norm: real input is THIS layer's own captured attn_residual output
    # -- the post-attention residual stream, never present in the base dump
    # (which only captures full-layer boundaries).
    h_mlp = site_records["layer4.attn_residual"].codes
    status, shadow_codes = S.rmsnorm_shadow_codes(h_mlp, mlp_norm_gain)
    assert status == "Ok"
    real_codes = site_records["layer4.mlp_norm"].codes
    assert np.array_equal(shadow_codes, real_codes), (
        f"mlp_norm mismatch: shadow={shadow_codes.tolist()} real={real_codes.tolist()}")


def test_rmsnorm_shadow_guard_vitality_a_wrong_gain_produces_a_mismatch():
    """Guard vitality, at hand-controlled scale (StandardsDocument Sec5.4:
    verified by execution before a check is trusted). This campaign's own
    6-layer fixture (used by the positive check above) turns out to saturate
    every element to +-127 regardless of the gain array or of a deliberately
    injected rounding-bias defect (`requant_chain_reference`'s own
    `drop_rounding_bias` knob, checked directly during this test's own
    authoring pass and not reproduced here as a passing assertion, since it
    would assert nothing) -- an extreme, degenerate fixture geometry, the
    same class of blind spot this campaign has already named once
    (`SoftmaxRowQ15` returning a constant at width==1, design Sec7 red-first
    proof part 3). This test instead uses a hand-chosen, non-saturating
    8-element case where a wrong gain array is confirmed, by direct
    execution, to change the requantized output."""
    h = np.array([10, -20, 30, -5, 7, -3, 15, -40], dtype=np.int8)
    correct_gain = np.array([1000, 2000, -500, 300, 900, -1200, 50, 700], dtype=np.int32)
    wrong_gain = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)

    status_correct, codes_correct = S.rmsnorm_shadow_codes(h, correct_gain)
    status_wrong, codes_wrong = S.rmsnorm_shadow_codes(h, wrong_gain)
    assert status_correct == "Ok" and status_wrong == "Ok"
    assert not np.array_equal(codes_correct, codes_wrong), (
        "expected a wrong gain array to change the requantized output -- "
        "the exact-equality check has no discriminating power at this input")
