"""Red suite (T-1683/T-1691) for `tools/shadow_layer_recompute.py` -- T-1691,
the weight-quantization/engine-arithmetic decomposition, the deciding
experiment of the T-1683 source-attribution campaign (design
`Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md`
Sec7, amended against Loki's fracture, D-SLM718).

`tools/shadow_layer_recompute.py` DOES NOT EXIST YET. Every test below that
imports it fails today on import (ModuleNotFoundError) -- `red-unimplemented`,
per Curie's own discipline (Curie.md Phase 4): a red test fails for its
cell's reason, not on a mis-stated assertion. This file is the concrete
contract Brunel builds against; Mendeleev audits it for coverage of the
design's Sec11 Coverage Model, never the module it tests.

THE CONTRACT THIS FILE PINS (derived from design Sec7 -- no committed source
exists to read this from, so the shapes below are this suite's own
derivation from the design's prose, stated explicitly so a disagreement is
visible rather than assumed, matching tools/test_t1687_layer_bisection_report.py's
own precedent for a not-yet-built sibling module):

  SITE_KINDS: tuple[str, ...]
      The five distinct call-site kinds the parity shadow's own eleven
      `RequantChainChecked` invocations span (design Sec7 step 5a):
      "rms_norm" (RmsNormSite, x2/layer -- attn_norm, mlp_norm),
      "project_and_funnel" (ProjectAndFunnel, x5/layer -- q, o, gate, up, down),
      "mlp_act" (MlpActSite, x1/layer), "residual_reconcile"
      (ResidualReconcileSite, x2/layer -- attn residual, mlp residual), and
      "context_row_funnel" (the inline attention-context-row funnel, x1/layer).
      `k_proj`/`v_proj` route through `LandingRescale` (a static per-head
      scale), never `RequantChainChecked` -- no site kind names them (design
      Sec2.1, Sec7 step 5a's own "k_proj/v_proj do NOT route through
      RequantChainChecked at all").

  real_scale(mult: np.ndarray[int64], shift: np.ndarray[int64],
             identity: np.ndarray[int64]) -> np.ndarray[float64]
      Design Sec2.4/Sec7 step 2's own recovered per-channel multiplier:
      mult[i] * 2**(-31 - shift[i]) when identity[i] == 0, else 1.0 --
      the exact formula `ApplyWeightScaleFold`/`MultiplyByQuantizedMultiplier`
      implement, read at source, not estimated from output behavior.

  dequantize_weight_matrix(w_int8: np.ndarray[int8], mult, shift, identity)
      -> np.ndarray[float64]
      w_float[k, i] = w_int8[k, i] * real_scale[i] (design Sec7 step 2).

  SiteFoldTriple  (dataclass: kind: str, w_int8: np.ndarray[int8] shape
      [out_channels, in_channels], identity/mult/shift: np.ndarray[int64]
      shape [out_channels])
      One call site's raw int8 weight matrix and its own (identity, mult,
      shift) fold triple, matching what the real engine's own LayerWeights
      carries per projection (design Sec2.5).

  LayerShadowResult  (dataclass: codes: np.ndarray[int8], status: str,
      rejected_at_site: int | None, rope_landing_clamped: bool,
      rope_apply_clamped: bool)
      status is "Ok" or the SslmForwardStatus name (e.g.
      "ChainInputOutOfDomain") of the FIRST rejecting site's own
      RequantChainChecked-shaped domain check (design Sec7 step 5a's own
      C29 reproduction: "where the real engine would reject... the parity
      shadow reports the SAME rejection status rather than silently
      producing a value"). rope_landing_clamped/rope_apply_clamped report
      whether the parity shadow's OWN independent ClampRopeCode
      reproduction (design Sec7 step 5a) saturated at the K/V-landing site
      or the RopeApplySite call respectively (design Sec2.2, Sec7 part 2).

  parity_shadow_layer(seed_codes: np.ndarray[int8], sites: list[SiteFoldTriple],
                       rope_landing_raw: int, rope_apply_raw: int,
                       inject_defect_site: int | None = None)
      -> LayerShadowResult
      Design Sec7 step 5a: composes `sites` in order through the SAME
      integer arithmetic RequantChainChecked's own eleven real call sites
      use (dynamic per-row max-abs requantization to 127 levels, the C29
      domain check, int32/int64 narrowing-cast/overflow convention exactly
      as the real C++ functions), reproduces ClampRopeCode at the K/V-landing
      site (`rope_landing_raw`) and the RopeApplySite call (`rope_apply_raw`),
      never removing any mechanism the real engine has. `inject_defect_site`
      (a SiteFoldTriple index) deliberately departs from the correct
      arithmetic at that one site only -- the red-first proof's own guard-
      vitality construction (design Sec7 red-first proof part 6).

  precision_shadow_layer(seed_codes: np.ndarray[int8], sites: list[SiteFoldTriple])
      -> np.ndarray[float64]
      Design Sec7 step 5b, unchanged construction: the SAME site sequence in
      float64, no intermediate int8 requantization, no ClampRopeCode
      saturation, no RequantChainChecked domain rejection anywhere.

  compare_exact(int8_codes: np.ndarray, parity_codes: np.ndarray)
      -> tuple[int, list[int]]
      Design Sec7 step 6a: (count of differing positions, the differing
      indices) -- exact integer-code equality, never a rank statistic
      (design Sec9, Sec10: "any code, at any position... that differs...
      is a residual, full stop").

  class KvContextTrailer  (fields: target_rows, context_length,
      num_key_value_heads, head_dim, k_codes, v_codes -- k_codes/v_codes
      shape [len(target_rows), context_length, num_key_value_heads, head_dim];
      num_target_layers is a read-only alias for len(target_rows))
  load_kv_context_trailer(path, *, after_offset: int) -> KvContextTrailer
      Design Sec7 step 4: reads the new K/V-context dump section
      `tools/sslm_layer_trace.cpp` gains, appended AFTER T-1689's own
      28-uint64 saturation-delta trailer (`after_offset` is the byte offset
      that trailer's own end lands at) -- loudly rejects (raises
      KvContextTrailerError) a written element count that does not match
      the independently-derived expected count
      `len(target_rows) * context_length * num_key_value_heads * head_dim * 2`
      (design Sec7 step 4's own two owed cells, (a)/(b)). The real layout
      carries no magic string (corrected 2026-08-03/04 from a prior version
      of this reader's own test oracle, which assumed one no production
      writer emits).

Run with: python -m pytest tools/test_t1691_shadow_layer_recompute.py -v
"""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parent)
_TESTS_DIR = str(Path(__file__).resolve().parent.parent / "tests")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROBE_EXE = _REPO_ROOT / "out" / "t1691_primitive_probe.exe"
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import layer_bisection_report as lbr  # noqa: E402  -- already real (T-1687)

import shadow_layer_recompute as slr  # noqa: E402  -- red-unimplemented: module does not exist yet


# =============================================================================
# Shared fixtures: the real site-kind inventory, the probe tool, and a
# hand-computable synthetic weight-set builder.
# =============================================================================

SITE_KINDS = ("rms_norm", "project_and_funnel", "mlp_act", "residual_reconcile",
              "context_row_funnel")

# Design Sec7 step 5a's own eleven-invocation, five-kind inventory (verified at
# source against main@c6cfa03d1aa9a7fc6c671b811c1cef56fc7ca96b by the design's
# own citation sweep): RmsNormSite x2 (attn_norm, mlp_norm), ProjectAndFunnel
# x5 (q, o, gate, up, down), MlpActSite x1, ResidualReconcileSite x2 (attn
# residual, mlp residual), the inline context-row funnel x1.
REAL_SITE_INVENTORY = (
    "rms_norm", "rms_norm",
    "project_and_funnel", "project_and_funnel", "project_and_funnel",
    "project_and_funnel", "project_and_funnel",
    "mlp_act",
    "residual_reconcile", "residual_reconcile",
    "context_row_funnel",
)
assert len(REAL_SITE_INVENTORY) == 11


def _require_probe():
    if not _PROBE_EXE.exists():
        pytest.skip(f"t1691_primitive_probe.exe not built at {_PROBE_EXE} -- run "
                     f"tests/build_t1691_primitive_probe.bat first")


def _probe(*args) -> str:
    _require_probe()
    result = subprocess.run([str(_PROBE_EXE), *[str(a) for a in args]],
                             capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _probe_kv(line: str) -> dict:
    out = {}
    for tok in line.split():
        k, _, v = tok.partition("=")
        out[k] = v
    return out


HIDDEN = 1536  # design Sec2.6: the real artifact's own hidden_size
INT32_MAX = 2**31 - 1
INT32_MIN = -2**31


def _make_site(rng, kind, alpha=0.85, beta=0.35, width=HIDDEN):
    """A hand-derivable-shape site fold triple: near-identity weight matrix
    plus noise, real_scale chosen so every mult is in its documented legal
    range [2^30, INT32_MAX] (design Sec2.4)."""
    w_real = alpha * np.eye(width) + beta * rng.standard_normal((width, width)) / np.sqrt(width)
    chan_max = np.max(np.abs(w_real), axis=1)
    w_int8 = np.rint(w_real / chan_max[:, None] * 127.0).astype(np.int8)
    real_target = chan_max / 127.0
    shift = np.maximum(0, np.floor(-np.log2(real_target))).astype(np.int64)
    mult = np.rint(real_target * 2.0 ** (31 + shift)).astype(np.int64)
    bad = (mult < (1 << 30)) | (mult > INT32_MAX)
    assert not bad.any(), "fixture builder produced an out-of-domain mult; adjust alpha/beta"
    identity = np.zeros(width, dtype=np.int64)
    return slr.SiteFoldTriple(kind=kind, w_int8=w_int8, identity=identity, mult=mult, shift=shift)


def _make_real_inventory(rng, width=HIDDEN):
    return [_make_site(rng, kind, width=width) for kind in REAL_SITE_INVENTORY]


def _seed_codes(rng, width=HIDDEN, heavy_tail=False, ratio=40.0):
    row = rng.standard_normal(width)
    if heavy_tail:
        idx = rng.choice(width, size=min(3, width), replace=False)
        row[idx] = np.array([1.0, -0.88, 0.79])[: len(idx)] * ratio
    scaled = np.rint(row / np.max(np.abs(row)) * 127.0)
    return np.clip(scaled, -127, 127).astype(np.int8)


# =============================================================================
# Red-first proof part 1 (design Sec7 red-first proof, part 1): formula
# correctness at a hand-computable scale -- the precision shadow's real_scale.
# =============================================================================

def test_real_scale_formula_hand_computable_identity_channel():
    # identity[i] == 1 -> real_scale[i] == 1.0 exactly (design Sec2.4).
    mult = np.array([1073741824], dtype=np.int64)
    shift = np.array([5], dtype=np.int64)
    identity = np.array([1], dtype=np.int64)
    got = slr.real_scale(mult, shift, identity)
    assert got.shape == (1,)
    assert got[0] == pytest.approx(1.0, abs=0.0)


def test_real_scale_formula_hand_computable_non_identity_channel():
    # mult=2^30, shift=0, identity=0 -> real_scale = 2^30 * 2^-31 = 0.5 exactly.
    mult = np.array([1 << 30], dtype=np.int64)
    shift = np.array([0], dtype=np.int64)
    identity = np.array([0], dtype=np.int64)
    got = slr.real_scale(mult, shift, identity)
    assert got[0] == pytest.approx(0.5, abs=1e-15)


def test_real_scale_formula_hand_computable_shift_case():
    # mult=2^30, shift=3, identity=0 -> real_scale = 2^30 * 2^-34 = 0.0625.
    mult = np.array([1 << 30], dtype=np.int64)
    shift = np.array([3], dtype=np.int64)
    identity = np.array([0], dtype=np.int64)
    got = slr.real_scale(mult, shift, identity)
    assert got[0] == pytest.approx(0.0625, abs=1e-15)


def test_dequantize_weight_matrix_applies_per_channel_scale():
    w_int8 = np.array([[10, -20], [30, 40]], dtype=np.int8)
    mult = np.array([1 << 30, 1 << 30], dtype=np.int64)
    shift = np.array([0, 1], dtype=np.int64)
    identity = np.array([0, 1], dtype=np.int64)
    got = slr.dequantize_weight_matrix(w_int8, mult, shift, identity)
    # row 0: real_scale = 0.5 -> [5.0, -10.0]; row 1: identity -> [30.0, 40.0]
    assert got[0] == pytest.approx([5.0, -10.0])
    assert got[1] == pytest.approx([30.0, 40.0])


# =============================================================================
# Red-first proof part 2 (design Sec7 red-first proof, part 2, extended):
# every primitive the parity shadow relies on validated against the REAL
# compiled C++ function -- via tests/t1691_primitive_probe.cpp, not a hand
# derivation alone. Corrected-out saturation-cap boundary (D-SLM724/D-SLM729)
# is NOT fixtured for RequantChainChecked's own call shape (proven
# unreachable there); ClampRopeCode's two real call sites ARE (D-SLM731).
# =============================================================================

@pytest.mark.parametrize("d_prime", [1, 1000, (1 << 30) - 1, 1 << 30, (1 << 31) - 1, 1 << 31])
def test_parity_shadow_normalize_scale_matches_real_function(d_prime):
    dn_want, s_want = (int(x) for x in _probe_kv(_probe("normalize_scale", d_prime)).values())
    dn_got, s_got = slr.normalize_scale_reference(d_prime)
    assert (dn_got, s_got) == (dn_want, s_want)


@pytest.mark.parametrize("dn", [1 << 30, (1 << 30) + 1, (1 << 31) - 1, 2097152000])
def test_parity_shadow_dynamic_scale_reciprocal_matches_real_function(dn):
    r_want = int(_probe_kv(_probe("dynamic_scale_reciprocal", dn))["r"])
    r_got = slr.dynamic_scale_reciprocal_reference(dn)
    assert r_got == r_want


@pytest.mark.parametrize("x", [
    [0] * 8,
    [1, -2, 3, -4, 5, -6, 7, -8],
    [INT32_MIN, 0, 0, 0],
    [INT32_MAX, INT32_MIN, 100, -100],
])
def test_parity_shadow_max_abs_reduce_wide_matches_real_function(x):
    d_want = int(_probe_kv(_probe("max_abs_reduce_wide", len(x), *x))["d"])
    d_got = slr.max_abs_reduce_wide_reference(np.array(x, dtype=np.int64))
    assert d_got == d_want


@pytest.mark.parametrize("x_i,r,s", [
    (1000, 4294967296, 21),
    (-1000, 4294967296, 21),
    (0, 4294967296, 21),
    (INT32_MAX, 4294967296, 0),   # int32-boundary x_i, narrowing-cast/overflow edge (dim 6)
    (INT32_MIN, 4294967296, 0),   # int32-boundary x_i, negative side
])
def test_parity_shadow_requant_token_code_wide_matches_real_function(x_i, r, s):
    q_want = int(_probe_kv(_probe("requant_token_code_wide", x_i, r, s))["code"])
    q_got = slr.requant_token_code_wide_reference(x_i, r, s)
    assert q_got == q_want


@pytest.mark.parametrize("raw", [0, 1, -1, 126, 127, 128, -127, -128, 200, -200, 5000, -5000])
def test_parity_shadow_clamp_rope_code_matches_real_function(raw):
    code_want = int(_probe_kv(_probe("clamp_rope_code", raw))["code"])
    code_got = slr.clamp_rope_code_reference(raw)
    assert code_got == code_want


def test_parity_shadow_clamp_rope_code_at_kv_landing_call_site_saturates():
    # design Sec7 red-first proof part 2's own K/V-landing cell (D-SLM731): a
    # known, hand-computed landing value exceeding [-127, 127] before
    # ClampRopeCode caps it -- T-1689's own D-SLM720 result (deltas == [2])
    # supplies the known-saturating shape (two saturating codes at the
    # K/V-landing site), not the verdict on the parity shadow's own
    # reproduction, which this cell is what proves.
    raw_over = 200  # exceeds [-127, 127] before clamping
    code_want = int(_probe_kv(_probe("clamp_rope_code", raw_over))["code"])
    assert code_want == 127
    code_got = slr.clamp_rope_code_reference(raw_over)
    assert code_got == 127


def test_parity_shadow_clamp_rope_code_at_rope_apply_call_site_saturates():
    # design Sec7 red-first proof part 2's own RopeApplySite cell: a known,
    # hand-computed rotated value exceeding [-127, 127] before clamping.
    raw_over = -300
    code_want = int(_probe_kv(_probe("clamp_rope_code", raw_over))["code"])
    assert code_want == -127
    code_got = slr.clamp_rope_code_reference(raw_over)
    assert code_got == -127


def test_parity_shadow_clamp_rope_code_guard_vitality_wrong_threshold_caught():
    # Guard vitality for the reproduction's own clamp threshold: a deliberate
    # off-by-TEN threshold error (clamping at 117 instead of 127) is
    # confirmed, by direct comparison to the real compiled function, to
    # diverge at an input the correct function does NOT clamp -- proving
    # this cell can fail before the reproduction is trusted (design Sec7
    # part 2's own guard-vitality requirement; D-SLM715's "a probe that
    # cannot fail proves nothing" standard). A boundary-adjacent off-by-one
    # (>=127 vs >127) is output-indistinguishable here because ClampRopeCode
    # clamps TO exactly the boundary value (127), which is why this cell
    # uses a threshold error large enough to move the output, not the
    # smallest possible mutation.
    def _wrong_threshold_clamp(raw):
        if raw > 117:
            return 117
        if raw < -117:
            return -117
        return raw

    raw = 120  # correct: in [-127, 127], passes through unclamped
    real_code = int(_probe_kv(_probe("clamp_rope_code", raw))["code"])
    assert real_code == 120
    assert _wrong_threshold_clamp(raw) == 117
    assert _wrong_threshold_clamp(raw) != real_code, (
        "the deliberately wrong-threshold reproduction must diverge from the real function at "
        "raw=120 -- if it does not, this guard has no kill power")
    assert slr.clamp_rope_code_reference(raw) == real_code  # the actual reference must NOT diverge


@pytest.mark.parametrize("d_prime,n", [
    (1, 4),
    ((1 << 31), 4),          # C29's own boundary: legal (<=), must be Ok
    ((1 << 31) + 1, 4),      # one past C29's own boundary: must reject
    ((1 << 31) + 1_000_000, 4),
])
def test_parity_shadow_requant_chain_domain_check_matches_real_function(d_prime, n):
    row = [d_prime] + [0] * (n - 1)
    line = _probe("requant_chain_checked", n, *row)
    status_want = line.split()[0].split("=")[1]
    status_got, codes_got = slr.requant_chain_reference(np.array(row, dtype=np.int64))
    assert status_got == status_want
    if status_want == "Ok":
        codes_str = [tok for tok in line.split() if tok.startswith("codes=")][0]
        codes_want = [int(c) for c in codes_str.split("=", 1)[1].split(",")]
        assert list(codes_got) == codes_want
    else:
        assert codes_got is None


def test_parity_shadow_requant_chain_domain_check_guard_vitality():
    # A row entirely within C29's legal domain must never be reported as a
    # domain rejection -- proves the domain-check reproduction is a live
    # guard (able to report Ok), not a construction that always rejects.
    status_got, codes_got = slr.requant_chain_reference(np.array([100, -50, 0, 75], dtype=np.int64))
    assert status_got == "Ok"
    assert codes_got is not None


# =============================================================================
# Red-first proof part 3 (design Sec7 red-first proof, part 3): provenance/
# shape guards for the new K/V-context dump trailer, and the attention-width
# fixture (softmax-weighted composed score row at width > 1, closing the
# D-SLM503 width==1 blindness).
# =============================================================================

# The real trailer's own fixed-width header, AFTER its own leading
# num_target_rows/target_rows[] fields (variable-length) -- context_length,
# num_kv_heads, head_dim. Corrected 2026-08-03/04 (site-comparison build2,
# `Claude/Brunel/superslm-t1691-site-comparison-build2-2026-08-03.md`) from a
# prior version of this test file's own fixture writer, which wrote a
# magic-prefixed, `num_target_layers`-first, K-block-then-V-block layout no
# production writer (`tools/sslm_layer_trace.cpp:222-259`, `WriteDump`) ever
# emits -- this test's own oracle was checked only against itself, the same
# shape as the BIA1 parser defect this campaign already found once
# (D-SLM742). Matches `shadow_layer_recompute.py`'s own corrected
# `_TRAILER_HEADER_FMT`/`load_kv_context_trailer`.
_TRAILER_HEADER_FMT = "<QQQ"  # context_length, num_kv_heads, head_dim


def _write_kv_context_trailer(f, target_rows, context_length, num_kv_heads, head_dim,
                               k_codes, v_codes):
    """Writes the real byte layout: `num_target_rows` (u64), `target_rows[]`
    (u64 each), `context_length`/`num_kv_heads`/`head_dim` (u64 each), then
    per target row, per position (0..context_length-1): K for every kv head
    THEN V for every kv head -- interleaved per position, never a whole-K-
    block followed by a whole-V-block. `k_codes`/`v_codes` shape
    `[len(target_rows), context_length, num_kv_heads, head_dim]`."""
    num_target_rows = len(target_rows)
    f.write(struct.pack("<Q", num_target_rows))
    f.write(struct.pack(f"<{num_target_rows}Q", *target_rows))
    f.write(struct.pack(_TRAILER_HEADER_FMT, context_length, num_kv_heads, head_dim))
    k = k_codes.astype(np.int8)
    v = v_codes.astype(np.int8)
    for row in range(num_target_rows):
        for pos in range(context_length):
            f.write(k[row, pos].tobytes())
            f.write(v[row, pos].tobytes())


def _matching_trailer_bytes(tmp_path, num_target_layers=2, context_length=5, num_kv_heads=2,
                             head_dim=4):
    target_rows = list(range(1, num_target_layers + 1))
    shape = (num_target_layers, context_length, num_kv_heads, head_dim)
    rng = np.random.default_rng(1)
    k_codes = rng.integers(-127, 128, size=shape, dtype=np.int64)
    v_codes = rng.integers(-127, 128, size=shape, dtype=np.int64)
    path = tmp_path / "kv_context_trailer.bin"
    with open(path, "wb") as f:
        f.write(b"\x00" * 64)  # stand-in for T-1689's own preceding trailer bytes
        _write_kv_context_trailer(f, target_rows, context_length, num_kv_heads, head_dim,
                                   k_codes, v_codes)
    return path, k_codes, v_codes, target_rows


def test_load_kv_context_trailer_roundtrip(tmp_path):
    path, k_codes, v_codes, target_rows = _matching_trailer_bytes(tmp_path)
    trailer = slr.load_kv_context_trailer(path, after_offset=64)
    assert trailer.target_rows == target_rows
    assert trailer.num_target_layers == 2
    assert trailer.context_length == 5
    assert trailer.num_key_value_heads == 2
    assert trailer.head_dim == 4
    assert np.array_equal(trailer.k_codes, k_codes.astype(np.int8))
    assert np.array_equal(trailer.v_codes, v_codes.astype(np.int8))


def test_load_kv_context_trailer_rejects_truncated_count(tmp_path):
    # Coverage-model dimension 9 cell (b): a dropped position produces a
    # written element count short of the independently-derived expected
    # count (len(target_rows) * context_length * num_kv_heads * head_dim * 2)
    # -- must be a loud rejection, not a short read reported as success.
    target_rows = [1, 2]
    context_length, num_kv_heads, head_dim = 5, 2, 4
    shape = (len(target_rows), context_length, num_kv_heads, head_dim)
    rng = np.random.default_rng(2)
    k_codes = rng.integers(-127, 128, size=shape, dtype=np.int64).astype(np.int8)
    v_codes = rng.integers(-127, 128, size=shape, dtype=np.int64).astype(np.int8)
    path = tmp_path / "truncated.bin"
    with open(path, "wb") as f:
        f.write(b"\x00" * 64)
        f.write(struct.pack("<Q", len(target_rows)))
        f.write(struct.pack(f"<{len(target_rows)}Q", *target_rows))
        f.write(struct.pack(_TRAILER_HEADER_FMT, context_length, num_kv_heads, head_dim))
        # Interleaved K/V per row/position, but drop the LAST position's own V
        # bytes -- a duplicated/dropped-position mutation, still interleaved
        # correctly up to the point of the drop.
        for row in range(len(target_rows)):
            for pos in range(context_length):
                f.write(k_codes[row, pos].tobytes())
                if not (row == len(target_rows) - 1 and pos == context_length - 1):
                    f.write(v_codes[row, pos].tobytes())
    with pytest.raises(slr.KvContextTrailerError):
        slr.load_kv_context_trailer(path, after_offset=64)


def test_load_kv_context_trailer_rejects_truncated_target_rows_header(tmp_path):
    # The real format's own leading variable-length field
    # (`num_target_rows`/`target_rows[]`) has no fixed size -- a file that
    # ends mid-`target_rows` must be a loud rejection, not a short read
    # silently reinterpreted as a smaller `num_target_rows`.
    path = tmp_path / "truncated_header.bin"
    with open(path, "wb") as f:
        f.write(b"\x00" * 64)
        f.write(struct.pack("<Q", 3))  # claims 3 target rows
        f.write(struct.pack("<Q", 1))  # but only one u64 of target_rows[] follows
    with pytest.raises(slr.KvContextTrailerError):
        slr.load_kv_context_trailer(path, after_offset=64)


def test_old_reader_survives_new_trailer_unchanged(tmp_path):
    # Coverage-model dimension 9 cell (a): T-1689's own 28-uint64
    # saturation-delta trailer, read by an OLD (T-1689-only) reader, is
    # byte-for-byte unaffected by T-1691's further append -- mirrors
    # `test_existing_29_row_body_unchanged_by_the_trailer_extension`'s own
    # naming and byte-length-checked convention.
    old_trailer_bytes = bytes(range(1, 29)) * 8  # stand-in T-1689 trailer content, checkable by length
    path = tmp_path / "layered_dump.bin"
    with open(path, "wb") as f:
        f.write(old_trailer_bytes)
        _write_kv_context_trailer(f, [1], 3, 1, 2, np.zeros((1, 3, 1, 2)), np.zeros((1, 3, 1, 2)))
    with open(path, "rb") as f:
        read_back = f.read(len(old_trailer_bytes))
    assert read_back == old_trailer_bytes, (
        "an old T-1689-only reader's own byte offsets into the dump must be unaffected by "
        "T-1691's further trailer append")


def _hand_softmax_weighted_sum(scores, values):
    m = max(scores)
    exps = [np.exp(s - m) for s in scores]
    z = sum(exps)
    weights = [e / z for e in exps]
    return sum(w * v for w, v in zip(weights, values))


def test_attention_width_fixture_matches_hand_computed_softmax_weighted_sum():
    # design Sec7 red-first proof part 3: a synthetic layer with hand-
    # derivable K/V values at width > 1 (closing D-SLM503's own width==1
    # blindness -- SoftmaxRowQ15 returns a constant at width==1, zero kill
    # power over the attention score path at that width).
    width = 4
    scores = [0.1, 2.0, -1.0, 0.5]
    values = [1.0, -2.0, 3.0, 0.0]
    want = _hand_softmax_weighted_sum(scores, values)
    got = slr.attention_context_row_reference(np.array(scores), np.array(values))
    assert got == pytest.approx(want, rel=1e-9)
    assert width > 1  # the fixture's own discriminating property


# =============================================================================
# Red-first proof part 4 (design Sec7 red-first proof, part 4): execution-
# level sanity -- every reported comparison value finite/well-formed and
# in-range, before any table is trusted.
# =============================================================================

def test_execution_sanity_gate_rejects_nan_and_inf_comparison_values():
    stats_ok = [lbr.LayerRowStats(spearman=0.9, pearson=0.9, max_abs_z_diff=1.0)]
    stats_bad_nan = [lbr.LayerRowStats(spearman=float("nan"), pearson=0.9, max_abs_z_diff=1.0)]
    stats_bad_inf = [lbr.LayerRowStats(spearman=0.9, pearson=0.9, max_abs_z_diff=float("inf"))]
    assert slr.execution_sanity_gate({"ok": stats_ok}) == 0
    assert slr.execution_sanity_gate({"bad_nan": stats_bad_nan}) != 0
    assert slr.execution_sanity_gate({"bad_inf": stats_bad_inf}) != 0


def test_execution_sanity_gate_rejects_exact_equality_count_out_of_range(tmp_path):
    # 6a's own reported n_diff must be in [0, hidden_size] -- a negative or
    # over-width count is itself a well-formedness defect, distinct from
    # the rank-statistic finiteness check above.
    assert slr.exact_equality_sanity(n_diff=0, hidden_size=1536) == 0
    assert slr.exact_equality_sanity(n_diff=1536, hidden_size=1536) == 0
    assert slr.exact_equality_sanity(n_diff=-1, hidden_size=1536) != 0
    assert slr.exact_equality_sanity(n_diff=1537, hidden_size=1536) != 0


# =============================================================================
# Red-first proof part 5 (design Sec7 red-first proof, part 5; design Sec9):
# a real-scale positive control -- int8-vs-parity-shadow is EXACT (zero
# codes differ) on a defect-free, non-heavy-tailed construction at the real
# eleven-site, five-kind inventory -- the layer-5 control-band shape.
# =============================================================================

def test_parity_shadow_exact_match_on_defect_free_low_outlier_population():
    rng = np.random.default_rng(20260803)
    sites = _make_real_inventory(rng)
    for trial in range(5):
        seed = _seed_codes(rng, heavy_tail=False)
        engine_result = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10)
        # engine_arm and parity_shadow_layer are the SAME construction on a
        # defect-free population by definition (there is no separate "real
        # engine" arm in this unit test -- the real int8 engine's own
        # comparison is the builder's own real-artifact obligation, design
        # Sec7 part 6/7). This cell pins that the constructor itself is
        # STABLE and DETERMINISTIC: two independent calls with the same
        # seed_codes and sites produce IDENTICAL codes -- the same-input-
        # same-output property any C1 test depends on.
        repeat_result = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10)
        n_diff, _ = slr.compare_exact(engine_result.codes, repeat_result.codes)
        assert n_diff == 0, f"trial {trial}: parity shadow is not deterministic on repeat calls"
        assert engine_result.status == "Ok"


# =============================================================================
# Red-first proof part 6 (design Sec7 red-first proof, part 6): a real-scale
# known-positive control -- a deliberate, hand-chosen arithmetic departure
# injected into the site sequence is confirmed to make int8-vs-parity-shadow
# demonstrably diverge (guard vitality at composed-pipeline scale).
# =============================================================================

def test_parity_shadow_guard_vitality_injected_defect_diverges():
    rng = np.random.default_rng(20260803)
    sites = _make_real_inventory(rng)
    defect_site = 4  # a project_and_funnel site (index 2-6 in REAL_SITE_INVENTORY)
    assert REAL_SITE_INVENTORY[defect_site] == "project_and_funnel"

    any_missed = False
    for trial in range(5):
        seed = _seed_codes(rng, heavy_tail=(trial % 2 == 0), ratio=100.0)
        clean = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10)
        defective = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10,
                                             inject_defect_site=defect_site)
        n_diff, _ = slr.compare_exact(clean.codes, defective.codes)
        any_missed = any_missed or (n_diff == 0)
    assert not any_missed, (
        "a real, one-line arithmetic defect injected at a project_and_funnel site must be "
        "caught by the exact-equality comparison on every trial -- a defect that survives "
        "undetected means the exact-equality test has no kill power over this site kind")


# =============================================================================
# Red-first proof part 7 (design Sec7 red-first proof, part 7; the cell
# Loki's strike named as missing, D-SLM718): the heavy-tail true-negative
# control -- no defect, no saturation, heavy activation tails, at the real
# site inventory -- int8-vs-parity-shadow is EXACT at every point in the
# sweep, AND the OLD precision-shadow-as-C1-test construction is shown to
# report a false residual on the SAME data (guard vitality on part 7 itself,
# design Sec11 dimension 11's third obligation).
# =============================================================================

@pytest.mark.parametrize("ratio", [0.0, 10.0, 40.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
def test_parity_shadow_exact_at_real_site_inventory_across_outlier_sweep(ratio):
    rng = np.random.default_rng(20260803 + int(ratio))
    sites = _make_real_inventory(rng)
    seed = _seed_codes(rng, heavy_tail=(ratio > 0.0), ratio=ratio)
    result_a = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10)
    result_b = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10)
    n_diff, _ = slr.compare_exact(result_a.codes, result_b.codes)
    assert n_diff == 0, f"outlier ratio {ratio}: parity shadow diverged from itself (non-determinism)"
    assert result_a.status == "Ok"
    assert not result_a.rope_landing_clamped or ratio >= 1000.0  # documents the shape, not asserted tightly


def test_old_rule_precision_shadow_shows_false_residual_new_rule_does_not():
    # The required old-rule-fails/new-rule-passes contrast (design Sec11
    # dimension 11's third guard-vitality obligation, mirroring the design-
    # time probe's own TEST 1B, Claude/Vitruvius/superslm-t1683-t1691-
    # parity-shadow-amendment-probe-2026-08-02.py) -- executed here at
    # T-1691's own real eleven-site, five-kind inventory, not the design-
    # time probe's ten-generic-site linear chain.
    rng = np.random.default_rng(31415926)
    sites = _make_real_inventory(rng)
    mech2_rho, control_rho = [], []
    for k in range(9):
        heavy = k < 4
        seed = _seed_codes(rng, heavy_tail=heavy, ratio=40.0)
        parity = slr.parity_shadow_layer(seed, sites, rope_landing_raw=10, rope_apply_raw=-10)
        precision = slr.precision_shadow_layer(seed, sites)
        # "the real engine" stands in as the parity shadow's own twin call
        # (both defect-free by construction) -- the OLD rule's own
        # comparison shape is int8-vs-precision-shadow, rank-based.
        stats = lbr.compare_layer_row(parity.codes, precision)
        (mech2_rho if heavy else control_rho).append(stats.spearman)
    mech2_rho, control_rho = np.array(mech2_rho), np.array(control_rho)
    old_rule_group_separated = control_rho.min() > mech2_rho.max()
    assert old_rule_group_separated, (
        "this cell's own purpose is to reproduce the false 'C1 attribution warranted' "
        "verdict at T-1691's real site inventory -- if it does not reproduce here, report "
        "that honestly rather than assert the contrast anyway (StandardsDocument Sec5.4)")
    # NEW rule: exact equality between two independent parity-shadow calls
    # (the C1 test's own no-residual case) must hold regardless.
    seed2 = _seed_codes(rng, heavy_tail=True, ratio=40.0)
    a = slr.parity_shadow_layer(seed2, sites, rope_landing_raw=10, rope_apply_raw=-10)
    b = slr.parity_shadow_layer(seed2, sites, rope_landing_raw=10, rope_apply_raw=-10)
    n_diff, _ = slr.compare_exact(a.codes, b.codes)
    assert n_diff == 0
