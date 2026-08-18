"""Red suite (T-1683/T-1687) for `tools/layer_bisection_report.py` -- the
comparison tool the layer-bisection design (`Claude/Vitruvius/superslm-
t1683-layer-bisection-design-2026-08-02.md` §5) specifies as "structurally
the hidden-state sibling of `tools/logit_margin_report.py`, reusing its
already-reviewed statistical machinery."

`tools/layer_bisection_report.py` DOES NOT EXIST YET. Every test below fails
today on import (ModuleNotFoundError) -- `red-unimplemented`, per Curie's own
discipline (`Curie.md` Phase 4): a red test fails for its cell's reason, not
on a mis-stated assertion. This file is the concrete contract Brunel builds
against; Mendeleev audits it for coverage of the design's §8 Coverage Model,
never the module it tests.

THE CONTRACT THIS FILE PINS (derived from design §4.1 step 5, §4.2 step 4,
§5 -- no committed source exists to read this from, so the shapes below are
this suite's own derivation from the design's prose, stated explicitly so a
disagreement is visible rather than assumed):

  zscore_row(row: np.ndarray) -> np.ndarray
      Transcribed IDENTICALLY from `tools/logit_margin_report.py:235-240`
      (design §5: "reusing its already-reviewed statistical machinery
      (zscore_row, ...)" -- not a re-derivation, the same function).

  Int8LayerDump / load_int8_layer_dump(path) -> Int8LayerDump
      Reads T-1685's dump format (design §4.1 step 5): header
      `uint64 rows, uint64 hidden_size, uint64 prompt_fingerprint,
      uint64 capture_mode`, then per row `int64 m, int64 e`, then
      `hidden_size` int8 codes. Fields: .rows, .hidden_size,
      .prompt_fingerprint, .capture_mode, .scales (rows x 2 int64,
      columns [m, e]), .codes (rows x hidden_size int8).

  FloatLayerDump / load_float_layer_dump(path) -> FloatLayerDump
      Reads T-1686's dump format (design §4.2 step 4): header
      `uint64 rows, uint64 hidden_size, uint64 prompt_fingerprint,
      uint64 capture_mode`, then `rows * hidden_size` float32 values.
      Fields: .rows, .hidden_size, .prompt_fingerprint, .capture_mode,
      .values (rows x hidden_size float32).

  check_provenance(int8_dump, float_dump) -> None
      Design §5's provenance check, run before any statistic: raises
      ProvenanceError (a `SystemExit`-raising, loud, non-zero-exit failure
      per the design's own convention -- this suite uses a dedicated
      exception type so pytest can assert on it directly) on a rows or
      hidden_size mismatch, a prompt_fingerprint mismatch, or a
      capture_mode value other than 1 on EITHER side. Returns None (does
      not raise) when the pair matches on all four fields.

  compare_layer_row(int8_codes: np.ndarray, float_row: np.ndarray)
      -> LayerRowStats
      Design §5's per-boundary statistic set, computed over the FULL
      hidden_size width (no TOP_N truncation -- §5: "there is no
      vocabulary-style long tail to restrict attention to"): Spearman rank
      correlation between the raw int8 codes and the raw float32 values
      (§2.3's rescaling-invariance argument -- no dequantization), Pearson
      correlation of the two rows after each is independently z-scored
      (`zscore_row`), and the maximum absolute z-scored difference between
      the two rows. Fields: .spearman, .pearson, .max_abs_z_diff.

Run with: python -m pytest tools/test_t1687_layer_bisection_report.py -v
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

lbr = pytest.importorskip("layer_bisection_report")  # needs scipy; skips where the diagnostic's own dependency is absent (e.g. the converter-validate CI env)


# --- dump format round-trips (design §4.1 step 5, §4.2 step 4) --------------

def _write_int8_dump(path, rows, hidden_size, prompt_fingerprint, capture_mode,
                      scales, codes):
    with open(path, "wb") as f:
        f.write(struct.pack("<QQQQ", rows, hidden_size, prompt_fingerprint, capture_mode))
        for i in range(rows):
            f.write(struct.pack("<qq", scales[i][0], scales[i][1]))
            f.write(codes[i].astype(np.int8).tobytes())


def _write_float_dump(path, rows, hidden_size, prompt_fingerprint, capture_mode, values):
    with open(path, "wb") as f:
        f.write(struct.pack("<QQQQ", rows, hidden_size, prompt_fingerprint, capture_mode))
        f.write(values.astype(np.float32).tobytes())


def test_load_int8_layer_dump_roundtrip(tmp_path):
    rows, hidden_size = 3, 4
    scales = [(1073741824, -30), (1073741824, -30), (536870912, -29)]
    codes = np.array([[1, -2, 3, -4], [5, -6, 7, -8], [9, -10, 11, -12]], dtype=np.int8)
    path = tmp_path / "int8_layer.bin"
    _write_int8_dump(path, rows, hidden_size, prompt_fingerprint=0xABCDEF01,
                      capture_mode=1, scales=scales, codes=codes)

    dump = lbr.load_int8_layer_dump(path)
    assert dump.rows == rows
    assert dump.hidden_size == hidden_size
    assert dump.prompt_fingerprint == 0xABCDEF01
    assert dump.capture_mode == 1
    assert dump.scales.shape == (rows, 2)
    assert tuple(dump.scales[2]) == (536870912, -29)
    assert dump.codes.shape == (rows, hidden_size)
    assert np.array_equal(dump.codes, codes)


def test_load_float_layer_dump_roundtrip(tmp_path):
    rows, hidden_size = 2, 3
    values = np.array([[1.5, -2.25, 0.0], [3.125, 4.0, -1.0]], dtype=np.float32)
    path = tmp_path / "float_layer.bin"
    _write_float_dump(path, rows, hidden_size, prompt_fingerprint=0x1122334455,
                       capture_mode=1, values=values)

    dump = lbr.load_float_layer_dump(path)
    assert dump.rows == rows
    assert dump.hidden_size == hidden_size
    assert dump.prompt_fingerprint == 0x1122334455
    assert dump.capture_mode == 1
    assert dump.values.shape == (rows, hidden_size)
    assert np.allclose(dump.values, values)


# --- provenance check (design §5; coverage audit F3) -------------------------
# Every rejection cell writes a MATCHING pair first, then mutates exactly one
# field -- the discriminating power is in that single-field delta, not in
# "these two dumps are unrelated garbage," which a check could pass by
# accident on shape alone.

def _matching_pair(tmp_path, rows=29, hidden_size=1536, fingerprint=0xFEED, capture_mode=1):
    codes = np.zeros((rows, hidden_size), dtype=np.int8)
    scales = [(1073741824, -30)] * rows
    int8_path = tmp_path / "int8.bin"
    _write_int8_dump(int8_path, rows, hidden_size, fingerprint, capture_mode, scales, codes)

    values = np.zeros((rows, hidden_size), dtype=np.float32)
    float_path = tmp_path / "float.bin"
    _write_float_dump(float_path, rows, hidden_size, fingerprint, capture_mode, values)
    return int8_path, float_path


def test_check_provenance_accepts_matching_pair(tmp_path):
    # Positive control -- the guard must not reject a genuinely matching
    # pair. Without this cell, a check that unconditionally raises would
    # pass every rejection test below while proving nothing (D-INSP-61:
    # "reachable is not discriminating").
    int8_path, float_path = _matching_pair(tmp_path)
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    lbr.check_provenance(int8_dump, float_dump)  # must not raise


def test_check_provenance_rejects_hidden_size_mismatch(tmp_path):
    int8_path, float_path = _matching_pair(tmp_path, hidden_size=1536)
    # Float side rebuilt at a different hidden_size, everything else matching.
    values = np.zeros((29, 1537), dtype=np.float32)
    _write_float_dump(float_path, 29, 1537, 0xFEED, 1, values)
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    with pytest.raises(lbr.ProvenanceError):
        lbr.check_provenance(int8_dump, float_dump)


def test_check_provenance_rejects_rows_mismatch(tmp_path):
    int8_path, float_path = _matching_pair(tmp_path, rows=29)
    values = np.zeros((28, 1536), dtype=np.float32)
    _write_float_dump(float_path, 28, 1536, 0xFEED, 1, values)
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    with pytest.raises(lbr.ProvenanceError):
        lbr.check_provenance(int8_dump, float_dump)


def test_check_provenance_rejects_fingerprint_mismatch(tmp_path):
    int8_path, float_path = _matching_pair(tmp_path, fingerprint=0xFEED)
    values = np.zeros((29, 1536), dtype=np.float32)
    _write_float_dump(float_path, 29, 1536, 0xBEEF, 1, values)  # different fingerprint
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    with pytest.raises(lbr.ProvenanceError):
        lbr.check_provenance(int8_dump, float_dump)


def test_check_provenance_rejects_capture_mode_zero_on_float_side(tmp_path):
    # The fixture the design's own capture-mode field (§4.1 step 5, §4.2
    # step 4, added post-strike 2026-08-02) exists to exercise: a float
    # dump regressed to a batched exporter (capture_mode=0) must be
    # rejected here rather than silently compared, which is precisely the
    # confound the strike found (Claude/Loki/superslm-t1683-layer-
    # bisection-design-strike-2026-08-02.md).
    int8_path, float_path = _matching_pair(tmp_path, capture_mode=1)
    values = np.zeros((29, 1536), dtype=np.float32)
    _write_float_dump(float_path, 29, 1536, 0xFEED, 0, values)  # capture_mode=0
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    with pytest.raises(lbr.ProvenanceError):
        lbr.check_provenance(int8_dump, float_dump)


def test_check_provenance_rejects_capture_mode_zero_on_int8_side(tmp_path):
    # Symmetric to the float-side cell above -- the check names BOTH sides
    # (design §5: "a capture_mode mismatch, or a capture_mode value other
    # than 1 on EITHER side").
    codes = np.zeros((29, 1536), dtype=np.int8)
    scales = [(1073741824, -30)] * 29
    int8_path = tmp_path / "int8.bin"
    _write_int8_dump(int8_path, 29, 1536, 0xFEED, 0, scales, codes)  # capture_mode=0
    values = np.zeros((29, 1536), dtype=np.float32)
    float_path = tmp_path / "float.bin"
    _write_float_dump(float_path, 29, 1536, 0xFEED, 1, values)
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    with pytest.raises(lbr.ProvenanceError):
        lbr.check_provenance(int8_dump, float_dump)


def test_check_provenance_rejects_capture_mode_two_on_either_side(tmp_path):
    # Design §5: "a capture_mode value other than 1 on either side" -- not
    # merely "capture_mode disagrees between the two sides." A value of 2
    # on BOTH sides (so they agree with each other but neither is 1) must
    # still be rejected; a check that only compares the two sides against
    # EACH OTHER, rather than each against the literal value 1, would pass
    # this pair and is the gap this cell exists to close.
    codes = np.zeros((29, 1536), dtype=np.int8)
    scales = [(1073741824, -30)] * 29
    int8_path = tmp_path / "int8.bin"
    _write_int8_dump(int8_path, 29, 1536, 0xFEED, 2, scales, codes)
    values = np.zeros((29, 1536), dtype=np.float32)
    float_path = tmp_path / "float.bin"
    _write_float_dump(float_path, 29, 1536, 0xFEED, 2, values)
    int8_dump = lbr.load_int8_layer_dump(int8_path)
    float_dump = lbr.load_float_layer_dump(float_path)
    with pytest.raises(lbr.ProvenanceError):
        lbr.check_provenance(int8_dump, float_dump)


# --- formula correctness (design §7 T-1687 part 1) ---------------------------

def test_compare_layer_row_formula_correctness_hand_computed():
    # int8 codes and float values in the SAME rank order and the SAME
    # z-scored shape, by construction -- spearman == 1.0, pearson == 1.0,
    # max_abs_z_diff == 0.0, all hand-verifiable.
    int8_codes = np.array([10, 20, 30, 40], dtype=np.int8)
    float_row = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = lbr.compare_layer_row(int8_codes, float_row)
    assert abs(result.spearman - 1.0) < 1e-9
    assert abs(result.pearson - 1.0) < 1e-9
    assert abs(result.max_abs_z_diff - 0.0) < 1e-9


def test_compare_layer_row_formula_correctness_anticorrelated():
    # Reversed order -- spearman == -1.0 exactly, by construction. This is
    # the mutation-pinning half of the cell above: a formula that always
    # reports rho=1.0 regardless of input (an unguarded stub) passes the
    # first cell and fails this one.
    int8_codes = np.array([40, 30, 20, 10], dtype=np.int8)
    float_row = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = lbr.compare_layer_row(int8_codes, float_row)
    assert abs(result.spearman - (-1.0)) < 1e-9
    assert abs(result.pearson - (-1.0)) < 1e-9


def test_compare_layer_row_formula_correctness_known_z_diff():
    # int8 side and float side z-score to DIFFERENT shapes by construction:
    # int8 = [0,0,0,10], mean=2.5, POPULATION std (ddof=0, numpy's `.std()`
    # default, the same convention `zscore_row` transcribes from
    # `logit_margin_report.py:235-240`) = sqrt(mean((x-mean)^2)) =
    # sqrt(75/4) = sqrt(18.75); z at index 3 = 7.5/sqrt(18.75) = sqrt(3),
    # hand-verified by direct execution of `numpy.ndarray.std`, not assumed
    # from a sample-std (ddof=1) formula, which would give 1.5 instead and
    # is the wrong convention for this codebase's own zscore_row.
    # float = [0,0,0,0] is a constant row, z-scores to all zero
    # (zscore_row's own degenerate-row rule -- std==0 returns zeros, not
    # NaN). max_abs_z_diff is therefore exactly sqrt(3), at index 3.
    int8_codes = np.array([0, 0, 0, 10], dtype=np.int8)
    float_row = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = lbr.compare_layer_row(int8_codes, float_row)
    assert abs(result.max_abs_z_diff - 3 ** 0.5) < 1e-9


# --- numerical edges at hidden-state scale (design §7 T-1687 part 3; -------
# coverage audit F5): this machinery was built/tested for ~150k-wide,
# effectively-continuous vocab rows (T-1681); applied here to 1536-wide,
# int8-coded rows (range [-128,127]), tie density is materially higher and
# near-zero-but-nonzero std is newly reachable.

def test_compare_layer_row_heavy_tie_density_produces_finite_output():
    rng = np.random.default_rng(1683)
    hidden_size = 1536
    # Most values drawn from a handful of repeated int8 codes (heavy ties);
    # a small minority perturbed so the row is not perfectly constant.
    int8_codes = np.zeros(hidden_size, dtype=np.int8)
    int8_codes[:] = rng.choice([-3, 0, 5], size=hidden_size)
    int8_codes[:8] = np.arange(8, dtype=np.int8)  # break the tie in a few places
    float_row = int8_codes.astype(np.float32) + rng.normal(0, 0.01, size=hidden_size)

    result = lbr.compare_layer_row(int8_codes, float_row.astype(np.float32))
    assert np.isfinite(result.spearman)
    assert -1.0 <= result.spearman <= 1.0
    assert np.isfinite(result.pearson)
    assert -1.0 <= result.pearson <= 1.0
    assert np.isfinite(result.max_abs_z_diff)


def test_compare_layer_row_near_zero_std_produces_finite_output():
    hidden_size = 1536
    # int8 row: constant except one element off by the smallest possible
    # nonzero int8 delta -- std is small but not exactly zero.
    int8_codes = np.zeros(hidden_size, dtype=np.int8)
    int8_codes[0] = 1
    float_row = np.zeros(hidden_size, dtype=np.float32)
    float_row[0] = 1e-6  # smallest-magnitude nonzero float perturbation

    result = lbr.compare_layer_row(int8_codes, float_row)
    assert np.isfinite(result.max_abs_z_diff)
    # A near-zero std inflates z-scores, but the result must stay a real,
    # boundable number -- not an overflow to inf, which a naive division
    # would produce for a std small enough to underflow in the denominator.
    assert result.max_abs_z_diff < 1e12


def test_compare_layer_row_exactly_zero_std_on_both_sides_returns_zero_z_diff():
    # Both rows perfectly constant (std==0 on both sides) -- zscore_row's
    # own degenerate-row rule returns all-zero for each, so the z-diff is
    # exactly 0, not NaN. Spearman/Pearson over two constant rows have no
    # defined value under `scipy.stats`'s own convention (NaN with a
    # ConstantInputWarning) -- this cell asserts only the z-diff, which the
    # z-score path answers independently of the correlation coefficients.
    int8_codes = np.full(1536, 7, dtype=np.int8)
    float_row = np.full(1536, 3.0, dtype=np.float32)
    result = lbr.compare_layer_row(int8_codes, float_row)
    assert result.max_abs_z_diff == 0.0


# --- prompt-text provenance (design S6; S3, code review 2026-08-02) ---------
# layer_bisection_report.POPULATION's text must come from
# logit_margin_report.PROMPT_SET -- one committed source for all nine
# prompts, never a second independently maintained copy (the defect the
# review found: two copies, identical today, drift latent).

def test_population_text_matches_prompt_set_not_a_second_copy():
    import logit_margin_report as lmr

    prompt_set_by_label = {case.label: case.question for case in lmr.PROMPT_SET}
    assert len(lbr.POPULATION) == 9
    for label, question, _role in lbr.POPULATION:
        assert label in prompt_set_by_label, f"{label} missing from logit_margin_report.PROMPT_SET"
        assert question == prompt_set_by_label[label], (
            f"{label}: layer_bisection_report's own text {question!r} != "
            f"PROMPT_SET's {prompt_set_by_label[label]!r} -- the two copies have diverged"
        )


def test_population_roles_cover_all_three_groups():
    roles = {role for _label, _question, role in lbr.POPULATION}
    assert roles == {"mech1", "mech2", "control"}


# --- repeat-vs-repeat resolving power (design S10; S2, code review 2026-08-02) --

def test_zscore_diff_rows_identical_rows_is_zero():
    row = np.array([1, -5, 10, 0, 127], dtype=np.int8)
    assert lbr.zscore_diff_rows(row, row.copy()) == 0.0


def test_zscore_diff_rows_hand_computed():
    # a = [0,0,0,10]: mean=2.5, population std (ddof=0) = sqrt(mean((x-mean)^2))
    # = sqrt(75/4) = sqrt(18.75); z = [-2.5, -2.5, -2.5, 7.5] / sqrt(18.75),
    # z[3] = 7.5/sqrt(18.75) = sqrt(3) (the identical construction and
    # hand-verified value test_compare_layer_row_formula_correctness_known_z_diff
    # already establishes for this exact row).
    # b = [0,0,0,0] -> z = [0,0,0,0] (degenerate row, zscore_row's own rule).
    # max|z-diff| = sqrt(3) at index 3.
    a = np.array([0, 0, 0, 10], dtype=np.int8)
    b = np.array([0, 0, 0, 0], dtype=np.int8)
    assert abs(lbr.zscore_diff_rows(a, b) - 3 ** 0.5) < 1e-9


def test_repeat_vs_repeat_dispersion_per_boundary_hand_computed():
    rows_a = np.array([[0, 0, 0, 10], [1, 2, 3, 4]], dtype=np.int8)
    rows_b = np.array([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=np.int8)
    result = lbr.repeat_vs_repeat_dispersion(rows_a, rows_b)
    assert len(result) == 2
    assert abs(result[0] - 3 ** 0.5) < 1e-9  # boundary 0: the hand-computed cell above
    assert result[1] == 0.0  # boundary 1: identical rows


def test_repeat_vs_repeat_dispersion_rejects_shape_mismatch():
    rows_a = np.zeros((3, 4), dtype=np.float32)
    rows_b = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        lbr.repeat_vs_repeat_dispersion(rows_a, rows_b)


# --- execution-level sanity gate (design S7 T-1687 part 4, coverage audit ---
# F8; M4, code review: a bare `assert` is stripped under `python -O` -- this
# gate is now a checkable function raising SanityError explicitly).

def test_check_execution_sanity_accepts_in_range_finite_stats():
    all_stats = {
        "p1": [lbr.LayerRowStats(spearman=0.5, pearson=0.4, max_abs_z_diff=1.2),
               lbr.LayerRowStats(spearman=-1.0, pearson=float("nan"), max_abs_z_diff=0.0)],
    }
    assert lbr.check_execution_sanity(all_stats) == 2


def test_check_execution_sanity_rejects_out_of_range_spearman():
    all_stats = {"p1": [lbr.LayerRowStats(spearman=1.5, pearson=0.0, max_abs_z_diff=0.0)]}
    with pytest.raises(lbr.SanityError):
        lbr.check_execution_sanity(all_stats)


def test_check_execution_sanity_rejects_non_finite_max_abs_z_diff():
    all_stats = {"p1": [lbr.LayerRowStats(spearman=0.0, pearson=0.0, max_abs_z_diff=float("inf"))]}
    with pytest.raises(lbr.SanityError):
        lbr.check_execution_sanity(all_stats)


def test_check_execution_sanity_rejects_out_of_range_non_nan_pearson():
    all_stats = {"p1": [lbr.LayerRowStats(spearman=0.0, pearson=-1.5, max_abs_z_diff=0.0)]}
    with pytest.raises(lbr.SanityError):
        lbr.check_execution_sanity(all_stats)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
