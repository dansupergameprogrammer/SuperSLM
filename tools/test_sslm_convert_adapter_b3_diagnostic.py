"""Brunel's T-2201 pins for the two D-SLM3724-investigation remedies (B3 review gate):

1. The `_b3_pair_diagnostic` mean-conjunct margins (`composed_mean`, `effect_mean`) no longer carry
   the spurious `+1.645` SE constant that came from grading a VALIDATION-partition `upper_ci`
   (which already embeds `+1.645*se`) against a Delta, rather than the partition's own `mean` --
   the exact fix specified against the design's own §26.9 lineage. The fixture values pinned below
   (`test_real_shopkeeper_*`, `test_consumer_five_artifact_band_*`) are not invented: they reproduce
   -- via the smallest two-point construction that reaches the same mean/SE the real run measured --
   the exact raw and corrected `margin_se` figures already executed and recorded in the T-2201
   investigation (`Claude/Brunel/t2201-b3-gate-investigation-2026-08-20.md` Sec1.2 and Sec4,
   Wizard repo): the real shopkeeper adapter's own three real pairs that fail to survive the
   correction, and Thomas's consumer-reported `layer4.up_proj` pair, whose raw margin sits in the
   3.0-3.5 SE band the investigation names as the shape shared by five of his six flagged pairs.
   Because `margin = (point - Delta)/se [+ 1.645 iff point is upper_ci]` is a pure algebraic
   identity, any two-point draw array reaching the same (mean, se, Delta) as the real run reproduces
   the real run's own margin figure exactly -- this is a reconstruction of the recorded arithmetic,
   not a new synthetic scenario.

2. The pooled primary gate's own `accepted=True`/`accepted=False` reading is printed by the CLI
   beside an explicit, ID-free notice, at every site the tool prints the pooled verdict (the
   `--verbose` build-log line inside `build_runtime_additive_sections`, and both the accept and
   reject branches of `main()`). T-2202 (D-SLM3730 lineage) rewrote this notice's text: it states
   plainly that the check can still refuse to write an artifact, that a refusal can come from
   sampling noise rather than a real defect, that the check cannot detect a genuine magnitude
   error, and that a repair replacing its statistic is in progress -- and it stops calling the
   per-pair diagnostics "the actionable signal" (they are uncommissioned as a general oracle --
   `StandardsDocument.md` §5.4 reserves that word for a must-accept and a must-reject, both
   producible by the instrument's real data path, authored independent of the instrument's
   builder; this file's own flag-fire pin below is a hand-built two-point construction that
   proves the flag-fire path is live above threshold, and proves nothing about what the
   diagnostic does on a real conversion -- see `Claude/Poirot/4299d84-t2206-b3-round2-
   confirmation.md` M3, Wizard repo) and stops saying a live reject "carries no information"
   (the reject branch still refuses to write the artifact, so that claim contradicted the branch
   it sat beside).
"""

import inspect
import re

import numpy as np
import pytest

import sslm_convert_adapter as A


def _two_point_draws(mean: float, se: float) -> np.ndarray:
    """The smallest draw array whose own sample mean is exactly `mean` and whose own sample SE
    (`np.std(ddof=1) / sqrt(n)`) is exactly `se`: two points `mean - se`, `mean + se` have sample
    std `sqrt(2)*se` and `n=2`, so `se_sample = sqrt(2)*se / sqrt(2) = se`, exactly."""
    return np.asarray([mean - se, mean + se], dtype=np.float64)


def _diagnostic_for(composed_mean_point: float, effect_mean_point: float, *,
                     delta_composed_mean: float = 0.0, delta_effect_mean: float = 0.0) -> dict:
    """Builds a single (layer, proj) pair's diagnostic via `A._b3_pair_diagnostic`, with `se=1.0`
    for both mean conjuncts (so the margin arithmetic reduces to `point - Delta` and the raw/
    corrected figures land on the exact literals this module's own docstring cites) and the tail
    conjuncts' own Deltas pinned far out of reach so they never flag -- this pin is scoped to the
    mean-conjunct fix (D-SLM3221's own lineage) alone, per T-2201's remedy 1."""
    composed_val = _two_point_draws(composed_mean_point, 1.0)
    effect_val = _two_point_draws(effect_mean_point, 1.0)
    raw = {"composed_val": composed_val, "effect_val": effect_val}
    own_check = {
        "delta_composed_mean": delta_composed_mean,
        "delta_composed_tail": 1.0e9,   # never flags -- out of this pin's scope
        "delta_effect_mean": delta_effect_mean,
        "delta_effect_tail": 1.0e9,     # never flags -- out of this pin's scope
        "accepted": False,
    }
    rng = np.random.default_rng(0xB3B3)
    return A._b3_pair_diagnostic("pinned_pair", raw, own_check, n_bootstrap_resamples=8, rng=rng)


# --- Remedy 1: the D-SLM3221 mean-conjunct fix, pinned against the investigation's own executed
#     real-shopkeeper-adapter figures (Sec1.2) ------------------------------------------------

@pytest.mark.parametrize(
    "corrected_mean, expected_raw_margin_if_unfixed",
    [
        # (name, corrected composed_mean the investigation recorded, the raw upper_ci-based
        #  margin the SAME pair's SAME data produces under the unfixed formula -- Sec1.2's own
        #  table: "layer7.gate_proj +3.622->1.977 ... layer19.o_proj +2.670->1.025 ...
        #  layer24.o_proj +2.038->0.393").
        (1.977, 3.622),   # layer7.gate_proj
        (1.025, 2.670),   # layer19.o_proj
        (0.393, 2.038),   # layer24.o_proj
    ],
)
def test_real_shopkeeper_flagged_pairs_correct_margin_matches_investigation_table(
    corrected_mean, expected_raw_margin_if_unfixed,
):
    d = _diagnostic_for(composed_mean_point=corrected_mean, effect_mean_point=-999.0)

    # The fixed formula reads the VALIDATION partition's own `mean`, so with `se=1.0` and
    # `Delta=0.0` the margin equals the point estimate exactly -- the corrected figure the
    # investigation recorded.
    assert d["margins_se"]["composed_mean"] == pytest.approx(corrected_mean, abs=1e-9)

    # Sanity-check the SAME raw data against the algebraic identity the investigation proved
    # (Sec1.2: "margin_composed_mean = (mean-Delta)/se + 1.645" under the unfixed `upper_ci` form)
    # -- confirms these fixtures reproduce the exact raw figures the investigation recorded, not
    # merely plausible-looking numbers.
    unfixed_margin = corrected_mean + A._B3_Z_95_ONE_SIDED
    assert unfixed_margin == pytest.approx(expected_raw_margin_if_unfixed, abs=1e-3)


def test_real_shopkeeper_flagged_pairs_no_longer_flag_composed_mean():
    """The three real shopkeeper pairs the investigation names as NOT surviving the correction
    (Sec1.2) must not appear in `flagged` once the fix is applied -- each corrected margin is
    below the 2.0 SE review threshold."""
    for corrected_mean in (1.977, 1.025, 0.393):
        d = _diagnostic_for(composed_mean_point=corrected_mean, effect_mean_point=-999.0)
        assert "composed_mean" not in d["flagged"], (
            f"corrected composed_mean margin {corrected_mean} is below the 2.0 SE review "
            f"threshold and must not flag post-fix"
        )


def test_consumer_five_artifact_band_pair_goes_red_then_green():
    """T-2201's own required cell: a construction whose RAW margin sits in the 3.0-3.5 SE band
    the investigation names as the shape shared by five of Thomas's six consumer-reported flagged
    pairs (Sec3, table in Sec4.2), and whose CORRECTED margin is sub-threshold -- reproducing
    `layer4.up_proj` exactly (raw 3.05, corrected 1.405, Sec4.2's own table)."""
    corrected = 1.405
    raw_if_unfixed = corrected + A._B3_Z_95_ONE_SIDED
    assert 3.0 <= raw_if_unfixed <= 3.5, "fixture must reproduce the consumer's 3.0-3.5 raw band"
    assert raw_if_unfixed == pytest.approx(3.05, abs=1e-3)

    d = _diagnostic_for(composed_mean_point=corrected, effect_mean_point=-999.0)

    # GREEN under the current (fixed) code: below threshold, not flagged.
    assert d["margins_se"]["composed_mean"] == pytest.approx(1.405, abs=1e-9)
    assert "composed_mean" not in d["flagged"]

    # RED under the pre-fix formula this pin replaces (`upper_ci` in place of `mean`): the same
    # underlying data, graded the old way, exceeds the 2.0 SE threshold and would have flagged --
    # this is the exact defect D-SLM3221 identified and this fix removes.
    assert raw_if_unfixed > A._B3_REVIEW_MARGIN_SE


def test_effect_mean_fix_removes_the_same_spurious_offset():
    """The same `+1.645` defect applied identically to `effect_mean` (T-2201's remedy 1 covers
    both mean conjuncts, `sslm_convert_adapter.py:811,813`) -- a purely constructed cell (no
    investigation-recorded `effect_mean` value ever crossed the 2.0 SE threshold, since every one
    of Thomas's six pairs read deeply negative on this conjunct, Sec4.4) demonstrating the fix
    flips flagged status here too."""
    corrected = 1.8
    raw_if_unfixed = corrected + A._B3_Z_95_ONE_SIDED  # 3.445 -- exceeds threshold pre-fix
    assert raw_if_unfixed > A._B3_REVIEW_MARGIN_SE

    d = _diagnostic_for(composed_mean_point=-999.0, effect_mean_point=corrected)
    assert d["margins_se"]["effect_mean"] == pytest.approx(1.8, abs=1e-9)
    assert "effect_mean" not in d["flagged"]


def test_tail_conjuncts_are_unaffected_by_the_mean_conjunct_fix():
    """Regression guard: the fix touches only the two mean-conjunct margin calls
    (`sslm_convert_adapter.py:811,813`); the tail conjuncts must keep reading a raw `p95` point
    estimate (never `upper_ci`), exactly as Significant 4 found already correct."""
    d = _diagnostic_for(composed_mean_point=0.0, effect_mean_point=0.0,
                        delta_composed_mean=0.0, delta_effect_mean=0.0)
    # own_check's tail Deltas are pinned to 1e9 above -- a `p95`-based tail margin must therefore
    # be a huge negative number (`p95 - 1e9`), never anything resembling a `mean`-based composed
    # figure near 0.
    assert d["margins_se"]["composed_tail"] < -1.0e8
    assert d["margins_se"]["effect_tail"] < -1.0e8


def test_a_genuinely_flagging_pair_actually_flags_composed_mean():
    """T-2202's own required cell (Poirot S3): every prior cell in this file asserts the NEGATIVE
    (`"composed_mean" not in flagged`, three times); none ever asserted the positive. Fixed via
    the same algebraic two-point construction `_diagnostic_for` already uses: a pair whose
    CORRECTED `composed_mean` margin is 4.495 SE -- Thomas's own consumer-reported `layer8.v_proj`
    pair, corrected via the D-SLM3221 fix this file's remedy-1 cells pin
    (`Claude/Brunel/t2201-b3-gate-investigation-2026-08-20.md` Sec4.2, Wizard repo) -- must appear
    in `flagged`, because 4.495 is more than double the 2.0 SE review threshold.

    Executed red-then-green: replacing `flagged = [conjunct for conjunct, m in margins.items() if
    m > _B3_REVIEW_MARGIN_SE]` at `sslm_convert_adapter.py:828` with `flagged = []` -- the
    reviewer's own executed mutation -- makes this cell fail (nothing can ever appear in an empty
    list); the current code, which actually computes `flagged` from the margins, passes."""
    d = _diagnostic_for(composed_mean_point=4.495, effect_mean_point=-999.0)
    assert d["margins_se"]["composed_mean"] == pytest.approx(4.495, abs=1e-9)
    assert "composed_mean" in d["flagged"], (
        "a corrected composed_mean margin of 4.495 SE is more than double the 2.0 SE review "
        "threshold and must flag -- a diagnostic that never flags anything passes every prior "
        "cell in this file while catching nothing"
    )


# --- Remedy 2: the notice printed beside the pooled gate's own verdict ----------------------

def test_status_notice_names_no_internal_decision_ids():
    notice = A._B3_POOLED_GATE_STATUS_NOTICE
    assert not re.search(r"D-SLM\d+", notice), (
        "the consumer-facing status notice must not cite internal decision IDs"
    )
    assert not re.search(r"\bT-\d+\b", notice), (
        "the consumer-facing status notice must not cite internal ticket IDs"
    )


def test_status_notice_states_what_the_check_can_and_cannot_do():
    """T-2202 (Poirot S1/S2): the notice states the check can still refuse to write an artifact,
    that the refusal can come from sampling noise rather than a real defect, that it cannot
    detect a genuine magnitude error, and that a repair is in progress -- and it must NOT claim a
    live reject "carries no information" (S2: the reject branch still refuses the artifact, which
    directly contradicts that phrase) or headline the per-pair diagnostics as "the actionable
    signal" (S1: they were never commissioned as a general oracle).

    T-2206 (Poirot S1): the per-pair caveat above qualified the false-POSITIVE direction only
    ("a pair named there is worth a closer look, not a confirmed finding") while the one thing
    actually measured about this diagnostic (`Claude/Brunel/t2201-b3-gate-investigation-2026-08-
    20.md` §3 finding 2, Wizard repo) runs the other way: the ×50-hot construction produces
    0/28 flags. The notice must also state the false-NEGATIVE direction -- an empty per-pair
    list is not evidence the adapter is sound."""
    notice = A._B3_POOLED_GATE_STATUS_NOTICE.lower()
    assert "refuse" in notice and "artifact" in notice
    assert "sampling noise" in notice
    assert "magnitude error" in notice
    assert "repair" in notice and "in progress" in notice
    assert "per-pair diagnostic" in notice
    assert "carries no information" not in notice
    assert "actionable signal" not in notice
    assert "empty" in notice and "not evidence" in notice


def test_status_notice_is_printed_at_every_pooled_gate_output_site():
    """Static wiring check: the notice constant must be referenced at its own definition plus
    every site the tool prints a pooled-gate verdict -- the `--verbose` build-log line inside
    `build_runtime_additive_sections`, and both the accept and reject branches of `main()`. A
    count of 1 (definition only, never printed) is exactly the pre-remedy defect T-2201 found:
    the pooled `accepted=True` printed with no caveat anywhere a consumer would see it."""
    source = inspect.getsource(A)
    occurrences = source.count("_B3_POOLED_GATE_STATUS_NOTICE")
    assert occurrences >= 4, (
        f"expected the status notice's definition plus 3 print sites (verbose pooled-gate "
        f"line, main() accept branch, main() reject branch); found {occurrences} references"
    )


def test_main_accept_path_prints_pooled_status_notice(monkeypatch, capsys, tmp_path):
    """End-to-end through `main()` (real CLI dispatch, real argument parsing, `--skip-verify` to
    avoid the C++ verifier dependency) on the ACCEPT branch: the pooled gate's `accepted=True` is
    printed beside the status notice, not bare."""
    fake_pooled = {
        "accepted": True, "n_pairs": 1,
        "per_pair_diagnostics": [{"name": "layer0.q_proj", "flagged": []}],
    }
    fake_verdict = {"domain_trip": False, "margin_exceeded": False,
                    "saturation_elevated": False, "pairs": [], "pooled": fake_pooled}

    def fake_build(adapter_dir, base_sslm_path):
        return [], fake_verdict, {}

    monkeypatch.setattr(A, "build_runtime_additive_sections", fake_build)
    monkeypatch.setattr(A.sf, "write_artifact", lambda path, sections: "deadbeefcafe")

    out_path = tmp_path / "out.sslm"
    monkeypatch.setattr(A.sys, "argv", [
        "sslm_convert_adapter.py",
        "--adapter", str(tmp_path), "--base", str(tmp_path / "base.sslm"),
        "--out", str(out_path), "--skip-verify",
    ])

    rc = A.main()
    assert rc == 0

    out = capsys.readouterr().out
    assert "accepted=True" in out
    assert "sampling noise" in out
    assert "per-pair diagnostic" in out.lower()


def test_main_reject_path_prints_pooled_status_notice(monkeypatch, capsys, tmp_path):
    """Same, on the REJECT branch (`margin_exceeded=True`, no `--fallback`) -- the branch the
    original consumer report's own `accepted=True` sat beside without qualification.

    T-2202 (Poirot M3): this cell no longer asserts the ABSENCE of a bare `accepted=False` line.
    That assertion pinned an incidental fact about this branch's current print statements, not
    this remedy -- making the reject branch symmetric with the accept branch (printing a bare
    `pooled B3 gate: accepted=...` line here too) is a defensible future improvement that would
    turn this cell red for a reason unrelated to what it guards. What the remedy actually
    requires -- the per-conjunct accepts and the notice both appear -- is asserted below and
    holds regardless of whether a future round adds the bare line."""
    fake_pooled = {
        "accepted": False, "n_pairs": 1,
        "composed_mean_accepts": False, "composed_tail_accepts": True,
        "effect_mean_accepts": True, "effect_tail_accepts": True,
        "per_pair_diagnostics": [{"name": "layer8.v_proj", "flagged": ["composed_mean"]}],
    }
    fake_verdict = {"domain_trip": False, "margin_exceeded": True,
                    "saturation_elevated": False, "pairs": [], "pooled": fake_pooled}

    def fake_build(adapter_dir, base_sslm_path):
        return None, fake_verdict, {}

    monkeypatch.setattr(A, "build_runtime_additive_sections", fake_build)

    out_path = tmp_path / "out.sslm"
    monkeypatch.setattr(A.sys, "argv", [
        "sslm_convert_adapter.py",
        "--adapter", str(tmp_path), "--base", str(tmp_path / "base.sslm"),
        "--out", str(out_path), "--skip-verify",
    ])

    rc = A.main()
    assert rc == 1

    err = capsys.readouterr().err
    assert "composed_mean_accepts=False" in err
    assert "sampling noise" in err
    assert "per-pair diagnostic" in err.lower()
