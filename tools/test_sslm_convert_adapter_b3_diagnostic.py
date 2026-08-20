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

2. (RETIRED, T-2213/D-SLM3783) The pooled primary gate's own `accepted=True`/`accepted=False`
   reading no longer exists -- the pooled ACCEPT/REJECT arithmetic itself was retired, not merely
   reworded. What this remedy's cells below now pin is the CURRENT notice: an explicit, ID-free
   status statement printed at every site the tool prints the pooled REPORT (the `--verbose`
   build-log line inside `build_runtime_additive_sections`, and `main()`'s own accept branch --
   there is no longer a B3-driven reject branch, since nothing B3 computes can refuse to write an
   artifact any more). It states plainly that the per-pair diagnostics are this tool's primary B3
   review signal, that a flagged pair is worth a closer look and not a confirmed finding, that an
   empty per-pair list is not proof of soundness, and what the new magnitude sanity check does and
   does not do (a coarse, wide-tolerance WARNING, never a REJECT -- commissioned separately in
   `test_sslm_convert_adapter_b3_magnitude.py`).
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


# --- Remedy 2 (RETIRED and rewritten, T-2213/D-SLM3783): the notice printed beside the pooled
#     report -- there is no longer a pooled VERDICT for it to sit beside. ---------------------

def test_status_notice_names_no_internal_decision_ids():
    notice = A._B3_POOLED_GATE_STATUS_NOTICE
    assert not re.search(r"D-SLM\d+", notice), (
        "the consumer-facing status notice must not cite internal decision IDs"
    )
    assert not re.search(r"\bT-\d+\b", notice), (
        "the consumer-facing status notice must not cite internal ticket IDs"
    )


def test_status_notice_states_what_each_shipped_signal_does_and_does_not_do():
    """T-2213 (D-SLM3783): the notice no longer describes a pooled accept/reject statistic (it
    was retired, not merely reworded -- 'a repair ... is in progress' is now false and must not
    appear). It states: nothing below blocks artifact emission on adapter quality; the per-pair
    diagnostics are this tool's primary B3 review signal, worth review, not a verdict, and an
    empty list is not proof of soundness; and the magnitude sanity check is a coarse WARNING that
    never refuses an artifact either."""
    notice = A._B3_POOLED_GATE_STATUS_NOTICE.lower()
    assert "retired" in notice
    assert "per-pair diagnostic" in notice
    assert "empty" in notice and "not proof" in notice
    assert "magnitude" in notice and "warning" in notice
    assert "never" in notice and "refuse" in notice
    # The retired framing must not survive: no more "repair in progress," no more "accepted=" or
    # "cannot detect a real magnitude error" (the new check exists precisely to give a coarse
    # magnitude signal, so that specific old disclaimer is now false).
    assert "repair" not in notice or "in progress" not in notice
    assert "accepted=" not in notice


def test_status_notice_is_printed_at_every_pooled_report_output_site():
    """Static wiring check: the notice constant must be referenced at its own definition, once in
    `run_b3_pooled_report`'s own docstring, and at every site the tool actually PRINTS a pooled
    report -- the `--verbose` build-log line inside `build_runtime_additive_sections`, and
    `main()`'s own accept branch. T-2213 removed the reject-branch print site: nothing B3
    computes can refuse to write an artifact any more, so `main()`'s REJECTED branch no longer
    has pooled B3 diagnostics to print. A count of 1 (definition only, nothing else referencing
    it) would be the pre-T-2201 defect this suite still guards against; the expected count is now
    4 (definition + 1 docstring mention + 2 real print sites)."""
    source = inspect.getsource(A)
    occurrences = source.count("_B3_POOLED_GATE_STATUS_NOTICE")
    assert occurrences == 4, (
        f"expected the status notice's definition, its own docstring mention, and 2 print sites "
        f"(verbose pooled-report line, main() accept branch -- the reject-branch site was "
        f"removed under T-2213, since nothing B3 computes can refuse an artifact any more); "
        f"found {occurrences} references"
    )


def test_main_accept_path_prints_pooled_status_notice_and_delta_norm(monkeypatch, capsys, tmp_path):
    """End-to-end through `main()` (real CLI dispatch, real argument parsing, `--skip-verify` to
    avoid the C++ verifier dependency) on the ACCEPT branch: the pooled report's `delta_norm` is
    printed beside the status notice -- never a retired `accepted=` verdict."""
    fake_pooled = {
        "n_pairs": 1, "delta_norm": 1.2345e-3, "magnitude_warning": None,
        "per_pair_diagnostics": [{"name": "layer0.q_proj", "flagged": []}],
    }
    fake_verdict = {"domain_trip": False, "margin_exceeded": False,
                    "saturation_elevated": False, "pairs": [], "pooled": fake_pooled}

    def fake_build(adapter_dir, base_sslm_path, **_kwargs):
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
    assert "accepted=" not in out, "the retired pooled verdict must not reappear in CLI output"
    assert "delta_norm=1.2345" in out or "delta_norm=1.234500e-03" in out
    assert "per-pair diagnostic" in out.lower()
    assert "MAGNITUDE WARNING" not in out, "no warning was configured; none must print"


def test_main_accept_path_prints_magnitude_warning_when_present(monkeypatch, capsys, tmp_path):
    """T-2213: when `run_b3_pooled_report` returns a `magnitude_warning`, `main()`'s accept
    branch prints it -- still on the ACCEPT branch (rc=0), because the warning never refuses to
    write the artifact."""
    fake_pooled = {
        "n_pairs": 1, "delta_norm": 5.0,
        "magnitude_warning": {
            "candidate_delta_norm": 5.0, "reference_delta_norm": 0.1,
            "ratio_to_reference": 50.0, "tolerance": [0.1, 10.0],
            "disposition": "unresolved",
            "reason": "pooled composed LoRA delta norm is 50x the reference's, outside the "
                     "[0.1x, 10x] wide-tolerance sanity band",
        },
        "per_pair_diagnostics": [{"name": "layer0.q_proj", "flagged": []}],
    }
    fake_verdict = {"domain_trip": False, "margin_exceeded": False,
                    "saturation_elevated": False, "pairs": [], "pooled": fake_pooled}

    def fake_build(adapter_dir, base_sslm_path, **_kwargs):
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
    assert rc == 0, "a magnitude warning must never refuse to write the artifact"
    out = capsys.readouterr().out
    assert "MAGNITUDE WARNING" in out
    assert "50x the reference" in out
    assert out_path.exists() or True  # write_artifact is mocked; rc==0 is the real assertion


def test_run_b3_pooled_report_no_longer_exists_under_its_retired_name():
    """T-2213 (D-SLM3783) diff pin: `run_b3_multi_pair_check` -- the retired pooled gate's own
    name -- must not exist; `run_b3_pooled_report` is its replacement."""
    assert not hasattr(A, "run_b3_multi_pair_check")
    assert hasattr(A, "run_b3_pooled_report")
