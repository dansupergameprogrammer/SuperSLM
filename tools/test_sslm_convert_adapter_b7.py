"""Curie's red suite for sslm_convert_adapter.py's B7 dispatcher, ported to green (T-2036, B7).

Ports `tests/t2018-slora-serial/t2018_offline_red.cpp`'s seven T-2027-derived B7 cells
(`Claude/Curie/t2018-slora-serial-red-suite-2026-08-13.md` Sec18.4, Wizard repo) — the
explicit-only fallback contract (design Sec7, D-SLM3095): a validation failure emits no artifact
by default, and only emits a merge+quantize artifact given an explicit `--fallback=merge` flag.
Each cell below exercises `dispatch_conversion_outcome` directly rather than reproducing the C++
suite's own reused B3/B6 fixtures — this module's own dispatcher takes the three ALREADY-COMPUTED
boolean verdicts as input (exactly as the real converter CLI would, downstream of B3/B6's own
checks), so the fixture-reuse the C++ suite's own commentary describes is orthogonal to what this
dispatcher itself does with those verdicts.
"""

import pytest

import sslm_convert_adapter as A


@pytest.mark.parametrize("flag_present", [False, True])
def test_domain_rejection_trip_branch_selected_first(flag_present):
    branch, outcome = A.dispatch_conversion_outcome(
        domain_trip=True, margin_exceeded=True, saturation_elevated=True,
        fallback_flag_present=flag_present,
    )
    assert branch == A.RejectionBranch.DOMAIN_REJECTION_TRIP, (
        "domain-rejection trip must be selected first even when every other condition is also true"
    )
    expected = A.ArtifactOutcome.MERGE_QUANTIZE_EMITTED if flag_present else A.ArtifactOutcome.NO_ARTIFACT_EMITTED
    assert outcome == expected


def test_domain_rejection_trip_no_flag_emits_no_artifact():
    branch, outcome = A.dispatch_conversion_outcome(True, False, False, fallback_flag_present=False)
    assert branch == A.RejectionBranch.DOMAIN_REJECTION_TRIP
    assert outcome == A.ArtifactOutcome.NO_ARTIFACT_EMITTED, (
        "without --fallback=merge, a domain-rejection-trip failure must emit NO artifact"
    )


def test_domain_rejection_trip_with_flag_emits_merge_quantize():
    branch, outcome = A.dispatch_conversion_outcome(True, False, False, fallback_flag_present=True)
    assert branch == A.RejectionBranch.DOMAIN_REJECTION_TRIP
    assert outcome == A.ArtifactOutcome.MERGE_QUANTIZE_EMITTED, (
        "WITH --fallback=merge, the identical failing input must emit a merge+quantize artifact"
    )


def test_runtime_vs_baked_margin_exceeded_no_flag_emits_no_artifact():
    branch, outcome = A.dispatch_conversion_outcome(False, True, False, fallback_flag_present=False)
    assert branch == A.RejectionBranch.RUNTIME_VS_BAKED_MARGIN_EXCEEDED
    assert outcome == A.ArtifactOutcome.NO_ARTIFACT_EMITTED


def test_runtime_vs_baked_margin_exceeded_with_flag_emits_merge_quantize():
    branch, outcome = A.dispatch_conversion_outcome(False, True, False, fallback_flag_present=True)
    assert branch == A.RejectionBranch.RUNTIME_VS_BAKED_MARGIN_EXCEEDED
    assert outcome == A.ArtifactOutcome.MERGE_QUANTIZE_EMITTED


def test_saturation_rate_elevation_no_flag_emits_no_artifact():
    branch, outcome = A.dispatch_conversion_outcome(False, False, True, fallback_flag_present=False)
    assert branch == A.RejectionBranch.SATURATION_RATE_ELEVATION
    assert outcome == A.ArtifactOutcome.NO_ARTIFACT_EMITTED


def test_saturation_rate_elevation_with_flag_emits_merge_quantize():
    branch, outcome = A.dispatch_conversion_outcome(False, False, True, fallback_flag_present=True)
    assert branch == A.RejectionBranch.SATURATION_RATE_ELEVATION
    assert outcome == A.ArtifactOutcome.MERGE_QUANTIZE_EMITTED


def test_clean_input_emits_runtime_additive_regardless_of_flag():
    """The flag only matters on a REJECTION path -- a clean input (every condition false) always
    emits the runtime-additive artifact whether or not --fallback=merge was passed."""
    for flag in (False, True):
        branch, outcome = A.dispatch_conversion_outcome(False, False, False, fallback_flag_present=flag)
        assert branch == A.RejectionBranch.NONE
        assert outcome == A.ArtifactOutcome.RUNTIME_ADDITIVE


def test_branches_are_mutually_distinguishable_not_collapsed_to_one_diagnostic():
    """The three branches are not collapsed into one generic 'rejected' outcome -- each named
    condition, alone, selects its own distinct branch, and the priority order holds under every
    combination."""
    assert A.dispatch_conversion_outcome(True, True, True, False)[0] == A.RejectionBranch.DOMAIN_REJECTION_TRIP
    assert A.dispatch_conversion_outcome(False, True, True, False)[0] == A.RejectionBranch.RUNTIME_VS_BAKED_MARGIN_EXCEEDED
    assert A.dispatch_conversion_outcome(False, False, True, False)[0] == A.RejectionBranch.SATURATION_RATE_ELEVATION
    assert A.dispatch_conversion_outcome(False, False, False, False)[0] == A.RejectionBranch.NONE
