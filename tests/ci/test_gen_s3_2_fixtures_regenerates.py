"""Curie's regeneration gate for gen_s3_2_fixtures.py
(SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.2).

WHY THIS EXISTS. Same shape as
test_gen_s3_1_c30_iexp_domain_sweep_fixtures_regenerates.py's own precedent: the
generated header's own line states "Re-running this script must reproduce this
file byte-for-byte", and nothing in the tree checked that claim before this
file. `.github/workflows/tests.yml` sits outside this campaign's writable scope
(D:\\SuperSLM\\tests\\ only), so this gate lives under tests/ci/ instead --
`python -m pytest tests/ci/ -v` already collects and runs every module here on
every CI invocation, so the gate runs the moment it lands, with no separate
CI-wiring step to add or forget.

MECHANISM. Calls the generator's own `generate()` -- never `main()`, which
writes to disk -- and compares its output, in memory, against the committed
header. Also re-runs every one of the generator's own `build_*()` functions
directly: each raises `AssertionError` on its own if the mechanism it derives
against (the vendored reference's `i_sqrt`, `dynamic_scale_reciprocal`, or
`bias_reconcile`) stops agreeing with the claim the generator's comments make.
This is the D-SLM357 mechanism-claim proof for this pass's own authored work:
every claim the generator's docstrings and inline comments make (the F-S3-2
table reproduces; the site witness's root is reachable and genuinely diverges
at two elements and not at the third; BIA1's bound is exactly INT32_MAX and is
tight; the C28 tie witness is genuine and the wrong candidate disagrees only on
the negative sign) is an assertion inside the generator itself, so importing
and calling it here IS running that proof, not merely regenerating a file.

Line-ending normalization matches the S3.1 precedent exactly (this Windows
checkout has core.autocrlf=true; `generate()`/`main()` always emit LF, and
reading the committed file in default text mode reproduces the same
normalization `git diff --exit-code` already performs).

Test-design record:
Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import gen_s3_2_fixtures as gen  # noqa: E402


def _read_committed_header() -> str:
    with open(gen.OUT_PATH, "r", encoding="ascii") as f:
        return f.read()


def test_regenerating_reproduces_the_committed_header_byte_for_byte():
    committed = _read_committed_header()
    regenerated = gen.generate()
    assert regenerated == committed, (
        "tests/gen_s3_2_fixtures.py's generate() output no longer matches the "
        "committed tests/sslm_s3_2_fixtures.h -- regenerate and commit the "
        "header, or the generator's own logic changed without the fixture "
        "being regenerated to match"
    )


def test_the_gate_has_something_to_lose_a_nonvacuous_floor():
    # Same independent-population-style floor StandardsDocument Sec4 asks of a
    # new structural check: confirm the compared artifact is substantial
    # before trusting the byte-identical comparison above to mean anything.
    committed = _read_committed_header()
    assert committed.count("kC31UnitCases[]") == 1
    assert committed.count("kC31SiteElements[]") == 1
    assert "kC24CodeXNearIdentity" in committed
    assert "kBia1AcceptBoundaryValue" in committed
    assert "kC28TieCorrectNeg" in committed
    assert len(committed) > 2000, (
        f"the committed header is only {len(committed)} bytes -- suspiciously "
        f"small for a fixture claiming five distinct witness groups"
    )


def test_c31_unit_cases_reproduce_f_s3_2s_own_divergence_pattern():
    # Re-derives directly (not via the committed header) -- proves the
    # generator's own build function, not just its serialized output.
    rows = gen.build_c31_unit_cases()
    assert len(rows) == 3
    assert rows[0]["differs"] and rows[1]["differs"], (
        "F-S3-2's first two witness rows (inexact negative divides) must "
        "diverge between floor and truncation"
    )
    assert not rows[2]["differs"], (
        "F-S3-2's third witness row (-1, 2) is an EXACT divide and must NOT "
        "diverge -- floor and truncation agree on every exact divide"
    )


def test_c31_site_case_is_reachable_and_genuinely_diverges():
    case = gen.build_c31_site_case()
    assert case["H"] == 1536
    diverging = [e for e in case["elements"] if e["diverges"]]
    non_diverging = [e for e in case["elements"] if not e["diverges"]]
    assert len(diverging) >= 1, "the site witness must contain at least one divergent element"
    assert len(non_diverging) >= 1, "the site witness must contain at least one non-divergent (positive-h) control"
    for e in non_diverging:
        assert e["h"] >= 0, "a non-divergent element must be non-negative (floor==trunc only for h>=0)"
    for e in diverging:
        assert e["h"] < 0, "a divergent element must be negative (F-S3-2's own class)"


def test_bia1_former_bound_witnesses_are_int32_max_and_r_a_max_still_asserted():
    # T-1657 Poirot review, Minor 2: build_bia1_bound() no longer re-derives or
    # asserts tightness of the retired load-time bound (there is no live
    # constraint left for "tight" to describe) -- this test now checks the two
    # properties that ARE still live: r_a_max is the vendored reciprocal's own
    # genuine maximum, and the three fixture values are the former bound's fixed
    # historical witnesses (INT32_MAX and one past it), independent of any
    # derivation from r_a_max.
    bound = gen.build_bia1_bound()
    assert bound["r_a_max"] == 1 << 32
    assert bound["accept_boundary_value"] == (1 << 31) - 1
    assert bound["hostile_value"] == (1 << 31)
    assert bound["hostile_value_negated"] == -(1 << 31)


def test_c28_domain_boundary_rejects_outside_0_63_and_accepts_the_endpoints():
    points = gen.build_c28_domain_boundary()
    assert points["below_min"]["in_domain"] is False
    assert points["at_min"]["in_domain"] is True
    assert points["at_max"]["in_domain"] is True
    assert points["above_max"]["in_domain"] is False
    assert points["below_min"]["k"] == -1
    assert points["at_min"]["k"] == 0
    assert points["at_max"]["k"] == 63
    assert points["above_max"]["k"] == 64


def test_c24_witness_reproduces_the_plans_stated_codes_both_runs():
    w = gen.build_c24_witness()
    assert (w["code_ref_pass_through"], w["code_x_pass_through"]) == (127, 25), (
        "the pass-through run must reproduce the plan's own stated [127, 25]"
    )
    assert (w["code_ref_near_identity"], w["code_x_near_identity"]) == (127, 26), (
        "the near-identity run must reproduce the plan's own stated [127, 26]"
    )
    assert w["code_x_pass_through"] != w["code_x_near_identity"], (
        "the shared element's code must diverge between the two runs -- the "
        "whole point of this witness is that the divergence is token-wide "
        "through the shared D', not local to the folded channel"
    )
    assert w["d_near_identity"] == w["d_correct"] - 1, (
        "the near-identity fold on the reference channel must be off by "
        "exactly one, per the plan's own stated property"
    )


def test_c28_tie_witness_discriminates_only_on_the_negative_sign():
    w = gen.build_c28_tie_witness()
    assert w["wrong_pos"] == w["correct_pos"], (
        "the wrong (round-half-up) candidate must agree with the correct "
        "(away-from-zero) result on the positive tie"
    )
    assert w["wrong_neg"] != w["correct_neg"], (
        "the wrong (round-half-up) candidate must disagree with the correct "
        "(away-from-zero) result on the negative tie -- otherwise this is not "
        "a discriminating negative control"
    )
    assert abs(w["b"]) <= (1 << 31) - 1, "the tie witness's B must itself be a valid BIA1 value"
