"""Curie's regeneration gate for gen_s3_3_rope_application_site_fixtures.py
(SuperSLM_S3a_WalkingSkeleton_Plan.md Sec6.2 step 3, Sec11 S3.3's own gate
line; D-SLM376, D-SLM383 -- the RoPE application site).

WHY THIS EXISTS. Same shape as test_gen_s3_3_fixtures_regenerates.py's own
precedent: the generated header states "Re-running this script must
reproduce this file byte-for-byte", and this gate is what checks that claim
on every CI run (`python -m pytest tests/ci/ -v` already collects every
module here).

MECHANISM. Calls the generator's own module-level `_build_*()` functions
directly (not `main()`, which writes to disk), so each of this pass's own
mechanism claims is proven live: position 0's cos/sin rows are genuinely the
identity per the PINNED table (not asserted from prose); the rotation at
position (context_cap - 1) is computed via the REAL vendored
`rope.rope_apply_pair`, never re-derived from its formula; the round-trip
witness rows are real, pinned trig-table entries that clear
`RopeApplyPair`'s own |cos|,|sin| <= 2^30 safety bound.

Line-ending normalization matches the S3.1/S3.2/S3.3 precedent (LF; this
Windows checkout has core.autocrlf=true).

Test-design record:
Claude/Curie/superslm-s3.3-rope-application-site-test-design-2026-07-28.md
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import gen_s3_3_rope_application_site_fixtures as gen  # noqa: E402


def _read_committed_header() -> str:
    with open(gen.OUT_PATH, "r", encoding="ascii") as f:
        return f.read()


def test_regenerating_reproduces_the_committed_header_byte_for_byte():
    committed = _read_committed_header()
    regenerated = gen.generate()
    assert regenerated == committed, (
        "tests/gen_s3_3_rope_application_site_fixtures.py's generate() output no "
        "longer matches the committed tests/sslm_s3_3_rope_application_site_"
        "fixtures.h -- regenerate and commit the header, or the generator's own "
        "logic changed without the fixture being regenerated to match"
    )


def test_the_gate_has_something_to_lose_a_nonvacuous_floor():
    committed = _read_committed_header()
    for needle in (
        "kRopeSitePositionZeroCase", "kRopeSitePositionCapMinusOneCase",
        "kRopeSiteRoundTripCosFlat", "kRopeSiteRoundTripSinFlat",
        "kRopeSitePinnedContextCap", "kRopeSiteRoundTripContextCap",
    ):
        assert needle in committed, f"expected witness group {needle!r} missing from the committed header"
    assert len(committed) > 2000, (
        f"the committed header is only {len(committed)} bytes -- suspiciously small "
        f"for a fixture claiming three distinct witness groups over 128-wide rows"
    )


def test_position_zero_is_the_exact_identity_rotation_against_the_pinned_table():
    c = gen._build_position_zero_case()
    assert all(v == 1 << 30 for v in c["cos_row"]), "position 0's pinned cos row must be all-identity (2^30)"
    assert all(v == 0 for v in c["sin_row"]), "position 0's pinned sin row must be all-zero"
    assert c["expected_out"] == c["row"], "position 0 must reproduce the input row exactly, unrotated"
    assert not c["any_raw_out_of_range"], "the identity rotation must never need the clamp"


def test_position_cap_minus_one_genuinely_rotates_and_engages_the_clamp():
    c = gen._build_position_cap_minus_one_case()
    assert c["position"] == gen.PINNED_CONTEXT_CAP - 1
    assert c["expected_out"] != c["row"], (
        "position (context_cap - 1) must produce a genuine, non-identity rotation -- "
        "otherwise this cell cannot distinguish a real rotation from an identity pass-through"
    )
    assert c["any_raw_out_of_range"], (
        "this witness is expected to exercise C33's clamp at the far position boundary; "
        "if this no longer holds after a pinned-table re-vendor, the witness row must be "
        "re-chosen so the clamp is still genuinely exercised, per StandardsDocument Sec4's "
        "own requirement that a discriminating fixture actually discriminate"
    )


def test_round_trip_rows_are_real_pinned_values_within_the_ropeapplypair_safety_bound():
    rt = gen._build_round_trip_rows()
    assert rt["context_cap"] == 4
    flat_cos = [v for row in rt["cos_rows"] for v in row]
    flat_sin = [v for row in rt["sin_rows"] for v in row]
    assert len(flat_cos) == len(flat_sin) == rt["context_cap"] * rt["pairs"]
    assert all(-(1 << 30) <= v <= (1 << 30) for v in flat_cos + flat_sin)
    # Rows 0..3 of the pinned table are reused verbatim -- confirm byte-for-byte
    # against the pinned json this module reads, not merely against its own copy.
    assert rt["cos_rows"] == gen._COS_TABLE[:4]
    assert rt["sin_rows"] == gen._SIN_TABLE[:4]


def test_q_row_generator_is_full_range_and_non_degenerate():
    row = gen._make_q_row(128)
    assert len(row) == 128
    assert min(row) < -100 and max(row) > 100, "the q-row must genuinely span int8 range, not cluster near zero"
    assert len(set(row)) > 64, "the q-row must not be a mostly-repeating (degenerate) pattern"
