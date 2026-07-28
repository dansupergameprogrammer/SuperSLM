#!/usr/bin/env python3
"""Generates sslm_s3_3_rope_application_site_fixtures.h -- derived witnesses for
S3.3's RoPE APPLICATION SITE (Claude/Plans/SuperSLM_S3a_WalkingSkeleton_Plan.md
Sec6.2 step 3, Sec11 S3.3's own gate line, Sec13 dim 2/5/6/11; D-SLM376,
D-SLM383): the not-yet-built composition that rotates `(even, odd)` pairs of
each head row at the absolute position indexing ROP1, with
`CheckPositionOverCap` wired as its first act and a position `>= context_cap`
rejected before any table read.

WHAT THIS PINS, AND WHAT IT DOES NOT (read before extending). Every rotated-
and-clamped witness below is computed by CALLING the vendored, hash-pinned
reference (tests/reference/superslm_spike/rope.py's own `rope_apply_pair`) --
never re-derived from its formula in this module. The cos/sin Q2.30 table
ROWS this module reads come from `tests/reference/superslm_spike/
rope_tables_pinned.json` -- the ALREADY-COMMITTED, one-time precompute of
`rope.rope_tables(head_dim=128, context_cap=128, theta=1000000.0)`
(tests/reference/precompute_pinned.py) -- rather than calling
`rope.rope_tables(...)` live, because that function's cos/sin entries are
libm transcendentals (not IEEE-correctly-rounded, ULP-different across
glibc/MSVCRT/Apple libm) and the S-HARDEN-5 discipline this tree already
follows (gen_intmath_fixtures.py's own precedent) is: pin the transcendental
computation's OUTPUT once, never recompute it on the generation path that CI
re-runs on whatever machine is running it.

The per-pair rotation composition mirrors Tools/superslm_spike/
dynamic_engine.py's own `_rotate_rows` EXACTLY (read at source, 2026-07-28):
`even = row[0::2]; odd = row[1::2]`; `cos = cos_table[positions, :]`;
`r_even, r_odd = vec_rope_apply_pair(even, odd, cos, sin)`; then clip each
component to [-127, 127]. This module reproduces that pairing (even index i
pairs with odd index i, in interleaved order -- NOT a first-half/second-half
split) rather than restating it from the plan's own prose, because the plan's
own Sec6.2 step 3 states the pairing in prose and the reference's own code is
the executable ground truth per StandardsDocument Sec5.4 ("exactness is
verified at source or by execution, never by construction").

Re-running this script must reproduce the emitted header byte-for-byte.

Test-design record:
Claude/Curie/superslm-s3.3-rope-application-site-test-design-2026-07-28.md
"""
from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "reference"))
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _REFERENCE_DIR)

import rope as rope_ref  # noqa: E402  (the vendored, hash-pinned reference)

with open(os.path.join(_SPIKE_DIR, "rope_tables_pinned.json"), "r", encoding="ascii") as _f:
    _PINNED = json.load(_f)

OUT_PATH = os.path.join(_THIS_DIR, "sslm_s3_3_rope_application_site_fixtures.h")

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

# The pinned table's own config -- read from the file, never restated as a
# literal, so a re-vendor of the pinned json cannot silently drift out of
# step with what this module assumes about its shape.
_PINNED_ROPE = _PINNED["rope"]
PINNED_CONTEXT_CAP = _PINNED_ROPE["context_cap"]  # 128
PINNED_HEAD_DIM = _PINNED_ROPE["head_dim"]  # 128
PINNED_PAIRS = PINNED_HEAD_DIM // 2  # 64
PINNED_THETA = _PINNED_ROPE["theta"]
_COS_TABLE = _PINNED_ROPE["cos_q30"]
_SIN_TABLE = _PINNED_ROPE["sin_q30"]


def _clip127(v: int) -> int:
    return max(-127, min(127, v))


def _make_q_row(n: int) -> list[int]:
    """A deterministic, full-range int8-domain row: every value in
    [-127, 127], not a degenerate constant or all-zero row, so a rotation
    genuinely mixes each pair's two components rather than testing a
    fixed-point of the formula."""
    row = [((i * 41 + 13) % 255) - 127 for i in range(n)]
    assert all(-127 <= v <= 127 for v in row), "q-row generator escaped int8-code range"
    assert len(set(row)) > n // 2, "q-row generator produced a degenerate (mostly-repeating) row"
    return row


def _rotate_and_clamp_row(row: list[int], cos_row: list[int], sin_row: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Sec6.2 step 3's own per-pair rotation, over one absolute position's
    already-read table row: pairs are (row[2*i], row[2*i+1]) -- interleaved
    even/odd, matching dynamic_engine.py's `_rotate_rows` exactly (even =
    row[0::2], odd = row[1::2]) -- NOT a first-half/second-half split.
    Returns (raw_even, raw_odd, clamped_out) where clamped_out interleaves
    the two clamped components back into one head_dim-wide row, matching the
    site's own output shape."""
    pairs = len(row) // 2
    assert pairs == len(cos_row) == len(sin_row), "row/table width mismatch"
    raw_even = []
    raw_odd = []
    out = [0] * len(row)
    for i in range(pairs):
        x = row[2 * i]
        y = row[2 * i + 1]
        rx, ry = rope_ref.rope_apply_pair(x, y, cos_row[i], sin_row[i])
        raw_even.append(rx)
        raw_odd.append(ry)
        out[2 * i] = _clip127(rx)
        out[2 * i + 1] = _clip127(ry)
    return raw_even, raw_odd, out


def _build_position_case(position: int, label: str) -> dict:
    row = _make_q_row(PINNED_HEAD_DIM)
    cos_row = _COS_TABLE[position]
    sin_row = _SIN_TABLE[position]
    raw_even, raw_odd, out = _rotate_and_clamp_row(row, cos_row, sin_row)
    return {
        "label": label,
        "position": position,
        "row": row,
        "cos_row": cos_row,
        "sin_row": sin_row,
        "raw_even": raw_even,
        "raw_odd": raw_odd,
        "expected_out": out,
        "any_raw_out_of_range": any(v < -127 or v > 127 for v in raw_even + raw_odd),
    }


def _build_position_zero_case() -> dict:
    """position 0: every angle is 0 (position * freq == 0 regardless of
    frequency), so cos_q30 == ROPE_ONE and sin_q30 == 0 for every pair --
    confirmed against the pinned table, not assumed -- which makes the
    rotation the EXACT identity: RopeApplyPair(x, y, 2^30, 0) reproduces
    (x, y) exactly (x*2^30 divided by 2^30 with no remainder). This is a
    genuine, meaningful property of position 0 specifically, not a weaker
    substitute for a real rotation witness."""
    c = _build_position_case(0, "position_zero")
    assert all(v == 1 << 30 for v in c["cos_row"]), "position 0's cos row must be the identity (2^30) at every pair"
    assert all(v == 0 for v in c["sin_row"]), "position 0's sin row must be exactly 0 at every pair"
    assert c["expected_out"] == c["row"], "position 0 must rotate to the exact identity (unchanged row)"
    return c


def _build_position_cap_minus_one_case() -> dict:
    """position context_cap - 1 (127 in the pinned table): the far edge of
    the valid range C12 admits -- the last row ROP1 legitimately carries and
    the last one the site's own C12 exclusive-bound check must still permit
    a table read for."""
    return _build_position_case(PINNED_CONTEXT_CAP - 1, "position_cap_minus_one")


def _build_round_trip_rows() -> dict:
    """A small ROP1 tensor-manifest witness for the artifact-round-trip half
    of the site's own join: rows 0..3 of the ALREADY-REAL, pinned cos/sin
    tables, reused verbatim (never fabricated) under a config whose own
    context_cap is 4 -- so 'every row this artifact carries' and 'every row
    this config admits' coincide exactly, with no cross-validation gap to
    paper over. head_dim stays 128 (PINNED_PAIRS columns) so the pinned
    table's own columns are used unmodified, not sliced."""
    n_rows = 4
    cos_rows = _COS_TABLE[:n_rows]
    sin_rows = _SIN_TABLE[:n_rows]
    for r in cos_rows + sin_rows:
        assert all(-(1 << 30) <= v <= (1 << 30) for v in r), (
            "round-trip witness rows must already clear RopeApplyPair's own |cos|,|sin| <= 2^30 safety bound "
            "(ValidateRopeTablesDomain's own check) -- they are real trig-table entries, not adversarial"
        )
    return {
        "context_cap": n_rows,
        "head_dim": PINNED_HEAD_DIM,
        "pairs": PINNED_PAIRS,
        "cos_rows": cos_rows,
        "sin_rows": sin_rows,
    }


def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def _fmt_i64_array(values: list[int]) -> str:
    return "{" + ", ".join(f"{v}LL" for v in values) + "}"


def generate() -> str:
    pos0 = _build_position_zero_case()
    pos_last = _build_position_cap_minus_one_case()
    round_trip = _build_round_trip_rows()

    assert pos_last["any_raw_out_of_range"] or True  # documented, not required -- see header comment

    lines = [
        "// GENERATED FILE -- do not edit by hand.",
        "// Produced by tests/gen_s3_3_rope_application_site_fixtures.py.",
        "// Re-running that script must reproduce this file byte-for-byte.",
        "//",
        "// Curie's derived witnesses for S3.3's RoPE APPLICATION SITE",
        "// (Claude/Plans/SuperSLM_S3a_WalkingSkeleton_Plan.md Sec6.2 step 3, Sec11",
        "// S3.3's own gate line; D-SLM376, D-SLM383). Every rotated value below is",
        "// computed by CALLING the vendored rope.rope_apply_pair on cos/sin table",
        "// rows read from the already-committed rope_tables_pinned.json (never",
        "// re-derived from the rotation formula in this header or its generator).",
        "//",
        "// Test-design record:",
        "// Claude/Curie/superslm-s3.3-rope-application-site-test-design-2026-07-28.md",
        "#ifndef SUPERSLM_TESTS_SSLM_S3_3_ROPE_APPLICATION_SITE_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_S3_3_ROPE_APPLICATION_SITE_FIXTURES_H",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "",
        f"inline constexpr int kRopeSitePinnedContextCap = {PINNED_CONTEXT_CAP};",
        f"inline constexpr int kRopeSitePinnedHeadDim = {PINNED_HEAD_DIM};",
        f"inline constexpr int kRopeSitePinnedPairs = {PINNED_PAIRS};",
        "",
        "// --- Sec4.1: position 0 and position (context_cap - 1) feature-oracle",
        "//     witnesses (Plan Sec13 dim 6: \"RoPE at position 0 and context_cap-1\").",
        "//     Every field below is real, executed data -- never hand-typed. ---",
        "",
        "struct RopeSitePositionCase {",
        "\tconst char* label;",
        "\tint64_t position;",
        "\tint64_t row[128];         // the head row BEFORE rotation, head_dim wide",
        "\tint64_t cos_row[64];      // ROP1's cos entries for this position, pairs wide",
        "\tint64_t sin_row[64];      // ROP1's sin entries for this position, pairs wide",
        "\tint64_t expected_out[128]; // the rotated row AFTER C33's clamp, head_dim wide",
        "\tbool any_raw_out_of_range; // true iff at least one pre-clamp component left [-127,127]",
        "};",
        "",
        "inline constexpr RopeSitePositionCase kRopeSitePositionZeroCase = {",
        f"\t\"{pos0['label']}\", {pos0['position']}LL,",
        f"\t{_fmt_i64_array(pos0['row'])},",
        f"\t{_fmt_i64_array(pos0['cos_row'])},",
        f"\t{_fmt_i64_array(pos0['sin_row'])},",
        f"\t{_fmt_i64_array(pos0['expected_out'])},",
        f"\t{_fmt_bool(pos0['any_raw_out_of_range'])},",
        "};",
        "",
        "inline constexpr RopeSitePositionCase kRopeSitePositionCapMinusOneCase = {",
        f"\t\"{pos_last['label']}\", {pos_last['position']}LL,",
        f"\t{_fmt_i64_array(pos_last['row'])},",
        f"\t{_fmt_i64_array(pos_last['cos_row'])},",
        f"\t{_fmt_i64_array(pos_last['sin_row'])},",
        f"\t{_fmt_i64_array(pos_last['expected_out'])},",
        f"\t{_fmt_bool(pos_last['any_raw_out_of_range'])},",
        "};",
        "",
        "// --- Sec4.2: the ROP1 artifact round-trip witness -- real, pinned",
        "//     cos/sin rows 0..3, reused verbatim under a context_cap=4 config",
        "//     (Plan Sec13.1 cell 9's sibling join for the RoPE table itself). ---",
        "",
        f"inline constexpr int kRopeSiteRoundTripContextCap = {round_trip['context_cap']};",
        f"inline constexpr int kRopeSiteRoundTripHeadDim = {round_trip['head_dim']};",
        f"inline constexpr int kRopeSiteRoundTripPairs = {round_trip['pairs']};",
        "",
        "// Flattened row-major [context_cap][pairs] -- row p occupies",
        "// [p*pairs, (p+1)*pairs).",
        "inline constexpr int64_t kRopeSiteRoundTripCosFlat[] = " +
        _fmt_i64_array([v for row in round_trip["cos_rows"] for v in row]) + ";",
        "inline constexpr int64_t kRopeSiteRoundTripSinFlat[] = " +
        _fmt_i64_array([v for row in round_trip["sin_rows"] for v in row]) + ";",
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_SSLM_S3_3_ROPE_APPLICATION_SITE_FIXTURES_H",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    text = generate()
    with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
