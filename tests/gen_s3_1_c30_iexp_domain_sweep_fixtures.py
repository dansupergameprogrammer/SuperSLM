#!/usr/bin/env python3
"""Generates sslm_s3_1_c30_iexp_domain_sweep_fixtures.h -- the C30 derived-operand
oracle-pinning cell for S3.1 (Claude/Plans/SuperSLM_S3a_WalkingSkeleton_Plan.md
Sec11 S3.1, Sec5's C30 references, Sec13 dim 6; D-SLM318, board T-132).

WHAT THIS PINS, AND WHAT IT DOES NOT (read before extending). The plan's S3.1 red-cell
list calls for "a differential cell ... asserting the design's C30 SITE and
IExpConstantsInDomain agree at every point" across e in [-90, +8] at mantissa extremes
m in {2**30, 2**31-1}. "The design's C30 site" is the production forward-composition
function that derives (q_ln2, q_b, q_c) from a per-query carried scale (m, e) and calls
IExpConstantsInDomain on the result -- Sec6.2 step 5 / Sec7.2's C30 predicate. That site
does not exist in D:\\SuperSLM yet (it is part of the attention interior, S3.3's build,
Sec11 S3.3) and no header in include/superslm/ declares it (verified: `grep -rn
"IExpConstantsInDomain\\|iexp_scale_constants" include/ src/` finds IExpConstantsInDomain's
own declaration/definition and nothing named for a C30 derivation wrapper). Per this
campaign's own test-design record Sec3, that half of the cell is BLOCKED on a
not-yet-existing entry point and is named there, not invented here.

What THIS module derives, and what it is good for: the (m, e) -> (q_ln2, q_b, q_c)
mapping IS already fully specified -- C30's derivation formula, `iexp_scale_constants`,
is pinned prose in Tools/superslm_spike/intmath.py (vendored copy at
tests/reference/superslm_spike/intmath.py) and is unaffected by S3a; only the C++ SITE
that will call it is missing. So this generator calls the vendored reference's
`iexp_scale_constants` (never re-derives its formula) to produce the exact
(q_ln2, q_b, q_c) triple C30 would hand to `IExpConstantsInDomain` at each (m, e), and
independently computes the oracle's own in-domain verdict via the SAME
arbitrary-precision five-line decomposition tests/gen_iexp_domain_fixtures.py already
established as IExpConstantsInDomain's independent oracle (never by calling the C++
primitive at generation time). The emitted fixture lets a C++ test call the REAL,
already-shipped `IExpConstantsInDomain` on these golden (q_ln2, q_b, q_c) triples and
compare its answer to this oracle -- which is the closest available substitute for the
full "design's C30 site" differential cell until that site exists, and is explicitly
named as a substitute, not a completion, in the test-design record.

The representative `q` used is 0 (the max-shifted-logit convention's own dominant case),
justified by intmath.h's own documented shortcut: "The one-call-per-triple shortcut
holds iff 2*q_b >= q_ln2 - 1 ... The first [q_p=0, i.e. q=0] dominates". Every emitted
row's `shortcut_condition_holds` field records whether that inequality holds for ITS
own (q_ln2, q_b) -- checked, not assumed -- so a C++ test can refuse to trust the q=0
representative on any row where the shortcut's own precondition fails.

Two further things this pins, both stated so a reader does not have to re-derive them:
  - `derivation_ok`: false if EITHER (a) the vendored `iexp_scale_constants` itself
    raises ValueError at (m, e) -- the offline construction has its OWN domain (every
    one of q_ln2/q_b/q_c's derivation shifts must be non-negative) -- OR (b) any of the
    three derived constants exceeds int64 range (observed: q_c reaches ~1.9e28 at
    e=-77, versus INT64_MAX ~9.2e18), which a real C++ call site could not even form as
    the `int64_t` IExpConstantsInDomain's signature requires. Both are upstream of and
    distinct from IExpConstantsInDomain's own representability question. Rows with
    `derivation_ok == false` carry no (q_ln2, q_b, q_c) triple and the C++ test must not
    call `IExpConstantsInDomain` on them -- there is nothing valid to call it with.
    Both construction-domain boundaries belong to the not-yet-built C30 derivation site
    (S3.3) and are recorded here, not asserted.
  - `expected_in_domain`, for rows with `derivation_ok == true`: the corrected,
    current-truth C30 domain rule this design pins (Sec7.2's table, D-SLM85/D-SLM107):
    `e >= -60, or (e == -61 and m >= 1268234713)`. This is transcribed from the design
    document's prose, independently of `iexp_scale_constants` and independently of
    IExpConstantsInDomain's own decomposition -- a THIRD, textual source -- so the cell
    genuinely cross-checks three things against each other rather than one thing
    against itself.

Re-running this script must reproduce the emitted header byte-for-byte.

Test-design record: Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "reference"))
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _REFERENCE_DIR)

import intmath as im  # noqa: E402  (the vendored, hash-pinned reference)

import json  # noqa: E402

with open(os.path.join(_SPIKE_DIR, "rope_tables_pinned.json"), "r", encoding="ascii") as _f:
    _PINNED_ROPE = json.load(_f)

# Same pin gen_intmath_fixtures.py applies: im._LN2 resolved from the pinned double,
# never from this runner's live libm math.log(2) (S-HARDEN-5 design Sec3.2).
im._LN2 = float.fromhex(_PINNED_ROPE["ln2"]["hex"])

OUT_PATH = os.path.join(_THIS_DIR, "sslm_s3_1_c30_iexp_domain_sweep_fixtures.h")

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
I_EXP_CLIP_N = 30
_IEXP_QFMT = 30

# pipeline.py's own derivation (NOT vendored into tests/reference/; recomputed here from
# the vendored intmath.py's own _POLY_A/_POLY_B/_POLY_C and the pinned ln2 double, never
# from a live math.log/math.floor(scale) call at a runtime scale):
#   _IEXP_LN2_Q = floor(ln2 * 2**QFMT); _IEXP_B_Q = floor(_POLY_B * 2**QFMT);
#   _IEXP_CA_Q  = floor((_POLY_C / _POLY_A) * 2**QFMT)
_IEXP_LN2_Q = int(im._LN2 * (1 << _IEXP_QFMT))
_IEXP_B_Q = int(im._POLY_B * (1 << _IEXP_QFMT))
_IEXP_CA_Q = int((im._POLY_C / im._POLY_A) * (1 << _IEXP_QFMT))


def _in_domain_oracle(q: int, q_ln2: int, q_b: int, q_c: int) -> bool:
    """IExpConstantsInDomain's own independent oracle -- verbatim the formula
    tests/gen_iexp_domain_fixtures.py establishes and this module does not re-derive:
    q_ln2 <= INT64_MAX // I_EXP_CLIP_N, then the five-line decomposition, then the
    result's int64 representability. Arbitrary-precision throughout."""
    if q_ln2 < 1 or q_ln2 > INT64_MAX // I_EXP_CLIP_N:
        return False
    clip_lo = -I_EXP_CLIP_N * q_ln2
    clipped = max(q, clip_lo)
    z = (-clipped) // q_ln2
    q_p = clipped + z * q_ln2
    base = q_p + q_b
    value = (base * base + q_c) >> z
    return INT64_MIN <= value <= INT64_MAX


def _corrected_c30_domain_rule(m: int, e: int) -> bool:
    """The design's own pinned current-truth rule (Sec7.2's table; D-SLM85 worst-case
    mantissa, D-SLM107 full mantissa sweep): e >= -60, or e == -61 with
    m >= 1,268,234,713. Transcribed from the design prose, independent of both
    `iexp_scale_constants` and `_in_domain_oracle`."""
    if e >= -60:
        return True
    if e == -61 and m >= 1268234713:
        return True
    return False


def _derive_row(m: int, e: int) -> dict:
    row = {"m": m, "e": e}
    try:
        q_ln2, q_b, q_c = im.iexp_scale_constants(
            m, e, _IEXP_LN2_Q, _IEXP_QFMT, _IEXP_B_Q, _IEXP_QFMT, _IEXP_CA_Q, _IEXP_QFMT
        )
    except ValueError:
        row["derivation_ok"] = False
        return row
    # A SECOND, distinct upstream-of-IExpConstantsInDomain guard, found by execution
    # rather than assumed: iexp_scale_constants's own arithmetic is arbitrary-precision
    # Python, so it happily returns a q_c many orders of magnitude past int64 range at
    # the more-negative end of this sweep (observed: ~1.9e28 at e=-77, versus INT64_MAX
    # ~9.2e18). `IExpConstantsInDomain`'s C++ signature takes q_c as `int64_t` -- a value
    # this large cannot even be FORMED as one, let alone passed to the predicate, so
    # calling it here would require silently truncating/wrapping a value the real
    # (not-yet-built) C++ derivation site would have to guard against forming in the
    # first place. That guard is upstream construction-domain work belonging to the
    # derivation site (S3.3), same bucket as the shift<0 rejection above, not a question
    # IExpConstantsInDomain itself answers -- so these rows are marked not-callable here
    # too, rather than misrepresented via a wrapped int64 that the real primitive never
    # actually receives.
    if not (INT64_MIN <= q_ln2 <= INT64_MAX and INT64_MIN <= q_b <= INT64_MAX
            and INT64_MIN <= q_c <= INT64_MAX):
        row["derivation_ok"] = False
        return row
    row["derivation_ok"] = True
    row["q_ln2"] = q_ln2
    row["q_b"] = q_b
    row["q_c"] = q_c
    row["shortcut_condition_holds"] = (2 * q_b >= q_ln2 - 1)
    row["expected_in_domain_via_oracle"] = _in_domain_oracle(0, q_ln2, q_b, q_c)
    row["expected_in_domain_via_corrected_rule"] = _corrected_c30_domain_rule(m, e)
    return row


def build_rows() -> list[dict]:
    rows = []
    mantissas = (1 << 30, (1 << 31) - 1)
    for e in range(-90, 9):  # [-90, +8] inclusive
        for m in mantissas:
            rows.append(_derive_row(m, e))
    return rows


def _fmt_row(row: dict) -> str:
    m = row["m"]
    e = row["e"]
    if not row["derivation_ok"]:
        return (
            "\t{ /*m=*/%dLL, /*e=*/%d, /*derivation_ok=*/false, /*q_ln2=*/0LL, "
            "/*q_b=*/0LL, /*q_c=*/0LL, /*shortcut_condition_holds=*/false, "
            "/*expected_in_domain=*/false, /*agrees_with_corrected_rule_text=*/true }," % (m, e)
        )
    # expected_in_domain is ALWAYS the arbitrary-precision decomposition oracle -- the
    # one IExpConstantsInDomain actually implements, verified correct independently of
    # C30 (tests/gen_iexp_domain_fixtures.py's own formula). It is never overridden by
    # the corrected-rule text below, which this generator discovered does NOT hold
    # uniformly across the full stated sweep (see build_rows()'s disagreement report,
    # printed to stderr and named in the test-design record rather than silently
    # patched here).
    oracle = row["expected_in_domain_via_oracle"]
    rule = row["expected_in_domain_via_corrected_rule"]
    return (
        "\t{ /*m=*/%dLL, /*e=*/%d, /*derivation_ok=*/true, /*q_ln2=*/%dLL, "
        "/*q_b=*/%dLL, /*q_c=*/%dLL, /*shortcut_condition_holds=*/%s, "
        "/*expected_in_domain=*/%s, /*agrees_with_corrected_rule_text=*/%s },"
        % (
            m,
            e,
            row["q_ln2"],
            row["q_b"],
            row["q_c"],
            "true" if row["shortcut_condition_holds"] else "false",
            "true" if oracle else "false",
            "true" if (oracle == rule) else "false",
        )
    )


def generate() -> str:
    rows = build_rows()
    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tests/gen_s3_1_c30_iexp_domain_sweep_fixtures.py: (q_ln2, q_b, q_c)",
        "// triples are computed by CALLING the vendored, hash-pinned reference's",
        "// iexp_scale_constants (tests/reference/superslm_spike/intmath.py) -- never",
        "// re-derived here. expected_in_domain is the arbitrary-precision",
        "// IExpConstantsInDomain decomposition oracle (authoritative -- it is what the",
        "// shipped primitive actually implements). agrees_with_corrected_rule_text",
        "// additionally records whether the D-SLM85/D-SLM107 textual e-rule",
        "// (e >= -60, or e == -61 with m >= 1268234713) predicts the SAME verdict --",
        "// false rows are a generator-discovered finding (the textual rule does not hold",
        "// uniformly across this sweep's positive tail), reported to stderr at generation",
        "// time and named in the test-design record, never silently reconciled here.",
        "// Re-running this script must reproduce this file byte-for-byte.",
        "//",
        "// Test-design record:",
        "// Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md",
        "#ifndef SUPERSLM_TESTS_SSLM_S3_1_C30_IEXP_DOMAIN_SWEEP_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_S3_1_C30_IEXP_DOMAIN_SWEEP_FIXTURES_H",
        "",
        "#include <cstddef>",
        "",
        "namespace superslm_test {",
        "",
        "struct C30IExpDomainSweepCase {",
        "\tlong long m;",
        "\tint e;",
        "\tbool derivation_ok;",
        "\tlong long q_ln2;",
        "\tlong long q_b;",
        "\tlong long q_c;",
        "\tbool shortcut_condition_holds;",
        "\tbool expected_in_domain;",
        "\tbool agrees_with_corrected_rule_text;",
        "};",
        "",
        "inline constexpr C30IExpDomainSweepCase kC30IExpDomainSweepCases[] = {",
    ]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append("};")
    lines.append("")
    lines.append(
        "inline constexpr size_t kC30IExpDomainSweepCasesCount = "
        "sizeof(kC30IExpDomainSweepCases) / sizeof(kC30IExpDomainSweepCases[0]);"
    )
    lines.append("")
    lines.append("}  // namespace superslm_test")
    lines.append("")
    lines.append("#endif  // SUPERSLM_TESTS_SSLM_S3_1_C30_IEXP_DOMAIN_SWEEP_FIXTURES_H")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows = build_rows()
    disagreements = [
        r for r in rows
        if r["derivation_ok"]
        and r["expected_in_domain_via_oracle"] != r["expected_in_domain_via_corrected_rule"]
    ]
    text = generate()
    with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT_PATH}: {len(rows)} rows, {len(disagreements)} disagree with the "
          f"corrected-rule text")
    if disagreements:
        es = sorted({r["e"] for r in disagreements})
        print(f"  disagreement e-range: [{min(es)}, {max(es)}] -- see the test-design "
              f"record's routed-finding section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
