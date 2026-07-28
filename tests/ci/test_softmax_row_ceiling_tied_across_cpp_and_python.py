"""Cross-artifact gate: the C32 softmax-row numerator ceiling is tied by NAME
AND CITATION only across the C++/Python language boundary (D-SLM367;
`include/superslm/checked_chain_funnel.h`'s own comment on
`kSoftmaxRowMaxSafeExponent`: "No `static_assert` can tie the two across the
C++/Python language boundary ... the tie is BY NAME AND CITATION only ... A
future edit to either side is not caught by this build; it is caught only by
re-reading this comment"). That is a rule someone must remember, and
StandardsDocument §4 names the fix: where a rule can be made structural, make
it structural.

MECHANISM. Reads BOTH definitions and asserts they agree, on the pattern of
this directory's own regeneration gates (`test_gen_s3_3_fixtures_regenerates.py`
et al.):
  - The Python side: `PROB_FRAC_BITS`/`PROB_WIDTH_CEILING` from the vendored,
    hash-pinned excerpt of `Tools/superslm_spike/pipeline.py`
    (`tests/reference/superslm_spike/pipeline_prob_width_ceiling.py`,
    PROVENANCE.md) — imported and executed, not re-typed.
  - The C++ side: `kProbFracBits` (`include/superslm/intmath.h`) and
    `kSoftmaxRowMaxSafeExponent`'s own formula
    (`include/superslm/checked_chain_funnel.h`) — read as source text (the
    same "text scan, no AST" discipline `check_no_forward_leaf_calls.py`
    already uses for a C++ source population) and evaluated in Python to the
    same value the C++ compiler would produce for `(int64_t{1} << 62) >>
    kProbFracBits`, never assumed to match by construction.

This gate does not care WHAT the value is (a future D-SLM367 revision is free
to change it), only that both sides agree on whatever it is — except for one
frozen regression check (the D-SLM367-ratified 2^47) that fails loudly if
either side silently drifts without a recorded ruling.

Test-design record:
Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
"""
from __future__ import annotations

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_REPO_ROOT = os.path.normpath(os.path.join(_TESTS_DIR, os.pardir))
_SPIKE_DIR = os.path.join(_TESTS_DIR, "reference", "superslm_spike")
if _SPIKE_DIR not in sys.path:
    sys.path.insert(0, _SPIKE_DIR)

import pipeline_prob_width_ceiling as py_ceiling  # noqa: E402  (the vendored excerpt)

_INTMATH_H = os.path.join(_REPO_ROOT, "include", "superslm", "intmath.h")
_FUNNEL_H = os.path.join(_REPO_ROOT, "include", "superslm", "checked_chain_funnel.h")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _cpp_prob_frac_bits() -> int:
    text = _read(_INTMATH_H)
    m = re.search(r"kProbFracBits\s*=\s*(\d+)\s*;", text)
    assert m is not None, (
        f"{_INTMATH_H}: kProbFracBits declaration not found -- the text-scan pattern "
        f"this gate depends on no longer matches the header"
    )
    return int(m.group(1))


def _cpp_softmax_row_max_safe_exponent(prob_frac_bits: int) -> int:
    text = _read(_FUNNEL_H)
    m = re.search(
        r"kSoftmaxRowMaxSafeExponent\s*=\s*\(int64_t\{1\}\s*<<\s*(\d+)\)\s*>>\s*kProbFracBits\s*;",
        text,
    )
    assert m is not None, (
        f"{_FUNNEL_H}: kSoftmaxRowMaxSafeExponent's expected formula "
        f"'(int64_t{{1}} << N) >> kProbFracBits' not found -- either the formula "
        f"changed shape (update this gate's own pattern to match) or the constant "
        f"was rewritten as a bare literal (StandardsDocument Sec4: the named-constant "
        f"tie this gate exists to enforce would then be gone entirely)"
    )
    shift_amount = int(m.group(1))
    return (1 << shift_amount) >> prob_frac_bits


def test_prob_frac_bits_agrees_across_cpp_and_python():
    cpp_bits = _cpp_prob_frac_bits()
    py_bits = py_ceiling.PROB_FRAC_BITS
    assert cpp_bits == py_bits, (
        f"kProbFracBits (C++, {_INTMATH_H}) == {cpp_bits}, but "
        f"PROB_FRAC_BITS (Python, pipeline.py) == {py_bits} -- the C32 softmax-row "
        f"fixed-point width has drifted between the two ports"
    )


def test_softmax_row_max_safe_exponent_agrees_across_cpp_and_python():
    cpp_bits = _cpp_prob_frac_bits()
    cpp_value = _cpp_softmax_row_max_safe_exponent(cpp_bits)
    py_value = py_ceiling.PROB_WIDTH_CEILING
    assert cpp_value == py_value, (
        f"kSoftmaxRowMaxSafeExponent (C++, {_FUNNEL_H}) == {cpp_value}, but "
        f"PROB_WIDTH_CEILING (Python, pipeline.py, D-SLM367) == {py_value} -- the "
        f"named constant D-SLM367 ties by citation only has drifted; this is exactly "
        f"the failure mode this gate exists to catch (checked_chain_funnel.h's own "
        f"comment: 'a future edit to either side is not caught by this build; it is "
        f"caught only by re-reading this comment' -- this gate is that re-read, "
        f"automated)"
    )


def test_the_tied_value_is_the_d_slm367_ratified_2_pow_47():
    # A frozen regression check, distinct from the two cross-artifact ties
    # above: this one DOES care what the value is, because D-SLM367 is a
    # specific, dated ruling (2^47, not 2^48-1, not unguarded) and a silent
    # drift to a different-but-still-self-consistent value on both sides
    # would pass the two tests above while quietly reversing Dan's ruling.
    assert py_ceiling.PROB_WIDTH_CEILING == 1 << 47 == 140737488355328
    cpp_bits = _cpp_prob_frac_bits()
    assert _cpp_softmax_row_max_safe_exponent(cpp_bits) == 1 << 47
