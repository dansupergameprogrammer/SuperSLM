"""Curie's red suite for check_intmath_softmax_row_q15_nodiscard.py.

Claude/Poirot/ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md
finding C.

Rule coverage (constructed text, StandardsDocument Sec4's population-
validation requirement -- a check shown able to FAIL on a fault it exists to
catch, not only shown to pass on unchanged input) plus the real assertion
against the committed header, which is the finding's own pin: it fails today
and must pass once the header is fixed.
"""
import check_intmath_softmax_row_q15_nodiscard as chk


# --- Rule coverage: constructed fixtures, not the real header. ---


def test_a_nodiscard_declaration_is_detected_as_such():
    sample = "[[nodiscard]] bool SoftmaxRowQ15(const int64_t* scores, size_t width);\n"
    assert chk.declaration_is_nodiscard(sample) is True


def test_a_plain_declaration_is_detected_as_not_nodiscard():
    sample = "bool SoftmaxRowQ15(const int64_t* scores, size_t width, int64_t q_ln2);\n"
    assert chk.declaration_is_nodiscard(sample) is False


def test_a_missing_declaration_raises_rather_than_silently_passing():
    try:
        chk.declaration_is_nodiscard("void SomeOtherFunction();\n")
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "declaration_is_nodiscard must raise when SoftmaxRowQ15 is not declared at all -- a "
            "silent False here would be indistinguishable from a genuinely unmarked declaration"
        )


# --- The pin itself: the real, committed header. ---


def test_softmax_row_q15_declaration_is_nodiscard_in_the_real_header():
    with open(chk.INTMATH_H, "r", encoding="utf-8") as f:
        text = f.read()
    assert chk.declaration_is_nodiscard(text), (
        "include/superslm/intmath.h: SoftmaxRowQ15's declaration must carry [[nodiscard]], matching "
        "this header's own stated doctrine for IExpConstruct/IExpConstantsInDomain -- "
        "'[[nodiscard]] is load-bearing, not decoration. Without it the outcome can be dropped and "
        "the untouched *out read anyway.' Without it here, a caller can silently discard whether a "
        "softmax row's construction was well-formed (as tests/test_main.cpp:9474 already does), "
        "exactly the failure mode commit 6eb1b76 fixed at the kernel and left open at the "
        "declaration (Poirot 2026-07-28 remediation-confirmation review, finding C)"
    )
