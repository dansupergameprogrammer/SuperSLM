"""Curie's red suite for check_checked_chain_funnel_position_cap_not_a_stub.py.

Claude/Poirot/ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md
finding D.

Rule coverage (constructed text, StandardsDocument Sec4's population-
validation requirement) plus the real assertion against the committed
header, which is the finding's own pin: it fails today (the header's opening
summary block calls CheckPositionOverCap "a standalone predicate stub") and
must pass once the header is corrected.
"""
import check_checked_chain_funnel_position_cap_not_a_stub as chk


# --- Rule coverage: constructed fixtures, not the real header. ---


def test_a_nearby_stub_claim_over_the_named_function_is_detected():
    sample = (
        "// This file also declares CheckPositionOverCap, a standalone\n"
        "// predicate stub for a still-unbuilt gate.\n"
    )
    hits = chk.find_stub_claims(sample)
    assert hits, "a 'stub' claim within a few lines of CheckPositionOverCap must be detected"


def test_an_unrelated_stub_claim_far_from_the_named_function_is_not_flagged():
    sample = (
        "// Some other function is a stub for now.\n" + ("// filler\n" * 10) +
        "SslmForwardStatus CheckPositionOverCap(int64_t position, int64_t context_cap);\n"
    )
    hits = chk.find_stub_claims(sample)
    assert not hits, (
        f"an unrelated 'stub' claim many lines away from CheckPositionOverCap must not be flagged "
        f"as describing it -- got {hits!r}"
    )


def test_a_declaration_with_no_stub_claim_at_all_is_clean():
    sample = (
        "// CheckPositionOverCap's own body is a complete, correct comparison.\n"
        "SslmForwardStatus CheckPositionOverCap(int64_t position, int64_t context_cap);\n"
    )
    assert chk.find_stub_claims(sample) == []


# --- The pin itself: the real, committed header. ---


def test_checked_chain_funnel_h_does_not_call_check_position_over_cap_a_stub():
    with open(chk.CHECKED_CHAIN_FUNNEL_H, "r", encoding="utf-8") as f:
        text = f.read()
    hits = chk.find_stub_claims(text)
    assert not hits, (
        "include/superslm/checked_chain_funnel.h describes CheckPositionOverCap as a 'stub' near: "
        f"{hits!r} -- its body (checked_chain_funnel.cpp) is a complete, correct comparison, "
        "verified on 32 operand pairs including both int64 extremes with zero disagreements "
        "against [0, cap); its own declaration comment forty lines below already says so. The word "
        "to change is 'stub' in the header's opening summary block; the predicate is built, only its "
        "WIRING into a real forward call site is still owed (Poirot 2026-07-28 remediation-"
        "confirmation review, finding D -- the same self-contradicting shape as the prior round's "
        "finding 4, introduced by the repair FOR finding 4)"
    )
