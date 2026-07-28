"""CI source check: include/superslm/checked_chain_funnel.h does not describe
`CheckPositionOverCap` as a "stub".

WHY THIS EXISTS. `CheckPositionOverCap`'s own body (checked_chain_funnel.cpp)
is a complete, correct comparison -- verified by execution on 32 operand
pairs including both int64 extremes, zero disagreements with `[0, cap)`
(Claude/Poirot/ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md
finding D). Its own declaration comment, forty lines below the file's opening
summary block, describes it accurately. The opening summary block does not:
it calls the function "a standalone predicate stub for S3.3's own
still-unbuilt position-cap gate" -- the same self-contradicting shape as the
prior round's finding 4, in the same header, introduced by the repair FOR
finding 4. The word to change is "stub"; the predicate is built. Its wiring
into a real forward call site (S3.6's own job) is still owed and this check
does not claim otherwise -- it pins only that the header stops describing an
already-real, already-verified function as unbuilt.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
CHECKED_CHAIN_FUNNEL_H = os.path.join(_REPO_ROOT, "include", "superslm", "checked_chain_funnel.h")

_NEEDLE = "CheckPositionOverCap"


def find_stub_claims(text: str, needle: str = _NEEDLE) -> list[str]:
    """Every line of `text` that both names `needle` and calls it (or the
    surrounding declaration) a "stub". A text scan over lines, not a full
    comment-block parse: over-inclusive on an unrelated "stub" a few lines
    away from `needle` is an accepted false positive for a ban like this one,
    not a soundness gap (matching tests/ci/check_no_forward_leaf_calls.py's
    own documented discipline for its own text scan)."""
    hits: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "stub" not in line.lower():
            continue
        # The claim spans a short paragraph in this header's prose comments;
        # a window of a few lines either side catches the whole sentence
        # without requiring the needle and "stub" share one physical line.
        window = "\n".join(lines[max(0, i - 3):min(len(lines), i + 4)])
        if needle in window:
            hits.append(line.strip())
    return hits


def main() -> int:
    with open(CHECKED_CHAIN_FUNNEL_H, "r", encoding="utf-8") as f:
        text = f.read()
    hits = find_stub_claims(text)
    if hits:
        print(
            f"check_checked_chain_funnel_position_cap_not_a_stub.py: FAILED -- "
            f"{CHECKED_CHAIN_FUNNEL_H} still describes {_NEEDLE} as a stub:",
            file=sys.stderr,
        )
        for h in hits:
            print(f"  - {h}", file=sys.stderr)
        return 1
    print(f"check_checked_chain_funnel_position_cap_not_a_stub.py: OK -- no stub claim over {_NEEDLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
