"""CI source check: SoftmaxRowQ15's declaration in include/superslm/intmath.h
carries `[[nodiscard]]`, matching this header's own stated doctrine for a
function of this shape.

WHY THIS EXISTS. `intmath.h`'s two siblings that answer the same kind of
question -- "did this construction succeed" -- are both `[[nodiscard]]`:
`IExpConstruct` and `IExpConstantsInDomain`. The header explains why, beside
`IExpConstruct`'s own declaration: "`[[nodiscard]]` is load-bearing, not
decoration. Without it the outcome can be dropped and the untouched `*out`
read anyway." Commit 6eb1b76 turned `SoftmaxRowQ15` from a `void` that
silently discarded `IExpConstruct`'s own outcome into a `bool` a caller can
now read -- but the declaration itself is plain `bool`, so a caller can still
silently discard IT, exactly the failure mode the fix closed one level in.
The pre-existing call site at tests/test_main.cpp:9474 already discards it.

Claude/Poirot/ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md
finding C.
"""
from __future__ import annotations

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
INTMATH_H = os.path.join(_REPO_ROOT, "include", "superslm", "intmath.h")

# An optional `[[nodiscard]]` immediately before the declaration, captured so
# callers can tell whether it was present without re-scanning.
_DECL_PATTERN = re.compile(r"(\[\[nodiscard\]\]\s+)?bool\s+SoftmaxRowQ15\s*\(")


def find_declaration(text: str) -> "re.Match[str] | None":
    """The regex match for SoftmaxRowQ15's declaration in `text`, or None if the
    declaration is not present at all (a fixture or a header that no longer
    declares this symbol)."""
    return _DECL_PATTERN.search(text)


def declaration_is_nodiscard(text: str) -> bool:
    """True iff SoftmaxRowQ15's declaration in `text` is immediately preceded by
    `[[nodiscard]]`. Raises if the declaration is not found at all -- a missing
    declaration is a different failure than an unmarked one, and this check
    exists to catch the second, not silently pass on the first."""
    m = find_declaration(text)
    if m is None:
        raise AssertionError("SoftmaxRowQ15's declaration was not found in the given text at all")
    return m.group(1) is not None


def main() -> int:
    with open(INTMATH_H, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        ok = declaration_is_nodiscard(text)
    except AssertionError as exc:
        print(f"check_intmath_softmax_row_q15_nodiscard.py: FAILED -- {exc}", file=sys.stderr)
        return 1
    if not ok:
        print(
            "check_intmath_softmax_row_q15_nodiscard.py: FAILED -- SoftmaxRowQ15's declaration in "
            f"{INTMATH_H} is not [[nodiscard]], against this header's own stated doctrine for "
            "IExpConstruct/IExpConstantsInDomain",
            file=sys.stderr,
        )
        return 1
    print("check_intmath_softmax_row_q15_nodiscard.py: OK -- SoftmaxRowQ15 is [[nodiscard]]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
