"""CI source check: no translation unit in the forward composition may name a
checked-chain-funnel leaf directly, outside the funnel's own file and the leaf
certification TUs (SuperSLM_S3a_WalkingSkeleton_Plan.md Sec7.3; T-200,
Sec17.3 cell 5; S3.1).

This is the structural half of the funnel (Sec7.3): a lint rule and a wrapper
type were both considered and rejected there in favour of a single-entry-point
funnel PLUS this check, which is what stops a future call site from reaching a
caller-ensures leaf directly instead of routing through the checked entry
points (`RequantChainChecked` / `NarrowRowChecked`).

Modelled on the precedent already in the tree, tests/check_no_pow_operator.py,
with the one property that precedent does NOT need and this check does: an
INPUT SET derived from a directory glob rather than a hardcoded file tuple
(Sec7.3 names check_no_pow_operator.py's hardcoded two-element tuple as "the
precedent's own weakness, not inherited"). One property does NOT transfer:
check_no_pow_operator.py gets its precision from `ast.parse` over Python; a
ban on C++ identifiers in forward TUs has no AST behind it here and is a text
scan (Sec7.3, verbatim) -- over-inclusive on a comment or a string literal
naming a leaf is an accepted false-positive, not a soundness gap, because the
rule is a ban, not a classifier.

WHERE THIS STANDS AS OF THE S3.1 HEADER-CONTRACT BUILD (2026-07-28, commit
32aca0c, T-200): `src/forward/checked_chain_funnel.cpp` now exists and
_DEFAULT_FORWARD_GLOBS resolves to exactly that one real file against the
tree. main()'s end-to-end run against it is no longer vacuous --
test_check_no_forward_leaf_calls.py's own
`test_main_end_to_end_against_the_real_default_glob_is_no_longer_vacuous`
asserts the real, exact population and that the real tree passes -- though
that cell proves the WIRING (the real glob and allowlist agree on the real
file), not the MECHANISM (a real banned-leaf call being caught): the funnel's
own file is allowlisted, and `scan_files` skips reading an allowlisted path's
content at all, so every mechanism cell still drives this module against
constructed fixture files standing in for "a forward TU," per
StandardsDocument Sec4's population-validation requirement. The site-
composition sources for S3.2-S3.9 (Claude/Plans/
SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11) still land under the same glob as
they are built; no further default change is anticipated for those, since the
glob root already covers `src/forward/**`.
"""
from __future__ import annotations

import glob
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# The eight funnel leaves named at Sec7.3, verbatim: the forward-leaf caller-ensures
# set the funnel exists to keep off every call site but its own.
BANNED_LEAVES = (
    "MaxAbsReduce",
    "MaxAbsReduceWide",
    "RowBoundsWide",
    "NormalizeScale",
    "DynamicScaleReciprocal",
    "RequantTokenCode",
    "RequantTokenCodeWide",
    "NarrowAccumulatorToI32",
)

# The forward-composition source root (Sec11): src/forward/checked_chain_funnel.cpp
# is the one real file under it as of the S3.1 header-contract build (2026-07-28,
# commit 32aca0c); S3.2-S3.9's site-composition sources land under the same glob
# root as they are built (see module docstring).
_DEFAULT_FORWARD_GLOBS = (
    "src/forward/**/*.cpp",
    "src/forward/**/*.h",
)

# Relative-to-repo-root paths permitted to name a banned leaf directly. Only a
# path _DEFAULT_FORWARD_GLOBS can match is meaningful here: scan_files's
# allowlist is consulted solely against the files main() globs in, so an entry
# outside src/forward/** is never reached by the default scan and would read
# as protection while providing none (Poirot ac34677 review finding N4). The
# leaf certification TUs and tests/test_main.cpp's own direct calls into the
# funnel's leaves live outside that glob root; a caller that needs them
# allowlisted supplies them explicitly to scan_files (see
# test_check_no_forward_leaf_calls.py's allowlist-control cells), rather than
# carrying them here where they can never be consulted.
_DEFAULT_ALLOWLIST = (
    "src/forward/checked_chain_funnel.cpp",
)

_LEAF_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _leaf_pattern(leaf: str) -> re.Pattern[str]:
    """A whole-identifier match for `leaf`: not preceded or followed by a character
    that could extend it into a different identifier. This is what keeps a call to
    `MaxAbsReduceWide` from also being reported as a hit on `MaxAbsReduce` (a strict
    prefix of it, and separately banned) -- the lookaround anchors both ends."""
    cached = _LEAF_PATTERN_CACHE.get(leaf)
    if cached is None:
        cached = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(leaf) + r"(?![A-Za-z0-9_])")
        _LEAF_PATTERN_CACHE[leaf] = cached
    return cached


def find_banned_leaf_uses(path: str, leaves: tuple[str, ...] = BANNED_LEAVES) -> list[tuple[int, str]]:
    """Every (1-based line number, leaf name) hit in `path`, in file order. A text
    scan, not an AST walk (Sec7.3) -- a leaf name inside a comment or a string
    literal is still reported; this is a deliberate over-approximation of a ban,
    not an attempt to classify intent."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, start=1):
        for leaf in leaves:
            if _leaf_pattern(leaf).search(line):
                hits.append((lineno, leaf))
    return hits


def _glob_files(globs: tuple[str, ...], repo_root: str) -> list[str]:
    out: list[str] = []
    for g in globs:
        out.extend(glob.glob(os.path.join(repo_root, g), recursive=True))
    return sorted(out)


def scan_files(
    file_paths: list[str],
    allowlist: tuple[str, ...] = _DEFAULT_ALLOWLIST,
    repo_root: str = _REPO_ROOT,
    leaves: tuple[str, ...] = BANNED_LEAVES,
) -> list[str]:
    """Scans every file in `file_paths` (absolute or repo-root-relative), skipping
    anything whose repo-root-relative path (normalized, so `/` and `\\` both match)
    is in `allowlist`. Returns one formatted failure string per hit, empty if clean."""
    allow_norm = {os.path.normpath(p) for p in allowlist}
    failures: list[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
        rel = os.path.relpath(abs_p, repo_root)
        rel_norm = os.path.normpath(rel)
        if rel_norm in allow_norm:
            continue
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        for lineno, leaf in find_banned_leaf_uses(abs_p, leaves):
            failures.append(f"{rel}:{lineno}: names banned leaf '{leaf}' outside the funnel's own file")
    return failures


def main(
    globs: tuple[str, ...] = _DEFAULT_FORWARD_GLOBS,
    allowlist: tuple[str, ...] = _DEFAULT_ALLOWLIST,
    repo_root: str = _REPO_ROOT,
) -> int:
    files = _glob_files(globs, repo_root)
    failures = scan_files(files, allowlist, repo_root)
    if failures:
        print("check_no_forward_leaf_calls.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"check_no_forward_leaf_calls.py: OK -- {len(files)} forward-composition file(s) "
        f"scanned, zero banned-leaf calls outside the allowlist"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
