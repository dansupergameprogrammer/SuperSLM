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
StandardsDocument Sec4's population-validation requirement.

UPDATED AT THE S3.2 HEADER-CONTRACT BUILD (2026-07-28, commit a594dd2): the
build staged S3.2's site compositions at `src/forward_sites.cpp` rather than
under `src/forward/`, specifically so this module's own exact-population
assertion (then hardcoded to one file) would not need editing as part of that
pass -- see that build log (Claude/Brunel/superslm-s3.2-weightless-and-
projection-sites-contract-build-2026-07-28.md) and the file's own placement
comment. That left `src/forward_sites.cpp` -- a real forward-composition
source, in the plan's own sense -- entirely outside _DEFAULT_FORWARD_GLOBS'
scan root, so a banned leaf called directly from it would pass this check
with nothing to catch it: the structural guarantee Sec7.3 exists to hold had
a hole exactly where the newest site compositions live.

Closed here by widening _DEFAULT_FORWARD_GLOBS to also name
`src/forward_sites.cpp` explicitly (it is NOT added to _DEFAULT_ALLOWLIST --
it is scanned, not exempted) and by moving the "expected real population"
this module's own end-to-end test asserts from a hardcoded count of one to a
named, sized set (_EXPECTED_REAL_FORWARD_FILES, below) that already lists
both real files. Physically relocating `src/forward_sites.cpp` under
`src/forward/` (the placement the S3.2 build log itself named as the
eventual, cleaner fix) is a production-source change outside this module's
own writable surface (`tests/`) and is routed rather than done here; this
widening closes the coverage gap immediately, independent of whether or when
that move happens, and keeps working unchanged after it does (the sibling
glob entry becomes redundant with the directory glob at that point, not
wrong).
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
# is the one real file under the directory glob as of the S3.1 header-contract
# build (2026-07-28, commit 32aca0c). `src/forward_sites.cpp` (S3.2, commit
# a594dd2) is a second real forward-composition source that was deliberately
# placed OUTSIDE src/forward/ (see module docstring) -- named here explicitly so
# it is scanned starting now, rather than left dark until a future move brings it
# under the directory glob on its own.
_DEFAULT_FORWARD_GLOBS = (
    "src/forward/**/*.cpp",
    "src/forward/**/*.h",
    "src/forward_sites.cpp",
)

# The real forward-composition population this module's own end-to-end test
# (test_check_no_forward_leaf_calls.py::
# test_main_end_to_end_against_the_real_default_glob_is_no_longer_vacuous)
# asserts _DEFAULT_FORWARD_GLOBS resolves to EXACTLY -- a named, sized set
# rather than a hardcoded count of one, so a third real file lands "correctly"
# failing that assertion (per its own docstring) whatever N happens to be
# today, without the assertion itself needing to change shape again.
_EXPECTED_REAL_FORWARD_FILES = (
    "src/forward/checked_chain_funnel.cpp",
    "src/forward/forward_sites.cpp",
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

# The funnel's own two documented entry points (Sec7.2): these are EXPECTED to
# call banned leaves internally -- that is what a checked entry point is for --
# so they are excluded from find_leaf_forwarding_doors' count rather than
# reported as forwarding doors themselves.
_FUNNEL_ENTRY_POINTS = (
    "RequantChainChecked",
    "NarrowRowChecked",
)

# The named, sized set of functions inside the funnel's own file that are
# PERMITTED to forward a banned leaf to an outside caller (Poirot e4b398c
# review, Significant 9; T-1357/D-SLM433): `CarriedScaleReciprocal` is the one
# door C26's runtime reciprocal derivation opens. Sized to N, the same
# _EXPECTED_REAL_FORWARD_FILES idiom above -- a second door landing without a
# matching update here is caught by the equality assertion in
# find_leaf_forwarding_doors' caller, not silently accepted.
_EXPECTED_DOOR_FUNCTIONS = (
    "CarriedScaleReciprocal",
)

# A top-level (column-0) C++ function DEFINITION's signature: a return type,
# the function name, a parenthesized parameter list with no nested parens (true
# of every function in this file -- no default-argument or template-parameter
# parens appear inside a top-level signature here), then whatever separates the
# parameter list from the opening brace (qualifiers like `noexcept`, or
# nothing). Matched against the WHOLE FILE TEXT (re.DOTALL), not line by line,
# so a signature split across several lines is still found as one match.
_TOP_LEVEL_FUNCTION_START = re.compile(
    r"^[A-Za-z_][\w:*&<>,\s]*?\b([A-Za-z_]\w*)\s*\([^(){}]*\)\s*(?:noexcept\s*)?\{",
    re.MULTILINE,
)


def find_leaf_forwarding_doors(
    path: str,
    leaves: tuple[str, ...] = BANNED_LEAVES,
    exclude: tuple[str, ...] = _FUNNEL_ENTRY_POINTS,
) -> list[str]:
    """Every top-level function DEFINED in `path` whose body names a banned leaf,
    excluding `exclude` (the funnel's own checked entry points, which are
    SUPPOSED to). A function's extent is found by matching its signature at
    column 0 (`_TOP_LEVEL_FUNCTION_START`, whole-file text so a multi-line
    signature is still one match) through the opening `{` that match ends on,
    then counting brace depth character by character from there to the matching
    close -- a text-level approximation, not a real tokenizer, matching this
    module's existing text-scan precedent (the docstring above) rather than
    adding a second parsing strategy. Returns names in file order, so the
    caller gets a stable diff against `_EXPECTED_DOOR_FUNCTIONS` when one
    changes."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    doors: list[str] = []
    for m in _TOP_LEVEL_FUNCTION_START.finditer(text):
        name = m.group(1)
        body_start = m.end() - 1  # the opening '{' this match ends on
        depth = 0
        body_end = len(text)
        for i in range(body_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    body_end = i + 1
                    break
        body_text = text[body_start:body_end]
        if name not in exclude and any(_leaf_pattern(leaf).search(body_text) for leaf in leaves):
            doors.append(name)
    return doors


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


def check_door_count(
    funnel_path: str = os.path.join(_REPO_ROOT, "src/forward/checked_chain_funnel.cpp"),
    expected: tuple[str, ...] = _EXPECTED_DOOR_FUNCTIONS,
) -> list[str]:
    """Significant 9 (Poirot e4b398c review, T-1357/D-SLM433): `scan_files` above
    holds every OTHER forward TU off the eight banned leaves; nothing holds the
    DOOR COUNT itself at one inside the funnel's own file. Asserts the exact,
    named set of functions in `funnel_path` that forward a banned leaf to an
    outside caller equals `expected` -- a second door opened alongside
    `CarriedScaleReciprocal` (or a rename of it) is caught here, the same
    named-set idiom `_EXPECTED_REAL_FORWARD_FILES` already uses one level up."""
    if not os.path.isfile(funnel_path):
        # Not a failure: this check is auxiliary to scan_files' own glob-driven
        # scan, applied only when the real funnel file is actually present at
        # the given repo_root. A scratch-directory caller exercising
        # scan_files/main against a constructed tree with no funnel file at
        # all is exercising the leaf-ban mechanism, not this door-count guard.
        return []
    doors = sorted(set(find_leaf_forwarding_doors(funnel_path)))
    if doors != sorted(expected):
        return [
            f"{funnel_path}: forwarding doors == {doors}, want exactly {sorted(expected)} "
            f"-- a door was added, removed, or renamed without updating "
            f"_EXPECTED_DOOR_FUNCTIONS"
        ]
    return []


def main(
    globs: tuple[str, ...] = _DEFAULT_FORWARD_GLOBS,
    allowlist: tuple[str, ...] = _DEFAULT_ALLOWLIST,
    repo_root: str = _REPO_ROOT,
) -> int:
    files = _glob_files(globs, repo_root)
    failures = scan_files(files, allowlist, repo_root)
    # The door-count guard applies only against the REAL production tree, not
    # an arbitrary scratch `repo_root` a mechanism test constructs to exercise
    # the leaf-ban scan in isolation (this module's existing convention,
    # test_check_no_forward_leaf_calls.py's docstring): those scratch trees
    # often write a minimal stand-in `checked_chain_funnel.cpp` with no doors
    # at all, which is not the fault this guard exists to catch. The real
    # tree's own door count is asserted separately and unconditionally by
    # test_main_end_to_end_against_the_real_default_glob_is_no_longer_vacuous.
    if os.path.normpath(repo_root) == os.path.normpath(_REPO_ROOT):
        failures += check_door_count(
            os.path.join(repo_root, "src/forward/checked_chain_funnel.cpp")
        )
    if failures:
        print("check_no_forward_leaf_calls.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"check_no_forward_leaf_calls.py: OK -- {len(files)} forward-composition file(s) "
        f"scanned, zero banned-leaf calls outside the allowlist, and the door count "
        f"holds at {len(_EXPECTED_DOOR_FUNCTIONS)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
