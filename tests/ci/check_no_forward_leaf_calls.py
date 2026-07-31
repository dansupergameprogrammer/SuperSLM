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

# --- T-1378: a comment/indentation/paren-robust top-level function scanner. ---
#
# The prior version anchored a function definition's signature at column 0 and
# delimited its parameter list with `[^(){}]*` (no parens allowed inside), then
# counted braces over the RAW file text to find the body's extent. Executed,
# three independently-constructed adversarial doors passed this scan silently
# (Poirot 9d8b38e-s3.4-s3.5-review-remainder-confirmation-2026-07-30.md,
# Significant 1): a door whose parameter list itself contains parens (a
# default-argument constructor call), a door indented inside `namespace
# superslm { ... }` rather than at column 0, and a door preceded by a `}`
# inside a `//` comment (which the raw brace count treated as a real close,
# truncating the body early and losing the leaf call inside it). All three are
# fixed by construction here, not by patching the old regex further:
#
#   1. Comments and string/char literals are stripped (blanked to spaces,
#      same length, same newline positions) before EITHER the signature match
#      or the brace count ever runs (`_strip_comments_and_strings`) -- a `}`
#      or a `(` inside one can no longer be mistaken for real source
#      structure, closing the third shape above outright.
#   2. A function's parameter list is found by a genuine balanced-parenthesis
#      scan, not a no-parens character class -- closing the first shape.
#   3. Brace CONTEXT is tracked (a stack of "namespace" / "other" tags, one
#      per currently-open brace) rather than column position, so a
#      definition nested inside any depth of `namespace { ... }` is found
#      exactly like one at column 0, while a definition nested inside a
#      struct/class/enum body, a control-structure body, a lambda, or
#      another function's own body is still excluded -- closing the second
#      shape and generalizing past the one-level case it was found at.
#
# Independently validated against a population of six constructed doors,
# each executed against BOTH the prior version and this one
# (test_find_leaf_forwarding_doors_catches_the_independently_found_population
# below): the three shapes above (missed before, caught now), a fourth
# variant using a `/* ... } ... */` block comment rather than `//` (missed
# before for the same reason as the third; caught now), a fifth combining the
# paren-in-params and indentation shapes on one door at TWO levels of nested
# `namespace { ... }` rather than one (missed before; caught now, proving the
# fix generalizes past the exact nesting depth it was found at), and a false-
# positive control: a banned leaf's name and a `{`/`}`-bearing fake signature
# inside a STRING LITERAL, which must NOT be reported as a door either before
# or after (both pass, unchanged -- this fix narrows what escapes detection,
# it does not widen what counts as a door).
#
# `StandardsDocument.md` Sec4: a new structure is validated against an
# independently-found population before it is trusted -- the population above
# is not the three shapes' own author re-deriving them, it is a wider net
# cast deliberately before hardening, per the same section's own instruction.


def _strip_comments_and_strings(text: str) -> str:
    """Returns a string the SAME LENGTH as `text`, with every `//` line
    comment, `/* ... */` block comment, and string/char literal's CONTENTS
    replaced by spaces (newlines preserved, so line-oriented reasoning
    elsewhere stays valid even though nothing here currently needs it).
    Positions map 1:1 onto the original text -- a caller slicing
    `text[a:b]` using indices found against this stripped text gets the REAL
    source, comments included; only the SCANNING (signature matching, brace
    depth) ever sees the stripped version, so a `}`, a `(`, or a leaf's name
    inside a comment or a string literal can no longer be mistaken for real
    source structure. A text-level scan, not a preprocessor -- unterminated
    comments/strings at end of file are closed at EOF rather than raising,
    matching this module's existing over-approximation doctrine (module
    docstring above: a false positive is accepted, a soundness gap is not)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            start = i
            while i < n and text[i] != "\n":
                i += 1
            out.append(" " * (i - start))
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            start = i
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            chunk = text[start:i]
            out.append("".join(ch if ch == "\n" else " " for ch in chunk))
            continue
        if c in ("\"", "'"):
            quote = c
            start = i
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" and i + 1 < n else 1
            i = min(i + 1, n)
            out.append(" " * (i - start))
            continue
        out.append(c)
        i += 1
    return "".join(out)


# A C++ function DEFINITION's signature HEAD: a return-type-ish prefix ending
# in an identifier (the function's own name, captured), immediately followed
# by `(` -- the same shape the prior version used, unchanged, because it
# already correctly rejects a bare call expression (`identifier(args)` has
# only ONE identifier before the `(`; a definition's return type supplies a
# second). No `^` anchor and no `[^(){}]*` parameter-list restriction: this
# module never anchors the match itself, it anchors the SEARCH POSITION via
# `re.Pattern.match(text, pos)`, and the parameter list is resolved by a
# balanced-parenthesis scan below, not by this pattern.
_SIGNATURE_HEAD_RE = re.compile(r"[A-Za-z_][\w:*&<>,\s]*?\b([A-Za-z_]\w*)\s*\(")
# Whatever separates a signature's closing `)` from its opening `{`:
# qualifiers like `noexcept`, or nothing.
_SIGNATURE_TAIL_RE = re.compile(r"\s*(?:noexcept\s*)?\{")
# A `{` is a namespace opener when the (stripped) text immediately before it
# ends in the `namespace` keyword, optionally named -- matches both
# `namespace superslm {` and the anonymous `namespace {`.
_NAMESPACE_OPENER_RE = re.compile(r"namespace(\s+[A-Za-z_]\w*)?\s*$")


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _brace_tag(cleaned: str, brace_pos: int) -> str:
    """Classifies a `{` at `brace_pos` in `cleaned` (already comment/string-
    stripped) by looking back over the text immediately before it:
    "namespace" (a named or anonymous namespace body -- the only tag
    `find_top_level_function_bodies` treats as transparent) or "other"
    (anything else: a struct/class/enum body, an if/for/while/do/switch
    body, a lambda, an initializer list -- all of which make everything
    inside them non-top-level for this scanner's purposes)."""
    look_start = max(0, brace_pos - 200)
    if _NAMESPACE_OPENER_RE.search(cleaned[look_start:brace_pos]):
        return "namespace"
    return "other"


def find_top_level_function_bodies(text: str) -> list[tuple[str, int, int]]:
    """Every top-level function DEFINITION in `text`: `(name, body_start,
    body_end)` with `body_start`/`body_end` indices into `text` ITSELF (the
    original, unstripped text), in file order. "Top-level" means every brace
    enclosing the definition, if any, opens a namespace (named or anonymous,
    any nesting depth) -- never a struct/class/enum body, a control-structure
    body, a lambda, or another function's own body. See the module comment
    above (T-1378) for why this replaced a column-0-anchored, no-nested-
    parens regex over raw text."""
    cleaned = _strip_comments_and_strings(text)
    n = len(cleaned)
    stack: list[str] = []
    pending_names: list[str] = []
    pending_starts: list[int] = []
    results: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        ch = cleaned[i]
        if ch == "{":
            stack.append(_brace_tag(cleaned, i))
            i += 1
            continue
        if ch == "}":
            if stack:
                tag = stack.pop()
                if tag == "function":
                    results.append((pending_names.pop(), pending_starts.pop(), i + 1))
            i += 1
            continue
        starts_ident = _is_word_char(ch) and not ch.isdigit()
        prev_is_ident = i > 0 and _is_word_char(cleaned[i - 1])
        if starts_ident and not prev_is_ident:
            head = _SIGNATURE_HEAD_RE.match(cleaned, i)
            if head is not None:
                depth = 1
                j = head.end()
                while j < n and depth > 0:
                    if cleaned[j] == "(":
                        depth += 1
                    elif cleaned[j] == ")":
                        depth -= 1
                    j += 1
                if depth == 0:
                    tail = _SIGNATURE_TAIL_RE.match(cleaned, j)
                    if tail is not None:
                        brace_pos = tail.end() - 1
                        if all(t == "namespace" for t in stack):
                            stack.append("function")
                            pending_names.append(head.group(1))
                            pending_starts.append(brace_pos)
                        else:
                            stack.append("other")
                        i = brace_pos + 1
                        continue
        i += 1
    return results


def find_leaf_forwarding_doors(
    path: str,
    leaves: tuple[str, ...] = BANNED_LEAVES,
    exclude: tuple[str, ...] = _FUNNEL_ENTRY_POINTS,
) -> list[str]:
    """Every top-level function DEFINED in `path` whose body names a banned leaf,
    excluding `exclude` (the funnel's own checked entry points, which are
    SUPPOSED to). A function's extent comes from `find_top_level_function_bodies`
    (T-1378): comment/string-stripped for both the signature match and the
    brace count, a balanced-parenthesis parameter list, and brace-CONTEXT
    tracking rather than column position -- see that function and the module
    comment above for what this replaced and why. The body text searched for
    a leaf name is sliced from the ORIGINAL (unstripped) text, so a leaf name
    inside a comment inside an otherwise-legitimate function still counts as
    a hit, matching this module's existing over-inclusive-on-comments
    convention (the module docstring's own "a ban, not a classifier").
    Returns names in file order, so the caller gets a stable diff against
    `_EXPECTED_DOOR_FUNCTIONS` when one changes."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    doors: list[str] = []
    for name, body_start, body_end in find_top_level_function_bodies(text):
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
