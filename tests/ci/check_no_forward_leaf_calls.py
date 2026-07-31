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
import json
import os
import re
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_INCLUDE_DIR = os.path.join(_REPO_ROOT, "include")

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

# --- T-1381: a Clang-AST derivation of the door count, replacing the text
# scanner (D-SLM464; SuperSLM_S3a_WalkingSkeleton_Plan.md Sec7.3a). ---
#
# T-1378 hardened the prior column-0-anchored, no-nested-parens regex scanner
# with comment/string stripping, a balanced-parenthesis parameter list, and
# brace-context tracking -- and an independent sweep against the HARDENED
# scanner (Poirot 5af6ab5-t1377-t1378-review-2026-07-31.md, Significant 2)
# still found three more missed shapes, one of them newly introduced by the
# hardening itself: a trailing return type (`auto Foo(...) -> int64_t {`,
# already a live style at src/bad_alloc_wrap.h:49) silently passed because
# `_SIGNATURE_TAIL_RE` admitted only optional `noexcept` between `)` and `{`;
# a door inside an `extern "C" { ... }` linkage block silently passed because
# the brace-context tracker only ever tagged a brace "namespace" or "other",
# never "extern C"; and a C++14 digit separator (`1'000`) appearing an ODD
# number of times in the file blanked EVERYTHING from it to end of file,
# because the comment/string stripper treated every `'` as a char-literal
# opener with no way to tell a digit separator from a real char literal by
# text alone -- silently erasing every door after it.
#
# Per `StandardsDocument.md` Sec4, a defect class that survives a second,
# carefully-executed repair is repaired by removing the class, not by
# patching a third special case onto the same recognizer: comments, string
# and char literals, digit separators, namespace/extern-"C" nesting at any
# depth, and trailing return types are not six special cases to enumerate by
# hand, they are ordinary nodes a real C++ grammar already resolves. This
# section replaces the hand-rolled text scanner (the comment/string stripper
# and the brace-tracking function-body finder, both removed) with a Clang AST
# walk over the funnel's own one file, mirroring the pattern already
# CI-wired in this repo at `tests/ci/derive_bad_alloc_membership.py`/
# `tools/ci/check_bad_alloc_contract.py` (same pinned `clang++-18`, same
# `ubuntu-latest` runner image; `.github/workflows/tests.yml`'s
# `bad-alloc-membership-check` job proves the exact binary is present
# there). The whole-tree leaf ban (`scan_files`/`find_banned_leaf_uses`,
# below) is UNCHANGED -- it is a deliberately over-approximate text scan by
# design (a ban, not a classifier; Sec7.3's own module docstring) and
# carries no soundness gap this replaces.
#
# `-fsyntax-only -Xclang -ast-dump=json` still emits a full AST for a scratch
# file with no `#include`s at all (every banned-leaf name and every type
# unresolved): Clang's error-recovery machinery represents an unresolved call
# as an `UnresolvedLookupExpr` carrying the callee's name directly, and an
# unresolved type does not prevent the enclosing `FunctionDecl`, its name, or
# its `CompoundStmt` body from appearing in the dump. This is what lets the
# same population of constructed, include-free scratch fixtures this suite
# already uses drive the AST-based mechanism exactly as they drove the text
# scanner, with no synthetic header prelude required (confirmed by execution:
# `clang++ -Xclang -ast-dump=json -fsyntax-only` on a bare
# `int64_t CarriedScaleReciprocal(int64_t m) { return
# DynamicScaleReciprocal(m); }` -- no includes, no declarations -- exits 0
# and dumps a `FunctionDecl` named `CarriedScaleReciprocal` whose body
# contains an `UnresolvedLookupExpr` named `DynamicScaleReciprocal`).
#
# Validated against the same population `test_check_no_forward_leaf_calls.py`
# already carries (five constructed doors plus the false-positive control)
# PLUS the three shapes this review found still missed (trailing return type,
# `extern "C" { }`, the odd-count digit separator) -- nine cases in total,
# each executed against BOTH the text-scan mechanism being replaced
# (reproduced verbatim in the test file, since this module no longer carries
# it) and this AST-based one, so the improvement is measured rather than
# asserted (StandardsDocument Sec4).
#
# CORRECTED 2026-07-31 (Poirot 5eff945-t1380-t1381-t1382-review-2026-07-31.md,
# Significant 1; T-1383): the paragraph just above this one shipped true; the
# ORIGINAL `_TOP_LEVEL_FUNCTION_KINDS` docstring (the comment directly below,
# now corrected in place) shipped a parity claim that was false the moment it
# was written -- admitting `FunctionDecl` only was argued to exclude a member
# function "exactly as the prior text scanner's brace-context tag 'other'
# did," which holds for an IN-CLASS member definition but not for an
# OUT-OF-LINE one (`Ret S::Door(...) { ... }`), which is not nested inside a
# struct body at all and which BOTH retired scanners caught. `find_leaf_
# forwarding_doors` now admits member-function kinds alongside `FunctionDecl`
# (both in-class and out-of-line member definitions surface as the identical
# `CXXMethodDecl` kind, confirmed by execution -- indistinguishable by kind
# alone), guarded so a lambda's own closure-type call operator -- itself a
# `CXXMethodDecl`, lexically nested inside a `LambdaExpr` -- is never counted
# as a separate top-level door: that guard is what keeps this widening from
# regressing `test_t1381_ast_mechanism_finds_a_door_hidden_inside_a_lambda_
# in_real_declared_code` into reporting two doors for one. Population widened
# to fourteen cases (the nine above, unchanged and re-verified against the
# fixed mechanism, plus the out-of-line member door this correction targets,
# plus two further shapes an independent sweep found -- an in-class member
# door, now also caught as a consequence of the same fix, and a namespace-
# scope function pointer initialised to a leaf's address, which remains
# missed by this mechanism and by both retired scanners alike: a `VarDecl`
# carries no `FunctionDecl`/`CXXMethodDecl` node for either generation to
# find, so this is documented parity, not a regression, and is not fixed
# here). See `test_check_no_forward_leaf_calls.py`'s own T-1383 section for
# all three new cases, each executed against a verbatim reproduction of the
# mechanism this correction replaces, exactly as the nine cases above are
# measured against the mechanism T-1381 replaced.

_DEFAULT_CLANGXX = os.environ.get("SUPERSLM_CLANGXX", "clang++")

# The AST node kinds `find_leaf_forwarding_doors` treats as door candidates:
# free/namespace-scope functions AND member functions, in-class or defined
# out-of-line -- both shapes surface as the SAME kind string in Clang's own
# dump (confirmed by execution: an out-of-line `Ret S::Door(...) { ... }`
# and an in-class `struct S { Ret Door(...) { ... } };` both produce a
# `CXXMethodDecl` node carrying a `CompoundStmt` body, indistinguishable by
# kind alone). (Corrected 2026-07-31, Poirot Significant 1 /T-1383: this set
# previously admitted `FunctionDecl` only, on a comment claiming parity with
# the prior text scanner's brace-context tag "other" -- false for an
# out-of-line member definition, which is not nested inside a struct body at
# all and was caught by both retired scanners. `find_leaf_forwarding_doors`'
# own lambda guard below is what keeps this widening from also reporting a
# lambda's closure-type call operator as a spurious extra door.)
_TOP_LEVEL_FUNCTION_KINDS = (
    "FunctionDecl",
    "CXXMethodDecl",
    "CXXConstructorDecl",
    "CXXDestructorDecl",
    "CXXConversionDecl",
)


class ClangUnavailable(RuntimeError):
    """Raised when clang++ cannot run -Xclang -ast-dump=json at all -- the
    environment problem, distinct from a genuine population mismatch (mirrors
    tests/ci/derive_bad_alloc_membership.py's own exception of the same
    name)."""


def _run_clang_ast_dump(source_path: str, clangxx: str = _DEFAULT_CLANGXX,
                         include_dir: str | None = None) -> dict:
    """Runs `clangxx -Xclang -ast-dump=json -fsyntax-only` against
    `source_path` and returns the parsed JSON root. A genuine environment
    failure (a missing clang++, or a timeout) raises `ClangUnavailable`. A
    RECOVERABLE semantic error -- an unresolved identifier or an unknown type
    name, the ordinary case for a scratch fixture with no `#include`s --
    does NOT raise, even though Clang's own exit code is nonzero in that
    case (confirmed by execution: `clang++ -fsyntax-only` on
    `int64_t Foo(int64_t m) { return Bar(m); }` with no declarations at all
    exits 1, reporting "unknown type name 'int64_t'" twice, yet still emits
    a complete, valid AST dump on stdout carrying `Foo`'s `FunctionDecl`, its
    `CompoundStmt` body, and an `UnresolvedLookupExpr` named `Bar` -- the
    exit code reflects the semantic errors, not whether the AST dump itself
    succeeded). What DOES distinguish a genuine parse failure (an
    unterminated brace, for instance) is that clang emits NO stdout at all
    in that case -- so the discriminator this function uses is "did valid
    JSON come back on stdout," not the process's exit code."""
    cmd = [clangxx, "-std=c++20", "-Xclang", "-ast-dump=json", "-fsyntax-only", "-x", "c++"]
    if include_dir is not None:
        cmd += ["-I", include_dir]
    cmd.append(source_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except FileNotFoundError as e:
        raise ClangUnavailable(f"{clangxx} not found on PATH: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ClangUnavailable(f"{clangxx} timed out dumping {source_path}") from e
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"clang++ produced no usable AST dump for {source_path} (exit {proc.returncode}):\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        ) from e


def _node_leaf_name(node: dict, leaves: tuple[str, ...]) -> str | None:
    """If `node` is a reference to one of `leaves` -- resolved (`DeclRefExpr`/
    `MemberExpr` whose `referencedDecl.name` is a leaf, the shape a call
    takes when the callee IS declared, e.g. the real production file with its
    real `#include`s) or unresolved (`UnresolvedLookupExpr` whose own `name`
    is a leaf directly, the shape an include-free scratch fixture takes) --
    returns the matched leaf name, else None."""
    referenced = node.get("referencedDecl")
    if isinstance(referenced, dict) and referenced.get("name") in leaves:
        return referenced["name"]
    if node.get("kind") == "UnresolvedLookupExpr" and node.get("name") in leaves:
        return node["name"]
    return None


def _body_calls_leaf(body: dict, leaves: tuple[str, ...]) -> bool:
    """Whether ANY node in `body`'s subtree (a `CompoundStmt`) references a
    banned leaf -- walked with no regard to further nesting (an `if`, a
    lambda, another block), matching the text scanner's own "anywhere in the
    body text" convention exactly, just precise about what counts as a
    reference instead of a raw substring match."""
    stack = [body]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        if _node_leaf_name(n, leaves) is not None:
            return True
        stack.extend(n.get("inner") or [])
    return False


def find_leaf_forwarding_doors(
    path: str,
    leaves: tuple[str, ...] = BANNED_LEAVES,
    exclude: tuple[str, ...] = _FUNNEL_ENTRY_POINTS,
    clangxx: str = _DEFAULT_CLANGXX,
    include_dir: str | None = _INCLUDE_DIR,
) -> list[str]:
    """Every top-level function DEFINED in `path` whose body names a banned
    leaf, excluding `exclude` (the funnel's own checked entry points, which
    are SUPPOSED to). T-1381 (D-SLM464): derived from a real Clang AST dump
    of `path` rather than a hand-rolled text scan -- see the module comment
    above for what this replaced and why. A "top-level function DEFINED" is
    an AST node whose kind is one of `_TOP_LEVEL_FUNCTION_KINDS` (a free or
    namespace-scope function, or a member function -- in-class or defined
    out-of-line, indistinguishable by kind alone; corrected 2026-07-31,
    Poirot Significant 1 / T-1383) located in `path` itself (tracked via the
    dump's own file-attribution, the same delta-encoded `loc.file`/
    `includedFrom` walk `derive_bad_alloc_membership.py` already uses) that
    carries a `CompoundStmt` body -- a declaration with no body (e.g. the
    header's own prototype, or an out-of-line member's own in-class
    declaration) is not a door. A candidate lexically nested inside a
    `LambdaExpr` -- the lambda's own closure-type call operator, which is a
    `CXXMethodDecl` like any other member function -- is never itself
    reported as a separate door: T-1383's widening from `FunctionDecl` alone
    would otherwise also flag that operator, double-counting a leaf call the
    enclosing door (the function the lambda is defined inside) already
    accounts for, exactly as
    `test_t1381_ast_mechanism_finds_a_door_hidden_inside_a_lambda_in_real_
    declared_code` requires. Returns names in file order, so the caller gets
    a stable diff against `_EXPECTED_DOOR_FUNCTIONS` when one changes."""
    root = _run_clang_ast_dump(path, clangxx, include_dir)
    target = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    state = {"file": None}
    doors: list[str] = []

    def walk(n: object, in_lambda: bool) -> None:
        if not isinstance(n, dict):
            return
        kind = n.get("kind")
        loc = n.get("loc") or {}
        if "file" in loc:
            state["file"] = loc["file"]
        cur_file = str(state["file"] or "").replace("\\", "/")
        if (
            not in_lambda
            and kind in _TOP_LEVEL_FUNCTION_KINDS
            and cur_file
            and os.path.normpath(os.path.abspath(cur_file)).replace("\\", "/") == target
        ):
            name = n.get("name")
            body = next(
                (c for c in (n.get("inner") or []) if isinstance(c, dict) and c.get("kind") == "CompoundStmt"),
                None,
            )
            if body is not None and name not in exclude and name not in doors and _body_calls_leaf(body, leaves):
                doors.append(name)
        child_in_lambda = in_lambda or kind == "LambdaExpr"
        for c in n.get("inner") or []:
            walk(c, child_in_lambda)

    walk(root, False)
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
        try:
            failures += check_door_count(
                os.path.join(repo_root, "src/forward/checked_chain_funnel.cpp")
            )
        except ClangUnavailable as e:
            # T-1381 (D-SLM464): the door-count guard is now Clang-AST-derived
            # and needs a real clang++ on PATH (or SUPERSLM_CLANGXX). Reported
            # as a clean gate failure, mirroring tools/ci/check_bad_alloc_
            # contract.py's own handling of the same exception -- an
            # environment gap, not a Python traceback.
            print(f"check_no_forward_leaf_calls.py: clang++ unavailable -- {e}", file=sys.stderr)
            return 1
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
