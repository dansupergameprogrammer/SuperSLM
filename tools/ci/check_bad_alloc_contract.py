#!/usr/bin/env python3
"""Production CI gate for the S-HARDEN-7 "throws only std::bad_alloc"
contract (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_Design-2026-07-23.md
Sec3.1, Sec3.3 steps 1-3; T-411). This is the "membership-check" job the
design commissions: it does, mechanically, on every CI run, what the design
document's Sec3.1 table did by hand.

Step 1 (derive membership): reuses tests/ci/derive_bad_alloc_membership.py's
AST walk over the nine `include/superslm/*.h` headers -- the corrected,
four-condition rule (not `=delete`d; not the enclosing class's own copy/move
special member; not `noexcept`; either an explicit artifact-sourced
parameter or a std::string/std::vector<T> return) -- so the population this
tool checks is re-derived from the headers on every run, never a maintained
list that can drift.

Step 2 (check wrapping): for each derived member, locates its public
definition in the corresponding .cpp file (by source-file convention: every
top-level function/method definition in this project starts at column 0)
and confirms the definition's body calls the shared wrap helper
(src/bad_alloc_wrap.h's WrapBadAllocContract) around a call to a
`<Name>Impl`-suffixed inner function -- design Sec3.1's rename-and-wrap
shape -- rather than containing the validation logic directly.

Step 3 (fail naming every unwrapped member): `--list-unwrapped` prints one
`header:line:name` row per currently-unwrapped derived member and exits 1 if
any exist, 0 if none do -- the population-validation cell design Sec3.2
commissions ("run membership-check against the current state with zero
wraps built, and confirm it fails naming exactly the eighteen sites").
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_CI_DIR = _THIS_DIR
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOLS_CI_DIR))
_TESTS_CI_DIR = os.path.join(_REPO_ROOT, "tests", "ci")

sys.path.insert(0, _TESTS_CI_DIR)
import derive_bad_alloc_membership as dbam  # noqa: E402  (path set above)

# header (as named in the derived population, e.g. "artifact.h") -> the .cpp
# file in src/ that carries every public definition for that header. Every
# derived member's Impl-suffixed inner function and its public wrapper live
# in this file (design Sec3.1's eighteen-site table; confirmed by direct
# read of each src/*.cpp during this tool's own construction).
_HEADER_TO_CPP = {
    "artifact.h": "artifact.cpp",
    "model.h": "model.cpp",
    "tokenizer.h": "tokenizer.cpp",
    "proof_manifest.h": "proof_manifest.cpp",
    "sha256.h": "sha256.cpp",
}


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_balanced_body(source: str, search_from: int) -> str:
    """From the first `{` at or after `search_from`, return the brace-balanced
    body text (including the enclosing braces). Empty string if unbalanced
    or no `{` is found."""
    brace_start = source.find("{", search_from)
    if brace_start == -1:
        return ""
    depth = 0
    i = brace_start
    n = len(source)
    while i < n:
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start:i + 1]
        i += 1
    return ""


def _find_definition_body(cpp_source: str, name: str, enclosing: str | None) -> str | None:
    """Find the qualified (`Enclosing::name` or `Enclosing<...>::name`) or free
    (`name`) function's DEFINITION in `cpp_source` -- distinguished from a mere
    call site by the project's own consistent style: every top-level definition
    starts at column 0 (no leading whitespace), while every call site in this
    codebase is indented inside another function's body. Returns the
    brace-balanced body text of the first such definition, or None if not
    found.

    T-2125: `Enclosing::name` is now `Enclosing(?:<...>)?::name` -- a class
    TEMPLATE's own member is qualified in its .cpp definition as
    `Enclosing<Kind>::name` (the template parameter, not a concrete type
    argument; this project's own convention, matching SslmAmplifyingFoldScaleView
    <Kind>::Parse), which the old literal `f"{enclosing}::{name}"` string could
    never match -- every templated member read as though its definition did not
    exist, misreporting a real, wrapped member as unwrapped regardless of its
    actual .cpp body (found live, `derive_bad_alloc_membership.py`'s own sibling
    correction is the other half of this same gap: `enclosing` itself was `None`
    for a template's implicit instantiations before that fix, so this one alone
    would not have been reachable for two of the member's three derived
    entries)."""
    if enclosing:
        qualified_pattern = re.escape(enclosing) + r"(?:<[^()]*>)?::" + re.escape(name)
    else:
        qualified_pattern = re.escape(name)
    # `\s*\(` immediately after the name excludes a `<name>Impl(` match (the
    # literal characters "Impl" between the name and "(" fail `\s*`).
    pattern = re.compile(
        r"^(?:[A-Za-z_][\w:<>,\s\*&]*[\s\*&])?" + qualified_pattern + r"\s*\(",
        re.MULTILINE,
    )
    for m in pattern.finditer(cpp_source):
        body = _extract_balanced_body(cpp_source, m.end())
        if body:
            return body
    return None


def _is_wrapped(body: str, name: str) -> bool:
    """True if `body` (a definition's brace-balanced text) matches design
    Sec3.1's rename-and-wrap shape: calls the shared helper around a call to
    `<name>Impl(` (unqualified -- called from within the definition's own
    scope, whether that scope is a class's static/instance method or a free
    function's enclosing translation unit)."""
    if "WrapBadAllocContract" not in body:
        return False
    impl_pattern = re.compile(r"\b" + re.escape(name) + r"Impl\s*\(")
    return bool(impl_pattern.search(body))


def find_unwrapped_members() -> list[dict]:
    """Step 1 + step 2: derive the population from the real headers, then
    check each member's own .cpp definition for the wrap shape. Returns the
    list of derived-but-unwrapped members, each the same dict shape
    derive_bad_alloc_membership.py produces (header, line, name, enclosing,
    ...)."""
    population = dbam.derive_population_per_header()
    cpp_cache: dict[str, str] = {}
    unwrapped: list[dict] = []
    for member in population:
        header = member["header"]
        cpp_name = _HEADER_TO_CPP.get(header)
        if cpp_name is None:
            unwrapped.append(member)  # no known .cpp -- cannot be wrapped
            continue
        cpp_path = os.path.join(_REPO_ROOT, "src", cpp_name)
        if cpp_path not in cpp_cache:
            if not os.path.isfile(cpp_path):
                cpp_cache[cpp_path] = ""
            else:
                cpp_cache[cpp_path] = _read(cpp_path)
        cpp_source = cpp_cache[cpp_path]
        body = _find_definition_body(cpp_source, member["name"], member["enclosing"])
        if body is None or not _is_wrapped(body, member["name"]):
            unwrapped.append(member)
    return unwrapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-unwrapped",
        action="store_true",
        help="Print one header:line:name row per currently-unwrapped derived "
        "member and exit 1 if any exist, 0 if none do.",
    )
    args = parser.parse_args()

    try:
        unwrapped = find_unwrapped_members()
    except dbam.ClangUnavailable as e:
        print(f"check_bad_alloc_contract.py: clang++ unavailable -- {e}", file=sys.stderr)
        return 1

    if not args.list_unwrapped:
        # Default behavior is the same as --list-unwrapped -- there is
        # nothing else this gate does.
        args.list_unwrapped = True

    if unwrapped:
        for m in sorted(unwrapped, key=lambda x: (x["header"], x["line"] or 0)):
            print(f"{m['header']}:{m['line']}:{m['name']}")
        print(
            f"check_bad_alloc_contract.py: FAILED -- {len(unwrapped)} member(s) of the "
            "\"throws only std::bad_alloc\" contract's derived population are not "
            "wrapped (design Sec3.1)",
            file=sys.stderr,
        )
        return 1

    print("check_bad_alloc_contract.py: OK -- every derived member is wrapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
