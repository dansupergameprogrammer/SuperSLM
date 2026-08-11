"""T-1899 (Curie, red suite for T-1894) -- structural-absence checker for
T-1822 design Sec12's "-60 load floor" cell, mechanism (2) of 2:

  "a tree-wide source scan of src/ for the retired symbol name and for any
  conditional wrapping the dynamic-gate call sites, asserted in the build's
  code review, PLUS the T-1898 probe re-run" (design Sec31.2.2's own text).

WHAT THIS CHECKS, AND WHY IT IS TEXT-SCAN, NOT AST. Two independent
population/assertion pairs, on the same "text scan, no AST" discipline this
tree's own check_no_forward_leaf_calls.py already establishes (that module's
own docstring: mirrors tests/test_check_no_pow_operator.py's convention):

  1. RETIRED SYMBOL ABSENCE. No file under `src/` may declare a symbol named
     `kOptionGFusedKvLandingExponentMin` -- T-1891's own spike-only pre-filter
     constant, explicitly retired (not repaired) by the design's own round-3
     fix (D-SLM2384/D-SLM2385/D-SLM2388): "no new production symbol
     comparable to the spike's own kOptionGFusedKvLandingExponentMin exists."
     A build that reintroduces a fused-site-specific floor constant under
     ANY name containing this exact pattern is caught.

  2. NO CONDITIONAL AHEAD OF THE DYNAMIC GATE, RE-ANCHORED 2026-08-11 round 4
     (T-1901 Critical 2/D-SLM2417, D-SLM2425). The prior version of this
     mechanism anchored on the literal string `out_magnitude_exceeded_int64`,
     with a fixed six-line backward look-back for an exponent-shaped `if`.
     Executed against the real built tree, that anchor matched only a
     COMMENT (the real call site passes `&exceeded0`/`&exceeded1` and
     refuses on `if (exceeded0 || exceeded1)` -- the literal string never
     appears in the actual conditional or the actual `LandingRescale` call),
     fifteen lines from the nearest real token, and the backward-only window
     could not reach code written after it regardless. An injected
     exponent-conditioned skip AND an outright deletion of the refusal both
     reported clean (T-1901 Critical 2). The corrected anchor: the two
     tokens the real fused call site actually contains --
       (a) `return SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain`/
           `OptionGWideRopeMagnitudeOutOfDomain` (the two refusal returns), and
       (b) a `LandingRescale(...)` call passing `&exceeded0`/`&exceeded1` as
           its own seventh (`out_magnitude_exceeded_int64`) argument
     -- scanning the ENCLOSING BLOCK each anchor sits inside (every
     currently-open brace at that point in the file, walked from innermost to
     outermost via a running brace-depth stack), never a fixed line count.
     Two, not one, assertions now compose this mechanism:
       (i) PRESENCE -- each of the two SslmForwardStatus tokens above must
           appear, at least once, as an actual `return SslmForwardStatus::...`
           statement somewhere in the population (never inside a comment or a
           string literal, both stripped before matching) -- catches outright
           deletion, which a skip-condition scan alone cannot, since removing
           the refusal trivially satisfies "no skip condition found".
       (ii) WRAPPING -- for every occurrence found by (i), or by the
           `LandingRescale`-seventh-argument anchor, none of its enclosing
           blocks' own opening `if` (found via the brace-depth stack, not a
           backward window) may be exponent-shaped (`e_t`, `kv_landing_e_t`,
           `ExponentMin`, or a numeric literal in [-128,0)) -- catches a
           conditional wrapping the refusal at ANY nesting depth, not only
           within a fixed number of lines above it.

MUTATION VITALITY (StandardsDocument.md Sec4: a new structure is validated
against an independently-found population before it is trusted; a check
shown able to fail on an INJECTED fault, not merely to pass on unchanged
input) -- RE-DERIVED 2026-08-11 round 4 against REAL-FILE mutations, never a
synthetic fixture (T-1901 Critical 2's own finding: the prior vitality
fixture carried a marker line, `(void) out_magnitude_exceeded_int64;`, that
exists nowhere in real code -- "a population of one chosen by the rule's
author", the exact failure StandardsDocument.md Sec4 names). This module's
own test (tests/ci/test_check_no_optionG_fused_site_skip.py) now copies the
REAL `src/forward/forward_sites.cpp` to a scratch location, applies the two
mutations T-1901 executed by hand (an injected exponent-conditioned `if`
wrapping the refusal; the refusal deleted outright), and requires both
copies to go RED -- the real, unmutated tree must report clean.

Test-design record: Claude/Curie/t1899-optionG-red-suite-2026-08-11.md
(mechanism 1, retired-symbol scan, T-1899's own original authoring).
Re-anchor record: Claude/Poirot/407981b-t1900-optionG-build.md Sec6;
Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md Sec12
"-60 load-time floor", mechanism (2), round 4 text.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_RETIRED_SYMBOL_PATTERN = re.compile(r"\bkOptionGFusedKvLandingExponentMin\b")

# T-1901 Critical 2 / D-SLM2425 -- the real call-site tokens, not a literal
# that only appears in a comment. Group 1 captures which status was
# returned, for readable reporting.
_PRIMARY_ANCHOR_PATTERN = re.compile(
    r"\breturn\s+SslmForwardStatus::"
    r"(OptionGFusedLandingExponentOutOfDomain|OptionGWideRopeMagnitudeOutOfDomain)\b"
)
_PRIMARY_ANCHOR_TOKENS = (
    "OptionGFusedLandingExponentOutOfDomain",
    "OptionGWideRopeMagnitudeOutOfDomain",
)
# The alternative anchor the design names: a `LandingRescale(...)` call
# passing its own seventh argument, `&exceeded0`/`&exceeded1`
# (`out_magnitude_exceeded_int64`). Detected on the call's own closing line
# (this codebase's multi-line call style always closes with the pointer
# argument immediately before the final paren), so no multi-line statement
# joining is needed.
_SECONDARY_ANCHOR_PATTERN = re.compile(r"&exceeded[01]\s*\)")

_IF_SHAPED = re.compile(r"^\s*(?:}\s*)?else\s+if\b|^\s*if\s*\(|^\s*if\b")
_EXPONENT_SHAPED = re.compile(
    r"\be_t\b|\bkv_landing_e_t\b|ExponentMin|(?<![\w.])-(?:1[01]\d|12[0-7]|[1-9]?\d)(?![\w.])"
)


@dataclass(frozen=True)
class RetiredSymbolHit:
    path: str
    line_no: int
    line: str


@dataclass(frozen=True)
class SkipConditionHit:
    path: str
    if_line_no: int
    if_line: str
    gate_line_no: int
    gate_line: str


@dataclass(frozen=True)
class MissingAnchorHit:
    """Mechanism 2, assertion (i): one of the two required refusal
    statements was not found anywhere in the population -- the refusal was
    deleted outright (T-1901 Critical 2's second executed mutation), or never
    built at all."""
    token: str


def find_retired_symbol_uses(path: str) -> List[RetiredSymbolHit]:
    hits: List[RetiredSymbolHit] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if _RETIRED_SYMBOL_PATTERN.search(line):
                hits.append(RetiredSymbolHit(path, i, line.rstrip("\n")))
    return hits


def _strip_comments_and_strings(text: str) -> str:
    """Replaces `//` comments, `/* */` comments, and the CONTENTS of
    string/char literals with spaces -- preserving every line break exactly,
    so brace-counting and anchor-token matching never fire on a comment or a
    string literal (this file's own header comments mention the anchor
    tokens by name; a real call site's own explanatory prose does too), and
    reported line numbers stay aligned with the original file."""
    out: List[str] = []
    i, n = 0, len(text)
    state = "code"  # code | line_comment | block_comment | string | char
    while i < n:
        c = text[i]
        if state == "code":
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                out.append("  ")
                i += 2
                state = "line_comment"
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                out.append("  ")
                i += 2
                state = "block_comment"
                continue
            if c == '"':
                out.append('"')
                i += 1
                state = "string"
                continue
            if c == "'":
                out.append("'")
                i += 1
                state = "char"
                continue
            out.append(c)
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                out.append("\n")
                state = "code"
                i += 1
                continue
            out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                out.append("  ")
                i += 2
                state = "code"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if state == "string":
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                out.append('"')
                i += 1
                state = "code"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if state == "char":
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == "'":
                out.append("'")
                i += 1
                state = "code"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
    return "".join(out)


def _enclosing_stacks(cleaned_text: str) -> Dict[int, List[Tuple[int, str]]]:
    """For each 1-indexed line number, the stack of (header_line_no,
    header_line_cleaned_text) for every '{' still open AT THE START of that
    line -- innermost last. Built by a single left-to-right scan of the
    (already comment/string-stripped) text, so a brace inside a comment or a
    string literal was already removed and cannot perturb the count."""
    lines = cleaned_text.split("\n")
    stack: List[Tuple[int, str]] = []
    stacks: Dict[int, List[Tuple[int, str]]] = {}
    for line_no, line in enumerate(lines, start=1):
        stacks[line_no] = list(stack)
        for ch in line:
            if ch == "{":
                stack.append((line_no, line))
            elif ch == "}":
                if stack:
                    stack.pop()
    return stacks


def find_dynamic_gate_skip_conditions(path: str) -> List[SkipConditionHit]:
    """Mechanism 2, assertion (ii): every PRIMARY or SECONDARY anchor
    occurrence in `path`, checked against its own real enclosing-block stack
    -- not a fixed backward window -- for an exponent-shaped `if` ancestor."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    cleaned = _strip_comments_and_strings(raw)
    raw_lines = raw.split("\n")
    cleaned_lines = cleaned.split("\n")
    stacks = _enclosing_stacks(cleaned)

    hits: List[SkipConditionHit] = []
    for idx, cleaned_line in enumerate(cleaned_lines):
        line_no = idx + 1
        is_anchor = bool(_PRIMARY_ANCHOR_PATTERN.search(cleaned_line)) or bool(
            _SECONDARY_ANCHOR_PATTERN.search(cleaned_line))
        if not is_anchor:
            continue
        # Innermost first: reversed() over the stack (built outermost-first).
        for header_line_no, header_cleaned in reversed(stacks.get(line_no, [])):
            if _IF_SHAPED.search(header_cleaned) and _EXPONENT_SHAPED.search(header_cleaned):
                hits.append(SkipConditionHit(
                    path=path,
                    if_line_no=header_line_no,
                    if_line=raw_lines[header_line_no - 1] if header_line_no - 1 < len(raw_lines) else header_cleaned,
                    gate_line_no=line_no,
                    gate_line=raw_lines[line_no - 1] if line_no - 1 < len(raw_lines) else cleaned_line,
                ))
    return hits


def find_missing_anchors(paths: Sequence[str]) -> List[MissingAnchorHit]:
    """Mechanism 2, assertion (i): each of the two required refusal
    statements (`_PRIMARY_ANCHOR_TOKENS`) must be found, as a real
    `return SslmForwardStatus::...` statement (comments/strings stripped),
    somewhere across the WHOLE population -- not merely "no skip condition
    found", which an outright-deleted refusal would trivially satisfy."""
    counts = {token: 0 for token in _PRIMARY_ANCHOR_TOKENS}
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            cleaned = _strip_comments_and_strings(f.read())
        for m in _PRIMARY_ANCHOR_PATTERN.finditer(cleaned):
            counts[m.group(1)] += 1
    return [MissingAnchorHit(token) for token, count in counts.items() if count == 0]


def _default_population(repo_root: str) -> List[str]:
    patterns = [
        os.path.join(repo_root, "src", "**", "*.cpp"),
        os.path.join(repo_root, "src", "**", "*.h"),
        os.path.join(repo_root, "include", "**", "*.h"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def check(
    repo_root: str, population: Sequence[str] = None
) -> Tuple[List[RetiredSymbolHit], List[SkipConditionHit], List[MissingAnchorHit]]:
    files = list(population) if population is not None else _default_population(repo_root)
    retired_hits: List[RetiredSymbolHit] = []
    skip_hits: List[SkipConditionHit] = []
    for path in files:
        retired_hits.extend(find_retired_symbol_uses(path))
        skip_hits.extend(find_dynamic_gate_skip_conditions(path))
    missing_hits = find_missing_anchors(files)
    return retired_hits, skip_hits, missing_hits


def main(repo_root: str, population: Sequence[str] = None) -> int:
    retired_hits, skip_hits, missing_hits = check(repo_root, population)
    if not retired_hits and not skip_hits and not missing_hits:
        return 0
    for h in retired_hits:
        print(f"{h.path}:{h.line_no}: retired symbol kOptionGFusedKvLandingExponentMin found: "
              f"{h.line}")
    for h in skip_hits:
        print(f"{h.path}:{h.gate_line_no}: dynamic-gate anchor preceded (enclosing line "
              f"{h.if_line_no}) by an exponent-shaped conditional: {h.if_line!r} -- possible "
              f"resurrected early-exit")
    for h in missing_hits:
        print(f"MISSING: no 'return SslmForwardStatus::{h.token}' statement found anywhere in "
              f"the population -- the dynamic gate's own refusal may have been deleted")
    return 1


if __name__ == "__main__":
    import sys

    _repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                os.pardir, os.pardir))
    sys.exit(main(_repo_root))
