"""Shared C++ source-scanning primitives for the test-suite-split structural
checks (T-1574): comment/string/char-literal blanking, brace matching, and
balanced-brace body extraction. Used by both check_test_registration_complete.py
(§7.2, static SSLM_TEST( count vs runtime --count-tests, plus the
name+order+body-hash diff, §7.1) and check_test_definitions_registered.py
(§7.2, D-SLM592, the definitions-vs-registrations diff) so the two checks scan
source text with one shared, tested implementation rather than two
independently-drifting copies.

Not a test file itself; tests/ci/test__source_scan.py exercises it directly.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")


def checked_source_files() -> list[str]:
    """Every C++ translation unit the test-suite split's structural checks
    cover: tests/test_main.cpp (shrinks stage by stage but always present
    through Stage 13) plus every tests/areas/*.cpp extracted so far. Sorted
    for a deterministic scan order. This is the population
    check_test_definitions_registered.py's own "test-shaped definition" scan
    is scoped to (§3) -- test content only, never tests/support/."""
    files = [os.path.join(TESTS_DIR, "test_main.cpp")]
    files += sorted(glob.glob(os.path.join(TESTS_DIR, "areas", "*.cpp")))
    return [f for f in files if os.path.isfile(f)]


def support_source_files() -> list[str]:
    """Every tests/support/*.cpp and *.h file -- the shared, non-test
    infrastructure a check_site_hashes.txt symbol may have relocated into
    (TwoLayerFixture is header-only and inline; ChainTraceSinkHookFn and
    RunsCrashProbeAndCrashes are .cpp-defined). Not part of
    checked_source_files(): this population is test-CONTENT-adjacent
    infrastructure, never scanned for test-shaped definitions."""
    files = sorted(glob.glob(os.path.join(TESTS_DIR, "support", "*.cpp")))
    files += sorted(glob.glob(os.path.join(TESTS_DIR, "support", "*.h")))
    return files


def mask_comments_and_literals(text: str) -> str:
    """Returns `text` with every `//` line comment, `/* */` block comment, and
    the contents of every string/char literal replaced by spaces (newlines
    preserved), so brace matching and identifier regexes never see a brace or
    an identifier-shaped token that lives inside a comment or a literal. Same
    length as `text`; every offset in the input maps 1:1 to the output."""
    n = len(text)
    out = list(text)
    i = 0
    while i < n:
        c = text[i]
        if text[i:i + 2] == "//":
            j = i
            while j < n and text[j] != "\n":
                out[j] = " "
                j += 1
            i = j
            continue
        if text[i:i + 2] == "/*":
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            for k in range(i, end):
                if text[k] != "\n":
                    out[k] = " "
            i = end
            continue
        if c == '"':
            j = i + 1
            out[i] = " "
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    out[j] = " "
                    out[j + 1] = " "
                    j += 2
                    continue
                if text[j] != "\n":
                    out[j] = " "
                j += 1
            if j < n:
                out[j] = " "
                j += 1
            i = j
            continue
        if c == "'":
            j = i + 1
            out[i] = " "
            while j < n and text[j] != "'":
                if text[j] == "\\" and j + 1 < n:
                    out[j] = " "
                    out[j + 1] = " "
                    j += 2
                    continue
                if text[j] != "\n":
                    out[j] = " "
                j += 1
            if j < n:
                out[j] = " "
                j += 1
            i = j
            continue
        i += 1
    return "".join(out)


def match_braces(masked_text: str) -> dict[int, int]:
    """Maps every `{` offset in `masked_text` to its matching `}` offset.
    `masked_text` must already have comments/literals blanked (see
    `mask_comments_and_literals`) so braces inside them are never counted."""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for idx, ch in enumerate(masked_text):
        if ch == "{":
            stack.append(idx)
        elif ch == "}":
            if stack:
                pairs[stack.pop()] = idx
    return pairs


def line_of(text: str) -> list[int]:
    """`result[offset]` is the 1-indexed source line containing `offset`,
    for every offset in `[0, len(text)]` (the length-th entry covers EOF)."""
    result = [1] * (len(text) + 1)
    line = 1
    for idx, ch in enumerate(text):
        result[idx] = line
        if ch == "\n":
            line += 1
    result[len(text)] = line
    return result


_FUNC_HDR_RE = re.compile(
    r"([A-Za-z_~][A-Za-z0-9_:<>,\*&\s]*?)\b([A-Za-z_~][A-Za-z0-9_]*)\s*"
    r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*(const)?\s*(noexcept)?\s*$",
    re.S,
)
_CTOR_HDR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*$", re.S)
_STRUCT_HDR_RE = re.compile(r"\b(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:[:][^{]*)?$", re.S)
_SKIP_FIRST_WORDS = {
    "if", "for", "while", "switch", "else", "do", "try", "catch", "struct", "class",
    "namespace", "enum", "union", "extern", "using", "return",
}
_NIADIC_VOID_RE = re.compile(r"^(?:static\s+)?void\s+[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)\s*$", re.S)


def find_function_definitions(text: str) -> list[dict]:
    """Every function-like definition in `text` -- at file scope, directly
    inside a namespace (transparent, contributes no qualifier), or as an
    inline member function directly inside a struct/class body (qualified
    `Struct::Method`) -- as a list of dicts with keys `name`, `qualified`,
    `open`, `close` (char offsets of the `{`/`}` pair), `header` (the raw
    header text immediately preceding the `{`), and `is_niladic_void` (True
    iff `header` is exactly `[static] void Name()`, matching the shape
    tests/ci/check_test_definitions_registered.py's own D-SLM592 definition of
    "test-shaped" requires).

    Skips anything nested inside another function's own body (a lambda, a
    local block) by rejecting any candidate whose ancestor chain contains
    another function-header candidate.
    """
    masked_text = mask_comments_and_literals(text)
    n = len(masked_text)

    stack: list[int] = []
    depth_at_open: dict[int, int] = {}
    open_ancestors: dict[int, tuple[int, ...]] = {}
    pairs: dict[int, int] = {}
    path_stack: list[int] = []
    depth = 0
    for idx, ch in enumerate(masked_text):
        if ch == "{":
            depth_at_open[idx] = depth
            open_ancestors[idx] = tuple(path_stack)
            path_stack.append(idx)
            stack.append(idx)
            depth += 1
        elif ch == "}":
            if stack:
                o = stack.pop()
                pairs[o] = idx
                path_stack.pop()
                depth -= 1

    def header_text(open_idx: int, back_limit: int = 6000) -> str:
        start = max(0, open_idx - back_limit)
        seg = masked_text[start:open_idx]
        candidates = []
        for ch in (";", "}", "{"):
            p = seg.rfind(ch)
            if p != -1:
                candidates.append(p)
        candidates.sort(reverse=True)
        for p in candidates:
            rest = seg[p + 1:].lstrip()
            if rest.startswith("while"):
                continue
            return seg[p + 1:].strip()
        return seg.strip()

    func_hdr_cache: dict[int, str | None] = {}
    struct_hdr_cache: dict[int, str | None] = {}

    def is_struct_header(open_idx: int) -> str | None:
        if open_idx in struct_hdr_cache:
            return struct_hdr_cache[open_idx]
        hdr = header_text(open_idx)
        m = _STRUCT_HDR_RE.search(hdr) if hdr else None
        res = m.group(1) if m else None
        struct_hdr_cache[open_idx] = res
        return res

    def is_func_header(open_idx: int) -> str | None:
        if open_idx in func_hdr_cache:
            return func_hdr_cache[open_idx]
        hdr = header_text(open_idx)
        res = None
        if hdr and "(" in hdr:
            fw = re.match(r"^\s*([A-Za-z_]+)", hdr)
            if not (fw and fw.group(1) in _SKIP_FIRST_WORDS):
                m = _FUNC_HDR_RE.search(hdr)
                if m and m.group(2) not in _SKIP_FIRST_WORDS:
                    res = m.group(2)
                else:
                    cm = _CTOR_HDR_RE.match(hdr)
                    if cm:
                        ancestors = open_ancestors.get(open_idx, ())
                        if ancestors:
                            innermost_struct = is_struct_header(ancestors[-1])
                            if innermost_struct and cm.group(1) == innermost_struct:
                                res = cm.group(1)
        func_hdr_cache[open_idx] = res
        return res

    results = []
    for o in depth_at_open:
        name = is_func_header(o)
        if not name:
            continue
        ancestors = open_ancestors[o]
        if any(is_func_header(a) for a in ancestors):
            continue  # nested inside another function's own body (lambda/local block)
        quals = [is_struct_header(a) for a in ancestors]
        quals = [q for q in quals if q]
        qualified = "::".join(quals + [name]) if quals else name
        close = pairs[o]
        header = header_text(o)
        results.append({
            "name": name,
            "qualified": qualified,
            "open": o,
            "close": close,
            "header": header,
            "is_niladic_void": bool(_NIADIC_VOID_RE.match(header)),
        })
    return results


def normalized_hash(body_text: str) -> str:
    """SHA-256 of `body_text` with every run of whitespace collapsed to one
    space and leading/trailing whitespace stripped, truncated to 16 hex
    chars -- the same normalization tests/support/test_order.txt and
    tests/support/check_site_hashes.txt were generated with."""
    normalized = re.sub(r"\s+", " ", body_text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
