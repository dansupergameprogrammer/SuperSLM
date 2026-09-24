"""Commissioning suite for tests/ci/check_no_library_stdout_write.py, the SuperSLM v1.7.1 CI tripwire
that refuses shipped library source naming a stdout-writing identifier.

The scan's promise, which is the whole specification this module tests:

    The scan walks `src/` and `include/` (files ending .c .cpp .h .hpp .inc .def .inl .ipp .hlsl
    .hlsli). It refuses a file that names, as a whole identifier outside a comment or a
    string/character literal, any of: stdout, STD_OUTPUT_HANDLE, cout, wcout, the printf family
    (printf, wprintf, vprintf, vwprintf, each also with _s), the _printf_l/_printf_p family (any
    identifier beginning `_` and containing `printf_l` or `printf_p`), puts, _putws, putchar,
    putwchar, WriteConsole/WriteConsoleA/WriteConsoleW, _write, write, __acrt_iob_func, fdopen,
    _fdopen, GetStdHandle, WriteFile; or that uses the token-paste operator `##` outside a comment
    or literal. It is a tripwire for those spellings, not a proof. Comments and literals are
    recognised exactly as C++ does, including C++14 digit separators (1'000 is not a character
    literal) and raw string literals (R"delim( ... )delim", with u8R/uR/UR/LR prefixes), so no
    literal or separator can hide the text after it. `stderr` and fprintf(stderr, ...) are
    allowed. Line numbers in a finding are exact.

Scanner interface (the only part of the scanner this module relies on):
    scan_for_stdout_writes(root_dirs: list[str]) -> list[str]   # "path:line: reason" strings
    main(argv: list[str]) -> int                                 # nonzero iff any hit

Sections, in order:

1. Tree-level checks. The real src/+include/ of the commit this module runs from must scan clean;
   the real src/+include/ of v1.7.0 (f43ab15), which still carries the known std::wprintf site in
   src/gpu/d3d12_harness.h, must be refused with that file named.

2. The commissioning population (TE-411). Written from the promise above by the test author, blind
   to the scanner's implementation and to every earlier population, so the scan's banned list is
   not fitted to it. Each case plants exactly one file into an otherwise empty src/+include/ tree
   and states the promise clause it tests.
     - must-reject: the scan must report the planted file at exactly the marked lines -- no line
       missing, no extra line -- and main() must return nonzero.
     - must-accept: the scan must report nothing and main() must return 0.
   Source lines prefixed `@@` in a case are the lines the scan must report; the prefix is removed
   before the file is written.

3. The TE-407 regression set: the reviewer's 15-form population from the round before this one.
   Kept as regression cases only. It is not the commissioning population, because the scanner's
   banned list was revised against it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# f43ab15: SuperSLM v1.7.0's tip. Its src/gpu/d3d12_harness.h still writes to stdout with
# std::wprintf -- the site v1.7.1 removes.
_MUST_REJECT_REF = "f43ab15"

sys.path.insert(0, _THIS_DIR)
try:
    import check_no_library_stdout_write as scanner  # type: ignore[import-not-found]

    _IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover -- exercised by the collection itself
    scanner = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _require_scanner() -> None:
    if scanner is None:
        pytest.skip(f"scanner module not importable: {_IMPORT_ERROR!r}")


# =============================================================================================
# 1. Tree-level checks
# =============================================================================================


def _git_archive_tree(ref: str, dest_dir: str) -> None:
    """Materializes `ref`'s src/ and include/ into `dest_dir` via `git archive` of that commit --
    never a copy of the working tree, which would pick up whatever is checked out."""
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, "archive.tar")
    subprocess.run(
        ["git", "archive", ref, "-o", archive_path, "--", "src", "include"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    # Plain `-xf`: python.exe resolves `tar` to Windows' System32 bsdtar, which refuses GNU tar's
    # --force-local. The cwd is a real Windows path, so no drive letter is misparsed as a host.
    subprocess.run(["tar", "-xf", archive_path], cwd=dest_dir, check=True, capture_output=True)
    os.remove(archive_path)


def _resolve_head() -> str:
    """The commit this module runs from. Never hardcoded: the tree CI checks out is the tree the
    scan must accept."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def test_scanner_module_exists():
    """Precondition for everything below, stated as one readable failure instead of every case
    skipping."""
    assert scanner is not None, (
        "tests/ci/check_no_library_stdout_write.py is missing or fails to import: "
        f"{_IMPORT_ERROR!r}"
    )


def test_must_reject_the_v1_7_0_tree_and_names_the_site():
    """f43ab15's real src/+include/ must be refused, and the refusal must name
    src/gpu/d3d12_harness.h -- a failure that does not point at the real site is not a
    diagnostic."""
    _require_scanner()
    with tempfile.TemporaryDirectory(prefix="te399_ci_scan_reject_") as tmp:
        _git_archive_tree(_MUST_REJECT_REF, tmp)
        roots = [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        hits = scanner.scan_for_stdout_writes(roots)
        assert hits, f"zero hits against {_MUST_REJECT_REF} (v1.7.0), which writes to stdout"
        assert any("d3d12_harness.h" in h for h in hits), (
            f"{len(hits)} hit(s) against {_MUST_REJECT_REF}, none naming d3d12_harness.h: {hits!r}"
        )
        code = scanner.main(roots)
        assert code != 0, f"main() returned {code} (success) against {_MUST_REJECT_REF}"


def test_must_accept_the_current_tree():
    """The real src/+include/ of the commit this module runs from must scan clean, over every
    file the scan walks."""
    _require_scanner()
    ref = _resolve_head()
    with tempfile.TemporaryDirectory(prefix="te399_ci_scan_accept_") as tmp:
        _git_archive_tree(ref, tmp)
        roots = [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        hits = scanner.scan_for_stdout_writes(roots)
        assert hits == [], f"{len(hits)} hit(s) against the current tree ({ref}): {hits!r}"
        code = scanner.main(roots)
        assert code == 0, f"main() returned {code} (failure) against the current tree {ref}"


# =============================================================================================
# 2. The commissioning population (TE-411)
# =============================================================================================

# Promise clauses, quoted or closely paraphrased. Every case names the one it tests.
C_IDENT = "refuses a listed identifier named as a whole identifier outside comments and literals"
C_PREFIX = (
    "refuses any identifier beginning `_` and containing `printf_l` or `printf_p` "
    "(the _printf_l/_printf_p family)"
)
C_PASTE = "refuses the token-paste operator `##` outside a comment or literal"
C_WHOLE = "whole identifier only: a longer identifier containing a listed word is not refused"
C_COMMENT = "comments are recognised exactly as C++ does"
C_LITERAL = "string and character literals are recognised exactly as C++ does"
C_SEPARATOR = "C++14 digit separators are not character literals and hide nothing after them"
C_RAW = "raw string literals, with u8R/uR/UR/LR prefixes, are recognised exactly as C++ does"
C_STDERR = "`stderr` and fprintf(stderr, ...) are allowed"
C_WALK = "the scan walks src/ and include/, files ending in the ten listed extensions"
C_LINES = "line numbers in a finding are exact"
C_SCOPE = "a listed identifier is refused wherever it appears in code, whatever the scope"

_MARK = "@@"


@dataclass(frozen=True)
class Case:
    id: str
    clause: str
    path: str  # relative to the tree root; always under src/ or include/ unless testing the walk
    source: str  # lines prefixed with _MARK are the lines the scan must report
    newline: str = "\n"
    final_newline: bool = True

    def render(self) -> tuple[str, frozenset[int]]:
        text = self.source[1:] if self.source.startswith("\n") else self.source
        if text.endswith("\n"):
            text = text[:-1]
        lines: list[str] = []
        expected: set[int] = set()
        for number, line in enumerate(text.split("\n"), start=1):
            if line.startswith(_MARK):
                expected.add(number)
                line = line[len(_MARK):]
            lines.append(line)
        body = self.newline.join(lines) + (self.newline if self.final_newline else "")
        return body, frozenset(expected)


# Every identifier the promise lists by name, with one realistic statement naming it. The
# statement is placed inside a function body at a line the case marks.
_LISTED_STATEMENTS: list[tuple[str, str]] = [
    ("stdout", "std::fflush(stdout);"),
    ("STD_OUTPUT_HANDLE", "const DWORD which = STD_OUTPUT_HANDLE;"),
    ("cout", "std::cout << 1;"),
    ("wcout", 'std::wcout << L"x";'),
    ("printf", 'std::printf("x");'),
    ("wprintf", 'wprintf(L"x");'),
    ("vprintf", "vprintf(fmt, ap);"),
    ("vwprintf", "vwprintf(wfmt, ap);"),
    ("printf_s", 'printf_s("x");'),
    ("wprintf_s", 'wprintf_s(L"x");'),
    ("vprintf_s", "vprintf_s(fmt, ap);"),
    ("vwprintf_s", "vwprintf_s(wfmt, ap);"),
    ("puts", 'std::puts("x");'),
    ("_putws", '_putws(L"x");'),
    ("putchar", "std::putchar(120);"),
    ("putwchar", "putwchar(L'x');"),
    ("WriteConsole", "WriteConsole(h, buf, n, &written, nullptr);"),
    ("WriteConsoleA", "WriteConsoleA(h, buf, n, &written, nullptr);"),
    ("WriteConsoleW", "WriteConsoleW(h, wbuf, n, &written, nullptr);"),
    ("_write", "_write(1, buf, n);"),
    ("write", "write(1, buf, n);"),
    ("__acrt_iob_func", "FILE* f = __acrt_iob_func(1);"),
    ("fdopen", 'FILE* f = fdopen(1, "w");'),
    ("_fdopen", 'FILE* f = _fdopen(1, "w");'),
    ("GetStdHandle", "HANDLE h2 = GetStdHandle(STD_ERROR_HANDLE);"),
    ("WriteFile", "WriteFile(h, buf, n, &written, nullptr);"),
]

# Members of the _printf_l/_printf_p family, by the promise's own rule: begins with `_`, contains
# `printf_l` or `printf_p`. `_fprintf_l(stderr, ...)` is in the family by that rule even though it
# targets stderr; `_printf_list` and `__printf_p_impl` are in it by the rule's literal text.
_PREFIX_STATEMENTS: list[tuple[str, str]] = [
    ("_printf_l", '_printf_l("x", loc);'),
    ("_printf_p", '_printf_p("x");'),
    ("_wprintf_p_l", '_wprintf_p_l(L"x", loc);'),
    ("_vprintf_l", "_vprintf_l(fmt, loc, ap);"),
    ("_vwprintf_p", "_vwprintf_p(wfmt, ap);"),
    ("_fprintf_l", '_fprintf_l(stderr, "x", loc);'),
    ("_printf_list", "int count = _printf_list;"),
    ("__printf_p_impl", '__printf_p_impl("x");'),
]


def _in_function(statement: str) -> str:
    return (
        "\n#include <cstdio>\n\n"
        "void te411_case(const char* fmt, va_list ap, const void* buf, unsigned n) {\n"
        f"{_MARK}    {statement}\n"
        "}\n"
    )


def _slug(word: str) -> str:
    return "hashhash" if word == "##" else word.strip("_").lower() or "u"


_REJECT: list[Case] = []
_ACCEPT: list[Case] = []

for _word, _stmt in _LISTED_STATEMENTS:
    _REJECT.append(
        Case(
            f"reject-ident-{_word}",
            C_IDENT,
            f"src/te411_rj_{_slug(_word)}_{len(_word)}.cpp",
            _in_function(_stmt),
        )
    )
for _word, _stmt in _PREFIX_STATEMENTS:
    _REJECT.append(
        Case(
            f"reject-prefix-{_word}",
            C_PREFIX,
            f"src/te411_rp_{_slug(_word)}_{len(_word)}.cpp",
            _in_function(_stmt),
        )
    )

# ---- must-reject: `##` -----------------------------------------------------------------------
_REJECT += [
    Case("reject-paste-adjacent", C_PASTE, "src/te411_paste1.h", """
#pragma once
@@#define TE411_CAT(a, b) a##b
"""),
    Case("reject-paste-spaced", C_PASTE, "src/te411_paste2.h", """
#pragma once
@@#define TE411_CAT2(a, b) a ## b
"""),
    Case("reject-paste-on-spliced-macro-line", C_PASTE, "src/te411_paste3.h", r"""
#pragma once
#define TE411_CAT3(a, b) \
@@    a##b
"""),
]

# ---- must-reject: scope and statement shapes -------------------------------------------------
_REJECT += [
    Case("reject-namespace-scope-stdout", C_SCOPE, "src/te411_ns1.cpp", """
#include <cstdio>

@@static FILE* const g_te411_out = stdout;
"""),
    Case("reject-namespace-scope-printf-address", C_SCOPE, "src/te411_ns2.cpp", """
#include <cstdio>

namespace te411 {
@@int (*g_sink)(const char*, ...) = &printf;
}
"""),
    Case("reject-macro-body", C_SCOPE, "src/te411_macro1.h", """
#pragma once
#include <cstdio>
@@#define TE411_LOG(msg) std::puts(msg)
"""),
    Case("reject-macro-body-continued-line", C_SCOPE, "src/te411_macro2.h", r"""
#pragma once
#include <cstdio>
#define TE411_LOG2(msg) \
    do { \
@@        std::puts(msg); \
    } while (0)
"""),
    Case("reject-after-using-namespace-std", C_SCOPE, "src/te411_using1.cpp", """
#include <iostream>
using namespace std;

void te411_case() {
@@    cout << 1;
}
"""),
    Case("reject-using-declaration-and-call", C_SCOPE, "src/te411_using2.cpp", """
#include <cstdio>
@@using std::printf;

void te411_case() {
@@    printf("x");
}
"""),
    Case("reject-two-statements-stderr-then-stdout", C_STDERR + "; " + C_IDENT, "src/te411_two1.cpp", """
#include <cstdio>

void te411_case() {
@@    std::fprintf(stderr, "a"); std::printf("b");
}
"""),
    Case("reject-two-statements-declaration-then-cout", C_IDENT, "src/te411_two2.cpp", """
#include <iostream>

void te411_case() {
@@    int x = 0; std::cout << x;
}
"""),
    Case("reject-argument-on-following-line", C_LINES, "src/te411_nextline1.cpp", """
#include <cstdio>

void te411_case() {
    std::fprintf(
@@        stdout, "x %d", 1);
}
"""),
    Case("reject-call-with-arguments-on-following-line", C_LINES, "src/te411_nextline2.cpp", """
#include <cstdio>

void te411_case() {
@@    std::printf(
        "x %d",
        1);
}
"""),
    Case("reject-qualifier-split-across-lines", C_LINES, "src/te411_nextline3.cpp", """
#include <iostream>

void te411_case() {
    std::
@@        cout << 1;
}
"""),
    Case("reject-member-write", C_IDENT, "src/te411_member.cpp", """
#include <ostream>

void te411_case(std::ostream& out, const char* buf, long n) {
@@    out.write(buf, n);
}
"""),
    Case("reject-inside-if-0", C_IDENT + " (only comments and literals are exempt)", "src/te411_if0.cpp", """
#include <cstdio>

#if 0
@@void te411_disabled() { std::printf("x"); }
#endif
"""),
    Case("reject-first-line-first-column", C_LINES, "src/te411_firstcol.cpp", """
@@putchar('x');
"""),
    Case("reject-last-line-without-newline", C_LINES, "src/te411_eof.cpp", """
#include <cstdio>

void te411_case() {}
@@void te411_last() { std::puts("x"); }
""", final_newline=False),
    Case("reject-crlf-line-endings", C_LINES, "src/te411_crlf.cpp", """
#include <cstdio>

void te411_case() {
@@    std::printf("x");
}
""", newline="\r\n"),
]

# ---- must-reject: comments recognised exactly as C++ does ------------------------------------
_REJECT += [
    Case("reject-after-nested-looking-block-comment", C_COMMENT, "src/te411_cm1.cpp", """
#include <cstdio>

void te411_case() {
@@    /* outer /* inner */ std::printf("x");
}
"""),
    Case("reject-block-opener-inside-line-comment", C_COMMENT, "src/te411_cm2.cpp", """
#include <cstdio>

void te411_case() {
    // a line comment does not open a block: /*
@@    std::printf("x");
}
"""),
    Case("reject-after-multiline-block-comment-naming-words", C_COMMENT + "; " + C_LINES, "src/te411_cm3.cpp", """
#include <cstdio>

void te411_case() {
    /*
       printf, cout, stdout and ## are all inside this comment
@@    */ std::printf("x");
}
"""),
    Case("reject-directly-after-block-comment-no-space", C_COMMENT, "src/te411_cm4.cpp", """
#include <cstdio>

void te411_case() {
@@    /*x*/std::printf("x");
}
"""),
    Case("reject-after-apostrophe-in-line-comment", C_COMMENT, "src/te411_cm5.cpp", """
#include <cstdio>

void te411_case() {
    // don't
@@    std::printf("x");
}
"""),
    Case("reject-after-apostrophe-in-block-comment", C_COMMENT, "src/te411_cm6.cpp", """
#include <cstdio>

void te411_case() {
@@    /* it's */ std::printf("x");
}
"""),
    Case("reject-after-quote-in-block-comment", C_COMMENT, "src/te411_cm7.cpp", """
#include <cstdio>

void te411_case() {
@@    /* " */ std::printf("x"); /* " */
}
"""),
    Case("reject-after-line-comment-with-mid-line-backslash", C_COMMENT, "src/te411_cm8.cpp", r"""
#include <cstdio>

void te411_case() {
    // a path like C:\temp is not a line splice
@@    std::printf("x");
}
"""),
]

# ---- must-reject: string and character literals recognised exactly as C++ does ---------------
_REJECT += [
    Case("reject-after-string-containing-line-comment-opener", C_LITERAL, "src/te411_lt1.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = "//"; std::printf("x");
}
"""),
    Case("reject-after-string-containing-block-comment-opener", C_LITERAL, "src/te411_lt2.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = "/*"; std::printf("x"); // */
}
"""),
    Case("reject-after-char-literal-double-quote", C_LITERAL, "src/te411_lt3.cpp", """
#include <cstdio>

void te411_case() {
@@    char q = '"'; std::printf("x");
}
"""),
    Case("reject-after-escaped-quote-in-string", C_LITERAL, "src/te411_lt4.cpp", r"""
#include <cstdio>

void te411_case() {
@@    const char* s = "a\"b"; std::printf("x");
}
"""),
    Case("reject-after-string-ending-in-escaped-backslash", C_LITERAL, "src/te411_lt5.cpp", r"""
#include <cstdio>

void te411_case() {
@@    const char* s = "a\\"; std::printf("x");
}
"""),
    Case("reject-after-escaped-apostrophe-char", C_LITERAL, "src/te411_lt6.cpp", r"""
#include <cstdio>

void te411_case() {
@@    char q = '\''; std::printf("x");
}
"""),
    Case("reject-after-escaped-backslash-char", C_LITERAL, "src/te411_lt7.cpp", r"""
#include <cstdio>

void te411_case() {
@@    char q = '\\'; std::printf("x");
}
"""),
    Case("reject-after-apostrophe-in-string", C_LITERAL, "src/te411_lt8.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = "it's"; std::printf("x");
}
"""),
    Case("reject-on-continuation-line-of-spliced-string", C_LITERAL + "; " + C_LINES, "src/te411_lt9.cpp", r"""
#include <cstdio>

void te411_case() {
    const char* s = "abc\
@@def"; std::printf("x");
}
"""),
    Case("reject-after-u8-char-literal-double-quote", C_LITERAL + "; " + C_SEPARATOR, "src/te411_lt10.cpp", """
#include <cstdio>

void te411_case() {
@@    auto c = u8'"'; std::printf("x");
}
"""),
    Case("reject-after-wide-char-literal-double-quote", C_LITERAL, "src/te411_lt11.cpp", """
#include <cstdio>

void te411_case() {
@@    auto c = L'"'; std::printf("x");
}
"""),
]

# ---- must-reject: digit separators -----------------------------------------------------------
_REJECT += [
    Case("reject-directly-after-digit-separated-constant", C_SEPARATOR, "src/te411_sep1.cpp", """
#include <cstdio>

void te411_case() {
@@    int n = 1'000; std::printf("x");
}
"""),
    Case("reject-after-odd-count-of-separators", C_SEPARATOR, "src/te411_sep2.cpp", """
#include <cstdio>

void te411_case() {
@@    long n = 1'2'3'4; std::puts("x");
}
"""),
    Case("reject-after-hex-separated-constant", C_SEPARATOR, "src/te411_sep3.cpp", """
#include <cstdio>

void te411_case() {
@@    unsigned n = 0xFF'FF; std::printf("x");
}
"""),
    Case("reject-after-binary-separated-constant", C_SEPARATOR, "src/te411_sep4.cpp", """
#include <cstdio>

void te411_case() {
@@    unsigned n = 0b1010'1010; std::printf("x");
}
"""),
    Case("reject-after-separated-floating-constant", C_SEPARATOR, "src/te411_sep5.cpp", """
#include <cstdio>

void te411_case() {
@@    double d = 1'000.000'1; std::printf("x");
}
"""),
    Case("reject-after-separated-constant-with-suffix", C_SEPARATOR, "src/te411_sep6.cpp", """
#include <cstdio>

void te411_case() {
@@    auto n = 1'000ull; std::printf("x");
}
"""),
    Case("reject-line-after-separated-constant", C_SEPARATOR + "; " + C_LINES, "src/te411_sep7.cpp", """
#include <cstdio>

void te411_case() {
    int n = 1'000;
@@    std::printf("x");
}
"""),
]

# ---- must-reject: raw string literals --------------------------------------------------------
_REJECT += [
    Case("reject-after-raw-string-containing-quote-and-paren", C_RAW, "src/te411_raw1.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = R"x(a")b)x"; std::printf("x");
}
"""),
    Case("reject-after-raw-string-containing-paren-quote", C_RAW, "src/te411_raw2.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = R"d()")d"; std::printf("x");
}
"""),
    Case("reject-after-raw-string-containing-line-comment-opener", C_RAW, "src/te411_raw3.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = R"(//)"; std::printf("x");
}
"""),
    Case("reject-after-empty-raw-string", C_RAW, "src/te411_raw4.cpp", """
#include <cstdio>

void te411_case() {
@@    const char* s = R"()"; std::printf("x");
}
"""),
    Case("reject-after-multiline-raw-string", C_RAW + "; " + C_LINES, "src/te411_raw5.cpp", """
#include <cstdio>

void te411_case() {
    const char* s = R"d(
printf inside the raw string, and a lone " and ) too
)d";
@@    std::printf("x");
}
"""),
    Case("reject-after-u8R-raw-string", C_RAW, "src/te411_raw6.cpp", """
#include <cstdio>

void te411_case() {
@@    auto s = u8R"(")"; std::printf("x");
}
"""),
    Case("reject-after-uR-raw-string", C_RAW, "src/te411_raw7.cpp", """
#include <cstdio>

void te411_case() {
@@    auto s = uR"(")"; std::printf("x");
}
"""),
    Case("reject-after-UR-raw-string", C_RAW, "src/te411_raw8.cpp", """
#include <cstdio>

void te411_case() {
@@    auto s = UR"(")"; std::printf("x");
}
"""),
    Case("reject-after-LR-raw-string", C_RAW, "src/te411_raw9.cpp", """
#include <cstdio>

void te411_case() {
@@    auto s = LR"(")"; std::printf("x");
}
"""),
    Case("reject-identifier-ending-in-R-is-not-a-raw-prefix", C_RAW, "src/te411_raw10.cpp", """
#include <cstdio>
#define TE411R

void te411_case() {
@@    const char* s = TE411R"("; std::printf("x"); // )"
}
"""),
]

# ---- must-reject: the walk -------------------------------------------------------------------
for _ext in ["c", "cpp", "h", "hpp", "inc", "def", "inl", "ipp", "hlsl", "hlsli"]:
    _REJECT.append(
        Case(
            f"reject-extension-{_ext}",
            C_WALK,
            f"src/te411_ext_{_ext}.{_ext}",
            '\n// scanned extension case\n@@void te411_ext() { printf("x"); }\n',
        )
    )
_REJECT += [
    Case("reject-nested-subdirectory-of-src", C_WALK, "src/te411_a/b/c/te411_deep.cpp", """
#include <cstdio>
@@void te411_deep() { std::puts("x"); }
"""),
    Case("reject-under-include", C_WALK, "include/superslm/te411_public.h", """
#pragma once
#include <cstdio>
@@inline void te411_public() { std::printf("x"); }
"""),
]

# ---- must-accept: every listed word, and `##`, inside each comment and literal form ----------
_ACCEPT_WORDS = [w for w, _ in _LISTED_STATEMENTS] + ["_printf_l", "_printf_p", "##"]
_ACCEPT_CONTEXTS: list[tuple[str, str, str]] = [
    ("line-comment", C_COMMENT, "int a = 0; // {w}"),
    ("block-comment", C_COMMENT, "int b = 0; /* {w} */ int c = 0;"),
    ("string", C_LITERAL, 'const char* s = "{w}";'),
    ("raw-string", C_RAW, 'const char* r = R"({w})";'),
    ("char", C_LITERAL, "const int c = '{w}';"),
]
for _word in _ACCEPT_WORDS:
    for _ctx, _clause, _tmpl in _ACCEPT_CONTEXTS:
        _ACCEPT.append(
            Case(
                f"accept-{_ctx}-{_word}",
                _clause,
                f"src/te411_ac_{_ctx.replace('-', '')}_{_slug(_word)}_{len(_word)}.cpp",
                "\nvoid te411_case() {\n    " + _tmpl.format(w=_word) + "\n}\n",
            )
        )

# ---- must-accept: longer identifiers containing a listed word --------------------------------
for _word in [
    "my_printf_helper", "rewrite", "stdout_guard_count", "printf_like", "sprintf", "snprintf",
    "swprintf", "vsnprintf", "fwrite", "fputs", "fputws", "overwrite", "writer", "write_all",
    "cout_count", "wcout_ready", "puts_count", "putchar_count", "putwchar_sink", "WriteFileEx",
    "WriteConsoleOutputW", "GetStdHandleCount", "STD_OUTPUT_HANDLE_COPY", "fdopen_count",
    "my_fdopen", "acrt_iob_func", "__acrt_iob_func_count", "STDOUT_FILENO", "StdoutSink",
    "my_printf_l", "printf_l", "printf_p", "x_printf_p", "_snprintf", "_printf_x", "_writev",
]:
    _ACCEPT.append(
        Case(
            f"accept-longer-identifier-{_word}",
            C_WHOLE + ("; " + C_PREFIX if "printf" in _word and _word.startswith("_") else ""),
            f"src/te411_lw_{_slug(_word)}_{len(_word)}.cpp",
            f"\nvoid te411_case(const char* buf, unsigned n) {{\n    {_word}(buf, n);\n}}\n",
        )
    )

# ---- must-accept: stderr ---------------------------------------------------------------------
_ACCEPT += [
    Case("accept-fprintf-stderr", C_STDERR, "src/te411_se1.cpp", """
#include <cstdio>

void te411_case() {
    std::fprintf(stderr, "x %d\\n", 1);
    fprintf(stderr, "y");
}
"""),
    Case("accept-fprintf-stderr-message-naming-words", C_STDERR + "; " + C_LITERAL, "src/te411_se2.cpp", """
#include <cstdio>

void te411_case() {
    std::fprintf(stderr, "stdout, printf and cout are not used here\\n");
}
"""),
    Case("accept-fprintf-stderr-arguments-on-following-lines", C_STDERR, "src/te411_se3.cpp", """
#include <cstdio>

void te411_case() {
    std::fprintf(
        stderr,
        "x");
}
"""),
    Case("accept-other-stderr-writers", C_STDERR, "src/te411_se4.cpp", """
#include <cstdio>
#include <iostream>

void te411_case(const char* fmt, va_list ap, const char* buf, unsigned n) {
    std::vfprintf(stderr, fmt, ap);
    std::fputs("x", stderr);
    std::fwrite(buf, 1, n, stderr);
    std::fflush(stderr);
    std::cerr << "x";
    std::clog << "x";
    std::perror("x");
}
"""),
    Case("accept-fprintf-stderr-in-macro", C_STDERR, "src/te411_se5.h", r"""
#pragma once
#include <cstdio>
#define TE411_ERR(msg) std::fprintf(stderr, "%s\n", msg)
"""),
]

# ---- must-accept: digit separators and prefixed char literals --------------------------------
_ACCEPT += [
    Case("accept-digit-separated-constant", C_SEPARATOR, "src/te411_as1.cpp", """
void te411_case() {
    long n = 1'000'000;
}
"""),
    Case("accept-separator-then-string-naming-word", C_SEPARATOR + "; " + C_LITERAL, "src/te411_as2.cpp", """
void te411_case() {
    int n = 1'000; const char* s = "printf";
}
"""),
    Case("accept-separators-of-each-radix", C_SEPARATOR, "src/te411_as3.cpp", """
void te411_case() {
    unsigned a = 0xFF'FF; unsigned b = 0b1010'1010; unsigned c = 07'7; double d = 1'000.000'1;
    const char* s = "cout";
}
"""),
    Case("accept-prefixed-char-literals-naming-words", C_LITERAL, "src/te411_as4.cpp", """
void te411_case() {
    auto a = u8'x'; auto b = L'x'; auto c = u'x'; auto d = U'x'; const char* s = "puts";
}
"""),
]

# ---- must-accept: comment and literal forms beyond the per-word matrix -----------------------
_ACCEPT += [
    Case("accept-line-comment-continued-by-splice", C_COMMENT, "src/te411_ax1.cpp", r"""
void te411_case() {
    // a line comment ending in a backslash continues onto the next line \
    std::printf("x");
}
"""),
    Case("accept-line-comment-continued-by-splice-crlf", C_COMMENT, "src/te411_ax2.cpp", r"""
void te411_case() {
    // a line comment ending in a backslash continues onto the next line \
    std::printf("x");
}
""", newline="\r\n"),
    Case("accept-multiline-block-comment-naming-every-word", C_COMMENT, "src/te411_ax3.cpp",
         "\n/*\n" + "\n".join(f"   {w}" for w in _ACCEPT_WORDS) + "\n*/\nvoid te411_case() {}\n"),
    Case("accept-multiline-raw-string-naming-words", C_RAW, "src/te411_ax4.cpp", """
void te411_case() {
    const char* s = R"d(
std::printf("x"); std::cout << 1; stdout )" ## "
)d";
}
"""),
    Case("accept-raw-string-with-paren-quote-before-word", C_RAW, "src/te411_ax5.cpp", """
void te411_case() {
    const char* s = R"d( )" printf )d";
}
"""),
    Case("accept-prefixed-raw-strings-naming-words", C_RAW, "src/te411_ax6.cpp", """
void te411_case() {
    auto a = u8R"(printf)"; auto b = uR"(cout)"; auto c = UR"(stdout)"; auto d = LR"(puts)";
}
"""),
    Case("accept-prefixed-strings-naming-words", C_LITERAL, "src/te411_ax7.cpp", """
void te411_case() {
    auto a = L"printf"; auto b = u8"cout"; auto c = u"puts"; auto d = U"write";
}
"""),
    Case("accept-spliced-string-naming-word", C_LITERAL, "src/te411_ax8.cpp", r"""
void te411_case() {
    const char* s = "abc\
printf";
}
"""),
    Case("accept-pragma-message-naming-word", C_LITERAL, "src/te411_ax9.cpp", """
#pragma message("printf is not called here")
void te411_case() {}
"""),
    Case("accept-stringize-single-hash", C_PASTE, "src/te411_ax10.h", """
#pragma once
#define TE411_STR(x) #x
# define TE411_STR2(x) # x
"""),
    Case("accept-string-containing-escaped-quote-then-word", C_LITERAL, "src/te411_ax11.cpp", r"""
void te411_case() {
    const char* s = "a\" printf \"b";
}
"""),
]

# ---- must-accept: files the walk does not include ---------------------------------------------
for _name in ["te411_notes.txt", "te411_gen.py", "te411_README.md", "te411_x.cpp.bak"]:
    _ACCEPT.append(
        Case(
            f"accept-unscanned-file-{_name}",
            C_WALK,
            f"src/{_name}",
            '\nvoid te411_case() { std::printf("x"); std::cout << 1; }\n',
        )
    )

# Population well-formedness: every must-reject marks at least one line, no must-accept marks
# any, ids and file names are unique, and every listed identifier has its own must-reject.
assert all(c.render()[1] for c in _REJECT), [c.id for c in _REJECT if not c.render()[1]]
assert not any(c.render()[1] for c in _ACCEPT), [c.id for c in _ACCEPT if c.render()[1]]
assert len({c.id for c in _REJECT + _ACCEPT}) == len(_REJECT) + len(_ACCEPT)
assert len({os.path.basename(c.path) for c in _REJECT + _ACCEPT}) == len(_REJECT) + len(_ACCEPT)
assert {w for w, _ in _LISTED_STATEMENTS} <= {c.id[len("reject-ident-"):] for c in _REJECT}


def _scan_planted(case: Case) -> tuple[list[str], int, frozenset[int]]:
    """Plants `case` alone into an empty src/+include/ tree, then returns the scan's hits, main()'s
    return code, and the lines the case marks."""
    body, expected = case.render()
    tmp = tempfile.mkdtemp(prefix="te411_")
    try:
        roots = [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        for root in roots:
            os.makedirs(root)
        full = os.path.join(tmp, *case.path.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(body)
        hits = scanner.scan_for_stdout_writes(roots)
        code = scanner.main(roots)
        return list(hits), code, expected
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _lines_for(hits: list[str], path: str) -> tuple[set[int], list[str]]:
    """Lines reported for the planted file, and any hit that names no line of it."""
    pattern = re.compile(r"(?:^|[/\\])" + re.escape(os.path.basename(path)) + r":(\d+):")
    lines: set[int] = set()
    other: list[str] = []
    for hit in hits:
        m = pattern.search(hit)
        if m:
            lines.add(int(m.group(1)))
        else:
            other.append(hit)
    return lines, other


@pytest.mark.parametrize("case", _REJECT, ids=[c.id for c in _REJECT])
def test_population_must_reject(case: Case):
    """The planted file must be reported at exactly its marked lines, and main() must refuse."""
    _require_scanner()
    hits, code, expected = _scan_planted(case)
    lines, other = _lines_for(hits, case.path)
    body, _ = case.render()
    assert lines == set(expected) and not other, (
        f"[{case.clause}] {case.path}: expected hits at lines {sorted(expected)}, "
        f"reported {sorted(lines)}; hits not naming the file: {other!r}; all hits: {hits!r}\n"
        f"--- planted source ---\n{body}"
    )
    assert code != 0, f"[{case.clause}] main() returned {code} (success) for {case.path}"


@pytest.mark.parametrize("case", _ACCEPT, ids=[c.id for c in _ACCEPT])
def test_population_must_accept(case: Case):
    """The planted file must produce no hit, and main() must return 0."""
    _require_scanner()
    hits, code, _ = _scan_planted(case)
    body, _ = case.render()
    assert hits == [], (
        f"[{case.clause}] {case.path} must be accepted; hits: {hits!r}\n"
        f"--- planted source ---\n{body}"
    )
    assert code == 0, f"[{case.clause}] main() returned {code} (failure) for {case.path}"


# =============================================================================================
# 3. The TE-407 regression set (not the commissioning population)
# =============================================================================================
# The reviewer's executed 15-form population from TE-407
# (Claude/Poirot/7f7436d-slm171-stdout-reconfirm.md, S1), reproduced verbatim. The scanner's banned
# list was revised against these forms, so they cannot commission it; they stay as regression
# cases. Each form is planted into a fresh `git archive` copy of the current tree, and the scan
# must name the planted file.

_TE407_MUTANT_FORMS = {
    "ctrl_fprintf_stdout.cpp": 'void f(){ std::fprintf(stdout, "x"); }\n',
    "ctrl_cout.cpp": "void f(){ std::cout << 1; }\n",
    "multiline_fprintf.cpp": 'void f(){ std::fprintf(\n    stdout, "x %d", 1); }\n',
    "multiline_fwrite.cpp": (
        "void f(const char* b, size_t n){ std::fwrite(b, 1, n,\n    stdout); }\n"
    ),
    "vprintf.cpp": (
        "#include <cstdarg>\nvoid f(const char* fmt, va_list ap){ std::vprintf(fmt, ap); }\n"
    ),
    "vfprintf_stdout.cpp": (
        "#include <cstdarg>\nvoid f(const char* fmt, va_list ap){ "
        "std::vfprintf(stdout, fmt, ap); }\n"
    ),
    "printf_s.cpp": 'void f(){ printf_s("x"); }\n',
    "putchar.cpp": "void f(){ std::putchar(120); }\n",
    "fputws_stdout.cpp": 'void f(){ std::fputws(L"x", stdout); }\n',
    "fputc_stdout.cpp": "void f(){ std::fputc(120, stdout); }\n",
    "using_cout.cpp": "using namespace std;\nvoid f(){ cout << 1; }\n",
    "posix_write_fd1.cpp": "void f(const char* b, unsigned n){ _write(1, b, n); }\n",
    "stderr_then_stdout_same_line.cpp": (
        'void f(){ std::fprintf(stderr, "a"); std::fprintf(stdout, "b"); }\n'
    ),
    "stdout_via_variable.cpp": 'void f(){ FILE* o = stdout; std::fprintf(o, "x"); }\n',
}
# The 15th form: a write appended to a shipped public `.inc` header, which
# include/superslm/sslm_abi.h pulls in with `#include "superslm/sslm_abi_functions.inc"`.
_TE407_INC_MUTANT_PATH = "include/superslm/sslm_abi_functions.inc"
_TE407_INC_MUTANT_BODY = '\ninline void TE407Noise(){ std::printf("x"); }\n'


def _plant_in_current_tree_and_scan(rel_path: str, body: str, prefix: str) -> list[str]:
    """Materializes the current tree, plants (or, for an existing file, appends) `body` at
    `rel_path`, and returns the hits naming that file."""
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        _git_archive_tree(_resolve_head(), tmp)
        full_path = os.path.join(tmp, *rel_path.split("/"))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "a" if os.path.exists(full_path) else "w"
        with open(full_path, mode, encoding="utf-8") as f:
            f.write(("#include <cstdio>\n#include <iostream>\n" if mode == "w" else "") + body)
        hits = scanner.scan_for_stdout_writes(
            [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        )
        needle = os.path.basename(rel_path)
        return [h for h in hits if needle in h]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("name,body", sorted(_TE407_MUTANT_FORMS.items()))
def test_regression_te407_form(name, body):
    """Each TE-407 form planted as a new file under src/ must be named by the scan."""
    _require_scanner()
    hits = _plant_in_current_tree_and_scan(f"src/{name}", body, prefix=f"te407_form_{name}_")
    assert hits, f"TE-407 regression form {name!r} was not caught -- content: {body!r}"


def test_regression_te407_inc_header():
    """A std::printf appended to the shipped public .inc header must be named by the scan."""
    _require_scanner()
    hits = _plant_in_current_tree_and_scan(
        _TE407_INC_MUTANT_PATH, _TE407_INC_MUTANT_BODY, prefix="te407_inc_"
    )
    assert hits, (
        f"appending a std::printf to {_TE407_INC_MUTANT_PATH} was not caught -- appended: "
        f"{_TE407_INC_MUTANT_BODY!r}"
    )


def test_regression_new_printf_in_a_new_translation_unit():
    """A plain std::printf in a new file under the current tree must be named by the scan -- the
    scan generalizes beyond the one site v1.7.0 carried."""
    _require_scanner()
    hits = _plant_in_current_tree_and_scan(
        "src/te399_synthetic_new_stdout_write.cpp",
        'void TE399SyntheticNoise() { std::printf("noise\\n"); }\n',
        prefix="te399_ci_scan_mutant_",
    )
    assert hits, "a std::printf planted in a new file under the current tree was not caught"
