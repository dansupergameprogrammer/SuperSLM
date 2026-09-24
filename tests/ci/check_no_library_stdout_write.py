"""CI source scan: a tripwire for stdout-writing spellings in the shipped `superslm`/`superslm_gpu`
library sources (TE-400, D-SLM7753; TE-402 S3; promise narrowed by D-SLM7783).

WHAT IT PROMISES. The scan refuses any `src/`/`include/` source file that names one of the banned
stdout-writing identifiers below, or the token-paste operator, outside a comment or a string/char
literal. It is a tripwire for those spellings, not a proof that the library writes nothing to
stdout: code that reaches stdout without naming a listed identifier can pass. That the 1.7.1
library writes nothing to stdout is a separate claim, verified for that release by a hand sweep
of the tree, not by this scan.

WHAT IS BANNED -- defined in exactly one place, `_BANNED_IDENTIFIERS`, `_BANNED_IDENTIFIER_FAMILY`
and `_BANNED_OPERATORS` below, each matched as a whole word and case-sensitively:

- stream and handle names: `stdout`, `STD_OUTPUT_HANDLE`, `cout`, `wcout`, `__acrt_iob_func`
  (the UCRT's own expansion of `stdout`), `GetStdHandle`;
- the printf family: `printf`, `wprintf`, `vprintf`, `vwprintf`, each also with a `_s` suffix;
  and the MSVC underscore-prefixed family -- any identifier starting with `_` that contains
  `printf` followed by a `_l` or `_p` suffix, optionally after `_s` (`_printf_l`, `_printf_p`,
  `_printf_p_l`, `_printf_s_l`, `_vwprintf_p`, ...; the stream and buffer variants in that family,
  such as `_fprintf_p` and `_sprintf_l`, are banned with it);
- character and string writers: `puts`, `_putws`, `putchar`, `putwchar`;
- raw writes: `write`, `_write`, `WriteFile`, `WriteConsole`, `WriteConsoleA`, `WriteConsoleW`;
- descriptor-to-stream openers: `fdopen`, `_fdopen`;
- the token-paste operator `##` and its digraph `%:%:`, because pasting can assemble any banned
  name from fragments no word-level ban can see.

`stderr`, `fprintf` and `fwprintf` are not banned: `fprintf(stderr, ...)` is this library's
diagnostic channel. A line that names them together with `stdout` is refused by the `stdout` ban.
Some banned identifiers also write to targets other than stdout (`write`, `WriteFile`), and the
family pattern catches some that never write to stdout (`_sprintf_l`). They are banned outright
because the library uses none of them, and a future legitimate use is a reviewed change to this
list rather than a silent pass.

HOW. A text scan, not a C++ parser. `_strip_comments_and_literals` blanks comments and
string/char literals with equal-length whitespace, keeping every newline, so reported line numbers
are exact. It recognises C++14 digit separators (a `'` inside a number such as `1'000` is not a
character-literal delimiter) and raw string literals (`R"delim(...)delim"`, with the prefixes
`u8R`, `uR`, `UR` and `LR`). An ordinary string or character literal ends at an unescaped newline,
which a well-formed literal cannot contain, so a quote the stripper misreads cannot blank the lines
after it. The ban then runs on what survives. Scanned extensions: `.c`, `.cpp`, `.h`, `.hpp`, `.inc`,
`.def`, `.inl`, `.ipp`, `.hlsl`, `.hlsli` (`include/superslm/sslm_abi_functions.inc` is part of the
public API, included by `include/superslm/sslm_abi.h`).

Known limits, each a way code can pass the scan: preprocessor tricks other than token pasting
(a macro whose expansion names a banned identifier defined outside `src/`/`include/`), function
pointers or handles obtained without naming a listed identifier, and writes to a descriptor or
handle that equals stdout's without being named as such.

The interface is pinned by tests/ci/test_te399_stdout_write_scan_commission.py:
`scan_for_stdout_writes(root_dirs)` and `main(argv)`. Exit code 0 iff no scanned file contains a
banned token outside a comment or literal; 1 otherwise, naming every `path:line: reason` hit.
"""
from __future__ import annotations

import os
import re
import sys

_SCANNED_EXTENSIONS = (".c", ".cpp", ".h", ".hpp", ".inc", ".def", ".inl", ".ipp", ".hlsl", ".hlsli")

# The banned list. The module docstring describes it; this is where it is defined.
_BANNED_IDENTIFIERS = (
    "stdout", "STD_OUTPUT_HANDLE", "cout", "wcout", "__acrt_iob_func", "GetStdHandle",
    "printf", "wprintf", "vprintf", "vwprintf",
    "printf_s", "wprintf_s", "vprintf_s", "vwprintf_s",
    "puts", "_putws", "putchar", "putwchar",
    "write", "_write", "WriteFile", "WriteConsole", "WriteConsoleA", "WriteConsoleW",
    "fdopen", "_fdopen",
)
# MSVC's underscore-prefixed printf family: `_printf_l`, `_printf_p`, `_printf_p_l`,
# `_printf_s_l`, `_vwprintf_p`, `_fprintf_p`, `_sprintf_l`, ...
_BANNED_IDENTIFIER_FAMILY = r"_\w*printf(?:_s)?_[lp]\w*"
# The token-paste operator and its digraph spelling.
_BANNED_OPERATORS = ("##", "%:%:")

_BANNED_TOKENS = re.compile(
    r"\b(?:"
    + "|".join(re.escape(name) for name in _BANNED_IDENTIFIERS)
    + "|"
    + _BANNED_IDENTIFIER_FAMILY
    + r")\b|"
    + "|".join(re.escape(op) for op in _BANNED_OPERATORS)
)

_RAW_STRING_PREFIXES = frozenset({"R", "u8R", "uR", "UR", "LR"})
_RAW_DELIMITER_MAX = 16
_RAW_DELIMITER_FORBIDDEN = frozenset(" ()\\\t\v\f\n")


def _is_identifier_char(c: str) -> bool:
    return c.isalnum() or c == "_" or c == "$"


def _blank(text: str) -> str:
    """Equal-length whitespace for `text`, keeping its newlines."""
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _raw_string_end(source: str, quote: int) -> int:
    """If `source[quote]` opens a raw string's body (`"delim(`), return the index just past its
    closing `)delim"`, or len(source) if it is never closed. Return -1 if the delimiter is not a
    valid raw-string delimiter, in which case the quote is an ordinary string's."""
    n = len(source)
    k = quote + 1
    while k < n and source[k] != "(":
        if source[k] in _RAW_DELIMITER_FORBIDDEN or k - (quote + 1) >= _RAW_DELIMITER_MAX:
            return -1
        k += 1
    if k >= n:
        return -1
    terminator = ")" + source[quote + 1:k] + '"'
    end = source.find(terminator, k + 1)
    return n if end == -1 else end + len(terminator)


def _pp_number_end(source: str, i: int) -> int:
    """Index just past the preprocessing number starting at `source[i]` (a digit): digits,
    identifier characters, `.`, an exponent sign after e/E/p/P, and a `'` digit separator
    followed by a digit or identifier character."""
    n = len(source)
    j = i + 1
    while j < n:
        ch = source[j]
        if ch in "eEpP" and j + 1 < n and source[j + 1] in "+-":
            j += 2
        elif _is_identifier_char(ch) or ch == ".":
            j += 1
        elif ch == "'" and j + 1 < n and _is_identifier_char(source[j + 1]):
            j += 2
        else:
            break
    return j


def _strip_comments_and_literals(source: str) -> str:
    """Replace every block comment, line comment, and string/char literal (raw strings included)
    in `source` with equal-length whitespace, keeping every newline so line numbers and column
    offsets match the original. Identifiers and preprocessing numbers are consumed whole, so a
    literal prefix (`L`, `u8`, `R`, ...) is recognised and a digit separator is never read as a
    character-literal delimiter. An ordinary string or character literal ends at an unescaped
    newline."""
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            end = source.find("\n", i)
            end = n if end == -1 else end
            out.append(" " * (end - i))
            i = end
            continue
        if c == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(_blank(source[i:end]))
            i = end
            continue
        if c.isdigit():
            end = _pp_number_end(source, i)
            out.append(source[i:end])
            i = end
            continue
        if _is_identifier_char(c):
            end = i + 1
            while end < n and _is_identifier_char(source[end]):
                end += 1
            word = source[i:end]
            out.append(word)
            i = end
            if word in _RAW_STRING_PREFIXES and i < n and source[i] == '"':
                raw_end = _raw_string_end(source, i)
                if raw_end != -1:
                    out.append(_blank(source[i:raw_end]))
                    i = raw_end
            continue
        if c == '"' or c == "'":
            j = i + 1
            while j < n:
                ch = source[j]
                if ch == "\\" and j + 1 < n:
                    j += 2
                    continue
                if ch == c:
                    j += 1
                    break
                if ch == "\n":
                    break
                j += 1
            out.append(_blank(source[i:j]))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _scan_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        original = f.read()
    stripped = _strip_comments_and_literals(original)
    hits: list[str] = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        m = _BANNED_TOKENS.search(line)
        if m:
            hits.append(f"{path}:{lineno}: stdout write (banned token '{m.group(0)}')")
    return hits


def scan_for_stdout_writes(root_dirs: list[str]) -> list[str]:
    """Every source line under any of `root_dirs` (walked recursively; files with an extension in
    `_SCANNED_EXTENSIONS`) that names a banned token outside a comment or string/char literal.
    Returns "path:line: reason" strings, sorted; empty if none."""
    hits: list[str] = []
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                if not name.endswith(_SCANNED_EXTENSIONS):
                    continue
                hits.extend(_scan_file(os.path.join(dirpath, name)))
    return sorted(hits)


def main(argv: list[str]) -> int:
    """CLI entry point: argv names one or more root directories (`src include` in CI); prints
    every hit and returns 1 if any exist, 0 if none -- the same contract as the sibling checks in
    this directory."""
    if not argv:
        print("check_no_library_stdout_write.py: no root directories given", file=sys.stderr)
        return 1
    hits = scan_for_stdout_writes(argv)
    if hits:
        for h in hits:
            print(h)
        print(
            f"check_no_library_stdout_write.py: FAILED -- {len(hits)} line(s) under {argv!r} "
            "name a banned stdout-writing token outside a comment or literal (see this "
            "script's module docstring for the list)",
            file=sys.stderr,
        )
        return 1
    print(f"check_no_library_stdout_write.py: OK -- no banned stdout-writing token under {argv!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
