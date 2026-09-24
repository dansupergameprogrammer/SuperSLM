"""CI source check: the shipped `superslm`/`superslm_gpu` libraries write nothing to stdout
(TE-400, D-SLM7753; TE-402 S3, remedy #4: "add a hosted-CI source scan that fails on any stdout
write in src/ and include/"). A one-time grep at the time of the fix is not a standing
guarantee -- a future `printf` added anywhere under `src/` or `include/` would ship silently,
exactly the gap S3 found. This makes the sweep TE-400's own build log performed by hand into
something CI runs on every push, refusing any stdout write it finds.

Commissioned, blind, by tests/ci/test_te399_stdout_write_scan_commission.py (Curie, TE-399 fix
round) -- this module implements exactly that commission's own interface contract; do not change
the two public names or their signatures without updating that test, which this module does not
own.

WHAT COUNTS AS A LIBRARY STDOUT WRITE (TE-407 S1 fix round). A TOKEN BAN, not a call-shape
match: any comment/string-stripped line containing the bare identifier `stdout`,
`STD_OUTPUT_HANDLE`, `cout`, `wcout`, the printf family (`printf`, `wprintf`, `vprintf`,
`vwprintf`, and each with a trailing `_s`), `puts`, `_putws`, the putchar family (`putchar`,
`putwchar`), the Win32 console-write family (`WriteConsoleA`/`WriteConsoleW`), or the POSIX
raw-descriptor write `_write` -- as a whole word, never as a call-shape regex. TE-402's own
call-shape matcher (an `fprintf(` regex plus "is the first argument `stdout`?") was proven, by
execution (TE-407 S1, `scan_mutants.py`, 13 of 15 escaped), to miss anything that does not look
exactly like the one site it was written from: `std::vprintf`, `printf_s`, `std::putchar`,
`std::fputc(..., stdout)`, `using namespace std; cout <<`, `stdout` held in a `FILE*` variable,
two calls on one line (only the first was inspected), a call whose `stdout` argument is on a
following line (each regex ran per line), and a `std::printf` planted in a public `.inc` header
(not in the old scanned-extension set) all escaped. A token ban has no call shape to miss: if the
word `stdout` (or any of the above) appears anywhere outside a comment or a string/char literal
in a shipped library source file, that is the finding, regardless of which function holds it,
which line the call opened on, or how many calls share a line. `_write` is banned unconditionally
(TE-407's own `posix_write_fd1.cpp` mutant, `_write(1, buf, n)`, names the stdout file descriptor
only by the numeral 1, which no word-level ban can single out from other integer arguments) --
verified absent from this tree already (zero pre-existing uses), so banning it outright costs
nothing here. `stderr` and `fprintf`/`fwprintf` alone (without the word `stdout` anywhere on the
line) are NOT banned: `fprintf(stderr, ...)` is this project's own established diagnostic channel
(d3d12_harness.h's own convention, and TE-400's own fix), and it is legal precisely because the
line naming it never also contains the word `stdout`.

The English word "puts" inside a comment (e.g. "puts the row maximum", legitimately present
several times in this tree) is not flagged -- comments are stripped before the ban runs, per
METHOD below, the same protection TE-402's implementation already had and this round keeps.

METHOD. Modelled on this directory's own precedent, check_no_forward_leaf_calls.py: a
comment-aware text scan, not full C++ parsing. Block comments, line comments, and string/char
literals are stripped with a small state machine (preserving line numbers exactly, the same
discipline check_no_forward_leaf_calls.py's own `_strip_comments_preserving_line_numbers` uses,
extended here to also blank string/char literals so a format string's own text cannot trip the
ban), unchanged from TE-402's implementation -- TE-407 S1 found the STRIPPING sound and the
MATCHING too narrow; only the matching changed this round. The token-ban regex then runs on what
survives.

`.inc` and `.def` (TE-407 S1's own remedy: "add `.inc`, `.def`, `.inl` and `.ipp` to the scanned
extensions" -- `.inl`/`.ipp` are not currently used anywhere in this tree but are added
preemptively for the same reason `.inc` was missed: a public API surface is not guaranteed to
stay confined to `.h`) are now scanned alongside `.c`/`.cpp`/`.h`/`.hpp`/`.hlsl`/`.hlsli` -- the
S1 escape that mattered most: `include/superslm/sslm_abi_functions.inc` is pulled into the public
API by `sslm_abi.h:324` and was not being scanned at all.

Exit code 0 iff no scanned file contains a banned token outside a comment or literal; 1
otherwise, naming every `path:line: reason` hit.
"""
from __future__ import annotations

import os
import re
import sys

_SCANNED_EXTENSIONS = (".c", ".cpp", ".h", ".hpp", ".inc", ".def", ".inl", ".ipp", ".hlsl", ".hlsli")

# TE-407 S1: a token ban, not a call-shape match (see the module docstring for why). Matches the
# reviewer's own tokenban.py probe pattern, plus `_write` (added to close the POSIX
# raw-file-descriptor escape that names no identifier the probe's own token set covers).
_BANNED_TOKENS = re.compile(
    r"\b(stdout|STD_OUTPUT_HANDLE|cout|wcout|v?w?printf(_s)?|puts|_putws|putw?char|"
    r"WriteConsole[AW]?|_write)\b"
)


def _strip_comments_and_literals(source: str) -> str:
    """Replace every block comment, line comment, and string/char literal in `source` with
    equal-length whitespace (preserving line numbers and column offsets exactly, so reported line
    numbers match the original file), leaving only code text for the forbidden-call regexes to
    see. Same discipline as check_no_forward_leaf_calls.py's own
    `_strip_comments_preserving_line_numbers`, extended to also blank string/char literals (a
    format string's own text, e.g. containing the word "printf", must not itself trip the
    regexes)."""
    out: list[str] = []
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
                out.append(c)
            else:
                out.append(" ")
            i += 1
            continue
        if in_block_comment:
            if c == "*" and nxt == "/":
                out.append("  ")
                in_block_comment = False
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                in_string = False
                out.append(" ")
                i += 1
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if in_char:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == "'":
                in_char = False
                out.append(" ")
                i += 1
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        # Not currently inside any comment/literal.
        if c == "/" and nxt == "/":
            in_line_comment = True
            out.append("  ")
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            out.append("  ")
            i += 2
            continue
        if c == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        if c == "'":
            in_char = True
            out.append(" ")
            i += 1
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
            hits.append(f"{path}:{lineno}: stdout write (banned token '{m.group(1)}')")
    return hits


def scan_for_stdout_writes(root_dirs: list[str]) -> list[str]:
    """Every source line under any of `root_dirs` (walked recursively; .c/.cpp/.h/.hpp/.inc/
    .def/.inl/.ipp/.hlsl/.hlsli by extension, matching this repo's own shipped-library source
    set) containing a banned stdout-writing token outside a comment or string/char literal --
    stdout/STD_OUTPUT_HANDLE/cout/wcout/the printf family/puts/_putws/the putchar family/
    WriteConsole*/_write (TE-407 S1: a token ban, not a call-shape match -- see the module
    docstring). Returns "path:line: reason" strings, empty if clean."""
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
    """CLI entry point: argv names one or more root directories (this repo's own src/ and
    include/ in normal CI use); prints every hit from scan_for_stdout_writes and returns 1 if any
    exist, 0 if the scan is clean -- the same contract check_no_pow_operator.main() and
    check_no_forward_leaf_calls.main() already use in this directory, so this scan wires into CI
    (.github/workflows/tests.yml) and a local `pytest tests/ci/` run the same way every sibling
    check here does."""
    if not argv:
        print("check_no_library_stdout_write.py: no root directories given", file=sys.stderr)
        return 1
    hits = scan_for_stdout_writes(argv)
    if hits:
        for h in hits:
            print(h)
        print(
            f"check_no_library_stdout_write.py: FAILED -- {len(hits)} stdout write(s) found "
            f"under {argv!r}; the engine library must write nothing to stdout (TE-400, "
            "D-SLM7753)",
            file=sys.stderr,
        )
        return 1
    print(f"check_no_library_stdout_write.py: OK -- no stdout write found under {argv!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
