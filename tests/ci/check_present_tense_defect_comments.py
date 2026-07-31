"""CI source check: T-1331's defect class -- a test-suite comment that cites one
of this project's review-severity findings (Critical N / Significant N / Weak N /
Structural N -- the labels this codebase's own review casebooks use to number a
finding, verbatim, e.g. "Critical 1", "Significant 5") must also show, in the same
comment block, that the finding is resolved: either the literal word "closed", or
a cited ticket ID (`T-<digits>`). A comment carrying the label with neither is
flagged (SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 intro, Sec14.12; D-SLM411,
D-SLM414, D-SLM418; board T-1337).

WHY THIS SHAPE, NOT THE PLAN'S LITERAL CANDIDATE. Sec14.12 proposes "a present-
tense defect comment must cite the ticket that pins it, so the comment fails when
that ticket closes without the comment changing." Read literally that is a TWO-
PART mechanism: (1) require a citation, (2) invalidate it when the cited ticket's
STATUS changes. Part (2) is not buildable inside this repository's CI: ticket
lifecycle lives on the board (`Claude/Zelda/Board.md`) and in the decision log
(`Claude/Decisions/DecisionLog.md`), both in a *different* project tree, not
checked out by this repo's CI jobs and not vendored here. A job running on this
repo's stock checkout has no way to ask "is T-1234 still open" -- it can only ask
"does this comment cite something." Confirmed against real data: recovering the
seven historical sites this class was found at (below) shows NONE of them cited a
literal ticket ID at all -- five cited a review-casebook filename plus a severity
label ("Critical 1 (Poirot fa3189a-...-review-2026-07-28.md)"), and the sixth (a
CHECK_MSG failure-message string) cited only the label. So "must cite A ticket" as
literally written would flag every one of these sites' own FIXED, current-truth
comments too (the fix rewrote five of the seven to cite "(closed; ...)" rather
than a ticket ID) -- a check built to the letter of the candidate would fire
forever on correct, current text, which fails the "does not fire on clean input"
half of the exit condition on its very first real-tree run. What is actually
buildable, and validated below against the real historical population: requiring
a **resolution marker** ("closed", or a cited ticket ID where one IS given, as the
seventh site's own fix does) alongside the severity-label citation. This is
Part (1) made real, plus the one form Part (2) can take without cross-repo access:
a human (or a future job with board access) revisits every "T-<n>, not yet closed"
comment; this module cannot tell the difference between an OPEN cited ticket and
a CLOSED one on its own, and says so below rather than silently mis-claiming it.

RULE COVERAGE, MEASURED. `tests/ci/present_tense_defect_historical_fixtures/`
holds the exact text of all seven sites T-1331/T-1334 found (D-SLM411, D-SLM414),
recovered from `D:\\SuperSLM` git history at fa5113d^ (six sites) and ee76dbd^ (the
seventh) -- the state BEFORE each was rewritten -- paired with each site's current,
fixed text at HEAD. `test_check_present_tense_defect_comments.py`'s
`test_fires_on_every_historical_pre_fix_site` and
`test_does_not_fire_on_any_historical_post_fix_site` replay both states through
this module's own scanner (never through a hand-transcribed stand-in) and assert
7/7 fire before the fix and 0/7 false-fire after it -- fixing nothing here, per
StandardsDocument Sec4's population-validation requirement.

INPUT COVERAGE. The default scan surface is `tests/**/*.cpp`, `tests/**/*.h`, and
`tests/**/*.py` -- the obligation's own text scopes it to "this campaign's test
suite" (plan Sec11 intro), not production source; `src/` and `include/` also carry
some of these same severity-label citations (as provenance on a fix, not as a live
defect claim) and are deliberately outside this check's scope for that reason.

KNOWN FALSE POSITIVE, NAMED RATHER THAN SUPPRESSED. A purely navigational section
header that cites several severity labels as an index (no defect language at all,
e.g. "the remediation red suite for Critical 1, Critical 2, and Significant 5")
also lacks a resolution marker and is flagged by this rule as written. Narrowing
the rule to require a negation word ("no"/"never"/"not") alongside the label
would suppress that false positive -- but it also suppresses a REAL, independently
found instance of this exact defect class (see the module's own casebook/build-log
citation for the discovered site), which contains no negation word at all
("... reads unmapped heap memory ..." stated as present fact, no "no"/"never").
Between a rule that over-flags a benign header and one that misses a real defect,
this module keeps the wider rule and reports both outcomes plainly rather than
tuning against either.

Modelled on tests/ci/check_no_forward_leaf_calls.py's own conventions: a text
scan (not a full C++ parse) over a glob-derived file set, a `scan_files` entry
point returning formatted failure strings, and a `main()` that fails loud.

BLOCK-BOUNDARY HARDENING (T-1485). The first real-tree run of this check found
two mechanical gaps in `_iter_blocks`, both scoped to `.py` files and both
fixed here with their own regression cells in the test module: (1) a Python
multi-line triple-quoted docstring's OPENING line happens to start with `"`
and so passed the plain-STRING check, but every following line does not start
with a quote character at all, so only the opening line was ever scanned --
silently hiding a resolution marker that in fact appeared later in the same
docstring, and reporting a fragment of a fully-cited docstring as an uncited
one. (2) `tests/gen_*.py`'s own convention -- a Python list of string
literals, one per generated-file physical line, used to build multi-line C++
output text -- was merged as ONE contiguous STRING block for every
consecutive line that happened to be a complete Python string starting with
`"`, whether or not that line's own content was a `//` comment; a five-line
generated comment was found merged, this way, into an eighteen-line block
that also swallowed a struct definition and blank-line placeholders between
witnesses. `_PY_QUOTED_CPP_COMMENT_PATTERN` fixes (2): a quoted `//` line
now merges only with adjacent quoted `//` lines.

TOKENIZE-DERIVED PYTHON STRING SPANS (T-1538). T-1485's fix for (1) tracked a
`.py` multi-line triple-quoted string by per-physical-line triple-quote
PARITY -- a line outside a string whose triple-quote count was odd was read
as opening one, with no whole-file notion of whether a string was already
open. Any ordinary line carrying an odd count of triple-quote markers --
including a comment merely naming the delimiter, or this module's own
`_TRIPLE_QUOTE_MARKERS` tuple declaration -- flipped that parity for the
remainder of the file: every later docstring's closing line then read as an
opener and every opening line as a closer, so code was scanned as string
text and docstring bodies were skipped entirely. Applied to `.cpp`/`.h`
inputs (no line in C++ ever closes a phantom Python string), one such stray
run collapsed the rest of the file into a single block, silencing every
citation below it.

`_iter_blocks` no longer does per-line triple-quote parity tracking at all.
For a `.py` file, the physical lines that fall inside a multi-line STRING
token are taken from Python's own `tokenize` module
(`_python_multiline_string_lines`) -- ground truth, not a heuristic
re-derived one line at a time -- and `.cpp`/`.h` inputs get no triple-quote
handling whatsoever, so a stray triple-quote marker inside a C++ comment can
no longer collapse the file. A `.py` file that `tokenize` cannot parse raises
`PythonTokenizeError`, which `scan_files` reports as a per-file failure
rather than silently treating the unparseable file as clean.
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys
import tokenize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# Poirot's own review-severity taxonomy, cited verbatim wherever a test-suite
# comment pins a claim to a specific review finding: "Critical 1", "Significant
# 5", etc. Requires a digit -- labels like "Significant B" (a coverage-dimension
# tag, a different convention entirely) do not match and are not this check's
# concern.
SEVERITY_LABEL_PATTERN = re.compile(r"\b(?:Critical|Significant|Weak|Structural)\s+\d+\b")

# A resolution marker: the literal word "closed" (case-insensitive; this
# codebase's own remediation convention, e.g. "Critical 1 (closed; Poirot ...)"),
# or a cited ticket ID in the form this codebase uses project-wide (T-<digits>;
# the one historical site that DOES cite a ticket -- "Before the T-1322 remedy"
# -- cites it in exactly this form). Either is accepted: this module cannot
# distinguish a still-open cited ticket from a closed one (see module docstring),
# so a cited ticket ID is treated as "this claim is tracked," not "verified closed."
RESOLUTION_MARKER_PATTERN = re.compile(r"closed|\bT-\d+\b", re.IGNORECASE)

_DEFAULT_TEST_GLOBS = (
    "tests/**/*.cpp",
    "tests/**/*.h",
    "tests/**/*.py",
)



# A Python source line that is itself a quoted C++ `//` comment line --
# the shape this codebase's fixture generators use (tests/gen_*.py: a list of
# Python string literals, one per generated-file physical line, some of which
# are `//` comments in the OUTPUT file). Every such line starts with a Python
# quote character immediately followed by `//` (e.g. `"// --- Witness 4 ...",`).
# Recognized as COMMENT, not the catch-all STRING below, so a run of these
# merges only with adjacent quoted `//` lines -- never with a neighboring
# quoted line of ordinary generated code or a blank-line placeholder (`"",`),
# which is exactly what a real `//` comment run in the generated .h/.cpp file
# itself would do. Without this, T-1485's own real-tree run found a Python
# generator's five-line `//` comment merged into an eighteen-line block that
# also swallowed a struct definition and blank-line separators between them,
# because every one of those unrelated lines is ALSO, coincidentally, its own
# complete Python string literal starting with `"`.
_PY_QUOTED_CPP_COMMENT_PATTERN = re.compile(r"""^["']//""")

# Raised when a `.py` file handed to _python_multiline_string_lines cannot be
# tokenized as Python. A file this check cannot parse must not be silently
# treated as clean (T-1538; StandardsDocument Sec4) -- scan_files reports this
# as a per-file failure, the same as a missing file, rather than swallowing it.
class PythonTokenizeError(Exception):
    """A `.py` input could not be tokenized; its string spans are unknown."""


def _python_multiline_string_lines(lines: list[str]) -> frozenset[int]:
    """Physical line numbers (1-based) that fall inside a multi-line STRING
    token, per Python's own tokenizer -- ground truth for a `.py` file's
    triple-quoted-string spans (T-1538).

    Replaces this module's earlier per-physical-line triple-quote-PARITY
    heuristic (T-1485), which carried no whole-file notion of whether a
    string was already open: any ordinary line outside a string whose
    triple-quote count was odd -- a comment merely naming the delimiter, or
    this module's own former `_TRIPLE_QUOTE_MARKERS` declaration -- flipped
    the tracked polarity for the remainder of the file, so every later
    docstring's closing line read as an opener and every opening line as a
    closer. `tokenize` has no such state to desynchronise: each STRING
    token's own `start`/`end` line numbers come from the interpreter's own
    lexer, not from re-counting delimiters on each line in isolation."""
    text = "".join(lines)
    result: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                result.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, SyntaxError, IndentationError, UnicodeDecodeError) as exc:
        raise PythonTokenizeError(f"could not tokenize as Python: {exc}") from exc
    return frozenset(result)


def _line_kind(line: str) -> str:
    """Classifies one physical line for block-grouping purposes. `_iter_blocks`
    calls this for every line except one it already knows, from a `.py` file's
    tokenize-derived multi-line-string spans, to be inside a Python string
    literal -- those lines are forced to STRING without consulting this
    function at all (see `_iter_blocks`). A line is part of a textual block
    if it is a `//` comment (any indentation), a
    Python-quoted `//` comment line (`_PY_QUOTED_CPP_COMMENT_PATTERN`), or a
    bare C string-literal continuation (the shape this codebase's multi-line
    CHECK_MSG messages take: one quoted string literal per physical line).
    Anything else -- code, blank lines -- is a break between blocks, so a
    resolution marker written for one comment can never silently satisfy a
    citation in an unrelated, later block."""
    stripped = line.strip()
    if stripped.startswith("//"):
        return "COMMENT"
    if _PY_QUOTED_CPP_COMMENT_PATTERN.match(stripped):
        return "COMMENT"
    if stripped.startswith('"'):
        return "STRING"
    return "OTHER"


def _iter_blocks(
    lines: list[str], python_string_lines: frozenset[int] | None = None
) -> list[tuple[int, int, str]]:
    """Every maximal run of same-kind (COMMENT or STRING) contiguous lines, as
    (1-based start line, 1-based end line, merged text). Lines are joined with a
    space so a citation split across a line break -- this codebase's own fixed
    text does this ("...(Significant " / "5, closed by...)") -- is still found as
    one continuous match against the merged text, never only against one line.

    `python_string_lines`, when given, is the set of 1-based physical line
    numbers a `.py` file's tokenizer places inside a multi-line STRING token
    (`_python_multiline_string_lines`). Every such line is forced to kind
    STRING regardless of what character it starts with, which is what keeps a
    multi-line docstring together as one block the same way the `//` and
    bare-C-string rules already keep their own multi-line shapes together --
    without re-deriving "is a string open" from each line's own content the
    way T-1485's triple-quote-parity tracker did (see
    `_python_multiline_string_lines`'s docstring for why that desynchronised).
    `.cpp`/`.h` inputs pass `None`: C++ has no triple-quoted strings, so no
    line in a C++ file should ever force this state, and none does."""
    blocks: list[tuple[int, int, str]] = []
    start: int | None = None
    kind: str | None = None
    buf: list[str] = []

    def _close() -> None:
        nonlocal start, kind, buf
        if kind in ("COMMENT", "STRING") and start is not None:
            blocks.append((start, start + len(buf) - 1, " ".join(buf)))
        start, kind, buf = None, None, []

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if python_string_lines is not None and i in python_string_lines:
            this_kind = "STRING"
        else:
            this_kind = _line_kind(line)

        if this_kind in ("COMMENT", "STRING") and this_kind == kind:
            buf.append(stripped)
            continue
        _close()
        if this_kind in ("COMMENT", "STRING"):
            start, kind, buf = i, this_kind, [stripped]
    _close()
    return blocks


def scan_text(lines: list[str], is_python: bool = False) -> list[tuple[int, int, list[str]]]:
    """Every (start_line, end_line, [labels]) block among `lines` that cites at
    least one severity-finding label with no resolution marker anywhere in the
    same block. Driven directly from an in-memory line list -- used both by
    find_uncited_defect_citations (file-backed) and by the historical-population
    validation cells, which replay recovered git-blob text without writing it to
    disk first (all seven recovered sites are `.cpp`-shaped text, so those cells
    correctly take the `is_python=False` default). A text scan, not a parsed
    AST -- see module docstring for why (matches check_no_forward_leaf_calls.py's
    own convention).

    `is_python=True` derives `lines`' multi-line-string spans from `tokenize`
    (T-1538) rather than skipping triple-quote tracking altogether; may raise
    `PythonTokenizeError` if `lines` cannot be tokenized as Python."""
    python_string_lines = _python_multiline_string_lines(lines) if is_python else None
    hits: list[tuple[int, int, list[str]]] = []
    for start, end, text in _iter_blocks(lines, python_string_lines):
        labels = SEVERITY_LABEL_PATTERN.findall(text)
        if not labels:
            continue
        if RESOLUTION_MARKER_PATTERN.search(text):
            continue
        hits.append((start, end, labels))
    return hits


def find_uncited_defect_citations(path: str) -> list[tuple[int, int, list[str]]]:
    """File-backed form of scan_text: reads `path` and applies the same rule.
    `.py` paths get tokenize-derived string-span tracking (T-1538); `.cpp`/`.h`
    paths get none -- C++ has no triple-quoted strings, so none is correct."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return scan_text(lines, is_python=path.endswith(".py"))


def _glob_files(globs: tuple[str, ...], repo_root: str) -> list[str]:
    out: list[str] = []
    for g in globs:
        out.extend(glob.glob(os.path.join(repo_root, g), recursive=True))
    return sorted(out)


def scan_files(file_paths: list[str], repo_root: str = _REPO_ROOT) -> list[str]:
    """Scans every file in `file_paths` (absolute or repo-root-relative).
    Returns one formatted failure string per uncited-defect-citation block,
    empty if clean.

    A `.py` file `tokenize` cannot parse is reported as its own failure line
    (T-1538) rather than skipped or treated as clean -- a file this check
    cannot parse is a file it cannot vouch for, the same posture as a missing
    file below."""
    failures: list[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
        rel = os.path.relpath(abs_p, repo_root)
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        try:
            hits = find_uncited_defect_citations(abs_p)
        except PythonTokenizeError as exc:
            failures.append(f"{rel}: {exc}")
            continue
        for start, end, labels in hits:
            failures.append(
                f"{rel}:{start}-{end}: cites {', '.join(sorted(set(labels)))} with no "
                f"resolution marker ('closed' or a cited T-<id>) in the same comment block"
            )
    return failures


def main(globs: tuple[str, ...] = _DEFAULT_TEST_GLOBS, repo_root: str = _REPO_ROOT) -> int:
    files = _glob_files(globs, repo_root)
    failures = scan_files(files, repo_root)
    if failures:
        print("check_present_tense_defect_comments.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"check_present_tense_defect_comments.py: OK -- {len(files)} test file(s) scanned, "
        f"every severity-label citation carries a resolution marker"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
