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
header that cites several severity labels as an index (no defect language at all)
also lacks a resolution marker and is flagged by this rule as written, when the
labels carry a digit the way this codebase's severity-label citations normally
do. This paragraph's own example is deliberately written without digits instead
-- "the remediation red suite for Critical A, Critical B, and Significant C",
letter-suffixed per this codebase's own "Significant B"-style coverage-tag
convention -- so `SEVERITY_LABEL_PATTERN`'s digit requirement (defined below)
does not match it and this module's own default-glob self-scan does not flag its
own text; T-1545's paragraph-granular fix (below) would otherwise have turned a
digit-labeled version of this same sentence into a live, uncited self-scan hit,
where before that fix a single docstring-wide block masked it behind an
unrelated "closed" citation elsewhere in the same then-one-block docstring
(28aa351 Significant 2/prior Significant 3). Narrowing the rule to require a
negation word ("no"/"never"/"not") alongside the label would suppress that false
positive -- but it also suppresses a REAL, independently found instance of this
exact defect class (see the module's own casebook/build-log citation for the
discovered site), which contains no negation word at all ("... reads unmapped
heap memory ..." stated as present fact, no "no"/"never"). Between a rule that
over-flags a benign header and one that misses a real defect, this module keeps
the wider rule and reports both outcomes plainly rather than tuning against
either.

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

F-STRING TOKEN COVERAGE AND PARAGRAPH-GRANULAR STRING BLOCKS (T-1545). T-1538's
`_python_multiline_string_lines` collected only `tokenize.STRING` spans. Since
Python 3.12 a multi-line f-string is not emitted as a `STRING` token at all --
it is `FSTRING_START` / `FSTRING_MIDDLE` (itself possibly multi-line) /
`FSTRING_END` -- so a multi-line f-string's physical lines fell through to
`_line_kind` uncovered: its opening line starts with `f`, not a quote
character, so it (and every following line, none of which starts with a
quote either) classified `OTHER` and no block formed at all, silently
admitting an uncited citation inside it (Poirot
884ee74-t1485-present-tense-gate-blocking-review-2026-07-31.md Significant
1, confirmed 28aa351 Significant 1). `_python_multiline_string_lines` now
tracks the span from each `FSTRING_START` to its matching `FSTRING_END`
(depth-counted, so a nested same-quote f-string does not close the span
early) as well as ordinary multi-line `STRING` tokens, on interpreters that
expose the f-string token family; on an interpreter without it (Python <
3.12), only ordinary `STRING` tokens are tracked, which is the complete
Python string-token vocabulary on that interpreter.

Separately: `_iter_blocks` merged every physical line the tokenizer placed
inside a multi-line STRING (or now f-string) span into one block regardless
of blank lines within it, because forced-STRING lines bypassed `_line_kind`
--  the classifier that already treats a blank line as a break for `//`
comments and bare-string continuations -- entirely. A resolution marker
anywhere in a multi-line docstring or module-level string therefore silently
satisfied a citation anywhere else in the same string, however far apart
and however many blank lines separated them (confirmed 28aa351 Significant
2/prior Significant 3: this module's own docstring was one such string at
that commit, and one `closed` inside it cleared six unrelated severity-label
citations -- T-1559 retired the two present-tense clauses that stood here,
both false by the time they were read: the line count had moved and the
docstring had long since been many blocks rather than one). A blank line now
breaks a block the same way inside a
tokenizer-forced string span as outside one, so a multi-line string is
scanned at paragraph granularity -- one block per blank-line-delimited
paragraph -- consistent with how `//` comment runs and bare C string-literal
continuations already break at a blank or non-continuing line.

LITERAL-IDENTITY-AWARE BLOCK BOUNDARIES (T-1553). T-1545's fix widened the
line pool `_python_multiline_string_lines` returns to cover f-strings, but
still flattened it to one `frozenset[int]` of physical line numbers with no
record of which literal each line belongs to. `_iter_blocks` merges by
adjacency alone, so two DISTINCT multi-line string literals with no blank
line between them -- an f-string ending on one line immediately followed by a
plain triple-quoted string starting on the next, exactly the shape T-1545's
own widened pool newly admits -- read as one block, and a resolution marker
in the first silently satisfied an uncited citation in the second (Poirot
d2a7eed-t1547-present-tense-confirmation-2026-07-31.md Significant 1; latent
on plain-plain adjacent literals since before T-1545 too, per the same
finding). `_python_multiline_string_lines` now returns each literal's own
`(start, end)` line range rather than a flattened set of line numbers, and
`_iter_blocks` closes the current block whenever a new literal's start line
is reached, the same way a blank line already closes it.

TOTAL BOUNDARIES: SPAN MEMBERSHIP, NOT JUST SPAN STARTS (T-1557). The break
above is directional -- it fires when a span BEGINS and not when one ENDS,
because a single-line literal produces no span and so contributes no start
line, while still classifying STRING through `_line_kind`. A multi-line
literal immediately followed by a single-line string literal therefore still
read as ONE block, and a resolution marker in the first silently satisfied an
uncited citation in the second: the same defect class as T-1553, at the one
boundary that remedy did not reach (Poirot
b2b00b4-t1556-adjacency-fold-confirmation-2026-07-31.md Significant 1, found
latent -- zero live instances, and the reverse ordering was already caught,
which is what exposed the asymmetry). `_iter_blocks` now carries whether the
previous line was inside any span and breaks when that changes, so the
boundary holds in both directions. Two adjacent literals with no blank line
between them get two blocks whenever at least one of them is multi-line.
Two SINGLE-line literals still merge, and so do two quoted-`//` lines: both
sit wholly outside every span, both are deliberate (the bare C string-literal
continuation rule and `_PY_QUOTED_CPP_COMMENT_PATTERN` respectively), and
splitting them would fire false positives across the fixture generators.
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


# On Python 3.12+ a multi-line f-string is not a single tokenize.STRING token
# -- it is FSTRING_START / FSTRING_MIDDLE / FSTRING_END, which do not exist as
# tokenize module attributes on earlier interpreters (T-1545). Resolved once,
# at import time: None on an interpreter that has no f-string token family, in
# which case _python_multiline_string_lines tracks only ordinary STRING
# tokens -- the complete Python string-token vocabulary on that interpreter.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


def _python_multiline_string_lines(lines: list[str]) -> frozenset[tuple[int, int]]:
    """Each multi-line STRING token's own `(start, end)` 1-based physical line
    range, or (T-1545) a multi-line f-string's own FSTRING_START..FSTRING_END
    range, per Python's own tokenizer -- ground truth for a `.py` file's
    triple-quoted-string spans (T-1538). One tuple per literal (T-1553) --
    NOT flattened into a single set of line numbers -- so a caller can tell
    where one literal ends and the next begins even when two literals are
    physically adjacent with no blank line between them.

    Replaces this module's earlier per-physical-line triple-quote-PARITY
    heuristic (T-1485), which carried no whole-file notion of whether a
    string was already open: any ordinary line outside a string whose
    triple-quote count was odd -- a comment merely naming the delimiter, or
    this module's own former `_TRIPLE_QUOTE_MARKERS` declaration -- flipped
    the tracked polarity for the remainder of the file, so every later
    docstring's closing line read as an opener and every opening line as a
    closer. `tokenize` has no such state to desynchronise: each STRING
    token's own `start`/`end` line numbers come from the interpreter's own
    lexer, not from re-counting delimiters on each line in isolation.

    A multi-line f-string is not a STRING token at all on Python 3.12+ (T-1545,
    Poirot 28aa351-t1544-present-tense-confirmation-2026-07-31.md Significant
    1): it is FSTRING_START, one or more FSTRING_MIDDLE segments (themselves
    possibly multi-line, and interleaved with the tokens of any interpolated
    expression), and FSTRING_END. This function tracks the whole span from
    each FSTRING_START to its matching FSTRING_END -- depth-counted so a
    nested same-quote f-string's own START/END pair does not close the outer
    span early -- rather than inspecting the interior tokens' own types,
    because an interpolated expression's tokens (a NAME, an OP) are still
    physically inside the f-string's source text and must not be read as
    code lines outside it.

    Flattening this into one `frozenset[int]` (T-1538 through T-1545) erased
    which literal a line belonged to, so two distinct literals with no blank
    line between them read as one merged block once both were in the pool
    (T-1553, Poirot d2a7eed-t1547-present-tense-confirmation-2026-07-31.md
    Significant 1): keeping each literal's own range is what lets a caller
    close a block at a literal's start line, the same way it already closes
    one at a blank line."""
    text = "".join(lines)
    result: set[tuple[int, int]] = set()
    try:
        fstring_depth = 0
        fstring_open_line: int | None = None
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if _FSTRING_START is not None and tok.type == _FSTRING_START:
                if fstring_depth == 0:
                    fstring_open_line = tok.start[0]
                fstring_depth += 1
                continue
            if _FSTRING_END is not None and tok.type == _FSTRING_END:
                if fstring_depth > 0:
                    fstring_depth -= 1
                    if fstring_depth == 0 and fstring_open_line is not None:
                        if tok.end[0] > fstring_open_line:
                            result.add((fstring_open_line, tok.end[0]))
                        fstring_open_line = None
                continue
            if fstring_depth > 0:
                # Inside an f-string span: every token here (an interpolated
                # expression's NAME/OP tokens, an FSTRING_MIDDLE segment) is
                # physically part of the f-string's own source text, already
                # covered when the matching FSTRING_END closes the span above.
                continue
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                result.add((tok.start[0], tok.end[0]))
    except (tokenize.TokenError, SyntaxError, IndentationError, UnicodeDecodeError) as exc:
        raise PythonTokenizeError(f"could not tokenize as Python: {exc}") from exc
    return frozenset(result)


def _line_kind(line: str) -> str:
    """Classifies one physical line for block-grouping purposes. `_iter_blocks`
    calls this for every line except a NON-BLANK one it already knows, from a
    `.py` file's tokenize-derived multi-line-string spans, to be inside a
    Python string literal -- such a line is forced to STRING without
    consulting this function at all (see `_iter_blocks`). A BLANK line inside
    such a span is still routed through this function like any other line, so
    it classifies OTHER and breaks the block the same as a blank line
    anywhere else (T-1545; see `_iter_blocks`'s own docstring for why). A line
    consulted here is part of a textual block if it is a `//` comment (any
    indentation), a Python-quoted `//` comment line
    (`_PY_QUOTED_CPP_COMMENT_PATTERN`), or a
    bare C string-literal continuation (the shape this codebase's multi-line
    CHECK_MSG messages take: one quoted string literal per physical line).
    Anything else -- code, blank lines -- is a break between blocks, so a
    resolution marker written for one comment can never silently satisfy a
    citation in an unrelated, later block.

    This function is never consulted for a line a `.py` file's
    tokenize-derived multi-line-string span forces to STRING (T-1545) -- but
    `_iter_blocks` applies the same blank-line break to those lines directly,
    rather than merging every line inside a multi-line string into one block
    regardless of blank lines within it (the shape prior Significant 3 /
    28aa351 Significant 2 named: a resolution marker anywhere in a multi-line
    docstring or module-level string silently satisfied a citation anywhere
    else in the same string), and since T-1557 it breaks on a change of span
    membership as well, which closes the exit boundary the start-line break
    left open.

    The invariant this docstring states -- a blank line breaks a block, and a
    resolution marker in one block cannot satisfy a citation in another -- is
    asserted here in prose and PINNED in
    `test_check_present_tense_defect_comments.py` by the span-exit and
    blank-line-control cells T-1557 added. That pairing is deliberate: this
    paragraph carried the same sentence through four review rounds while being
    falsifiable in each, because a claim in a docstring cannot fail a build
    (Poirot b2b00b4-t1556-adjacency-fold-confirmation-2026-07-31.md, standing
    routing). Read the cells as the statement of record; if this paragraph and
    those cells ever disagree, the cells are right."""
    stripped = line.strip()
    if stripped.startswith("//"):
        return "COMMENT"
    if _PY_QUOTED_CPP_COMMENT_PATTERN.match(stripped):
        return "COMMENT"
    if stripped.startswith('"'):
        return "STRING"
    return "OTHER"


def _iter_blocks(
    lines: list[str], python_string_spans: frozenset[tuple[int, int]] | None = None
) -> list[tuple[int, int, str]]:
    """Every maximal run of same-kind (COMMENT or STRING) contiguous lines, as
    (1-based start line, 1-based end line, merged text). Lines are joined with a
    space so a citation split across a line break -- this codebase's own fixed
    text does this ("...(Significant " / "5, closed by...)") -- is still found as
    one continuous match against the merged text, never only against one line.

    `python_string_spans`, when given, is the set of each individual literal's
    own 1-based `(start, end)` physical line range that a `.py` file's
    tokenizer places inside a multi-line STRING or f-string token
    (`_python_multiline_string_lines`) -- NOT a single flattened set of line
    numbers (T-1553): keeping each literal's own range is what lets this
    function tell a line that starts a NEW literal from a line that continues
    the one before it, even when the two are physically adjacent with no
    blank line between them. A non-blank line inside any span is forced to
    kind STRING regardless of what character it starts with, which is what
    keeps a multi-line docstring paragraph together as one block the same way
    the `//` and bare-C-string rules already keep their own multi-line shapes
    together -- without re-deriving "is a string open" from each line's own
    content the way T-1485's triple-quote-parity tracker did (see
    `_python_multiline_string_lines`'s docstring for why that desynchronised).
    `.cpp`/`.h` inputs pass `None`: C++ has no triple-quoted strings, so no
    line in a C++ file should ever force this state, and none does.

    A BLANK line inside a tokenizer-forced string span is NOT forced to
    STRING (T-1545, Poirot
    28aa351-t1544-present-tense-confirmation-2026-07-31.md Significant 2):
    it classifies via `_line_kind` like any other line, which returns OTHER
    for it, breaking the block. Without this, every physical line inside a
    multi-line docstring or module-level string -- however many blank lines
    separate its paragraphs -- merged into ONE block, so a resolution marker
    anywhere in the string silently satisfied a citation anywhere else in
    it. This module's own module docstring was exactly such a span before
    this fix: one `closed` inside it cleared six unrelated severity-label
    citations, which is precisely what `_line_kind`'s own docstring already
    states must not happen. Forcing only non-blank lines gives a multi-line
    string paragraph-granular blocks -- one block per run of non-blank
    lines -- the same shape blank lines already produce for `//` comment
    runs and bare C string-literal continuations.

    A literal's OWN start line also closes the current block, even when the
    previous line was already kind STRING (T-1553, Poirot
    d2a7eed-t1547-present-tense-confirmation-2026-07-31.md Significant 1):
    without this, two distinct literals with no blank line between them --
    an f-string ending on one line immediately followed by a plain
    triple-quoted string starting on the next -- read as a single block, so a
    resolution marker in the first silently satisfied an uncited citation in
    the second.

    Span MEMBERSHIP is part of block identity too, so a line inside any span
    never merges with a line outside every span (T-1557, Poirot
    b2b00b4-t1556-adjacency-fold-confirmation-2026-07-31.md Significant 1).
    The literal-start break above is directional: it closes a block when a
    span BEGINS, and left the exit unguarded, because a single-line literal
    produces no span and therefore no start line while still classifying
    STRING through `_line_kind`. A multi-line literal immediately followed by
    a single-line string literal therefore read as one block -- the same
    defect class as T-1553, at the boundary that remedy did not reach. The
    two shapes that merge ACROSS literals by design still do: single-line to
    single-line (the bare C string-literal continuation rule) and quoted-`//`
    to quoted-`//` (`_PY_QUOTED_CPP_COMMENT_PATTERN`), both of which sit
    wholly outside every span and so compare equal. That is why this break
    keys on span membership rather than on giving every literal a start line,
    which would split both and fire false positives across the fixture
    generators.

    Where two literals share one physical line -- implicit or `+`
    concatenation whose first literal ends on the same line the second
    begins -- that line is both inside the first literal's span and a start
    line of the second, so the literal-start break fires INSIDE the first
    literal and can separate its own citation from its own resolution marker
    (T-1558, same casebook Minor 2). This is a false positive rather than a
    merge, and it is the deliberate choice: a line-granular scanner cannot
    decompose two literals sharing a physical line, so it must err one way,
    and a false positive on a blocking gate is loud and self-announcing where
    the class this module exists to prevent is silent. Reach is zero across
    the tree measured; if it ever fires, the shape is correct code and the
    fix is a line break, not a suppression."""
    python_string_lines: frozenset[int] = frozenset()
    literal_start_lines: frozenset[int] = frozenset()
    if python_string_spans is not None:
        python_string_lines = frozenset(
            n for span_start, span_end in python_string_spans for n in range(span_start, span_end + 1)
        )
        literal_start_lines = frozenset(span_start for span_start, _span_end in python_string_spans)

    blocks: list[tuple[int, int, str]] = []
    start: int | None = None
    kind: str | None = None
    buf: list[str] = []

    def _close() -> None:
        nonlocal start, kind, buf
        if kind in ("COMMENT", "STRING") and start is not None:
            blocks.append((start, start + len(buf) - 1, " ".join(buf)))
        start, kind, buf = None, None, []

    prev_in_span = False
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        in_span = i in python_string_lines

        if in_span and stripped:
            this_kind = "STRING"
        else:
            this_kind = _line_kind(line)

        if (
            this_kind in ("COMMENT", "STRING")
            and this_kind == kind
            and i not in literal_start_lines
            and in_span == prev_in_span
        ):
            buf.append(stripped)
            prev_in_span = in_span
            continue
        _close()
        if this_kind in ("COMMENT", "STRING"):
            start, kind, buf = i, this_kind, [stripped]
        prev_in_span = in_span
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
    python_string_spans = _python_multiline_string_lines(lines) if is_python else None
    hits: list[tuple[int, int, list[str]]] = []
    for start, end, text in _iter_blocks(lines, python_string_spans):
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
