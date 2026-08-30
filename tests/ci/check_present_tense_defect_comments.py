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

A SECOND, INDEPENDENT DEFECT CLASS: A "NOT YET BUILT" CLAIM SURVIVING ITS OWN
BUILD ROUND (T-2385, fold round 43, closing T-2382 finding S3). A test-suite
docstring or comment can assert, in present tense, that a surface is unbuilt
and its cells `xfail`-marked ("NONE OF THIS IS BUILT YET", "scan_build_
output.py does not read archives at all yet", "the still-unbuilt half") while
the build round that lands the surface removes every `xfail` marker the
prose was describing and leaves the prose itself untouched. This is NOT the
class `SEVERITY_LABEL_PATTERN`/`RESOLUTION_MARKER_PATTERN` above catch: that
pair requires a specific citation shape ("Critical N"/"Significant N"/etc.)
this prose never uses, so a docstring saying "not yet built" is invisible to
it by construction -- confirmed twice on the same red suite
(`tests/t2296-fp-free-open-red-suite/`), one round apart: D-SLM5018 S1/S3
filed it first, and it recurred at scale (nine sites across two files) in
the very build round that closed it (T-2382 finding S3).

RULE COVERAGE, MEASURED. `_UNBUILT_CLAIM_PATTERN` (below) is a set of the
exact phrasings this codebase has used for "not built yet," recovered from
the real T-2382 finding rather than authored in the abstract: a bare
`not yet built` (the `pytest.fail` guards this suite's own cells raised
before the production module existed), `NONE OF THIS IS BUILT`, `still
globs` (naming the pre-archive directory-glob path by name), `genuinely
red-unimplemented`, a bare `unbuilt` (catching `still-unbuilt`,
`(unbuilt -- xfail)`, and `unbuilt archive-composition`), `does not read
archives`, `coincidentally the correct exit code`/`coincidentally exits N`
(the "right answer for the wrong reason" shape), and a future-tense
`will ... once ... lands/ships/builds` construction (T-2382's own finding
that a docstring describing a landed interface in future tense is the same
defect wearing different grammar). Validated (T-2385, this session) against
the real historical population at commit `7a77a07^{tree}`'s own two files,
`test_archive_composition.py` and `test_archive_gate.py`: both flag before
the fix (nine sites, D-SLM5018-shaped), and both are clean after it --
`test_check_present_tense_unbuilt_class.py` (`tests/t2296-fp-free-open-red-
suite/`, this round's own test pin, since this module's own test file is
outside this ticket's writable scope) replays both states, now from
fixture files vendored in the pin's own directory rather than `git show`
(see "POPULATION RECOVERY IS VENDORED, NOT LIVE `git show`" below).

WHITESPACE-TOLERANT MATCHING (T-2387, Poirot
8788b01-t2386-archive-gate-confirmation.md N2). Every alternative above
that spans more than one word originally spelled its internal spaces as a
literal `" "`, and this suite's own docstrings are hard-wrapped near
column 72 -- so a trigger phrase that happens to straddle a line break was
invisible to the pattern. Measured against the real historical population
at `7a77a07^{tree}`: the literal-space pattern matched 12 of the real
occurrences in `test_archive_composition.py`/`test_archive_gate.py`; two
more -- `does not read` / `archives` and `coincidentally` / `the correct
exit code`, each pair split across a line break -- were missed for no
reason other than the line wrap, reaching 14 once each internal literal
space is replaced with `\\s+` (already the case for the future-tense
alternative's own `[^.]{0,N}` gaps, which admit newlines). Neither miss
changed this check's own verdict on either
historical file (both still flagged, on a different phrase), but the
pattern's OWN rule-coverage cells fed it single-line synthetic strings and
so could not see the gap -- exactly the commissioning failure
`StandardsDocument.md` Sec5.4 names: a maker-authored control the real
input can, and did, fail to reproduce. `_UNBUILT_CLAIM_PATTERN` below is
now whitespace-tolerant throughout, and the pin adds one wrapped-fixture
cell per multi-word alternative alongside the existing single-line ones.

POPULATION RECOVERY IS VENDORED, NOT LIVE `git show` (T-2387, Poirot
8788b01-t2386-archive-gate-confirmation.md N1). The pin's two
historical-population cells used to recover `7a77a07`'s pre-fix text with
a live `git show <sha>` subprocess call, skipping (not failing) when that
command could not resolve the commit. Measured: a real
`git clone --depth 1` of this repository -- the checkout every job in
`.github/workflows/tests.yml` performs, none of them setting
`fetch-depth` -- resolves that `git show` with exit 128, so both cells
skipped and the pin reported green in the exact environment that gates it.
The pin now reads the pre-fix text of both files from two fixture files
vendored in its own directory
(`present_tense_unbuilt_class_historical_fixtures/`), recovered once by
this fix and committed alongside it; recovery no longer depends on git
history being present at all, in CI or anywhere else.

INPUT COVERAGE, DELIBERATELY NARROWER THAN THE SEVERITY-LABEL CHECK ABOVE.
`_UNBUILT_CLAIM_GLOBS` scans only `tests/t2296-fp-free-open-red-suite/`,
not the whole `tests/` tree `_DEFAULT_TEST_GLOBS` covers. Measured (T-2385):
the identical trigger phrases scanned against the whole `tests/` tree hit
two unrelated, correctly-worded files
(`tests/ci/check_checked_chain_funnel_position_cap_not_a_stub.py` and its
own test file) that quote a PAST "still-unbuilt"-shaped defect as the
worked example inside a check built to detect that exact class on a
DIFFERENT function -- accurate historical narration, not a live claim about
current build status, and outside this ticket's own writable scope to
correct even if it were the same defect. Widening this check's own input
coverage to the whole tree the day a real instance is found outside this
suite is the same discipline `_DEFAULT_TEST_GLOBS` above was widened under
(T-1485 through T-1564); until then, scoping to the two files this defect
has recurred in twice keeps the check honest about what it has actually
been validated against.

THE MARKER, AND WHY FILE-GRANULARITY RATHER THAN BLOCK-GRANULARITY. The
severity-label check above requires the citation and its resolution marker
in the SAME comment/docstring block, because a citation and an unrelated
"closed" elsewhere in the file must not silently satisfy each other
(T-1545's own paragraph-granularity fix exists for exactly this reason).
This check's own marker is checked at FILE granularity on purpose: the
defect this check pins is not "a citation and its marker disagree within
one block," it is "this file's prose describes a whole surface as
unbuilt-and-xfailed, and the file's own marker count is zero" -- a fact
about the file as a whole, which is exactly what T-2382 found (zero
`xfail(strict=True)` markers survive in either file, confirmed by direct
count, while the prose describing them does).

WHAT COUNTS AS THE MARKER (T-2387, Poirot
8788b01-t2386-archive-gate-confirmation.md N3/N4). File-granularity means
ONE thing has to be true of the file, not that ANY `@pytest.mark.xfail`
anywhere clears ANY trigger phrase anywhere -- the original form did the
latter, and it was wrong twice over. First (N3): `test_check_fp_free_scan.
py` carried a stale `# ... -- NOT YET BUILT` import-guard comment (line
245, predating the archive round) that was cleared by three markers whose
own `reason=` text is D-SLM5009's and T-2347's genuinely open questions --
having nothing to do with whether that import is built. The file "has not
had its markers stripped out from under its prose" was true of the file's
own markers and false of that comment: a marker with no relationship to a
claim should not launder it. Second (N4): a live marker was found by a bare
`_XFAIL_MARKER_PATTERN.search(text)` over the WHOLE raw file text, which
cannot tell a real `@pytest.mark.xfail(...)` decorator from a Python string
literal that merely spells the same characters -- this module's own pin
file (`test_check_present_tense_unbuilt_class.py`) carries every trigger
phrase verbatim as fixture DATA and was saved from flagging only because
one of its fixture strings happens to also spell `@pytest.mark.xfail(`
literally; a checker that cannot be caught by its own rule for an
accidental reason is the shape of the defect it hunts.

Both are closed the same way: `_xfail_decorator_source_texts` finds every
REAL decorator via `ast.parse` (a string that merely spells decorator
syntax is not a decorator node and is never returned), and
`find_stale_unbuilt_claim` requires that at least one such decorator's own
source text -- the whole `@pytest.mark.xfail(...)` call, `reason=` included
-- ITSELF matches `_UNBUILT_CLAIM_PATTERN` before treating the file as
covered. A marker about an unrelated open question no longer launders a
claim it says nothing about, and a fixture string that only spells marker
syntax is never mistaken for one. File granularity is preserved -- this is
still "does the file, as a whole, carry a marker for this claim," not a
per-block or per-function scope -- but the marker itself is real and
on-topic rather than merely present.

THE PIN'S OWN FILE IS EXCLUDED BY NAME, NOT BY ACCIDENT (T-2387, N4). Even
with the tightened marker rule above, this module's own pin has no real
`@pytest.mark.xfail` decorators of its own (it decorates nothing; it
constructs fixture text), so its fixture strings would flag under the
tightened rule exactly as they would have flagged the moment its one
accidental marker-spelling string was refactored away. `_UNBUILT_CLAIM_
GLOBS`'s own matches are filtered against `_UNBUILT_CLAIM_EXCLUDE_
BASENAMES` in `main()` before scanning -- the same "known false positive,
scoped out by name and pinned by a test proving the exclusion is load-
bearing" shape the pin's own `_KNOWN_FALSE_POSITIVE_FILES` uses for the two
`tests/ci/` files named above -- so the exclusion no longer depends on what
any fixture string happens to spell.
"""
from __future__ import annotations

import ast
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


# T-2385 (fold round 43, closing T-2382 finding S3): the exact phrasings
# this codebase has used to claim a surface is "not built yet," recovered
# from the real historical population rather than authored in the
# abstract -- see the module docstring's own "A SECOND, INDEPENDENT DEFECT
# CLASS" section for the provenance of each alternative and the validation
# against commit 7a77a07's own two files. T-2387 (Poirot
# 8788b01-t2386-archive-gate-confirmation.md N2): every internal literal
# space is `\s+`, not `" "` -- this suite's own prose is hard-wrapped near
# column 72, and a phrase that happens to straddle a line break is real
# text on the historical population, not a hypothetical (see the module
# docstring's own "WHITESPACE-TOLERANT MATCHING" section).
_UNBUILT_CLAIM_PATTERN = re.compile(
    r"\bnot\s+yet\s+built\b"
    r"|NONE\s+OF\s+THIS\s+IS\s+BUILT"
    r"|\bstill\s+globs\b"
    r"|genuinely\s+red-unimplemented"
    r"|\bunbuilt\b"
    r"|does\s+not\s+read\s+archives"
    r"|coincidentally\s+(?:the\s+correct\s+exit\s+code|exits\s+\d+)"
    r"|\bwill\b[^.]{0,100}\bonce\b[^.]{0,60}\b(?:lands?|ships?|builds?)\b",
    re.IGNORECASE,
)

# A live, real xfail marker -- either decorator spelling this codebase uses
# (`@pytest.mark.xfail(...)` or the bare attribute reference). Matched
# against one AST decorator's own source text at a time
# (`_xfail_decorator_source_texts`), never against raw whole-file text --
# see the module docstring's own "WHAT COUNTS AS THE MARKER" section for
# why (T-2387, N3/N4).
_XFAIL_MARKER_PATTERN = re.compile(r"pytest\.mark\.xfail\b")

# T-2385: scoped to the one suite this defect class has recurred in twice
# (D-SLM5018, then T-2382 finding S3) -- see the module docstring's own
# "INPUT COVERAGE, DELIBERATELY NARROWER" section for the measured reason
# this is not yet `tests/**/*.py`.
_UNBUILT_CLAIM_GLOBS = (
    "tests/t2296-fp-free-open-red-suite/**/*.py",
)

# T-2387 (Poirot 8788b01-t2386-archive-gate-confirmation.md N4): this
# module's own pin file lives inside `_UNBUILT_CLAIM_GLOBS`'s own scan
# surface and carries every `_UNBUILT_CLAIM_PATTERN` alternative verbatim
# as fixture DATA -- see the module docstring's own "THE PIN'S OWN FILE IS
# EXCLUDED BY NAME" section. Matched by basename, not full path, since
# there is exactly one file to exclude and no risk of collision within the
# scanned suite.
_UNBUILT_CLAIM_EXCLUDE_BASENAMES = frozenset({
    "test_check_present_tense_unbuilt_class.py",
})



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
    `test_check_present_tense_defect_comments.py`, by these cells and no
    others (T-1564, Poirot
    83260be-t1560-span-membership-boundary-confirmation-2026-07-31.md Minor 3,
    which found this paragraph citing a cell that cannot fail):

        blank line breaks inside a span
            `test_a_resolution_marker_in_one_paragraph_of_a_multiline_string
             _does_not_clear_an_unrelated_paragraph`
        block boundary holds leaving a span
            `test_a_multiline_literal_does_not_merge_with_a_following
             _single_line_literal`
        membership, not per-literal starts, is the right key
            `test_two_single_line_string_literals_still_merge_by_design`

    Each was verified to go red under a mutation that destroys the rule it
    names. The blank-line control beside the span-exit primary is NOT in this
    list: its blank line sits outside the span, so it survives every mutation
    of the blank-line rule and pins nothing. It is kept as a regression guard
    on the primary's own shape and its docstring says so.

    This pairing is deliberate: this paragraph carried the same sentence
    through four review rounds while being falsifiable in each, because a
    claim in a docstring cannot fail a build (Poirot
    b2b00b4-t1556-adjacency-fold-confirmation-2026-07-31.md, standing
    routing). Read the cells as the statement of record; if this paragraph and
    those cells ever disagree, the cells are right -- which is exactly how the
    error this paragraph previously contained was found."""
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
    shapes that merge ACROSS literals by design still do: single-line to
    single-line (the bare C string-literal continuation rule), quoted-`//` to
    quoted-`//` (`_PY_QUOTED_CPP_COMMENT_PATTERN`), and bare `//` to
    quoted-`//` in either order -- both classify COMMENT and the pairing is
    reachable in `.cpp`/`.h` input, though its measured reach in this tree is
    zero (T-1564, same casebook Observation). All of them sit wholly outside
    every span and so compare equal. That is why this break keys on span
    membership rather than on giving every literal a start line, which would
    split them and fire false positives across the fixture generators.

    Where two literals SHARE one physical line -- implicit or `+`
    concatenation whose first literal ends on the same line the second
    begins -- this scanner is line-granular and there is no line boundary
    between them to break at, so it cannot decompose them at all. What it
    does instead depends entirely on whether the SECOND literal is multi-line,
    and the four orderings do not behave alike (T-1562, Poirot
    83260be-t1560-span-membership-boundary-confirmation-2026-07-31.md
    Significant 1, which found the earlier text here claiming they did):

        multi-line -> multi-line   SPLITS   (loud: the second literal's own
                                             start line falls on the shared
                                             line, so the break fires INSIDE
                                             the first literal and separates
                                             its citation from its marker)
        multi-line -> single-line  MERGES   (silent)
        single-line -> multi-line  MERGES   (silent)
        single-line -> single-line MERGES   (silent)

    Only the first errs loud. **The other three are a genuine blind spot of
    this module's own defect class** -- a resolution marker in one authored
    literal satisfying an uncited citation in a different one -- and they are
    silent, which is the failure mode this module exists to prevent. They are
    not a deliberate trade-off and must not be described as one. The earlier
    text here asserted the loud behaviour for all four, which read as
    reassurance in the direction of the module being safer than it is.

    Why they are not simply fixed: a single-line literal produces no tokenize
    span and therefore no start line, so at line granularity there is nothing
    to key a break on. Closing them means decomposing a physical line into its
    constituent literals -- a sub-line rewrite of `_iter_blocks`'s whole
    contract, and a design call rather than a text fix.

    Reach, measured over the scanned tree: shared-physical-line literal pairs
    are common (high hundreds; the exact total depends on how adjacent
    literals are counted and two independent measurements disagreed on it), but
    **0 of them have a multi-line first literal** -- which both measurements
    agree on and which is the load-bearing half. So every live instance takes
    a merging path, and none currently manifests a hit (the gate is green).
    No total is quoted here on purpose: D-SLM550 retired a self-describing
    count from this same docstring for drifting, and a number two people
    derive differently is the same defect wearing a different hat. All four
    orderings are pinned by cells, so this paragraph is checkable rather than
    asserted.

    Separately, the span-exit break introduced above fires between a
    multi-line literal and a single-line literal on the FOLLOWING line, which
    is the defect shape when they are two authored messages and a **false
    positive** when they are one authored message written as implicit
    concatenation (T-1563, same casebook Minor 2). The `+` spelling was
    already flagged before this change; only the implicit spelling is new.
    Reach is zero and a false positive on a blocking gate is self-announcing,
    so it is accepted and named here rather than suppressed -- but it is a
    real behaviour change against correct, fully-cited code, and if it fires
    the fix is a line break in the source, not a suppression."""
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


def _xfail_decorator_source_texts(text: str) -> list[str]:
    """Every real `@pytest.mark.xfail(...)`-shaped decorator's own full
    source text, one entry per decorator, found anywhere in `text` via
    Python's own `ast` parser -- never a regex re-derivation of decorator
    syntax. A decorator is included only when it is an actual AST decorator
    node attached to a function, async function, or class; a string literal
    that merely SPELLS `@pytest.mark.xfail(` (this suite's own fixture data
    for the rule-coverage cells above) is not a decorator node and is never
    included (T-2387, Poirot 8788b01-t2386-archive-gate-confirmation.md N4
    -- see the module docstring's own "WHAT COUNTS AS THE MARKER" section).
    Returns an empty list, rather than raising, when `text` cannot be
    parsed as Python: `find_stale_unbuilt_claim` treats that the
    conservative direction -- no marker can be verified, so none is
    credited, and a real trigger phrase in unparseable text still flags
    rather than being silently cleared."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    sources: list[str] = []
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", None) or []:
            func = dec.func if isinstance(dec, ast.Call) else dec
            func_src = ast.get_source_segment(text, func) or ""
            if _XFAIL_MARKER_PATTERN.search(func_src):
                sources.append(ast.get_source_segment(text, dec) or func_src)
    return sources


def _rendered_string_constant_match(text: str):
    """R9 (T-2404, D-SLM5209/D-SLM5211/D-SLM5215): the raw-text-only search
    above cannot see a claim split across two adjacent Python string
    literals -- e.g. `"...is not yet "` immediately followed by
    `"built..."`, where the raw file carries a closing quote, a newline,
    indentation, and an opening quote between "yet" and "built" (not
    whitespace, so `_UNBUILT_CLAIM_PATTERN`'s own `\\s+` does not bridge
    it), while Python's parser concatenates the two literals into one
    string at parse time and the RENDERED value matches the pattern
    cleanly. Parses `text` (`ast.parse`, the same machinery
    `_xfail_decorator_source_texts` already uses in this module) and runs
    `_UNBUILT_CLAIM_PATTERN` against the rendered `.value` of every
    `ast.Constant` string node, returning the first match object found or
    None. Returns None, rather than raising, when `text` cannot be parsed
    as Python -- the same conservative direction
    `_xfail_decorator_source_texts` takes; the raw-text surface above still
    sees a real trigger phrase in unparseable text even when this one
    cannot."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found = _UNBUILT_CLAIM_PATTERN.search(node.value)
            if found:
                return found
    return None


def find_stale_unbuilt_claim(text: str) -> str | None:
    """T-2385: the second, independent defect class this module checks for
    (see the module docstring's own "A SECOND, INDEPENDENT DEFECT CLASS"
    section) -- `text` (a whole file's own content) asserts, in one of the
    phrasings `_UNBUILT_CLAIM_PATTERN` names, that some surface is unbuilt,
    while carrying no live `@pytest.mark.xfail` marker whose OWN source
    text is itself about the same claim. Returns the matched phrase, or
    None if the file is clean by this rule. File-granularity by design --
    see the module docstring's own "THE MARKER, AND WHY FILE-GRANULARITY"
    and "WHAT COUNTS AS THE MARKER" sections (T-2387, N3/N4): this file
    either does or does not carry a covering marker, and that fact is
    checked once for the file, but "covering" now means the marker's own
    `reason=` text matches `_UNBUILT_CLAIM_PATTERN` too -- not merely that
    some unrelated marker exists anywhere in the file.

    R9 (T-2404): two scan surfaces are unioned -- the raw text search
    above, and a second pass (`_rendered_string_constant_match`) over
    every string constant's RENDERED value, which catches a claim split
    across adjacent literals that the raw-text surface cannot see. A file
    is flagged if either surface matches; neither surface's own scope
    (file-granularity, the covering-marker rule) changes."""
    match = _UNBUILT_CLAIM_PATTERN.search(text)
    if match is None:
        match = _rendered_string_constant_match(text)
    if match is None:
        return None
    covering = any(
        _UNBUILT_CLAIM_PATTERN.search(src) for src in _xfail_decorator_source_texts(text)
    )
    if covering:
        return None
    return match.group(0)


def find_stale_unbuilt_claims(path: str) -> str | None:
    """File-backed form of `find_stale_unbuilt_claim`: reads `path` and
    applies the same rule."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return find_stale_unbuilt_claim(text)


def scan_unbuilt_claims(file_paths: list[str], repo_root: str = _REPO_ROOT) -> list[str]:
    """Scans every file in `file_paths` for the stale-unbuilt-claim defect
    class (T-2385). Returns one formatted failure string per flagged file,
    empty if clean. A missing file is reported the same way `scan_files`
    reports one, for the same reason (a file this check cannot read is a
    file it cannot vouch for)."""
    failures: list[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
        rel = os.path.relpath(abs_p, repo_root)
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        phrase = find_stale_unbuilt_claims(abs_p)
        if phrase is not None:
            failures.append(
                f"{rel}: contains a not-yet-built-shaped claim ({phrase!r}) with no "
                f"live @pytest.mark.xfail marker in the file whose own reason covers it"
            )
    return failures


def main(globs: tuple[str, ...] = _DEFAULT_TEST_GLOBS, repo_root: str = _REPO_ROOT) -> int:
    files = _glob_files(globs, repo_root)
    failures = scan_files(files, repo_root)

    unbuilt_files = [
        p for p in _glob_files(_UNBUILT_CLAIM_GLOBS, repo_root)
        if os.path.basename(p) not in _UNBUILT_CLAIM_EXCLUDE_BASENAMES
    ]
    unbuilt_failures = scan_unbuilt_claims(unbuilt_files, repo_root)

    if failures or unbuilt_failures:
        print("check_present_tense_defect_comments.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        for f in unbuilt_failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"check_present_tense_defect_comments.py: OK -- {len(files)} test file(s) scanned, "
        f"every severity-label citation carries a resolution marker; "
        f"{len(unbuilt_files)} file(s) scanned for stale not-yet-built claims, none found"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
