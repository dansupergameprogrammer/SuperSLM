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
"""
from __future__ import annotations

import glob
import os
import re
import sys

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


def _line_kind(line: str) -> str:
    """Classifies one physical line for block-grouping purposes. A line is part
    of a textual block if it is a `//` comment (any indentation) or a bare C
    string-literal continuation (the shape this codebase's multi-line CHECK_MSG
    messages take: one quoted string literal per physical line). Anything else --
    code, blank lines -- is a break between blocks, so a resolution marker
    written for one comment can never silently satisfy a citation in an
    unrelated, later block."""
    stripped = line.strip()
    if stripped.startswith("//"):
        return "COMMENT"
    if stripped.startswith('"'):
        return "STRING"
    return "OTHER"


def _iter_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Every maximal run of same-kind (COMMENT or STRING) contiguous lines, as
    (1-based start line, 1-based end line, merged text). Lines are joined with a
    space so a citation split across a line break -- this codebase's own fixed
    text does this ("...(Significant " / "5, closed by...)") -- is still found as
    one continuous match against the merged text, never only against one line."""
    blocks: list[tuple[int, int, str]] = []
    start: int | None = None
    kind: str | None = None
    buf: list[str] = []
    for i, line in enumerate(lines, start=1):
        this_kind = _line_kind(line)
        if this_kind in ("COMMENT", "STRING") and this_kind == kind:
            buf.append(line.strip())
            continue
        if kind in ("COMMENT", "STRING") and start is not None:
            blocks.append((start, i - 1, " ".join(buf)))
        if this_kind in ("COMMENT", "STRING"):
            start, kind, buf = i, this_kind, [line.strip()]
        else:
            start, kind, buf = None, None, []
    if kind in ("COMMENT", "STRING") and start is not None:
        blocks.append((start, len(lines), " ".join(buf)))
    return blocks


def scan_text(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Every (start_line, end_line, [labels]) block among `lines` that cites at
    least one severity-finding label with no resolution marker anywhere in the
    same block. Driven directly from an in-memory line list -- used both by
    find_uncited_defect_citations (file-backed) and by the historical-population
    validation cells, which replay recovered git-blob text without writing it to
    disk first. A text scan, not a parsed AST -- see module docstring for why
    (matches check_no_forward_leaf_calls.py's own convention)."""
    hits: list[tuple[int, int, list[str]]] = []
    for start, end, text in _iter_blocks(lines):
        labels = SEVERITY_LABEL_PATTERN.findall(text)
        if not labels:
            continue
        if RESOLUTION_MARKER_PATTERN.search(text):
            continue
        hits.append((start, end, labels))
    return hits


def find_uncited_defect_citations(path: str) -> list[tuple[int, int, list[str]]]:
    """File-backed form of scan_text: reads `path` and applies the same rule."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return scan_text(lines)


def _glob_files(globs: tuple[str, ...], repo_root: str) -> list[str]:
    out: list[str] = []
    for g in globs:
        out.extend(glob.glob(os.path.join(repo_root, g), recursive=True))
    return sorted(out)


def scan_files(file_paths: list[str], repo_root: str = _REPO_ROOT) -> list[str]:
    """Scans every file in `file_paths` (absolute or repo-root-relative).
    Returns one formatted failure string per uncited-defect-citation block,
    empty if clean."""
    failures: list[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
        rel = os.path.relpath(abs_p, repo_root)
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        for start, end, labels in find_uncited_defect_citations(abs_p):
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
