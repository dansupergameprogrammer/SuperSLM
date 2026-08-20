"""CI source check: no public document restates a present-tense CI-execution
claim outside `docs/platform-support.md`'s own "CI execution status" section --
the one place this project's public prose is allowed to state what its hosted
CI has actually run (T-2195 remedy S3; `Claude/Poirot/
1381076-t2195-t2189-closing-confirmation.md`).

WHY THIS SHAPE. The same overclaim recurred across four consecutive review
rounds (T-2191, T-2192, T-2195, T-2196), each time in different words the
prior round's own fix did not anticipate: "exercised at continuous-integration
extent", "CI-probed on hosted runners", "proven bit-identical and exercised in
continuous integration", "runs on every push, and is fully green". A literal
phrase list closes exactly the phrasings already found and nothing else. This
module is scoped by MEANING instead: any clause naming CI (the token `CI`,
`continuous integration` (also hyphenated), `GitHub Actions`, `hosted
runner(s)`, `the matrix`/`the workflow(s)`, or the compound `CI-<verb>` shape)
alongside an unhedged present-tense execution or passing verb (`is
exercised/validated/tested`, `runs on every`, `is green/passing`, `executes`,
`CI-exercised/probed/tested/validated`) is flagged, unless the same clause
carries a negation cue (`not`/`never`/`capped`/`no further`/etc). A NEW
paraphrase of the same claim is caught by the rule without a new literal
entry -- what the phrase-list approach this module replaces could not do.

T-2196'S OWN FINDING (S1): THE FIRST VERSION OF THIS MODULE DID NOT CATCH THE
SENTENCE IT WAS BUILT FOR. Two independent causes, both fixed here and both
pinned by their own regression cells:

1. THE CLAUSE SPLITTER SHREDDED THE SENTENCE. The first version split on
   every `.` and `;` unconditionally (`re.split(r'[.;]| -- | — ', text)`).
   Applied to "The workflows exist and define the full matrix
   ([.github/workflows/tests.yml](.github/workflows/tests.yml)), runs on
   every push, and is **fully green on the 1.0 release commit**", the
   periods inside the filename `.github/workflows/tests.yml` and the link
   target cut the one sentence into six fragments -- the noun landed in one
   fragment, the verb in another, and neither fragment carried both. Fixed
   here by masking every period that lives inside a markdown link target, an
   inline code span, a bracketed link label, or a version number (`1.0`,
   `1.1.0`) BEFORE searching for clause-boundary periods (`_mask_non_
   sentence_periods`), so a split point is only ever found at a period that
   is genuinely a sentence boundary in the ORIGINAL, un-masked text.
2. AN EXEMPTION MATCHED EXACTLY ONE CLAUSE IN THE WHOLE SCANNED CORPUS. The
   negation word list included `no further`, present in this project's
   public prose in exactly one place (`README.md`'s own "CI extent" defining
   sentence: "...and no further hardware-specific measurement has been made
   beyond what that gives"). Under the FIRST version's broken splitter, that
   phrase's clause boundary was also wrong, and the exemption's only
   measured effect on this tree was to silently clear the one real
   violation the check would otherwise have caught in the neighboring,
   un-hedged sentence. Per `StandardsDocument.md` Sec4, an exemption pattern
   that matches exactly one clause in the corpus IS the finding: audited
   here and kept, because with the splitter fixed it correctly scopes to
   its OWN sentence only ("CI extent" means... and no further measurement
   has been made -- a genuine, single-sentence hedge) and no longer bleeds
   into the neighboring sentence the splitter used to merge it with. Every
   other negation word (`not`, `never`, `hasn't`, `has not`, `have not`,
   `capped`) was checked against the same historical corpus below and each
   has more than one legitimate hit; none is a lone-match exemption.

RULE COVERAGE, MEASURED. `tests/ci/ci_claims_historical_fixtures/` holds the
exact recovered text of every site four review rounds found this class at:
`CHANGELOG.md`'s and `docs/platform-support.md`'s Known-gaps AVX-512 bullets,
`README.md`'s AVX-512 roadmap bullet, the SSE2/AVX2/AVX-512 kernel-tier table
rows in `docs/platform-support.md`, and `README.md`'s own CI-extent paragraph
(the sentence T-2196 S1 found this module's first version could not catch).
`test_check_ci_claims.py`'s `test_fires_on_every_historical_pre_fix_site` and
`test_does_not_fire_on_any_historical_post_fix_site` replay both states
through this module's own scanner and assert 7/7 fire before the fix and 0/7
false-fire on the current, corrected text -- fixing nothing here, per
`StandardsDocument.md` Sec4's population-validation requirement.

INPUT COVERAGE AND ENVIRONMENT (T-2196 S2). The prior version of this check
lived as an inline `python3 -c "..."` block inside a single hosted-runner-only
GitHub Actions job (`workflow-lint`, `ubuntu-latest`), was not reachable by
`pytest tests/ci/`, and had no self-test -- unlike every sibling structural
checker in this directory (`check_no_forward_leaf_calls.py`, `check_present_
tense_defect_comments.py`, `check_gpu_guard_status_parity.py`), each a
`tests/ci/check_*.py` module with a `tests/ci/test_check_*.py` self-test,
invoked from CI by the same two-step shape (self-test, then the gate script)
and runnable locally by any pytest invocation. This module now matches that
convention: it runs anywhere `pytest tests/ci/` runs (the machine this
project actually develops and tests on, per `docs/platform-support.md`'s own
CI execution status section -- GitHub Actions hosted runs have not executed
since 2026-07-23), not only on a capped hosted runner. Path comparisons use
`os.path.normpath` rather than a literal `'docs/platform-support.md'` string
(T-2196 S2's third face: `glob.glob('docs/*.md')` yields backslash-separated
paths on Windows, so a literal forward-slash comparison never matched and the
prior version reported a false violation against a clean tree on the one
platform this project runs anything on).

CHANGELOG.MD SCOPING. Widening the verb pattern to catch the historical
corpus's bare-past-participle shape ("...is proven bit-identical and
exercised in continuous integration") also caught a real hit outside the
seven fixture sites: `CHANGELOG.md`'s own 1.0 release entry, "Linux and
macOS are exercised at continuous-integration extent only" -- exactly the
one surviving hit T-2196's own tree-wide grep found and accepted as "true
when written and remains true," not a live overclaim. `CHANGELOG.md` is
dated release history (Keep a Changelog convention, newest section first);
a claim inside an already-shipped, older `## [x.y.z]` section is a dated
record of what was true at that release, not a claim about today's CI. Only
the LATEST release section (or the un-headed preamble, if any) is scanned
for `CHANGELOG.md` (`_scope_changelog_to_latest_release`) -- the release
actually being published stays in scope, and shipped history is not
re-litigated on every run.

Modelled on `tests/ci/check_no_forward_leaf_calls.py`'s and `check_present_
tense_defect_comments.py`'s own conventions: a text scan (not a markdown
parse) over a fixed target-file list, a `scan_files`/`scan_text` pair
returning formatted failure strings, and a `main()` that fails loud.
"""
from __future__ import annotations

import glob
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# A CI-referring noun: the bare token "CI" (matches inside a hyphenated
# compound like "CI-exercised" too, since the boundary between "CI" and "-"
# is a word boundary), "continuous integration" with either a space or a
# hyphen between the two words (the historical corpus has both --
# `site1_pre.txt`'s "continuous-integration extent"), "GitHub Actions",
# "hosted runner(s)", or "the matrix"/"the workflow(s)".
CI_NOUN_PATTERN = re.compile(
    r"\bCI\b"
    r"|continuous[\s-]integration"
    r"|GitHub Actions"
    r"|hosted runners?"
    r"|\bthe (matrix|workflows?)\b",
    re.IGNORECASE,
)

# An unhedged present-tense execution or passing verb. Three shapes:
# "is exercised/validated/tested" with "is" immediately adjacent; the bare
# past participle alone ("...is proven bit-identical and exercised in
# continuous integration" -- `site2_pre.txt`/`site6_pre.txt`'s own
# compound-predicate shape, where "is" governs "proven" and never sits next
# to "exercised" at all); and the compound `CI-<verb>` adjectival shape
# (`site3_pre.txt`: "CI-exercised"; `site4_pre.txt`/`site5_pre.txt`:
# "CI-probed"), which has no separate "is" token either. The bare-participle
# alternative subsumes the "is <verb>" one; both are kept so the pattern
# reads as the shapes actually found, not as a minimal regex.
EXEC_VERB_PATTERN = re.compile(
    r"\bis (currently )?(exercised|validated|tested)\b"
    r"|\b(exercises|validates|tests)\b"
    r"|\b(exercised|validated|tested)\b"
    r"|\bruns? on every\b"
    r"|\bis (fully )?(green|passing)\b"
    r"|\bexecutes\b"
    r"|\bcurrently (passing|green|executing)\b"
    r"|\bCI-(exercised|probed|tested|validated|checked)\b",
    re.IGNORECASE,
)

# A negation/hedge cue that exempts the clause it appears in. Every entry
# was audited against `ci_claims_historical_fixtures/` (T-2196 S1's own
# directive: an exemption matching exactly one clause in the corpus IS the
# finding). `no further` has exactly one hit in the historical corpus
# (`site7_pre.txt`/`site7_post.txt`'s own "CI extent" defining sentence) and
# is kept because, with the clause splitter fixed (below), it correctly
# scopes to that one sentence and no longer bleeds into the neighboring,
# un-hedged sentence the way it did under the broken splitter -- the
# defect T-2196 S1 found was the splitter, not this exemption.
NEGATION_PATTERN = re.compile(
    r"\bnot\b|\bnever\b|\bhasn.t\b|\bhas not\b|\bhave not\b|\bcapped\b|\bno further\b",
    re.IGNORECASE,
)

# Markdown/text spans whose internal periods are never a sentence boundary:
# an inline code span (`` `tests.yml` ``), a bracketed link label
# (`[docs/platform-support.md]`), a parenthesized link target
# (`(docs/platform-support.md)`), and a bare version number (`1.0`, `1.1.0`).
_CODE_SPAN_PATTERN = re.compile(r"`[^`]*`")
_BRACKET_LABEL_PATTERN = re.compile(r"\[[^\]]*\]")
_PAREN_TARGET_PATTERN = re.compile(r"\([^()]*\)")
_VERSION_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+){1,3}\b")

# The clause-boundary token itself: a period or semicolon, or an em/en-dash
# used as punctuation (surrounded by spaces). Unchanged from the module this
# replaces -- the reported defect (T-2196 S1) was periods splitting inside a
# masked span, not the dash/semicolon boundaries, and the historical corpus
# has no dash or semicolon living inside a link, code span, or version
# number.
_CLAUSE_BREAK_PATTERN = re.compile(r"[.;]|\s--\s|\s—\s")

_TARGET_FILES = ("README.md", "CHANGELOG.md")

# The one section allowed to make a present-tense CI-execution claim: the
# consolidated statement every other public document points to instead of
# restating it (T-2195 S3). Built with os.path.join/os.path.normpath rather
# than a literal forward-slash string so the comparison is correct on
# Windows, where glob.glob('docs/*.md') yields backslash-separated paths
# (T-2196 S2's third face: the literal-string version never matched here and
# reported a false violation against the clean tree).
_EXEMPT_FILE = os.path.normpath(os.path.join("docs", "platform-support.md"))
_EXEMPT_SECTION_HEADING = "CI execution status"


def _mask_non_sentence_periods(text: str) -> str:
    """Returns a SAME-LENGTH copy of `text` with every period inside an
    inline code span, a bracketed link label, a parenthesized link target,
    or a version number replaced with a NUL placeholder. Same length is
    load-bearing: `_split_into_clauses` finds its break points against this
    masked copy and slices the ORIGINAL text at the identical offsets, so a
    masked period can never itself become (or hide) a split point, and no
    other character shifts position."""

    def _blank_periods(m: re.Match) -> str:
        return m.group(0).replace(".", "\x00")

    masked = text
    masked = _CODE_SPAN_PATTERN.sub(_blank_periods, masked)
    masked = _BRACKET_LABEL_PATTERN.sub(_blank_periods, masked)
    masked = _PAREN_TARGET_PATTERN.sub(_blank_periods, masked)
    masked = _VERSION_NUMBER_PATTERN.sub(_blank_periods, masked)
    return masked


def _split_into_clauses(text: str) -> list[str]:
    """Splits `text` into clause-shaped fragments at real sentence/clause
    boundaries -- a period or semicolon, or an em/en-dash used as
    punctuation -- read from a period-masked copy (`_mask_non_sentence_
    periods`) so a period inside a filename, a URL, or a version number is
    never mistaken for one (T-2196 S1). The ORIGINAL text is what gets
    sliced; the masked copy is used only to locate the break points, and the
    two are guaranteed the same length so an offset found in one is valid in
    the other."""
    original = text.replace("\n", " ")
    masked = _mask_non_sentence_periods(original)
    assert len(masked) == len(original)
    clauses: list[str] = []
    start = 0
    for m in _CLAUSE_BREAK_PATTERN.finditer(masked):
        clauses.append(original[start : m.start()])
        start = m.end()
    clauses.append(original[start:])
    return clauses


def find_ci_execution_claims(text: str) -> list[str]:
    """Every clause in `text` that names CI (or a synonym) alongside an
    unhedged present-tense execution/passing verb, with no negation cue in
    the same clause. Returns the stripped clause text for each hit."""
    hits: list[str] = []
    for clause in _split_into_clauses(text):
        if not CI_NOUN_PATTERN.search(clause):
            continue
        if not EXEC_VERB_PATTERN.search(clause):
            continue
        if NEGATION_PATTERN.search(clause):
            continue
        stripped = clause.strip()
        if stripped:
            hits.append(stripped)
    return hits


def _strip_exempt_section(path: str, text: str) -> str:
    """Removes the one section this claim class is allowed to live in
    (`docs/platform-support.md`'s own "CI execution status" section) from
    `text` before scanning -- every OTHER section of that same file, and
    every other public document, stays in scope."""
    if os.path.normpath(path) != _EXEMPT_FILE:
        return text
    sections = re.split(r"(?m)^## ", text)
    kept = [s for s in sections if not s.startswith(_EXEMPT_SECTION_HEADING)]
    return "## ".join(kept)


# CHANGELOG.md is dated release history (Keep a Changelog convention: newest
# release at the top). A present-tense CI-execution claim in an OLDER,
# already-shipped release section is a dated statement of what was true at
# that release, not a live claim about today's CI -- exactly the "true when
# written and remains true" disposition T-2196's own tree-wide grep found for
# 1.0's own "Linux and macOS are exercised at continuous-integration extent
# only" entry, the one hit that audit accepted rather than flagged. Scoping
# the scan to only the LATEST `## [x.y.z]` section (or the un-headed
# preamble above the first such heading, if any) keeps a live overclaim in
# the release actually being published in scope while not re-litigating
# history on every run -- the same asymmetry `docs/platform-support.md`'s
# own CI execution status section gets in the other direction (one place is
# ALLOWED to make the claim there; here, one section is EXCUSED from being
# read as a current claim at all).
_CHANGELOG_RELEASE_HEADING_PATTERN = re.compile(r"(?m)^## \[")


def _scope_changelog_to_latest_release(path: str, text: str) -> str:
    if os.path.basename(os.path.normpath(path)) != "CHANGELOG.md":
        return text
    matches = list(_CHANGELOG_RELEASE_HEADING_PATTERN.finditer(text))
    if len(matches) < 2:
        return text
    return text[: matches[1].start()]


def scan_text(path: str, text: str) -> list[str]:
    """Scans one file's already-read `text` (with `path` used only to decide
    whether the CI-execution-status exemption or the CHANGELOG
    latest-release scoping applies) and returns one formatted failure string
    per violation, empty if clean."""
    scoped = _strip_exempt_section(path, text)
    scoped = _scope_changelog_to_latest_release(path, scoped)
    return [f"{path}: {hit}" for hit in find_ci_execution_claims(scoped)]


def _target_paths(repo_root: str) -> list[str]:
    paths = list(_TARGET_FILES)
    docs_glob = os.path.join(repo_root, "docs", "*.md")
    for p in sorted(glob.glob(docs_glob)):
        paths.append(os.path.relpath(p, repo_root))
    return paths


def scan_files(paths: list[str], repo_root: str = _REPO_ROOT) -> list[str]:
    """Scans every file in `paths` (repo-root-relative) and returns one
    formatted failure string per violation, empty if clean. A missing file
    is reported as its own failure, the same posture as every sibling
    checker in this directory."""
    failures: list[str] = []
    for rel in paths:
        abs_p = os.path.join(repo_root, rel)
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        with open(abs_p, "r", encoding="utf-8") as f:
            text = f.read()
        failures.extend(scan_text(rel, text))
    return failures


def main(repo_root: str = _REPO_ROOT) -> int:
    paths = _target_paths(repo_root)
    failures = scan_files(paths, repo_root)
    if failures:
        print("check_ci_claims.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"check_ci_claims.py: OK -- {len(paths)} public doc(s) scanned, "
        f"no present-tense CI-execution claim outside "
        f"{_EXEMPT_FILE}'s {_EXEMPT_SECTION_HEADING!r} section"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
