"""Curie's regeneration gate for gen_s3_1_c30_iexp_domain_sweep_fixtures.py
(SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1; Poirot review ac34677,
finding S10).

WHY THIS EXISTS. The generated header's own line states "Re-running this
script must reproduce this file byte-for-byte" -- a claim nothing in this
tree checked before this file. The `generators` job in
.github/workflows/tests.yml runs eight OTHER generators and then a single
"Fail if regenerating changed any tracked file" step that runs
`git diff --exit-code` over a NAMED file list; that list does not include
this generator or its output header, so a regeneration drift here would go
undetected by a job that already exists and already runs on every CI
invocation.

WHY THIS IS A TEST FILE, NOT A tests.yml EDIT. Adding this generator's output
to that named list would be the more conventional fix and matches the
pattern the other eight generators use -- but it requires editing
.github/workflows/tests.yml, a file outside this campaign's writable scope
(D:\\SuperSLM\\tests\\ only). The check need not wait on that edit: this
module lives under tests/ci/, which `python -m pytest tests/ci/ -v` already
collects and runs on every CI invocation (confirmed directly: the review's
own B2 executed this exact command against this tree). A pytest-collected
gate placed here runs the moment it lands, with no separate CI-wiring step
to add, forget, or omit -- which is itself the failure class Poirot's
finding C1 names elsewhere in this same review (a check authored but never
wired into anything that runs it). Whoever later adds this generator to the
`generators` job's named file list makes the check redundant, not wrong;
this file is not superseded until that happens, and Records Standards Sec6.6
governs deleting it at that point, not before.

MECHANISM. Calls the generator's own `generate()` -- never `main()`, which
writes to disk -- and compares its output, in memory, against the committed
header. This test never writes into the working tree, so it cannot itself
violate the regeneration discipline it exists to enforce.

Line-ending normalization is deliberate, not an oversight: `generate()` joins
its lines with a literal "\n" and `main()` writes with `newline="\n"`
(explicit, so the committed blob is LF regardless of the platform that ran
it) -- but a clone with `core.autocrlf=true` (verified present on this
machine) rewrites the WORKING-TREE COPY to CRLF at checkout time, which
`git diff --exit-code` itself already normalizes back before comparing
(that normalization is exactly what lets the `generators` job's own
git-diff step work on any clone regardless of this setting). Reading the
committed file here with universal-newline translation (the default text
mode, not `newline=""`) reproduces that same normalization at the Python
level, so this gate agrees with `git diff --exit-code` rather than failing
on a line-ending artifact `git diff` would not have flagged. Verified: a
one-value injected drift (a q_ln2 field changed from 0 to 1) still fails
this test correctly with the normalization in place (see the test-design
record's execution log) -- the normalization suppresses only the
line-ending false positive, not genuine content drift.

Test-design record:
Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import gen_s3_1_c30_iexp_domain_sweep_fixtures as gen  # noqa: E402


def _read_committed_header() -> str:
    # Universal-newline translation (the default text-mode behavior, no
    # `newline=""`): collapses a working tree that `core.autocrlf=true`
    # rewrote to CRLF back to the LF `generate()` itself always produces --
    # see the module docstring's line-ending note.
    with open(gen.OUT_PATH, "r", encoding="ascii") as f:
        return f.read()


def test_regenerating_reproduces_the_committed_header_byte_for_byte():
    committed = _read_committed_header()
    regenerated = gen.generate()
    assert regenerated == committed, (
        "tests/gen_s3_1_c30_iexp_domain_sweep_fixtures.py's generate() output no "
        "longer matches the committed tests/sslm_s3_1_c30_iexp_domain_sweep_"
        "fixtures.h -- the generator's own header comment promises byte-for-byte "
        "reproduction (\"Re-running this script must reproduce this file "
        "byte-for-byte\"); regenerate and commit the header, or the generator's "
        "own logic changed without the fixture being regenerated to match"
    )


def test_the_gate_has_something_to_lose_a_nonvacuous_floor():
    # A comparison that is byte-identical only because both sides are
    # (accidentally) empty or degenerate would pass this file's own primary
    # assertion vacuously. This is the independent-population-style floor
    # this project's StandardsDocument Sec4 asks of a new structural check:
    # confirm the compared artifact is actually substantial before trusting
    # the byte-identical comparison above to mean anything. 198 == the sweep
    # domain's own row count (e in [-90, 8] inclusive, at 2 mantissas each --
    # gen_s3_1_c30_iexp_domain_sweep_fixtures.py's own build_rows()), which
    # Curie's original authoring of this fixture (2026-07-28) already
    # verified and the review reconfirmed (B4: "198 rows, 79 disagreements,
    # matching Sec7.2's stated figure").
    rows = gen.build_rows()
    assert len(rows) == 198, (
        f"gen_s3_1_c30_iexp_domain_sweep_fixtures.py's build_rows() now returns "
        f"{len(rows)} rows, not the expected 198 -- either the sweep's own domain "
        f"changed (a planner-owed decision, Sec5/D-SLM318) or this floor is stale "
        f"and needs updating alongside it"
    )
    committed = _read_committed_header()
    assert committed.count("\t{ /*m=*/") == 198, (
        "the committed header's own row count no longer matches 198 -- either it "
        "was hand-edited (the header's own first line: \"GENERATED FILE. Do not "
        "hand-edit.\") or it was committed stale against a since-changed generator"
    )
