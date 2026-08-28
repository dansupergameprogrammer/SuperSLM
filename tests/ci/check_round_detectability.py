"""CI structural check: T-2352, the detectability gate.

`Claude/CLAUDE.md`'s standing project rule ("A fix round that adds a
production change routes a test pin for THAT change, in the same round")
names an owed structure and calls itself a discipline until it exists: *"a
check that compares a round's production diff against its new-cell list and
refuses the mismatch."* This module is that structure.

THE CLASS THIS CLOSES. A production change lands in a fix or build round and
no cell in the gating suite can detect it -- revert the change, the suite
stays green. Ten verified instances, found by hand across two review rounds
on one arc (D-SLM4909): `Claude/Poirot/8a28460-t2344-fp-scan-fix-round-
confirmation.md` S2 (six unpinned, four proven undetectable by reverting) and
`Claude/Poirot/3cd1a2b-t2349-fp-scan-fold35-build-review.md` S3 (four proven
undetectable by a hand-rolled mutation battery). The mechanism named there --
a pytest import hook that rewrites one remedy's source before collection --
is generalised here into a reusable, round-scoped, hunk-level sweep, so the
next round runs it before shipping rather than waiting for the next review to
find the same shape by hand.

MECHANISM. Given a BASE ref, a HEAD ref, a whitelist of production-file
globs, and a pytest target (the gating suite), this module:

  1. Computes `git diff BASE HEAD -- <production globs>`, restricted to
     tracked files that still exist at HEAD.
  2. Splits each file's diff into its own hunks, at git's own `@@ ... @@`
     boundaries -- one hunk is one contiguous change region.
  3. Establishes the BASELINE signature: the gating suite's own outcome at
     HEAD, unmutated -- the set of (nodeid, outcome) pairs pytest reports,
     plus the raw passed/failed/xfailed/error counts.
  4. For each hunk, builds a MUTANT: the file at HEAD with that ONE hunk
     reverse-applied (`git apply -R`) and every other hunk, and every other
     file, left exactly as HEAD shipped it. Runs the gating suite against
     the mutant tree; restores the file; compares the mutant's signature to
     the baseline.
  5. A hunk whose mutant signature is IDENTICAL to the baseline is
     UNDETECTED -- the round's own new cells, and every pre-existing cell,
     pass whether or not that hunk shipped. A hunk whose mutant differs (a
     new failure, a flipped xfail, a collection error) is DETECTED.
  6. A hunk that will not reverse-apply alone (its context depends on a
     sibling hunk in the same file -- e.g. a signature changed in one hunk
     and a call site updated in another) is retried jointly with each
     subsequent hunk in the same file, up to a bounded window, and reported
     as a COMBINED unit if that succeeds. A hunk that never isolates is
     reported UNRESOLVED, never silently dropped from the report.

WHAT THIS CATCHES. Any hunk-granular change to a production file the gating
suite can import and execute, whose effect no cell's outcome depends on. This
subsumes line coverage (an unreached hunk is trivially undetected) and goes
beyond it: a reached hunk whose effect nothing asserts is undetected too --
`StandardsDocument.md` Sec5.4, the commissioning rule's must-accept /
must-reject shape, and the "test the real workload" rule this module applies
at hunk scope rather than corpus scope.

WHAT THIS CANNOT CATCH -- stated here because a check asserted is not a check
demonstrated (T-2352 brief):

  - A file the gating suite never imports at all makes every hunk in it read
    as undetected by construction, correctly -- this module does not
    distinguish "unreached because untested" from "unreached because the
    suite cannot reach this file's module boundary at all" (e.g. a driver
    script invoked only as a subprocess, never imported). Both need a cell;
    only the second needs a NEW import path before one can be written at
    all, which is a finding for the test author, not something this checker
    can propose.
  - A file this module cannot express as Python-importable production code
    under the given pytest target is out of its mechanism entirely -- a
    batch file, a YAML workflow, a linker script. Reverting one line of a
    `.bat` file and re-running a Python suite proves nothing, because the
    suite never executes the batch file. This module's default production
    globs are Python-only for exactly this reason; a caller who widens them
    to a non-Python file gets an UNRESOLVED report for every hunk in it, not
    a false PASS.
  - Two hunks that are individually redundant covers for the same behaviour
    (either alone reverts it) each show DETECTED when mutated alone, so a
    combination that is jointly unpinned reads as covered. This module
    mutates one hunk (or one bounded combined unit) at a time, never an
    arbitrary subset.
  - A hunk that is pure comment, docstring, or dead string content is
    correctly reported UNDETECTED and is not a defect -- this module does
    not parse Python well enough to tell prose from behaviour. Its report
    names the hunk's own text so a human (or the calling gate) makes that
    call; it does not auto-fail a round over a comment edit.
  - A mutant that fails to even collect (an exception during import) is
    scored DETECTED against the baseline signature, which is correct for
    "the suite's outcome changed" but may be detecting a syntax dependency
    rather than a behavioural one -- reported with its own error text so a
    reader does not mistake a coincidental crash for a targeted assertion.
  - MEASURED, not hypothetical: a git hunk that BUNDLES a genuinely-covered
    remedy with a genuinely-uncovered one reads DETECTED as a whole, hiding
    the uncovered half -- this is not a theoretical edge case, it is what
    this module's own hunk-level sweep produced on 5 of the 10 historical
    T-2352 instances (`Claude/Brunel/t2352-detectability-check-build-
    2026-08-27.md` Sec4), each requiring a hand-isolated, source-level
    mutation (keeping the bundled remedy's surrounding refactor intact) to
    reveal the uncovered sub-change underneath a DETECTED hunk. A caller
    who trusts a raw DETECTED verdict on a large, multi-remedy hunk without
    that finer check is trusting the coupling, not the coverage. Splitting
    each remedy into its own commit-time hunk (already good practice) is
    the structural fix; this module does not do that decomposition for a
    caller today.

USAGE.
    python tests/ci/check_round_detectability.py \\
        --base <ref> --head <ref> \\
        --suite tests/t2296-fp-free-open-red-suite \\
        [--prod tests/ci/check_fp_free_scan.py] \\
        [--prod tests/ci/run_fp_free_scan_real_corpus.py] \\
        [--json out.json]

Exits 0 if every hunk in scope is DETECTED or explicitly UNRESOLVED-and-
reported; exits 1 if any hunk is UNDETECTED. UNRESOLVED hunks are printed
loudly but do not by themselves fail the run -- an unresolvable hunk is a
report about this checker's own reach, not a verdict about the round, and
folding it into the same exit code as UNDETECTED would let a hard-to-isolate
hunk hide behind "the checker choked," which is worse than naming it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def _git_allow_fail(repo: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@dataclasses.dataclass
class Hunk:
    path: str
    index: int  # 0-based position within this file's diff
    header: str  # the "@@ -a,b +c,d @@ context" line
    patch_text: str  # a complete, appliable patch for this hunk alone


def _file_diff_header(repo: Path, base: str, head: str, path: str) -> tuple[str, str]:
    """Return (old_path_line, new_path_line) i.e. the '--- a/x' / '+++ b/x' pair."""
    raw = _git(repo, "diff", "--no-color", base, head, "--", path)
    old_line = new_line = ""
    for line in raw.splitlines():
        if line.startswith("--- "):
            old_line = line
        elif line.startswith("+++ "):
            new_line = line
            break
    if not old_line or not new_line:
        raise RuntimeError(f"could not locate --- / +++ header for {path}")
    return old_line, new_line


def get_hunks(repo: Path, base: str, head: str, path: str) -> list[Hunk]:
    """Split BASE..HEAD's diff of one file into individually-appliable hunks."""
    raw = _git(repo, "diff", "--no-color", "-U3", base, head, "--", path)
    if not raw.strip():
        return []
    old_line, new_line = _file_diff_header(repo, base, head, path)

    lines = raw.splitlines(keepends=True)
    # Find where the hunk bodies start (after the +++ line).
    body_start = None
    for i, line in enumerate(lines):
        if line.startswith("+++ "):
            body_start = i + 1
            break
    if body_start is None:
        return []

    hunk_starts = [
        i for i in range(body_start, len(lines)) if lines[i].startswith("@@ ")
    ]
    hunks: list[Hunk] = []
    for n, start in enumerate(hunk_starts):
        end = hunk_starts[n + 1] if n + 1 < len(hunk_starts) else len(lines)
        body = "".join(lines[start:end])
        patch_text = old_line + "\n" + new_line + "\n" + body
        if not patch_text.endswith("\n"):
            patch_text += "\n"
        header = lines[start].rstrip("\n")
        hunks.append(Hunk(path=path, index=n, header=header, patch_text=patch_text))
    return hunks


def try_reverse_apply(repo: Path, patch_text: str) -> tuple[bool, str]:
    """Attempt `git apply -R` of a single-hunk (or combined-hunk) patch against
    the current working tree. Returns (ok, stderr-or-empty).

    Patch bytes are piped as raw bytes, never through a text-mode stdin --
    on Windows, `subprocess.run(..., text=True)` translates `\\n` to the
    platform line separator on write, which corrupts a unified diff's own
    line endings and makes `git apply` reject a hunk that applies cleanly
    from a file. Encoded once, sent once, unmangled.
    """
    patch_bytes = patch_text.encode("utf-8")
    check_proc = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", "-R", "--whitespace=nowarn", "-"],
        input=patch_bytes,
        capture_output=True,
    )
    if check_proc.returncode != 0:
        return False, check_proc.stderr.decode("utf-8", errors="replace")
    apply_proc = subprocess.run(
        ["git", "-C", str(repo), "apply", "-R", "--whitespace=nowarn", "-"],
        input=patch_bytes,
        capture_output=True,
    )
    if apply_proc.returncode != 0:
        return False, apply_proc.stderr.decode("utf-8", errors="replace")
    return True, ""


def restore(repo: Path, path: str, ref: str) -> None:
    _git(repo, "checkout", ref, "--", path)


# ---------------------------------------------------------------------------
# Running the gating suite and reading its outcome
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(
    r"(?P<counts>(?:\d+ \w+(?:, )?)+) in [\d.]+s"
)
_RESULT_LINE_RE = re.compile(r"^(PASSED|FAILED|XFAIL|XPASS|ERROR)\s+(\S+)")


@dataclasses.dataclass(frozen=True)
class Signature:
    passed: int
    failed: int
    xfailed: int
    xpassed: int
    errors: int
    failed_ids: tuple[str, ...]
    error_ids: tuple[str, ...]
    collection_error: bool

    def __eq__(self, other: object) -> bool:  # explicit: outcome, not counts alone
        if not isinstance(other, Signature):
            return NotImplemented
        return (
            self.passed == other.passed
            and self.failed == other.failed
            and self.xfailed == other.xfailed
            and self.xpassed == other.xpassed
            and self.errors == other.errors
            and self.failed_ids == other.failed_ids
            and self.error_ids == other.error_ids
            and self.collection_error == other.collection_error
        )

    def brief(self) -> str:
        if self.collection_error:
            return "COLLECTION ERROR"
        return (
            f"{self.passed} passed, {self.failed} failed, "
            f"{self.xfailed} xfailed, {self.xpassed} xpassed, "
            f"{self.errors} errors"
        )


def run_suite(repo: Path, suite: str, extra_args: list[str]) -> Signature:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", suite, "-q", "--tb=line", "-rfE", *extra_args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = proc.stdout + "\n" + proc.stderr

    failed_ids: list[str] = []
    error_ids: list[str] = []
    for line in out.splitlines():
        m = _RESULT_LINE_RE.match(line.strip())
        if m:
            kind, nodeid = m.groups()
            if kind == "FAILED":
                failed_ids.append(nodeid)
            elif kind == "ERROR":
                error_ids.append(nodeid)

    counts = {"passed": 0, "failed": 0, "xfailed": 0, "xpassed": 0, "error": 0, "errors": 0}
    summary_line = ""
    for line in out.splitlines()[::-1]:
        if " in " in line and re.search(r"\d+\.\d+s", line):
            summary_line = line
            break
    collection_error = "ERRORS" in out and "error" in out.lower() and not summary_line
    if summary_line:
        for count, word in re.findall(r"(\d+) (\w+)", summary_line):
            counts[word] = int(count)
    elif "no tests ran" not in out.lower():
        collection_error = True

    return Signature(
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        xfailed=counts.get("xfailed", 0),
        xpassed=counts.get("xpassed", 0),
        errors=counts.get("error", 0) or counts.get("errors", 0),
        failed_ids=tuple(sorted(set(failed_ids))),
        error_ids=tuple(sorted(set(error_ids))),
        collection_error=collection_error,
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HunkResult:
    path: str
    index: int
    header: str
    status: str  # "detected" | "undetected" | "unresolved"
    signature: str
    detail: str = ""


def sweep_round(
    repo: Path,
    base: str,
    head: str,
    prod_files: list[str],
    suite: str,
    pytest_args: list[str] | None = None,
    combine_window: int = 2,
) -> tuple[Signature, list[HunkResult]]:
    pytest_args = pytest_args or []

    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    # Untracked files are ignored here on purpose: this checker's own script
    # (and a caller's scratch files) sit untracked in the same tree while it
    # runs. What must be clean is every TRACKED file this sweep might touch
    # -- an uncommitted edit to a production file would silently become part
    # of every mutant and every restore.
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise RuntimeError(f"working tree not clean before sweep:\n{dirty}")

    _git(repo, "checkout", "--quiet", head)
    try:
        baseline = run_suite(repo, suite, pytest_args)

        results: list[HunkResult] = []
        for path in prod_files:
            hunks = get_hunks(repo, base, head, path)
            n = len(hunks)
            resolved = [False] * n
            i = 0
            while i < n:
                if resolved[i]:
                    i += 1
                    continue
                window_end = min(n, i + 1 + combine_window)
                applied = False
                for j in range(i + 1, window_end + 1):
                    group = hunks[i:j]
                    combined_patch = "".join(h.patch_text for h in group)
                    # A combined patch needs a single file header, not one per hunk.
                    old_line, new_line = _file_diff_header(repo, base, head, path)
                    bodies = []
                    for h in group:
                        # strip the per-hunk header/+++/--- we added, keep body only
                        body_lines = h.patch_text.splitlines(keepends=True)[2:]
                        bodies.append("".join(body_lines))
                    combined_patch = old_line + "\n" + new_line + "\n" + "".join(bodies)
                    ok, err = try_reverse_apply(repo, combined_patch)
                    if ok:
                        try:
                            sig = run_suite(repo, suite, pytest_args)
                        finally:
                            restore(repo, path, head)
                        label = (
                            hunks[i].header
                            if j == i + 1
                            else f"{hunks[i].header} .. {hunks[j-1].header} (combined)"
                        )
                        status = "undetected" if sig == baseline else "detected"
                        results.append(
                            HunkResult(
                                path=path,
                                index=i,
                                header=label,
                                status=status,
                                signature=sig.brief(),
                            )
                        )
                        for k in range(i, j):
                            resolved[k] = True
                        applied = True
                        break
                if not applied:
                    results.append(
                        HunkResult(
                            path=path,
                            index=i,
                            header=hunks[i].header,
                            status="unresolved",
                            signature="",
                            detail="could not reverse-apply alone or combined "
                            f"with the next {combine_window} hunk(s)",
                        )
                    )
                    resolved[i] = True
                i += 1
        return baseline, results
    finally:
        # Leave the tree exactly as it was: HEAD checked out, nothing dirty.
        leftover = _git(repo, "status", "--porcelain").strip()
        if leftover:
            _git(repo, "checkout", "--", ".")
        if current and current != "HEAD":
            _git(repo, "checkout", "--quiet", current)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root")
    parser.add_argument("--base", required=True, help="prior round's tip ref")
    parser.add_argument("--head", required=True, help="this round's tip ref")
    parser.add_argument(
        "--prod",
        action="append",
        required=True,
        help="a production file path to sweep (repeatable)",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="the gating suite path passed to pytest",
    )
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="extra argument forwarded to pytest (repeatable)",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    baseline, results = sweep_round(
        repo, args.base, args.head, args.prod, args.suite, args.pytest_arg
    )

    print(f"baseline: {baseline.brief()}")
    undetected = [r for r in results if r.status == "undetected"]
    unresolved = [r for r in results if r.status == "unresolved"]
    detected = [r for r in results if r.status == "detected"]
    for r in results:
        print(f"[{r.status.upper():10}] {r.path}#{r.index}  {r.header}  {r.detail}")

    print(
        f"\n{len(detected)} detected, {len(undetected)} UNDETECTED, "
        f"{len(unresolved)} unresolved, of {len(results)} hunks swept"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "base": args.base,
                    "head": args.head,
                    "baseline": baseline.brief(),
                    "results": [dataclasses.asdict(r) for r in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return 1 if undetected else 0


if __name__ == "__main__":
    raise SystemExit(main())
