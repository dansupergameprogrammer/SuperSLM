#!/usr/bin/env python3
"""Cross-repo drift check for the vendored reference (T-1521, SuperSLM_S3a_
WalkingSkeleton_Plan.md Sec11 S3.1c item 2b).

A silently-forked reference is worse than an absent one (D-SLM534): a vendored
copy that has quietly diverged from `D:\\Wizard` reports agreement with a thing
nobody is maintaining, and reports it with the authority of a green gate.
`check_provenance.py` (S-HARDEN-5/T-1520/T-1529) proves the vendored copy on
disk matches its OWN recorded hash and that files sharing a Group agree on
their recorded Source commit -- it has no access to `D:\\Wizard` and cannot
prove the recorded hash and commit are still faithful to that source. This
script is the other half: given a path to a checkout of the source repository,
it proves the vendored copy still matches the source AT ITS PINNED COMMIT, and
separately proves the source has not moved past that commit on the file
without a re-vendor.

Reads the source repository path from exactly one explicit environment
variable, SUPERSLM_REFERENCE_SOURCE -- never a guessed default path, so this
can never silently pass by resolving against an unrelated directory that
happens to exist. When the variable is unset, or set to a path that is not a
git repository, this script exits non-zero immediately, naming the missing
variable. This is a hard failure, not a skip -- the same convention the
spike's own suite already follows (`Tools/superslm_spike/README.md`: "The
suite deliberately fails rather than skips on a missing module... A skipped
test reads as swept on every future audit; a failing one does not"), applied
here to a cross-repo dependency instead of an intra-repo one.

Covers the full-file vendors only -- files whose vendored content is claimed
to be byte-identical to one literal path in the source repository at one
pinned commit. `pipeline_prob_width_ceiling.py` (a narrow, hand-extracted
excerpt of pipeline.py, D-SLM367) and `rope_tables_pinned.json` (the
precomputed output of `precompute_pinned.py`, not a copy of any upstream
file) make no such claim -- neither's content equals any one source-repo path
byte-for-byte, so "git show <commit>:<path> equals the vendored copy" is not
a claim either can carry, and both are deliberately excluded from
_DRIFT_CHECKED_FILES rather than silently mismatched.

**This check cannot be a per-commit CI leg** (SuperSLM_S3a_WalkingSkeleton_
Plan.md Sec11 S3.1c item 2b): a public runner building `D:\\SuperSLM` has no
sibling checkout of `D:\\Wizard`. It runs only where both repositories are
checked out side by side -- a publish-gate obligation, named but not wired
here; cadence is Dan's call (item 5).

Three distinct failure modes against a present, valid source repository,
each reported with its own message so one is never confused with another:

  1. Hash mismatch -- the vendored file on disk does not match the source
     file's content AT THE PINNED COMMIT. Catches a hand-edit of either side
     (the vendored copy in this repo, or -- via the PROVENANCE.md-recorded
     commit no longer describing what's on disk -- a stale record).
  2. Drift-ahead -- the source repository's current branch tip carries a
     later commit touching the file than the pinned one. Distinct from a
     hash mismatch: the pin is stale (a re-vendor is owed), not that anyone
     hand-edited anything.
  3. Commit not found -- the source path resolves to a valid git repository
     that does not contain the pinned commit at all (a shallow clone, or a
     checkout of a different repository). Distinct from both: `git show
     <commit>:<path>` fails with `fatal: bad object`/`fatal: Not a valid
     object name`, caught here rather than surfaced as an uncaught
     subprocess error or misreported as a hash mismatch (T-1533).
"""

from __future__ import annotations

import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import check_provenance as _cp  # noqa: E402  (path set above)

ENV_VAR = "SUPERSLM_REFERENCE_SOURCE"

# (vendored relpath under this directory, source relpath under the source repo).
# See the module docstring for why the excerpt and the precomputed derivative
# are excluded.
_DRIFT_CHECKED_FILES: tuple[tuple[str, str], ...] = (
    ("superslm_spike/intmath.py", "Tools/superslm_spike/intmath.py"),
    ("superslm_spike/rope.py", "Tools/superslm_spike/rope.py"),
    ("superslm_spike/dynamic_engine.py", "Tools/superslm_spike/dynamic_engine.py"),
    ("superslm_spike/pipeline.py", "Tools/superslm_spike/pipeline.py"),
    ("superslm_spike/silu_lut.py", "Tools/superslm_spike/silu_lut.py"),
    ("superslm_spike/constrain.py", "Tools/superslm_spike/constrain.py"),
)


class SourceUnavailable(RuntimeError):
    """The source path is unset, does not exist, or is not a git repository --
    the missing-source failure mode, distinct from a found repository that
    lacks the pinned commit (CommitNotFound) or disagrees on content
    (reported as a plain mismatch string, not an exception)."""


class CommitNotFound(RuntimeError):
    """The source path is a valid git repository, but does not contain the
    commit PROVENANCE.md pins for one of the checked files -- a shallow
    clone, or a checkout of an unrelated repository (T-1533)."""


def _run_git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", cwd] + args, capture_output=True, text=False)


def resolve_source(source_dir: str | None) -> str:
    """Validates `source_dir` names a real git repository, raising
    SourceUnavailable naming the problem if not. Never guesses a default --
    an unset or blank `source_dir` is exactly the failure this function
    exists to catch, not a signal to fall back to anything."""
    if not source_dir:
        raise SourceUnavailable(
            f"{ENV_VAR} is not set -- drift cannot be checked without an explicit "
            f"path to a checkout of the source repository (D:\\Wizard)"
        )
    if not os.path.isdir(source_dir):
        raise SourceUnavailable(f"{ENV_VAR}={source_dir!r} is not a directory")
    proc = _run_git(["rev-parse", "--is-inside-work-tree"], source_dir)
    if proc.returncode != 0 or proc.stdout.strip() != b"true":
        raise SourceUnavailable(f"{ENV_VAR}={source_dir!r} is not a git repository")
    return source_dir


def _show_at_commit(source_dir: str, commit: str, source_relpath: str) -> bytes:
    """Returns the content of `source_relpath` at `commit` in `source_dir`.
    Raises CommitNotFound if the commit itself is absent from the repository
    (a shallow clone, or an unrelated checkout) -- distinguished from the
    commit existing but the path being wrong at that commit, which is a
    'path does not exist' error from the same command; both are reported as
    CommitNotFound here since neither reflects a real vendored file's content
    and both mean the pin cannot be evaluated against this source."""
    cat_file = _run_git(["cat-file", "-e", f"{commit}^{{commit}}"], source_dir)
    if cat_file.returncode != 0:
        raise CommitNotFound(
            f"commit {commit} not found in source repository at {source_dir!r} "
            f"(checking {source_relpath})"
        )
    show = _run_git(["show", f"{commit}:{source_relpath}"], source_dir)
    if show.returncode != 0:
        raise CommitNotFound(
            f"commit {commit} not found in source repository at {source_dir!r} "
            f"(path {source_relpath} unreadable at that commit: "
            f"{show.stderr.decode('utf-8', errors='replace').strip()})"
        )
    return show.stdout


def _drift_ahead_commits(source_dir: str, commit: str, source_relpath: str) -> list[str]:
    """Commits on the source repository's current branch tip (HEAD) that
    touch `source_relpath` and are not ancestors of `commit` -- i.e. commits
    strictly after the pinned one, in `git log`'s own reachability sense.
    Empty if the pin is current."""
    proc = _run_git(["log", "--format=%H", f"{commit}..HEAD", "--", source_relpath], source_dir)
    if proc.returncode != 0:
        # HEAD..pin ordering issues (detached, pin not an ancestor at all) are
        # not this function's concern -- resolve_source/_show_at_commit above
        # already proved the commit exists; a log failure here is treated as
        # "nothing found" rather than raised, since the commit's own presence
        # is already established.
        return []
    return [line for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]


def check_drift(
    repo_root: str = _THIS_DIR,
    source_dir: str | None = None,
    checked_files: tuple[tuple[str, str], ...] = _DRIFT_CHECKED_FILES,
    provenance_path: str | None = None,
) -> list[str]:
    """Runs all three drift checks against every entry in `checked_files`,
    returning one formatted failure string per finding (empty if clean).
    Raises SourceUnavailable if `source_dir` itself cannot be resolved to a
    real git repository -- that failure applies to the whole run, not to one
    file, so it is not folded into the per-file failure list."""
    resolved_source = resolve_source(source_dir)

    prov_path = provenance_path or os.path.join(repo_root, "PROVENANCE.md")
    with open(prov_path, "r", encoding="utf-8") as f:
        recorded = _cp._parse_provenance(f.read())

    failures: list[str] = []
    for vendored_relpath, source_relpath in checked_files:
        row = recorded.get(vendored_relpath)
        if row is None or row.source_commit in ("", "PENDING"):
            failures.append(
                f"{vendored_relpath}: no Source commit recorded in PROVENANCE.md -- cannot check drift"
            )
            continue
        commit = row.source_commit

        vendored_abs = os.path.join(repo_root, vendored_relpath)
        if not os.path.isfile(vendored_abs):
            failures.append(f"{vendored_relpath}: vendored file not found on disk at {vendored_abs}")
            continue
        with open(vendored_abs, "rb") as f:
            vendored_bytes = f.read()

        try:
            source_bytes = _show_at_commit(resolved_source, commit, source_relpath)
        except CommitNotFound as e:
            failures.append(f"{vendored_relpath}: {e}")
            continue

        if vendored_bytes != source_bytes:
            failures.append(
                f"{vendored_relpath}: mismatch -- vendored copy differs from "
                f"{source_relpath}@{commit} in the source repository "
                f"(hand-edit of the vendored copy, or a stale pin)"
            )
            continue

        drift_ahead = _drift_ahead_commits(resolved_source, commit, source_relpath)
        if drift_ahead:
            failures.append(
                f"{vendored_relpath}: drift-ahead -- {source_relpath} was touched by "
                f"{len(drift_ahead)} commit(s) in the source repository after the pinned "
                f"{commit} (newest {drift_ahead[0]}) -- a re-vendor is owed"
            )

    return failures


def main(
    repo_root: str = _THIS_DIR,
    source_dir: str | None = None,
    checked_files: tuple[tuple[str, str], ...] = _DRIFT_CHECKED_FILES,
    provenance_path: str | None = None,
) -> int:
    if source_dir is None:
        source_dir = os.environ.get(ENV_VAR)
    try:
        failures = check_drift(repo_root, source_dir, checked_files, provenance_path)
    except SourceUnavailable as e:
        print(f"check_source_drift.py: FAILED -- {e}", file=sys.stderr)
        return 1

    if failures:
        print("check_source_drift.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"check_source_drift.py: OK -- {len(checked_files)} vendored files match their pinned source state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
