"""Plan rev 1.18.9 §15.28 item 2 (§8): the reference anchor fixture refuses a shallow history.

In a shallow clone `git merge-base` sees only the commits above the boundary, so it can answer, or answer wrongly,
where §8 says no merge-base is recorded. This cell builds a small full repository of its own (main: A-B-C; the
post-M1 branch: C-D), so it never depends on how the source tree was cloned, and runs write_merge_base.py:

  (i)   a `git clone --depth 1` of it, at a HEAD where `git merge-base` still answers (HEAD at main's tip) and at the
        branch tip: the fixture exits nonzero, records the failure marker (never an anchor), and names the shallow
        history and the remedy `git fetch --unshallow`;
  (ii)  the same clone after `git fetch --unshallow`: the fixture records the true merge-base and exits 0.

Mutant: the fixture's shallow check removed. (i)'s first clone then records C as an anchor and exits 0, and this cell
goes red.

This cell grades the fixture only. The rest of §15.28 (i)'s chain is not asserted here: the reference cells take the
anchor through ctest's FIXTURES_REQUIRED, so a fixture refusal leaves them Not Run (counted failed, never skipped), and
the loader (precache_reference.h) refuses any anchor that is not 40 characters. Its one-time receipt, a --depth 1 clone
of the post-M1 branch where `ctest -R '^superslm_precache_e2_reference$'` reports the anchor refusal and e2_reference
Not Run, is in the plan's build record post-m1-cloud-2026-10-22.md.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

FIXTURE = Path(__file__).resolve().with_name("write_merge_base.py")
ENV = dict(os.environ, GIT_AUTHOR_NAME="pc", GIT_AUTHOR_EMAIL="pc@example.invalid",
           GIT_COMMITTER_NAME="pc", GIT_COMMITTER_EMAIL="pc@example.invalid",
           GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
FAILURES = []


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, text=True, capture_output=True,
                          check=True).stdout.strip()


def check(condition: bool, message: str) -> None:
    print(("ok   " if condition else "FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def run_fixture(repo: Path, out: Path):
    if out.exists():
        out.unlink()
    result = subprocess.run([sys.executable, str(FIXTURE), str(repo), "origin/main", str(out)],
                            env=ENV, text=True, capture_output=True)
    recorded = out.read_text(encoding="ascii").strip() if out.exists() else ""
    return result.returncode, recorded, result.stdout + result.stderr


def is_anchor(text: str) -> bool:
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        full = root / "full"
        full.mkdir()
        git(full, "init", "-q", "-b", "main")
        commits = {}
        for name in ("A", "B", "C"):
            (full / "f.txt").write_text(name + "\n", encoding="ascii")
            git(full, "add", "f.txt")
            git(full, "commit", "-q", "-m", name)
            commits[name] = git(full, "rev-parse", "HEAD")
        git(full, "checkout", "-q", "-b", "post-m1")
        (full / "g.txt").write_text("D\n", encoding="ascii")
        git(full, "add", "g.txt")
        git(full, "commit", "-q", "-m", "D")
        out = root / "merge-base.txt"

        for label, branch in (("HEAD at main's tip", "main"), ("HEAD at the branch tip", "post-m1")):
            clone = root / f"shallow-{branch}"
            git(root, "clone", "-q", "--depth", "1", "--no-single-branch", "--branch", branch,
                full.as_uri(), str(clone))
            check(git(clone, "rev-parse", "--is-shallow-repository") == "true",
                  f"(i) {label}: the --depth 1 clone is shallow")
            code, recorded, text = run_fixture(clone, out)
            check(code != 0, f"(i) {label}: the fixture refuses (exit {code})")
            check(not is_anchor(recorded), f"(i) {label}: no anchor is recorded (file reads {recorded!r})")
            check("shallow" in text and "git fetch --unshallow" in text,
                  f"(i) {label}: the refusal names the shallow history and the remedy")

            git(clone, "fetch", "-q", "--unshallow")
            check(git(clone, "rev-parse", "--is-shallow-repository") == "false",
                  f"(ii) {label}: the clone is full after git fetch --unshallow")
            code, recorded, text = run_fixture(clone, out)
            check(code == 0 and recorded == commits["C"],
                  f"(ii) {label}: the fixture records the true merge-base {commits['C'][:8]} "
                  f"(exit {code}, file {recorded[:12]!r})")
    print(f"merge-base shallow cell: {len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
