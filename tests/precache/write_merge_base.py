"""CTest setup: recompute the reference anchor immediately before grading."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    repo, line, output = sys.argv[1:]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Plan §8 (rev 1.18.9 §15.28 item 2): a shallow history records no merge-base. Above the boundary
        # git merge-base can answer, and below it can name the wrong commit, so every shallow clone refuses.
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo,
                                 text=True, capture_output=True, check=True).stdout.strip()
        if shallow != "false":
            raise ValueError(f"shallow history ({repo} --is-shallow-repository reads {shallow!r}); "
                             "run git fetch --unshallow, then rerun")
        result = subprocess.run(["git", "merge-base", "HEAD", line], cwd=repo,
                                text=True, capture_output=True, check=True)
        anchor = result.stdout.strip()
        if len(anchor) != 40 or any(c not in "0123456789abcdef" for c in anchor):
            raise ValueError(f"invalid merge-base output: {anchor!r}")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        target.write_bytes(f"FAIL {error}\n".encode("ascii", errors="backslashreplace"))
        print(f"reference merge-base fixture refused: {error}", file=sys.stderr)
        return 1
    target.write_bytes((anchor + "\n").encode("ascii"))
    print(f"reference merge-base fixture: {line} -> {anchor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
