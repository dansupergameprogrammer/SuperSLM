"""Compare v1.6.0 and candidate private bytes after one tied flag-clear map."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


def reading(exe: Path, model: Path):
    run = subprocess.run([str(exe.resolve()), str(model.resolve())],
                         cwd=exe.resolve().parent, text=True, capture_output=True, timeout=120)
    if run.returncode:
        raise RuntimeError(f"{exe}: exit {run.returncode}\n{run.stdout}{run.stderr}")
    found = re.fullmatch(r"PRIVATE_BYTES (\d+) HEAD_BYTES (\d+)", run.stdout.strip())
    if not found:
        raise RuntimeError(f"{exe}: malformed reading {run.stdout!r}")
    return int(found[1]), int(found[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    base, head1 = reading(args.base, args.model)
    candidate, head2 = reading(args.candidate, args.model)
    if head1 != head2 or candidate >= base:
        print(f"FAIL private bytes base={base} candidate={candidate} head={head1}/{head2}")
        return 1
    delta = base - candidate
    print(f"PRIVATE_BYTES base={base} candidate={candidate} saved={delta} "
          f"head={head1} error={delta - head1}")
    if delta < head1 // 2:
        print("FAIL tied host copy saving below half the head size")
        return 1
    print("PASS tied host copy private-byte saving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
