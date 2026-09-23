"""T-2851 RU's independent 16-token baseline/candidate comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
from run_identity import RU, execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()
    baseline, _ = execute(args.base.resolve(), "cpu", RU, "-", 16, 0, False, "0,1,2")
    candidate, _ = execute(args.candidate.resolve(), "gpu", RU, "-", 16, 0,
                           args.device, "0,1,2")
    if candidate != baseline:
        print(f"FAIL RU {'device' if args.device else 'host'} tokens={candidate} baseline={baseline}")
        return 1
    print(f"PASS RU {'device' if args.device else 'host'} tokens={candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
