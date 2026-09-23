"""Public-ABI tie across rows 0/64, compared with v1.6.0 on both backends."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

from make_tie import EXPECTED_SHA256
from run_identity import execute

HERE = Path(__file__).resolve().parent
MODEL = HERE / "out" / "tie" / "tie.sslm"
PROMPT = "0,1,2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if not MODEL.is_file() or hashlib.sha256(MODEL.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("V=70 tie fixture missing or digest mismatch; run make_tie.py")
    failures = 0
    for backend in ("cpu", "gpu"):
        baseline, _ = execute(args.base, backend, MODEL, "-", 16, 0, False, PROMPT)
        if baseline != [0] * 16:
            raise RuntimeError(f"baseline {backend} does not select lowest tie: {baseline}")
        for tasks in (0, 3, 4, 7):
            got, log = execute(args.candidate, backend, MODEL, "-", 16, tasks,
                               False, PROMPT)
            if tasks > 1:
                line = next(line for line in log.splitlines() if line.startswith("THREADS "))
                calls = re.search(r"hook_calls=(\d+) task_calls=(\d+)", line)
                if not calls or (int(calls[1]), int(calls[2])) != (16, 32):
                    raise RuntimeError(f"{backend}/T{tasks} did not cross two partitions: {line}")
            if got != baseline:
                print(f"FAIL integrated tie {backend}/T{tasks} got={got} baseline={baseline}")
                failures += 1
            else:
                print(f"PASS integrated tie {backend}/T{tasks} tokens=16 "
                      f"mode={'two partitions' if tasks > 1 else 'serial'}")
    device, _ = execute(args.candidate, "gpu", MODEL, "-", 16, 0, True, PROMPT)
    if device != [0] * 16:
        print(f"FAIL integrated tie gpu/device got={device} baseline={[0] * 16}")
        failures += 1
    else:
        print("PASS integrated tie gpu/device tokens=16")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
