"""T-2851 row 10: reject a thread born inside the caller's parallel hook."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from run_identity import R05

HERE = Path(__file__).resolve().parent
CELL = HERE / "out" / "cell_real_decode.exe"


def run(probe: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("T2956_THREAD_SPAWN_PROBE", None)
    if probe:
        env["T2956_THREAD_SPAWN_PROBE"] = "1"
    return subprocess.run([str(CELL), "cpu", str(R05), "-", "1", "4", "0"],
                          cwd=CELL.parent, env=env, text=True, capture_output=True,
                          timeout=120)


def main() -> int:
    normal = run(False)
    ordinary = normal.stdout + normal.stderr
    if normal.returncode or not re.search(r"THREADS .*new=0 .*changed_at_hook=0", ordinary):
        print(f"FAIL census normal exit={normal.returncode}\n{ordinary}")
        return 1
    probe = run(True)
    injected = probe.stdout + probe.stderr
    if (probe.returncode != 29 or not re.search(r"THREAD_NEW id=\d+ start=[0-9A-Fa-f]+ module=.+", injected)
            or not re.search(r"THREADS .*new=1 .*changed_at_hook=1", injected)):
        print(f"FAIL census must-reject exit={probe.returncode}\n{injected}")
        return 1
    print("PASS census normal new=0 changed_at_hook=0")
    print(next(line for line in injected.splitlines() if line.startswith("THREAD_NEW ")))
    print("PASS census must-reject exit=29 new=1 changed_at_hook=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
