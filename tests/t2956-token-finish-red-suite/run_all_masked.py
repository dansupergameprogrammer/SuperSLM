"""Compare the live CPU/GPU degenerate finish row with v1.6.0."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

from run_identity import R05

BUILDS = Path(r"D:\_t2956-mutants\build")


def main() -> int:
    failures = 0
    for version, folder in (("v1.6.0", "allmasked_v160"),
                            ("candidate", "allmasked_current")):
        root = BUILDS / folder
        env = os.environ.copy()
        env["T2956_ALL_MASKED"] = "1"
        env["T2956_SHADER_DIR"] = str(root / "shaders")
        for backend in ("cpu", "gpu"):
            for tasks in ((0,) if version == "v1.6.0" else (0, 3)):
                run = subprocess.run([str(root / "cell_real_decode.exe"), backend, str(R05),
                                      "potion_shop_order", "1", str(tasks), "0"],
                                     cwd=root, env=env, text=True, capture_output=True,
                                     timeout=120)
                expected = f"ALL_MASKED backend={backend} status=0 token=-2 walk_same=1"
                if run.returncode or expected not in run.stdout:
                    print(f"FAIL all-masked {version}/{backend}/T{tasks} exit={run.returncode} "
                          f"stdout={run.stdout!r} stderr={run.stderr!r}")
                    failures += 1
                else:
                    print(f"PASS all-masked {version}/{backend}/T{tasks} "
                          "status=0 token=-2 walk_same=1")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
