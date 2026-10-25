"""Runs a command and reports its wall time and peak memory (Windows peak working set, or Linux
peak RSS, of the process and its children, sampled every 50 ms).

    python measure.py <log file> -- <command> [args ...]
"""
from __future__ import annotations

import subprocess
import sys
import time

import psutil


def peak_of(p: psutil.Process) -> int:
    info = p.memory_info()
    return getattr(info, "peak_wset", 0) or info.rss


def main(argv: list[str]) -> int:
    log, cmd = argv[0], argv[argv.index("--") + 1:]
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        ps = psutil.Process(proc.pid)
        peak = 0
        while proc.poll() is None:
            try:
                peak = max(peak, peak_of(ps) + sum(peak_of(c) for c in ps.children(recursive=True)))
            except psutil.Error:
                pass
            time.sleep(0.05)
    wall = time.perf_counter() - t0
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"measure: exit={proc.returncode} wall_s={wall:.2f} peak_bytes={peak}\n")
    print(f"exit={proc.returncode} wall_s={wall:.2f} peak_bytes={peak}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
