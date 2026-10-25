"""Runs a command and reports its wall time and peak memory: the command's own peak (Windows peak
working set, else RSS), and the largest peak of any child it starts, sampled every 50 ms.

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
        main_peak = child_peak = 0
        while proc.poll() is None:
            try:
                main_peak = max(main_peak, peak_of(ps))
                for c in ps.children(recursive=True):
                    child_peak = max(child_peak, peak_of(c))
            except psutil.Error:
                pass
            time.sleep(0.05)
    wall = time.perf_counter() - t0
    line = f"exit={proc.returncode} wall_s={wall:.2f} main_peak_bytes={main_peak} max_child_peak_bytes={child_peak}"
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"measure: {line}\n")
    print(line)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
