"""On/off CPU load for C6 7.9 construction runs (four plan-pinned forms).

The worker count is calibrated on the box to a 2.8x decode slowdown before
commissioning. This program records the exact schedule; it never grades a pair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys
import time

FORMS = {"square-0.1": .1, "square-1": 1.0, "square-60": 60.0}

def schedule(form: str, duration: float, seed: int) -> list[tuple[float, bool]]:
    if duration <= 0:
        raise ValueError("positive duration required")
    events = []
    rng = random.Random(seed)
    at, active = 0.0, True
    while at < duration:
        events.append((at, active))
        dwell = (FORMS[form] / 2 if form in FORMS else rng.expovariate(1.0))
        at += max(dwell, 1e-6)
        active = not active
    return events

def burn(active: mp.Event, stop: mp.Event) -> None:
    value = 1
    while not stop.is_set():
        if active.is_set():
            for _ in range(10000):
                value = ((value * 1664525) + 1013904223) & 0xFFFFFFFF
        else:
            time.sleep(.001)
    if value < 0:
        raise AssertionError("unreachable")

def run(form: str, duration: float, workers: int, seed: int, record: Path,
        calibration_probe=None, calibration_source: Path | None = None) -> dict:
    if workers < 1:
        raise ValueError("at least one worker required")
    events = schedule(form, duration, seed)
    active, stop = mp.Event(), mp.Event()
    processes = [mp.Process(target=burn, args=(active, stop)) for _ in range(workers)]
    start = time.monotonic()
    started = []
    calibration = None
    def publish(phase: str) -> None:
        record.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {"form": form, "pid": os.getpid(), "worker_pids": [p.pid for p in started],
                    "phase": phase, "complete": False, "calibration": calibration,
                    "seed": seed, "started_monotonic": start}
        temporary = record.with_suffix(record.suffix + ".partial")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, record)
    try:
        for process in processes:
            process.start()
            started.append(process)
        if calibration_probe is not None:
            active.clear()
            publish("off")
            off = float(calibration_probe("off", [p.pid for p in started]))
            active.set()
            publish("on")
            on = float(calibration_probe("on", [p.pid for p in started]))
            if not math.isfinite(off) or not math.isfinite(on) or off <= 0 or on / off < 2.8:
                raise ValueError("live load calibration below 2.8x")
            calibration = {"off": off, "on": on, "slowdown": on / off, "qualified": True}
            if calibration_source is not None:
                source = calibration_source.resolve()
                calibration["probe_path"] = str(source)
                calibration["probe_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        for offset, on in events:
            while time.monotonic() - start < offset:
                time.sleep(min(.01, max(0.0, offset - (time.monotonic() - start))))
            if on:
                active.set()
            else:
                active.clear()
            publish("on" if on else "off")
        while time.monotonic() - start < duration:
            time.sleep(.01)
    finally:
        stop.set()
        for process in started:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
    result = {"form": form, "duration_requested_s": duration, "elapsed_s": time.monotonic() - start,
              "workers": workers, "seed": seed, "pid": os.getpid(), "events": events,
              "worker_pids": [p.pid for p in processes], "phase": "stopped", "complete": True,
              "calibration": calibration, "worker_exits": [p.exitcode for p in processes]}
    record.parent.mkdir(parents=True, exist_ok=True)
    temporary = record.with_suffix(record.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, record)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", choices=[*FORMS, "random-dwell"], required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--calibration-probe", type=Path, required=True,
                        help="independent decode probe; called as PROBE on|off WORKER_PID... and prints median time")
    args = parser.parse_args()
    def probe(phase: str, workers: list[int]) -> float:
        command = [sys.executable, str(args.calibration_probe), phase, *(str(pid) for pid in workers)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=True)
        return float(result.stdout.strip())
    print(json.dumps(run(args.form, args.duration, args.workers, args.seed, args.record,
                         calibration_probe=probe, calibration_source=args.calibration_probe)))
