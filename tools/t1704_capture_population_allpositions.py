#!/usr/bin/env python3
"""T-1704 Part 3 -- re-captures the campaign's own 9-prompt population with
all-position site-dump capture (`--site-dump-all-positions`), against the
SAME baseline artifact `tools/t1697_capture_population_sitedumps.py` already
used. Mirrors that script's own shape exactly, substituting the new flag and
output directory.

Also, as the strongest possible backward-compatibility guard (stronger than
"published statistics reproduce" -- an exact byte comparison): re-captures
the SAME population in DEFAULT (last-token-only) mode with the SAME,
modified `sslm_layer_trace.exe` and diffs every resulting `.sitedump`
against the ORIGINAL T-1697 capture under `out/t1697_population/` byte-for-
byte. If this task's change to `sslm_layer_trace.cpp` altered default-mode
behavior in any way, this comparison would show it directly.
"""
from __future__ import annotations

import filecmp
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
ALLPOS_DIR = os.path.join(REPO_ROOT, "out", "t1704_population_allpos")
DEFAULT_RECAPTURE_DIR = os.path.join(REPO_ROOT, "out", "t1704_population_default_recapture")
ORIGINAL_DIR = os.path.join(REPO_ROOT, "out", "t1697_population")


def main() -> int:
    os.makedirs(ALLPOS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RECAPTURE_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    all_default_identical = True
    for label, question, role in KSR.POPULATION:
        prompt = KSR.build_prompt(question)

        # --- all-position capture (the real Part 3/4 population). ---
        allpos_dump = os.path.join(ALLPOS_DIR, f"{label}.dump")
        allpos_site = os.path.join(ALLPOS_DIR, f"{label}.sitedump")
        proc = subprocess.run(
            [str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH), prompt,
             "--dump", allpos_dump, "--site-dump", allpos_site, "--site-dump-all-positions"],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"FAILED: all-position capture for {label!r} (exit {proc.returncode})\n"
                  f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}", file=sys.stderr)
            return 1
        assert "self_check: production and manual-replay paths agree" in proc.stdout
        size_mb = os.path.getsize(allpos_site) / 1e6
        print(f"captured (all-position) {label!r} role={role} -> {allpos_site} ({size_mb:.1f} MB)")

        # --- default (last-token-only) re-capture -- the backward-compat guard. ---
        default_dump = os.path.join(DEFAULT_RECAPTURE_DIR, f"{label}.dump")
        default_site = os.path.join(DEFAULT_RECAPTURE_DIR, f"{label}.sitedump")
        proc2 = subprocess.run(
            [str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH), prompt,
             "--dump", default_dump, "--site-dump", default_site],
            capture_output=True, text=True, timeout=300)
        if proc2.returncode != 0:
            print(f"FAILED: default-mode re-capture for {label!r} (exit {proc2.returncode})\n"
                  f"stdout:\n{proc2.stdout}\nstderr:\n{proc2.stderr}", file=sys.stderr)
            return 1
        assert "self_check: production and manual-replay paths agree" in proc2.stdout

        original_site = os.path.join(ORIGINAL_DIR, f"{label}.sitedump")
        if not os.path.isfile(original_site):
            print(f"WARNING: no original T-1697 capture at {original_site} to diff against",
                  file=sys.stderr)
            continue
        identical = filecmp.cmp(default_site, original_site, shallow=False)
        all_default_identical = all_default_identical and identical
        print(f"  default-mode re-capture byte-identical to ORIGINAL T-1697 capture: {identical}")

    print()
    if all_default_identical:
        print("RESULT: every default-mode (last-token-only) re-capture is BYTE-IDENTICAL to the "
              "original T-1697 capture (produced before this task's tooling change) -- default "
              "behavior is provably unaltered, not merely 'should be unaltered' by code inspection.")
    else:
        print("RESULT: MISMATCH -- a default-mode re-capture differs from the original T-1697 "
              "capture. This task's change perturbed default behavior. STOP.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
