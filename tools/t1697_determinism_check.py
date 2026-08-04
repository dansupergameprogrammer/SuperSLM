#!/usr/bin/env python3
"""T-1697 same-device determinism re-check: the artifact changed, so the
bit-exact guarantee needs re-checking at minimum on THIS device. Runs
`sslm_layer_trace.exe --dump --site-dump` twice against the SAME migrated
artifact and the same prompt, and asserts the two dump files are byte-
identical (both the base dump and the site-dump). Cross-device/cross-vendor
re-verification (the G-1 hardware) is out of this task's scope -- recorded
as owed in the build log, not attempted here.
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


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: t1697_determinism_check.py <artifact_path>", file=sys.stderr)
        return 1
    artifact_path = sys.argv[1]
    out_dir = os.path.join(REPO_ROOT, "out", "t1697_determinism")
    os.makedirs(out_dir, exist_ok=True)

    label, question, _role = KSR.POPULATION[0]  # digit_symbol_spaced -- a mech2 prompt
    prompt = KSR.build_prompt(question)

    dumps = []
    sitedumps = []
    for run in (1, 2):
        dump_path = os.path.join(out_dir, f"run{run}.dump")
        site_dump_path = os.path.join(out_dir, f"run{run}.sitedump")
        proc = subprocess.run(
            [DRIVER_EXE, artifact_path, str(KSR.TOKENIZER_PATH), prompt,
             "--dump", dump_path, "--site-dump", site_dump_path],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"FAILED: run {run} exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
            return 1
        dumps.append(dump_path)
        sitedumps.append(site_dump_path)

    dump_identical = filecmp.cmp(dumps[0], dumps[1], shallow=False)
    site_identical = filecmp.cmp(sitedumps[0], sitedumps[1], shallow=False)
    print(f"RESULT: base dump byte-identical across 2 runs: {dump_identical}")
    print(f"RESULT: site dump byte-identical across 2 runs: {site_identical}")
    print(f"same-device determinism: {'PASS' if dump_identical and site_identical else 'FAIL'}")
    return 0 if dump_identical and site_identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
