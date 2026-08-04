#!/usr/bin/env python3
"""T-1697 step 0: capture `--site-dump`/`--dump` output for the campaign's own
9-prompt population, at all 28 rows, against the BASELINE artifact -- this
supplies the real captured `mlp_act` codes (down_proj's own real input, per
channel) and `down_proj.requant` records this task's SmoothQuant-style
derivation needs. Reuses `kv_saturation_report.POPULATION`/`build_prompt`
verbatim (the same established population every other campaign drive reads
from one source, T-1683 S3 precedent) and the SAME driver invocation shape
`tools/t1691_mlp_drive.py` already established -- no reimplementation of the
capture step, only a fresh run in this task's own worktree/out directory.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1697_population")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    for label, question, role in KSR.POPULATION:
        prompt = KSR.build_prompt(question)
        dump_path = os.path.join(OUT_DIR, f"{label}.dump")
        site_dump_path = os.path.join(OUT_DIR, f"{label}.sitedump")
        proc = subprocess.run(
            [str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH), prompt,
             "--dump", dump_path, "--site-dump", site_dump_path],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"FAILED: driver failed for prompt {label!r} (exit {proc.returncode})\n"
                  f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}", file=sys.stderr)
            return 1
        assert "self_check: production and manual-replay paths agree" in proc.stdout, (
            f"prompt {label!r}: self-check line missing from stdout")
        print(f"captured {label!r} role={role} -> {site_dump_path}")

    print("RESULT: all 9 prompts captured, all 28 rows, baseline artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
