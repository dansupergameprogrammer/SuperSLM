#!/usr/bin/env python3
"""T-1813 -- aggregate the k0-arm dual-run parity self-check from the per-
invocation logs written by tools/t1813_run_arms.py.

WHY THIS EXISTS (not a duplicate of the run script's own totals line): the
run script's "SELF-CHECK TOTALS" line sums only over invocations that
EXECUTED within that particular process, and this ticket's arms run was
resumed after an OOM kill (D-SLM1487) -- the resumed pass skipped every
already-complete k0/k4/k8 dump (717 of 956 invocations) via the driver's own
idempotent skip-on-exists logic, and executed fresh only the 161 remaining
`rot`-arm invocations, which never engage the dual-run path by construction
(gated on cfg.k==0, and rot's cfg.k is always 8 --
tools/t1797_residual_arms.cpp:465, 592, 741, 355-372, read at source). The
resumed pass's own totals line therefore correctly reported 0/0 for THAT
PASS and was wrongly read as "the check never ran" -- it ran, in the earlier
(killed) pass, and its per-invocation logs survived the kill. This script
reads the ground truth (every saved log file) rather than trusting either
pass's own in-process running total.

Usage:
    python tools/t1813_selfcheck_aggregate.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
LOG_DIR = REPO_ROOT / "out" / "t1813_logs"

LINE_RE = re.compile(
    r"self_check: norm dual-runs (\d+) mismatches (\d+); "
    r"reconcile dual-runs (\d+) mismatches (\d+)"
)


def aggregate_arm(arm: str) -> dict:
    logs = sorted((LOG_DIR / arm).glob("*.log"))
    n_logs = len(logs)
    n_with_line = 0
    norm_calls = norm_mismatch = recon_calls = recon_mismatch = 0
    missing = []
    for lp in logs:
        text = lp.read_text(encoding="utf-8", errors="replace")
        m = LINE_RE.search(text)
        if not m:
            missing.append(lp.name)
            continue
        n_with_line += 1
        nc, nm, rc, rm = (int(x) for x in m.groups())
        norm_calls += nc
        norm_mismatch += nm
        recon_calls += rc
        recon_mismatch += rm
    return {
        "arm": arm,
        "n_logs": n_logs,
        "n_logs_with_self_check_line": n_with_line,
        "logs_missing_self_check_line": missing,
        "norm_dual_runs": norm_calls,
        "norm_mismatches": norm_mismatch,
        "reconcile_dual_runs": recon_calls,
        "reconcile_mismatches": recon_mismatch,
    }


def main() -> int:
    out = {}
    for arm in ("k0", "k4", "k8", "rot"):
        r = aggregate_arm(arm)
        out[arm] = r
        print(f"arm={arm}: n_logs={r['n_logs']} (self_check line present in {r['n_logs_with_self_check_line']}) "
              f"norm dual-runs={r['norm_dual_runs']} mismatches={r['norm_mismatches']} "
              f"reconcile dual-runs={r['reconcile_dual_runs']} mismatches={r['reconcile_mismatches']}")
        if r["logs_missing_self_check_line"]:
            print(f"  WARNING: {len(r['logs_missing_self_check_line'])} logs missing the self_check line: "
                  f"{r['logs_missing_self_check_line'][:5]}...")
    (REPO_ROOT / "out" / "t1813_selfcheck_aggregate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
