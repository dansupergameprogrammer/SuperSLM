#!/usr/bin/env python3
"""T-1813 -- Route A driver, run phase.

Invokes the EXISTING, UNMODIFIED t1797_residual_arms.exe (built from
D:\\SuperSLM\\.worktrees\\t1797-rotation-solve\\tools\\t1797_residual_arms.cpp,
commit 464b71d, per Claude/Vitruvius/t1812-rotation-grading-route-2026-08-07.md
Route A) once per (document, arm) over T-1777's own frozen 239-document corpus.
No production file is touched. No file under the t1797/t1777 worktrees is
written by this script -- every dump lands under THIS worktree's own out/
directory.

This is a subprocess-invocation wrapper only (per the route's own "no new C++
strictly required" note); it does not reimplement any of the tool's
arithmetic. Parallelized across a process pool because the tool is
scalar/single-threaded per its own header comment, and the corpus x arms
product (239 x 4 = 956 invocations) is too slow to run serially in one
session (measured: ~16s/invocation including model load).

Usage:
    python tools/t1813_run_arms.py --arms k0 k4 k8 rot --workers 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

T1777_CORPUS = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out\t1777_corpus")
EXE = Path(r"D:\SuperSLM\.worktrees\t1797-rotation-solve\out\t1797_residual_arms.exe")
MODEL = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER = Path(r"D:\SuperSLM\.worktrees\t1797-rotation-solve\tests\fixtures\qwen2.5-1.5b.tok.sslm")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DUMP_ROOT = REPO_ROOT / "out" / "t1813_arms"
LOG_DIR = REPO_ROOT / "out" / "t1813_logs"
MASTER_LOG = REPO_ROOT / "out" / "t1813_run.log"


def read_prompts_tsv(path: Path) -> list[tuple[str, str]]:
    """Returns [(label, prompt_text)], reversing the '<NL>' placeholder back
    to real newlines -- the same reversal t1777_pooled_trace_batch.cpp's own
    ReadPromptsTsv performs (read at source, tools/t1777_pooled_trace_batch.cpp:112-116)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, text = line.split("\t", 1)
            out.append((label, text.replace("<NL>", "\n")))
    return out


def run_one(label: str, prompt: str, arm: str) -> tuple[str, str, int, str]:
    """Runs one (label, arm) invocation. Returns (label, arm, returncode, stdout+stderr)."""
    dump_dir = DUMP_ROOT / arm
    dump_dir.mkdir(parents=True, exist_ok=True)
    bin_path = dump_dir / f"{label}_{arm}.bin"
    if bin_path.exists() and bin_path.stat().st_size > 0:
        return (label, arm, 0, "SKIPPED (dump already present)")
    proc = subprocess.run(
        [str(EXE), str(MODEL), str(TOKENIZER), prompt, label, "--arm", arm,
         "--dump-dir", str(dump_dir)],
        capture_output=True, text=True, timeout=300,
    )
    log_path = LOG_DIR / arm / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n---stderr---\n" + proc.stderr, encoding="utf-8")
    return (label, arm, proc.returncode, proc.stdout + proc.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["k0", "k4", "k8", "rot"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N docs (smoke test)")
    args = ap.parse_args()

    if not EXE.exists():
        print(f"FATAL: exe not found at {EXE}", file=sys.stderr)
        return 2
    if not MODEL.exists():
        print(f"FATAL: model not found at {MODEL}", file=sys.stderr)
        return 2

    docs = read_prompts_tsv(T1777_CORPUS / "prompts.tsv")
    if args.limit:
        docs = docs[: args.limit]
    DUMP_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(label, prompt, arm) for arm in args.arms for (label, prompt) in docs]
    total = len(jobs)

    with open(MASTER_LOG, "a", buffering=1, encoding="utf-8") as mlog:
        mlog.write(f"\n=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"arms={args.arms} docs={len(docs)} total_jobs={total} workers={args.workers} ===\n")
        mlog.flush()
        done = 0
        failures = []
        norm_calls_total = 0
        norm_mismatch_total = 0
        reconcile_calls_total = 0
        reconcile_mismatch_total = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, label, prompt, arm) for (label, prompt, arm) in jobs]
            for fut in as_completed(futs):
                label, arm, rc, out = fut.result()
                done += 1
                if rc != 0:
                    failures.append((label, arm, rc))
                    mlog.write(f"[{done}/{total}] FAIL label={label} arm={arm} rc={rc}\n{out}\n")
                else:
                    # parse self_check line if present (only meaningful for k0, but
                    # printed -- 0/0 -- for every arm; we sum across all invocations)
                    for line in out.splitlines():
                        if line.startswith("self_check:"):
                            # self_check: norm dual-runs N mismatches N; reconcile dual-runs N mismatches N
                            parts = line.replace(";", "").split()
                            try:
                                nc = int(parts[3]); nm = int(parts[5])
                                rc_ = int(parts[8]); rm = int(parts[10])
                                norm_calls_total += nc
                                norm_mismatch_total += nm
                                reconcile_calls_total += rc_
                                reconcile_mismatch_total += rm
                            except (IndexError, ValueError):
                                pass
                    mlog.write(f"[{done}/{total}] ok label={label} arm={arm}\n")
                if done % 20 == 0 or done == total:
                    elapsed = time.time() - t0
                    mlog.write(f"--- progress {done}/{total} elapsed={elapsed:.1f}s ---\n")
                    mlog.flush()
        elapsed = time.time() - t0
        mlog.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={elapsed:.1f}s "
                    f"failures={len(failures)} ===\n")
        mlog.write(f"=== SELF-CHECK TOTALS (summed ONLY over invocations that freshly executed IN "
                    f"THIS PASS -- skipped/resumed invocations from a prior pass contribute nothing "
                    f"here even though they carry their own self_check line in their saved log; the "
                    f"dual-run path is gated to the k0 arm only (cfg.k==0), so a pass that freshly "
                    f"executes no k0 invocations reports 0/0 here by construction, not because no "
                    f"parity check ran anywhere -- run tools/t1813_selfcheck_aggregate.py for the true "
                    f"corpus-wide total read from every saved per-invocation log): "
                    f"norm dual-runs={norm_calls_total} mismatches={norm_mismatch_total}; "
                    f"reconcile dual-runs={reconcile_calls_total} mismatches={reconcile_mismatch_total} ===\n")
        if failures:
            mlog.write(f"FAILURES: {failures}\n")
        mlog.flush()

    summary = {
        "total_jobs": total,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "self_check_totals": {
            "norm_dual_runs": norm_calls_total,
            "norm_mismatches": norm_mismatch_total,
            "reconcile_dual_runs": reconcile_calls_total,
            "reconcile_mismatches": reconcile_mismatch_total,
        },
    }
    (REPO_ROOT / "out" / "t1813_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
