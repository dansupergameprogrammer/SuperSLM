"""TE-425 -- runs the SuperSLM 1.8.0 GPU allocation-status red suite's jobs and aggregates their verdicts.

Usage:
  run_red_suite.py --bin DIR --out DIR [--only PATTERN[,PATTERN...]] [--list] [--summary]
                   --qwen3 PATH --r15 PATH --adapter PATH --g5 PATH
  --bin      a build_red_suite.bat output directory (te425_cells.exe, te425_r13_ordinals.exe,
             cell_alloc_faults.exe, shaders\\)
  --out      where each job's output is written, one <job>.txt per job
  --only     fnmatch patterns over job names; default every job
  --summary  do not run anything: aggregate the job outputs already in --out

Each job is one process, bounded well under ten minutes, so a partial run can be resumed by name. A job
that a previous run already completed is re-run only when named explicitly.

Verdicts (te425_cells.exe's contract): GREEN every leg conforms to plan Sec3.3; RED at least one does not
(the reading this suite exists to give at v1.7.1); INVALID the job could not decide (a hook that never
fired, a selector that did not resolve, a setup failure); UNCONSTRUCTIBLE the dimension-10 ballast made
no engine allocation fail.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import time
from pathlib import Path

# Legs that must run alone in a fresh process: te425_cells.exe hooks --list prints isolate=1 for them.
ISOLATED_HOOK_LEGS = [
    "P.close1.A2.oom", "P.setevent1.oom", "P.setevent1.fail", "P.setevent2.fail",
    "F.pso1.oom", "F.pso1.fail", "F.rootsig1.oom", "F.rootsig1.removed",
    "D.close1.A2.oom", "D.setevent1.fail",
    "C.close1.A2.oom", "C.signal1.oom", "M.signal1.oom",
    "R.signal2.oom", "R.close2.A2.oom",
    "L.setevent1.oom", "S.setevent1.fail",
]
REMOVED_SITES = ["prompt_window", "decode_window", "prompt_close", "prompt_readback", "restore_reset",
                 "seq_create_map", "devlogits", "context_queue", "map"]


def jobs(a) -> list[tuple[str, list[str], str]]:
    """(name, argv after the executable, executable) for every job, in run order."""
    Q, R, A, G = f"--qwen3={a.qwen3}", f"--r15={a.r15}", f"--adapter={a.adapter}", f"--g5={a.g5}"
    out = str(Path(a.out))
    J: list[tuple[str, list[str], str]] = []
    cells = "te425_cells.exe"
    J.append(("r13", [], "te425_r13_ordinals.exe"))
    J.append(("refgen", ["refgen", Q, "--len=5", f"--ref={out}\\ref_prompt5.bin"], cells))
    for n in (1, 5, 40):
        J.append((f"r2-len{n}", ["r2", Q, f"--len={n}"], cells))
    for n in (1, 5):
        J.append((f"r3-len{n}", ["r3", Q, f"--len={n}"], cells))
    for lo in range(1, 701, 140):
        J.append((f"r3-len40-k{lo:03d}", ["r3", Q, "--len=40", f"--from={lo}", f"--to={lo + 139}"], cells))
    J.append(("r3len-len5-edges", ["r3len", Q, "--len=5", "--select=window-edges"], cells))
    J.append(("r10-len40-edges", ["r10", Q, "--len=40", "--select=window-edges"], cells))
    J.append(("r10-len5-edges", ["r10", Q, "--len=5", "--select=window-edges"], cells))
    J.append(("r10dec-edges", ["r10dec", Q, "--path=step", "--select=window-edges"], cells))
    for n in (1, 5):
        J.append((f"r4seam-len{n}", ["r4seam", G, f"--len={n}"], cells))
        J.append((f"r4new-len{n}", ["r4new", G, f"--len={n}"], cells))
    for path in ("step", "bridge"):
        J.append((f"r5seam-{path}", ["r5seam", Q, f"--path={path}"], cells))
        J.append((f"r5new-{path}", ["r5new", Q, f"--path={path}"], cells))
    J.append(("r6", ["r6", Q], cells))
    J.append(("r7cso", ["r7cso", Q, f"--tmp={out}\\tmp"], cells))
    J.append(("hooks-qwen3", ["hooks", "--group=qwen3", Q], cells))
    J.append(("hooks-r15", ["hooks", "--group=r15", R, A], cells))
    J.append(("hooks-g5", ["hooks", "--group=g5", G], cells))
    for leg in ISOLATED_HOOK_LEGS:
        J.append((f"hook-{leg}", ["hooks", f"--only={leg}", Q, R, A, G, f"--ref={out}\\ref_prompt5.bin"], cells))
    for route in ("prefill", "decode", "batch", "schema", "restore"):
        art = G if route == "schema" else Q
        ref = f"--ref={out}\\ref_e7_{route}.bin"
        J.append((f"e7ref-{route}", ["e7ref", f"--route={route}", art, ref], cells))
        J.append((f"e7-alloc-{route}", ["e7", f"--route={route}", "--kind=alloc", art, ref], cells))
        J.append((f"e7-other-{route}", ["e7", f"--route={route}", "--kind=other", art, ref], cells))
        if route in ("prefill", "restore"):
            J.append((f"e7-queue_oom-{route}", ["e7", f"--route={route}", "--kind=queue_oom", art, ref], cells))
    J.append(("e7direct", ["e7direct", Q, f"--ref={out}\\ref_e7_prefill.bin"], cells))
    for src in ("seam", "new", "close"):
        J.append((f"r14-{src}", ["r14", Q, f"--src={src}"], cells))
    for s in ("qwen3", "adapter", "g5"):
        J.append((f"r15zero-{s}", ["r15zero", f"--set={s}", Q, R, A, G], cells))
    for c in ("embed", "read", "finish", "finish_devlogits"):
        J.append((f"r15sweep-{c}", ["r15sweep", f"--call={c}", Q, R], cells))
    J.append(("r1ctx", ["r1ctx", Q], cells))
    for c in ("context", "seq_create", "restore"):
        J.append((f"r1new-{c}", ["r1new", f"--call={c}", Q], cells))
    J.append(("r1new-map0", ["r1new", "--call=map0", Q, "--retry-every=25"], cells))
    J.append(("r1new-map_head", ["r1new", "--call=map_head", R, "--retry-every=25"], cells))
    J.append(("r1new-adapter", ["r1new", "--call=adapter", R, A, "--retry-every=5"], cells))
    J.append(("cell_alloc_faults", [a.r15, a.adapter], "cell_alloc_faults.exe"))
    for site in REMOVED_SITES:
        J.append((f"removed-{site}", ["removed", f"--site={site}", Q, R], cells))
    for lo in (0, 16, 32):
        J.append((f"r8-{lo:02d}", ["r8", Q, f"--start={3072 + 128 * lo}", "--step=128", "--steps=16"], cells))
    J.append(("ballast", ["ballast", Q, "--margin-mib=512"], cells))
    return J


VERDICT = re.compile(r"^VERDICT (\S+) (\S+)", re.M)
SUMMARY = re.compile(r"^SUMMARY (\S+) legs=(\d+) conform=(\d+) nonconform=(\d+) invalid=(\d+)", re.M)


def summarize(out: Path, names: list[str]) -> int:
    rows = []
    for name in names:
        p = out / f"{name}.txt"
        if not p.is_file():
            rows.append((name, "NOT-RUN", "", ""))
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        v = VERDICT.findall(text)
        s = SUMMARY.findall(text)
        verdict = v[-1][1] if v else ("RED" if "FAIL" in text and name == "cell_alloc_faults" else "NO-VERDICT")
        if name == "cell_alloc_faults":
            m = re.search(r"^FAIL (\d+) of (\d+) faulted", text, re.M)
            verdict = "RED" if m else ("GREEN" if "PASS all counted" in text else "NO-VERDICT")
            legs = f"{m.group(2)} faulted, {m.group(1)} nonconforming" if m else ""
        else:
            legs = f"legs={s[-1][1]} conform={s[-1][2]} nonconform={s[-1][3]} invalid={s[-1][4]}" if s else ""
        w = re.findall(r"^WALL \S+ ([\d.]+) s", text, re.M)
        rows.append((name, verdict, legs, (w[-1] + " s") if w else ""))
    width = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{width}}  {r[1]:<15} {r[2]:<55} {r[3]}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    print("TOTAL " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--qwen3", default="")
    ap.add_argument("--r15", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--g5", default="")
    ap.add_argument("--timeout", type=int, default=560)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    all_jobs = jobs(a)
    pats = [p.strip() for p in a.only.split(",") if p.strip()]
    chosen = [j for j in all_jobs if any(fnmatch.fnmatch(j[0], p) for p in pats)]
    if a.list:
        for name, argv, exe in chosen:
            print(name, exe, " ".join(argv))
        return 0
    if a.summary:
        return summarize(a.out, [j[0] for j in chosen])
    for name, argv, exe in chosen:
        t0 = time.time()
        cmd = [str(a.bin / exe)] + argv
        try:
            r = subprocess.run(cmd, cwd=a.bin, capture_output=True, text=True, timeout=a.timeout,
                               encoding="utf-8", errors="replace")
            text = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else "") + f"\nEXIT {r.returncode}\n"
        except subprocess.TimeoutExpired as e:
            text = (e.stdout or "") + f"\nTIMEOUT after {a.timeout} s\nVERDICT {name} INVALID\n"
            if isinstance(text, bytes):
                text = text.decode("utf-8", "replace")
        (a.out / f"{name}.txt").write_text(f"$ {' '.join(cmd)}\n{text}", encoding="utf-8")
        v = VERDICT.findall(text)
        print(f"{name}: {v[-1][1] if v else 'no verdict'} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
