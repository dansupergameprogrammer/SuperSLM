"""Freeze the scratch production-mutant readings into a tracked evidence manifest.

A mutant whose target is gone from the engine under test (its anchor lands 0 times, or its anchoring
function is gone) is recorded as `target_gone` with the reason, not as a crash: either its
mutation.txt holds apply_mutant.py's "expected one landing, found 0" refusal (run_mutant.ps1 stopped
at exit 2, so there is no cell.txt), or it has no log folder at all and apply_mutant.py --check
against --tree reports the miss. A missing log for a mutant whose target still lands is still an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LOGS = Path(r"D:\_t2956-mutants\logs")
IDS = ("a b b2 c c_exact d e f g h h2 i j k l1 l2 l3 l4 l5 l6 m n "
       "n2_upload n2_restore n3_upload n3_restore o o2 p q_cpu q_gpu").split()
LANDING = {
    "a": "forward_sites.cpp:2838",
    "b": "forward_sites.cpp:2848-2851",
    "b2": "forward_sites.cpp:2848-2851",
    "c": "forward_sites.cpp:2792-2802 (semantic proxy only)",
    "c_exact": "forward_sites.cpp:2792-2864 (task-local argmax, last-wins merge)",
    "d": "logits_site.hlsl:75",
    "e": "gpu_1p0.cpp:985-988",
    "f": "gpu_1p0.cpp:985-988",
    "g": "sslm_abi.cpp prefill path",
    "h": "gpu_1p0.cpp:1180",
    "h2": "gpu_1p0.cpp:1200-1207",
    "i": "gpu_1p0.cpp:2944-2951",
    "j": "gpu_1p0.cpp:851-858",
    "k": "gpu_1p0.cpp:851-858",
    "m": "gpu_1p0.cpp:1200-1207",
    "n": "gpu_1p0.cpp:890",
    "o": "gpu_1p0.cpp:922",
    "o2": "gpu_1p0.cpp:4142",
    "p": "d3d12_harness.h:412-418",
    "q_cpu": "sslm_abi.cpp:2592 (all-masked injection disconnected)",
    "q_gpu": "gpu_1p0.cpp:2970 (all-masked injection disconnected)",
}


def landing_for(ident: str) -> str:
    landing = LANDING.get(ident)
    if landing is None:
        if ident.startswith("l"):
            landing = "gpu_1p0.cpp:851, bundle allocation " + ident[1]
        elif ident.startswith("n2_") or ident.startswith("n3_"):
            landing = ("gpu_1p0.cpp:618-633" if ident.endswith("upload") else
                       "superslm_gpu.cpp:4587-4596")
        else:
            raise RuntimeError(f"{ident}: missing landing")
    return landing


def read_log(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith(b"\xff\xfe") else "utf-8",
                       errors="replace")


GONE_APPLY = re.compile(r"[^\r\n]*(?:expected one landing, found 0|function \S+ not found exactly once)[^\r\n]*")


def target_gone_reason(ident: str, folder: Path, tree: Path) -> str | None:
    """The reason `ident`'s target is gone, or None when it still lands (or the logs show a run)."""
    mutation_path = folder / "mutation.txt"
    if mutation_path.exists():
        if (folder / "cell.txt").exists():
            return None
        found = GONE_APPLY.search(read_log(mutation_path))
        return found.group(0).strip().removeprefix("RuntimeError: ") if found else None
    proc = subprocess.run([sys.executable, str(HERE / "apply_mutant.py"), ident, "--check",
                           "--tree", str(tree)], capture_output=True, text=True)
    gone = []
    for line in proc.stdout.splitlines():
        found = re.match(rf"^MUTANT {re.escape(ident)} MISS (.+?): (lands 0 times|function-not-found): (.*)$",
                         line)
        if found:
            what = ("expected one landing, found 0" if found.group(2) == "lands 0 times" else
                    "function not found exactly once")
            gone.append(f"{found.group(1)}: {what}: {found.group(3)}")
    if not gone:
        return None
    return f"apply_mutant.py --check at {tree}: " + "; ".join(gone)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=LOGS, help="the runner's log root (<logs>/<id>/...)")
    parser.add_argument("--tree", type=Path, default=HERE.parents[1],
                        help="engine tree apply_mutant.py --check reads for a mutant with no logs "
                             "(default: this checkout)")
    parser.add_argument("--out", type=Path, default=HERE / "mutant_evidence.json")
    parser.add_argument("--builder-commit", default=None,
                        help="the commit the mutant builds came from (default: --tree's HEAD)")
    args = parser.parse_args()
    builder = args.builder_commit or subprocess.run(
        ["git", "-C", str(args.tree), "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True).stdout.strip()
    results = []
    for ident in IDS:
        folder = args.logs / ident
        gone = target_gone_reason(ident, folder, args.tree.resolve())
        if gone is not None:
            results.append({"id": ident, "landing": landing_for(ident), "classification": "target_gone",
                            "reason": gone})
            continue
        mutation = read_log(folder / "mutation.txt")
        matches = re.findall(r"^MUTANT \S+ (\S+) sha256=([0-9a-f]{64})\r?$",
                             mutation, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"{ident}: missing single production-source digest")
        lines = read_log(folder / "cell.txt").splitlines()
        failed = [line.strip().split(" : ", 1)[-1] for line in lines
                  if re.search(r"(?:^|: )FAIL (?!HR )", line)]
        if ident.startswith("q_"):
            failed = [line.strip().split(" : ", 1)[-1] for line in lines
                      if "FAIL all-masked " in line]
        if not failed:
            raise RuntimeError(f"{ident}: no failing cell assertion")
        landing = landing_for(ident)
        results.append({"id": ident, "landing": landing, "source": matches[0][0],
                        "source_sha256": matches[0][1], "reading": failed[0],
                        "classification": "semantic proxy only" if ident == "c" else
                                          "executed exact task-local mutant killed" if ident == "c_exact" else
                                          "executed production mutant killed"})
    target = args.out
    target.write_text(json.dumps({"builder_commit":
        builder, "mutants": results},
        indent=2) + "\n", encoding="utf-8")
    gone_ids = [r["id"] for r in results if r["classification"] == "target_gone"]
    print(f"WROTE {target} mutants={len(results)} target_gone={len(gone_ids)}"
          + (f" ({' '.join(gone_ids)})" if gone_ids else ""))


if __name__ == "__main__":
    main()
