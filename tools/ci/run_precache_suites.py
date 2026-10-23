#!/usr/bin/env python3
"""Run the precache CPU cells the cloud can run, for the branch-coverage job.

The two precache CPU suites, superslm_precache_e1 and superslm_precache_unpinned, also hold cells that read box
artifacts (real models, E0 artifacts, references, GPU captures) or the synthetic 0.5B/1.5B models, which are not
generated in the cloud. So this driver runs an allowlist, tools/ci/precache_coverage_cells.json: the cells that are
green on Linux with only tools/gen_paged_kv_fixture.py's four generated fixtures. Every listed cell must still be
registered (a stale entry fails), and each is run by its exact id. The cells a suite registers but the list leaves out
are printed and recorded in the receipt, so a new cell is seen there until it is listed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

SUITES = ("e1", "unpinned")
CELLS = Path(__file__).with_name("precache_coverage_cells.json")


def select_cells(registered: list[str], listed: list[str]) -> list[str]:
    if not registered or len(set(registered)) != len(registered):
        raise ValueError("cell inventory must be nonempty and contain unique ids")
    if not listed or len(set(listed)) != len(listed):
        raise ValueError("the allowlist must be nonempty and contain unique ids")
    if any(not cell or any(c.isspace() for c in cell) for cell in listed):
        raise ValueError("the allowlist contains an invalid id")
    missing = [cell for cell in listed if cell not in registered]
    if missing:
        raise ValueError(f"allowlisted cells absent from executable: {missing}")
    wanted = set(listed)
    return [cell for cell in registered if cell in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", required=True, type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    manifest = json.loads(CELLS.read_text(encoding="utf-8"))
    if {key for key in manifest if not key.startswith("_")} != set(SUITES):
        parser.error("the allowlist must name exactly the two precache CPU suites")
    env = os.environ.copy()
    fixture = args.fixture_dir or Path(env.get("SUPERSLM_PAGED_KV_FIXTURE_DIR", ""))
    if not all((fixture / f"{name}.sslm").is_file() for name in ("pkv_def", "pkv_qk", "pkv_odd", "pkv_32k")):
        parser.error("all four generated paged-KV fixtures are required")
    env["SUPERSLM_PAGED_KV_FIXTURE_DIR"] = str(fixture.resolve())
    plans = []
    for suite in SUITES:  # preflight every suite before running any cell
        binary = args.binary_dir.resolve() / f"superslm_precache_{suite}{'.exe' if os.name == 'nt' else ''}"
        inventory_env = dict(env, LLVM_PROFILE_FILE=os.devnull)
        inventory = subprocess.run([str(binary), "--list"], env=inventory_env, text=True, capture_output=True, check=True)
        registered = inventory.stdout.splitlines()
        try:
            selected = select_cells(registered, list(manifest[suite]))
        except ValueError as e:
            parser.error(f"{suite}: {e}")
        plans.append((suite, binary, registered, selected))
    receipt = {"suites": []}
    failed = False
    for suite, binary, registered, selected in plans:
        run_env = env.copy()
        if args.profile_dir:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            run_env["LLVM_PROFILE_FILE"] = str(args.profile_dir.resolve() / f"precache_{suite}.profraw")
        omitted = [cell for cell in registered if cell not in selected]
        print(f"{suite}: running {len(selected)}/{len(registered)} cells; not in the allowlist: {omitted}", flush=True)
        started = time.monotonic()
        result = subprocess.run([str(binary), *(f"={cell}" for cell in selected)], env=run_env)
        receipt["suites"].append({"suite": suite, "registered": registered, "selected": selected,
                                  "omitted": omitted, "exit_code": result.returncode,
                                  "elapsed_seconds": time.monotonic() - started})
        failed |= result.returncode != 0
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
