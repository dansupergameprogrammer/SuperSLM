"""T-2851 rows 4/6: scalar, SSE2 and AVX2 row and token identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from run_identity import R05, R15, RU, execute

HERE = Path(__file__).resolve().parent
FORCED = Path(r"D:\_t2956-mutants\forced")
BASE = HERE / "out" / "cell_real_decode_base.exe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("R05", "R15", "RU"))
    parser.add_argument("--tier", choices=("scalar", "sse2", "avx2"))
    args = parser.parse_args()
    failures = []
    for name, artifact, count, prompt in (
        ("R05", R05, 64, "-"),
        ("R15", R15, 64, "-"),
        ("RU", RU, 16, "0,1,2"),
    ):
        if args.only and name != args.only:
            continue
        baseline_cache = HERE / "out" / f"forced_baseline_{name}.json"
        identity_log = HERE / "out" / "identity" / f"{name}_cpu_base.txt"
        if baseline_cache.exists():
            expected = json.loads(baseline_cache.read_text())
        elif identity_log.exists():
            lines = [line for line in identity_log.read_text().splitlines()
                     if line.startswith("TOKENS ")]
            if len(lines) != 1:
                raise RuntimeError(f"bad baseline log: {identity_log}")
            expected = [int(token) for token in lines[0].split()[1:]]
            baseline_cache.write_text(json.dumps(expected))
        else:
            expected, _ = execute(BASE, "cpu", artifact, "-", count, 0, False, prompt)
            baseline_cache.write_text(json.dumps(expected))
        for tier in ("scalar", "sse2", "avx2"):
            if args.tier and tier != args.tier:
                continue
            out = FORCED / tier
            rows = subprocess.run([str(out / "cell_rows.exe"), str(artifact)], cwd=out,
                                  text=True, capture_output=True, timeout=120)
            if rows.returncode:
                raise RuntimeError(f"{name}/{tier} row exit={rows.returncode}: "
                                   f"{rows.stdout}{rows.stderr}")
            try:
                got, log = execute(out / "cell_real_decode.exe", "cpu", artifact, "-", count,
                                 4, False, prompt)
                for line in log.splitlines():
                    if line.startswith(("THREAD_EXIT", "THREADS")):
                        print(f"{name}/{tier} {line}", flush=True)
                if got != expected:
                    raise AssertionError(f"tokens={got} baseline={expected}")
                print(f"PASS {name} {tier} rows=103x9 tokens={len(got)}", flush=True)
            except (RuntimeError, AssertionError) as error:
                failures.append(f"{name}/{tier}: {error}")
                print(f"RED {name} {tier} row=PASS token_cell={error}", flush=True)
    if failures:
        raise RuntimeError(f"forced-tier red cells={len(failures)}")


if __name__ == "__main__":
    main()
