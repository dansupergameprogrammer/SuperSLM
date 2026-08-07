#!/usr/bin/env python3
"""Popper probe 2 -- generate the library-based float reference (T-1686)
dump for all nine campaign prompts, so the T-1690 headline can be compared
against the arm it claims to corroborate, on this machine, today."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\SuperSLM\.worktrees\popper-t1802")
OUT = ROOT / "out" / "popper_t1802"
OUT.mkdir(parents=True, exist_ok=True)
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

PROMPTS = {
    "digit_symbol_spaced": "What is 12 + 15? Give just the number.",
    "digit_symbol_nospace": "What is 12+15? Give just the number.",
    "plain_language_plus": "What is 12 plus 15? Give just the number.",
    "digit_symbol_mult": "What is 17 x 23? Give just the number.",
    "capital_of_germany": "What is the capital of Germany?",
    "capital_of_france": "What is the capital of France?",
    "capital_of_japan": "What is the capital of Japan?",
    "largest_planet": "Name the largest planet in the solar system.",
    "days_in_week": "How many days are in a week? Give just the number.",
}

for label, q in PROMPTS.items():
    dump = OUT / f"{label}.libfloat.bin"
    if dump.exists():
        print(f"skip {label}")
        continue
    cmd = [sys.executable, str(ROOT / "tools" / "float_reference_layer_dump.py"), q,
           "--system", SYSTEM, "--dump", str(dump)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(f"{label}: exit {p.returncode}")
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        sys.exit(1)
print("done")
