#!/usr/bin/env python3
"""Popper probe 9 -- how much does the one surviving mutation move the
headline?

M9 (removing the float32 upcast around the attention softmax) survives all
19 red-first checks, because every one of those checks runs its fixtures in
float64 where the upcast is inert. This probe runs the M9-mutated witness at
real checkpoint scale on four campaign prompts and reports the change in the
headline Spearman values -- i.e. the size of the defect the suite cannot
see."""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

T1690 = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness")
ROOT = Path(r"D:\SuperSLM\.worktrees\popper-t1802")
WORK = ROOT / "out" / "m9"
WORK.mkdir(parents=True, exist_ok=True)
FLOAT_MODEL = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
                   r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PROMPTS = {
    "digit_symbol_spaced": "What is 12 + 15? Give just the number.",
    "digit_symbol_mult": "What is 17 x 23? Give just the number.",
    "capital_of_france": "What is the capital of France?",
    "days_in_week": "How many days are in a week? Give just the number.",
}
OLD = "    weights_sm = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)"
NEW = "    weights_sm = torch.softmax(scores, dim=-1)"
LOCUS = list(range(21, 29))


def load_int8(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        codes = np.zeros((rows, hs), dtype=np.int8)
        for i in range(rows):
            struct.unpack("<qq", f.read(16))
            codes[i, :] = np.frombuffer(f.read(hs), dtype=np.int8)
    return codes


def load_float(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        d = np.frombuffer(f.read(), dtype=np.float32, count=rows * hs)
    return d.reshape(rows, hs)


def rho(a, b):
    r, _ = spearmanr(a.astype(np.float64), b.astype(np.float64))
    return float(r)


def main():
    src = (T1690 / "tools" / "independent_layer_reference.py").read_text(encoding="utf-8")
    assert src.count(OLD) == 1
    (WORK / "independent_layer_reference.py").write_text(src.replace(OLD, NEW), encoding="utf-8")

    for dt in ("bfloat16", "float32"):
        print(f"\n=== M9-mutated witness at real scale, dtype={dt} ===")
        print(f"  {'prompt':<22}{'layer':<7}{'clean rho':>11}{'M9 rho':>11}{'delta':>10}")
        for label, q in PROMPTS.items():
            dump = WORK / f"{label}.m9_{dt}.bin"
            if not dump.exists():
                p = subprocess.run([sys.executable, str(WORK / "independent_layer_reference.py"), q,
                                    "--system", SYSTEM, "--model", str(FLOAT_MODEL),
                                    "--dtype", dt, "--dump", str(dump)],
                                   capture_output=True, text=True, timeout=1800)
                if p.returncode != 0:
                    print(p.stdout[-2000:], p.stderr[-2000:])
                    return 1
            codes = load_int8(T1690 / "out" / "t1690" / f"{label}.int8.bin")
            clean = load_float(T1690 / "out" / "t1690" / f"{label}.witness_{dt}.bin")
            mut = load_float(dump)
            for l in (24, 28):
                c, m = rho(codes[l], clean[l]), rho(codes[l], mut[l])
                print(f"  {label:<22}L{l:<6}{c:11.4f}{m:11.4f}{m-c:+10.4f}")
            print(f"  {label:<22}rank agreement clean-vs-M9 at L28: {rho(clean[28], mut[28]):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
