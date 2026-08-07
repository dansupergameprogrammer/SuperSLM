#!/usr/bin/env python3
"""Popper probe 8 -- overfit attack. The nine campaign prompts and the 4/5
grouping were both fixed before T-1690 ran; every number in the packet is
measured on that same tuned-on population. This probe brings its own
held-out prompts, chosen here and never seen by the campaign, and asks
whether the arithmetic/factual separation reproduces.

Group A (held out, arithmetic with digit-symbol surface): predicted LOW
Group B (held out, factual/plain): predicted HIGH
Group C (probes of what the grouping is actually about -- spelled-out
         arithmetic, and a digit-answer factual question)
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

T1690 = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness")
ROOT = Path(r"D:\SuperSLM\.worktrees\popper-t1802")
OUT = ROOT / "out" / "heldout"
OUT.mkdir(parents=True, exist_ok=True)
EXE = T1690 / "out" / "sslm_layer_trace.exe"
SSLM = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOK = T1690 / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
                   r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

HELDOUT = [
    ("ho_add_34_51", "A", "What is 34 + 51? Give just the number."),
    ("ho_add_9_8", "A", "What is 9+8? Give just the number."),
    ("ho_mult_6_7", "A", "What is 6 x 7? Give just the number."),
    ("ho_sub_100_37", "A", "What is 100 - 37? Give just the number."),
    ("ho_capital_italy", "B", "What is the capital of Italy?"),
    ("ho_smallest_planet", "B", "Name the smallest planet in the solar system."),
    ("ho_largest_ocean", "B", "What is the largest ocean on Earth?"),
    ("ho_colour_sky", "B", "What colour is a clear daytime sky?"),
    ("ho_spelled_arith", "C", "What is twelve plus fifteen? Give just the number."),
    ("ho_months_in_year", "C", "How many months are in a year? Give just the number."),
]
LOCUS = list(range(21, 29))


def build_prompt(q):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def load_int8(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        codes = np.zeros((rows, hs), dtype=np.int8)
        for i in range(rows):
            struct.unpack("<qq", f.read(16))
            codes[i, :] = np.frombuffer(f.read(hs), dtype=np.int8)
    return codes, fp


def load_float(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        d = np.frombuffer(f.read(), dtype=np.float32, count=rows * hs)
    return d.reshape(rows, hs), fp


def rho(a, b):
    r, _ = spearmanr(a.astype(np.float64), b.astype(np.float64))
    return float(r)


def main():
    vals = {}
    for label, group, q in HELDOUT:
        i8 = OUT / f"{label}.int8.bin"
        if not i8.exists():
            p = subprocess.run([str(EXE), str(SSLM), str(TOK), build_prompt(q), "--dump", str(i8)],
                               capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                print(f"{label} int8 FAILED\n{p.stdout}{p.stderr}")
                return 1
        for dt in ("float32", "bfloat16"):
            w = OUT / f"{label}.witness_{dt}.bin"
            if not w.exists():
                p = subprocess.run([sys.executable, str(T1690 / "tools" / "independent_layer_reference.py"),
                                    q, "--system", SYSTEM, "--model", str(FLOAT_MODEL),
                                    "--dtype", dt, "--dump", str(w)],
                                   capture_output=True, text=True, timeout=1800)
                if p.returncode != 0:
                    print(f"{label} witness {dt} FAILED\n{p.stdout}{p.stderr}")
                    return 1
        codes, ifp = load_int8(i8)
        vals[label] = {}
        for dt in ("float32", "bfloat16"):
            wv, wfp = load_float(OUT / f"{label}.witness_{dt}.bin")
            assert wfp == ifp, f"{label}/{dt} fingerprint mismatch"
            vals[label][dt] = {l: rho(codes[l], wv[l]) for l in LOCUS}
        print(f"  ran {label}")

    for dt in ("float32", "bfloat16"):
        print(f"\n=== HELD-OUT, dtype={dt}: spearman(int8, witness) ===")
        print(f"  {'prompt':<20}{'grp':<5}" + "".join(f"L{l:<6}" for l in LOCUS))
        for label, group, _ in HELDOUT:
            print(f"  {label:<20}{group:<5}" + "".join(f"{vals[label][dt][l]:7.4f}" for l in LOCUS))
        A = [l for l, g, _ in HELDOUT if g == "A"]
        B = [l for l, g, _ in HELDOUT if g == "B"]
        print("\n  per-layer gap on HELD-OUT prompts only (min(B) - max(A)):")
        for l in LOCUS:
            a = [vals[x][dt][l] for x in A]
            b = [vals[x][dt][l] for x in B]
            print(f"    L{l}: A(arith) [{min(a):.4f},{max(a):.4f}]  B(factual) [{min(b):.4f},{max(b):.4f}]"
                  f"  gap={min(b)-max(a):+.4f}")
        print("  group C probes at L28: " + ", ".join(
            f"{x}={vals[x][dt][28]:.4f}" for x, g, _ in HELDOUT if g == "C"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
