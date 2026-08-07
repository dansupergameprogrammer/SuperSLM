#!/usr/bin/env python3
"""Popper probe 10 -- classify the held-out prompts the way the campaign
classifies its own.

"mechanism-2" is not a surface category: `tools/logit_margin_report.py`
groups the four campaign prompts as fail / control-arith / arith-other, and
the operational signal is a token-choice divergence between the int8 engine
and the float reference that is not explained by a near-tie. This probe
takes the first-position reading of that signal for every campaign prompt
and every held-out prompt: the int8 engine's own next token (printed by
`sslm_layer_trace.exe`'s self-check) against the float reference's argmax.
"""
import re
import subprocess
import sys
from pathlib import Path

T1690 = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness")
ROOT = Path(r"D:\SuperSLM\.worktrees\popper-t1802")
OUT = ROOT / "out" / "classify"
OUT.mkdir(parents=True, exist_ok=True)
EXE = T1690 / "out" / "sslm_layer_trace.exe"
SSLM = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOK = T1690 / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
                   r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

CAMPAIGN = [
    ("digit_symbol_spaced", "mech2", "What is 12 + 15? Give just the number."),
    ("digit_symbol_nospace", "mech2", "What is 12+15? Give just the number."),
    ("plain_language_plus", "mech2", "What is 12 plus 15? Give just the number."),
    ("digit_symbol_mult", "mech2", "What is 17 x 23? Give just the number."),
    ("capital_of_germany", "other5", "What is the capital of Germany?"),
    ("capital_of_france", "other5", "What is the capital of France?"),
    ("capital_of_japan", "other5", "What is the capital of Japan?"),
    ("largest_planet", "other5", "Name the largest planet in the solar system."),
    ("days_in_week", "other5", "How many days are in a week? Give just the number."),
]
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


def build_prompt(q):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def int8_token(label, q):
    p = subprocess.run([str(EXE), str(SSLM), str(TOK), build_prompt(q),
                        "--dump", str(OUT / f"{label}.tmp.bin")],
                       capture_output=True, text=True, timeout=1800)
    m = re.search(r"self_check:.*?\(token=(\d+)", p.stdout)
    if not m:
        raise SystemExit(f"{label}: no self_check token in\n{p.stdout}\n{p.stderr}")
    return int(m.group(1))


def float_tokens(label, q):
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "float_reference_logits.py"), q,
                        "--system", SYSTEM, "--model", str(FLOAT_MODEL), "--max-new", "4",
                        "--dump", str(OUT / f"{label}.logits.bin")],
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise SystemExit(f"{label}: float logits failed\n{p.stdout[-1500:]}{p.stderr[-1500:]}")
    m = re.search(r"output_ids: (.+)", p.stdout)
    return [int(x) for x in m.group(1).split()]


def main():
    print(f"  {'prompt':<22}{'packet group':<14}{'int8 tok':>10}{'float tok':>11}  agree?")
    for name, rows in (("CAMPAIGN", CAMPAIGN), ("HELD-OUT", HELDOUT)):
        print(f"\n--- {name} ---")
        for label, grp, q in rows:
            i8 = int8_token(label, q)
            fl = float_tokens(label, q)
            print(f"  {label:<22}{grp:<14}{i8:>10}{fl[0]:>11}  {'YES' if i8 == fl[0] else 'NO  <-- token-choice divergence'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
