#!/usr/bin/env python3
"""Popper probe 7 -- the shared input the packet's provenance check never
compares.

`t1690_witness_report.py::check_provenance` compares an FNV-1a hash of the
prompt TEXT across the three arms. It never compares the TOKEN ID SEQUENCE.
The witness forwards ids from `transformers.AutoTokenizer`; the int8 engine
forwards ids from its own `tokenizer.sslm` artifact. If those disagree on
any prompt, the two arms are running different token streams and every
Spearman value for that prompt compares different computations -- and the
mechanism-2 prompts are exactly the digit/symbol-dense ones where a BPE
disagreement is most likely.

Executed for all nine campaign prompts."""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"D:\SuperSLM\.worktrees\popper-t1802")
TOK_EXE = ROOT / "out" / "popper_tok_ids.exe"
TOK_SSLM = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\tests\fixtures\qwen2.5-1.5b.tok.sslm")
MODEL = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
             r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
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


def build_prompt(q):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    bad = []
    for label, q in PROMPTS.items():
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert text == build_prompt(q), f"{label}: chat template != engine-side hand-built prompt"
        hf_ids = tok(text, add_special_tokens=False)["input_ids"]

        tmp = Path(tempfile.gettempdir()) / f"popper_prompt_{label}.txt"
        tmp.write_bytes(text.encode("utf-8"))
        p = subprocess.run([str(TOK_EXE), str(TOK_SSLM), str(tmp)],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            print(f"{label}: tok exe failed\n{p.stdout}{p.stderr}")
            return 1
        lines = p.stdout.strip().splitlines()
        n = int(lines[0].split(":")[1])
        sslm_ids = [int(x) for x in lines[1].split(":")[1].split()]
        same = sslm_ids == hf_ids
        if not same:
            bad.append(label)
            first = next((i for i, (a, b) in enumerate(zip(sslm_ids, hf_ids)) if a != b), min(n, len(hf_ids)))
            print(f"  {label:<22} MISMATCH  hf_n={len(hf_ids)} sslm_n={n}  first divergence at index {first}: "
                  f"hf={hf_ids[first:first+6]} sslm={sslm_ids[first:first+6]}")
        else:
            print(f"  {label:<22} identical  n={n}")
    print(f"\nprompts with divergent tokenization: {bad if bad else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
