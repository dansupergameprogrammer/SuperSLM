#!/usr/bin/env python3
"""Popper probe 6 -- claim 1's independence, attacked by removal.

I1. SABOTAGE. Import the witness with every Qwen2 modeling symbol in the
    installed `transformers` replaced by an object that raises on any use
    (construction, call, or attribute access). If the witness still produces
    a byte-identical dump, no part of the library's Qwen2 composition is
    reachable from it. If it raises, the independence claim is false.
I2. IMPORT AUDIT. Record every module the witness actually loads, and check
    for `float_reference_layer_dump`, any `transformers.models.qwen2`
    modeling module, and any `.sslm` reader.
I3. THE UNCHECKED SHARED INPUT. The report tool's provenance check compares
    the FNV-1a hash of the prompt TEXT across arms. It never compares the
    TOKEN ID SEQUENCE. Print the witness's own token ids and length, so the
    int8 engine's own tokenization can be checked against them.
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

WITNESS = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\tools\independent_layer_reference.py")
MODEL = Path(r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
             r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
QUESTION = "What is 12 + 15? Give just the number."
BASELINE = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\out\t1690"
                r"\digit_symbol_spaced.witness_bfloat16.bin")

SABOTAGE = r'''
import sys, importlib
from pathlib import Path

class Poison:
    def __init__(self, name): self._n = name
    def __call__(self, *a, **k): raise RuntimeError(f"WITNESS TOUCHED LIBRARY SYMBOL {self._n}")
    def __getattr__(self, item): raise RuntimeError(f"WITNESS TOUCHED LIBRARY SYMBOL {self._n}.{item}")

import transformers
mod = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
poisoned = []
for name in dir(mod):
    if name.startswith("Qwen2") or name in ("apply_rotary_pos_emb", "repeat_kv", "eager_attention_forward",
                                            "rotate_half", "ALL_ATTENTION_FUNCTIONS"):
        setattr(mod, name, Poison(name))
        poisoned.append(name)
for name in dir(transformers):
    if name.startswith("Qwen2"):
        setattr(transformers, name, Poison("transformers." + name))
        poisoned.append("transformers." + name)
try:
    import transformers.cache_utils as cu
    for name in ("DynamicCache", "Cache", "StaticCache"):
        if hasattr(cu, name):
            setattr(cu, name, Poison("cache_utils." + name)); poisoned.append("cache_utils." + name)
except Exception as e:
    print("cache_utils poison skipped:", e)
print(f"POISONED {len(poisoned)} library symbols, e.g. {poisoned[:8]}")

sys.path.insert(0, str(Path(WITNESS_DIR)))
import independent_layer_reference as ilr
rc = ilr.main([QUESTION, "--system", SYSTEM, "--model", MODEL, "--dtype", "bfloat16", "--dump", DUMP])
print("witness exit:", rc)
loaded = sorted(m for m in sys.modules if "qwen2" in m.lower() or "float_reference" in m
                or "sslm" in m.lower())
print("SUSPECT MODULES LOADED:", loaded)
'''

TOKENS = r'''
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)["input_ids"]
print("prompt_text_repr:", repr(text))
print("n_tokens:", len(ids))
print("ids:", ids)
'''


def render(body: str) -> str:
    header = (
        f"WITNESS_DIR = r'{WITNESS.parent}'\n"
        f"MODEL = r'{MODEL}'\n"
        f"SYSTEM = {SYSTEM!r}\n"
        f"QUESTION = {QUESTION!r}\n"
        f"DUMP = r'{DUMP_PATH}'\n"
    )
    return header + body


DUMP_PATH = Path(tempfile.gettempdir()) / "popper_t1802_sabotage.bin"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    print("=== I1/I2. sabotage run ===")
    script = Path(tempfile.gettempdir()) / "popper_t1802_sabotage.py"
    script.write_text(render(SABOTAGE), encoding="utf-8")
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=1800)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-3000:])
    print("sabotage run exit:", p.returncode)
    if DUMP_PATH.exists():
        print("sabotaged dump sha256 :", sha(DUMP_PATH))
        print("packet   dump sha256  :", sha(BASELINE))
        print("BYTE-IDENTICAL:", sha(DUMP_PATH) == sha(BASELINE))

    print("\n=== I3. token ids the witness actually runs ===")
    script2 = Path(tempfile.gettempdir()) / "popper_t1802_tokens.py"
    script2.write_text(render(TOKENS), encoding="utf-8")
    p2 = subprocess.run([sys.executable, str(script2)], capture_output=True, text=True, timeout=600)
    print(p2.stdout)
    if p2.returncode != 0:
        print(p2.stderr[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
