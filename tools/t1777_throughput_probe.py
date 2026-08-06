"""T-1777 throughput probe (scratch, not committed as a durable tool): times
the float-side per-position/per-layer capture (t1740_pooled_float_dump.py's
own capture_all_positions, imported and reused verbatim -- not reimplemented)
on a handful of short documents, loading the HF model ONCE per process rather
than once per document (t1740_pooled_float_dump.py's own __main__ reloads the
~1.5B-param checkpoint every invocation, which is fine for 3 prompts but is
the dominant cost at corpus scale). Also times the int8 engine side via
subprocess, matching the existing sslm_layer_trace-family invocation pattern.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import t1740_pooled_float_dump as fd  # noqa: E402

MODEL_PATH = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER_PATH = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL_PATH = fd._resolve_default_model(fd.DEFAULT_MODEL)
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
EXE = REPO_ROOT / "out" / "t1740_pooled_trace.exe"


def build_prompt(user_text: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")


def main():
    docs = []
    with open(r"D:\Wizard\Claude\Docs\spike\shopkeeper_corpus_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec["utterance"] if "utterance" in rec else " ".join(rec["turns"]))
    sample = docs[:5]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(FLOAT_MODEL_PATH), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(FLOAT_MODEL_PATH), local_files_only=True, torch_dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"float model+tokenizer load: {load_s:.2f}s device={device}")

    float_times = []
    tok_counts = []
    for text in sample:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated["input_ids"].to(device)
        n_pos = input_ids.shape[1]
        t1 = time.perf_counter()
        captured = fd.capture_all_positions(model, input_ids, device)
        fd.endpoint_self_check_last_position(model, captured, input_ids, device)
        dt = time.perf_counter() - t1
        float_times.append(dt)
        tok_counts.append(n_pos)
        print(f"  float doc n_pos={n_pos} capture+selfcheck={dt:.3f}s")

    print(f"float per-doc marginal mean={sum(float_times)/len(float_times):.3f}s "
          f"min={min(float_times):.3f}s max={max(float_times):.3f}s "
          f"tok_counts={tok_counts}")

    engine_times = []
    for i, text in enumerate(sample):
        prompt = build_prompt(text)
        dump_path = REPO_ROOT / "out" / "t1777" / f"probe_{i}.int8.bin"
        cmd = [str(EXE), str(MODEL_PATH), str(TOKENIZER_PATH), prompt, "--dump", str(dump_path)]
        t2 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        dt = time.perf_counter() - t2
        engine_times.append(dt)
        ok = proc.returncode == 0
        print(f"  engine doc {i} ok={ok} time={dt:.3f}s")
        if not ok:
            print(proc.stdout[-2000:]); print(proc.stderr[-2000:])

    print(f"engine per-doc (incl. model load+marshal each invocation) mean="
          f"{sum(engine_times)/len(engine_times):.3f}s min={min(engine_times):.3f}s max={max(engine_times):.3f}s")


if __name__ == "__main__":
    main()
