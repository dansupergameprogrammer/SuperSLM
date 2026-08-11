#!/usr/bin/env python3
"""T-1909 -- VRAM headroom + s/doc probe for Qwen2.5-3B-Instruct, before committing to the
full 239-document, 3-arm (int8 engine / float-bf16 / float-fp32) capture.

DISPOSABLE. Reuses tools/t1740_pooled_float_dump.py's own capture_all_positions and
endpoint_self_check_last_position VERBATIM (imported, not reimplemented) -- the same
functions tools/t1777_pooled_float_dump_batch.py calls for the real capture -- pointed at
the merged single-file 3B checkpoint T-1911 produced instead of the 1.5B DEFAULT_MODEL path.
This is "running an existing tool against a new checkpoint" (T-1909 brief SS5's carve-out),
not new capture logic.

Measures, for ONE representative document from the frozen T-1777 corpus, at each of the two
float precisions the real capture needs (bf16 = this checkpoint's actual compute dtype,
fp32 = the reference-self-consistency baseline's other side):
  - peak CUDA memory allocated during model load + one forward capture
  - wall-clock for model load and for the one-document forward capture

so the s/doc figure and VRAM headroom can be reported and a full-corpus wall-clock
projected BEFORE the long run is committed to, per the brief.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement")
sys.path.insert(0, str(REPO_ROOT / "tools"))
import t1740_pooled_float_dump as fd  # noqa: E402  (reused verbatim, not reimplemented)

MERGED_3B = Path(r"D:\SuperSLM\.worktrees\t1911-merge-checkpoint-shards\out\qwen2.5-3b-instruct-merged")
CORPUS = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out\t1777_corpus\docs.jsonl")
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def sample_docs(n=6):
    docs = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    docs.sort(key=lambda d: len(d["text"]))
    if len(docs) <= n:
        return docs
    # spread across the length distribution: shortest, longest, and evenly spaced between
    idxs = sorted(set(round(i * (len(docs) - 1) / (n - 1)) for i in range(n)))
    return [docs[i] for i in idxs]


def fmt(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.2f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def probe(dtype_label: str, torch_dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== {dtype_label} ===")
    if not torch.cuda.is_available():
        print("STOP: CUDA not available in this process")
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_3B), local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_3B), local_files_only=True, torch_dtype=torch_dtype)
        model.to(device)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"OOM during load/`.to(cuda)`: {exc}")
        return {"dtype": dtype_label, "oom": True}
    model.eval()
    t_load = time.perf_counter() - t0
    mem_after_load = torch.cuda.max_memory_allocated() / 2**30
    print(f"load+to(cuda): {fmt(t_load)}  peak_vram_after_load={mem_after_load:.3f} GiB")

    docs = sample_docs(6)
    doc_times = []
    doc_positions = []
    peak_vram_total = mem_after_load
    for doc in docs:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": doc["text"]}]
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated["input_ids"].to(device)
        n_positions = input_ids.shape[1]
        try:
            t1 = time.perf_counter()
            captured = fd.capture_all_positions(model, input_ids, device)
            fd.endpoint_self_check_last_position(model, captured, input_ids, device)
            t_doc = time.perf_counter() - t1
        except torch.cuda.OutOfMemoryError as exc:
            print(f"OOM during forward capture on doc={doc['label']}: {exc}")
            return {"dtype": dtype_label, "oom": True, "load_s": t_load,
                    "peak_vram_after_load_gib": mem_after_load, "n_positions": n_positions,
                    "docs_completed_before_oom": len(doc_times)}
        peak_vram_total = max(peak_vram_total, torch.cuda.max_memory_allocated() / 2**30)
        print(f"doc={doc['label']:<16} prompt_tokens={n_positions:<4} time={fmt(t_doc)}")
        doc_times.append(t_doc)
        doc_positions.append(n_positions)
        del captured
        torch.cuda.empty_cache()

    mean_doc_s = sum(doc_times) / len(doc_times)
    del model
    torch.cuda.empty_cache()

    print(f"mean s/doc over {len(doc_times)} sampled docs (token range "
          f"{min(doc_positions)}-{max(doc_positions)}): {mean_doc_s:.3f}s  "
          f"peak_vram_total={peak_vram_total:.3f} GiB")

    return {"dtype": dtype_label, "oom": False, "load_s": t_load, "mean_doc_s": mean_doc_s,
            "n_sampled": len(doc_times), "token_range": [min(doc_positions), max(doc_positions)],
            "peak_vram_after_load_gib": mem_after_load, "peak_vram_total_gib": peak_vram_total}


def main():
    import torch
    print(f"torch: {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  total_vram={props.total_memory / 2**30:.2f} GiB")

    results = {}
    results["bf16"] = probe("bfloat16 (compute dtype)", torch.bfloat16)
    results["fp32"] = probe("float32 (self-consistency reference)", torch.float32)

    print("\n=== SUMMARY ===")
    for k, r in results.items():
        print(k, "->", r)


if __name__ == "__main__":
    main()
