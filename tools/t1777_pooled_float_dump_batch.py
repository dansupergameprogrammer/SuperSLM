#!/usr/bin/env python3
"""T-1777 scratch instrument: float-reference per-position, per-layer
capture over MANY documents in one process (loads the ~1.5B-param HF
checkpoint ONCE), reusing tools/t1740_pooled_float_dump.py's own
capture_all_positions and endpoint_self_check_last_position verbatim
(imported, not reimplemented) -- this is the float-side counterpart to
tools/t1777_pooled_trace_batch.cpp, which does the identical thing for the
int8 engine side. tools/t1740_pooled_float_dump.py's own __main__ reloads the
checkpoint every invocation, which is fine for T-1740's own three prompts but
is the dominant per-document cost at the corpus scale this ticket needs (a
throughput probe measured ~3.5s of one-time load and ~3.3s of per-document
marginal cost when loaded once vs. reloading every time).

Usage
-----
    python tools\\t1777_pooled_float_dump_batch.py --docs out\\t1777\\docs.jsonl \\
        --out-dir out\\t1777

--docs is a JSONL file, one document per line: {"label": ..., "text": ...}
(the raw user-turn text; this script applies the same chat template
tools/t1740_pooled_float_dump.py's own __main__ applies). Writes
<out-dir>/<label>.float.bin per document, identical binary layout to
tools/t1740_pooled_float_dump.py's own dump.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import t1740_pooled_float_dump as fd  # noqa: E402  (reused verbatim, not reimplemented)

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=str(fd.DEFAULT_MODEL))
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float32"],
        help="'auto' resolves to this checkpoint's config.json torch_dtype (bfloat16, verified by "
             "execution -- T-1782/D-SLM1212-1225: the float32 the sibling scripts mention is the "
             "SERIALIZATION dtype, not the forward-pass compute dtype). 'float32' forces an fp32 "
             "forward, for measuring how much of any engine-vs-reference disagreement is actually "
             "reference-precision noise rather than engine error.")
    args = parser.parse_args(argv)

    docs = []
    with open(args.docs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = fd._resolve_default_model(Path(args.model))
    torch_dtype = torch.float32 if args.dtype == "float32" else "auto"
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"model+tokenizer load: {time.perf_counter() - t0:.2f}s device={device} documents={len(docs)} "
          f"resolved_model_dtype={model.dtype}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_failed = 0
    t_start = time.perf_counter()
    for i, doc in enumerate(docs):
        label = doc["label"]
        text = doc["text"]
        t_doc = time.perf_counter()
        messages = [{"role": "system", "content": args.system}, {"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated["input_ids"].to(device)
        n_positions = input_ids.shape[1]

        try:
            captured = fd.capture_all_positions(model, input_ids, device)
            fd.endpoint_self_check_last_position(model, captured, input_ids, device)
        except AssertionError as e:
            print(f"FAILED label={label}: {e}", flush=True)
            n_failed += 1
            continue

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        fingerprint = fd.fnv1a64(prompt_text)
        dump_path = out_dir / f"{label}.float.bin"
        with open(dump_path, "wb") as fbin:
            fbin.write(struct.pack("<QQQQ", n_positions, n_layers + 1, hidden_size, fingerprint))
            for pos in range(n_positions):
                for idx in range(n_layers + 1):
                    fbin.write(captured[idx][pos].cpu().numpy().astype("float32").tobytes())
        del captured
        torch.cuda.empty_cache()
        n_ok += 1
        dt = time.perf_counter() - t_doc
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(docs)}] label={label} n_pos={n_positions} this_doc={dt:.2f}s "
                  f"elapsed={elapsed:.1f}s avg={elapsed/(i+1):.2f}s/doc", flush=True)

    total = time.perf_counter() - t_start
    print(f"batch_done: {n_ok} ok, {n_failed} failed (of {len(docs)}), capture_total={total:.1f}s "
          f"avg={total/max(1,len(docs)):.2f}s/doc", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
