#!/usr/bin/env python3
r"""T-1822 fold 8 -- the site-1 oracle arm: the forgone refinement IN versus OUT.

The design of record (Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md,
Sec 8.5) refuses M1 grouping at site 1 (`embed`) and bounds the forgone prize at
<= 0.4341 quantum of the embedding table's own grid.  The step from that bound to "no
measurable oracle effect" had been an INFERENCE from a different cell (2.1's weights-only
+0.000000).  This tool replaces the inference with the direct measurement: two arms through
the T-1809 instrument, differing in exactly one variable -- whether site 1's landing keeps
the per-channel power-of-two refinement (G = 1, k_cap = 3, the refusal's own ceiling
setting) or lands on the row grid (the adopted ruling).

    arm C      site 1 on the ROW grid       (T-1809's arm C, re-run this session so the
                                             two arms share run conditions)
    arm C-s1r  site 1 on the REFINED grid   (per-channel k_i = largest k <= 3 with
                                             |x_i| * 2^k <= D'; the refinement is RETAINED
                                             through the landing -- the value a consumer
                                             with k[] in hand would read)

Everything else -- the other seventeen modelled sites, the K/V static scales, the float32
attention accumulation, the capture, the corpus -- is the T-1809 tool's own code, imported
unmodified from its worktree.  The ONLY difference between the two arms is the embed
landing hook, so the paired contrast is the site-1 refinement and nothing else.

Usage (from D:\SuperSLM\.worktrees\t1822-fold8):
    python tools\t1822_fold8_site1_oracle_arm.py --site1 row     --out-dir out\t1822_arm_c_rerun [...]
    python tools\t1822_fold8_site1_oracle_arm.py --site1 refined --out-dir out\t1822_arm_s1r    [...]
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

T1809_TOOLS_DEFAULT = r"D:\SuperSLM\.worktrees\t1809-activation-interaction\tools"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site1", required=True, choices=("row", "refined"),
                        help="row = the adopted ruling (arm C verbatim); "
                             "refined = the forgone per-channel refinement retained")
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--t1777-tools", required=True)
    parser.add_argument("--t1809-tools", default=T1809_TOOLS_DEFAULT)
    parser.add_argument("--spike-root", default=r"D:\Wizard\Tools")
    parser.add_argument(
        "--artifact-metadata",
        default=r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8\metadata.json")
    parser.add_argument("--k-cap", type=int, default=3,
                        help="site 1's own k_cap (design Sec 6.1: 3, the norm-consumer bound)")
    parser.add_argument("--embed-weights", choices=("bf16", "int8rt"), default="bf16",
                        help="bf16 = the checkpoint's own continuous embedding (the T-1809 "
                             "instrument's convention; site 1's landing then quantizes a "
                             "continuous row -- the INSTRUMENT's site-1 cell). int8rt = the "
                             "embedding table round-tripped through the engine's own per-tensor "
                             "symmetric int8 rule (pipeline._quantize_tensor: scale = "
                             "max|W|/127, round half away, clamp) -- the row the ENGINE's "
                             "site 1 actually re-grids, i.e. the design's Sec 8.5 cell, where "
                             "the reachable input domain is the already-int8 table")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1809_tools).resolve()))
    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1809_activation_arm_dump as t1809  # noqa: E402
    import t1740_pooled_float_dump as fd       # noqa: E402

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = []
    with open(args.docs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if args.limit is not None:
        docs = docs[: args.limit]

    model_arg = args.model if args.model is not None else str(fd.DEFAULT_MODEL)
    model_path = fd._resolve_default_model(Path(model_arg))
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype="auto",
        attn_implementation="eager")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"site1={args.site1} model+tokenizer load: {time.perf_counter() - t0:.2f}s "
          f"device={device} documents={len(docs)} dtype={model.dtype}", flush=True)

    if args.embed_weights == "int8rt":
        # The engine's own rule for the embed table (pipeline._quantize_tensor,
        # the non-projection branch): ONE scale per tensor, scale = max|W|/127,
        # codes = clamp(round_half_away(W/scale)), dequantized W' = codes*scale.
        # After this, every embedding row lies on the artifact's int8 grid, so
        # site 1's landing re-grids an already-8-bit signal -- the engine's own
        # site-1 input domain (design Sec 8.5: D' <= 127, injective re-gridding).
        w = model.model.embed_tokens.weight.detach()
        wf = w.to(torch.float32)
        peak = float(wf.abs().max())
        scale = peak / 127.0 if peak > 0 else 1.0
        codes = t1809._round_half_away_from_zero(wf / scale).clamp(-127, 127)
        deq = (codes * scale).to(w.dtype)
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(deq)
        n_moved = int((deq.to(torch.float32) != wf).sum())
        print(f"embed table round-tripped through the engine's per-tensor int8 rule: "
              f"scale {scale:.6g}, {n_moved} of {wf.numel()} values moved", flush=True)

    k_scales, v_scales = t1809.read_kv_landing_scales(
        args.artifact_metadata, model.config.num_hidden_layers,
        model.config.num_key_value_heads)

    # Arm C's forward, verbatim: weights float, activations quantized at every site.
    handles = t1809.install_forward(model, k_scales, v_scales, quantize_activations=True)
    # install_forward appends the embed hook FIRST, then 2 norm hooks per layer
    # (read at source, t1809_activation_arm_dump.py:303-307).  Asserted, not assumed:
    n_expected = 1 + 2 * model.config.num_hidden_layers
    assert len(handles) == n_expected, (len(handles), n_expected)

    if args.site1 == "refined":
        # Remove arm C's row-grid embed hook and install the refined landing.
        handles[0].remove()

        INT8_MAX = t1809.INT8_MAX
        k_cap = args.k_cap

        def refined_embed_hook(_module, _args, output):
            """Site 1's landing with the per-channel PoT refinement RETAINED.

            Design Sec 4.1 at G = 1, k_cap = 3: D'_g = |x_i| (C20's >= 1 guard is
            value-irrelevant at zero elements, which land at 0 on any grid);
            k_i = largest k <= k_cap with |x_i| * 2^k <= D'; the element is quantized
            on the grid D' / (127 * 2^k_i) and RECONSTRUCTED on it -- the value a
            consumer holding k[] would read.  Float32 throughout, the instrument's
            own convention (rt_dynamic's docstring).
            """
            xf = output.to(torch.float32)
            d = xf.abs().amax(dim=-1, keepdim=True)
            d = torch.where(d > 0, d, torch.ones_like(d))
            ax = xf.abs()
            k = torch.zeros_like(xf)
            for kk in range(1, k_cap + 1):
                k = torch.where(ax * float(1 << kk) <= d, float(kk), k)
            quantum = d / (INT8_MAX * torch.pow(torch.tensor(2.0, device=xf.device), k))
            q = t1809._round_half_away_from_zero(xf / quantum).clamp(-INT8_MAX, INT8_MAX)
            return (q * quantum).to(output.dtype)

        handles[0] = model.model.embed_tokens.register_forward_hook(refined_embed_hook)
        print(f"site-1 landing REPLACED: per-channel refined grid, G=1 k_cap={k_cap}",
              flush=True)
    else:
        print("site-1 landing: arm C's row grid, unmodified", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = 0
    t_start = time.perf_counter()
    for i, doc in enumerate(docs):
        label = doc["label"]
        t_doc = time.perf_counter()
        messages = [{"role": "system", "content": t1809.SYSTEM_PROMPT},
                    {"role": "user", "content": doc["text"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt")
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
        with open(out_dir / f"{label}.float.bin", "wb") as fbin:
            fbin.write(struct.pack("<QQQQ", n_positions, n_layers + 1, hidden_size,
                                   fingerprint))
            for pos in range(n_positions):
                for idx in range(n_layers + 1):
                    fbin.write(captured[idx][pos].cpu().numpy().astype("float32").tobytes())
        del captured
        if device == "cuda":
            torch.cuda.empty_cache()
        n_ok += 1
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(docs)}] label={label} n_pos={n_positions} "
                  f"this_doc={time.perf_counter() - t_doc:.2f}s elapsed={elapsed:.1f}s "
                  f"avg={elapsed/(i+1):.2f}s/doc", flush=True)

    total = time.perf_counter() - t_start
    print(f"batch_done: site1={args.site1} {n_ok} ok, {n_failed} failed (of {len(docs)}), "
          f"capture_total={total:.1f}s avg={total/max(1,len(docs)):.2f}s/doc", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
