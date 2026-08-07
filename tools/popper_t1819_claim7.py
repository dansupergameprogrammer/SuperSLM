#!/usr/bin/env python3
"""T-1819 attack on claim 7: "probability quantization sits 6.4x closer to a boundary
than uniform -- the largest deviation measured".

Two nulls are executed against it.

NULL A -- it is a property of the softmax distribution, not of the engine's quantizer.
  The float32 reference model's OWN attention probabilities are pushed through the same
  Q15 floor geometry (x = 32768*p, boundaries at integers of x) at the same layers, the
  same query position and the same heads T-1796's Arm B captured.  If the float model
  reproduces the 6.4x, nothing about the engine's integer softmax is being measured.

NULL B -- the excess is entirely the near-zero mass, where a code flip is inconsequential.
  Under floor geometry a value with x < 0.01 is "within 1% of a boundary" by construction:
  its distance to the boundary at 0 IS its own magnitude.  The same statistic is therefore
  recomputed split by code magnitude (x < 1, i.e. probabilities that quantize to 0 or 1,
  versus x >= 1).  If the excess lives entirely below x = 1, the "conversion curve" is
  counting flips of attention weights of size <= 1/32768 of a row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def floor_distance(x: np.ndarray) -> np.ndarray:
    """Distance to the nearest truncation boundary, in [0, 0.5], T-1796's convention:
    boundaries at integers of the pre-round quotient."""
    frac = x - np.floor(x)
    return np.minimum(frac, 1.0 - frac)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    a = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    model.to(dev)
    print("model.dtype:", model.dtype, "device:", dev)

    allx = []
    for u in a.prompts:
        text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n")
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        rows = []
        for L in CHECKPOINT_LAYERS:
            att = out.attentions[L][0, :, -1, :].double().cpu().numpy()  # heads x keys
            rows.append(att.reshape(-1))
        x = np.concatenate(rows) * 32768.0
        print(f"prompt tokens={ids.shape[1]}  captured probability values={x.size}")
        allx.append(x)

    res = {"per_prompt": [], "pooled": {}}
    for eps in (0.01, 0.05, 0.10):
        pass
    print()
    print("=== NULL A: float32 model's own attention probabilities under the SAME Q15 "
          "floor geometry ===")
    print(f"{'prompt':>7} {'n':>7} {'mean dist':>10} {'%<1%':>8} {'%<5%':>8} {'%<10%':>8}")
    for i, x in enumerate(allx, 1):
        d = floor_distance(x)
        row = dict(prompt=i, n=int(d.size), mean=float(d.mean()),
                   w1=float(100 * (d < 0.01).mean()), w5=float(100 * (d < 0.05).mean()),
                   w10=float(100 * (d < 0.10).mean()))
        res["per_prompt"].append(row)
        print(f"{i:>7} {row['n']:>7} {row['mean']:>10.4f} {row['w1']:>7.2f}% "
              f"{row['w5']:>7.2f}% {row['w10']:>7.2f}%")
    d_all = floor_distance(np.concatenate(allx))
    res["pooled"] = dict(n=int(d_all.size), mean=float(d_all.mean()),
                         w1=float(100 * (d_all < 0.01).mean()),
                         w5=float(100 * (d_all < 0.05).mean()),
                         w10=float(100 * (d_all < 0.10).mean()))
    print(f"{'pooled':>7} {res['pooled']['n']:>7} {res['pooled']['mean']:>10.4f} "
          f"{res['pooled']['w1']:>7.2f}% {res['pooled']['w5']:>7.2f}% "
          f"{res['pooled']['w10']:>7.2f}%")
    print("T-1796 engine measurement: mean 0.2193, %<1% 12.88, %<5% 20.11, %<10% 29.77")
    print("uniform null:              mean 0.2500, %<1%  2.00, %<5% 10.00, %<10% 20.00")

    print()
    print("=== NULL B: split by code magnitude (x = 32768 * p) ===")
    X = np.concatenate(allx)
    D = floor_distance(X)
    for lab, mask in (("x < 1 (code 0 or 1)", X < 1.0), ("x >= 1", X >= 1.0),
                      ("x >= 32 (p >= 1e-3)", X >= 32.0)):
        n = int(mask.sum())
        if n == 0:
            continue
        print(f"{lab:>22}: n={n:>6} ({100*n/X.size:5.1f}% of all)  mean={D[mask].mean():.4f}  "
              f"%<1%={100*(D[mask] < 0.01).mean():6.2f}  %<5%={100*(D[mask] < 0.05).mean():6.2f}  "
              f"%<10%={100*(D[mask] < 0.10).mean():6.2f}")
        res.setdefault("by_magnitude", {})[lab] = dict(
            n=n, mean=float(D[mask].mean()), w1=float(100 * (D[mask] < 0.01).mean()),
            w5=float(100 * (D[mask] < 0.05).mean()), w10=float(100 * (D[mask] < 0.10).mean()))
    near = D < 0.01
    res["share_of_near_boundary_below_x1"] = float((X[near] < 1.0).mean())
    print(f"\nshare of ALL within-1%-of-boundary values that have x < 1: "
          f"{100*res['share_of_near_boundary_below_x1']:.1f}%")
    res["share_all_below_x1"] = float((X < 1.0).mean())
    print(f"share of all values with x < 1: {100*res['share_all_below_x1']:.1f}%")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
