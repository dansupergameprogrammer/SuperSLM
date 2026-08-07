#!/usr/bin/env python3
"""T-1795: sub-block substitution at the 6 checkpoint layers.

Reads a checkpoint layer's own real (engine-captured, self-checked bit-for-bit against
production by tools/t1795_residual_probe.cpp) dequantized intermediate quantities --
h_in, normed_attn, attn_branch, attn_stream, normed_mlp, mlp_branch, h_out -- and
substitutes ONE sub-block's own quantization/rounding step with its float-exact form,
computed by calling the checkpoint's REAL float32 module directly
(`layer.input_layernorm`, `layer.post_attention_layernorm`, `layer.mlp`) on the engine's
own real captured input at that stage. Every substitution below is POSITIONWISE -- it
needs only the single captured token-position vector, never another position's K/V --
which is what makes it computable without re-deriving the engine's own attention
machinery a second time. Attention is NOT positionwise (it reads every prior position's
K/V) and is therefore reported separately, not isolated by this method (named, per this
ticket's own instruction against guessing a share for an un-isolable stage).

Four sub-blocks, individual + one joint (positionwise-only) arm:

  - norm_attn:  RmsNormSite's own int8 quantization of the attn-norm output, isolated by
                comparing the engine's real `normed_attn` (dequantized) against
                `layer.input_layernorm(h_in)` computed in float32 from the SAME real h_in.
  - add_attn:   ResidualReconcileSite's own int8 requantization of h_in + attn_branch,
                isolated by comparing the engine's real `attn_stream` (dequantized)
                against the EXACT float sum h_in + attn_branch (both the engine's own
                real dequantized values -- no model call needed, a pure arithmetic check).
  - norm_mlp:   the mlp-norm's own quantization, isolated the same way as norm_attn, from
                the engine's real `attn_stream`.
  - mlp:        the MLP branch's own weight/activation quantization, isolated by
                comparing the engine's real `mlp_branch` (dequantized) against
                `layer.mlp(normed_mlp_exact)` -- note this uses the EXACT (not the
                engine's quantized) normed_mlp as input, so it measures the MLP branch's
                own quantization ceiling GIVEN an exact input, isolating it from
                norm_mlp's own separate contribution.
  - add_mlp:    the second residual add's own requantization, isolated the same way as
                add_attn, from the engine's real `attn_stream` and `mlp_branch`.
  - joint_positionwise: every POSITIONWISE stage made exact simultaneously (both norms,
                both adds, the MLP branch), attention held at the engine's real value --
                h_out_joint = h_in + attn_branch_real (exact add)
                              + layer.mlp(layer.post_attention_layernorm(h_in + attn_branch_real))
                compared against the engine's real h_out. The gap between this arm and
                baseline-vs-h_out is the ceiling every positionwise sub-block together can
                close; what remains is attention's own quantization plus any
                cross-sub-block interaction, neither cleanly separable from the other by
                this method (named, not guessed).

Attention's OWN branch is additionally compared -- as a DIFFERENT, clearly-labeled
quantity, not conflated with the isolated shares above -- against the float reference's
own recursively-computed attn_out at that layer (a different population: the float
model's own upstream-consistent input, not the engine's drifted one), per
StandardsDocument.md 5.4's requirement that both sides of a comparison be the same
quantity over the same population.

Model loaded exactly as tools/t1795_float_residual_reference.py (float32, eager,
local_files_only, TF32 disabled and verified by execution) -- the identical precision
discipline, re-verified here rather than inherited by assertion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_MODEL = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)


def _resolve_default_model(p: Path) -> Path:
    snaps = p / "snapshots"
    if snaps.is_dir():
        entries = [d for d in snaps.iterdir() if d.is_dir()]
        if len(entries) == 1:
            return entries[0]
    return p


def read_vec(f):
    parts = f.readline().split()
    n = int(parts[0])
    vals = [float(x) for x in parts[1 : 1 + n]]
    assert len(vals) == n
    return vals


def read_stages(path: Path):
    layers = {}
    with open(path) as f:
        count = int(f.readline().strip())
        for _ in range(count):
            header = f.readline().split()
            assert header[0] == "layer"
            layer = int(header[1])
            entry = {
                "h_in": read_vec(f),
                "normed_attn": read_vec(f),
                "attn_branch": read_vec(f),
                "attn_stream": read_vec(f),
                "normed_mlp": read_vec(f),
                "mlp_branch": read_vec(f),
                "h_out": read_vec(f),
            }
            layers[layer] = entry
    return layers


def l2(a, b) -> float:
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b))


def relnorm(a, b) -> float:
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--stages", required=True, nargs="+", help="one or more *_stages.txt dumps")
    parser.add_argument("--float-attn-out", required=True, nargs="+",
                         help="matching *_float_attnout.txt dumps (float model's own recursive attn_out per layer)")
    parser.add_argument("--out", required=True, help="path to write the per-prompt, per-layer result table (json)")
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
    import json

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    param_dtypes = {p.dtype for p in model.parameters()}
    if param_dtypes != {torch.float32}:
        raise SystemExit(f"FAILED: model parameters are not genuine float32: {param_dtypes}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        print(f"allow_tf32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    model.to(device)
    print(f"model.dtype: {model.dtype}, device: {device}")

    results = []
    for prompt_idx, (stages_path, float_attnout_path) in enumerate(
        zip(args.stages, args.float_attn_out), start=1
    ):
        layers = read_stages(Path(stages_path))
        float_attn_out = {}
        with open(float_attnout_path) as f:
            n = int(f.readline().strip())
            for _ in range(n):
                header = f.readline().split()
                layer_idx = int(header[1])
                float_attn_out[layer_idx] = read_vec(f)

        for layer_idx in CHECKPOINT_LAYERS:
            e = layers[layer_idx]
            layer_mod = model.model.layers[layer_idx]

            h_in_t = torch.tensor(e["h_in"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            attn_stream_t = torch.tensor(e["attn_stream"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                normed_attn_exact = layer_mod.input_layernorm(h_in_t)[0, 0].double().tolist()
                normed_mlp_exact = layer_mod.post_attention_layernorm(attn_stream_t)[0, 0].double().tolist()

                normed_mlp_exact_t = torch.tensor(normed_mlp_exact, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                mlp_from_exact_norm = layer_mod.mlp(normed_mlp_exact_t)[0, 0].double().tolist()

                # joint_positionwise: exact norms + exact MLP, attention held real, exact adds
                attn_branch_real_t = torch.tensor(e["attn_branch"], dtype=torch.float32, device=device)
                h_in_f = torch.tensor(e["h_in"], dtype=torch.float64)
                attn_branch_f = torch.tensor(e["attn_branch"], dtype=torch.float64)
                attn_stream_joint = (h_in_f + attn_branch_f)  # exact add, float64
                attn_stream_joint_t = attn_stream_joint.float().to(device).unsqueeze(0).unsqueeze(0)
                normed_mlp_joint = layer_mod.post_attention_layernorm(attn_stream_joint_t)
                mlp_joint = layer_mod.mlp(normed_mlp_joint)[0, 0].double()
                h_out_joint = (attn_stream_joint + mlp_joint.cpu()).tolist()

            # --- individual sub-block errors, all against the SAME engine-real comparand ---
            row = {
                "prompt": prompt_idx,
                "layer": layer_idx,
                # baseline layer-level growth (for context)
                "h_in_norm": l2(e["h_in"], [0.0] * len(e["h_in"])),
                "h_out_norm": l2(e["h_out"], [0.0] * len(e["h_out"])),
                "layer_delta_norm": l2(e["h_out"], e["h_in"]),
                # norm_attn: engine's real normed_attn vs exact, same h_in
                "norm_attn_abs": l2(e["normed_attn"], normed_attn_exact),
                "norm_attn_rel": relnorm(e["normed_attn"], normed_attn_exact),
                # add_attn: engine's real attn_stream vs exact sum of engine's own real parts
                "add_attn_abs": l2(e["attn_stream"], [a + b for a, b in zip(e["h_in"], e["attn_branch"])]),
                "add_attn_rel": relnorm(e["attn_stream"], [a + b for a, b in zip(e["h_in"], e["attn_branch"])]),
                # norm_mlp: engine's real normed_mlp vs exact, same attn_stream
                "norm_mlp_abs": l2(e["normed_mlp"], normed_mlp_exact),
                "norm_mlp_rel": relnorm(e["normed_mlp"], normed_mlp_exact),
                # mlp: engine's real mlp_branch vs exact MLP over the EXACT normed_mlp
                "mlp_abs": l2(e["mlp_branch"], mlp_from_exact_norm),
                "mlp_rel": relnorm(e["mlp_branch"], mlp_from_exact_norm),
                # add_mlp: engine's real h_out vs exact sum of engine's own real parts
                "add_mlp_abs": l2(e["h_out"], [a + b for a, b in zip(e["attn_stream"], e["mlp_branch"])]),
                "add_mlp_rel": relnorm(e["h_out"], [a + b for a, b in zip(e["attn_stream"], e["mlp_branch"])]),
                # joint_positionwise: engine's real h_out vs the fully-exact-except-attention chain
                "joint_positionwise_abs": l2(e["h_out"], h_out_joint),
                "joint_positionwise_rel": relnorm(e["h_out"], h_out_joint),
                # attention's OWN branch: engine's real branch vs the float model's own
                # recursively-computed attn_out at this layer -- a DIFFERENT population
                # (float's own upstream-consistent input), named as such, not an isolation.
                "attn_branch_vs_float_recursive_abs": l2(e["attn_branch"], float_attn_out[layer_idx]),
                "attn_branch_vs_float_recursive_rel": relnorm(e["attn_branch"], float_attn_out[layer_idx]),
                "attn_branch_norm_engine": l2(e["attn_branch"], [0.0] * len(e["attn_branch"])),
                "attn_branch_norm_float_recursive": l2(float_attn_out[layer_idx], [0.0] * len(float_attn_out[layer_idx])),
                "mlp_branch_norm_engine": l2(e["mlp_branch"], [0.0] * len(e["mlp_branch"])),
            }
            results.append(row)
            print(f"P{prompt_idx} layer={layer_idx}: "
                  f"norm_attn_abs={row['norm_attn_abs']:.4f} add_attn_abs={row['add_attn_abs']:.4f} "
                  f"norm_mlp_abs={row['norm_mlp_abs']:.4f} mlp_abs={row['mlp_abs']:.4f} "
                  f"add_mlp_abs={row['add_mlp_abs']:.4f} joint_pw_abs={row['joint_positionwise_abs']:.4f} "
                  f"layer_delta_norm={row['layer_delta_norm']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {args.out} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
