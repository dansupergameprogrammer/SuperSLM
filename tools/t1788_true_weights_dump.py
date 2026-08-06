#!/usr/bin/env python3
"""T-1788: dump the checkpoint's own TRUE float32 Q/K projection weights and biases for the 6
checkpoint layers -- the INDEPENDENT source this ticket's "weight-exact" candidate substitutes
for the engine's int8 Q/K weight. Loaded exactly the way `t1788_float_prescore_reference.py`
(itself an unmodified copy of T-1787's/T-1786's/T-1778's own float reference) already loads the
checkpoint: `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32, local_files_only=True)`
-- the SAME model object, so these weights are provably the same tensors the float reference's own
oracle computation used internally, with NO import of `superslm`/`intmath`/`dynamic_engine` and no
input taken from the engine's own int8 construction (StandardsDocument.md 5.4's independence
requirement for a reference).

Dumped once (weights don't depend on prompt or position) as raw float64 binary, per checkpoint
layer: q_proj.weight [hidden_size, hidden_size], q_proj.bias [hidden_size] (zeros if the layer
carries none), k_proj.weight [kv_hidden_size, hidden_size], k_proj.bias [kv_hidden_size].
nn.Linear's own layout is [out_features, in_features] -- W[out,in] -- matching
GemmInt8AccumulateRow's own indexing (matmul.cpp:145, `weights + j*in_channels`), so no transpose
is needed to line this dump up with the engine's own int8 weight dump (`--weights-dump`,
t1788_weight_activation_probe.cpp), which uses the identical [out,in] layout.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

DEFAULT_MODEL = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(DEFAULT_MODEL), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    hidden_size = model.config.hidden_size
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim if getattr(model.config, "head_dim", None) else hidden_size // model.config.num_attention_heads
    kv_hidden_size = num_kv_heads * head_dim

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/t1788/true_weights.bin")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        for layer_idx in CHECKPOINT_LAYERS:
            attn = model.model.layers[layer_idx].self_attn
            qw = attn.q_proj.weight.detach().double().numpy()  # [hidden_size, hidden_size]
            qb = (
                attn.q_proj.bias.detach().double().numpy()
                if attn.q_proj.bias is not None
                else np.zeros(hidden_size, dtype=np.float64)
            )
            kw = attn.k_proj.weight.detach().double().numpy()  # [kv_hidden_size, hidden_size]
            kb = (
                attn.k_proj.bias.detach().double().numpy()
                if attn.k_proj.bias is not None
                else np.zeros(kv_hidden_size, dtype=np.float64)
            )
            assert qw.shape == (hidden_size, hidden_size), qw.shape
            assert qb.shape == (hidden_size,), qb.shape
            assert kw.shape == (kv_hidden_size, hidden_size), kw.shape
            assert kb.shape == (kv_hidden_size,), kb.shape
            f.write(qw.tobytes())
            f.write(qb.tobytes())
            f.write(kw.tobytes())
            f.write(kb.tobytes())
            has_qb = attn.q_proj.bias is not None
            has_kb = attn.k_proj.bias is not None
            print(
                f"layer={layer_idx}: q_proj.weight {qw.shape} q_proj.bias present={has_qb} "
                f"k_proj.weight {kw.shape} k_proj.bias present={has_kb}"
            )

    print(f"written: {out_path} ({len(CHECKPOINT_LAYERS)} layers x "
          f"(q_weight[{hidden_size}x{hidden_size}] + q_bias[{hidden_size}] + "
          f"k_weight[{kv_hidden_size}x{hidden_size}] + k_bias[{kv_hidden_size}]), float64)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
