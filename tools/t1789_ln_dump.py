#!/usr/bin/env python3
"""T-1789: dump the checkpoint's own float32 input_layernorm.weight (RMSNorm gamma) for the 6
checkpoint layers, to out/t1789/ln_gamma.bin (float64 raw, 6 x 1536, checkpoint-layer order).

Independent source discipline (StandardsDocument.md 5.4, same as t1788_true_weights_dump.py):
reads the HF checkpoint's own safetensors directly; zero import of superslm/intmath/
dynamic_engine; no derived input from the engine's own int8 construction. Reads tensors without
instantiating the model (safetensors direct open)."""

import json
import struct
from pathlib import Path

from safetensors import safe_open

SNAPSHOT = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)
OUT = Path(r"D:\SuperSLM\.worktrees\t1789-attention-solve\out\t1789\ln_gamma.bin")


def main():
    files = sorted(SNAPSHOT.glob("*.safetensors"))
    assert files, f"no safetensors under {SNAPSHOT}"
    tensors = {}
    for fp in files:
        with safe_open(str(fp), framework="pt") as f:
            for name in f.keys():
                for layer in CHECKPOINT_LAYERS:
                    want = f"model.layers.{layer}.input_layernorm.weight"
                    if name == want:
                        tensors[layer] = f.get_tensor(name).double().tolist()
    missing = [l for l in CHECKPOINT_LAYERS if l not in tensors]
    assert not missing, f"missing input_layernorm.weight for layers {missing}"
    with open(OUT, "wb") as f:
        for layer in CHECKPOINT_LAYERS:
            vals = tensors[layer]
            assert len(vals) == 1536, f"layer {layer}: unexpected gamma length {len(vals)}"
            f.write(struct.pack(f"<{len(vals)}d", *vals))
    print(f"ln_gamma_written: {len(CHECKPOINT_LAYERS)} layers x 1536 -> {OUT}")


if __name__ == "__main__":
    main()
