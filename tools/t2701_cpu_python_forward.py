"""T-2701 whole-forward C++/Python fused-QK parity cell.

Build ``t2701_cpu_forward_probe`` first, then run this file with its path.
The fixture deliberately carries QK norms and QKC1, so its full integer
forward reaches the raw-K -> wide-RMSNorm -> wide-RoPE -> QKC1 path.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import convert_model as converter
import sslm_format as artifact_format
from reference_pipeline import pipeline


def fixture() -> pipeline.QuantizedModel:
    cfg = pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16)
    base = pipeline.fixture_model(cfg)
    weights = dict(base.weights)
    scales = dict(base.weight_scales)
    constants = dict(base.composition_constants)
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        weights[f"{prefix}.q_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
        weights[f"{prefix}.k_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
        scales[f"{prefix}.q_norm.gain"] = [1.0] * cfg.head_dim
        scales[f"{prefix}.k_norm.gain"] = [1.0] * cfg.head_dim
        constants[f"{prefix}.q_norm"] = (1 << 30, -30)
        constants[f"{prefix}.k_norm"] = (1 << 30, -30)
    seeded = replace(base, weights=weights, weight_scales=scales, composition_constants=constants)
    peaks = {f"layer{layer}": np.asarray(
        [[0.25 + 0.125 * (head + channel) for channel in range(cfg.head_dim)]
         for head in range(cfg.num_key_value_heads)], dtype=np.float64)
        for layer in range(cfg.num_hidden_layers)}
    return pipeline.with_provisional_qk_channel_table(seeded, peaks)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <t2701_cpu_forward_probe.exe>")
    probe = Path(sys.argv[1])
    if not probe.is_file():
        raise SystemExit(f"missing compiled probe: {probe}")
    model = fixture()
    artifact = Path("out/t2701/t2701_fused_qk_fixture.sslm")
    sections, _ = converter.build_sections(model)
    artifact_format.write_artifact(str(artifact), sections,
                                   flags=converter.artifact_flags_for_model(model))
    tokens = [1, 2, 3, 4, 5]
    py_digest = hashlib.sha256(
        np.asarray(pipeline.forward(model, tokens), dtype="<i4").tobytes()).hexdigest()
    run = subprocess.run([str(probe), str(artifact), ",".join(map(str, tokens))],
                         check=True, text=True, capture_output=True)
    cpp_digest = run.stdout.strip().rsplit("=", 1)[-1]
    print(f"fixture={artifact} tokens={tokens}")
    print(f"python_logits_sha256={py_digest}")
    print(f"cpp_logits_sha256={cpp_digest}")
    if cpp_digest != py_digest:
        raise SystemExit("C++/Python whole-forward mismatch")
    print("C++/Python whole-forward: IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
