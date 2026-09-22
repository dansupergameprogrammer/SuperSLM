"""Generate the compact vocab-384 model used by the CPU terminal-reset cells."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "reference_pipeline"))

import convert_model  # noqa: E402
import pipeline  # noqa: E402
import sslm_format  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_t2922_cpu_reset_base.py <out.sslm>")
    config = pipeline.ModelConfig(
        hidden_size=32,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=64,
        vocab_size=384,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        context_cap=64,
    )
    model = pipeline.fixture_model(config)
    sections, fold_error = convert_model.build_sections(model)
    flags = convert_model.artifact_flags_for_model(
        model, option_g=False, enable_damped_greedy=False)
    data, fingerprint = sslm_format.build_artifact(sections, flags=flags)
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"wrote {output} bytes={len(data)} fingerprint={fingerprint} fold_error={fold_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
