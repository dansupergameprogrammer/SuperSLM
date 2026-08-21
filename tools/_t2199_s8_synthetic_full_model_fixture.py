"""T-2199 Phase D review S8 closure (conductor's commission, 2026-08-20): a complete,
runnable synthetic .sslm model artifact -- every named tensor `BuildEngineCache`
(src/sslm_abi.cpp) requires, not the single-placeholder-weight shape
`sslm_pinned_calibration_fixture.py` provides (which the suite owner's own S8 evaluation
found DOES pass the converter/verifier but is REJECTED by the runtime engine construction,
Claude/Curie/t2199-phaseD-red-2026-08-20.md, item 3).

Reuses `tools/reference_pipeline/pipeline.py`'s own `fixture_model`/`ModelConfig` -- the "Sec11
fixture model," already the load-bearing small transformer `reference_pipeline`'s entire test
suite exercises the PYTHON reference forward pass against (real per-channel calibrated weight
scales via `_derive_scales`, real RoPE tables, real KV-landing scales/reciprocals, real
composition constants -- not a hand-typed placeholder). Its own output shape (a `QuantizedModel`
with `config`/`weights`/`weight_scales`/`rope_tables`/`biases`/`composition_constants`/
`kv_landing_scales`/`kv_landing_reciprocals`) is the SAME interface `convert_model.build_sections`
already consumes for the real converter path -- no new glue needed, no reimplementation of the
production writer.

Deterministic, hermetic, regenerated fresh every run (S-HARDEN-5's own discipline, matching
`sslm_pinned_calibration_fixture.py`'s own precedent) -- never committed as a binary blob.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference_pipeline"))

import convert_model as C  # noqa: E402
import pipeline as P  # noqa: E402
import sslm_format as F  # noqa: E402


# The "Sec11 fixture model" shape (tools/reference_pipeline/tests/test_pipeline.py's own
# fixture_config_kwargs) -- a known-good, extensively-exercised small transformer, not a new
# guess at dimensions. Kept identical to that precedent rather than shrunk further: every field
# here is already proven to clear every reference-pipeline domain check (RoPE pairwise rotation,
# GQA group-size divisibility, KV-landing derivation) at this exact size.
def build_config():
    # num_hidden_layers widened 2 -> 8 from the Sec11 fixture's own canonical shape
    # (test_pipeline.py's fixture_config_kwargs): tools/t2139_dim9_current_token_pin.cpp (the
    # C1-discriminating pin this fixture exists to run for real) hardcodes layer_budget=8 to
    # exercise a genuine partial-layer-budget resumption -- num_hidden_layers must be >= that.
    # Every OTHER dimension is kept at the proven Sec11 values; only the layer count changes,
    # and MarshalLayer's own per-layer construction is uniform across layers (no new domain
    # check is exercised by adding more of the identical, already-verified per-layer shape).
    return P.ModelConfig(
        hidden_size=32, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def build_artifact_bytes():
    """Returns (data, fingerprint) -- a complete, real .sslm model artifact byte string."""
    cfg = build_config()
    model = P.fixture_model(cfg)
    sections, fold_approximation_error = C.build_sections(model)
    data, fingerprint = F.build_artifact(sections)
    return data, fingerprint, fold_approximation_error


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "t2199_s8_fixture.sslm"
    data, fingerprint, fold_err = build_artifact_bytes()
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"wrote {out_path}: {len(data)} bytes, fingerprint={fingerprint}, "
          f"fold_approximation_error={fold_err}")
