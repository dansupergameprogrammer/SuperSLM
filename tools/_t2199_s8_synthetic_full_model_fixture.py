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
from dataclasses import replace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference_pipeline"))

import convert_model as C  # noqa: E402
import pipeline as P  # noqa: E402
import sslm_convert_schema as SC  # noqa: E402
import sslm_format as F  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402


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
    # The vocabulary and context are also widened for the product cells in the T-2138 ABI
    # suite: dim7 deliberately uses token 97 and a 20-token prompt. The old 32/16 fixture made
    # those cells exercise fixture rejection rather than the contract they grade.
    return P.ModelConfig(
        hidden_size=32, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=128,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=64,
    )


def build_artifact_bytes():
    """Returns (data, fingerprint) -- a complete, real .sslm model artifact byte string."""
    cfg = build_config()
    model = P.fixture_model(cfg)
    # This compact Phase-D fixture has head_dim=8.  Fused-QK's QKC1 contract is
    # deliberately head_dim=128-only, so keep this unrelated decoder fixture on
    # the ordinary (no-QK) path rather than serializing an invalid QKC1 table.
    model = replace(
        model,
        weights={k: v for k, v in model.weights.items()
                 if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
        weight_scales={k: v for k, v in model.weight_scales.items()
                       if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
    )
    # S8 is the hermetic model used by the Phase D damped-greedy gate, so carry the same DGC1
    # section and feature flag that production conversion emits with --enable-damped-greedy.
    sections, fold_approximation_error = C.build_sections(model, enable_damped_greedy=True)
    # One real compiled schema keeps the Phase D gate's mask-first composition live without a
    # checkpoint dependency. Token 0 spells the complete accepted document; the remaining
    # vocabulary entries are deliberately unrelated pieces.
    vocab = [b'{"ok":true}'] + [f"<unused-{i}>".encode("ascii") for i in range(1, cfg.vocab_size)]
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
    }
    # T-2917 (folding TE-370 C1, D-SLM7600): this synthetic vocabulary carries no tokenizer
    # special ids at all (every entry is either the one meaningful token or an inert filler
    # piece), so there is nothing to exclude -- special_ids=frozenset() is the correct,
    # deliberate "no exclusion" call, not an omission.
    masks = SC.compile_schema_to_mask_pages(schema, vocab, special_ids=frozenset())
    sections.append(F.Section(
        F.SectionType.SCHEMA_MASKS,
        _serialize_scm1([("g5_minimal_one_field", masks)], cfg.vocab_size)))
    # Keep this synthetic writer on the product writer's flag rule: QKC1 is
    # present exactly for a QK-norm-bearing model, and the container bit says so.
    data, fingerprint = F.build_artifact(
        sections, flags=C.artifact_flags_for_model(model) | F.DAMPED_GREEDY_CONSTANTS_FLAG)
    return data, fingerprint, fold_approximation_error


def build_plain_artifact_bytes():
    """T-2234 (SuperSLM 1.2.1): the SAME fixture model with NO damped-greedy opt-in -- the
    default conversion path (no DGC1 section, no feature bit), i.e. exactly the artifact
    shape a pre-1.2 caller holds. The workspace-layout cells in
    tests/t2138-abi-red-suite/dim7_contract_red.cpp need this twin so both variants differ
    in nothing but `damped_greedy_available`."""
    cfg = build_config()
    model = P.fixture_model(cfg)
    model = replace(
        model,
        weights={k: v for k, v in model.weights.items()
                 if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
        weight_scales={k: v for k, v in model.weight_scales.items()
                       if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
    )
    sections, _ = C.build_sections(model)  # default: no DGC1 section, no feature bit
    return F.build_artifact(sections)


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "t2199_s8_fixture.sslm"
    data, fingerprint, fold_err = build_artifact_bytes()
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"wrote {out_path}: {len(data)} bytes, fingerprint={fingerprint}, "
          f"fold_approximation_error={fold_err}")
    if len(sys.argv) > 2:
        plain_out_path = sys.argv[2]
        plain_data, plain_fingerprint = build_plain_artifact_bytes()
        with open(plain_out_path, "wb") as f:
            f.write(plain_data)
        print(f"wrote {plain_out_path}: {len(plain_data)} bytes, "
              f"fingerprint={plain_fingerprint} (no DGC1 section/flag)")
