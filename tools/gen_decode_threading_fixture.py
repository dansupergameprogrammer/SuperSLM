"""Decode-threading plan (rev 1.2) §3.9: the CI fixtures for the one-row (M = 1) matvec cells.

Builds runnable `.sslm` artifacts through the reference pipeline on `main`, by the path
`_t2199_s8_synthetic_full_model_fixture.py` already takes (`pipeline.fixture_model`'s pinned
weights and calibration, `convert_model.build_sections`, `sslm_format.build_artifact`). Only the
geometry and one calibration knob differ. Deterministic and hermetic; generated fresh by every CI
leg that runs a suite binary and never committed (S-HARDEN-5). `superslm_tests` reads them from
the directory named by SUPERSLM_DECODE_THREADING_FIXTURE_DIR, and a missing file fails the cell.

Variants (two layers, vocabulary 256, context cap 160 each):

  fdef.sslm      F-DEF: hidden 192, 4 query heads and 2 KV heads of 48, intermediate 512,
                 QK-norm stripped. Carries the damped-greedy constants and one schema (the S8
                 fixture's), for cell 8.1. Moves kv, kv_landing, rope_q and rope_k.
  fqk.sslm       F-QK: the same hidden and MLP widths with QK-norm kept. The loader admits a
                 QK-norm artifact only at head_dim 128 (src/model.cpp, ValidateFusedKHeadDim), so
                 F-QK has 4 query heads and 2 KV heads of 128 (query width 512). Moves
                 k_channel_landing.
  fnoclamp.sslm  F-DEF with the K and V calibration peaks inflated, so that K/V landing never
                 clamps: the vitality variant of cell 11.1, on which the F1 preamble must fail.

The knob. The calibration peak of every layer's K (and, for fnoclamp, V) projection is multiplied
by a per-variant factor before the scales are derived. A factor below 1 makes the static K landing
scale finer, so landed K codes are larger and K/V landing and K RoPE clamp more often; that is
the plan's "raise the generator's activation scale" (§8 preamble, R6). Nothing else changes: the
weights, the other peaks and every derivation are fixture_model's own.

Usage: python tools/gen_decode_threading_fixture.py OUT_DIR
"""

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference_pipeline"))

import convert_model as C  # noqa: E402
import pipeline as P  # noqa: E402
import sslm_convert_schema as SC  # noqa: E402
import sslm_format as F  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402

VOCAB = 256
CONTEXT_CAP = 160

# Measured on this generator's output (docs/decode-threading/fixture-premise.txt). The guard cells'
# F1 preamble needs a decode step whose second layer clamps kv, kv_landing, rope_q and rope_k, with
# rope_q and rope_k moving by different amounts: at 1.0 no step of the cells' prompt does, at 0.5
# none does either (rope_q and rope_k clamp about once a layer), at 0.3 three steps in 60 do, and at
# 0.2 twelve do, the first at step 2. rope_q is the scarce one; the Q peak does not move it.
K_PEAK_FACTOR_DEF = 0.2
KV_PEAK_FACTOR_NOCLAMP = 8.0


def config(head_dim):
    return P.ModelConfig(
        hidden_size=192, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=head_dim, intermediate_size=512, vocab_size=VOCAB, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=CONTEXT_CAP)


_CALIBRATION = {}


def fixture_model(cfg, peak_factors):
    """`pipeline.fixture_model`, with the named calibration peaks scaled before derivation."""
    P.attention_group_size(cfg)
    weights, weight_scales, floats = P._pinned_weights(cfg)
    float_weight = P._dict_float_source(floats)
    tokenize_prompt = P._fixture_tokenize_prompt(cfg)
    if cfg not in _CALIBRATION:  # the calibration pass is the slow step; variants share it
        _CALIBRATION[cfg] = P._calibrate(
            cfg, float_weight, P.calibration_records(),
            lambda record: tokenize_prompt(P.run_prompt_messages(record)),
            return_channel_peaks=True)
    maxima, channel_peaks = _CALIBRATION[cfg]
    maxima = dict(maxima)
    for layer in range(cfg.num_hidden_layers):
        for site, factor in peak_factors.items():
            key = f"layer{layer}.{site}"
            if key not in maxima:
                raise KeyError(f"calibration has no peak named {key}")
            maxima[key] = maxima[key] * factor
    scales, residual_scales, biases = P._derive_scales(cfg, maxima, weight_scales, {})
    constants, kv_scales, kv_recips = P._derive_composition_constants(cfg, weight_scales, scales)
    model = P.QuantizedModel(
        config=cfg, scales=scales, weights=weights, weight_scales=weight_scales,
        residual_scales=residual_scales, rope_tables=P._build_rope_tables(cfg), biases=biases,
        float_source=float_weight, tokenize_prompt=tokenize_prompt,
        calibration=P._calibration_record(P._FIXTURE_TOKENIZATION), gemm_weights={},
        composition_constants=constants, kv_landing_scales=kv_scales,
        kv_landing_reciprocals=kv_recips)
    return P.with_provisional_qk_channel_table(model, channel_peaks) if channel_peaks else model


def strip_qk_norm(model):
    def keep(k):
        return not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))
    return dataclasses.replace(
        model, weights={k: v for k, v in model.weights.items() if keep(k)},
        weight_scales={k: v for k, v in model.weight_scales.items() if keep(k)})


def artifact(model, with_schema):
    sections, _ = C.build_sections(model, enable_damped_greedy=with_schema)
    flags = C.artifact_flags_for_model(model)
    if with_schema:
        # The S8 fixture's schema (token 0 spells the one accepted document), for cell 8.1.
        vocab = [b'{"ok":true}'] + [f"<unused-{i}>".encode("ascii") for i in range(1, VOCAB)]
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        masks = SC.compile_schema_to_mask_pages(schema, vocab, special_ids=frozenset())
        sections.append(F.Section(F.SectionType.SCHEMA_MASKS,
                                  _serialize_scm1([("g5_minimal_one_field", masks)], VOCAB)))
        flags |= F.DAMPED_GREEDY_CONSTANTS_FLAG
    data, fingerprint = F.build_artifact(sections, flags=flags)
    return data, fingerprint


VARIANTS = (
    ("fdef.sslm", 48, True, {"k": K_PEAK_FACTOR_DEF}, True),
    ("fqk.sslm", 128, False, {}, False),
    ("fnoclamp.sslm", 48, True, {"k": KV_PEAK_FACTOR_NOCLAMP, "v": KV_PEAK_FACTOR_NOCLAMP}, False),
)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out_dir = argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name, head_dim, strip, factors, with_schema in VARIANTS:
        start = time.time()
        model = fixture_model(config(head_dim), factors)
        if strip:
            model = strip_qk_norm(model)
        data, fingerprint = artifact(model, with_schema)
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {path}: {len(data)} bytes, fingerprint={fingerprint}, "
              f"{time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
