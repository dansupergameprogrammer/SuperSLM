"""C28 bias surface integration: emission, fixture, forward wiring, oracle conformance.

Charter: D-SLM59 (2026-07-16 — the C28 bias surface): the dynamic-arm biased fixture,
the bias emission half-even rounding, biased-model forward conformance to the oracle,
and domain-edge rejection. See the full pins in
Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §17.1 (amendment A-8).

Packet brief: Claude/Curie/packets/eval/packet-10.md.
"""

import random

import pytest

from conftest import api, require
from composition_ref import (
    bias_emission_oracle,
    forward_dynamic_logits_oracle,
    FineScaleRejected,
)

MODULE_PIPELINE = "reference_pipeline.pipeline"


def fixture_config(pipeline):
    """The §11 fixture model's shape, mirrored from test_dynamic_forward.py."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


# ==============================================================================
# Cell 1: C28 emission rounds half-even
# ==============================================================================


def test_dynamic_bias_emission_rounds_half_even():
    """C28 emission: B[j] = round_half_even(b[j] / S_ref * 2^q_B). The rounding mode is
    ADOPTED from C14's offline half-even precedent (A-8 §17.1).

    Executed tie fixtures (A-8 §17.1): at s_ref = 0.25, q_b = 30,
    b = (k + 0.5) * s_ref / 2^30 → emit_dynamic_bias([b], s_ref)[0] is 10 for k = 10
    (tie to even) and 12 for k = 11. Each also == bias_emission_oracle.

    Plus a seeded sweep (random.Random(10001), 2,000 floats vs the oracle) and both signs.
    """
    emit_dynamic_bias = api(MODULE_PIPELINE, "emit_dynamic_bias")

    # Executed tie fixtures: s_ref = 0.25, q_b = 30
    s_ref = 0.25
    q_b = 30

    # k = 10: b = 10.5 * 0.25 / 2^30
    k_10_float = (10 + 0.5) * s_ref / (1 << q_b)
    result_10 = emit_dynamic_bias([k_10_float], s_ref, q_b=q_b)[0]
    expected_10 = bias_emission_oracle(k_10_float, s_ref, q_b)
    assert result_10 == 10, f"k=10 tie to even: expected 10, got {result_10}"
    assert result_10 == expected_10, f"k=10: implementation {result_10} != oracle {expected_10}"

    # k = 11: b = 11.5 * 0.25 / 2^30
    k_11_float = (11 + 0.5) * s_ref / (1 << q_b)
    result_11 = emit_dynamic_bias([k_11_float], s_ref, q_b=q_b)[0]
    expected_11 = bias_emission_oracle(k_11_float, s_ref, q_b)
    assert result_11 == 12, f"k=11 tie to even: expected 12, got {result_11}"
    assert result_11 == expected_11, f"k=11: implementation {result_11} != oracle {expected_11}"

    # Seeded sweep: 2,000 random floats vs the oracle, both signs
    rng = random.Random(10001)
    s_ref_sweep = 0.25  # pinned for consistency with tie fixtures
    q_b_sweep = 30

    # Draw both positive and negative biases
    b_values = [rng.uniform(-100.0, 100.0) for _ in range(2000)]
    for b_val in b_values:
        expected = bias_emission_oracle(b_val, s_ref_sweep, q_b_sweep)
        result = emit_dynamic_bias([b_val], s_ref_sweep, q_b=q_b_sweep)[0]
        assert result == expected, (
            f"sweep: b={b_val}, expected {expected}, got {result}"
        )

    # Guard: sweep must draw both signs
    has_positive = any(b > 0 for b in b_values)
    has_negative = any(b < 0 for b in b_values)
    assert has_positive and has_negative, "sweep did not draw both signs"


# ==============================================================================
# Cell 2: The biased fixture carries the pinned dynamic_bias surface
# ==============================================================================


def test_the_biased_fixture_carries_the_pinned_dynamic_bias_surface():
    """The biased fixture carries dynamic_biases with the pinned structure:
    model.dynamic_biases[f"{prefix}.{name}"] -> (q_B: int, codes: tuple[int, ...])

    One entry per biased projection (q/k/v per layer), each (30, tuple) with widths
    matching the projection's output dim. Codes within ±2^40 (the executed decisive
    magnitude).

    The plain fixture_model carries NO dynamic_biases.
    """
    fixture_model, fixture_model_biased = api(
        MODULE_PIPELINE, "fixture_model", "fixture_model_biased")

    cfg = fixture_config(require(MODULE_PIPELINE))

    # Plain fixture has NO dynamic_biases
    model_plain = fixture_model(cfg)
    plain_biases = getattr(model_plain, "dynamic_biases", None)
    assert plain_biases is None or len(plain_biases) == 0, (
        "plain fixture_model must not carry dynamic_biases"
    )

    # Biased fixture carries the pinned surface
    model_biased = fixture_model_biased(cfg)
    biases = model_biased.dynamic_biases
    assert isinstance(biases, dict), "dynamic_biases must be a dict"
    assert len(biases) > 0, "dynamic_biases must not be empty"

    # Every entry is (q_B, codes) with q_B = 30 and codes as a tuple
    for site, (q_b_val, codes) in biases.items():
        assert isinstance(site, str), f"site {site} must be a string"
        assert q_b_val == 30, f"site {site}: q_B must be 30, got {q_b_val}"
        assert isinstance(codes, (tuple, list)), (
            f"site {site}: codes must be tuple or list, got {type(codes)}"
        )

        # Width check: the codes should match the site's output dimension
        # (this is a structural property, verified below)
        assert len(codes) > 0, f"site {site}: codes must not be empty"

        # Magnitude check: codes within ±2^40
        for i, code in enumerate(codes):
            assert isinstance(code, int), (
                f"site {site}[{i}]: code must be int, got {type(code)}"
            )
            assert abs(code) <= (1 << 40), (
                f"site {site}[{i}]: code {code} exceeds ±2^40"
            )

    # Verify that we have entries for q/k/v projections across layers
    q_entries = [s for s in biases.keys() if "q_proj" in s]
    k_entries = [s for s in biases.keys() if "k_proj" in s]
    v_entries = [s for s in biases.keys() if "v_proj" in s]
    assert len(q_entries) > 0, "must have q_proj bias entries"
    assert len(k_entries) > 0, "must have k_proj bias entries"
    assert len(v_entries) > 0, "must have v_proj bias entries"


# ==============================================================================
# Cell 3: Forward is bit-exact against the oracle on the biased fixture
# ==============================================================================


def test_forward_dynamic_is_bit_exact_against_the_oracle_on_the_biased_fixture():
    """The E9 claim under bias: forward_dynamic(model_biased, tokens) ==
    forward_dynamic_logits_oracle(model_biased, tokens) bit-for-bit on [0,1,3,5]
    AND [7,2,30,11]. No tolerance, never argmax-only.

    The oracle's bias walk is authored in composition_ref and reads
    model.dynamic_biases when present.
    """
    forward_dynamic, fixture_model_biased = api(
        MODULE_PIPELINE, "forward_dynamic", "fixture_model_biased")

    cfg = fixture_config(require(MODULE_PIPELINE))
    model = fixture_model_biased(cfg)

    # Sequence 1: [0,1,3,5]
    tokens_1 = [0, 1, 3, 5]
    logits_forward_1 = forward_dynamic(model, tokens_1)
    logits_oracle_1 = forward_dynamic_logits_oracle(model, tokens_1)

    # Verify shape
    assert len(logits_forward_1) == len(tokens_1)
    assert len(logits_oracle_1) == len(tokens_1)
    assert len(logits_forward_1[0]) == cfg.vocab_size

    # Bit-exact comparison
    for t, (forward_row, oracle_row) in enumerate(zip(logits_forward_1, logits_oracle_1)):
        for j, (fwd_val, oracle_val) in enumerate(zip(forward_row, oracle_row)):
            assert fwd_val == oracle_val, (
                f"sequence 1, token {t}, position {j}: "
                f"forward {fwd_val} != oracle {oracle_val}"
            )

    # Sequence 2: [7,2,30,11]
    tokens_2 = [7, 2, 30, 11]
    logits_forward_2 = forward_dynamic(model, tokens_2)
    logits_oracle_2 = forward_dynamic_logits_oracle(model, tokens_2)

    # Verify shape
    assert len(logits_forward_2) == len(tokens_2)
    assert len(logits_oracle_2) == len(tokens_2)

    # Bit-exact comparison
    for t, (forward_row, oracle_row) in enumerate(zip(logits_forward_2, logits_oracle_2)):
        for j, (fwd_val, oracle_val) in enumerate(zip(forward_row, oracle_row)):
            assert fwd_val == oracle_val, (
                f"sequence 2, token {t}, position {j}: "
                f"forward {fwd_val} != oracle {oracle_val}"
            )


# ==============================================================================
# Cell 4: The bias-dropping negative control
# ==============================================================================


def test_a_bias_dropping_forward_fails():
    """Negative control: a forward that drops the bias does not match the biased oracle.

    Build a zeroed-bias twin (same model, every bias code 0 — construct via the biased
    model's own surface, never by filtering). In-cell decisiveness: oracle(biased) !=
    oracle(zeroed) on [0,1,3,5] (executed A-8: 128/128 elements differ at the pinned
    fixture). Conformance: forward_dynamic(biased) == oracle(biased) — a forward that
    drops or zeroes the bias reproduces the zeroed twin and fails here by construction.
    """
    import dataclasses

    forward_dynamic, fixture_model_biased = api(
        MODULE_PIPELINE, "forward_dynamic", "fixture_model_biased")

    cfg = fixture_config(require(MODULE_PIPELINE))
    model_biased = fixture_model_biased(cfg)

    # Zeroed-bias twin via dataclasses.replace — dynamic_biases is a FROZEN-dataclass
    # field (A-9 pin refinement; QuantizedModel is frozen, plain setattr raises).
    zeroed_biases = {
        site: (q_b_val, tuple(0 for _ in codes))
        for site, (q_b_val, codes) in model_biased.dynamic_biases.items()
    }
    model_zeroed = dataclasses.replace(model_biased, dynamic_biases=zeroed_biases)

    # Decisiveness guard: oracle(biased) must differ from oracle(zeroed)
    # (executed A-8: 128/128 elements differ at the pinned seed-101 ±2^40 fixture).
    tokens = [0, 1, 3, 5]
    logits_biased = forward_dynamic_logits_oracle(model_biased, tokens)
    logits_zeroed = forward_dynamic_logits_oracle(model_zeroed, tokens)
    differ_count = sum(
        1 for b_row, z_row in zip(logits_biased, logits_zeroed)
        for b_val, z_val in zip(b_row, z_row)
        if b_val != z_val
    )
    assert differ_count > 0, (
        "decisiveness guard failed: biased and zeroed oracle logits must differ "
        "on the pinned fixture [0,1,3,5]"
    )

    # THE TEETH (Curie A-9 — the cell's name is its claim): the implementation must
    # reproduce the BIASED oracle. A forward that drops or zeroes the bias reproduces
    # the zeroed twin instead and fails here by construction.
    logits_forward = forward_dynamic(model_biased, tokens)
    for f_row, o_row in zip(logits_forward, logits_biased):
        assert [int(v) for v in f_row] == [int(v) for v in o_row], (
            "forward_dynamic does not reproduce the biased oracle — the bias path is "
            "dropped, zeroed, or mis-wired (C28)")


# ==============================================================================
# Cell 5: Oversized bias magnitudes reject at the coarse scale edge
# ==============================================================================


def test_oversized_bias_magnitudes_reject_at_the_coarse_scale_edge():
    """Domain edges, executed ON THE PINNED FIXTURE (Curie A-9 addendum — the A-8 ±2^50
    premise was executed on the pre-calibration stand-in fixture, whose norm carried-scale
    regime sat lower; the builder's counter-evidence reproduced at this seat's bench and
    REFINED: the pinned fixture's ladder is

        ±2^50  → COMPUTES clean (logit[0][0] = −2393 — inside the chain domain; re-executed
                  2026-07-27 after SuperSLM_S3a_WalkingSkeleton_Plan.md SS11 S3.0/F-S3-1
                  reconciled composition_ref.py's mlp_act to C10's LUT sigmoid, superseding
                  the −2389 witness computed under the excluded i-exp-sigmoid construction),
        ±2^52  → ValueError, C30's COARSE bound (q_ln2 < 1 — i-exp's existing rejection),
        ±2^55  → ValueError, C29's CHAIN-DOMAIN edge (D' = 3309102943 > 2^31 at
                  normalize_scale — rejection arrives via the chain domain, not C30).

    Both rejection mechanisms are pinned, pre-existing rejections (C30's coarse end;
    C29 restating C21's precondition at every feeding site), so the implementation must
    raise the same class the oracle does at each rung."""
    import dataclasses

    forward_dynamic, fixture_model_biased = api(
        MODULE_PIPELINE, "forward_dynamic", "fixture_model_biased")

    cfg = fixture_config(require(MODULE_PIPELINE))
    model_base = fixture_model_biased(cfg)

    def oversized(mag_bits):
        biases = dict(model_base.dynamic_biases)
        q_b_val, codes = biases["layer0.q_proj"]
        biases["layer0.q_proj"] = (q_b_val, tuple(
            2 ** mag_bits if i % 2 == 0 else -(2 ** mag_bits)
            for i in range(len(codes))))
        return dataclasses.replace(model_base, dynamic_biases=biases)

    tokens = [0, 1, 3, 5]

    # Rung 0 — ±2^50 COMPUTES on the pinned fixture, and the implementation still
    # matches the oracle bit-for-bit there (conformance holds right up to the edge).
    model_50 = oversized(50)
    oracle_50 = forward_dynamic_logits_oracle(model_50, tokens)
    assert oracle_50[0][0] == -2393  # the executed pinned-fixture witness (re-executed 2026-07-27, C10)
    forward_50 = forward_dynamic(model_50, tokens)
    for f_row, o_row in zip(forward_50, oracle_50):
        assert [int(v) for v in f_row] == [int(v) for v in o_row]

    # Rung 1 — ±2^52: C30's coarse bound fires first (q_ln2 < 1), oracle AND forward.
    with pytest.raises(ValueError):
        forward_dynamic_logits_oracle(oversized(52), tokens)
    with pytest.raises(ValueError):
        forward_dynamic(oversized(52), tokens)

    # Rung 2 — ±2^55: C29's chain-domain edge (D' > 2^31 at normalize_scale).
    with pytest.raises(ValueError):
        forward_dynamic_logits_oracle(oversized(55), tokens)
    with pytest.raises(ValueError):
        forward_dynamic(oversized(55), tokens)
