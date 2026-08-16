"""T-1945 -- mutation pin for the located T-1944 root cause: `calibrate_kv_landing_arm`'s
per-head branch (`pipeline.py`, arms C/D/E) must write `kv_reciprocals[...]` as the
canonical `S_kh / S_ref` RATIO mantissa/exponent/reciprocal triple the runtime's
`LandingRescale` actually consumes, never a reciprocal of the landed scale's own bare
canonical.

Design/grounding of record: `Claude/Popper/t1944-t1942-armcde-collapse-debunk-2026-08-12.md`
§3.1 (the located defect, proven by counter-experiment) and §1's headline; the legacy
per-layer path's own derivation and comment at `pipeline.py:2082-2097`
(`_derive_composition_constants`); the engine consumer at
`D:\\SuperSLM\\include\\superslm\\forward_sites.h:140-175` and `src/model.cpp:736-765`
(`main@727e63e`, read-only grounding) -- `LandingRescale` reads `(r_t, e_t)` from
`KvLandingReciprocals` alone, `KvLandingScales` is never one of its arguments.

**No cell in the suite this file joins ever read the reciprocal triples' own quantity**
before this pin (T-1944's own finding on why four review rounds and 1882 tests missed the
defect: the toy-fixture cells in `test_armd_arme_kv_calibration.py` pin selection distinctness,
softmax_khead regeneration, and sweep behaviour, but assert nothing about what
`kv_reciprocals` itself carries). This file closes exactly that class.

Proven RED against the pre-fix line (`pipeline.py:2458`,
`kv_reciprocals[...] = (m, e, intmath.dynamic_scale_reciprocal(m))`, `weight_scales` gated
to arms D/E only) and GREEN after the fix, by executing both states of the file in-process
-- see `Claude/Brunel/t1945-kv-reciprocal-ratio-fix-2026-08-12.md` for the quoted failure.
"""
from fractions import Fraction

import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"


def fixture_config(pipeline):
    """Identical to the sibling suite's own §11 fixture shape
    (`test_armd_arme_kv_calibration.py::fixture_config`) -- two kv_heads is the minimum
    needed to state a per-head distinctness claim at all, and this pin reuses the same
    fixture rather than inventing a second one for the same contract."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


@pytest.fixture(scope="module")
def _kv_reciprocal_fixture(request):
    """Compute `calibrate_kv_landing_arm` once per arm (C, D, E -- every per-head arm the
    defect's own write site serves) on the shared toy fixture, plus the `weight_scales`
    this pin independently re-derives its own expected ratio from. Module-scoped so the
    (cheap but non-trivial) forward passes run once, not once per assertion."""
    pipeline = require(MODULE)
    calibrate_kv_landing_arm = api(MODULE, "calibrate_kv_landing_arm")
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    results = {
        arm: calibrate_kv_landing_arm(cfg, float_weight, records, tokenize, arm=arm)
        for arm in ("C", "D", "E")
    }
    return {"cfg": cfg, "weight_scales": weight_scales, "results": results, "pipeline": pipeline}


@pytest.mark.parametrize("arm", ["C", "D", "E"])
def test_kv_reciprocal_triple_equals_the_ratio_landingrescale_consumes(
        _kv_reciprocal_fixture, arm):
    """§3.1's own located property, stated as a positive contract: for every
    `(layer, k_head)` site of every per-head arm, `kv_reciprocals[key]` equals
    `(m_t, e_t, dynamic_scale_reciprocal(m_t))` where `(m_t, e_t) =
    canonical_scale(landed_scale / k_s_ref)`, `landed_scale` reconstructed from the SAME
    `kv_landing[key]` the converter actually wrote, and `k_s_ref` the K projection's own
    reference-fold denominator (`_reference_fold(weight_scales[f"{prefix}.k_proj"])[1]`)
    -- the identical quantity and formula the legacy per-layer path already uses
    (`pipeline.py:2088-2093`) and the engine's `LandingRescale` already consumes
    (`forward_sites.h:140-175`, `model.cpp:736-765`: `(r_t, e_t)` read from
    `KvLandingReciprocals` alone).

    This assertion is computed via a DIFFERENT path than the production write it checks:
    it starts from the OBSERVED `kv_landing[key]` (the converter's own already-written
    landed scale) and this fixture's own independently-fetched `weight_scales`, not from
    any intermediate the production code path shares internally beyond the two primitives
    (`_reference_fold`, `canonical_scale`) both the shipped legacy path and this pin treat
    as already-trusted arithmetic -- the property under test is which OPERANDS feed them,
    not their own correctness.

    **The exponent is asserted exact; the mantissa is asserted within +/-1.** Production
    computes `canonical_scale(selected_scale / k_s_ref)` from the RAW pre-canonicalization
    `selected_scale` (single rounding); this pin can only reconstruct `selected_scale` by
    replaying `kv_landing[key]` (already `canonical_scale`d once) back into a `Fraction`
    and dividing, which rounds a SECOND time. Confirmed by execution: the two paths agree
    on `e_t` exactly at all 12 fixture sites (2 layers x 2 kv_heads x 3 arms), with mantissas
    within +/-1: measured 5 exact, 6 at +1, 1 at -1 (T-1948/D-SLM2689 corrected the earlier
    "exactly 1 at every site" wording) -- the identical double-rounding-
    tie class `Claude/Popper/t1944-t1942-armcde-collapse-debunk-2026-08-12.md` §3.1 point 2
    already names ("double rounding in the reconstruction path -- value agreement <=1e-9,
    not a second defect"). This tolerance cannot mask the actual defect: the pre-fix write
    carries `e_t == kv_landing`'s own unshifted exponent, while the correct ratio's `e_t`
    is shifted by `~log2(1/k_s_ref)` (~7 bits on this fixture, `k_s_ref` measured
    `~1/127`) -- an exponent divergence the exact `e_t` check below catches regardless of
    the mantissa tolerance, and the reciprocal `r_t` is checked for INTERNAL consistency
    with whichever mantissa was actually observed, not tolerance-widened itself.

    RED against the pre-fix line (`kv_reciprocals[...] = (m, e,
    dynamic_scale_reciprocal(m))`, the scale's own bare canonical): every site's observed
    `e_t` equals `kv_landing`'s own unshifted exponent, which disagrees with the ratio's
    shifted `e_t` computed here at every one of the fixture's 4 (layer, k_head) sites
    (`num_hidden_layers=2 * num_key_value_heads=2`) whenever `k_s_ref != 1` (confirmed
    true on this fixture's own pinned weights, verified by execution -- see the build log
    for the quoted failure).
    """
    pipeline = _kv_reciprocal_fixture["pipeline"]
    cfg = _kv_reciprocal_fixture["cfg"]
    weight_scales = _kv_reciprocal_fixture["weight_scales"]
    result = _kv_reciprocal_fixture["results"][arm]

    checked = 0
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        _, k_s_ref = pipeline._reference_fold(weight_scales[f"{prefix}.k_proj"])
        for head in range(cfg.num_key_value_heads):
            key = f"{prefix}.k_head{head}"
            m, e = result["kv_landing"][key]
            landed_scale = Fraction(m) * Fraction(2) ** e
            m_t_expected, e_t_expected = pipeline.canonical_scale(landed_scale / Fraction(k_s_ref))
            m_t_observed, e_t_observed, r_t_observed = result["kv_reciprocals"][key]

            assert e_t_observed == e_t_expected, (
                f"arm {arm} {key}: kv_reciprocals exponent={e_t_observed}, but the ratio "
                f"canonical(landed_scale / k_s_ref) LandingRescale actually consumes has "
                f"exponent {e_t_expected} (k_s_ref={k_s_ref}) -- the converter wrote the "
                f"wrong quantity (this is the defect signature: an unshifted exponent "
                f"means the scale's own canonical was written where the ratio belongs)"
            )
            assert abs(m_t_observed - m_t_expected) <= 1, (
                f"arm {arm} {key}: kv_reciprocals mantissa={m_t_observed}, expected "
                f"{m_t_expected} (+/-1 double-rounding tolerance) -- a divergence this "
                f"large is not the reconstruction tie this pin tolerates"
            )
            assert r_t_observed == pipeline.intmath.dynamic_scale_reciprocal(m_t_observed), (
                f"arm {arm} {key}: reciprocal={r_t_observed} is not "
                f"dynamic_scale_reciprocal of the triple's own mantissa ({m_t_observed}) "
                f"-- internally inconsistent triple"
            )
            checked += 1
    assert checked == cfg.num_hidden_layers * cfg.num_key_value_heads, (
        "fixture sanity: expected to check every (layer, k_head) site"
    )


def test_fixture_sanity_k_s_ref_is_not_identity_at_every_layer(_kv_reciprocal_fixture):
    """Fixture-validity precondition (T-1899's own convention): if `k_s_ref == 1` at every
    layer, the ratio quantity this pin checks would coincide with the bare scale at every
    site, and the pin above would pass even against the pre-fix write by accident. Confirms
    by execution that this fixture's own K-projection reference-fold denominator is
    non-trivial (`!= 1`) at both layers, so the two quantities this pin discriminates
    genuinely diverge on this fixture."""
    pipeline = _kv_reciprocal_fixture["pipeline"]
    cfg = _kv_reciprocal_fixture["cfg"]
    weight_scales = _kv_reciprocal_fixture["weight_scales"]
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        _, k_s_ref = pipeline._reference_fold(weight_scales[f"{prefix}.k_proj"])
        assert k_s_ref != 1, (
            f"{prefix}: k_s_ref == 1 -- this fixture cannot discriminate the ratio "
            f"quantity from the bare scale at this layer, the pin above would be vacuous "
            f"here"
        )
