"""§15 W6 fidelity pair for forward_dynamic (behavioral + sentinel).

Charter: §15's W6 harness requirement extended to the full stack (D-SLM55):
"D and R are computed by the pinned C19/C20 integer constructions — a float per-token scale
(torch `amax` + a floating-point divide) ... measures a pipeline that is not the one that ships."
The rung-1 sentinels (test_rung1_harness_wiring.py, green at HEAD) cover `forward_dynamic_scale`'s
narrow chain; a full-stack forward whose MID-NETWORK sites reached for a float scale would pass them.
This packet closes that gap.

Test-design record: Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md amendment A-3
(referenced in packet E4 as the rung-1 amendment A-3 for the forbidden list), plus this packet:
Claude/Curie/packets/eval/packet-04.md.
"""

import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"
INTMATH = "reference_pipeline.intmath"


def fixture_config(pipeline):
    """The §11 fixture model's shape, mirrored from test_pipeline.py and test_rung1_harness_wiring.py."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def test_every_forward_dynamic_trace_record_recomputes_through_intmath():
    """Behavioral cell: every trace record, every site, recomputes through the pinned chain.

    The feature oracle: whatever sites the implementation enumerates (implementation-defined),
    each record's (Dprime, Dn, s, R, codes) must be the pinned chain's own output recomputed
    from the recorded x_int by intmath's scalar constructions. A site whose scale came from
    a float amax+divide diverges at the decisive Dn values; more importantly, a site that
    recorded a scale it did not use cannot manufacture matching codes.
    """
    forward_dynamic, fixture_model = api(
        MODULE, "forward_dynamic", "fixture_model")
    max_abs_reduce, normalize_scale, dynamic_scale_reciprocal, requant_token_code = api(
        INTMATH, "max_abs_reduce", "normalize_scale", "dynamic_scale_reciprocal", "requant_token_code")
    from fractions import Fraction
    import rung1_ref
    from composition_ref import canonical_scale_oracle, residual_reconcile_oracle

    module = require(MODULE)
    model = fixture_model(fixture_config(module))

    trace = []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace)
    assert trace, "an empty trace proves nothing — the forward must have dynamic sites"

    # A-7 amendment (D-SLM58 / the A-4 per-(token, head) k/v record shape): the trace
    # carries TWO record types and each recomputes through ITS pinned construction.
    # k/v landing records have no chain fields — the landing is the one-rounding C26
    # composite with the OFFLINE reciprocal, not the C19-C22 chain.
    chain_records = [r for r in trace if "Dprime" in r]
    kv_records = [r for r in trace
                  if r["site"].endswith(("k_proj.requant", "v_proj.requant"))]
    assert chain_records and kv_records, (
        "both record types must be present — a trace with only one type is either "
        "missing the dynamic chain sites or missing the k/v landing records")

    for record in chain_records:
        d_prime = max_abs_reduce(record["x_int"])
        dn, s = normalize_scale(d_prime)
        r = dynamic_scale_reciprocal(dn)
        assert record["Dprime"] == d_prime
        assert record["Dn"] == dn
        assert record["s"] == s
        assert record["R"] == r
        assert tuple(record["codes"]) == tuple(
            requant_token_code(int(x_i), r, s) for x_i in record["x_int"])

    for record in kv_records:
        kind = "k" if record["site"].endswith("k_proj.requant") else "v"
        prefix = record["site"].rsplit(".", 2)[0]
        s_target = model.scales.scale(f"{prefix}.{kind}_head{record['head']}.scale")
        s_ref = max(model.weight_scales[f"{prefix}.{kind}_proj"])
        m_t, e_t = canonical_scale_oracle(Fraction(s_target) / Fraction(s_ref))
        r_t = rung1_ref.reciprocal_oracle(m_t)      # OFFLINE reciprocal (D-SLM58)
        assert tuple(record["codes"]) == tuple(
            max(-127, min(127, residual_reconcile_oracle(
                int(x), record["m_in"], r_t, record["e_in"], e_t)))
            for x in record["x_int"])


def test_forward_dynamic_never_calls_float_amax_or_float_divide(monkeypatch):
    """The W6 full-stack sentinel (S-2): catches the library call at its origin.

    Mirrors test_rung1_harness_wiring.py::test_forward_dynamic_scale_never_calls_float_amax_or_float_divide
    exactly, changing only the function under test (forward_dynamic, not forward_dynamic_scale)
    and the token sequence (the E3 pinned sequence [0, 1, 3, 5], not the rung-1 convention tokens).

    If forward_dynamic reaches torch.amax/torch.Tensor.amax/np.amax, or a float divide via
    torch.div/torch.true_divide/torch.Tensor.div on ANY site's scale-computation path, this
    fails — regardless of whether a given fixture happened to expose a numeric divergence
    (the behavioral cell's job). Pairs with the behavioral cell exactly as the rung-1 pattern
    does.

    The forbidden-call list is pinned (design doc amendment A-3): np.amax, torch.amax,
    torch.Tensor.amax, torch.div, torch.true_divide, torch.Tensor.div, torch.Tensor.true_divide.
    np.ndarray.max is EXCLUDED (A-3): unpatchable and false-positive on the static path's
    envelope checks.

    Apparatus smoke status (A-3 discipline): this exact patch set runs GREEN today inside
    test_rung1_harness_wiring.py's sentinel (HEAD's suite is green), so the patching lines
    are proven runnable. While red (api() raises first), the apparatus does not execute;
    nonetheless the apparatus is proven by the green rung-1 sentinel that shares it, which is
    what A-3 requires. No new smoke stand-in owed.
    """
    import numpy as np
    forward_dynamic, fixture_model = api(
        MODULE, "forward_dynamic", "fixture_model")

    def forbidden(name):
        def _raise(*a, **k):
            raise AssertionError(
                f"{name} was called on the dynamic-scale reproducible path; §15/W6 requires "
                f"D and R to be computed by intmath's pinned C19/C20/C21 integer constructions"
            )
        return _raise

    monkeypatch.setattr(np, "amax", forbidden("numpy.amax"), raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "amax", forbidden("torch.amax"), raising=False)
    monkeypatch.setattr(torch.Tensor, "amax", forbidden("torch.Tensor.amax"), raising=False)
    monkeypatch.setattr(torch, "div", forbidden("torch.div"), raising=False)
    monkeypatch.setattr(torch, "true_divide", forbidden("torch.true_divide"), raising=False)
    monkeypatch.setattr(torch.Tensor, "div", forbidden("torch.Tensor.div"), raising=False)
    monkeypatch.setattr(torch.Tensor, "true_divide", forbidden("torch.Tensor.true_divide"), raising=False)

    module = require(MODULE)
    model = fixture_model(fixture_config(module))
    forward_dynamic(model, [0, 1, 3, 5])  # must complete without touching any forbidden call
