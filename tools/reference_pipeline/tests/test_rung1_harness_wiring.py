"""The W6 rung-1 sentinels — dynamic per-token scale hashing.

§15 (SuperSLM_Plan.md): *"a float per-token scale (torch `amax` + a floating-point divide)
differs from `round_half_up(2^62/D')` at the edges and measures a pipeline that is not the one
that ships"*. These two cells close that gap.

The two EXISTING harness-fidelity sentinels (`test_pipeline.py:421`
`test_no_library_rounding_is_called_on_the_reproducible_path`, `test_pipeline.py:659`
`test_a_forward_pass_does_not_compute_a_scale`) are scoped to the **static** baseline — they
say nothing about the **dynamic** forward's scale-COMPUTATION step. A rung-1 harness wired to
`x.abs().amax(dim=-1)` + a float divide for `1/D'` would pass every other rung-1 cell in this
suite while the *pipeline that actually runs the eval* never calls them — exactly the W6 failure
mode, arriving through the one code path G-8 exists to cover.

Test-design record: Claude/Curie/packets/packet-08.md
"""

import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"
INTMATH = "reference_pipeline.intmath"


def fixture_config(pipeline):
    """The §11 fixture model's shape, mirrored from test_pipeline.py."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def test_forward_dynamic_scale_per_token_scales_match_the_pinned_integer_construction():
    """Behavioural cell: the traced scale computation matches intmath's scalar functions exactly.

    Mirrors `test_pipeline.py`'s decisive-fixture pattern. A token whose max-abs element
    drives Dn to a value where a float amax+divide (IEEE double, round-to-nearest-even)
    disagrees with round_half_up's exact tie rule — confirmed inline, not merely asserted.

    NOTE: the fixture-model plumbing follows test_pipeline.py's calling convention
    (`module = require(MODULE); model = fixture_model(fixture_config(module))`), resolved
    against the existing harness conventions in this file's `fixture_config` helper.

    NARROWING (packet-08's stated fallback, taken): the decisive-Dn proof below stands
    alone -- the forward-pass loop asserts the general trace-vs-intmath property on an
    arbitrary token, without engineering the fixture model's activations to reach
    dn_probe exactly. The dn_probe block is the standalone proof the property is decisive
    in principle; the sentinel cell below catches the float path at its origin regardless.
    """
    forward_dynamic_scale, dynamic_scale_trace, fixture_model = api(
        MODULE, "forward_dynamic_scale", "dynamic_scale_trace", "fixture_model")
    max_abs_reduce, normalize_scale, dynamic_scale_reciprocal = api(
        INTMATH, "max_abs_reduce", "normalize_scale", "dynamic_scale_reciprocal")
    from rung1_ref import reciprocal_oracle

    # Decisive fixture: a Dn where a float amax+divide (IEEE double, round-to-nearest-even)
    # disagrees with round_half_up's exact tie rule -- confirmed inline, not merely asserted.
    # Curie 2026-07-16 (verification pass): the brief's first value, 2**30 + 1, is NOT
    # decisive -- at ratio magnitude ~2**32 the double's error (~2**-20 relative) only
    # crosses a rounding boundary for ~1-in-500k Dn values, and no small offset from the
    # domain floor qualifies. This value is bench-derived decisive: exact R = 2152302821,
    # IEEE-double round gives 2152302820.
    dn_probe = 2142675266
    exact_r = reciprocal_oracle(dn_probe)
    float_r = round(2 ** 62 / dn_probe)   # IEEE double divide + Python's own round()
    assert float_r != exact_r, (
        "fixture is not decisive -- float divide happens to agree with round_half_up at "
        f"Dn={dn_probe}; re-derive a decisive value (see packet-08, amended 2026-07-16)"
    )

    # Build model exactly as test_pipeline.py's own cells do.
    module = require(MODULE)
    model = fixture_model(fixture_config(module))
    tokens = [1, 5, 9, 2]
    trace = dynamic_scale_trace(model, tokens)
    for record in trace:
        expected_d_prime = max_abs_reduce(record["x_int"])   # the token's own int32 activation vector
        expected_dn, expected_s = normalize_scale(expected_d_prime)
        expected_r = dynamic_scale_reciprocal(expected_dn)
        assert record["Dprime"] == expected_d_prime
        assert record["Dn"] == expected_dn
        assert record["s"] == expected_s
        assert record["R"] == expected_r


def test_forward_dynamic_scale_never_calls_float_amax_or_float_divide(monkeypatch):
    """The W6 rung-1 sentinel (S-1): catches the library call at its origin.

    If forward_dynamic_scale reaches torch.amax/torch.Tensor.amax/np.amax, or a float divide
    via torch.div/torch.true_divide/torch.Tensor.div on the scale-computation path, this
    fails -- regardless of whether a given fixture happened to expose a numeric divergence
    (cell 1's job). Pairs with cell 1 exactly as test_no_library_rounding_is_called_on_the_
    reproducible_path pairs with the tie-direction behavioural cells in test_pipeline.py.

    The forbidden-call list is pinned (design doc amendment A-3): `np.amax`,
    `torch.amax`, `torch.Tensor.amax`, `torch.div`, `torch.true_divide`,
    `torch.Tensor.div`, `torch.Tensor.true_divide`. `np.ndarray.max` was in the
    first pin and is REMOVED (A-3, 2026-07-16): CPython refuses setattr on immutable
    C-extension types (`raising=False` covers a missing attribute, not an immutable
    type), and independently the static path's envelope checks call `ndarray.max` on
    code the dynamic change never touches, so the entry was both unpatchable and a
    false positive. A raw `.max()` float misuse on the scale path is covered by the
    behavioural cell above -- which is why the two cells exist as a pair.
    """
    import numpy as np
    forward_dynamic_scale, fixture_model = api(
        MODULE, "forward_dynamic_scale", "fixture_model")

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
    forward_dynamic_scale(model, [1, 5, 9, 2])   # must complete without touching any forbidden call
