"""Curie's red suite for D-SLM394 / audit F4 (Claude/Mendeleev/superslm-s3.3-
attention-interior-coverage-audit-2026-07-28.md Sec4): the premise every S3.3
`ctx_fold` cell in `tests/test_main.cpp` depends on -- that the shipped
converter's `layer{L}.ctx_fold` emitter (`convert_model._ctx_fold_tensor`)
emits only identity `(1, 0, 0)` rows -- had no committed, CI-executed test
inside this repository. Its only proof was a probe script under `D:\\Wizard`'s
records tree, run once, historically, outside this repo's CI.

RUNS THE REAL EMITTER, NEVER A REIMPLEMENTATION. `convert_model._ctx_fold_tensor`
is the actual function `convert_model.build_sections` calls for every layer of
every real conversion (tools/convert_model.py, `_ctx_fold_tensor`) -- this
module calls that function directly, exactly as shipped, against a
representative checkpoint config (the attribute contract `_ctx_fold_tensor`/
`_ctx_channel_scales` read: `model.config.num_attention_heads`,
`model.config.num_key_value_heads`, `model.scales.scale(key)` -- the same
minimal-contract pattern `sslm_pinned_calibration_fixture.py` and
`test_sslm_convert_validate.py` already use for the rest of this module
family). Never re-derives `quantize_multiplier`'s arithmetic itself: plan
Sec4.3's own finding (SuperSLM_S3a_WalkingSkeleton_Plan.md, vendored at
tests/reference/superslm_spike/pipeline_prob_width_ceiling.py's own docstring)
is that reading the emitter's source to build a "join" oracle produces a
correlated oracle, not an independent one -- this suite instead RUNS the real
emitter and reads its actual return value.

SKIPPED, NOT SILENTLY PASSED, IF `reference_pipeline` IS EVER UNIMPORTABLE.
`_ctx_fold_tensor` unconditionally calls `convert_model._load_spike()`
(`from reference_pipeline import artifact_cache, pipeline` as of T-2123/T-2137
B0/B5 -- vendored in-tree at `tools/reference_pipeline/`, no longer a
cross-tree `D:\\Wizard\\Tools` import) before it ever inspects a scale value --
this is a hard, unconditional dependency of the real emitter itself (not a
choice this test module makes), and it is why every OTHER module in this
`tools/` family (sslm_convert_validate.py's own docstring;
sslm_pinned_calibration_fixture.py's own docstring) deliberately injects a
fixture stand-in instead of calling `convert_model._ctx_fold_tensor` directly.
Before B5, this dependency was a cross-tree `D:\\Wizard` import and this suite
could not run in CI at all (`.github/workflows/tests.yml`'s
`converter-validate` job runs `pytest tools/` on `ubuntu-latest` with no
`D:\\Wizard` sibling present); after B5, `reference_pipeline` is vendored
in-tree and importable everywhere this repository is checked out, so the skip
condition below is now a defensive guard against `reference_pipeline` itself
being broken or absent, not a routine CI condition -- it is
`pytest.mark.skipif`-gated on the same import this file's own function under
test performs, the identical pattern `tools/test_pinned_calibration_gate.py`
already uses for `sslm_verify` not being built locally. This suite runs the
real emitter for real and fails loudly the day a calibration change starts
emitting a non-identity row -- closing D-SLM394's "checked nowhere" gap.

Test-design record:
Claude/Curie/superslm-s3.3-t1317-ctx-fold-identity-premise-test-design-2026-07-28.md
"""

import types

import pytest

import convert_model as C


def _spike_available():
    try:
        C._load_spike()
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _spike_available(),
    reason="reference_pipeline (vendored in-tree, T-2123/T-2137) is not importable in this "
           "environment -- see module docstring; this is the real emitter's own unconditional "
           "dependency, not a choice this test module makes")


class _Scales:
    """Stands in for the spike's `model.scales` -- `.scale(key)` returns a
    positive float per named site, the same minimal contract
    `sslm_pinned_calibration_fixture.py`'s `_Scales` already uses."""

    def __init__(self, table):
        self._table = table

    def scale(self, key):
        return self._table[key]


def _representative_model(kv_head_scales_by_layer, num_attention_heads=4, num_key_value_heads=2):
    """A representative checkpoint config -- the minimal attribute contract
    `convert_model._ctx_fold_tensor`/`_ctx_channel_scales` read, geometrically
    coherent (num_attention_heads divisible by num_key_value_heads, GQA
    group=2), never a real calibrated artifact (none is available in this
    repository or its CI -- see module docstring). `kv_head_scales_by_layer`
    is `{layer_index: [scale_per_kv_head]}`, letting each cell below choose
    whether the per-KV-head V scales agree (today's shipped calibration
    behavior) or diverge (the future-calibration-change risk plan Sec14.9
    names as live)."""
    cfg = types.SimpleNamespace(num_attention_heads=num_attention_heads,
                                num_key_value_heads=num_key_value_heads)
    table = {}
    for layer, kv_scales in kv_head_scales_by_layer.items():
        assert len(kv_scales) == num_key_value_heads
        for h, s in enumerate(kv_scales):
            table[f"layer{layer}.v_head{h}.scale"] = s
    return types.SimpleNamespace(config=cfg, scales=_Scales(table))


def test_ctx_fold_emitter_emits_only_identity_rows_on_a_representative_checkpoint():
    """D-SLM394 / audit F4's own cell: a representative checkpoint whose per-KV-head V
    scales agree within every layer (today's shipped calibration behavior -- the premise
    every S3.3 hand-built `ctx_fold` cell in tests/test_main.cpp depends on) must emit
    ONLY identity `(1, 0, 0)` rows from the REAL `convert_model._ctx_fold_tensor`, for
    every attention head of every layer."""
    num_layers = 3
    model = _representative_model({L: [0.75, 0.75] for L in range(num_layers)})

    for layer in range(num_layers):
        rows = C._ctx_fold_tensor(model, layer)
        assert rows.shape == (4, 3), f"layer{layer}.ctx_fold: shape {rows.shape}, want (4, 3)"
        for head, row in enumerate(rows):
            identity, mult, shift = (int(x) for x in row)
            assert (identity, mult, shift) == (1, 0, 0), (
                f"layer{layer}.ctx_fold head{head}: real convert_model._ctx_fold_tensor emitted "
                f"({identity}, {mult}, {shift}), want the identity row (1, 0, 0) -- the premise "
                f"every S3.3 ctx_fold cell in tests/test_main.cpp depends on (D-SLM394) no longer "
                f"holds for this representative checkpoint")


def test_ctx_fold_emitter_emits_a_non_identity_row_when_kv_head_scales_diverge():
    """Guard-vitality half of D-SLM394 / F4: proves the previous cell's assertion is not
    vacuously true by construction of the emitter -- when per-KV-head V scales genuinely
    DIVERGE (the future-calibration-change risk plan Sec14.9 names as live), the SAME
    real `convert_model._ctx_fold_tensor` must emit a non-identity row for the head NOT
    at the layer's max scale. Without this cell, a suite that only ever fed
    equal-scale fixtures could not tell "the emitter always emits identity" from "this
    fixture never gave it a reason not to.\""""
    model = _representative_model({0: [0.75, 0.90]}, num_attention_heads=4, num_key_value_heads=2)

    rows = C._ctx_fold_tensor(model, 0)
    assert rows.shape == (4, 3)
    identities = [int(row[0]) for row in rows]
    # group = 4 // 2 = 2: heads 0-1 share kv_head0 (scale 0.75, below the layer max 0.90,
    # so NOT identity); heads 2-3 share kv_head1 (scale 0.90, the layer max, identity).
    assert identities == [0, 0, 1, 1], (
        f"layer0.ctx_fold identity flags {identities}, want [0, 0, 1, 1] -- kv_head0 (0.75) is "
        f"below the layer's max V scale (0.90, kv_head1) and must fold; kv_head1, already at the "
        f"max, must stay identity. If this fails as [1, 1, 1, 1], the real emitter is not reacting "
        f"to diverging per-head scales at all, and the previous cell's all-identity assertion would "
        f"be vacuous.")
    for head in (0, 1):
        _, mult, shift = (int(x) for x in rows[head])
        assert mult != 0 or shift != 0, f"head{head}: non-identity row carries a zero (mult, shift)"
