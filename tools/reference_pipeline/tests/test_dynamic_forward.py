"""The W8A8-dynamic full-stack integer forward -> int32 logits (§15 harness, D-SLM48/D-SLM55).

Charter: D-SLM55 (2026-07-16 — the full-stack dynamic-scale simulated forward is §15 harness
work) realizing §15's harness requirement and §6.2 C19–C22 (D-SLM48). See the full pin in
Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §10.1 (forward_dynamic contract),
and this packet: Claude/Curie/packets/eval/packet-03.md.

Test-design record: Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §10
"""

import numpy as np
import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"


def fixture_config(pipeline):
    """The §11 fixture model's shape, mirrored from test_pipeline.py and test_rung1_harness_wiring.py."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def test_forward_dynamic_returns_int32_logits_one_row_per_token():
    """Shape and width: forward_dynamic returns int32 logits with one row per token.

    §6.5: the W8A8-dynamic forward returns int32 logits in the same envelope as forward().
    """
    forward_dynamic, fixture_model = api(MODULE, "forward_dynamic", "fixture_model")
    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    logits = forward_dynamic(model, [0, 1, 3, 5])
    arr = np.asarray(logits)
    assert arr.shape == (4, cfg.vocab_size)
    assert arr.dtype == np.int32


def test_forward_dynamic_embed_site_trace_equals_dynamic_scale_trace():
    """The "embed"-site subsumption: the full stack's entry quantization matches the A-3 narrow chain.

    The feature oracle tying the two public surfaces (dynamic_scale_trace and forward_dynamic).
    The full stack must subsume the narrow chain at entry, exactly — a full-stack forward whose
    entry quantization drifted from the pinned C19–C22 chain fails here against code that is
    already green (dynamic_scale_trace).
    """
    forward_dynamic, fixture_model, dynamic_scale_trace = api(
        MODULE, "forward_dynamic", "fixture_model", "dynamic_scale_trace")
    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    trace = []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace)
    embed_records = [r for r in trace if r["site"] == "embed"]
    narrow = dynamic_scale_trace(model, [0, 1, 3, 5])

    assert len(embed_records) == len(narrow) == 4
    for got, expected in zip(embed_records, narrow):
        for key in ("x_int", "Dprime", "Dn", "s", "R", "codes"):
            assert got[key] == expected[key]
    assert [r["token_index"] for r in embed_records] == [0, 1, 2, 3]


def test_forward_dynamic_scales_vary_per_token():
    """Per-token variation: the dynamic activation scales vary across the token sequence.

    This is what "dynamic" means observably: a per-token scale varies with the token.
    Executed 2026-07-16: the embedding rows of [0,1,3,5] carry D' = 119,122,127,125 —
    pairwise distinct. The rung-1 convention tokens [1,5,9,2] carry D' = 122,125,125,122 —
    NOT pairwise distinct — and are therefore NOT the variation fixture. The pinned decisive
    sequence [0,1,3,5] is required. Cell 3 uses this fixture and no other.

    The static arm cannot produce this trace at all (it computes no scale — the existing green
    sentinel test_a_forward_pass_does_not_compute_a_scale proves it).
    """
    forward_dynamic, fixture_model = api(MODULE, "forward_dynamic", "fixture_model")
    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    trace = []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace)
    embed_dprimes = [r["Dprime"] for r in trace if r["site"] == "embed"]
    assert len(set(embed_dprimes)) == 4  # pairwise distinct


def test_forward_dynamic_cached_equals_uncached_bit_for_bit():
    """Cache bit-identity: cached and uncached forwards produce identical logits.

    The KVCache contract unchanged (pipeline.forward's docstring): the cached form is
    bit-identical to the uncached forward over the whole prefix. Note: the cache is REAL
    here (not A-3's no-op forward_dynamic_scale), and each cached position carries the
    codes computed under its own token's scale.
    """
    forward_dynamic, fixture_model, new_kv_cache = api(
        MODULE, "forward_dynamic", "fixture_model", "new_kv_cache")
    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    uncached = np.asarray(forward_dynamic(model, [0, 1, 3, 5]))
    cache = new_kv_cache(model)
    first = np.asarray(forward_dynamic(model, [0, 1], cache=cache))
    rest = np.asarray(forward_dynamic(model, [3, 5], cache=cache))
    assert np.array_equal(np.concatenate([first, rest]), uncached)


def test_forward_dynamic_is_deterministic_across_calls():
    """Determinism: calling forward_dynamic twice with the same inputs produces identical results.

    A consistency oracle, deliberately paired beside cells 2/3's feature oracles, never
    substituting for them — §17 dim 6's shape (the consistency oracle pattern).
    """
    forward_dynamic, fixture_model = api(MODULE, "forward_dynamic", "fixture_model")
    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    trace_a = []
    logits_a = np.asarray(forward_dynamic(model, [0, 1, 3, 5], trace=trace_a))

    trace_b = []
    logits_b = np.asarray(forward_dynamic(model, [0, 1, 3, 5], trace=trace_b))

    assert np.array_equal(logits_a, logits_b)
    assert len(trace_a) == len(trace_b)
    for rec_a, rec_b in zip(trace_a, trace_b):
        assert rec_a == rec_b


def test_forward_dynamic_site_set_is_constant_across_inputs():
    """Site-set constancy: the ordered list of sites is constant across different inputs of equal length.

    §14's data-independence shape: the ordered site list is a function of the configuration,
    never of the data. Same assertion shape as the rung-1 trace-length cells.
    """
    forward_dynamic, fixture_model = api(MODULE, "forward_dynamic", "fixture_model")
    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    trace_a, trace_b = [], []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace_a)
    forward_dynamic(model, [7, 2, 30, 11], trace=trace_b)  # different tokens, same length
    assert [r["site"] for r in trace_a] == [r["site"] for r in trace_b]

    # Curie 2026-07-16 (verification pass, design record amendment A-2): the site set must
    # extend BEYOND the entry site. D-SLM55 charters per-token scales "at every requant site
    # through all layers"; a forward whose only traced site is "embed" is the static pipeline
    # wearing a fabricated entry trace (dynamic_scale_trace is green and callable by anyone),
    # and every other cell in this file would pass it. Site names stay implementation-
    # enumerated; the COUNT is the structural floor.
    assert len({r["site"] for r in trace_a}) > 1, (
        "the trace names only the entry site; a full-stack dynamic forward must trace "
        "mid-network requant sites (D-SLM55: 'at every requant site through all layers')"
    )
