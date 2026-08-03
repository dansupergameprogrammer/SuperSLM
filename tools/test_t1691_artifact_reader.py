"""T-1691 step 1's own red-first proof: `sslm_artifact_reader.py`'s
independent `.sslm` parser, checked against KNOWN, hand-authored artifact
contents -- the two existing hand-derived fixtures this campaign already
built (`kv_saturation_fixture.py`, `kv_context_trailer_fixture.py`), whose
every weight, fold triple, and site constant is a value this test already
knows independent of the reader (it is the SAME value the fixture writer
itself set, module-level constants in each fixture file) -- not a value
the reader's own output is compared against itself. A defect that made the
reader silently return the wrong tensor, the wrong layer's data, or a
byte-shifted value would be caught here, before this reader is trusted on
the real 28-layer artifact (design S7 step 1's own "no shared apparatus"
argument: this reader shares no code with either fixture writer or with
the production C++ loader/marshal)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_context_trailer_fixture as CTX_FIX  # noqa: E402
import kv_saturation_fixture as SAT_FIX  # noqa: E402
import sslm_artifact_reader as R  # noqa: E402


def test_config_matches_the_multi_layer_fixtures_own_geometry(tmp_path):
    _tok, model_path = CTX_FIX.write_fixture(tmp_path)
    art = R.read_artifact(str(model_path))
    assert art.config == {
        "hidden_size": CTX_FIX.HIDDEN_SIZE,
        "num_hidden_layers": CTX_FIX.NUM_HIDDEN_LAYERS,
        "num_attention_heads": CTX_FIX.NUM_ATTENTION_HEADS,
        "num_key_value_heads": CTX_FIX.NUM_KEY_VALUE_HEADS,
        "head_dim": CTX_FIX.HEAD_DIM,
        "intermediate_size": CTX_FIX.INTERMEDIATE_SIZE,
        "vocab_size": CTX_FIX.VOCAB_SIZE,
        "context_cap": CTX_FIX.CONTEXT_CAP,
        "tie_word_embeddings": True,
        "kv_precision": 0,
        "kv_block_size": 1,
    }
    assert len(art.layers) == CTX_FIX.NUM_HIDDEN_LAYERS


def test_weights_and_folds_match_the_fixtures_own_known_construction(tmp_path):
    """Every layer of kv_context_trailer_fixture.py uses the IDENTICAL
    identity-K/V / all-zero-branch / identity-down construction (its own
    module docstring) -- checked at every layer, not just layer 0, so a
    defect that only shows up past the first `layerN.` prefix (an
    off-by-one in the per-layer name construction) is caught."""
    _tok, model_path = CTX_FIX.write_fixture(tmp_path)
    art = R.read_artifact(str(model_path))
    for l, layer in enumerate(art.layers):
        assert (layer.q_weight == 0).all(), f"layer{l}.q_proj expected all-zero"
        assert (layer.o_weight == 0).all(), f"layer{l}.o_proj expected all-zero"
        assert (layer.gate_weight == 0).all(), f"layer{l}.gate_proj expected all-zero"
        assert (layer.up_weight == 0).all(), f"layer{l}.up_proj expected all-zero"
        assert (layer.k_weight == [[1, 0], [0, 1]]).all(), f"layer{l}.k_proj expected identity"
        assert (layer.v_weight == [[1, 0], [0, 1]]).all(), f"layer{l}.v_proj expected identity"
        assert (layer.down_weight == [[1, 0], [0, 1]]).all(), f"layer{l}.down_proj expected identity"
        # WSC1 fold: every projection's identity flag is 1, mult 0, shift 0
        # (the fixture's own `_identity_fold`).
        for fold, channels in ((layer.q_fold, CTX_FIX.HIDDEN_SIZE), (layer.k_fold, 2),
                                (layer.v_fold, 2), (layer.o_fold, CTX_FIX.HIDDEN_SIZE),
                                (layer.gate_fold, CTX_FIX.INTERMEDIATE_SIZE),
                                (layer.up_fold, CTX_FIX.INTERMEDIATE_SIZE),
                                (layer.down_fold, CTX_FIX.HIDDEN_SIZE)):
            assert fold.shape == (channels, 3)
            assert (fold[:, 0] == 1).all() and (fold[:, 1] == 0).all() and (fold[:, 2] == 0).all()
        assert layer.q_bias is None and layer.k_bias is None and layer.v_bias is None
        assert layer.kv_landing_e_t_k == [CTX_FIX.KV_LANDING_E_T]
        assert layer.kv_landing_r_t_k == [CTX_FIX.KV_LANDING_R_T]
        assert layer.kv_landing_e_t_v == [CTX_FIX.KV_LANDING_E_T]
        assert layer.kv_landing_r_t_v == [CTX_FIX.KV_LANDING_R_T]
        assert layer.iexp_softmax_khead_m == [CTX_FIX.SOFTMAX_KHEAD_CONST[0]]
        assert layer.iexp_softmax_khead_e == [CTX_FIX.SOFTMAX_KHEAD_CONST[1]]
        for sc in (layer.attn_norm_site_constant, layer.q_site_constant, layer.o_site_constant,
                   layer.ctx_fold_site_constant, layer.attn_residual_site_constant,
                   layer.mlp_norm_site_constant, layer.gate_site_constant, layer.up_site_constant,
                   layer.down_site_constant, layer.mlp_residual_site_constant):
            assert sc == CTX_FIX.CANONICAL
        assert layer.mlp_act_site_constant == CTX_FIX.MLP_ACT_CONST


def test_embed_final_norm_rope_and_sigmoid_lut(tmp_path):
    _tok, model_path = CTX_FIX.write_fixture(tmp_path)
    art = R.read_artifact(str(model_path))
    assert (art.embed[:3] == [5, -5]).all()
    assert (art.final_norm_gain == 1).all()
    assert art.embed_site_constant == CTX_FIX.EMBED_CONST
    assert art.final_norm_site_constant == CTX_FIX.CANONICAL
    assert art.rope_cos.shape == (CTX_FIX.CONTEXT_CAP,)
    assert (art.rope_cos == 1073741824).all()
    assert (art.rope_sin == 0).all()
    assert art.sigmoid_lut.shape == (1025,)
    assert art.lm_head is None  # tied embeddings


def test_second_independent_fixture_kv_saturation_geometry(tmp_path):
    """A second, independently-authored fixture (T-1689's own, single-layer,
    single-token) -- cross-checks the reader against a SECOND known-ground-
    truth construction, not only the one this file's own author (this same
    build) wrote."""
    tok_path, sat_path, nosat_path = SAT_FIX.write_fixture(tmp_path)
    sat = R.read_artifact(str(sat_path))
    nosat = R.read_artifact(str(nosat_path))
    assert len(sat.layers) == 1 == len(nosat.layers)
    assert (sat.layers[0].k_weight == [[127, 0], [0, 0]]).all()
    assert (sat.layers[0].v_weight == [[127, 0], [0, 0]]).all()
    assert (nosat.layers[0].k_weight == [[1, 0], [0, 1]]).all()
    assert sat.embed[0].tolist() == [5, -5]
    assert sat.layers[0].kv_landing_e_t_k == [SAT_FIX.KV_LANDING_E_T]
    assert sat.layers[0].kv_landing_r_t_k == [SAT_FIX.KV_LANDING_R_T]


def test_structural_rejection_on_a_truncated_file(tmp_path):
    """Reject-over-degrade (module docstring): a truncated artifact is
    rejected loudly (ValueError), never silently mis-parsed."""
    _tok, model_path = CTX_FIX.write_fixture(tmp_path)
    truncated = tmp_path / "truncated.sslm"
    truncated.write_bytes(model_path.read_bytes()[:100])
    try:
        R.read_artifact(str(truncated))
        assert False, "expected SslmArtifactFormatError on a truncated file"
    except R.SslmArtifactFormatError:
        pass
