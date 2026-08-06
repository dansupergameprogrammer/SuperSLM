"""Red suite for run_criterion2_trace.py (T-1522, producer half; SuperSLM_S3a_
WalkingSkeleton_Plan.md Sec11 S3.1c item 3).

Uses `superslm_spike.pipeline.fixture_model` -- the same deterministic,
real-weight-free constructor `Tools/superslm_spike/tests/test_dynamic_engine.py`
uses for its own bit-equality suite -- so these cells are hermetic and need no
real checkpoint. A separate, explicitly-marked cell runs the producer against
the real calibrated artifact this build has read access to, proving the
mechanism genuinely executes end to end; it is skipped, not failed, where that
artifact is unavailable (this project's own repo-availability convention,
matching check_source_drift.py's real-tree cell).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_criterion2_trace as rct
from superslm_spike import pipeline


def _fixture_config():
    """The vendored spike's own small fixture shape (matches
    Tools/superslm_spike/tests/test_dynamic_engine.py's fixture_config)."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _build_tiny_model(cfg):
    """A hermetic, real-checkpoint-free `QuantizedModel`, built WITHOUT
    `pipeline.fixture_model()`.

    `fixture_model()` cannot be called from this vendored location: its
    `calibration_records()` reads `CALIBRATION_CORPUS_PATH`, a path computed
    relative to `pipeline.py`'s own file location (`parents[2]` -- two
    directories up), which resolves inside `D:\\Wizard` where the file
    normally lives but resolves to a nonexistent path once the whole file is
    vendored two directories shallower here (`tests/reference/superslm_spike/
    pipeline.py`); and its calibration lambda calls `run_prompt_messages`,
    which imports `superslm_spike.baseline` -- not part of the vendored
    closure (Sec11 S3.1c item 1 deliberately excludes it: "baseline.py alone
    would pull torch and transformers into a build that must never acquire a
    real-weight dependency"). Neither gap is a defect in this build: nothing
    in criterion 2's own design ever calls `fixture_model()` from the
    vendored copy -- the real execution path loads a genuine calibrated
    artifact instead (see test_the_producer_runs_end_to_end_against_the_real_
    calibrated_artifact below). This helper reproduces `fixture_model()`'s
    OWN body (pinned weights, the fixture's byte-level tokenizer, `_calibrate`)
    with hand-built calibration records in place of the corpus, so this
    file's hermetic cells need neither the corpus nor `baseline.py`.
    """
    pipeline.attention_group_size(cfg)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    fixture_tokenize = pipeline._fixture_tokenize_prompt(cfg)

    def tokenize_record(record):
        return fixture_tokenize([{"role": "user", "content": record}])

    records = ["hello world", "a different calibration string", "a third one"]
    maxima = pipeline._calibrate(cfg, float_weight, records, tokenize_record)
    scales, residual_scales, biases = pipeline._derive_scales(cfg, maxima, weight_scales, {})
    composition_constants, kv_landing_scales, kv_landing_reciprocals = (
        pipeline._derive_composition_constants(cfg, weight_scales, scales)
    )
    calibration = pipeline.CalibrationRecord(
        corpus_sha256="0" * 64, tokenization="test-only tiny fixture, not the real corpus", classes=()
    )
    return pipeline.QuantizedModel(
        config=cfg, scales=scales, weights=weights, weight_scales=weight_scales,
        residual_scales=residual_scales, rope_tables=pipeline._build_rope_tables(cfg),
        biases=biases, float_source=float_weight, tokenize_prompt=fixture_tokenize,
        calibration=calibration, gemm_weights={}, composition_constants=composition_constants,
        kv_landing_scales=kv_landing_scales, kv_landing_reciprocals=kv_landing_reciprocals,
    )


def _write_model_pickle(tmp: str) -> str:
    model = _build_tiny_model(_fixture_config())
    # forward_dynamic_vec touches neither float_source (calibration-only) nor
    # tokenize_prompt (prompt-rendering-only) -- both are closures over a
    # live callable and are not reliably picklable (the real artifact's own
    # loader carries the identical shape: `artifact_cache.load_artifact`
    # reopens a live, lazy float_source, or a closure stub when the
    # checkpoint path is unavailable). run_criterion2_trace.py's own
    # load_model() docstring states this is the expected shape of a
    # producer-facing model pickle.
    picklable = dataclasses.replace(model, float_source=None, tokenize_prompt=None)
    path = os.path.join(tmp, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(picklable, f)
    return path


def _write_prompts(tmp: str, members: list[dict]) -> str:
    path = os.path.join(tmp, "prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"members": members}, f)
    return path


_SMALL_MEMBERS = [
    {"index": 1, "axis": "tiny", "source": "hand-built", "expected_verdict": "n/a", "input_ids": [0, 1, 3, 5]},
    {"index": 2, "axis": "tiny-2", "source": "hand-built", "expected_verdict": "n/a", "input_ids": [2, 4]},
]


# --- load_prompt_pack / load_model: the two named failure modes. ---


def test_missing_prompts_file_raises_naming_the_path_and_the_precompute_script():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "does_not_exist.json")
        with pytest.raises(FileNotFoundError, match="precompute_criterion2_prompt_pack"):
            rct.load_prompt_pack(missing)


def test_missing_model_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "does_not_exist.pkl")
        with pytest.raises(FileNotFoundError):
            rct.load_model(missing)


def test_main_exits_nonzero_on_missing_prompts():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _write_model_pickle(tmp)
        code = rct.main(["--model", model_path, "--prompts", os.path.join(tmp, "nope.json"), "--out", os.path.join(tmp, "out.jsonl")])
        assert code == 1


def test_main_exits_nonzero_on_missing_model():
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = _write_prompts(tmp, _SMALL_MEMBERS)
        code = rct.main(["--model", os.path.join(tmp, "nope.pkl"), "--prompts", prompts_path, "--out", os.path.join(tmp, "out.jsonl")])
        assert code == 1


def test_prompt_pack_is_read_in_index_order_regardless_of_file_order():
    with tempfile.TemporaryDirectory() as tmp:
        shuffled = [_SMALL_MEMBERS[1], _SMALL_MEMBERS[0]]
        path = _write_prompts(tmp, shuffled)
        members = rct.load_prompt_pack(path)
        assert [m["index"] for m in members] == [1, 2]


# --- _order_record: field-order and value-conversion contract. ---


def test_chain_record_is_reordered_to_the_pinned_field_order():
    raw = {
        "e_out": 5, "m_out": 4, "codes": (1, 2, 3), "R": 3, "s": 2, "Dn": 1,
        "Dprime": 10, "x_int": (7, 8), "token_index": 0, "site": "embed",
    }
    ordered = rct._order_record(raw)
    assert list(ordered.keys()) == list(rct._CHAIN_FIELDS)
    assert ordered["x_int"] == [7, 8]
    assert ordered["codes"] == [1, 2, 3]


def test_kv_landing_record_is_reordered_to_the_pinned_field_order():
    raw = {
        "e_out": 1, "m_out": 2, "codes": (9, 9), "e_in": 3, "m_in": 4,
        "x_int": (5, 6), "head": 0, "token_index": 1, "site": "layer0.k_proj.requant",
    }
    ordered = rct._order_record(raw)
    assert list(ordered.keys()) == list(rct._KV_FIELDS)


def test_a_record_missing_a_pinned_field_raises_naming_the_field():
    """Mutation proof: dynamic_engine's own record shape drifting (a field
    silently dropped) must be caught, not silently serialized as a partial
    record -- a silent partial trace is exactly the shape a real forward
    defect could take."""
    raw = {
        "site": "embed", "token_index": 0, "x_int": (1,), "Dprime": 1,
        "Dn": 1, "s": 1, "R": 1, "codes": (1,),
        # "m_out" deliberately missing
        "e_out": 1,
    }
    with pytest.raises(KeyError, match="m_out"):
        rct._order_record(raw)


def test_numpy_scalar_values_are_converted_to_plain_python_ints():
    np = pytest.importorskip("numpy")
    raw = {
        "site": "embed", "token_index": np.int64(0), "x_int": (np.int64(7), np.int64(8)),
        "Dprime": np.int64(1), "Dn": np.int64(1), "s": np.int64(1), "R": np.int64(1),
        "codes": (np.int64(1),), "m_out": np.int64(1), "e_out": np.int64(1),
    }
    ordered = rct._order_record(raw)
    # json.dumps must not choke on the result -- proves every value converted.
    encoded = json.dumps(ordered)
    assert "int64" not in encoded


# --- End-to-end against the fixture model: the wiring cell. ---


def test_run_pack_against_fixture_model_writes_valid_jsonl_with_final_logits():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _write_model_pickle(tmp)
        prompts_path = _write_prompts(tmp, _SMALL_MEMBERS)
        out_path = os.path.join(tmp, "out.jsonl")

        code = rct.main(["--model", model_path, "--prompts", prompts_path, "--out", out_path])
        assert code == 0

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # At least one site record plus one final_logits record per member.
        final_logits_lines = [l for l in lines if l.get("kind") == "final_logits"]
        assert [l["member_index"] for l in final_logits_lines] == [1, 2]
        site_lines = [l for l in lines if l.get("kind") != "final_logits"]
        assert len(site_lines) > 0, "expected at least one per-site trace record"

        for record in site_lines:
            if "head" in record:
                assert list(record.keys()) == list(rct._KV_FIELDS)
            else:
                assert list(record.keys()) == list(rct._CHAIN_FIELDS)

        for fl in final_logits_lines:
            assert isinstance(fl["logits"], list)
            assert len(fl["logits"]) > 0


def test_run_pack_is_deterministic_across_two_runs():
    """Same model, same prompts -> byte-identical output -- the reference
    forward has no hidden randomness or ordering nondeterminism."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _write_model_pickle(tmp)
        prompts_path = _write_prompts(tmp, _SMALL_MEMBERS)
        out1 = os.path.join(tmp, "out1.jsonl")
        out2 = os.path.join(tmp, "out2.jsonl")
        assert rct.main(["--model", model_path, "--prompts", prompts_path, "--out", out1]) == 0
        assert rct.main(["--model", model_path, "--prompts", prompts_path, "--out", out2]) == 0
        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            assert f1.read() == f2.read()


def test_a_single_member_pack_produces_exactly_one_final_logits_record():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _write_model_pickle(tmp)
        prompts_path = _write_prompts(tmp, [_SMALL_MEMBERS[0]])
        out_path = os.path.join(tmp, "out.jsonl")
        assert rct.main(["--model", model_path, "--prompts", prompts_path, "--out", out_path]) == 0
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        final_logits_lines = [l for l in lines if l.get("kind") == "final_logits"]
        assert len(final_logits_lines) == 1
        assert final_logits_lines[0]["member_index"] == 1


# --- Present truth: the real pinned reference pack, and (if available) the
# real calibrated artifact. ---


def test_the_real_pinned_prompt_pack_loads_and_matches_the_design_records_counts():
    members = rct.load_prompt_pack(rct.DEFAULT_PROMPTS_PATH)
    assert [m["index"] for m in members] == [1, 2, 3, 4, 5]
    expected_counts = {1: 20, 2: 460, 3: 467, 4: 484, 5: 571}
    for m in members:
        assert len(m["input_ids"]) == expected_counts[m["index"]], (
            f"member {m['index']} ({m['axis']}) has {len(m['input_ids'])} tokens, "
            f"the closed design record (D-SLM350) pins {expected_counts[m['index']]}"
        )


def test_the_producer_runs_end_to_end_against_the_real_calibrated_artifact():
    """The strongest available proof this build can offer: the producer,
    unmodified, run against genuine calibrated weights and the real, full
    five-member reference pack -- not a fixture model or a truncated pack.
    Skipped, not failed, where the real artifact is unavailable in this
    environment (a public CI runner never has it; this cell is this build's
    own end-to-end evidence, not a standing CI leg)."""
    artifact_dir = os.environ.get(
        "SUPERSLM_CALIBRATED_ARTIFACT", r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8"
    )
    if not os.path.isdir(artifact_dir):
        pytest.skip(f"no calibrated artifact at {artifact_dir!r} in this environment")

    wizard_source = os.environ.get("SUPERSLM_REFERENCE_SOURCE")
    if not wizard_source or not os.path.isdir(wizard_source):
        pytest.skip(f"SUPERSLM_REFERENCE_SOURCE not set to a real D:\\Wizard checkout")

    # artifact_cache.py (the loader) is NOT part of the vendored closure (it
    # is not in dynamic_engine.py's own import graph) -- it is loaded here,
    # directly from D:\Wizard, ONLY to construct a pickle for this one-time
    # proof run, exactly mirroring precompute_criterion2_prompt_pack.py's own
    # cross-repo, hand-run convention. The producer under test never imports
    # it; it only ever unpickles the QuantizedModel this test hands it.
    #
    # Loaded via importlib, by explicit file path, rather than
    # `from superslm_spike import artifact_cache`: this process has ALREADY
    # imported `superslm_spike` (this file's own top-level `from
    # superslm_spike import pipeline`), which permanently caches it in
    # sys.modules pointing at THIS repository's vendored copy (which has no
    # artifact_cache.py at all -- it is outside the closure by design). Once
    # cached, `from superslm_spike import X` never re-searches sys.path for
    # `superslm_spike` itself, so no amount of sys.path surgery makes that
    # form resolve to D:\Wizard's copy instead. Loading the file directly
    # sidesteps package resolution entirely; artifact_cache.py's own
    # `from superslm_spike import pipeline` (inside load_artifact) still
    # resolves through the normal, already-cached route -- to the VENDORED
    # pipeline module, which is byte-identical to D:\Wizard's at the pinned
    # commit (PROVENANCE.md), so the QuantizedModel it builds is
    # cross-compatible with dynamic_engine.forward_dynamic_vec either way.
    import importlib.util

    artifact_cache_path = os.path.join(wizard_source, "Tools", "superslm_spike", "artifact_cache.py")
    if not os.path.isfile(artifact_cache_path):
        pytest.skip(f"artifact_cache.py not found at {artifact_cache_path!r}")
    spec = importlib.util.spec_from_file_location("artifact_cache_realpath", artifact_cache_path)
    artifact_cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(artifact_cache)
    model = artifact_cache.load_artifact(artifact_dir)

    picklable = dataclasses.replace(model, float_source=None, tokenize_prompt=None)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "real_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(picklable, f)
        out_path = os.path.join(tmp, "real_out.jsonl")

        code = rct.main(["--model", model_path, "--prompts", rct.DEFAULT_PROMPTS_PATH, "--out", out_path])
        assert code == 0

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        final_logits_lines = [l for l in lines if l.get("kind") == "final_logits"]
        assert [l["member_index"] for l in final_logits_lines] == [1, 2, 3, 4, 5]
        site_lines = [l for l in lines if l.get("kind") != "final_logits"]
        # Five real members, dozens of layers each -- thousands of site
        # records is the expected order of magnitude, not a handful.
        assert len(site_lines) > 1000
