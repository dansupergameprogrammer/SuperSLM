"""Calibrated-artifact serialization: round-trip bit-identity + the staleness law.

Charter: D-SLM59 stage 4's precondition — the 3.3-hour in-process calibration paid once;
a cached artifact then loads in minutes per process. Pins in design record A-8 §17.3.

Red suite for Packet E12 (Claude/Curie/packets/eval/packet-12.md).

Test design: Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §17.3.

T-408 (2026-07-23) adds the checkpoint-path retention red suite (design record
Claude/Vitruvius/SuperSLM_T408_ProofManifestMetrics_Design-2026-07-23.md §4/§8 steps
2/3/6): `save_artifact` persists the originating checkpoint path; `load_artifact`
reconstructs a live, lazy `float_source` from it, falling back to today's raising stub
when the path is absent, missing from an old-schema `metadata.json`, or no longer
resolves on disk. A synthetic on-disk checkpoint (one tiny `.safetensors` file, no
`config.json` needed — the reconstruction path never calls `load_config`) stands in
for the real 1.5B-parameter checkpoint, per this design's own step-2 red cell text.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import api, require


MODULE = "reference_pipeline.artifact_cache"
PIPELINE = "reference_pipeline.pipeline"


def fixture_config(pipeline):
    """The §11 fixture model's shape."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def test_a_loaded_artifact_produces_bit_identical_forwards_on_both_arms(tmp_path):
    """Round-trip: save the fixture model, load it back, and assert bit-identity.

    Both arms (forward, forward_dynamic) must produce identical output on both pinned
    sequences [0,1,3,5] and [7,2,30,11]. Additionally, loaded.scales and
    loaded.dynamic_biases (when present) must equal the original.

    §15 format lock's premise: a cached artifact is faithfully reconstructed when loaded.
    """
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    forward, forward_dynamic, fixture_model = api(
        PIPELINE, "forward", "forward_dynamic", "fixture_model"
    )
    pipeline = require(PIPELINE)
    cfg = fixture_config(pipeline)
    model = fixture_model(cfg)

    # Save the artifact to tmp_path.
    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path)

    # Load it back.
    loaded = load_artifact(artifact_path)

    # Test round-trip on both arms, both sequences.
    for seq in [[0, 1, 3, 5], [7, 2, 30, 11]]:
        # Static arm.
        original_logits = forward(model, seq)
        loaded_logits = forward(loaded, seq)
        assert np.array_equal(original_logits, loaded_logits)

        # Dynamic arm.
        original_dynamic_logits = forward_dynamic(model, seq)
        loaded_dynamic_logits = forward_dynamic(loaded, seq)
        assert np.array_equal(original_dynamic_logits, loaded_dynamic_logits)

    # Assert scales match.
    assert loaded.scales == model.scales

    # The BIASED half, behind its own gate (Curie A-9: the delivered `if hasattr` on
    # the plain fixture was dead code — the plain fixture never carries biases, so the
    # biased round-trip was never exercised). Reds independently on E10's fixture.
    fixture_model_biased = api(PIPELINE, "fixture_model_biased")
    model_b = fixture_model_biased(cfg)
    path_b = tmp_path / "artifact_biased"
    save_artifact(model_b, path_b)
    loaded_b = load_artifact(path_b)
    assert loaded_b.dynamic_biases == model_b.dynamic_biases
    for seq in [[0, 1, 3, 5], [7, 2, 30, 11]]:
        assert np.array_equal(forward_dynamic(model_b, seq), forward_dynamic(loaded_b, seq))


def test_the_manifest_carries_the_format_version_and_source_fingerprint(tmp_path):
    """The manifest (read as JSON) carries format_version and a 64-hex source_fingerprint.

    Determinism: saving the SAME model twice yields the SAME fingerprint.
    Decisiveness: a model saved from a different config yields a DIFFERENT fingerprint.

    Both properties are tested in-cell (not assumed) per the oracle discipline.
    """
    save_artifact, load_artifact, ARTIFACT_FORMAT_VERSION = api(
        MODULE, "save_artifact", "load_artifact", "ARTIFACT_FORMAT_VERSION"
    )
    fixture_model = api(PIPELINE, "fixture_model")
    pipeline = require(PIPELINE)
    cfg = fixture_config(pipeline)
    model = fixture_model(cfg)

    # Save once.
    artifact_path_1 = tmp_path / "artifact_1"
    save_artifact(model, artifact_path_1)

    # Read manifest.
    manifest_path_1 = artifact_path_1 / "manifest.json"
    assert manifest_path_1.exists()
    manifest_1 = json.loads(manifest_path_1.read_text())

    # Assert format_version and fingerprint keys exist and have expected types.
    assert "format_version" in manifest_1
    assert manifest_1["format_version"] == ARTIFACT_FORMAT_VERSION
    assert "source_fingerprint" in manifest_1
    assert isinstance(manifest_1["source_fingerprint"], str)
    assert len(manifest_1["source_fingerprint"]) == 64
    # Hex check: all chars in [0-9a-f].
    assert all(c in "0123456789abcdef" for c in manifest_1["source_fingerprint"].lower())

    # Determinism: save the same model again, verify fingerprint matches.
    artifact_path_2 = tmp_path / "artifact_2"
    save_artifact(model, artifact_path_2)
    manifest_path_2 = artifact_path_2 / "manifest.json"
    manifest_2 = json.loads(manifest_path_2.read_text())
    assert manifest_2["source_fingerprint"] == manifest_1["source_fingerprint"]

    # Decisiveness: save a model from a different config, verify fingerprint differs.
    artifact_path_3 = tmp_path / "artifact_3"
    cfg_different = pipeline.ModelConfig(
        hidden_size=64,  # Different!
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )
    model_different = fixture_model(cfg_different)
    save_artifact(model_different, artifact_path_3)
    manifest_path_3 = artifact_path_3 / "manifest.json"
    manifest_3 = json.loads(manifest_path_3.read_text())
    assert manifest_3["source_fingerprint"] != manifest_1["source_fingerprint"]


def test_a_format_version_mismatch_is_a_documented_rejection(tmp_path):
    """Staleness, version axis: a format_version mismatch raises a documented rejection.

    Never a silent load. The error message must name the version mismatch.
    """
    save_artifact, load_artifact, ARTIFACT_FORMAT_VERSION = api(
        MODULE, "save_artifact", "load_artifact", "ARTIFACT_FORMAT_VERSION"
    )
    fixture_model = api(PIPELINE, "fixture_model")
    pipeline = require(PIPELINE)
    cfg = fixture_config(pipeline)
    model = fixture_model(cfg)

    # Save artifact.
    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path)

    # Hand-edit manifest to a different format_version.
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    wrong_version = ARTIFACT_FORMAT_VERSION + 1
    manifest["format_version"] = wrong_version
    manifest_path.write_text(json.dumps(manifest))

    # Load should raise a documented error.
    with pytest.raises((ValueError, Exception)) as exc_info:
        load_artifact(artifact_path)

    # Assert the error message mentions the version mismatch.
    error_msg = str(exc_info.value).lower()
    assert "version" in error_msg or "format" in error_msg


def test_a_fingerprint_mismatch_is_a_documented_rejection(tmp_path):
    """Staleness, fingerprint axis: expect_fingerprint mismatch raises a documented rejection.

    Happy path: expect_fingerprint matching the manifest's own value loads.
    Sad path: a mismatched expect_fingerprint raises a documented error naming the mismatch.
    """
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    fixture_model = api(PIPELINE, "fixture_model")
    pipeline = require(PIPELINE)
    cfg = fixture_config(pipeline)
    model = fixture_model(cfg)

    # Save artifact.
    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path)

    # Read the correct fingerprint from manifest.
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    correct_fingerprint = manifest["source_fingerprint"]

    # Happy path: load with matching fingerprint.
    loaded = load_artifact(artifact_path, expect_fingerprint=correct_fingerprint)
    assert loaded is not None

    # Sad path: load with mismatched fingerprint.
    wrong_fingerprint = "0" * 64
    with pytest.raises((ValueError, Exception)) as exc_info:
        load_artifact(artifact_path, expect_fingerprint=wrong_fingerprint)

    # Assert the error message mentions the fingerprint mismatch.
    error_msg = str(exc_info.value).lower()
    assert "fingerprint" in error_msg


# ==============================================================================
# T-408 §4/§8 steps 2/3/6 — checkpoint-path retention and float-source reopen
# ==============================================================================


def _tiny_checkpoint_cfg(pipeline):
    """A config small enough that `fixture_model` calibrates in well under a second."""
    return pipeline.ModelConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, intermediate_size=8, vocab_size=4, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=4,
    )


def _write_synthetic_safetensors(path, tensors):
    """The minimal on-disk shape `pipeline._SafeTensors` reads: an 8-byte little-endian
    header length, the JSON header (name -> {dtype, shape, data_offsets}), then the raw
    bytes back to back. F32 (one of `_SafeTensors._DIRECT`) so no BF16 reinterpretation
    is in play — this fixture is about the checkpoint-path plumbing, not the widening."""
    header = {}
    payload = bytearray()
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    header["__metadata__"] = {}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(bytes(payload))


def _build_synthetic_checkpoint(tmp_path, pipeline, cfg, *, embed_values=None):
    """A tiny on-disk checkpoint directory holding one `.safetensors` file with just the
    `embed` tensor at its production upstream name -- `_CheckpointFloatSource` resolves
    the tensor it is actually asked for lazily, so nothing else `_upstream_names` maps
    needs to be present on disk for this fixture's cells."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    names = pipeline._upstream_names(cfg)          # upstream -> ours
    embed_upstream = next(u for u, o in names.items() if o == "embed")
    if embed_values is None:
        embed_values = np.arange(cfg.vocab_size * cfg.hidden_size, dtype=np.float32).reshape(
            cfg.vocab_size, cfg.hidden_size)
    _write_synthetic_safetensors(checkpoint_dir / "model.safetensors", {embed_upstream: embed_values})
    return checkpoint_dir, embed_values


def _model_pointed_at_checkpoint(pipeline, cfg, checkpoint_dir):
    """`fixture_model(cfg)` (cheap, deterministic) with its dict-backed `float_source`
    swapped for a real `_CheckpointFloatSource` reading the synthetic checkpoint --
    exactly the shape a real `load_model`-built model carries, at fixture cost."""
    import dataclasses

    model = pipeline.fixture_model(cfg)
    names = pipeline._upstream_names(cfg)
    tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)
    float_source = pipeline._CheckpointFloatSource(tensors, names, cfg)
    return dataclasses.replace(model, float_source=float_source)


def test_save_artifact_persists_the_checkpoint_path_in_metadata(tmp_path):
    """T-408 §4 step 2: `save_artifact(model, path, checkpoint_path)` writes the path into
    `metadata.json` verbatim; a model with no checkpoint (fixture-built) writes `null`, not
    a placeholder string -- absence is truthful."""
    save_artifact = api(MODULE, "save_artifact")
    pipeline = require(PIPELINE)
    cfg = _tiny_checkpoint_cfg(pipeline)
    checkpoint_dir, _ = _build_synthetic_checkpoint(tmp_path, pipeline, cfg)
    model = _model_pointed_at_checkpoint(pipeline, cfg, checkpoint_dir)

    artifact_path = tmp_path / "artifact_with_checkpoint"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    assert metadata.get("checkpoint_path") == str(checkpoint_dir)

    fixture_only = pipeline.fixture_model(cfg)
    artifact_path_2 = tmp_path / "artifact_without_checkpoint"
    save_artifact(fixture_only, artifact_path_2)
    metadata_2 = json.loads((artifact_path_2 / "metadata.json").read_text())
    assert metadata_2.get("checkpoint_path") is None


def test_load_artifact_reopens_a_live_float_source_from_the_persisted_checkpoint_path(tmp_path):
    """T-408 §4 step 2/3, happy path: a present, resolvable checkpoint path round-trips to
    a `float_weight` that succeeds and reads back the exact planted tensor -- the reload
    is a real lazy reader over the checkpoint, not the raising stub."""
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    pipeline = require(PIPELINE)
    cfg = _tiny_checkpoint_cfg(pipeline)
    checkpoint_dir, embed_values = _build_synthetic_checkpoint(tmp_path, pipeline, cfg)
    model = _model_pointed_at_checkpoint(pipeline, cfg, checkpoint_dir)

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))
    loaded = load_artifact(artifact_path)

    reopened = loaded.float_weight("embed")
    assert np.array_equal(np.asarray(reopened, dtype=np.float64),
                          embed_values.astype(np.float64))


def test_load_artifact_stub_raises_when_no_checkpoint_path_was_ever_recorded(tmp_path):
    """T-408 §7 fork 1: a `fixture_model()`-built artifact (no checkpoint, `null` in
    `metadata.json`) reloads without error, but calling `float_weight` on it raises --
    the expected, narrow stub contract, reached by calling `float_weight` directly, not
    through `verify_and_merge` (that is the separate manifest-merge failure §6 item 2
    gates, tested on the SuperSLM side)."""
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    pipeline = require(PIPELINE)
    cfg = _tiny_checkpoint_cfg(pipeline)
    model = pipeline.fixture_model(cfg)

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path)          # no checkpoint_path argument at all
    loaded = load_artifact(artifact_path)         # must not raise on load

    with pytest.raises(RuntimeError):
        loaded.float_weight("embed")


def test_load_artifact_stub_raises_identically_for_an_old_schema_missing_key(tmp_path):
    """T-408 §4/§7, dim-9 gap: a `metadata.json` with the `checkpoint_path` key absent
    entirely (constructed directly, simulating a cache directory written before this
    change) takes the identical stub-fallback branch as an explicit `null` -- proving
    `metadata.get("checkpoint_path")` collapses "missing key" and "explicit null" into
    one code path rather than a silent second behavior."""
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    pipeline = require(PIPELINE)
    cfg = _tiny_checkpoint_cfg(pipeline)
    model = pipeline.fixture_model(cfg)

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path)
    metadata_path = artifact_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert "checkpoint_path" in metadata          # today's save_artifact always writes it
    del metadata["checkpoint_path"]                # simulate a pre-T-408 cache directory
    metadata_path.write_text(json.dumps(metadata))

    loaded = load_artifact(artifact_path)          # must not raise on load
    with pytest.raises(RuntimeError):
        loaded.float_weight("embed")


def test_load_artifact_stub_raises_when_the_checkpoint_path_no_longer_resolves(tmp_path):
    """T-408 §7 fork 2: a checkpoint path that WAS present and resolvable at save time but
    has since moved or been deleted falls back to the stub on load (no crash), and raises
    only when `float_weight` is actually called -- the new failure mode §4 explicitly
    accepts, distinct from "no checkpoint was ever recorded" (a `try`/`except` around the
    reopen, not a pre-check that could itself race the filesystem)."""
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    pipeline = require(PIPELINE)
    cfg = _tiny_checkpoint_cfg(pipeline)
    checkpoint_dir, _ = _build_synthetic_checkpoint(tmp_path, pipeline, cfg)
    model = _model_pointed_at_checkpoint(pipeline, cfg, checkpoint_dir)

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))
    shutil.rmtree(checkpoint_dir)                  # the checkpoint is gone by load time

    loaded = load_artifact(artifact_path)           # must not raise on load
    with pytest.raises(RuntimeError):
        loaded.float_weight("embed")


def test_reopened_float_source_reads_are_warm_object_bit_identical(tmp_path):
    """T-408 §7/§8 step 6, this project's `TestSil1WarmObjectRepeatedReadsShowNoDrift`
    precedent applied to the reopened float source: two reads of the same tensor within
    one `load_artifact` call, and a second independent `load_artifact` call on the same
    directory within the same process, all return bit-identical values -- no drift from
    the memory-mapped file, no stale handle after a first read."""
    save_artifact, load_artifact = api(MODULE, "save_artifact", "load_artifact")
    pipeline = require(PIPELINE)
    cfg = _tiny_checkpoint_cfg(pipeline)
    checkpoint_dir, embed_values = _build_synthetic_checkpoint(tmp_path, pipeline, cfg)
    model = _model_pointed_at_checkpoint(pipeline, cfg, checkpoint_dir)

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))

    loaded = load_artifact(artifact_path)
    first_read = np.asarray(loaded.float_weight("embed"), dtype=np.float64).copy()
    second_read = np.asarray(loaded.float_weight("embed"), dtype=np.float64).copy()
    assert np.array_equal(first_read, second_read)

    reloaded = load_artifact(artifact_path)
    third_read = np.asarray(reloaded.float_weight("embed"), dtype=np.float64)
    assert np.array_equal(first_read, third_read)
    assert np.array_equal(first_read, embed_values.astype(np.float64))
