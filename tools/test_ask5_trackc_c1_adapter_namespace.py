"""T-2543 C-1: `sslm_convert_adapter.build_merged_checkpoint` merges a bare-convention
adapter correctly, rather than silently merging zero deltas.

Poirot 2a46a85-t2540-ask5-trackc-review.md C-1: T-2539 threaded `present` into this
module's own `_upstream_names` call, which let a bare-convention checkpoint reach the
LoRA merge loop for the first time -- but the adapter's own key set six lines below was
still built with a hardcoded `model.` prefix. On a bare-convention checkpoint, EVERY
lookup in `adapted_upstream.get(upstream)` misses (the checkpoint's real keys carry no
`model.` prefix), so `n_merged` is 0, the tool prints success, and the emitted "merged"
checkpoint is byte-equal to the base -- a silent wrong model. Pre-fold (`fdd4739`), the
identical bare-convention input raised `KeyError` on the map's own first tensor read,
because the pre-fold map was itself hardcoded `model.`-prefixed and rejected the
checkpoint entirely before ever reaching this function.

Base checkpoint shapes bind against `tools/_calibrate_checkpoint_fixture.py`'s own
constants (`HIDDEN_SIZE=8`, `NUM_ATTENTION_HEADS=2`, `HEAD_DIM=4` -> `q_proj` is `[8, 8]`),
matching `tools/_t2194_bf16_lora_fixture.py`'s own adapter fixture -- built to target
`q_proj` at that exact width, so the same adapter fixture binds against a base built by
either `build_fixture_checkpoint` (legacy) or `build_parameterized_fixture_checkpoint`
(any convention) with no shape changes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _calibrate_checkpoint_fixture import build_parameterized_fixture_checkpoint
from _t2194_bf16_lora_fixture import build_fp32_reference_fixture

import sslm_convert_adapter as A


def _build_bare_adapter_and_base(tmp_path):
    """A bare-convention (`prefix=""`) base checkpoint plus an F32 PEFT adapter targeting
    its own `layer0.q_proj`, bound together via `adapter_config.json`'s own
    `base_model_name_or_path`."""
    checkpoint_dir = build_parameterized_fixture_checkpoint(
        tmp_path / "base_checkpoint", prefix="", biased=False,
        tie_word_embeddings=True, qk_norm=False)
    adapter_dir = tmp_path / "adapter"
    build_fp32_reference_fixture(
        adapter_dir, base_model_name_or_path=str(checkpoint_dir), seed=0)
    return checkpoint_dir, adapter_dir


def test_merging_a_bare_convention_adapter_actually_merges_nonzero_deltas(tmp_path, capsys):
    """The Critical's own green twin: on a bare-convention base, the merged `q_proj`
    tensor at layer 0 differs from the base checkpoint's own raw `q_proj` -- a real
    `scaling*(B@A)` delta was added, not a silent zero-tensor merge. Before the fix, EVERY
    tensor's key lookup missed and the emitted checkpoint was byte-identical to the base.
    """
    checkpoint_dir, adapter_dir = _build_bare_adapter_and_base(tmp_path)
    out_dir = tmp_path / "merged"

    A.build_merged_checkpoint(adapter_dir, out_dir, verbose=True)
    printed = capsys.readouterr().out
    assert "0 adapted tensors merged" not in printed, (
        f"the merge loop matched zero adapter keys: {printed!r}"
    )

    from reference_pipeline import pipeline
    base_tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)
    merged_tensors = pipeline._open_checkpoint_tensors(out_dir)

    base_q = np.asarray(base_tensors.tensor("layers.0.self_attn.q_proj.weight"))
    merged_q = np.asarray(merged_tensors.tensor("layers.0.self_attn.q_proj.weight"))
    assert not np.array_equal(base_q, merged_q), (
        "layer0 q_proj is byte-identical between base and \"merged\" checkpoint -- the "
        "adapter's own delta was not applied"
    )

    # Every OTHER tensor (untouched by the adapter's own target_modules=["q_proj"]) is
    # still copied through unchanged -- proving the fix is scoped to the adapted
    # projection, not a wholesale rewrite.
    base_embed = np.asarray(base_tensors.tensor("embed_tokens.weight"))
    merged_embed = np.asarray(merged_tensors.tensor("embed_tokens.weight"))
    assert np.array_equal(base_embed, merged_embed), (
        "an untouched tensor (embed_tokens.weight) changed across the merge"
    )


def test_merging_a_legacy_convention_adapter_is_unregressed(tmp_path, capsys):
    """The identical merge, on the ORIGINAL `model.`-prefixed convention every existing
    incumbent (and this repo's own pre-existing e2e suite,
    `test_sslm_convert_adapter_b3_notice_e2e.py`) uses -- confirms the namespace-detection
    fix does not merely swap which convention silently fails."""
    from _calibrate_checkpoint_fixture import build_fixture_checkpoint

    checkpoint_dir = build_fixture_checkpoint(tmp_path / "legacy_base")
    adapter_dir = tmp_path / "legacy_adapter"
    build_fp32_reference_fixture(
        adapter_dir, base_model_name_or_path=str(checkpoint_dir), seed=0)
    out_dir = tmp_path / "legacy_merged"

    A.build_merged_checkpoint(adapter_dir, out_dir, verbose=True)
    printed = capsys.readouterr().out
    assert "0 adapted tensors merged" not in printed, printed

    from reference_pipeline import pipeline
    base_tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)
    merged_tensors = pipeline._open_checkpoint_tensors(out_dir)
    base_q = np.asarray(base_tensors.tensor("model.layers.0.self_attn.q_proj.weight"))
    merged_q = np.asarray(merged_tensors.tensor("model.layers.0.self_attn.q_proj.weight"))
    assert not np.array_equal(base_q, merged_q)


def test_read_base_projection_weight_honors_the_detected_namespace(tmp_path):
    """The sibling site (`:492`, `read_base_projection_weight`, reached from
    `build_runtime_additive_sections` at `:1420`) -- unit-level: reading `layer0.q_proj`
    from a bare-convention checkpoint's own opened tensor source, with the detected
    namespace threaded through, returns the real tensor rather than raising `KeyError` on
    a hardcoded `model.` key that does not exist on this convention."""
    from reference_pipeline import pipeline

    checkpoint_dir = build_parameterized_fixture_checkpoint(
        tmp_path / "bare_base", prefix="", biased=False,
        tie_word_embeddings=True, qk_norm=False)
    tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)
    ns = pipeline.detect_namespace(set(tensors.keys()))
    assert ns == "", "the fixture's own bare convention was not detected as bare"

    weight = A.read_base_projection_weight(tensors, 0, "q_proj", ns)
    direct = tensors.tensor("layers.0.self_attn.q_proj.weight")
    assert np.array_equal(np.asarray(weight), np.asarray(direct))

    with pytest.raises(KeyError):
        A.read_base_projection_weight(tensors, 0, "q_proj", "model.")
