"""Engine 1.19: `sslm_convert_adapter --fallback=merge` applies the tied-head rule.

`build_merged_checkpoint` reads the base checkpoint named by `adapter_config.json` directly,
so it must give the same answer `pipeline.load_model` gives on that checkpoint: on a config
with `tie_word_embeddings: true`, an `lm_head.weight` that is a stored copy of the embedding
(same dtype, shape and bytes) is accepted and dropped, and any other `lm_head.weight` is
refused with `TiedLmHeadMismatch`.

Before the fix the merge path never mapped a tied `lm_head.weight`, so a differing head was
silently replaced by the embedding (code review 1.19, finding 2).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import _calibrate_checkpoint_fixture as fixture_mod
from _t2194_bf16_lora_fixture import build_fp32_reference_fixture

import sslm_convert_adapter as A

EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"


def _tied_base(checkpoint_dir, *, lm_head=None):
    """A tied, `model.`-prefixed base checkpoint the LoRA fixture binds to. Element [0, 0] of
    the embedding is +0.0 so the signed-zero case has a zero to negate. `lm_head` is
    `None` (absent), or a function from the embedding array to the head array."""
    checkpoint_dir = fixture_mod.build_parameterized_fixture_checkpoint(
        checkpoint_dir, prefix="model.", biased=False, tie_word_embeddings=True,
        lm_head_present=False, qk_norm=False)
    tensors = fixture_mod._parameterized_tensors(
        prefix="model.", biased=False, lm_head_present=False, qk_norm=False, seed=0)
    embed = np.array(tensors[EMBED], dtype=np.float32)
    embed[0, 0] = 0.0
    tensors[EMBED] = embed
    if lm_head is not None:
        tensors[LM_HEAD] = lm_head(embed.copy())
    fixture_mod._write_safetensors_shard(checkpoint_dir, tensors)
    return checkpoint_dir


def _adapter_for(tmp_path, checkpoint_dir, name):
    return build_fp32_reference_fixture(
        tmp_path / name, base_model_name_or_path=str(checkpoint_dir), seed=0)


def _last_byte_flipped(embed):
    raw = bytearray(embed.tobytes())
    raw[-1] ^= 0x01
    return np.frombuffer(bytes(raw), dtype=np.float32).reshape(embed.shape).copy()


def _signed_zero(embed):
    embed[0, 0] = -0.0
    return embed


_DIFFERING = {"last-byte-flipped": _last_byte_flipped, "signed-zero": _signed_zero}


def test_merge_of_a_tied_base_with_a_byte_identical_head_equals_the_head_free_merge(tmp_path):
    """The identical duplicate is accepted, and the merged checkpoint is byte-equal to the
    one merged from the same base without `lm_head.weight`."""
    absent = _tied_base(tmp_path / "absent")
    duplicate = _tied_base(tmp_path / "duplicate", lm_head=lambda embed: embed)
    out_absent = A.build_merged_checkpoint(
        _adapter_for(tmp_path, absent, "adapter-absent"), tmp_path / "merged-absent", verbose=False)
    out_duplicate = A.build_merged_checkpoint(
        _adapter_for(tmp_path, duplicate, "adapter-duplicate"), tmp_path / "merged-duplicate",
        verbose=False)
    merged = (out_duplicate / "model.safetensors").read_bytes()
    assert merged == (out_absent / "model.safetensors").read_bytes()
    from reference_pipeline import pipeline
    assert LM_HEAD not in pipeline._open_checkpoint_tensors(out_duplicate).keys()


@pytest.mark.parametrize("case", sorted(_DIFFERING))
def test_merge_of_a_tied_base_with_a_differing_head_is_refused_by_the_named_error(case, tmp_path):
    """A differing tied head is refused by the converter's own `TiedLmHeadMismatch`, naming
    both tensors and the differing bytes, and no merged checkpoint is written."""
    pipeline = A._load_spike()
    base = _tied_base(tmp_path / "base", lm_head=_DIFFERING[case])
    adapter = _adapter_for(tmp_path, base, "adapter")
    out_dir = tmp_path / "merged"
    with pytest.raises(pipeline.TiedLmHeadMismatch) as caught:
        A.build_merged_checkpoint(adapter, out_dir, verbose=False)
    message = str(caught.value)
    assert LM_HEAD in message and EMBED in message, message
    assert "stored bytes differ" in message, message
    assert not (out_dir / "model.safetensors").exists()


@pytest.mark.parametrize("case", sorted(_DIFFERING))
def test_fallback_merge_conversion_refuses_a_differing_tied_head_before_writing(case, tmp_path):
    """The `--fallback=merge` conversion entry point refuses the same checkpoint
    `pipeline.load_model` refuses, and writes no artifact."""
    pipeline = A._load_spike()
    base = _tied_base(tmp_path / "base", lm_head=_DIFFERING[case])
    adapter = _adapter_for(tmp_path, base, "adapter")
    out_path = tmp_path / "merged.sslm"
    with pytest.raises(pipeline.TiedLmHeadMismatch):
        A.run_merge_quantize_conversion(adapter, out_path, skip_verify=True, verbose=False)
    with pytest.raises(pipeline.TiedLmHeadMismatch):
        pipeline.load_model(base)
    assert not out_path.exists()
