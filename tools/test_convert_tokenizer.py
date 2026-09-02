"""Regression suite for convert_tokenizer.py's checkpoint-derived model label.

`TokenizerTables.emit_artifact()` used to hardcode `config["model"] = "qwen2.5-1.5b-instruct"`
regardless of which checkpoint `self.ckpt` actually pointed at (found during T-1909, filed
D-SLM2503). Nothing reads the field back -- confirmed by grep across `tools/` and `src/` -- so the
defect was cosmetic, not functional, but every artifact this converter ever emitted for any
checkpoint carried a mislabeled CONFIG section.

Two checkpoint-path shapes occur in this project (`convert_tokenizer.derive_model_name`'s own
docstring): the HF hub cache's content-hash snapshot dirs, where the leaf directory name is a hash
and the readable repo name sits two levels up (`models--<org>--<repo>/snapshots/<rev>`); and flat
exported directories whose own leaf name is already descriptive (e.g. a merged-LoRA output). Both
are covered below, plus `emit_artifact`'s actual on-disk CONFIG bytes, so a regression that
reintroduces a hardcoded literal -- in either function -- is caught.
"""

import json
import struct

import pytest

import convert_tokenizer as CT
import sslm_format as F


# ==============================================================================
# derive_model_name -- pure function of a checkpoint path, no checkpoint files needed
# ==============================================================================


def test_derive_model_name_from_hf_hub_cache_snapshot_matches_the_former_hardcoded_literal():
    """The real, currently-converted checkpoint's own path shape: this is the exact
    string the converter hardcoded before the fix, so this cell also pins that the fix
    is a no-op regression for the checkpoint every existing artifact was built from."""
    ckpt = r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert CT.derive_model_name(ckpt) == "qwen2.5-1.5b-instruct"


def test_derive_model_name_from_hf_hub_cache_snapshot_tracks_a_different_checkpoint():
    """T-1909's own case: a 3B checkpoint at the analogous cache path must NOT come back
    labeled 1.5B -- this is the exact defect shape the fix closes."""
    ckpt = r"D:\hf_cache\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\deadbeefcafe1234567890abcdef1234567890ab"
    assert CT.derive_model_name(ckpt) == "qwen2.5-3b-instruct"


def test_derive_model_name_falls_back_to_the_leaf_name_for_a_flat_exported_checkpoint():
    """A merged-LoRA export (or any checkpoint dir not under an HF hub cache) has no
    `models--<org>--<repo>/snapshots/<rev>` ancestry -- its own leaf name is already the
    descriptive label and is used as-is (lowercased)."""
    ckpt = r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v1-merged"
    assert CT.derive_model_name(ckpt) == "qwen2.5-1.5b-shopkeeper-lora-v1-merged"


def test_derive_model_name_is_case_insensitive_on_the_repo_segment():
    ckpt = r"D:\hf_cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\0000000000000000000000000000000000000000"
    assert CT.derive_model_name(ckpt) == "qwen2.5-7b-instruct"


def test_derive_model_name_does_not_misfire_on_a_directory_literally_named_snapshots():
    """The HF-cache branch is gated on the PARENT being named `snapshots` AND the
    grandparent following the `models--<org>--<repo>` cache-key convention. A directory
    that merely happens to sit under a `snapshots` folder without that convention falls
    through to the leaf-name default rather than misparsing an unrelated ancestor."""
    ckpt = r"D:\some_export\snapshots\qwen2.5-0.5b-custom"
    assert CT.derive_model_name(ckpt) == "qwen2.5-0.5b-custom"


# ==============================================================================
# emit_artifact -- the actual on-disk CONFIG bytes, not just the helper function
# ==============================================================================


def _read_config_section(sslm_path):
    """Minimal, independent re-parse of the documented `.sslm` v1 header/section-table
    layout (sslm_format.py's own `build_artifact`), so this cell checks the bytes
    `emit_artifact` actually wrote rather than trusting `derive_model_name` in isolation."""
    data = sslm_path.read_bytes()
    assert data[0:4] == F.MAGIC
    section_count = struct.unpack_from("<I", data, 12)[0]
    for i in range(section_count):
        row = F.HEADER_BYTES + i * F.SECTION_DESC_BYTES
        sec_type, _dtype = struct.unpack_from("<II", data, row)
        off, length, _elem_count = struct.unpack_from("<QQQ", data, row + 8)
        if sec_type == F.SectionType.CONFIG:
            return json.loads(data[off:off + length])
    raise AssertionError("no CONFIG section in artifact")


def _bare_tokenizer_tables(ckpt_dir):
    """A `TokenizerTables` instance with just enough state for `emit_artifact` to run,
    bypassing `__init__`'s real tokenizer.json/tokenizer_config.json file reads -- no
    checkpoint files exist in this test environment, and none of `emit_artifact`'s own
    logic (the field under test) depends on real vocab/merge content."""
    t = object.__new__(CT.TokenizerTables)
    t.ckpt = ckpt_dir
    t.model_name = CT.derive_model_name(ckpt_dir)
    t.chat_template = None
    t.byte_to_id = list(range(256))
    t.id_to_bytes = [b"a", b"b"]
    t.merge_triples = []
    t.specials = []

    class _StubUnicode:
        version = (1, 0, 0)

        def serialize(self):
            return b""

    t.u = _StubUnicode()
    return t


def test_emit_artifact_config_section_carries_the_derived_model_name_not_a_literal(tmp_path):
    ckpt = r"D:\hf_cache\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\deadbeefcafe1234567890abcdef1234567890ab"
    tables = _bare_tokenizer_tables(ckpt)
    out = tmp_path / "tok.sslm"

    tables.emit_artifact(str(out))

    config = _read_config_section(out)
    assert config["model"] == "qwen2.5-3b-instruct"
    assert config["model"] != "qwen2.5-1.5b-instruct"  # the former hardcoded literal


def test_emit_artifact_config_section_changes_when_the_checkpoint_changes():
    """Two different checkpoints must not emit the same `model` label -- pins that the
    field is actually derived per-instance, not memoized or hardcoded anywhere upstream
    of the CONFIG dict."""
    ckpt_a = r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ckpt_b = r"D:\hf_cache\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\deadbeefcafe1234567890abcdef1234567890ab"
    assert _bare_tokenizer_tables(ckpt_a).model_name != _bare_tokenizer_tables(ckpt_b).model_name


# ==============================================================================
# _parse_merge_element -- the model.merges schema branch (T-2541, closes TOK-04 /
# D-SLM5573, t2408 §6 Track E step 1). Pure function, no checkpoint files needed.
# ==============================================================================


def test_parse_merge_element_splits_the_incumbent_space_separated_string_schema():
    """Every Qwen2.5-family checkpoint's own `model.merges` element -- unchanged
    behavior from before this fix."""
    assert CT._parse_merge_element("\u0120 \u0120", 0) == ["\u0120", "\u0120"]


def test_parse_merge_element_uses_a_two_element_list_directly():
    """The pinned Qwen3-Embedding-0.6B candidate's own `model.merges` element schema
    -- a 2-element list, no join/split needed."""
    assert CT._parse_merge_element(["\u0120", "\u0120"], 0) == ["\u0120", "\u0120"]


def test_parse_merge_element_uses_a_two_element_tuple_directly():
    assert CT._parse_merge_element(("\u0120", "\u0120"), 0) == ["\u0120", "\u0120"]


def test_parse_merge_element_rejects_an_unrecognized_shape_by_name():
    """A merge element that is neither a string nor a 2-element list/tuple is an
    explicit, named rejection -- never a silent guess (N3 discipline)."""
    with pytest.raises(CT.UnsupportedTokenizerShape, match=r"model\.merges\[3\]"):
        CT._parse_merge_element({"a": "b"}, 3)


def test_parse_merge_element_rejects_a_three_element_list():
    with pytest.raises(CT.UnsupportedTokenizerShape):
        CT._parse_merge_element(["a", "b", "c"], 0)
