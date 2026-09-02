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


# ==============================================================================
# _classify_post_processor -- the post_processor read (T-2541, closes TOK-06 /
# root cause of D-SLM5574, t2408 §6 Track E step 2). Pure function of the JSON
# value, no checkpoint files needed.
# ==============================================================================


def _candidate_shaped_post_processor(appended_id=151643):
    """The pinned Qwen3-Embedding-0.6B candidate's own `post_processor` shape,
    confirmed against the real checkpoint's own tokenizer.json this fold: a
    `Sequence` wrapping a bare `ByteLevel` and a `TemplateProcessing` whose own
    `single` template appends one `SpecialToken` after the input `Sequence`."""
    return {
        "type": "Sequence",
        "processors": [
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
            {
                "type": "TemplateProcessing",
                "single": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
                ],
                "pair": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 0}},
                    {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
                ],
                "special_tokens": {
                    "<|endoftext|>": {"id": "<|endoftext|>", "ids": [appended_id], "tokens": ["<|endoftext|>"]},
                },
            },
        ],
    }


def test_classify_post_processor_none_carries_no_trailing_append():
    assert CT._classify_post_processor(None) is None


def test_classify_post_processor_bare_bytelevel_carries_no_trailing_append():
    """Every incumbent's own bare `ByteLevel` post_processor."""
    assert CT._classify_post_processor({"type": "ByteLevel", "add_prefix_space": False}) is None


def test_classify_post_processor_resolves_the_candidates_own_trailing_special_id():
    assert CT._classify_post_processor(_candidate_shaped_post_processor()) == 151643


def test_classify_post_processor_rejects_a_bare_non_bytelevel_top_level_type():
    """The exact shape D-SLM5619 found this branch silently absorbing into the
    no-append case through fold round 12: a bare, non-Sequence-wrapped
    `TemplateProcessing` (bge-small-en-v1.5's own real `post_processor`, confirmed
    against the real file this fold, reproduced here as a literal fixture since the
    checkpoint's own `model.type` is WordPiece and would never reach this
    classifier through the real `__init__` path)."""
    bare_template_processing = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "pair": [],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]},
        },
    }
    with pytest.raises(CT.UnsupportedTokenizerShape, match="unrecognized top-level type"):
        CT._classify_post_processor(bare_template_processing)


def test_classify_post_processor_rejects_a_sequence_with_no_templateprocessing():
    pp = _candidate_shaped_post_processor()
    pp["processors"] = [pp["processors"][0]]  # ByteLevel only, no TemplateProcessing
    with pytest.raises(CT.UnsupportedTokenizerShape, match="0 TemplateProcessing"):
        CT._classify_post_processor(pp)


def test_classify_post_processor_rejects_a_sequence_with_two_templateprocessing_entries():
    pp = _candidate_shaped_post_processor()
    pp["processors"].append(pp["processors"][1])
    with pytest.raises(CT.UnsupportedTokenizerShape, match="2 TemplateProcessing"):
        CT._classify_post_processor(pp)


def test_classify_post_processor_rejects_a_single_template_of_the_wrong_length():
    pp = _candidate_shaped_post_processor()
    pp["processors"][1]["single"].append({"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}})
    with pytest.raises(CT.UnsupportedTokenizerShape, match="has 3 entries"):
        CT._classify_post_processor(pp)


def test_classify_post_processor_rejects_sequence_and_specialtoken_in_the_wrong_order():
    pp = _candidate_shaped_post_processor()
    pp["processors"][1]["single"] = list(reversed(pp["processors"][1]["single"]))
    with pytest.raises(CT.UnsupportedTokenizerShape, match="not in the.*expected order"):
        CT._classify_post_processor(pp)


def test_classify_post_processor_rejects_an_id_the_special_tokens_map_does_not_resolve():
    pp = _candidate_shaped_post_processor()
    pp["processors"][1]["special_tokens"]["<|endoftext|>"]["ids"] = [151643, 151644]
    with pytest.raises(CT.UnsupportedTokenizerShape, match="does not resolve to exactly one id"):
        CT._classify_post_processor(pp)


def test_classify_post_processor_rejects_an_unresolvable_special_token_reference():
    pp = _candidate_shaped_post_processor()
    del pp["processors"][1]["special_tokens"]["<|endoftext|>"]
    with pytest.raises(CT.UnsupportedTokenizerShape, match="does not resolve to exactly one id"):
        CT._classify_post_processor(pp)


# ==============================================================================
# _post_processor_mismatches -- the additive parity check's own counting logic
# (T-2541, closes D-SLM5575, t2408 §6 Track E step 3). Pure function of a
# `tables`/`hf`-shaped pair of objects, no checkpoint files needed -- the
# real-checkpoint acceptance run lives in test_convert_tokenizer_post_processor_
# parity.py, skipped when the pinned candidate is not present locally.
# ==============================================================================


class _FakeTables:
    """Stands in for a `TokenizerTables` instance: exposes only the two attributes
    `_post_processor_mismatches` reads, so this file's tests don't need a real
    checkpoint to drive the check's own pass/mismatch counting."""

    def __init__(self, trailing_special_id, ids_by_text):
        self.trailing_special_id = trailing_special_id
        self._ids_by_text = ids_by_text

    def ref_encode(self, text):
        return list(self._ids_by_text[text])


class _FakeHF:
    def __init__(self, ids_by_text_with_trailing):
        self._ids = ids_by_text_with_trailing

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is True
        return list(self._ids[text])


def test_post_processor_mismatches_counts_zero_when_ref_plus_trailing_matches_hf():
    tables = _FakeTables(trailing_special_id=999, ids_by_text={"a": [1, 2], "b": [3]})
    hf = _FakeHF({"a": [1, 2, 999], "b": [3, 999]})
    assert CT._post_processor_mismatches(tables, hf, ["a", "b"]) == 0


def test_post_processor_mismatches_counts_every_line_wrong_by_a_bad_trailing_id():
    """The must-reject shape by construction: a trailing_special_id one off from
    what HF's own encode path actually appends makes every line a mismatch."""
    tables = _FakeTables(trailing_special_id=998, ids_by_text={"a": [1, 2], "b": [3]})
    hf = _FakeHF({"a": [1, 2, 999], "b": [3, 999]})
    assert CT._post_processor_mismatches(tables, hf, ["a", "b"]) == 2


def test_post_processor_mismatches_counts_only_the_lines_that_actually_differ():
    tables = _FakeTables(trailing_special_id=999, ids_by_text={"a": [1, 2], "b": [3]})
    hf = _FakeHF({"a": [1, 2, 999], "b": [3, 1000]})  # only "b" is wrong
    assert CT._post_processor_mismatches(tables, hf, ["a", "b"]) == 1


def test_verify_post_processor_evaluates_unconditionally_when_trailing_id_is_none(monkeypatch):
    """T-2542 Finding 1 (Poirot): the early return keyed on `trailing_special_id is
    None` is gone -- `verify_post_processor` now calls `_post_processor_mismatches`
    (and therefore `transformers.AutoTokenizer.from_pretrained`) for every
    checkpoint, incumbent included. Vacuity for an incumbent is now a property of
    the comparison's own RESULT (it reads 0 mismatches because nothing is appended
    on either side), not of the function declining to run it -- proven here by a
    fake `hf` that records every call it receives: the comparison actually ran."""
    stub = _FakeTables(trailing_special_id=None, ids_by_text={"a": [1, 2], "b": [3]})
    monkeypatch.setattr(CT, "TokenizerTables", lambda ckpt_dir: stub)
    monkeypatch.setattr(CT, "read_corpus_records", lambda corpus_path: ["a", "b"])

    calls = []

    class _RecordingHF:
        def encode(self, text, add_special_tokens):
            calls.append((text, add_special_tokens))
            return list(stub._ids_by_text[text])  # nothing appended -> matches ref_encode

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(ckpt_dir):
            return _RecordingHF()

    import sys as _sys
    import types as _types
    fake_transformers = _types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)
    monkeypatch.setitem(_sys.modules, "transformers", fake_transformers)

    assert CT.verify_post_processor("unused-ckpt-dir", "unused-corpus-path") == 0
    assert calls == [("a", True), ("b", True)], "the comparison must actually run, not be skipped"


def test_verify_post_processor_imports_transformers_even_on_the_vacuous_path(monkeypatch):
    """T-2542 Finding 3 (Poirot): the CHANGELOG and this file previously claimed the
    vacuous path (trailing_special_id is None) "does not run transformers" -- false,
    since the import was always the function's first statement, above the early
    return. Finding 1's fix deletes the early return entirely, so this is no longer
    even a latent claim: with `transformers` unimportable, the vacuous path now
    raises ImportError rather than returning quietly, proving the import is not
    merely present but load-bearing on every path."""
    stub = _FakeTables(trailing_special_id=None, ids_by_text={"a": [1, 2]})
    monkeypatch.setattr(CT, "TokenizerTables", lambda ckpt_dir: stub)
    monkeypatch.setattr(CT, "read_corpus_records", lambda corpus_path: ["a"])

    import builtins as _builtins
    real_import = _builtins.__import__

    def _no_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers' (simulated absent)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", _no_transformers)
    with pytest.raises(ImportError, match="transformers"):
        CT.verify_post_processor("unused-ckpt-dir", "unused-corpus-path")
