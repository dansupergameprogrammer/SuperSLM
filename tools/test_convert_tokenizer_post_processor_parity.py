"""Real-checkpoint acceptance tests for the additive post-processor-parity check
(T-2541 Track E step 3, closes D-SLM5575, `t2408` §6 Track E). Requires the pinned
Qwen3-Embedding-0.6B candidate and a Qwen2.5-family incumbent in the local HF hub
cache -- skipped, not failed, when a checkpoint is not present locally, since
fetching one is a separate step from running the Python suite (mirrors
`test_sslm_convert_loader_join.py`'s own skip-if-absent convention for its compiled
binary dependency).

THIS CHECK IS ITSELF A DECIDING INSTRUMENT AND IS NOT COMMISSIONED BY THIS BUILD.
`convert_tokenizer.verify_post_processor`'s own docstring, and `t2408` §6 Track E /
§9, state a must-accept and a must-reject construction are owed to a seat
independent of this build, blind to its own controls, before the check's readings
are load-bearing. The tests below confirm the check RUNS and reads the numbers this
ticket's own build log quotes; they are authored by the same seat that built the
check and do NOT satisfy that independent commissioning.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import convert_tokenizer as CT  # noqa: E402

_HF_HOME = os.environ.get("HF_HOME", r"D:\hf_cache")
_CANDIDATE = os.path.join(
    _HF_HOME, "hub", "models--Qwen--Qwen3-Embedding-0.6B", "snapshots",
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
)
_INCUMBENT = os.path.join(
    _HF_HOME, "hub", "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots",
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)
_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "tokenizer_golden_corpus.jsonl",
)

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(_CANDIDATE) and os.path.isdir(_INCUMBENT) and os.path.isfile(_CORPUS)),
    reason="pinned Qwen3-Embedding-0.6B candidate and/or Qwen2.5-1.5B-Instruct "
           "incumbent not present in the local HF hub cache",
)


def test_new_check_passes_on_the_real_candidates_own_corpus():
    from transformers import AutoTokenizer
    tables = CT.TokenizerTables(_CANDIDATE)
    assert tables.trailing_special_id == 151643
    hf = AutoTokenizer.from_pretrained(_CANDIDATE)
    lines = CT.read_corpus_records(_CORPUS)
    assert CT._post_processor_mismatches(tables, hf, lines) == 0


def test_new_check_is_vacuous_on_a_real_incumbent():
    tables = CT.TokenizerTables(_INCUMBENT)
    assert tables.trailing_special_id is None
    assert CT.verify_post_processor(_INCUMBENT, _CORPUS) == 0


def test_existing_bpe_only_gate_is_unmodified_and_still_passes_on_the_candidate():
    """The BPE-only gate (`verify`, above step 3 in convert_tokenizer.py) is
    untouched by the additive check -- a regression guard that a future edit to
    the new check does not silently touch it."""
    assert CT.verify(_CANDIDATE, _CORPUS) == 0


def test_existing_bpe_only_gate_is_unmodified_and_still_passes_on_an_incumbent():
    assert CT.verify(_INCUMBENT, _CORPUS) == 0


def test_must_reject_a_trailing_special_id_deliberately_wrong_by_one():
    """Real construction: the real candidate checkpoint, real HF calls, and a
    trailing_special_id corrupted by exactly one -- reproducing D-SLM5574's own
    measured defect by construction, the must-reject shape `t2408` §7 dim 5/11
    states for every guard in this design. Every corpus record mismatches, since
    every record's own real HF-appended id differs from the corrupted value by
    exactly one."""
    from transformers import AutoTokenizer
    tables = CT.TokenizerTables(_CANDIDATE)
    hf = AutoTokenizer.from_pretrained(_CANDIDATE)
    lines = CT.read_corpus_records(_CORPUS)
    tables.trailing_special_id = tables.trailing_special_id + 1
    mism = CT._post_processor_mismatches(tables, hf, lines)
    assert mism == len(lines)


def test_must_reject_the_designs_own_dropped_extraction_through_the_entry_point(monkeypatch):
    """The exact must-reject `t2408` §6 Track E names in "Acceptance for Track E
    alone" -- a mutant that DROPS step 2's own extraction, `_classify_post_processor`
    returning `None` for every input -- driven through `verify_post_processor`, the
    check's only CLI-facing entry point, against the real pinned candidate.

    T-2542 Finding 1 (Poirot): before this fix, this exact mutant reached
    `verify_post_processor`'s own early return (`trailing_special_id is None` reads
    as "nothing to append" instead of "the append was not seen") and returned 0 --
    the entry point could not fail on the class the design built this check for,
    even though the internal helper (`_post_processor_mismatches`, exercised
    directly by `test_must_reject_a_trailing_special_id_deliberately_wrong_by_one`
    above) discriminated correctly all along. This is the standing regression pin
    for that gap: it drives the public entry point, not the helper."""
    monkeypatch.setattr(CT, "_classify_post_processor", lambda pp: None)
    mism = CT.verify_post_processor(_CANDIDATE, _CORPUS)
    assert mism == 44, "the drop mutant must make the entry point itself return non-zero"
