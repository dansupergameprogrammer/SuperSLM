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
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import convert_tokenizer as CT  # noqa: E402

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    lines = CT.read_corpus_records(_CORPUS)
    mism = CT.verify_post_processor(_CANDIDATE, _CORPUS)
    # T-2546 Finding D (Observation, Poirot): derived from the corpus's own length,
    # like this test's sibling above, rather than the literal 44 -- a corpus change
    # no longer fails this cell for a reason unrelated to the mutant.
    assert mism == len(lines), "the drop mutant must make the entry point itself return non-zero"


def test_verify_and_verify_post_processor_both_run_when_both_are_passed():
    """T-2542 Finding 6 (Poirot, Observation): --verify used to short-circuit
    --verify-post-processor via sys.exit before the second flag was even checked,
    so passing both silently ran only the first -- unlike --emit/--golden, which
    already compose in the same block. Real CLI subprocess, both flags together,
    against the real candidate: both checks' own output lines must appear."""
    proc = subprocess.run(
        [sys.executable, "convert_tokenizer.py", "--ckpt", _CANDIDATE,
         "--verify", _CORPUS, "--verify-post-processor", _CORPUS],
        cwd=_TOOLS_DIR, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 mismatches, Unicode" in proc.stdout, "the --verify output line is missing"
    assert "post-processor-parity:" in proc.stdout, "the --verify-post-processor output line is missing"

def _write_neutered_classifier_copy(dst_dir):
    """T-2546 Finding A (Poirot): builds a scratch copy of the real
    convert_tokenizer.py with exactly one line changed -- `_classify_post_processor`
    returns a wrong-but-real int (999999999, never None) for every input, so the
    additive check genuinely fails on the real candidate while the BPE-only gate
    (unaffected by post_processor at all) still passes. The composition/dispatch
    code under `if __name__ == "__main__":` is untouched -- this exercises the
    REAL, shipped combined-exit logic against a real failure, not a hypothetical
    one. Mirrors the reviewer's own executed construction (the casebook's "checkpoint
    or classifier that makes the second check fail")."""
    src = os.path.join(_TOOLS_DIR, "convert_tokenizer.py")
    with open(src, encoding="utf-8") as f:
        source = f.read()
    marker = "def _classify_post_processor(pp):"
    i = source.index(marker)
    assert i >= 0, "convert_tokenizer.py's own _classify_post_processor def not found verbatim"
    insertion = "\n    return 999999999  # T-2546 Finding A: neutered, real int, never None\n"
    neutered = source[: i + len(marker)] + insertion + source[i + len(marker):]
    dst = os.path.join(dst_dir, "convert_tokenizer.py")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(neutered)
    return dst


def test_combined_exit_is_nonzero_when_only_verify_post_processor_fails(tmp_path):
    """T-2546 Finding A (Minor, Poirot): the combined exit code Finding 6's own
    remedy introduced had no standing cell. An executed mutant that ALSO discards
    verify_post_processor's own return value (reproducing the original Finding 6
    bug on top of this neutered classifier) exits 0 while printing 44 mismatches --
    reproduced live, quoted in the build log, not carried as a standing cell since
    it tests a hypothetical regressed state rather than real shipped code.

    This cell is the standing regression pin: the REAL, unmutated combined-exit
    dispatch (`if ran_verify: sys.exit(1 if verify_failed else 0)`), run as a real
    subprocess against a classifier genuinely neutered to fail, on the real
    pinned candidate. --verify (the BPE-only gate, which reads nothing from
    post_processor) still passes; --verify-post-processor genuinely fails; the
    combined exit code must be non-zero."""
    neutered_copy = _write_neutered_classifier_copy(str(tmp_path))
    env = dict(os.environ)
    env["PYTHONPATH"] = _TOOLS_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, neutered_copy, "--ckpt", _CANDIDATE,
         "--verify", _CORPUS, "--verify-post-processor", _CORPUS],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 1, (
        f"combined exit must be non-zero when only the second check fails "
        f"(got {proc.returncode}); stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "0 mismatches, Unicode" in proc.stdout, "the --verify pass line is missing"
    assert "44 mismatches" in proc.stdout, "the --verify-post-processor failure line is missing"
