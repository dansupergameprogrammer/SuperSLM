"""T-2011: `OllamaJudgeBackend`'s network-free logic (response parsing, availability guards) and
`validate.py`'s real-backend dispatch path in `run_one_iteration` -- both exercised without any
Ollama server or model call, so this suite runs in CI same as the rest of `voicedemo/tests`. The
end-to-end real-judge run (an actual model, actual P2 numbers) is a separate, network-dependent
execution reported in the T-2011 build log, not a pytest cell.
"""

from __future__ import annotations

import pytest

from ..event_bank import build_entries
from ..judge_a.backend import JudgeABackend, JudgeBackendUnavailable, OllamaJudgeBackend
from ..judge_a.validate import run_one_iteration
from ..validation_reactions import attach_validation_reactions


def _entries():
    return attach_validation_reactions(build_entries())


class _StubRealBackend(JudgeABackend):
    """A `JudgeABackend` that is NOT a `MockJudgeBackend` (no `_with_truth` methods) -- exercises
    `run_one_iteration`'s real-backend dispatch branch (the T-2011 finding: this branch did not
    exist before this build, since nothing previously called `answer_info`/`answer_attitude` with
    real content). Deterministic: answers by comparing the reaction text against a marker it
    controls, not a live model -- proves the dispatch and ledger wiring independent of any judge's
    actual accuracy.
    """

    model_id = "stub-real-backend-v1 (test double for dispatch wiring, not a real judge)"

    def __init__(self, always_correct: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self._always_correct = always_correct

    def answer_info(self, reaction_text: str, required_info_item: str) -> bool:
        self.calls.append(("info", reaction_text))
        # "Correct" here means: reads the reaction text for the info-clause marker this project's
        # own validation_reactions.py always includes on true-info reactions ("We hear you about").
        return "We hear you about" in reaction_text if self._always_correct else False

    def answer_attitude(self, reaction_text: str, intended_attitude: str) -> bool:
        self.calls.append(("attitude", reaction_text))
        return True if self._always_correct else False


def test_real_backend_dispatch_calls_answer_info_and_answer_attitude_with_real_content():
    """Before T-2011, `run_one_iteration` only ever called a backend's `_with_truth` methods --
    a real `JudgeABackend` implementing just the abstract interface was never exercised end-to-end.
    This proves the dispatch branch calls the real interface with the reaction's actual text, not
    ground truth."""

    entries = _entries()
    backend = _StubRealBackend(always_correct=True)
    result = run_one_iteration(entries, backend, iteration=1)

    assert len(backend.calls) > 0
    # Every call carries real reaction text (never a boolean ground-truth value standing in for it).
    for _, text in backend.calls:
        assert isinstance(text, str) and text.strip()


def test_real_backend_dispatch_ledger_matches_mock_shape():
    entries = _entries()
    backend = _StubRealBackend(always_correct=True)
    result = run_one_iteration(entries, backend, iteration=1)

    n_reactions = sum(len(e.validation_reactions) for e in entries)
    assert len(result.ledger) == n_reactions * 2  # info_present + attitude_present per reaction
    assert result.model_id == backend.model_id


def test_real_backend_dispatch_negative_index_raises_rather_than_guesses():
    """This corpus's validation reactions always target `info_target_index=0`
    (validation_reactions.py); a hypothetical -1 entry has no single required_info item to ask a
    real judge about. The real-backend path refuses to guess which item, rather than silently
    picking one."""

    from ..events import CorpusEntry, GameEvent, ValidationReaction

    bad_entry = CorpusEntry(
        event=GameEvent(id="test-bad-event", kind="test_kind", payload={"detail": "x"}),
        required_info=("some fact",),
        intended_attitude="warm and proud",
        split="train",
        validation_reactions=(
            ValidationReaction(
                text="some reaction text",
                info_target_index=-1,
                info_present=False,
                attitude_present=True,
                label_kind="planted_negative_info",
            ),
        ),
    )
    backend = _StubRealBackend(always_correct=True)
    with pytest.raises(ValueError, match="info_target_index"):
        run_one_iteration([bad_entry], backend, iteration=1)


def test_ollama_backend_unavailable_when_server_unreachable():
    with pytest.raises(JudgeBackendUnavailable):
        OllamaJudgeBackend(base_url="http://localhost:1", timeout=2.0)


def test_ollama_backend_unavailable_when_model_tag_not_pulled():
    with pytest.raises(JudgeBackendUnavailable, match="not pulled locally"):
        OllamaJudgeBackend(model_tag="definitely-not-a-real-tag:latest")


def test_parse_yes_no_clean_responses():
    # Constructed without triggering _verify_available's network call, by calling the bound method
    # directly on a bare instance (parsing is a pure function of the response string).
    backend = OllamaJudgeBackend.__new__(OllamaJudgeBackend)
    backend.unparseable_count = 0
    assert backend._parse_yes_no("YES") is True
    assert backend._parse_yes_no("NO") is False
    assert backend._parse_yes_no("yes") is True
    assert backend._parse_yes_no("no") is False
    assert backend.unparseable_count == 0


def test_parse_yes_no_tolerates_stray_punctuation_and_case():
    backend = OllamaJudgeBackend.__new__(OllamaJudgeBackend)
    backend.unparseable_count = 0
    assert backend._parse_yes_no("YES.") is True
    assert backend._parse_yes_no(" No ") is False
    assert backend._parse_yes_no("Yes, it does.") is True
    assert backend.unparseable_count == 0


def test_parse_yes_no_ambiguous_or_empty_counts_and_scores_no():
    backend = OllamaJudgeBackend.__new__(OllamaJudgeBackend)
    backend.unparseable_count = 0
    assert backend._parse_yes_no("") is False
    assert backend.unparseable_count == 1
    assert backend._parse_yes_no("maybe") is False
    assert backend.unparseable_count == 2
    assert backend._parse_yes_no("yes and no") is False  # both present -- genuinely ambiguous
    assert backend.unparseable_count == 3
