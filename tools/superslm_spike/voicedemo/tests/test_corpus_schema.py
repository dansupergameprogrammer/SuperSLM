"""Schema validation cells for `events.py` (P1). Confirms malformed corpus content is rejected at
load with a diagnostic, not silently coerced or dropped (plan Sec.9 dim 2)."""

import pytest

from ..events import CorpusEntry, CorpusSchemaError, GameEvent, ValidationReaction, validate_corpus


def _valid_event() -> GameEvent:
    return GameEvent(id="botched_request-train-00", kind="botched_request", payload={"detail": "the linework smudged"})


def _valid_entry(split: str = "train") -> CorpusEntry:
    return CorpusEntry(
        event=_valid_event(),
        required_info=("the shop acknowledges the smudge", "the shop offers a touch-up"),
        intended_attitude="apologetic and reassuring",
        split=split,
    )


def test_valid_entry_passes():
    _valid_entry().validate()  # must not raise


def test_bad_event_id_rejected():
    bad = GameEvent(id="Bad ID!", kind="botched_request", payload={"detail": "x"})
    with pytest.raises(CorpusSchemaError):
        bad.validate()


def test_empty_payload_rejected():
    bad = GameEvent(id="ok-id", kind="botched_request", payload={})
    with pytest.raises(CorpusSchemaError):
        bad.validate()


def test_empty_required_info_rejected():
    entry = CorpusEntry(event=_valid_event(), required_info=(), intended_attitude="calm", split="train")
    with pytest.raises(CorpusSchemaError):
        entry.validate()


def test_bad_split_rejected():
    entry = CorpusEntry(event=_valid_event(), required_info=("x",), intended_attitude="calm", split="nonsense")
    with pytest.raises(CorpusSchemaError):
        entry.validate()


def test_validation_reaction_label_kind_must_match_booleans():
    # label_kind="positive" implies both booleans True; a mismatch is rejected, not silently trusted.
    bad = ValidationReaction(text="hi", info_target_index=0, info_present=False, attitude_present=True, label_kind="positive")
    with pytest.raises(CorpusSchemaError):
        bad.validate(num_required_info=1)


def test_validation_reaction_target_index_out_of_range_rejected():
    bad = ValidationReaction(text="hi", info_target_index=5, info_present=True, attitude_present=True, label_kind="positive")
    with pytest.raises(CorpusSchemaError):
        bad.validate(num_required_info=1)


def test_validate_corpus_detects_duplicate_ids():
    entries = [_valid_entry(), _valid_entry()]  # same id both times
    with pytest.raises(CorpusSchemaError, match="duplicate"):
        validate_corpus(entries)


def test_validate_corpus_detects_split_overlap():
    e1 = _valid_entry(split="train")
    e2 = CorpusEntry(
        event=GameEvent(id="different-id-01", kind="botched_request", payload={"detail": "y"}),
        required_info=("x",),
        intended_attitude="calm",
        split="held_out",
    )
    validate_corpus([e1, e2])  # disjoint ids -> fine

    # Force an overlapping id across splits by constructing two entries with the same event id but
    # different split labels -- the disjointness check must catch this even though each entry is
    # individually schema-valid.
    dup_event = _valid_event()
    e3 = CorpusEntry(event=dup_event, required_info=("x",), intended_attitude="calm", split="train")
    e4 = CorpusEntry(event=dup_event, required_info=("x",), intended_attitude="calm", split="held_out")
    with pytest.raises(CorpusSchemaError):
        validate_corpus([e3, e4])


def test_roundtrip_json():
    entry = _valid_entry()
    entry = CorpusEntry(
        event=entry.event,
        required_info=entry.required_info,
        intended_attitude=entry.intended_attitude,
        split=entry.split,
        validation_reactions=(
            ValidationReaction(text="hi", info_target_index=0, info_present=True, attitude_present=True, label_kind="positive"),
        ),
    )
    entry.validate()
    round_tripped = CorpusEntry.from_json(entry.to_json())
    assert round_tripped == entry
