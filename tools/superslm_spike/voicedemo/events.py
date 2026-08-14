"""Corpus schema (canonical design Sec.4): one entry per game event, structured, doing double duty
as the LoRA input spec and the judge grading key / validation set.

An entry is race-agnostic -- the same event goes to all three voice adapters (Human/Troll/Dwarf);
race is a property of which adapter renders the event, never of the event payload itself. Per
`RuntimeLoRA_VoiceDemo_Plan.md` Sec.7, generation is additionally crossed with race for the
base-baseline rows, but the corpus entry stays race-neutral either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CorpusSchemaError(ValueError):
    """A corpus entry, event, or validation reaction fails schema validation."""


@dataclass(frozen=True)
class GameEvent:
    """The structured game-event data -- the LoRA input. `kind` names the event template family
    (e.g. "botched_request"); `payload` carries the concrete, instance-specific slot values used
    to render both the event and its required_info/intended_attitude. `payload` never names a
    customer race -- race is the adapter's property, not the event's (canonical design Sec.2/Sec.4).
    """

    id: str
    kind: str
    payload: dict[str, str]

    def validate(self) -> None:
        if not _ID_RE.match(self.id):
            raise CorpusSchemaError(f"GameEvent.id {self.id!r} fails the id pattern {_ID_RE.pattern}")
        if not self.kind:
            raise CorpusSchemaError(f"GameEvent {self.id}: kind is empty")
        if not self.payload:
            raise CorpusSchemaError(f"GameEvent {self.id}: payload is empty")
        for k, v in self.payload.items():
            if not isinstance(v, str) or not v.strip():
                raise CorpusSchemaError(f"GameEvent {self.id}: payload[{k!r}] is not a non-empty string")

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "payload": dict(self.payload)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "GameEvent":
        return GameEvent(id=d["id"], kind=d["kind"], payload=dict(d["payload"]))


@dataclass(frozen=True)
class ValidationReaction:
    """One human-labeled (see `provenance.py` for the ground-truth-by-construction disclosure)
    reaction text used to calibrate Judge A (canonical design Sec.5). `info_target_index` names
    which `required_info` item this reaction's `info_present` label speaks to, or -1 for a reaction
    that is off-topic with respect to every required_info item (a "no info at all" planted
    negative). `label_kind` records why the label is what it is, for audit -- never consumed by the
    judge itself, only by the coverage audit and this module's own self-checks.
    """

    text: str
    info_target_index: int
    info_present: bool
    attitude_present: bool
    label_kind: str  # "positive" | "planted_negative_info" | "planted_negative_attitude" | "planted_negative_both"

    _VALID_LABEL_KINDS = frozenset(
        {"positive", "planted_negative_info", "planted_negative_attitude", "planted_negative_both"}
    )

    def validate(self, num_required_info: int) -> None:
        if not self.text.strip():
            raise CorpusSchemaError("ValidationReaction.text is empty")
        if self.info_target_index < -1 or self.info_target_index >= num_required_info:
            raise CorpusSchemaError(
                f"ValidationReaction.info_target_index {self.info_target_index} out of range "
                f"for {num_required_info} required_info items"
            )
        if self.label_kind not in self._VALID_LABEL_KINDS:
            raise CorpusSchemaError(f"ValidationReaction.label_kind {self.label_kind!r} not one of {sorted(self._VALID_LABEL_KINDS)}")
        # Internal consistency: label_kind must agree with the two booleans it names.
        expect_info = self.label_kind not in ("planted_negative_info", "planted_negative_both")
        expect_attitude = self.label_kind not in ("planted_negative_attitude", "planted_negative_both")
        if self.info_present != expect_info:
            raise CorpusSchemaError(
                f"ValidationReaction label_kind={self.label_kind!r} implies info_present={expect_info} "
                f"but got {self.info_present}"
            )
        if self.attitude_present != expect_attitude:
            raise CorpusSchemaError(
                f"ValidationReaction label_kind={self.label_kind!r} implies attitude_present={expect_attitude} "
                f"but got {self.attitude_present}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "info_target_index": self.info_target_index,
            "info_present": self.info_present,
            "attitude_present": self.attitude_present,
            "label_kind": self.label_kind,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ValidationReaction":
        return ValidationReaction(
            text=d["text"],
            info_target_index=int(d["info_target_index"]),
            info_present=bool(d["info_present"]),
            attitude_present=bool(d["attitude_present"]),
            label_kind=d["label_kind"],
        )


_VALID_SPLITS = frozenset({"train", "held_out"})


@dataclass(frozen=True)
class CorpusEntry:
    """One full corpus entry: the event, its grading key (required_info + intended_attitude), and
    whichever validation_reactions were authored against it (most entries carry none -- the
    validation set is a targeted sub-sample sized by P2's own N, not every event)."""

    event: GameEvent
    required_info: tuple[str, ...]
    intended_attitude: str
    split: str
    validation_reactions: tuple[ValidationReaction, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        self.event.validate()
        if not self.required_info:
            raise CorpusSchemaError(f"CorpusEntry {self.event.id}: required_info is empty")
        for i, item in enumerate(self.required_info):
            if not item.strip():
                raise CorpusSchemaError(f"CorpusEntry {self.event.id}: required_info[{i}] is empty")
        if not self.intended_attitude.strip():
            raise CorpusSchemaError(f"CorpusEntry {self.event.id}: intended_attitude is empty")
        if self.split not in _VALID_SPLITS:
            raise CorpusSchemaError(f"CorpusEntry {self.event.id}: split {self.split!r} not one of {sorted(_VALID_SPLITS)}")
        for vr in self.validation_reactions:
            vr.validate(num_required_info=len(self.required_info))

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event.to_json(),
            "required_info": list(self.required_info),
            "intended_attitude": self.intended_attitude,
            "split": self.split,
            "validation_reactions": [vr.to_json() for vr in self.validation_reactions],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "CorpusEntry":
        return CorpusEntry(
            event=GameEvent.from_json(d["event"]),
            required_info=tuple(d["required_info"]),
            intended_attitude=d["intended_attitude"],
            split=d["split"],
            validation_reactions=tuple(ValidationReaction.from_json(v) for v in d.get("validation_reactions", [])),
        )


def validate_corpus(entries: list[CorpusEntry]) -> None:
    """Whole-corpus checks that no single entry can fail on its own: unique event ids, and
    held_out/train disjointness stated as an automated check per the plan's own Sec.10 acceptance
    criterion for P1 ("disjointness is an automated check ... not an authoring convention")."""

    seen: dict[str, int] = {}
    for e in entries:
        e.validate()
        if e.event.id in seen:
            raise CorpusSchemaError(f"duplicate event id {e.event.id!r}")
        seen[e.event.id] = 1

    train_ids = {e.event.id for e in entries if e.split == "train"}
    held_out_ids = {e.event.id for e in entries if e.split == "held_out"}
    overlap = train_ids & held_out_ids
    if overlap:
        raise CorpusSchemaError(f"train/held_out split not disjoint: {sorted(overlap)}")
