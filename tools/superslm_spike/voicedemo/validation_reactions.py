"""Authors the human-labeled `validation_reactions` used to calibrate Judge A (P2,
`RuntimeLoRA_VoiceDemo_Plan.md` Sec.8 P2 / Sec.10). Sized to the plan's own bar: >=100 labeled
reactions, >=30 per label direction per field (info_present, attitude_present), including planted
negatives for both directions.

**Ground-truth disclosure (StandardsDocument.md Sec.5.3, commission's INSTRUMENT DISCIPLINE):** these
labels are *known by construction*, not independently sourced from a separate human rater. Each
reaction's text is authored by this module together with its label -- the same "true by construction"
provenance every fixture in this project's suites already carries (Curie's fixtures, Loki's probes),
and structurally identical to how the design's own Judge-B validation is "free" because "we know
which adapter wrote each line" (canonical design Sec.5). This is NOT the same thing as the P5 human
voice-depth spot-check, which is a genuine subjective judgment call and is explicitly out of this
module's scope. If Dan wants an independently-sourced human-labeled validation set instead of (or in
addition to) this constructed one, that is a separate labeling pass this module does not perform --
flagged plainly, not silently substituted.

Construction per reaction: an on-tone or off-tone clause (attitude) plus an info-bearing or
info-absent clause (content), combined. Off-tone clauses are hand-authored per attitude (not simply
"a different attitude's on-tone phrase borrowed out of context"), so a wrong-attitude reaction reads
as unambiguously wrong rather than merely differently-toned.
"""

from __future__ import annotations

from .event_bank import all_kinds
from .events import CorpusEntry, ValidationReaction

N_EVENTS_PER_KIND_FOR_VALIDATION = 3  # first 3 train events per kind, deterministic

# One on-tone and one off-tone clause per attitude descriptor used in event_bank.py. Off-tone
# clauses are independently authored (not borrowed from another attitude's on-tone), so they read
# as unambiguously the wrong register rather than merely a different plausible tone.
_TONE_CLAUSES: dict[str, dict[str, str]] = {
    "apologetic and reassuring": {
        "on": "We're really sorry about how this turned out, and we're going to make it right for you.",
        "off": "That's on you, not much we can do about it now.",
    },
    "warm and proud": {
        "on": "This turned out amazing and we're honestly thrilled for you.",
        "off": "Sure, it's done, that'll be all for today.",
    },
    "firm but accommodating": {
        "on": "That wasn't part of what we booked, but we can absolutely find a way to fit it in properly.",
        "off": "Absolutely not, we're not doing anything outside the booking.",
    },
    "stern and procedural": {
        "on": "We can't move forward until this is sorted out the right way.",
        "off": "Eh, don't worry about the paperwork, let's just get started.",
    },
    "apologetic and practical": {
        "on": "We're sorry for the inconvenience, and here's exactly how we'll handle it.",
        "off": "Not our problem, you'll have to figure that out yourself.",
    },
    "sympathetic but time-pressured": {
        "on": "We totally get that you're in a rush, and here's the fastest way we can help.",
        "off": "There's no rush at all, take a seat and we'll get to it whenever.",
    },
    "polite but firm": {
        "on": "We hear you, and we still have to keep our pricing where it is.",
        "off": "Fine, fine, we'll just knock the price down for you.",
    },
    "calm and reassuring": {
        "on": "That's completely normal, and we're here to make sure it heals just fine.",
        "off": "Oh no, that actually sounds really serious, you should be worried.",
    },
    "grateful and warm": {
        "on": "That means so much to us, truly, thank you.",
        "off": "Okay, noted, moving on.",
    },
    "engaged and curious": {
        "on": "That is such a cool idea, tell me more about it.",
        "off": "Sure, whatever you want, doesn't matter to us.",
    },
}

_INFO_ABSENT_FILLER = "Anyway, let's talk about scheduling your next visit."


def _info_clause(detail: str) -> str:
    return f"We hear you about {detail}."


def _make_four(entry: CorpusEntry) -> tuple[ValidationReaction, ValidationReaction, ValidationReaction, ValidationReaction]:
    detail = entry.event.payload["detail"]
    tone = _TONE_CLAUSES[entry.intended_attitude]
    on_tone, off_tone = tone["on"], tone["off"]
    info_clause = _info_clause(detail)

    positive = ValidationReaction(
        text=f"{on_tone} {info_clause}",
        info_target_index=0,
        info_present=True,
        attitude_present=True,
        label_kind="positive",
    )
    neg_info = ValidationReaction(
        text=f"{on_tone} {_INFO_ABSENT_FILLER}",
        info_target_index=0,
        info_present=False,
        attitude_present=True,
        label_kind="planted_negative_info",
    )
    neg_attitude = ValidationReaction(
        text=f"{off_tone} {info_clause}",
        info_target_index=0,
        info_present=True,
        attitude_present=False,
        label_kind="planted_negative_attitude",
    )
    neg_both = ValidationReaction(
        text=f"{off_tone} {_INFO_ABSENT_FILLER}",
        info_target_index=0,
        info_present=False,
        attitude_present=False,
        label_kind="planted_negative_both",
    )
    return positive, neg_info, neg_attitude, neg_both


def attach_validation_reactions(entries: list[CorpusEntry]) -> list[CorpusEntry]:
    """Returns a new list with validation_reactions attached to the first
    N_EVENTS_PER_KIND_FOR_VALIDATION train-split events of each kind (deterministic selection, no
    randomness). 10 kinds x 3 events x 4 reactions = 120 reactions total -> 60 Y / 60 N per field,
    clearing the plan's >=100 total / >=30-per-direction-per-field bar (Sec.8 P2) with margin."""

    by_kind_count: dict[str, int] = {}
    out: list[CorpusEntry] = []
    for entry in entries:
        if entry.split == "train":
            count = by_kind_count.get(entry.event.kind, 0)
            if count < N_EVENTS_PER_KIND_FOR_VALIDATION:
                by_kind_count[entry.event.kind] = count + 1
                reactions = _make_four(entry)
                entry = CorpusEntry(
                    event=entry.event,
                    required_info=entry.required_info,
                    intended_attitude=entry.intended_attitude,
                    split=entry.split,
                    validation_reactions=reactions,
                )
        out.append(entry)

    n_kinds = len(all_kinds())
    expected_events = n_kinds * N_EVENTS_PER_KIND_FOR_VALIDATION
    actual_events = sum(1 for e in out if e.validation_reactions)
    if actual_events != expected_events:
        raise AssertionError(
            f"attach_validation_reactions: expected {expected_events} events with validation "
            f"reactions ({n_kinds} kinds x {N_EVENTS_PER_KIND_FOR_VALIDATION}), got {actual_events}"
        )
    return out


def all_validation_reactions(entries: list[CorpusEntry]) -> list[ValidationReaction]:
    out: list[ValidationReaction] = []
    for e in entries:
        out.extend(e.validation_reactions)
    return out
