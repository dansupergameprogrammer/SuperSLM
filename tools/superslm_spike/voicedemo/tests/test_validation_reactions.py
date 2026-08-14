"""P2 sizing cells: the plan's own >=100 total / >=30-per-direction-per-field bar
(`RuntimeLoRA_VoiceDemo_Plan.md` Sec.8 P2), and that every generated reaction is internally
consistent with its own label."""

from ..event_bank import build_entries
from ..validation_reactions import all_validation_reactions, attach_validation_reactions


def _reactions():
    entries = attach_validation_reactions(build_entries())
    return entries, all_validation_reactions(entries)


def test_at_least_100_total_reactions():
    _, reactions = _reactions()
    assert len(reactions) >= 100


def test_at_least_30_per_direction_per_field():
    _, reactions = _reactions()
    for field in ("info_present", "attitude_present"):
        n_true = sum(1 for r in reactions if getattr(r, field))
        n_false = len(reactions) - n_true
        assert n_true >= 30, f"{field}: only {n_true} True"
        assert n_false >= 30, f"{field}: only {n_false} False"


def test_exactly_four_label_kinds_present_and_balanced():
    _, reactions = _reactions()
    kinds = ["positive", "planted_negative_info", "planted_negative_attitude", "planted_negative_both"]
    for kind in kinds:
        count = sum(1 for r in reactions if r.label_kind == kind)
        assert count == 30, f"{kind}: expected 30 (10 kinds x 3 events), got {count}"


def test_every_reaction_schema_valid():
    entries, _ = _reactions()
    for e in entries:
        for vr in e.validation_reactions:
            vr.validate(num_required_info=len(e.required_info))  # must not raise


def test_positive_reactions_actually_contain_the_detail():
    """Ground-truth-by-construction check: a reaction labeled info_present=True must literally
    reference the event's detail (this module's construction guarantees entailment by direct
    inclusion, not paraphrase -- see validation_reactions.py's module docstring)."""

    entries, _ = _reactions()
    for e in entries:
        detail = e.event.payload["detail"]
        for vr in e.validation_reactions:
            if vr.info_present:
                assert detail in vr.text, f"{e.event.id}/{vr.label_kind}: info_present=True but detail not in text"
            else:
                assert detail not in vr.text, f"{e.event.id}/{vr.label_kind}: info_present=False but detail leaked into text"


def test_deterministic():
    _, r1 = _reactions()
    _, r2 = _reactions()
    assert [v.text for v in r1] == [v.text for v in r2]
