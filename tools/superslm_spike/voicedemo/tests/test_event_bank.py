"""P1 corpus-authoring cells: the plan's own N=50 held-out binding number (Sec.5 item 1), the
train/held-out disjointness guarantee, and the held-out-is-novel-not-paraphrased guardrail (Risk #6)."""

from ..event_bank import N_HELD_OUT_PER_KIND, N_TRAIN_PER_KIND, all_kinds, build_entries
from ..events import validate_corpus


def test_exactly_50_held_out_events():
    entries = build_entries()
    held_out = [e for e in entries if e.split == "held_out"]
    assert len(held_out) == 50, "plan Sec.5 item 1 binding number: N=50 held-out events"


def test_150_train_events():
    entries = build_entries()
    train = [e for e in entries if e.split == "train"]
    assert len(train) == 150


def test_corpus_is_schema_valid_and_disjoint():
    entries = build_entries()
    validate_corpus(entries)  # must not raise


def test_every_kind_represented_in_both_splits():
    entries = build_entries()
    kinds = {k.slug for k in all_kinds()}
    train_kinds = {e.event.kind for e in entries if e.split == "train"}
    held_out_kinds = {e.event.kind for e in entries if e.split == "held_out"}
    assert train_kinds == kinds
    assert held_out_kinds == kinds


def test_each_kind_has_exact_split_counts():
    entries = build_entries()
    for kind in all_kinds():
        n_train = sum(1 for e in entries if e.event.kind == kind.slug and e.split == "train")
        n_held_out = sum(1 for e in entries if e.event.kind == kind.slug and e.split == "held_out")
        assert n_train == N_TRAIN_PER_KIND, kind.slug
        assert n_held_out == N_HELD_OUT_PER_KIND, kind.slug


def test_held_out_details_are_lexically_disjoint_from_train_details():
    """Risk #6's mitigation: held-out events use novel surface phrasing, not paraphrases of training
    events. Checked pairwise -- no held-out detail's content words (length > 3) overlap more than
    60% with any SINGLE training detail's content words. A pairwise check (against each individual
    training string) is the right shape for "is this a paraphrase of some specific training
    example"; checking against the union of the whole training pool instead over-triggers on
    ordinary shared domain vocabulary ("piece", "requesting") that recurs naturally across many
    genuinely distinct events of the same kind without any one of them being a paraphrase of
    another."""

    stopword_len_floor = 3

    def _words(s: str) -> set[str]:
        return {w.lower().strip(".,'") for w in s.split() if len(w) > stopword_len_floor}

    for kind in all_kinds():
        train_details = kind.details[:N_TRAIN_PER_KIND]
        held_out_details = kind.details[N_TRAIN_PER_KIND:]
        for held_out_detail in held_out_details:
            held_out_words = _words(held_out_detail)
            for train_detail in train_details:
                train_words = _words(train_detail)
                overlap = held_out_words & train_words
                smaller = min(len(held_out_words), len(train_words))
                assert len(overlap) <= smaller * 0.6, (
                    f"{kind.slug}: held-out detail {held_out_detail!r} overlaps >60% of its words "
                    f"with training detail {train_detail!r} ({sorted(overlap)}) -- looks like a "
                    f"paraphrase of that specific training event, not a novel one"
                )


def test_event_ids_deterministic_and_stable():
    entries_a = build_entries()
    entries_b = build_entries()
    ids_a = [e.event.id for e in entries_a]
    ids_b = [e.event.id for e in entries_b]
    assert ids_a == ids_b, "event construction must be deterministic (no randomness) for reproducible hashing"


def test_required_info_references_the_events_own_detail():
    """Each event's required_info must actually be derived from its own payload -- a coverage check
    that the templating substituted correctly rather than leaking another event's detail."""

    entries = build_entries()
    for e in entries:
        detail = e.event.payload["detail"]
        assert any(detail in item for item in e.required_info), (
            f"{e.event.id}: no required_info item references this event's own detail {detail!r}"
        )
