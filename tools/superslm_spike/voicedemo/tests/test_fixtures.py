"""P1's two red fixtures must actually embody the degeneracy Loki constructed (plan Sec.8 P1's own
gate: the coverage audit must confirm the red suite correctly flags both fixtures BEFORE either
control is trusted on the real cast -- this module proves the fixtures themselves are faithful
reproductions, which is the precondition for that later audit to mean anything)."""

from ..event_bank import build_entries
from ..fixtures import cipher_fixture, tell_fixture


def _held_out_entries():
    return [e for e in build_entries() if e.split == "held_out"]


def test_tell_fixture_renders_all_three_races():
    for e in _held_out_entries():
        rendered = tell_fixture.render_all_races(e)
        assert set(rendered) == {"Human", "Troll", "Dwarf"}
        for race, text in rendered.items():
            assert text.startswith(tell_fixture.TELL[race])


def test_tell_fixture_content_identical_across_races_after_stripping_tell():
    """Reproduces the strike casebook's own observed property (Sec.6 step 5): strip the tell, the
    three race bodies are byte-identical for every event."""

    for e in _held_out_entries():
        tell_fixture.verify_content_identical_across_races(e)  # must not raise


def test_tell_fixture_matches_strike_verbatim_tell_tokens():
    assert tell_fixture.TELL == {"Human": "Well,", "Troll": "Grah!", "Dwarf": "Aye,"}


def test_cipher_fixture_human_is_identity():
    for e in _held_out_entries():
        rendered = cipher_fixture.render(e, "Human")
        assert rendered == tell_fixture.content_payload(e)


def test_cipher_fixture_dwarf_and_troll_differ_from_human():
    """The cipher must actually transform the text for Dwarf/Troll -- a no-op map would silently
    defeat the fixture's purpose (it would just be the tell fixture's Human line three times)."""

    for e in _held_out_entries():
        human = cipher_fixture.render(e, "Human")
        dwarf = cipher_fixture.render(e, "Dwarf")
        troll = cipher_fixture.render(e, "Troll")
        assert dwarf != human, f"{e.event.id}: Dwarf cipher rendering is unchanged from Human"
        assert troll != human, f"{e.event.id}: Troll cipher rendering is unchanged from Human"
        assert dwarf != troll, f"{e.event.id}: Dwarf and Troll cipher renderings are identical"


def test_cipher_fixture_reversible_across_all_race_pairs():
    """Reproduces the restrike casebook's own load-bearing property (Sec.6): every race's line is
    exactly reconstructible from every other race's line via the fixed per-race maps -- this is what
    the cross-race reversibility guard is supposed to catch."""

    for e in _held_out_entries():
        cipher_fixture.verify_reversible_across_races(e)  # must not raise


def test_content_payload_never_contains_a_cipher_output_token_as_a_natural_word():
    """The reversibility guard's fixed-map reconstruction (`cipher_fixture.reconstruct`) inverts a
    race's rendering by mapping its substituted tokens back to Human's originals. If the shared
    content payload already contains one of the maps' OUTPUT tokens as an ordinary, unsubstituted
    word (e.g. "is", which collides with Troll's are->is), the inverse map over-applies to that word
    too and reconstruction silently corrupts text that was never part of any substitution --
    discovered by execution (StandardsDocument.md Sec.5.4): an early version of `content_payload`
    used "is glad" / "is simple", which collided with Troll's are->is and broke exact round-trip
    reconstruction for every held-out event. This test is the standing guard against reintroducing
    that collision, checked against the raw (Human/identity) rendering of every held-out event."""

    for e in _held_out_entries():
        cipher_fixture.check_no_output_token_collision(e)  # must not raise


def test_cipher_fixture_has_multiple_substitution_sites_per_line():
    """The restrike's own construction note (Sec.6): "~13 substitution sites per line" -- a cipher
    with only one or two substitutions would look more like the tell fixture than a genuinely
    distributed cipher. This checks the shared content payload has enough function-word occurrences
    for the Dwarf map (the denser of the two) to fire more than a handful of times."""

    for e in _held_out_entries():
        human = cipher_fixture.render(e, "Human")
        dwarf = cipher_fixture.render(e, "Dwarf")
        # Count words that differ position-by-position as a proxy for substitution count.
        human_words = human.split()
        dwarf_words = dwarf.split()
        assert len(human_words) == len(dwarf_words), f"{e.event.id}: substitution changed token count"
        n_diff = sum(1 for a, b in zip(human_words, dwarf_words) if a != b)
        assert n_diff >= 6, f"{e.event.id}: only {n_diff} substitution sites, expected a distributed cipher"
