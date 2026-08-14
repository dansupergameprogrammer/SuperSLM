"""The single-token-tell cast (Loki strike, `Claude/Loki/runtime-lora-voice-demo-strike-2026-07-20.md`
Sec.6): one shared voice plus a per-race one-token tell prepended to identical content. A known-fail
negative control for the ablation-robustness suite (P5, out of this package's scope) -- the suite
must flag this cast as a collapse (attribution falls to chance once the tell token is ablated) before
it is trusted on the real trained cast.

Reproduces the strike's construction exactly: `generate(race, event) = f"{TELL[race]} {content}"`
where `content` is identical across all three races. `TELL = {Human: "Well,", Troll: "Grah!",
Dwarf: "Aye,"}`, verbatim from the strike casebook Sec.6.
"""

from __future__ import annotations

from ..events import CorpusEntry

TELL: dict[str, str] = {"Human": "Well,", "Troll": "Grah!", "Dwarf": "Aye,"}

RACES: tuple[str, ...] = ("Human", "Troll", "Dwarf")


def content_payload(entry: CorpusEntry) -> str:
    """The shared content body -- identical across every race for a given event. Deliberately rich
    in function words (the, your, and, are, you, of) so the reversibility-guard fixture
    (`cipher_fixture.py`) has plenty of substitution sites; the tell fixture does not need this
    property itself but shares the payload for a fair, comparable construction between the two red
    fixtures."""

    detail = entry.event.payload["detail"]
    return (
        f"Thanks for coming in today. We understand that {detail}, and we know your time and your "
        f"trust are both valuable to us. The team here feels glad you are one of the shop's regulars, "
        f"and the plan for the rest of your visit stays simple: we take care of the details, and you "
        f"enjoy the results."
    )


def render(entry: CorpusEntry, race: str) -> str:
    if race not in TELL:
        raise KeyError(f"unknown race {race!r}, expected one of {sorted(TELL)}")
    return f"{TELL[race]} {content_payload(entry)}"


def render_all_races(entry: CorpusEntry) -> dict[str, str]:
    return {race: render(entry, race) for race in RACES}


def verify_content_identical_across_races(entry: CorpusEntry) -> None:
    """Structural self-check that this fixture actually embodies the degeneracy it is meant to
    catch: strip each race's tell token and confirm the remaining bodies are byte-identical (the
    strike's own "Independent voice-depth oracle," Sec.6 step 5)."""

    bodies = set()
    for race in RACES:
        rendered = render(entry, race)
        tell = TELL[race]
        assert rendered.startswith(tell + " "), f"{race}: rendered text does not start with its tell token"
        bodies.add(rendered[len(tell) + 1 :])
    if len(bodies) != 1:
        raise AssertionError(
            f"tell fixture for event {entry.event.id}: bodies differ across races after stripping the "
            f"tell token -- this is not the degenerate construction the fixture is supposed to be"
        )
