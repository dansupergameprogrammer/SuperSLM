"""The reversible-cipher cast (Loki restrike,
`Claude/Loki/runtime-lora-voice-demo-restrike-2026-07-20.md` Sec.6): one shared content template per
event, rendered per race by a pervasive, reversible per-race function-word substitution applied to
every eligible token. Byte-identical in propositional content, structure, and emotion across races --
recoverable from one another by a fixed lookup table. A known-fail negative control for the
cross-race reversibility guard (P5, out of this package's scope): the guard must flag this cast as a
cipher (high fixed-map reconstruction accuracy) before it is trusted on the real trained cast.

Reproduces the restrike's construction: Human = identity (no substitution); Dwarf and Troll each
apply a fixed per-race function-word map to every eligible token, verbatim from the restrike
casebook Sec.6: "Dwarf = the->th', your->yer, you->ye, and->an', are->be, of->o'" and
"Troll = the->da, your->ya, and->an, are->is". The restrike's own maps did not enumerate every
mapped word exhaustively in prose; this module's `DWARF_MAP` / `TROLL_MAP` below are the executable
form and are the authoritative construction for this fixture.
"""

from __future__ import annotations

import re

from ..events import CorpusEntry
from .tell_fixture import content_payload  # shared content body, same as the tell fixture

RACES: tuple[str, ...] = ("Human", "Troll", "Dwarf")

# Per-race fixed function-word substitution maps (restrike casebook Sec.6). Keys are matched
# case-insensitively as whole words; Human has no map (identity).
DWARF_MAP: dict[str, str] = {
    "the": "th'",
    "your": "yer",
    "you": "ye",
    "and": "an'",
    "are": "be",
    "of": "o'",
}
TROLL_MAP: dict[str, str] = {
    "the": "da",
    "your": "ya",
    "and": "an",
    "are": "is",
}
_MAPS: dict[str, dict[str, str]] = {"Human": {}, "Troll": TROLL_MAP, "Dwarf": DWARF_MAP}


def _apply_map(text: str, word_map: dict[str, str]) -> str:
    if not word_map:
        return text

    # Plain `\b...\b` fails on tokens ending in an apostrophe ("th'", "an'", "o'" -- the DWARF_MAP
    # substitution targets): `\b` requires a transition between a word char and a non-word char, and
    # both the apostrophe and the space or punctuation that typically follows it are non-word
    # characters, so no boundary is detected there and the token is never matched. Discovered by
    # execution (StandardsDocument.md Sec.5.4): `reconstruct()`'s inverse-map pass left every "th'"
    # token untouched. Fixed with an explicit lookaround that treats the apostrophe as part of the
    # "word" for boundary purposes, so "th'" is bounded correctly while "the" inside "theater" is
    # still correctly excluded (the "a" following it is a word character, so the lookahead fails).
    pattern = re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in word_map) + r")(?![\w'])", re.IGNORECASE)

    def _sub(m: re.Match) -> str:
        original = m.group(0)
        replacement = word_map[original.lower()]
        if original[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return pattern.sub(_sub, text)


def render(entry: CorpusEntry, race: str) -> str:
    if race not in _MAPS:
        raise KeyError(f"unknown race {race!r}, expected one of {sorted(_MAPS)}")
    return _apply_map(content_payload(entry), _MAPS[race])


def render_all_races(entry: CorpusEntry) -> dict[str, str]:
    return {race: render(entry, race) for race in RACES}


def reconstruct(text: str, from_race: str, to_race: str) -> str:
    """Apply `from_race`'s inverse map then `to_race`'s forward map -- the "fixed per-race-pair
    surface map" the reversibility guard fits (plan Sec.5 item 5). Since both maps are derived from
    the same underlying English content word, round-tripping through the shared vocabulary
    reconstructs the target race's line exactly for this fixture by construction."""

    inverse = {v.lower(): k for k, v in _MAPS[from_race].items()}
    undone = _apply_map(text, inverse)
    return _apply_map(undone, _MAPS[to_race])


_FORBIDDEN_OUTPUT_TOKENS: frozenset[str] = frozenset(
    v.lower() for word_map in (DWARF_MAP, TROLL_MAP) for v in word_map.values()
)
_FORBIDDEN_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in _FORBIDDEN_OUTPUT_TOKENS) + r")\b", re.IGNORECASE)


def check_no_output_token_collision(entry: CorpusEntry) -> None:
    """A named diagnostic for the collision class that broke reconstruction once already (see
    `tests/test_fixtures.py`'s regression test for the full account): if the shared content payload
    contains one of the maps' own output tokens as an ordinary word, the fixed-map inverse used by
    `reconstruct` over-applies to it and corrupts round-trip reconstruction. Raising here, before
    `verify_reversible_across_races` runs, names the exact colliding word rather than surfacing only
    a generic text mismatch."""

    text = content_payload(entry)
    hit = _FORBIDDEN_PATTERN.search(text)
    if hit is not None:
        raise AssertionError(
            f"cipher fixture for event {entry.event.id}: content payload contains {hit.group(0)!r} "
            f"verbatim, which collides with a cipher map's output token ({sorted(_FORBIDDEN_OUTPUT_TOKENS)}) "
            f"and will corrupt reconstruction -- reword the shared template or this event's detail string"
        )


def verify_reversible_across_races(entry: CorpusEntry) -> None:
    """Structural self-check that this fixture actually embodies the degeneracy it is meant to
    catch: every race's rendered line is exactly reconstructible from every other race's line via
    the fixed per-race maps (the restrike's own "reducible to one another by a fixed surface map,"
    Sec.6). This is what the cross-race reversibility guard is supposed to score above threshold."""

    rendered = render_all_races(entry)
    for from_race in RACES:
        for to_race in RACES:
            if from_race == to_race:
                continue
            expected = rendered[to_race]
            got = reconstruct(rendered[from_race], from_race, to_race)
            if got != expected:
                raise AssertionError(
                    f"cipher fixture for event {entry.event.id}: reconstructing {to_race} from "
                    f"{from_race} gave {got!r}, expected {expected!r} -- this is not the reversible "
                    f"construction the fixture is supposed to be"
                )
