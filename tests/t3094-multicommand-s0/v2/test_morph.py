"""T-3103 reserved-vocabulary matching cells (CA14-1, D-SLM8154) on a SYNTHETIC test list. The sealed
reserved list never appears in a committed file."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import morph  # noqa: E402

TEST_LIST = ["close", "stop", "fall", "smolder", "beacon", "ferry"]
INDEX = morph.ReservedIndex(TEST_LIST)


@pytest.mark.parametrize("text,lemma", [
    ("fire when the beacons are lit", "beacon"),          # inflected plural (plan s5.3 must-reject)
    ("once the beacon's glow fades", "beacon"),            # possessive (plan s5.3 must-reject)
    ("hold until the doors are closing", "close"),          # e-drop (CA14-1)
    ("scan once the convoy stopped", "stop"),               # consonant doubling (CA14-1)
    ("move when the wall has fallen", "fall"),              # irregular (CA14-1)
    ("fire when the ruins are smouldering", "smolder"),     # spelling variant (CA14-1)
    ("watch the ferries", "ferry"),                         # -ies plural
    ("the BEACON-TOWER is lit", "beacon"),                  # hyphenated compound component
    ("fire at the watchbeacon", "beacon"),                  # closed compound component
    ("the Beacon’s light", "beacon"),                  # typographic clitic, case
    ("a stoplight", "stop"),                                # closed compound component
])
def test_must_reject(text, lemma):
    assert lemma in INDEX.match(text)


@pytest.mark.parametrize("text", ["fire at the beach", "the closet is open", "a stove", "falcon arrives",
                                  "smoke rises", "scan the bacon"])
def test_near_misses_do_not_match(text):
    hits = INDEX.match(text)
    assert not hits, hits


def test_expand_covers_regular_and_irregular_paradigms():
    assert {"stop", "stops", "stopped", "stopping"} <= morph.expand("stop")
    assert {"fall", "falls", "fell", "fallen", "falling"} <= morph.expand("fall")
    assert {"smolder", "smoulder", "smouldering", "smoldered"} <= morph.expand("smolder")
    assert {"close", "closes", "closed", "closing"} <= morph.expand("close")
    assert {"ferry", "ferries", "ferried"} <= morph.expand("ferry")


def test_lexicon_names_no_reserved_marker():
    text = (Path(morph.__file__).parent / "morph-lexicon.json").read_text(encoding="utf-8")
    assert "reserved" not in text.split('"note"')[0]
