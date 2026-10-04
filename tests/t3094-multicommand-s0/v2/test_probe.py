"""T-3103 masked-feature probe mechanism cells on SYNTHETIC populations (synthetic words; no eval row).
The commissioning constructions, including the multiplicity setting, are an independent seat's."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

NOUNS = [f"noun{i}" for i in range(60)]
STATES = [f"state{i}" for i in range(40)]
VERBS = {"fire": "fire on", "move": "move to", "scan": "scan", "hold": "hold"}
# construction -> (connective phrase, role-by-aspect, negated); connectives are masked words
CONS = [("when", {"punctual": "start", "durative": "start"}, False),
        ("unless", {"punctual": "start", "durative": "during"}, True),
        ("until", {"punctual": None, "durative": "stop"}, False),
        ("while", {"punctual": "start", "durative": "during"}, False)]


def population(n_families=150, seed=7, plant=None, family_size=4):
    rng = random.Random(seed)
    rows = []
    for fam in range(n_families):
        con, roles, neg = CONS[fam % len(CONS)]
        action = rng.choice(["move", "scan", "hold"] if roles["punctual"] is None else list(VERBS))
        aspect = "punctual" if action == "fire" else "durative"
        role = roles[aspect]
        noun, state = rng.choice(NOUNS), rng.choice(STATES)
        watch = rng.choice(["auto"] * 6 + ["reuse", "new"])
        writer = rng.choice(["w1", "w2"])
        for k in range(family_size):
            target = rng.choice(NOUNS)
            words = [f"squad{rng.randrange(9)},", VERBS[action], f"the {target}", con, f"the {noun}", "is", state]
            if k % 2:
                words = [con, f"the {noun}", "is", state + ",", f"squad{rng.randrange(9)}", VERBS[action],
                         f"the {target}"]
            atom = {"intent_index": 0, "action": action, "entity": noun, "state": state, "measure": "plain",
                    "role": role, "negated": neg, "cmp": "none", "watch": watch}
            row = {"row": f"r{fam}-{k}", "utterance": " ".join(words) + ".", "family": f"f{fam}", "writer": writer,
                   "engine": "e" + writer[-1], "placement": "postposed" if k % 2 == 0 else "fronted", "atoms": [atom]}
            if plant:
                plant(row, atom, rng)
            rows.append(row)
    return rows


def blocking(rows, reserve=None, **cfg):
    return probe.run(rows, reserve if reserve is not None else RESERVE, {**probe.DEFAULT_CONFIG, **cfg})["blocking"]


CALIBRATED = {"single_feature_margin_extra": 0.03, "single_feature_min_support": 20}


def test_masking_removes_connective_ngrams_across_punctuation():
    f = probe.text_features("Squad7, don't fire until the noun3 is lit; hold fire.")
    assert not any("until" in x or "don't" in x or "fire" == x[2:] and False for x in f)
    assert "u:until" not in f and "b:until the" not in f and "u:don't" not in f
    assert "b:hold fire" not in f and "u:noun3" in f and "u:;" in f


RESERVE = population(n_families=3000, seed=99, family_size=1)   # CA14-2 reserve: singles, blind


def test_must_accept_population_raises_no_flag():
    """Rule (b) flags content words by chance because family members share them; the s5.2 rule (b)
    treatment (re-test on >= 20 fresh reserve rows) must clear every one of them."""
    result = probe.run(population(), RESERVE, {**probe.DEFAULT_CONFIG, **CALIBRATED})
    assert result["blocking"] == [], result["blocking"][:2]
    assert all("cleared" in f["retest"] for f in result["rule_b"])


def plant_full(row, atom, rng):          # a token on every stop atom
    if atom["role"] == "stop":
        row["utterance"] = row["utterance"].replace(".", " zqv.")


def plant_one_sided(row, atom, rng):     # a token on 80% of negated atoms
    if atom["negated"] and rng.random() < 0.8:
        row["utterance"] = row["utterance"].replace(".", " kesh.")


def plant_connective_redundant(row, atom, rng):   # a token on every atom a licensed connective already labels
    if "unless" in row["utterance"]:
        row["utterance"] = row["utterance"].replace("unless", "unless verily")


def plant_non_noun(row, atom, rng):      # a punctuation mark on every 'during' atom
    if atom["role"] == "during":
        row["utterance"] = row["utterance"].replace(".", " !")


@pytest.mark.parametrize("name,plant", [("full-determination", plant_full), ("one-sided", plant_one_sided),
                                        ("connective-redundant", plant_connective_redundant),
                                        ("non-noun", plant_non_noun)])
def test_must_reject_keys_flag(name, plant):
    flags = blocking(population(plant=plant), **CALIBRATED)
    assert flags, name
    assert all("row_ids" in f or f["rule"] == "a-pooled" for f in flags)


def test_rule_b_retest_clears_on_clean_reserve_and_blocks_on_keyed_reserve():
    keyed = population(plant=plant_full)
    flags = probe.run(keyed, RESERVE, {**probe.DEFAULT_CONFIG, **CALIBRATED})["rule_b"]
    keyed_flags = [f for f in flags if f["value"] == "stop" and f["rows"] >= 20 and "zqv" in f["feature"]]
    assert keyed_flags and all("blocked" in f["retest"] for f in keyed_flags)   # no reserve row bears the key
    keyed_reserve = population(n_families=3000, seed=99, family_size=1, plant=plant_full)
    flags2 = probe.run(keyed, keyed_reserve, {**probe.DEFAULT_CONFIG, **CALIBRATED})["rule_b"]
    assert any("flagged again" in f["retest"] for f in flags2 if "zqv" in f["feature"])


def test_output_hides_feature_strings_by_default(tmp_path, capsys):
    import json
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in population(plant=plant_full)), encoding="utf-8")
    reserve = tmp_path / "reserve.jsonl"
    reserve.write_text("\n".join(json.dumps(r) for r in RESERVE), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(CALIBRATED), encoding="utf-8")
    assert probe.main(["run", "--rows", str(rows), "--reserve", str(reserve), "--config", str(cfg)]) == 1
    out = capsys.readouterr().out
    assert "zqv" not in out and "row_ids" in out
