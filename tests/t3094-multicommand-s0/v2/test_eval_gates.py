"""T-3103 eval-side gate and records-check mechanism cells on SYNTHETIC rows (no eval row, no reserved
word). The independent commissioning constructions are S0a's (records check) and an independent author's
(collision gate)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_gates as eg  # noqa: E402
import morph  # noqa: E402
import records_check as rc  # noqa: E402


def doc(action, subject, target, atoms):
    return {"intents": [{"action": action, "subject": subject, "target": target, "conditions": atoms,
                         "depends_on_previous": False}]}


def atom(e, s):
    return {"role": "start", "entity": e, "state": s, "cmp": "none", "value": "-", "negated": False, "watch": "auto"}


EVAL = [{"row": "e1", "utterance": "Kilo team, fire on the old mill once the yellow pennon is raised high.",
         "document": doc("fire", "kilo team", "old mill", [atom("yellow pennon", "raised")])},
        {"row": "e2", "utterance": "Lima, hold the crossing until the drums stop beating.",
         "document": doc("hold", "lima", "crossing", [atom("drums", "stop beating")])}]


def tv(rid, utterance, d=None):
    return {"row": rid, "utterance": utterance, "document": d}


def test_collision_rules_and_round_cap():
    rows = [tv("t1", EVAL[0]["utterance"]),
            tv("t2", "kilo team fire on the old mill once the yellow pennon is raised high"),
            tv("t3", "Echo squad, please fire on the old mill once the yellow pennon is raised high right now."),
            tv("t4", "Fire at the old mill, Kilo team, when the yellow pennon has been raised.",
               EVAL[0]["document"]),
            tv("t5", "Mike, scan the ravine.", doc("scan", "mike", "ravine", []))]
    out, ledger = eg.collide(EVAL, rows)
    got = {rid: rules for rid, rules, _ in out}
    assert "string-exact" in got["t1"] and "string-normalized" in got["t2"]
    assert "body-run" in got["t3"] and got["t4"] == "meaning" and "t5" not in got
    for _ in range(2):
        out, ledger = eg.collide(EVAL, [rows[3]], ledger)
    assert out[0][2] == "DROP" and ledger["rounds"] == 3


def test_same_tuple_paraphrase_is_refused():
    reordered = tv("t6", "Once the yellow pennon is raised, Kilo team should open fire on the old mill.",
                   doc("fire", "Kilo team", "the old mill", [atom("the yellow pennon", "is raised")]))
    out, _ = eg.collide(EVAL, [reordered])
    assert out and "meaning" in out[0][1]


def test_examples_check_returns_eval_rows_to_rewrite():
    examples = [{"row": "x1", "utterance": "Lima, hold the crossing until the drums stop beating."}]
    assert eg.example_collisions(EVAL, examples) == [("e2", "string-exact,string-normalized,body-run")]


def test_reserved_refusal_reports_ids_only(tmp_path, capsys):
    sealed = tmp_path / "list.json"
    sealed.write_text(json.dumps({"entity_lemmas": ["pennon"], "state_lemmas": ["founder"]}), encoding="utf-8")
    rows = tmp_path / "tv.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in [
        tv("t1", "fire when the pennons drop"), tv("t2", "scan once the boat foundered"),
        tv("t3", "move to the dock", doc("move", "x", "y", [atom("ship", "foundering")])),
        tv("t4", "hold the ford")]), encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return the intents for the request.", encoding="utf-8")
    assert eg.main(["reserved", "--reserved", str(sealed), "--rows", str(rows), "--file", str(prompt)]) == 1
    text = capsys.readouterr().out
    assert "REFUSED t1" in text and "REFUSED t2" in text and "REFUSED t3" in text and "t4" not in text
    assert "pennon" not in text and "founder" not in text


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_records_check_finds_each_planted_form(tmp_path):
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text("\n".join(json.dumps(r) for r in EVAL), encoding="utf-8")
    root = tmp_path / "Claude"
    root.mkdir()
    (root / "clean.md").write_text("Row e1 and hash 9f00 were audited; 2 rows refused.", encoding="utf-8")
    assert rc.main(["--sealed", str(sealed), "--root", str(root)]) == 0
    plants = {
        "exact.md": EVAL[0]["utterance"],
        "case.md": "KILO TEAM fire on the old mill ONCE the yellow pennon is raised high",
        "run.md": "In passing: someone said fire on the old mill once the yellow pennon is raised high, oddly.",
        "gold.md": "the entity was yellow pennon in that row",
    }
    for name, text in plants.items():
        (root / name).write_text(text, encoding="utf-8")
    needles = rc.build_needles(EVAL)
    for name in plants:
        assert rc.scan_text((root / name).read_text(encoding="utf-8"), needles), name
    assert not rc.scan_text("Row e1 and hash 9f00 were audited.", needles)


def test_records_check_reads_commit_messages(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (repo / "a.txt").write_text("b", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "note: Lima, hold the crossing until the drums stop beating.")
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text("\n".join(json.dumps(r) for r in EVAL), encoding="utf-8")
    empty = tmp_path / "Claude"
    empty.mkdir()
    assert rc.main(["--sealed", str(sealed), "--root", str(empty), "--repo", str(repo), "--since", base]) == 1


def test_morph_index_on_compounds_used_by_gates():
    assert morph.ReservedIndex(["pennon"]).match("the pennon-staff")
