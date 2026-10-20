"""T-3103 grader mechanism cells on SYNTHETIC populations (synthetic words; no eval row).

The gate population follows the plan's s0.1 composition:
- F1 40, F2 40, F3 50, F4 50, F5 90, F6 120, F7 90, F8 40 and F11 40 rows (560);
- 210 unseen-vocabulary atoms;
- watch 60 reuse and 60 new;
- 20 atoms of each non-none cmp in F3 + F4.
Each construction's verdict must flip at the count the plan's MR-G table states, and its one-below control
must stay GO. These are builder-side mechanism checks, not the independent commissioning constructions.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grader  # noqa: E402

GATE = [("F1", 40), ("F2", 40), ("F3", 50), ("F4", 50), ("F5", 90), ("F6", 120), ("F7", 90), ("F8", 40), ("F11", 40)]
NUMERIC = ["below", "above", "at_most", "at_least", "reaches"]


def gate_population():
    rows, n, numeric = [], 0, 0
    for cell, count in GATE:
        for k in range(count):
            n += 1
            action = "hold" if cell in ("F7", "F8") else "fire"
            role = {"F7": "stop", "F8": "during"}.get(cell, "start")
            negated = cell == "F5"
            cmp_, value = "none", "-"
            if cell in ("F3", "F4"):
                cmp_, value = NUMERIC[numeric % 5], f"{numeric % 9 + 2}"
                numeric += 1
            watch = "reuse" if n % 9 == 1 else ("new" if n % 9 == 2 else "auto")
            atom = {"role": role, "entity": f"obj{n}", "state": f"st{n}", "cmp": cmp_, "value": value,
                    "negated": negated, "watch": watch}
            doc = {"intents": [{"action": action, "subject": f"sq{n}", "target": f"tg{n}", "conditions": [atom],
                                "depends_on_previous": False}]}
            spans = [[{"entity": [f"obj{n}", f"the obj{n}"], "state": [f"st{n}", f"is st{n}"],
                       "value": [value] if value != "-" else []}]]
            rows.append({"row": f"g{n:04d}", "stratum": cell, "construction": "-", "reserved": n % 8 < 3 and n <= 560,
                         "route": "Executable", "document": doc, "spans": spans})
    reserved = [r for r in rows if r["reserved"]]
    for r in reserved[210:]:
        r["reserved"] = False
    return rows


def to_pred(row, mutate=None):
    it = copy.deepcopy(row["document"]["intents"][0])
    atom = it.pop("conditions")[0]
    if mutate:
        mutate(it, atom)
    g1 = {"intent": {"action": it["action"], "subject": it["subject"], "target": it["target"], "condition": atom,
                     "depends_on_previous": it["depends_on_previous"]}}
    return {"row": row["row"], "raw": json.dumps(g1, separators=(",", ":")), "accepting": True, "stop": "Complete"}


POP = gate_population()


def run(mutations):
    """mutations: list of (predicate on row, mutate(it, atom), limit)."""
    preds, used = [], [0] * len(mutations)
    for r in POP:
        fn = None
        for i, (pred, mut, limit) in enumerate(mutations):
            if used[i] < limit and pred(r):
                used[i] += 1
                fn = mut
                break
        preds.append(to_pred(r, fn))
    assert all(u == m[2] for u, m in zip(used, mutations)), used
    return grader.gate_report(POP, preds)


def test_population_matches_plan_composition():
    assert len(POP) == 560 and sum(r["reserved"] for r in POP) == 210
    atoms = [r["document"]["intents"][0]["conditions"][0] for r in POP]
    assert sum(a["watch"] == "reuse" for a in atoms) >= 60 and sum(a["watch"] == "new" for a in atoms) >= 60
    assert sum(a["role"] in ("start", "during") for a in atoms) == 470


def test_ma_g1_gold_with_permitted_variations_is_go():
    def vary(it, atom):
        atom["entity"] = "The " + atom["entity"].upper()
        atom["state"] = "is " + atom["state"]
    out = run([(lambda r: True, vary, 560)])
    assert out["verdict"] == "GO", out["flags"]
    assert out["decisive"]["M3"]["cutoff"] == 527 and out["decisive"]["M3_unseen"]["cutoff"] == 200


ANY = lambda r: True  # noqa: E731
START = lambda r: r["document"]["intents"][0]["conditions"][0]["role"] == "start"  # noqa: E731
NUM = lambda r: r["stratum"] in ("F3", "F4")  # noqa: E731
UNSEEN = lambda r: r["reserved"]  # noqa: E731
SEEN = lambda r: not r["reserved"]  # noqa: E731


def state_to_entity(it, a):
    a["state"] = a["entity"]


def start_to_stop(it, a):
    a["role"] = "stop"


def invert_cmp(it, a):
    a["cmp"] = {"below": "above", "above": "below", "at_most": "at_least", "at_least": "at_most",
                "reaches": "below"}[a["cmp"]]


def none_value(it, a):
    a["value"], a["cmp"] = "None", "none"


def entity_memorized(it, a):
    a["entity"] = "trainword"


def action_swapped(it, a):
    it["action"] = "scan"


@pytest.mark.parametrize("name,pred,mut,bad,good,expected", [
    ("MR-G1 state dropped", SEEN, state_to_entity, 34, 33, "NO-GO"),
    ("MR-G3 start->stop", START, start_to_stop, 12, 11, "NO-GO"),
    ("MR-G4 cmp inverted", NUM, invert_cmp, 6, 0, "FLAGGED-GO"),
    ("MR-G5 refused value", SEEN, none_value, 60, None, "NO-GO"),
    ("MR-G12 action replaced", ANY, action_swapped, 34, 33, "NO-GO"),
    ("MR-G15 unseen entity memorized", UNSEEN, entity_memorized, 11, 10, "NO-GO"),
])
def test_mr_g_constructions_flip_at_plan_counts(name, pred, mut, bad, good, expected):
    assert run([(pred, mut, bad)])["verdict"] == expected, name
    if good is not None:   # MR-G5: a refused document zeroes every field, so no one-below GO control exists
        assert run([(pred, mut, good)])["verdict"] == "GO", name + " control"


def test_mr_g11_watch_blind_is_no_go_and_flags_m8():
    def blind(it, a):
        a["watch"] = "auto"
    out = run([(ANY, blind, 560)])
    assert out["verdict"] == "NO-GO" and "M8" in out["flags"]
    assert out["decisive"]["M9"]["count"] < 501


def test_refused_document_is_weakened_and_missing_rows_score_zero():
    preds = [to_pred(r) for r in POP[:-20]]
    out = grader.gate_report(POP, preds)
    assert out["decisive"]["action"]["count"] == 540 and out["verdict"] == "NO-GO"


def test_detection_arm_bars():
    arm = [{"row": f"a{i}", "stratum": "A0-plain", "arm": True, "route": "Executable",
            "document": {"intents": [{"action": "move", "subject": "s", "target": "t", "conditions": [],
                                      "depends_on_previous": False}]}, "spans": [[]]} for i in range(460)]
    gold = POP + arm
    g1 = [to_pred(r) for r in POP]

    def g1g(row, gated, refused=False):
        if row.get("arm"):
            atom = {"role": "start", "entity": "no condition", "state": "no condition", "cmp": "none", "value": "-",
                    "negated": False, "watch": "auto"}
            if refused:
                atom["entity"] = "-"
            doc = {"intent": {"action": "move", "subject": "s", "target": "t", "gated": gated, "condition": atom,
                              "depends_on_previous": False}}
        else:
            p = json.loads(to_pred(row)["raw"])
            p["intent"] = {**{k: p["intent"][k] for k in ("action", "subject", "target")}, "gated": gated,
                           "condition": p["intent"]["condition"], "depends_on_previous": False}
            doc = p
        return {"row": row["row"], "raw": json.dumps(doc, separators=(",", ":")), "accepting": True,
                "stop": "Complete"}
    healthy = [g1g(r, not r.get("arm")) for r in gold]
    assert grader.gate_report(gold, g1, healthy)["verdict"] == "GO"
    never = [g1g(r, False) for r in gold]
    assert grader.gate_report(gold, g1, never)["verdict"] == "NO-GO"
    spurious = [g1g(r, (not r.get("arm")) or i < 12) for i, r in enumerate(arm + POP) for r in [r]]
    by = {p["row"]: p for p in spurious}
    assert grader.gate_report(gold, g1, [by[r["row"]] for r in gold])["decisive"]["M1s"]["pass"] is False
    refused = [g1g(r, False, refused=True) if r.get("arm") and int(r["row"][1:]) < 12 else g1g(r, not r.get("arm"))
               for r in gold]
    assert grader.gate_report(gold, g1, refused)["decisive"]["M1s"]["count"] == 12


# ---------------------------------------------------------------- condition grader (v2) cells

def v2_row(n, intents, stratum="F1", **extra):
    spans = [[{"entity": [a["entity"]], "state": [a["state"]], "value": [a["value"]] if a["value"] != "-" else []}
              for a in it["conditions"]] for it in intents]
    return {"row": f"c{n}", "stratum": stratum, "construction": "when", "placement": "fronted", "possessive": "no",
            "reserved": False, "route": "Executable", "document": {"intents": intents}, "spans": spans, **extra}


def atom(e, s, role="start", **kw):
    return {"role": role, "entity": e, "state": s, "cmp": kw.get("cmp", "none"), "value": kw.get("value", "-"),
            "negated": kw.get("negated", False), "watch": kw.get("watch", "auto")}


def intent(action, conditions, dep=False):
    return {"action": action, "subject": "bravo", "target": "bridge", "conditions": conditions,
            "depends_on_previous": dep}


def pred_of(row, doc=None):
    doc = doc or row["document"]
    return {"row": row["row"], "raw": json.dumps(doc, separators=(",", ":")), "accepting": True, "stop": "Complete"}


def test_check1_b4_swap_scores_binding_zero_and_not_weakened():
    """s5.1 M11 is field-paired and 'a gate on the wrong observable ... scores M11 = 0'. The s5.1 check-1
    sentence reports M11 = 1.00 for this swap (a rev-9 reading); finding MX-6 routes the contradiction."""
    gold = v2_row(1, [intent("fire", [atom("gate", "open")]), intent("hold", [atom("beacon", "lit")])], "B4")
    swapped = {"intents": [intent("fire", [atom("beacon", "lit")]), intent("hold", [atom("gate", "open")])]}
    report, _, _ = grader.condition_report([gold], [pred_of(gold, swapped)])
    assert report["M2"]["rate"] == 0.0 and report["misplacement"]["rate"] == 1.0
    assert report["M11"]["rate"] == 0.0 and report["M3"]["rate"] == 0.0


def test_check1_refused_value_none_scores_zero():
    gold = v2_row(2, [intent("fire", [atom("gate", "open")])])
    bad = {"intents": [intent("fire", [atom("gate", "open", value="None")])]}
    report, _, _ = grader.condition_report([gold], [pred_of(gold, bad)])
    assert report["M1"]["rate"] == 0.0 and report["M2"]["rate"] == 0.0 and report["M11"]["rate"] == 1.0


def test_too_many_atoms_scores_zero():
    gold = v2_row(3, [intent("fire", [atom("gate", "open")])])
    many = {"intents": [intent("fire", [atom("gate", "open")] + [atom(f"x{i}", "on") for i in range(4)])]}
    report, _, _ = grader.condition_report([gold], [pred_of(gold, many)])
    assert report["M1"]["rate"] == 0.0


def test_identity_pairing_uses_state_when_entity_is_shared():
    gold = v2_row(4, [intent("fire", [atom("pulse", "pending", negated=True), atom("pulse", "completes")])])
    swapped = {"intents": [intent("fire", [atom("pulse", "completes", negated=True), atom("pulse", "pending")])]}
    report, _, _ = grader.condition_report([gold], [pred_of(gold, swapped)])
    assert report["M9"]["rate"] == 0.0 and report["M2"]["rate"] == 1.0
    report, _, _ = grader.condition_report([gold], [pred_of(gold)])
    assert report["M9"]["rate"] == 1.0 and report["M10"]["rate"] == 1.0


def test_decoy_and_d2_route():
    d3 = {"row": "d1", "stratum": "D3", "decoy": True, "route": "Executable",
          "document": {"intents": [intent("fire", [])]}, "spans": [[]]}
    d2 = {"row": "d2", "stratum": "D2", "decoy": True, "route": "NoAction", "document": None, "spans": []}
    hallucinated = {"intents": [intent("fire", [atom("beacon", "lit", role="during")])]}
    report, _, _ = grader.condition_report([d3, d2], [pred_of(d3, hallucinated),
                                                     {"row": "d2", "raw": "x", "accepting": False,
                                                      "stop": "Complete", "route": "NoAction"}])
    assert report["M12|D3"]["rate"] == 0.0


def test_cli_exit_codes(tmp_path):
    gold, pred = tmp_path / "g.jsonl", tmp_path / "p.jsonl"
    gold.write_text("\n".join(json.dumps(r) for r in POP), encoding="utf-8")
    pred.write_text("\n".join(json.dumps(to_pred(r)) for r in POP), encoding="utf-8")
    assert grader.main(["--gold", str(gold), "--pred", str(pred), "--mode", "gate"]) == 0
    pred.write_text("\n".join(json.dumps(to_pred(r, state_to_entity)) for r in POP), encoding="utf-8")
    assert grader.main(["--gold", str(gold), "--pred", str(pred), "--mode", "gate"]) == 1
    assert grader.main(["--gold", str(tmp_path / "missing"), "--pred", str(pred), "--mode", "gate"]) == 4
