"""T-3103 S0 condition grader and gate grader (plan rev 15 s5.1, s0.1). Decision-bearing; registered
QUARANTINED until independently commissioned (MA/MR and MA-G/MR-G constructions).

Gold rows (JSONL, sealed store), one per row:
  {"row", "stratum", "construction", "placement", "possessive", "reserved", "family", "decoy", "arm",
   "novel_engine" (optional), "route", "document": IntentDoc v2 gold or null,
   "spans": [[{"entity": [...], "state": [...], "value": [...]}, ...], ...],
   "subject_sets", "target_sets" (optional, per intent; default: the gold string)}
Predictions (JSONL): {"row", "raw": decoded output text, "accepting": bool,
   "stop": "Complete"|"Budget"|"DeadEnd"|"StopToken", "route": optional router status}

Rules (s5.1):
- The product predicate runs first. A refused document, or one the adapter would refuse as
  TooManyConditionAtoms, scores zero.
- Intents align by index.
- Identity pairing (entity eligibility, plus state eligibility when two gold atoms share an entity) feeds
  binding, whole atom, whole set and the dropped gate.
- Field pairing (no eligibility) feeds M3 to M8 and M11.
- Field metrics are taken over every gold atom; an unpaired gold atom counts as wrong.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bars  # noqa: E402
import canon  # noqa: E402
import contract  # noqa: E402

WEAKENING = {"start": {"stop", "start_any"}, "start_any": {"stop"}, "during": {"stop", "start", "start_any"}}
GATING = ("start", "start_any", "during")
FIELDS = ("role", "entity", "state", "cmp", "value", "negated", "watch")
MAX_CONDITION_ATOMS = 4          # S0 fixture value of the adapter's positive cap (plan s4.2)


# ------------------------------------------------------------------ agreement and pairing

def _in(pred_text, field, gold_set):
    return canon.canon(pred_text, field) in {canon.canon(g, field) for g in gold_set}


def field_ok(p, g, spans, field):
    if field in ("role", "cmp", "negated", "watch"):
        return p[field] == g[field]
    if field == "value":
        if g["value"] == contract.SENTINEL or p["value"] == contract.SENTINEL:
            return p["value"] == g["value"]
        return _in(p["value"], "value", spans["value"])
    return _in(p[field], field, spans[field])


def agree(p, g, spans):
    return sum(field_ok(p, g, spans, f) for f in FIELDS)


def best_pairing(preds, golds, eligible):
    """One-to-one pairing maximizing (#pairs, #agreeing fields). Both lists hold at most
    MAX_CONDITION_ATOMS items here, so exhaustive search is exact."""
    n, m = len(preds), len(golds)
    for size in range(min(n, m), 0, -1):
        best, best_key = None, -1
        for gsel in itertools.combinations(range(m), size):
            for perm in itertools.permutations(range(n), size):
                pairs = list(zip(perm, gsel))
                if all(eligible(pi, gi) for pi, gi in pairs):
                    key = sum(agree(preds[pi], golds[gi][0], golds[gi][1]) for pi, gi in pairs)
                    if key > best_key:
                        best, best_key = pairs, key
        if best is not None:
            return best
    return []


def _sets(row, field, i, git):
    sets = row.get(field + "_sets")
    return sets[i] if sets else [git[field]]


# ------------------------------------------------------------------ per-row scoring

def score_row(row, pred, mode="v2"):
    """Per-row facts. mode: 'v2', 'g1' or 'g1g'."""
    facts = {"row": row["row"], "refused": None, "intents": [], "pred_atoms": 0, "misplaced": 0}
    doc = None
    if pred is None:
        facts["refused"] = "Missing"
    else:
        try:
            doc = contract.product_predicate(pred["raw"].encode("utf-8"), accepting=pred["accepting"],
                                             stop=pred["stop"], mode=mode)
            if mode == "v2" and any(len(it["conditions"]) > MAX_CONDITION_ATOMS for it in doc["intents"]):
                raise contract.Refusal("TooManyConditionAtoms")
            if mode != "v2":
                facts["gated_pred"] = doc["intent"].get("gated", True)
                doc = contract.g1_to_v2(doc)
        except contract.Refusal as refusal:
            facts["refused"], doc = refusal.status, None
    facts["route_ok"] = (pred or {}).get("route", row.get("route")) == row.get("route")
    gold = row.get("document")
    pdoc = doc["intents"] if doc is not None else []
    facts["pred_atoms"] = sum(len(it["conditions"]) for it in pdoc)
    if not gold:
        return facts
    gsets = row["spans"]
    shared = defaultdict(int)
    for it in gold["intents"]:
        for a in it["conditions"]:
            shared[canon.canon(a["entity"], "entity")] += 1
    for i, git in enumerate(gold["intents"]):
        pit = pdoc[i] if i < len(pdoc) else None
        golds = [(a, gsets[i][j]) for j, a in enumerate(git["conditions"])]
        preds = pit["conditions"] if pit is not None else []

        def eligible(pi, gi, golds=golds, preds=preds):
            g, s = golds[gi]
            if not _in(preds[pi]["entity"], "entity", s["entity"]):
                return False
            return shared[canon.canon(g["entity"], "entity")] <= 1 or _in(preds[pi]["state"], "state", s["state"])
        ident = {gi: pi for pi, gi in best_pairing(preds, golds, eligible)}
        fieldp = {gi: pi for pi, gi in best_pairing(preds, golds, lambda pi, gi: True)}
        ifacts = {"gold_atoms": len(golds), "pred_atoms": len(preds), "present": pit is not None,
                  "action_ok": pit is not None and pit["action"] == git["action"],
                  "subject_ok": pit is not None and _in(pit["subject"], "entity", _sets(row, "subject", i, git)),
                  "target_ok": pit is not None and _in(pit["target"], "entity", _sets(row, "target", i, git)),
                  "dep_ok": pit is not None and pit["depends_on_previous"] == git["depends_on_previous"],
                  "atoms": []}
        for gi, (g, s) in enumerate(golds):
            fp, ip = fieldp.get(gi), ident.get(gi)
            fok = {f: (fp is not None and field_ok(preds[fp], g, s, f)) for f in FIELDS}
            whole = ip is not None and all(field_ok(preds[ip], g, s, f) for f in FIELDS)
            weakened = g["role"] in GATING and not (
                fp is not None and preds[fp]["role"] not in WEAKENING[g["role"]]
                and preds[fp]["negated"] == g["negated"] and fok["cmp"] and fok["value"])
            ifacts["atoms"].append({"gold": g, "field_ok": fok, "bound": ip is not None, "whole": whole,
                                    "weakened": weakened})
        ifacts["whole_set"] = bool(golds) and all(a["whole"] for a in ifacts["atoms"]) and len(preds) == len(golds)
        ifacts["gated_weakened"] = any(a["weakened"] for a in ifacts["atoms"])
        ifacts["has_gating"] = any(g["role"] in GATING for g, _ in golds)
        for pi, p in enumerate(preds):          # misplacement
            if pi in ident.values():
                continue
            if any(_in(p["entity"], "entity", gsets[k][j]["entity"])
                   for k, other in enumerate(gold["intents"]) if k != i
                   for j in range(len(other["conditions"]))):
                facts["misplaced"] += 1
        facts["intents"].append(ifacts)
    return facts


# ------------------------------------------------------------------ condition grader (S4)

def frac(num, den):
    return {"num": num, "den": den, "rate": (num / den) if den else None}


GROUPS = {"M5_per_cmp": "M5|", "M10_per_F": "M10_F|", "M10_per_placement": "M10_placement|",
          "M10_per_B": "M10_B|", "M10_per_construction": "M10_construction|", "M12_per_family": "M12|"}
ALIAS = {"M6_F5": "M6|F5", "M6_F6": "M6|F6"}


def condition_report(gold_rows, preds):
    """Section 5.1 metrics and the CB verdict over one population. CB-6 needs the reference resolver."""
    by = {p["row"]: p for p in preds}
    m = defaultdict(lambda: [0, 0])

    def add(key, ok):
        m[key][0] += int(ok)
        m[key][1] += 1
    families = defaultdict(list)
    pred_atoms = misplaced = 0
    for r in gold_rows:
        f = score_row(r, by.get(r["row"]), "v2")
        pred_atoms += f["pred_atoms"]
        misplaced += f["misplaced"]
        if r.get("decoy"):
            fam = r["stratum"]
            add(("M12", fam), f["refused"] is None and f["pred_atoms"] == 0 and (fam != "D2" or f["route_ok"]))
            continue
        if not r.get("document"):
            continue
        for it in f["intents"]:
            if it["gold_atoms"]:
                add("M1", it["present"] and it["pred_atoms"] > 0)
                add("M10", it["whole_set"])
                if r["stratum"].startswith("F"):
                    add(("M10_F", r["stratum"]), it["whole_set"])
                    if r.get("placement") not in (None, "-"):
                        add(("M10_placement", r["placement"]), it["whole_set"])
                    if r.get("construction") not in (None, "-"):
                        add(("M10_construction", r["stratum"] + "/" + r["construction"]), it["whole_set"])
                if r["stratum"].startswith("B"):
                    add(("M10_B", r["stratum"]), it["whole_set"])
            else:
                add("M1s", it["present"] and it["pred_atoms"] > 0)
            if it["has_gating"]:
                add("M11", it["gated_weakened"])
            for a in it["atoms"]:
                g, fok = a["gold"], a["field_ok"]
                add("M2", a["bound"])
                add("M9", a["whole"])
                for key, field in (("M3", "entity"), ("M4", "state"), ("M6", "negated"), ("M7", "role"),
                                   ("M8", "watch")):
                    add(key, fok[field])
                if r["stratum"] in ("F5", "F6"):
                    add(("M6", r["stratum"]), fok["negated"])
                if r.get("possessive") == "yes":
                    add("M3_possessive", fok["entity"])
                if r.get("reserved"):
                    add("M3_unseen", fok["entity"])
                    add("M4_unseen", fok["state"])
                if g["cmp"] != "none":
                    add("M5", fok["cmp"] and fok["value"])
                    add(("M5", g["cmp"]), fok["cmp"] and fok["value"])
                else:
                    add("M5_sentinel", fok["cmp"] and fok["value"])
        if r.get("family"):
            families[r["family"]].append(all(it["whole_set"] for it in f["intents"] if it["gold_atoms"]))
    report = {k if isinstance(k, str) else "|".join(map(str, k)): frac(*v) for k, v in m.items()}
    report["misplacement"] = frac(misplaced, pred_atoms)
    report["M13"] = frac(sum(all(v) for v in families.values()), len(families))
    verdict, failures = {}, []
    for bar, rules in bars.CB.items():
        ok = True
        for metric, cmp_, val in rules:
            if metric in GROUPS:
                keys = sorted(k for k in report if k.startswith(GROUPS[metric]))
                if not keys:
                    ok = False
                    failures.append(f"{bar} {metric}: no population")
            else:
                keys = [ALIAS.get(metric, metric)]
            for k in keys:
                entry = report.get(k)
                if entry is None or entry.get("rate") is None:
                    ok = False
                    failures.append(f"{bar} {k}: not evaluable here")
                    continue
                rate = entry["rate"]
                good = rate >= val if cmp_ == ">=" else (rate <= val if cmp_ == "<=" else rate == val)
                if not good:
                    ok = False
                    failures.append(f"{bar} {k}: {entry['num']}/{entry['den']} vs {cmp_} {val}")
        verdict[bar] = ok
    unseen, all_atoms = m["M3_unseen"][1], m["M3"][1]
    if all_atoms and unseen < bars.CB_UNSEEN_SHARE_MIN * all_atoms:
        verdict["CB-3"] = False
        failures.append(f"CB-3 unseen-vocabulary atoms {unseen}/{all_atoms} below 30%")
    return report, verdict, failures


# ------------------------------------------------------------------ gate grader (S0g)

def gate_report(gold_rows, preds, preds_g1g=None):
    """EGB over the gate split from the G1 pass. With a G1g pass (detection arm, D-SLM8098 pending), M1
    recall over the condition rows and M1s over the arm's ungated and decoy rows join the decisive set."""
    by = {p["row"]: p for p in preds}
    cond = [r for r in gold_rows if r.get("document") and not r.get("arm")]
    arm = [r for r in gold_rows if r.get("arm")]
    c = defaultdict(lambda: [0, 0])

    def add(key, ok):
        c[key][0] += int(ok)
        c[key][1] += 1
    novel = defaultdict(lambda: [0, 0])
    for r in cond:
        f = score_row(r, by.get(r["row"]), "g1")
        it, g = f["intents"][0], r["document"]["intents"][0]["conditions"][0]
        a = it["atoms"][0]
        refused = f["refused"] is not None
        fok = a["field_ok"]
        vals = {"action": not refused and it["action_ok"], "subject": not refused and it["subject_ok"],
                "target": not refused and it["target_ok"], "M3": fok["entity"], "M4": fok["state"],
                "M9": a["whole"]}
        for k, v in vals.items():
            add(k, v)
            if r.get("novel_engine"):
                novel[k][0] += int(v)
                novel[k][1] += 1
        add("W", all(vals[k] for k in ("action", "subject", "target")) and vals["M9"])
        if r.get("reserved"):
            add("M3_unseen", fok["entity"])
            add("M4_unseen", fok["state"])
        if g["role"] in GATING:
            add("M11g", a["weakened"])
        if r["stratum"] in ("F5", "F6"):
            add("M6_F5F6", fok["negated"])
        if r["stratum"] in ("F6", "F7"):
            add("M7_F6F7", fok["role"])
        if g["cmp"] != "none":
            add("M5", fok["cmp"] and fok["value"])
        for key, field in (("M6", "negated"), ("M7", "role"), ("M8", "watch")):
            add(key, fok[field])
    if preds_g1g is not None:
        by2 = {p["row"]: p for p in preds_g1g}
        for r in cond:
            f = score_row(r, by2.get(r["row"]), "g1g")
            add("M1_recall", f["refused"] is None and f.get("gated_pred") is True)
        for r in arm:
            f = score_row(r, by2.get(r["row"]), "g1g")
            add("M1s", f["refused"] is not None or f.get("gated_pred") is True)
    cut = bars.egb_cutoffs({k: v[1] for k, v in c.items()}, arm=preds_g1g is not None)
    decisive = {k: {"count": c[k][0], **cut[k], "pass": bars.passes(cut[k], c[k][0])} for k in cut}
    flags = [k for k, line in bars.EGB_DIAGNOSTIC.items() if c[k][1] and c[k][0] / c[k][1] < line]
    nominal = {name: n for name, _, n, _, _ in bars.EGB_DECISIVE}
    for k, (num, den) in sorted(novel.items()):   # CA14-N1: novel-engine half, as diagnostics
        if den and num / den < nominal[k]:
            flags.append(f"novel_engine:{k}")
    if not all(d["pass"] for d in decisive.values()):
        verdict = "NO-GO"
    elif flags:
        verdict = "FLAGGED-GO"
    else:
        verdict = "GO"
    return {"verdict": verdict, "decisive": decisive, "flags": flags,
            "diagnostic": {k: frac(*c[k]) for k in bars.EGB_DIAGNOSTIC}, "W": frac(*c["W"]),
            "novel_engine": {k: frac(*v) for k, v in novel.items()}}


# ------------------------------------------------------------------ command line

def _load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    """Exit 0 = pass/GO; 1 = fail/NO-GO; 2 = FLAGGED-GO; 4 = construction error (never a refusal)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mode", choices=("v2", "gate"), default="v2")
    ap.add_argument("--pred-g1g", help="gate mode: the detection arm's G1g pass, only if the arm is adopted")
    args = ap.parse_args(argv)
    try:
        gold, pred = _load(args.gold), _load(args.pred)
        g1g = _load(args.pred_g1g) if args.pred_g1g else None
        if args.mode == "v2":
            report, verdict, failures = condition_report(gold, pred)
        else:
            out = gate_report(gold, pred, g1g)
    except Exception as error:  # noqa: BLE001 - malformed input is a construction error, not a refusal
        print(f"CONSTRUCTION ERROR: {type(error).__name__}: {error}")
        return 4
    if args.mode == "v2":
        print(json.dumps({"CB": verdict, "failures": failures}, sort_keys=True))
        if all(ok for bar, ok in verdict.items() if bar != "CB-6"):
            print("VERDICT CB-PASS (CB-6 owed: reference resolver)")
            return 0
        print("VERDICT CB-FAIL")
        return 1
    print(json.dumps({k: v for k, v in out.items() if k != "novel_engine"}, sort_keys=True))
    print("VERDICT " + out["verdict"])
    return {"GO": 0, "NO-GO": 1, "FLAGGED-GO": 2}[out["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
