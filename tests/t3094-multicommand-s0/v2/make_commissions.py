"""T-3103 S0 eval-writer commissions (plan rev 15 s5.2 writer process, s5.3 seal, CA14-2, CA14-N1).

Reads only the eval part of the conductor's partition (D-SLM8169) and the sealed reserved list. Writes
into the sealed store only:
  commissions/EW1.jsonl and commissions/EW2.jsonl: one commission per assignment unit, carrying each
    slot's row ID and its plain-language meaning (no field names, example sentences or fixed phrasings);
  commissions/manifest.json: hashes and counts.
It prints counts and hashes only.

Writer slots:
- At least two writers (s5.2 item 2). EW1 and EW2 each write 40-60% of every F, B and D cell.
- EW2 must be an engine family that wrote no T or V row and does not run the paraphrase generator, and it
  writes at least half of the gate split (D-SLM8155, CA14-N1). The conductor binds each slot to a seat and
  records the engine.
- A unit (paraphrase family, published pair or single) stays with one writer.
- Sessions hold 10 to 40 rows within one split, with one speaker each.
- The probe reserve RE is written by the same writers before the probe first runs, and is never inspected
  (CA14-2).

Usage: py -3 make_commissions.py --eval-part <eval-part.json> --reserved <sealed list> --out <sealed dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ACTION = {"fire": "fire on a target", "move": "move to a place", "scan": "scan a target or area",
          "hold": "hold a position", "wait": "stand and watch something"}
PLACEMENT = {
    "fronted": "the condition comes before the command",
    "postposed": "the condition comes after the command",
    "mid_subject_verb": "the condition sits between the addressed unit and the verb",
    "mid_verb_object": "the condition sits between the verb and its target",
    "mid_vocative": "the condition comes right after the unit is addressed by name, before the rest",
    "anaphoric": "the command is given first; a separate second sentence states the condition and refers back to it",
}
ETYPE = {"structure": "a building or fixed structure", "unit": "a vehicle, machine or unit that can move or act",
         "signal": "a signal, light, sound or marker", "zone": "an area or site",
         "resource": "a store or supply of something", "task": "a task another squad is already carrying out",
         "-": "something in the world"}
WATCH = {"auto": "", "reuse": " The player explicitly asks to rely on the watch already being kept on that thing.",
         "new": " The player explicitly asks for a new watch to be set up on that thing."}
F_MEANING = {
    ("F1",): "The action may start only once an observed thing of {etype} is in a state the player names "
             "(a lasting state that may already be true). Introduce the condition with '{con}'.",
    ("F2",): "The action may start only once an event the player names happens to an observed thing of {etype}. "
             "Introduce the condition with '{con}'.",
    ("F3",): "The action may start only once a measured property of an observed thing of {etype} meets a quantity, "
             "using the comparison '{con}' and a quantity said in words or digits.",
    ("F4",): "The action may start only once the number of things of {etype} that are in a state meets a count, "
             "using the comparison '{con}'.",
    ("F5",): "The action is wanted only while a condition about an observed thing of {etype} does NOT hold. "
             "Express it with '{con}'.",
    ("F6",): "The player holds the action back until something happens to an observed thing of {etype}. "
             "Express it with '{con}'.",
    ("F7",): "The ongoing action continues and ends when something holds of an observed thing of {etype}. "
             "Express it with '{con}'.",
    ("F8",): "The ongoing action continues only while something holds of an observed thing of {etype}. "
             "Express it with '{con}'.",
    ("F9",): "The action may start only when {atoms} separate conditions about observed things all hold, joined "
             "with '{con}'.",
    ("F10",): "The action may start when any one of {atoms} alternative conditions about observed things holds, "
              "offered with '{con}'.",
    ("F11",): "The action may start only once a task another squad is already carrying out is finished.",
    ("F12",): "Two squads get orders in the same request; the second squad's action may start only on an event "
              "the first squad causes.",
}
CON = {"as_soon_as": "as soon as", "the_moment": "the moment", "drops_below": "drops below", "at_least": "at least",
       "or_less": "or less", "fewer_than": "fewer than", "none_of": "none of", "more_than": "more than",
       "if_not": "if not", "neg_action_while": "a forbidden action plus 'while'", "dont_until": "don't ... until",
       "hold_fire_till": "hold fire till", "not_before": "not before", "never_until": "never ... until",
       "dont_before": "don't ... before", "keep_till": "keep ... till", "as_long_as": "as long as",
       "and_also": "and also", "either": "either ... or"}
B_MEANING = {"B1": "Two to four actions in one request; only the last one waits on an observed condition.",
             "B2": "Two to four actions in one request; only the first or a middle one waits on an observed "
                   "condition.",
             "B3": "One condition, said before two or more coordinated actions, gates all of them.",
             "B4": "Two actions, each with its own different condition.",
             "B5": "Several actions chained with 'then', some of which wait on observed conditions."}
D_MEANING = {
    "D1": "Uses 'while' to give one unit an action during another friendly unit's action; nothing is a condition.",
    "D2": "A question or report that uses 'when' or 'once' and asks for no action.",
    "D3": "A plain command whose target is described by a state word (the state describes the target; it is "
          "not a condition).",
    "D4": "Two requested actions ordered with 'before' or 'after'; the order is between the actions, no observed "
          "condition.",
    "D5": "A command with a purpose or result clause ('so that ...', 'to make ...'); nothing gates the action.",
    "D6": "A command plus a separate statement of an observation as plain fact; the fact does not gate the action.",
    "D7": "Uses 'once' to mean a single time, not a condition.",
    "D8": "Uses 'if' as 'whether', to say what to check, not a condition.",
    "D9": "Uses 'as soon as possible' or 'when you can'; there is no observed thing.",
    "D10": "Uses 'until' inside a fixed phrase with no observed thing ('until further notice' style).",
    "D11": "Uses 'as long as' about time or effort, with no observed thing.",
}
S_MEANING = {"S1": "A condition that needs two separate groups of alternatives.",
             "S2": "A condition on a condition (one observation only counts after another).",
             "S3": "A condition marked as a repeat or next occurrence.",
             "S4": "A threshold whose boundary inclusiveness matters (or less / below).",
             "S5": "A durative action until both of two things hold.",
             "S6": "A non-negated action before an observed event (a deadline).",
             "S7": "A repeated punctual action until a condition."}
GEN_MEANING = {
    "conjunction": "Two or more actions joined in one request.",
    "punctuation_free": "A transcript-style request with no punctuation.",
    "correction": "The player corrects part of the request partway through.",
    "shared_argument": "One unit or target shared across several actions.",
    "pronoun": "Later actions refer back to an earlier unit or target by pronoun.",
    "nested_condition": "An action gated on something observed in the world, said naturally.",
    "parallel": "Different units get different simultaneous orders.",
    "cancellation": "Revokes an earlier order and asks for nothing new.",
    "zero_action": "Talk, a question or a remark that asks for no action.",
    "above_cap": "Asks for many actions at once, more than four.",
}
CAP_MEANING = {"four_intents": "Exactly four actions: a move, a scan, a hold, and a fire that waits on an observed "
                               "condition.",
               "five_requested": "The four actions above plus one more separately requested action.",
               "inserted_wait": "Exactly four actions: a move, a scan, a hold and a fire whose start waits on an "
                                "observed condition.",
               "two_waits": "Three actions, one of which may start only when two observed conditions both hold."}
PC2_MEANING = {"simple": "Fire on a target once a cue the player names happens.",
               "dependency": "Fire on a target once a scan another unit is already running is finished.",
               "ordering": "Fire on a target once a move another unit is already making is finished."}


def _con(c):
    return CON.get(c, c.replace("_", " "))


def slot_meaning(r, reserved_pair):
    s = r["stratum"]
    if s.startswith("F"):
        text = "Request: " + ACTION.get(r["action"], r["action"]) + ", addressed to a unit the player names. "
        text += F_MEANING[(s,)].format(etype=ETYPE.get(r["entity_type"], "something"), con=_con(r["construction"]),
                                       atoms=r["atoms"])
        text += WATCH.get(r["watch"], "")
        if r["placement"] not in ("-", None):
            text += " Placement: " + PLACEMENT[r["placement"]] + "."
        if r["possessive"] == "yes":
            text += " Name the observed thing with a possessive (whose it is, or your/my)."
    elif s.startswith("B"):
        text = B_MEANING[s] + f" Conditions in total: {r['atoms']}." + WATCH.get(r["watch"], "")
    elif s.startswith("D"):
        text = D_MEANING[s]
    elif s.startswith("S"):
        text = S_MEANING[s]
    elif s.startswith("GEN-") or s.startswith("UC-"):
        text = GEN_MEANING[s.split("-", 1)[1]]
    elif s.startswith("CAP-"):
        text = CAP_MEANING[r["construction"]]
    elif s.startswith("PC2-"):
        text = PC2_MEANING[s[4:]] + WATCH[r["watch"]]
    elif s == "N-ambiguous_executable":
        text = "Begins like a cancellation but ends with a plain new request."
    elif s == "A0-plain":
        text = "Request: " + ACTION.get(r["action"], r["action"]) + ", with no condition."
    else:
        raise ValueError(s)
    if reserved_pair:
        text += f" Use the word '{reserved_pair[0]}' for the observed thing and '{reserved_pair[1]}' for what must " \
                "hold of it (any natural form)."
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-part", required=True)
    ap.add_argument("--reserved", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    raw = Path(args.eval_part).read_bytes()
    part = json.loads(raw)
    sealed = json.loads(Path(args.reserved).read_text(encoding="utf-8"))
    cols = part["columns"]
    rows = [dict(zip(cols, v)) for v in part["rows"]]
    units = defaultdict(list)
    for r in rows:
        units[r["unit"]].append(r)
    for u in units.values():
        u.sort(key=lambda r: r["member"])
    # reserved words: balanced round-robin over type pools and states, per unit
    pools = sealed["entity_types"]
    states = sealed["state_lemmas"]
    turn = Counter()
    st_turn = 0
    assign = {}
    for uid in sorted(units):
        first = units[uid][0]
        if any(r["reserved"] for r in units[uid]):
            et = first["entity_type"] if first["entity_type"] in pools else "structure"
            pool = pools[et]
            ent = pool[turn[et] % len(pool)]
            turn[et] += 1
            assign[uid] = (ent, states[(st_turn * 7) % len(states)])
            st_turn += 1
    # writer slots: greedy by rows. Each unit goes to the writer with fewer rows in its stratum (pooled over the
    # eval splits, the s5.2 cell), then fewer in its split cell. Gate ties go to EW2, which must write at least
    # half of the gate split.
    writer = {}
    cell_rows = defaultdict(Counter)
    split_rows = defaultdict(Counter)
    order = sorted(units, key=lambda u: (units[u][0]["split"] != "G", units[u][0]["stratum"], -len(units[u]), u))
    for uid in order:
        rs = units[uid]
        stratum, split = rs[0]["stratum"], rs[0]["split"]
        gate_bias = 0 if split == "G" else 1

        def load(w):
            return (cell_rows[stratum][w], split_rows[(split, stratum)][w],
                    (0 if w == "EW2" else 1) if gate_bias == 0 else (0 if w == "EW1" else 1))
        w = min(("EW1", "EW2"), key=load)
        writer[uid] = w
        cell_rows[stratum][w] += len(rs)
        split_rows[(split, stratum)][w] += len(rs)
    out = Path(args.out) / "commissions"
    out.mkdir(parents=True, exist_ok=True)
    per_writer = defaultdict(list)
    for uid in sorted(units, key=lambda u: (units[u][0]["split"], units[u][0]["stratum"], u)):
        per_writer[writer[uid]].append(uid)
    files, stats = {}, {}
    for w, uids in sorted(per_writer.items()):
        lines, session, n_in_session, sess_split = [], 0, 0, None
        for uid in uids:
            rs = units[uid]
            split = rs[0]["split"]
            if split != sess_split or n_in_session + len(rs) > 30:
                session += 1
                n_in_session, sess_split = 0, split
            n_in_session += len(rs)
            pair = assign.get(uid)
            kind = "family" if len(rs) > 1 and len({r["stratum"] for r in rs}) == 1 and \
                not rs[0]["stratum"].startswith("PC2") else ("pair" if len(rs) > 1 else "single")
            lines.append({"unit": uid, "writer": w, "split": split, "session": f"{w}-s{session:03d}",
                          "speaker": f"{w}-p{session:03d}", "kind": kind,
                          "instruction": ("Write one independently worded utterance per slot; every slot of this "
                                          "unit expresses the same meaning in different words." if kind == "family"
                                          else "Write one utterance per slot; the slots share their observed-thing "
                                          "and state words and differ only as their meanings say." if kind == "pair"
                                          else "Write one utterance."),
                          "reserve_pool": split == "RE",
                          "slots": [{"row": r["row"], "meaning": slot_meaning(r, pair)} for r in rs]})
        data = "".join(json.dumps(l, ensure_ascii=False, sort_keys=True) + "\n" for l in lines).encode()
        (out / f"{w}.jsonl").write_bytes(data)
        files[f"{w}.jsonl"] = hashlib.sha256(data).hexdigest()
        stats[w] = {"units": len(uids), "rows": sum(len(units[u]) for u in uids), "sessions": session}
    gate_rows = Counter(writer[u] for u in units for r in units[u] if r["split"] == "G")
    manifest = {"eval_part_sha256": hashlib.sha256(raw).hexdigest(), "files": files, "writers": stats,
                "gate_rows_by_writer": dict(gate_rows), "reserved_units": len(assign)}
    shares = {k: c["EW1"] / sum(c.values()) for k, c in cell_rows.items() if k[0] in "FBD"}
    core_gate = Counter(writer[u] for u in units for r in units[u] if r["split"] == "G" and not r["arm"])
    manifest.update({"gate_core_rows_by_writer": dict(core_gate),
                     "EW1_share_range_F_B_D_cells": [round(min(shares.values()), 3), round(max(shares.values()), 3)]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"writers": stats, "gate_rows_by_writer": dict(gate_rows),
                      "gate_core_rows_by_writer": dict(core_gate), "reserved_units": len(assign),
                      "EW1_share_range_F_B_D_cells": manifest["EW1_share_range_F_B_D_cells"],
                      "manifest_files": files}, sort_keys=True))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
