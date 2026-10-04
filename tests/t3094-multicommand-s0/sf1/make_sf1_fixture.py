"""Writes sf1-fixture.json and encoder-e1-reference.json (T-3103, plan rev 7 section 6 row 12).

The SF1 batch is re-authored for plan rev 15 s6 row 12: an independently specified, ordered three-command
batch of rev-9 resolved commands (action, subject, target, trigger as ResolvedCondition, after) with one
ungated command, one with a two-entry start_all condition, and one durative hold with a numeric stop_any
entry whose value is the exact rational 75/2. It is hand-specified, not resolver output.
The encoder reference vectors (encoder-e1-reference.json, D-SLM8077) stay on their frozen legacy inputs.

Cell SF1-ORTHO: three orthogonal unit rows in D = 3 (padded stride 4), deliberately permuted
against the dimension order so a harness confusing rows with lanes fails. Its rows, probes,
expected indices and scores are literal constants below, not computed.
Cell SF1-E1: the same commands through encoder E1 (D = 64), each probe its own row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvb_e1  # noqa: E402

HERE = Path(__file__).resolve().parent
ONE, ZERO = "3f800000", "00000000"

BATCH_ID = "t3103-sf1-batch-0002"
CATALOGUE, STATE = "act-sf1-1+obs-sf1-1", "state-sf1-1"
EMPTY = {"start_all": [], "start_any": [], "stop_any": [], "during_all": []}


def entry(instance, predicate, kind, cmp="none", value=None):
    return {"instance": instance, "predicate": predicate, "kind": kind, "cmp": cmp, "value": value,
            "negated": False, "binding": {"mode": "passive", "ref": None}}


COMMANDS = [
    {"action": "move", "subject": "alpha squad", "target": "north ridge", "trigger": dict(EMPTY), "after": []},
    {"action": "fire", "subject": "bravo squad", "target": "east gate",
     "trigger": {**EMPTY, "start_all": [entry("beacon-1", "Beacon.Lit", "state"),
                                         entry("gate-east", "Gate.Open", "state")]}, "after": []},
    {"action": "hold", "subject": "charlie squad", "target": "river ford",
     "trigger": {**EMPTY, "stop_any": [entry("charlie squad", "Unit.Health", "numeric", "below",
                                              {"num": 75, "den": 2})]}, "after": []},
]
PERMUTED = {**COMMANDS[1], "trigger": {**EMPTY, "start_all": list(reversed(COMMANDS[1]["trigger"]["start_all"]))}}
STALE_LIVE_CATALOGUE = "act-sf1-1+obs-sf1-2"

LEGACY_COMMANDS = [
    {"action": "move", "subject": "alpha squad", "target": "north ridge", "trigger": "immediate",
     "depends_on_previous": False},
    {"action": "scan", "subject": "bravo squad", "target": "east gate", "trigger": "immediate",
     "depends_on_previous": False},
    {"action": "hold", "subject": "charlie squad", "target": "river ford", "trigger": "immediate",
     "depends_on_previous": False},
]

ORTHO = {
    "name": "SF1-ORTHO", "dims": 3, "padded_dims": 4, "capacity": 3,
    # ordinal 0 -> e2, ordinal 1 -> e0, ordinal 2 -> e1
    "rows": [[ZERO, ZERO, ONE], [ONE, ZERO, ZERO], [ZERO, ONE, ZERO]],
    "probes": [[ZERO, ZERO, ONE], [ONE, ZERO, ZERO], [ZERO, ONE, ZERO]],
    "expected_out_index": [0, 1, 2],
    "expected_top1": [{"index": 0, "score_bits": ONE}, {"index": 1, "score_bits": ONE},
                      {"index": 2, "score_bits": ONE}],
    "expected_row_readback": [[ZERO, ZERO, ONE, ZERO], [ONE, ZERO, ZERO, ZERO], [ZERO, ONE, ZERO, ZERO]],
    "expected_tombstone_words": [0],
}

ENCODER_CASES = [
    {"case": "sf1-ordinal-0", "command": LEGACY_COMMANDS[0]},
    {"case": "sf1-ordinal-1", "command": LEGACY_COMMANDS[1]},
    {"case": "sf1-ordinal-2", "command": LEGACY_COMMANDS[2]},
    {"case": "key-order-insensitive", "command": dict(reversed(list(LEGACY_COMMANDS[0].items())))},
    {"case": "nfd-cafe", "command": {**LEGACY_COMMANDS[1], "target": "cafe\u0301"}},
    {"case": "nfc-cafe", "command": {**LEGACY_COMMANDS[1], "target": "caf\u00e9"}},
    {"case": "dependency-true", "command": {**LEGACY_COMMANDS[2], "depends_on_previous": True}},
    {"case": "structured-trigger-mechanism", "command": {**LEGACY_COMMANDS[0], "trigger": {"b": [1, None], "a": "x"}},
     "note": "encoder mechanism input only; not condition gold"},
    {"case": "control-and-quote", "command": {**LEGACY_COMMANDS[0], "target": "line\n\"two\"\t\u0001"}},
]
ENCODER_REFUSALS = [
    {"case": "float-value", "command": {**LEGACY_COMMANDS[0], "depends_on_previous": 0.5},
     "status": "EncoderUnsupportedValue"},
    {"case": "empty-object", "command": {}, "status": "EncoderUnsupportedValue"},
]


def main() -> None:
    e1_rows = [["%08x" % b for b in rvb_e1.encode_bits(c)] for c in COMMANDS]
    e1 = {"name": "SF1-E1", "dims": rvb_e1.DIMS, "padded_dims": 64, "capacity": 3, "rows": e1_rows,
          "probes": e1_rows, "expected_out_index": [0, 1, 2], "expected_top1_index": [0, 1, 2],
          "expected_tombstone_words": [0]}
    fixture = {
        "fixture": "T-3103 S0-SF1", "plan": "Claude/Plans/superslm-multicommand-array/plan.md rev 7 s4.2, s6 row 12",
        "superfaiss": {"repo": "D:/SuperFAISS", "ref": "master",
                       "commit": "cd42dc228244d1db54864ef69b5ee5f07d16fc2d"},
        "batch": {"batch_id": BATCH_ID, "catalogue_version": CATALOGUE, "state_version": STATE,
                  "encoder_version": rvb_e1.ENCODER_VERSION, "commands": COMMANDS},
        "expected_host_map": [{"batch_id": BATCH_ID, "ordinal": i, "command": c,
                               "catalogue_version": CATALOGUE, "state_version": STATE}
                              for i, c in enumerate(COMMANDS)],
        "permuted_entry_command": PERMUTED, "stale_live_catalogue_version": STALE_LIVE_CATALOGUE,
        "metric": "dot", "quant": "float32", "query_k": 1,
        "cells": [ORTHO, e1],
        "superfaiss_observations": [
            {"name": "OBS-WRONG-DIMS", "expect_status": "DimsMismatch", "expect_count_after": 0},
            {"name": "OBS-NONFINITE", "expect_status": "NonFiniteQuery", "expect_count_after": 0},
            {"name": "OBS-PARTIAL", "expect_statuses": ["Ok", "NonFiniteQuery"], "expect_count_after": 1,
             "why": "Append publishes row by row; a later failure leaves the earlier row visible, so the "
                    "bridge must stage a fresh bank and destroy it on any failure"},
            {"name": "OBS-OVER-CAPACITY", "expect_statuses": ["Ok", "Ok", "Ok", "OutOfMemory"],
             "expect_count_after": 3},
        ],
    }
    (HERE / "sf1-fixture.json").write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8", newline="\n")
    reference = {"encoder": rvb_e1.ENCODER_VERSION, "dims": rvb_e1.DIMS, "cases": [], "refusals": []}
    for case in ENCODER_CASES:
        reference["cases"].append({**case, "bits": ["%08x" % b for b in rvb_e1.encode_bits(case["command"])]})
    for case in ENCODER_REFUSALS:
        try:
            rvb_e1.encode_bits(case["command"])
            raise SystemExit(f"refusal case {case['case']} encoded")
        except rvb_e1.EncoderError as error:
            assert error.status == case["status"], case
        reference["refusals"].append(case)
    (HERE / "encoder-e1-reference.json").write_text(json.dumps(reference, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
