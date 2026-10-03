"""Writes sf1-fixture.json and encoder-e1-reference.json (T-3103, plan rev 7 section 6 row 12).

The SF1 batch is a hand-specified resolved batch: three immediate, independent commands. It
carries no condition or cue (D-SLM8075 holds condition-bearing gold), and it is not the output
of the game-side resolver, whose reference is held for the trigger revision.

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

BATCH_ID = "t3103-sf1-batch-0001"
CATALOGUE, STATE = "catalogue-sf1-v1", "state-sf1-v1"
COMMANDS = [
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
    {"case": "sf1-ordinal-0", "command": COMMANDS[0]},
    {"case": "sf1-ordinal-1", "command": COMMANDS[1]},
    {"case": "sf1-ordinal-2", "command": COMMANDS[2]},
    {"case": "key-order-insensitive", "command": dict(reversed(list(COMMANDS[0].items())))},
    {"case": "nfd-cafe", "command": {**COMMANDS[1], "target": "cafe\u0301"}},
    {"case": "nfc-cafe", "command": {**COMMANDS[1], "target": "caf\u00e9"}},
    {"case": "dependency-true", "command": {**COMMANDS[2], "depends_on_previous": True}},
    {"case": "structured-trigger-mechanism", "command": {**COMMANDS[0], "trigger": {"b": [1, None], "a": "x"}},
     "note": "encoder mechanism input only; not condition gold"},
    {"case": "control-and-quote", "command": {**COMMANDS[0], "target": "line\n\"two\"\t\u0001"}},
]
ENCODER_REFUSALS = [
    {"case": "float-value", "command": {**COMMANDS[0], "depends_on_previous": 0.5},
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
