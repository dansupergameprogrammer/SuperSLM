"""T-3103 S0-SF1 and ResolvedVectorBatch v1 / encoder E1 cells (plan rev 7 s4.2, s6 row 12).

Reference self-controls pass now; they prove the references compute, not that any product
path works. The real-SuperFAISS receipt cell is red until run_sf1.ps1 executes against master
cd42dc2; while the box timing lock exists that run is owed and the cell fails for that reason.
"""
from __future__ import annotations

import copy
import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rvb_e1  # noqa: E402
import sf1_check  # noqa: E402

FIXTURE = json.loads((HERE / "sf1-fixture.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((HERE / "encoder-e1-reference.json").read_text(encoding="utf-8"))
RECEIPT = Path(os.environ.get("T3103_SF1_RECEIPT", r"D:\_artifacts\SuperSLM\t3103\sf1-receipt.json"))
LOCK = Path(r"D:\_scratch\chains\box-timing.lock")
COMMANDS = FIXTURE["batch"]["commands"]


# ------------------------------------------------------------------ encoder E1 (frozen)

@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["case"])
def test_e1_reference_vectors_are_frozen(case):
    assert ["%08x" % b for b in rvb_e1.encode_bits(case["command"])] == case["bits"]


def test_e1_shape_and_equivalences():
    by = {c["case"]: c["bits"] for c in REFERENCE["cases"]}
    assert REFERENCE["encoder"] == "E1" and REFERENCE["dims"] == 64
    for bits in by.values():
        values = rvb_e1.bits_to_floats(int(b, 16) for b in bits)
        assert len(values) == 64 and all(math.isfinite(v) for v in values)
        assert abs(math.sqrt(sum(v * v for v in values)) - 1.0) < 1e-6
    assert by["key-order-insensitive"] == by["sf1-ordinal-0"]
    assert by["nfd-cafe"] == by["nfc-cafe"]
    assert by["dependency-true"] != by["sf1-ordinal-2"]
    assert by["structured-trigger-mechanism"] != by["sf1-ordinal-0"]
    assert len({tuple(by["sf1-ordinal-%d" % i]) for i in range(3)}) == 3


@pytest.mark.parametrize("case", REFERENCE["refusals"], ids=lambda c: c["case"])
def test_e1_refusals(case):
    with pytest.raises(rvb_e1.EncoderError) as caught:
        rvb_e1.encode_bits(case["command"])
    assert caught.value.status == case["status"]


# ---------------------------------------------------- ResolvedVectorBatch v1 publish rule

def _healthy():
    batch = rvb_e1.build_batch(FIXTURE["batch"]["batch_id"], FIXTURE["batch"]["catalogue_version"],
                               FIXTURE["batch"]["state_version"], COMMANDS)
    return batch, list(COMMANDS)


def _status(batch, commands, cap=4, live=(FIXTURE["batch"]["catalogue_version"], FIXTURE["batch"]["state_version"]), appends=None):
    return rvb_e1.publish_status(batch, commands, cap=cap, live_catalogue_version=live[0],
                                 live_state_version=live[1], append_statuses=appends)


def test_rvb_healthy_control_publishes():
    batch, commands = _healthy()
    assert batch["dims"] == 64 and batch["metric"] == "dot" and batch["quant"] == "float32"
    assert batch["channels"] == 0 and len(batch["rows"]) == 3
    assert _status(batch, commands, appends=["Ok"] * 3) == "Published"


def _mutate(kind):
    batch, commands = _healthy()
    cap, live, appends = 4, (FIXTURE["batch"]["catalogue_version"], FIXTURE["batch"]["state_version"]), ["Ok"] * 3
    if kind == "dropped_row":
        batch["rows"].pop()
    elif kind == "swapped_rows":
        batch["rows"][0], batch["rows"][1] = batch["rows"][1], batch["rows"][0]
    elif kind == "duplicated_map":
        batch["host_map"][2] = copy.deepcopy(batch["host_map"][1])
    elif kind == "swapped_map_commands":
        batch["host_map"][0]["command"], batch["host_map"][1]["command"] = COMMANDS[1], COMMANDS[0]
    elif kind == "wrong_dimension":
        batch["rows"][1] = batch["rows"][1][:-1]
    elif kind == "nonfinite_row":
        batch["rows"][2][5] = struct.unpack("<I", struct.pack("<f", float("nan")))[0]
    elif kind == "partial_append":
        appends = ["Ok", "NonFiniteQuery", "Ok"]
    elif kind == "stale_state":
        live = (FIXTURE["batch"]["catalogue_version"], "state-sf1-2")
    elif kind == "over_cap_five_resolved":
        extra = [{**COMMANDS[0], "subject": "delta squad"}, {**COMMANDS[1], "subject": "echo squad"}]
        commands = commands + extra
        batch = rvb_e1.build_batch(batch["batch_id"], batch["catalogue_version"], batch["state_version"], commands)
        appends = ["Ok"] * 5
    return _status(batch, commands, cap=cap, live=live, appends=appends)


@pytest.mark.parametrize("kind,status", [
    ("dropped_row", "MapMismatch"), ("swapped_rows", "MapMismatch"), ("duplicated_map", "MapMismatch"),
    ("swapped_map_commands", "MapMismatch"), ("wrong_dimension", "WrongDimension"),
    ("nonfinite_row", "NonFiniteRow"), ("partial_append", "PartialAppend"), ("stale_state", "StaleContext"),
    ("over_cap_five_resolved", "OverCap"),
])
def test_rvb_reference_refusals(kind, status):
    assert _mutate(kind) == status


# ------------------------------------------------------------ checker self-controls

def _simulated_receipt():
    """What a correct library would print, built from fixture literals (self-control only)."""
    cells = []
    for spec in FIXTURE["cells"]:
        n, dims, padded = spec["capacity"], spec["dims"], spec["padded_dims"]
        readback = spec.get("expected_row_readback") or [r + ["00000000"] * (padded - dims) for r in spec["rows"]]
        top = spec.get("expected_top1") or [{"index": i, "score_bits": "3f800000"} for i in spec["expected_top1_index"]]
        cells.append({"cell": spec["name"], "create": "Ok", "padded_dims": padded,
                      "appends": [{"status": "Ok", "out_index": i} for i in spec["expected_out_index"]],
                      "snapshot": "Ok", "count": n, "view_dims": dims, "view_padded": padded,
                      "row_bytes": n * padded * 4, "readback": readback,
                      "tombstone_words": spec["expected_tombstone_words"],
                      "queries": [{"status": "Ok", "hit_count": 1, **t} for t in top]})
    obs = [{"observation": o["name"], "create": "Ok",
            "statuses": o.get("expect_statuses") or [o["expect_status"]], "count_after": o["expect_count_after"]}
           for o in FIXTURE["superfaiss_observations"]]
    return copy.deepcopy({"cells": cells, "observations": obs})


def test_checker_accepts_simulated_correct_receipt():
    assert sf1_check.check(FIXTURE, _simulated_receipt()) == []


@pytest.mark.parametrize("damage", ["swap_hits", "shift_out_index", "drop_row", "pad_lane", "partial_obs"])
def test_checker_refuses_simulated_damage(damage):
    receipt = _simulated_receipt()
    ortho = receipt["cells"][0]
    if damage == "swap_hits":
        ortho["queries"][0]["index"], ortho["queries"][1]["index"] = 1, 0
    elif damage == "shift_out_index":
        ortho["appends"][2]["out_index"] = 3
    elif damage == "drop_row":
        ortho["count"] = 2
    elif damage == "pad_lane":
        ortho["readback"][0][3] = "3f800000"
    elif damage == "partial_obs":
        receipt["observations"][2]["count_after"] = 0
    assert sf1_check.check(FIXTURE, receipt)


# ----------------------------------------------------- S0-SF1: the real SuperFAISS run

def test_sf1_real_superfaiss_round_trip():
    if not RECEIPT.is_file():
        reason = "box timing lock present" if LOCK.exists() else "run_sf1.ps1 not yet executed"
        pytest.fail(f"red-owed: S0-SF1 receipt absent ({reason}); run tests/t3094-multicommand-s0/sf1/run_sf1.ps1")
    assert RECEIPT.stat().st_mtime >= (HERE / "sf1-fixture.json").stat().st_mtime, "stale receipt"
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8-sig"))
    assert sf1_check.check(FIXTURE, receipt) == []


# ------------------------------------------------- rev-9 resolved-command cells (plan rev 15 s6 row 12)

import resolved_condition as rc  # noqa: E402


def test_sf1_batch_is_canonical_rev9_shape():
    rc.guard_batch(COMMANDS)
    assert [len(c["trigger"]["start_all"]) for c in COMMANDS] == [0, 2, 0]
    assert COMMANDS[2]["trigger"]["stop_any"][0]["value"] == {"num": 75, "den": 2}
    assert all(not any(c["trigger"].values()) for c in COMMANDS[:1])


def test_permuted_entry_pair_changes_row_and_is_refused_before_encoding():
    permuted = FIXTURE["permuted_entry_command"]
    assert rvb_e1.encode_bits(permuted) != rvb_e1.encode_bits(COMMANDS[1])
    with pytest.raises(rc.GuardRefusal) as caught:
        rc.guard_batch([COMMANDS[0], permuted, COMMANDS[2]])
    assert caught.value.status == "InvalidResolvedCommand"
    assert rc.canonicalize(permuted["trigger"]) == COMMANDS[1]["trigger"]


def test_observable_catalogue_change_alone_is_stale():
    batch, commands = _healthy()
    stale = FIXTURE["stale_live_catalogue_version"]
    assert stale.split("+")[0] == FIXTURE["batch"]["catalogue_version"].split("+")[0]
    assert _status(batch, commands, live=(stale, FIXTURE["batch"]["state_version"])) == "StaleContext"


@pytest.mark.parametrize("damage", ["float_value", "unreduced", "duplicate", "wait_ref_later", "ref_on_passive"])
def test_guard_refuses_non_canonical(damage):
    cmds = copy.deepcopy(COMMANDS)
    e = cmds[2]["trigger"]["stop_any"][0]
    if damage == "float_value":
        e["value"] = 37.5
    elif damage == "unreduced":
        e["value"] = {"num": 150, "den": 4}
    elif damage == "duplicate":
        cmds[1]["trigger"]["start_all"].append(copy.deepcopy(cmds[1]["trigger"]["start_all"][-1]))
    elif damage == "wait_ref_later":
        e["binding"] = {"mode": "wait", "ref": 2}
    elif damage == "ref_on_passive":
        e["binding"]["ref"] = 1
    with pytest.raises(rc.GuardRefusal):
        rc.guard_batch(cmds, wait_ordinals={0, 1, 2})
