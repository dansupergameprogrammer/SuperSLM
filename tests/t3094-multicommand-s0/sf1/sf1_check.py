"""T-3103 S0-SF1 independent checker (instrument T-3103-SF1-CHECK).

Decides whether a harness receipt shows the resolved batch round-tripping through the real
SuperFAISS ScratchBank Create/Append/Snapshot/Query seam: every row ingested at its ordinal,
the stored bytes equal to the fixture rows, each orthogonal probe returning its intended row,
and each returned index resolving through the host map to the exact original command, ordinal
and batch ID. Expected values come from sf1-fixture.json literals, never from the receipt.

Exit 0 + "ACCEPTED" = cell passes. Exit 1 + one "REFUSED:" line = cell fails.
Exit 2 + "CONSTRUCTION ERROR" = unreadable input; never a refusal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvb_e1  # noqa: E402

ZERO = "00000000"


def check(fixture: dict, receipt: dict) -> list[str]:
    errors: list[str] = []
    batch = fixture["batch"]
    host_map = rvb_e1.build_batch(batch["batch_id"], batch["catalogue_version"], batch["state_version"],
                                  batch["commands"])["host_map"]
    if host_map != fixture["expected_host_map"]:
        errors.append("reference host map differs from the fixture's literal map")
    cells = {c.get("cell"): c for c in receipt.get("cells", [])}
    if set(cells) != {c["name"] for c in fixture["cells"]}:
        errors.append(f"receipt cells {sorted(cells)} do not match fixture cells")
    for spec in fixture["cells"]:
        got = cells.get(spec["name"])
        if got is None:
            continue
        tag = spec["name"]
        n, dims, padded = spec["capacity"], spec["dims"], spec["padded_dims"]
        if got["create"] != "Ok" or got["snapshot"] != "Ok":
            errors.append(f"{tag}: create/snapshot not Ok")
        if got["padded_dims"] != padded or got["view_padded"] != padded or got["view_dims"] != dims:
            errors.append(f"{tag}: geometry {got['view_dims']}/{got['view_padded']} != {dims}/{padded}")
        appends = got["appends"]
        if [a["status"] for a in appends] != ["Ok"] * n:
            errors.append(f"{tag}: append statuses {[a['status'] for a in appends]}")
        if [a["out_index"] for a in appends] != spec["expected_out_index"]:
            errors.append(f"{tag}: out_index {[a['out_index'] for a in appends]} != {spec['expected_out_index']}")
        if got["count"] != n:
            errors.append(f"{tag}: published count {got['count']} != {n}")
        if got["row_bytes"] != n * padded * 4:
            errors.append(f"{tag}: row bytes {got['row_bytes']} != {n * padded * 4}")
        expected_readback = spec.get("expected_row_readback") or [
            list(row) + [ZERO] * (padded - dims) for row in spec["rows"]]
        if got["readback"] != expected_readback:
            errors.append(f"{tag}: stored row bytes differ from fixture rows")
        if got["tombstone_words"] != spec["expected_tombstone_words"]:
            errors.append(f"{tag}: tombstone words {got['tombstone_words']}")
        queries = got["queries"]
        if len(queries) != n:
            errors.append(f"{tag}: {len(queries)} queries for {n} probes")
            continue
        for probe, q in enumerate(queries):
            want_index = (spec["expected_top1"][probe]["index"] if "expected_top1" in spec
                          else spec["expected_top1_index"][probe])
            if q["status"] != "Ok" or q["hit_count"] != 1 or q["index"] != want_index:
                errors.append(f"{tag}: probe {probe} -> {q}, want index {want_index}")
                continue
            if "expected_top1" in spec and q["score_bits"] != spec["expected_top1"][probe]["score_bits"]:
                errors.append(f"{tag}: probe {probe} score bits {q['score_bits']}")
            entry = host_map[q["index"]]
            want = fixture["expected_host_map"][probe]
            if (entry["command"], entry["ordinal"], entry["batch_id"]) != \
                    (want["command"], want["ordinal"], want["batch_id"]):
                errors.append(f"{tag}: hit {q['index']} maps to ordinal {entry['ordinal']}, want {want['ordinal']}")
        if spec["name"] == "SF1-E1":
            e1_rows = [["%08x" % b for b in rvb_e1.encode_bits(c)] for c in batch["commands"]]
            if e1_rows != spec["rows"]:
                errors.append("SF1-E1: fixture rows are not the E1 encoding of the batch")
    observations = {o.get("observation"): o for o in receipt.get("observations", [])}
    for spec in fixture["superfaiss_observations"]:
        got = observations.get(spec["name"])
        if got is None:
            errors.append(f"{spec['name']}: missing observation")
            continue
        want = spec.get("expect_statuses") or [spec["expect_status"]]
        if got["create"] != "Ok" or got["statuses"] != want or got["count_after"] != spec["expect_count_after"]:
            errors.append(f"{spec['name']}: observed {got['statuses']} count {got['count_after']}")
    return errors


def main() -> int:
    try:
        fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        receipt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
        errors = check(fixture, receipt)
    except Exception as error:  # noqa: BLE001 - any unreadable input is a construction error
        print(f"CONSTRUCTION ERROR: {type(error).__name__}: {error}")
        return 2
    if errors:
        print("REFUSED: " + " | ".join(errors))
        return 1
    print("ACCEPTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
