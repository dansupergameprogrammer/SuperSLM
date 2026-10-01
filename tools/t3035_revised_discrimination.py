"""Quarantined T-3035 revised stream and restored-state observation reducer.

The independent debunker supplies commissioning constructions. No result from
this file is load-bearing before the two registered instruments are commissioned.
"""

import argparse
import hashlib
import json
from pathlib import Path


STREAM_ID = "T-3035-precache-stream-v2"
BYTE_ID = "T-3035-precache-byte-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def reference_row(path: Path, fmt: str, mode: str, target_hash: str) -> tuple[dict, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("reference header missing")
    header = fields(lines[0])
    if header["prompt_ids_sha256"].lower() != target_hash:
        raise ValueError("reference prompt hash differs from probe target")
    adopter = "legacy" if fmt == "SSB5" else "budget"
    candidates = [fields(line) for line in lines[1:] if line.startswith("prefix_id=")]
    candidates = [row for row in candidates if row.get("adopter_mode") == adopter
                  and row.get("decode_mode") == mode]
    if not candidates:
        raise ValueError("no matching reference row")
    tokens = [[int(x) for x in row["tokens"].split(",") if x] for row in candidates]
    if len({tuple(row) for row in tokens}) != 1:
        raise ValueError("matching reference rows disagree")
    return header, tokens[0]


def exact_address(name: str, h: dict[str, str]) -> dict:
    fmt = h["format"]
    hidden_offset = 156 if fmt == "SSB5" else 172
    if name == "M-res":
        return {"kind": "field", "byte_offset": hidden_offset,
                "byte_count": int(h["hidden"])}
    if name == "M-scale":
        return {"kind": "field", "byte_offset": 76, "byte_count": 1}
    if name == "M-cur":
        return {"kind": "field", "byte_offset": 72, "byte_count": 4}
    if name.startswith("M-count("):
        return {"kind": "field", "byte_offset": int(name[8:-1]), "byte_count": 1}
    if name == "M-row0-changed":
        return {"kind": "field", "byte_offset": int(h["rows_offset"]), "byte_count": 1}
    page = int(h["B"])
    length = int(h["L"])
    if name.startswith("M-row("):
        positions = [int(name[6:-1])]
    elif name.startswith("M-page("):
        first = int(name[7:-1]) * page
        positions = list(range(first, first + page))
    elif name == "M-tail":
        positions = list(range(length // page * page, length))
    elif name.startswith("M-swap("):
        a, b = (int(x) for x in name[7:-1].split(","))
        positions = list(range(a * page, (a + 1) * page)) + list(range(b * page, (b + 1) * page))
    else:
        raise ValueError(f"unknown mutant {name}")
    if not positions or min(positions) < 0 or max(positions) >= length:
        raise ValueError(f"mutant row outside prefix: {name}")
    return {"kind": "KV", "rows_offset": int(h["rows_offset"]),
            "run_first": 0, "run_count": int(h["runs"]),
            "stride_positions": int(h["stride"]),
            "row_bytes": int(h["row_bytes"]), "positions": positions,
            "byte_offset_formula": "rows_offset + (run * stride_positions + position) * row_bytes"}


def reduce(raw: Path, reference: Path, target: Path, donor: Path,
           stream_out: Path, byte_out: Path) -> None:
    lines = raw.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("probe header missing")
    header = fields(lines[0])
    if header["format"] not in {"SSB5", "SSB6"}:
        raise ValueError("unknown format")
    if int(header["unedited_state_diff_bytes"]) != 0:
        raise ValueError("unedited round trip changed saved state")
    baseline = [int(x) for x in header["tokens"].split(",") if x]
    target_hash = digest(target)
    ref_header, expected = reference_row(reference, header["format"], header["mode"], target_hash)
    if baseline != expected or len(baseline) != 64:
        raise ValueError("unedited restore stream differs from prefill reference")
    if int(ref_header["B"]) != int(header["B"]):
        raise ValueError("page geometry differs from reference")
    common = {"status": "QUARANTINED", "raw_sha256": digest(raw),
              "probe_source_sha256": digest(Path(__file__).resolve().parents[1] / "tests/t3035_mutant_probe.cpp"),
              "reducer_source_sha256": digest(Path(__file__).resolve()),
              "reference_sha256": digest(reference), "reference_path": str(reference),
              "model_sha256": ref_header["model_sha256"],
              "target_ids_sha256": target_hash, "donor_ids_sha256": digest(donor),
              "format": header["format"], "adopter_mode": "legacy" if header["format"] == "SSB5" else "budget",
              "decode_mode": header["mode"], "length": int(header["L"]),
              "page_positions": int(header["B"]), "unedited_roundtrip_tokens": baseline,
              "unedited_state_diff_bytes": 0}
    stream_rows = []
    byte_rows = []
    names = set()
    for line in lines[1:]:
        parts = line.split()
        name = parts[0]
        if name in names:
            raise ValueError(f"duplicate mutant {name}")
        names.add(name)
        item = fields(line)
        changed = int(item["changed_bytes"])
        restored_diff = int(item["restored_diff_bytes"])
        killed = bool(int(item["killed"]))
        first = int(item["first_diff"])
        observed = int(item["tokens"])
        status = int(item["status"])
        if status != 0 or (first < 0) == killed or (not killed and observed != 64):
            raise ValueError(f"inconsistent mutant observation: {line}")
        if changed == 0 and (killed or restored_diff != 0):
            raise ValueError(f"identity edit changed state or stream: {name}")
        classification = "IDENTITY" if changed == 0 else "CHANGED"
        address = exact_address(name, header)
        stream_rows.append({"mutant": name, "edit": classification,
                            "changed_bytes": changed, "restore_status": status,
                            "stream_killed": killed if changed else None,
                            "first_divergent_token": first if changed else None,
                            "observed_tokens": observed, "address": address})
        byte_rows.append({"mutant": name, "edit": classification,
                          "changed_bytes": changed, "restore_status": status,
                          "restored_operand_mismatched_bytes": restored_diff,
                          "exact_state_rejected": (restored_diff > 0 if changed and name != "M-cur" else None),
                          "scope": "SPX1-field-only" if name == "M-cur" else "restored-sequence-operand",
                          "address": address})
    if not names or (header.get("selection") != "fields" and
                     ("M-res" not in names or "M-row0-changed" not in names)):
        raise ValueError("required mutant classes missing")
    for instrument, rows, out in ((STREAM_ID, stream_rows, stream_out),
                                  (BYTE_ID, byte_rows, byte_out)):
        payload = dict(common, instrument=instrument, mutants=rows,
                       note="Observation only; independent commissioning and cell assertions pending")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def summarize(directory: Path, output: Path) -> None:
    paths = sorted(directory.glob("*.stream.json"))
    groups: dict[str, list[dict]] = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = f'{record["model_sha256"]}:{record["adopter_mode"]}:{record["decode_mode"]}'
        groups.setdefault(key, []).append(record)
    result = {"instrument": STREAM_ID, "status": "QUARANTINED", "populations": {}}
    for key, records in groups.items():
        classes: dict[str, dict] = {}
        for record in records:
            for row in record["mutants"]:
                name = row["mutant"]
                if name == "M-res":
                    cls = "residual"
                elif name.startswith("M-row(") and name != "M-row(0)":
                    cls = "noninitial_row"
                elif name.startswith("M-page("):
                    cls = "full_page"
                elif name == "M-tail":
                    cls = "partial_tail"
                else:
                    cls = name
                group = classes.setdefault(cls, {"changed": 0, "killed": 0, "identities": 0, "survivors": []})
                if row["edit"] == "IDENTITY":
                    group["identities"] += 1
                else:
                    group["changed"] += 1
                    group["killed"] += int(row["stream_killed"])
                    if not row["stream_killed"]:
                        group["survivors"].append({"reference_sha256": record["reference_sha256"],
                                                   "mutant": name})
        result["populations"][key] = {"prefixes": len(records), "classes": classes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mutation")
    for arg in ("raw", "reference", "target", "donor", "stream_out", "byte_out"):
        m.add_argument(arg, type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("directory", type=Path)
    s.add_argument("output", type=Path)
    a = parser.parse_args()
    if a.command == "mutation":
        reduce(a.raw, a.reference, a.target, a.donor, a.stream_out, a.byte_out)
    else:
        summarize(a.directory, a.output)


if __name__ == "__main__":
    main()
