"""Quarantined T-3035 population discrimination instrument.

The screen subcommand reads raw token streams emitted by t3035_screen. The
mutation subcommand will reduce shipped-verb save/edit/restore observations.
Commissioning constructions are authored by an independent seat.
"""

import argparse
import hashlib
import json
from pathlib import Path


EOS = {151643, 151645}


def onset(tokens: list[int]) -> int:
    for start in range(max(0, len(tokens) - 31)):
        for period in range(1, 5):
            if all(tokens[i] == tokens[i - period] for i in range(start + period, len(tokens))):
                return start
    return len(tokens)


def screen(input_path: Path, output_path: Path) -> int:
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        head, body = line.split(" tokens=", 1)
        fields = head.split()
        token_field = body.split()[0]
        tokens = [int(x) for x in token_field.split(",") if x]
        eos_at = next((i for i, t in enumerate(tokens) if t in EOS), -1)
        unique_tail = len(set(tokens[-16:]))
        periodic_at = onset(tokens)
        rows.append({"prompt": fields[0], "length": int(fields[1].split("=", 1)[1]),
                     "mode": fields[2].split("=", 1)[1], "tokens": len(tokens),
                     "eos_index": eos_at, "last16_distinct": unique_tail,
                     "periodic_onset": periodic_at,
                     "pass": len(tokens) == 64 and eos_at == -1 and unique_tail > 2})
    late_count = sum(row["periodic_onset"] >= 32 for row in rows)
    passed = bool(rows) and all(row["pass"] for row in rows) and 10 * late_count >= 9 * len(rows)
    result = {"instrument": "T-3035-discrimination", "status": "QUARANTINED",
              "input": str(input_path), "rows": rows,
              "summary": {"streams": len(rows), "late_periodic": late_count,
                          "required_late_periodic": 0.9, "pass": passed}}
    output_path.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return 0 if passed else 1


def mutation(input_path: Path, output_path: Path, target: Path,
             donor: Path, screen_path: Path | None) -> int:
    """Transcribe one shipped-verb probe without licensing its population."""
    lines = input_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("missing probe header")
    fields = dict(part.split("=", 1) for part in lines[0][2:].split()
                  if "=" in part)
    if int(fields["baseline_tokens"]) != 64:
        raise ValueError("unedited round trip did not reach 64 tokens")
    baseline = [int(value) for value in fields["tokens"].split(",")]
    if len(baseline) != 64:
        raise ValueError("baseline token count mismatch")
    results = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"malformed mutant row: {line}")
        name = parts[0]
        attrs = dict(part.split("=", 1) for part in parts[1:])
        killed = bool(int(attrs["killed"]))
        first = int(attrs["first_diff"])
        observed = int(attrs["tokens"])
        if (first < 0) == killed or (not killed and observed != 64):
            raise ValueError(f"inconsistent mutant result: {line}")
        results.append({"mutant": name, "killed": killed,
                        "first_divergent_token": first,
                        "observed_tokens": observed,
                        "donor_prompt_id": donor.stem})
    if len({row["mutant"] for row in results}) != len(results):
        raise ValueError("duplicate mutant row")
    screen_rows = []
    if screen_path:
        screen_data = json.loads(screen_path.read_text(encoding="utf-8"))
        screen_rows = [row for row in screen_data["rows"]
                       if Path(row["prompt"]).stem == target.stem
                       and row["mode"] == fields["mode"]]
        if len(screen_rows) != 1:
            raise ValueError("matching screen row missing or duplicated")
    result = {
        "instrument": "T-3035-precache-discrimination",
        "status": "QUARANTINED",
        "population_adequate": False,
        "reason": "M-swap and system-first M-row(0) source contradictions; commissioning unauthored",
        "probe_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "target_prompt_id": target.stem,
        "target_ids_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "donor_prompt_id": donor.stem,
        "donor_ids_sha256": hashlib.sha256(donor.read_bytes()).hexdigest(),
        "length": int(fields["L"]), "page_positions": int(fields["B"]),
        "decode_mode": fields["mode"],
        "unedited_roundtrip_tokens": baseline,
        "screen": screen_rows[0] if screen_rows else None,
        "mutants": results,
        "threshold_observation": {
            "M_res_killed": next((row["killed"] for row in results
                                  if row["mutant"] == "M-res"), None),
            "every_page_killed": all(row["killed"] for row in results
                                     if row["mutant"].startswith("M-page")),
            "tail_killed": next((row["killed"] for row in results
                                 if row["mutant"] == "M-tail"), None),
            "M_swap_killed": next((row["killed"] for row in results
                                   if row["mutant"].startswith("M-swap")), None),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return 0


def summarize(directory: Path) -> int:
    paths = sorted(directory.glob("*.discrimination.json"))
    if len(paths) != 10:
        raise ValueError("a ten-prefix population is required")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    strata = ("0", "1..B-1", "B-1", "B", "floor(L/2)", "L-B-1", "L-1")
    counts = {stratum: 0 for stratum in strata}
    summary = {"prefixes": 10, "M_res_killed": 0,
               "all_pages_killed": 0, "tail_killed_where_defined": 0,
               "tail_defined": 0, "row_killed": counts,
               "M_swap_killed": 0, "M_swap_measured": 0,
               "thresholds_met": False,
               "source_limits": ["M-swap is a K/V pair permutation",
                                 "system-first chat makes M-row(0) donor-identical",
                                 "commissioning constructions are unauthored"]}
    for record in records:
        rows = {row["mutant"]: row for row in record["mutants"]}
        length, page = record["length"], record["page_positions"]
        if rows.get("M-res", {}).get("killed"):
            summary["M_res_killed"] += 1
        for name, position in zip(strata,
                                  (0, 1, page-1, page, length//2,
                                   length-page-1, length-1)):
            if rows.get(f"M-row({position})", {}).get("killed"):
                counts[name] += 1
        pages = [row for row in record["mutants"] if row["mutant"].startswith("M-page")]
        if len(pages) == 3 and all(row["killed"] for row in pages):
            summary["all_pages_killed"] += 1
        if length % page:
            summary["tail_defined"] += 1
            if rows.get("M-tail", {}).get("killed"):
                summary["tail_killed_where_defined"] += 1
        swaps = [row for row in record["mutants"] if row["mutant"].startswith("M-swap")]
        if swaps:
            summary["M_swap_measured"] += 1
            summary["M_swap_killed"] += int(swaps[0]["killed"])
    summary["required"] = {"M_res": "10/10 in at least one decode mode",
                           "each_row_stratum": "8/10, and B-1, B, L-1 each 10/10",
                           "pages": "10/10", "tail": "all defined",
                           "swap": "10/10"}
    for path, record in zip(paths, records):
        record["population_summary"] = summary
        path.write_bytes((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return 0


def composition(input_path: Path, output_path: Path, kind: str,
                walk_path: Path | None) -> int:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("missing composition header")
    header = dict(part.split("=", 1) for part in lines[0][2:].split() if "=" in part)
    stream_rows = []
    dispositions = []
    for line in lines[1:]:
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        if line.startswith("disposition "):
            dispositions.append(fields)
        else:
            fields["tokens"] = [int(value) for value in fields["tokens"].split(",")]
            stream_rows.append(fields)
    if not stream_rows or any(len(row["tokens"]) != 64 for row in stream_rows):
        raise ValueError("composition did not produce 64-token streams")
    result = {"instrument": "T-3035-precache-discrimination",
              "status": "QUARANTINED", "population_adequate": False,
              "probe_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "model_sha256": header["model_sha256"],
              "prompt_ids_sha256": header["prompt_ids_sha256"],
              "prefix_id": stream_rows[0]["prefix_id"],
              "kind": kind}
    if kind == "adapter":
        by_mode = {(row["adopter_mode"], row["adapter"]): row for row in stream_rows}
        controls = []
        for mode in ("budget", "legacy"):
            on = by_mode[(mode, "bound")]["tokens"]
            off = by_mode[(mode, "off")]["tokens"]
            different = [i for i, pair in enumerate(zip(on, off)) if pair[0] != pair[1]]
            controls.append({"adopter_mode": mode, "adapter_on_off_different_tokens": len(different),
                             "first_divergent_token": different[0] if different else -1,
                             "required_different_tokens": 16,
                             "killed": len(different) >= 16})
        result.update({"adapter_sha256": header["adapter_sha256"],
                       "controls": controls,
                       "reason": "Adapter controls measured; discrimination instrument uncommissioned and other §5.4 source limits remain"})
    elif kind == "schema":
        if any(int(row["walk"]) == 0 for row in stream_rows):
            raise ValueError("schema walk was not progressed")
        walk_result = None
        if walk_path:
            probe = dict(part.split("=", 1) for part in walk_path.read_text(encoding="utf-8").split())
            if (probe["schema"] != stream_rows[0]["schema_name"] or
                    int(probe["original_walk"]) != int(stream_rows[0]["walk"])):
                raise ValueError("walk probe does not match schema reference")
            walk_result = {"probe_sha256": hashlib.sha256(walk_path.read_bytes()).hexdigest(),
                           "donor_schema": probe["donor_schema"],
                           "donor_schema_hash": probe["donor_schema_hash"],
                           "mutant_walk": int(probe["mutant_walk"]),
                           "restore_status": int(probe["restore_status"]),
                           "killed": bool(int(probe["killed"])),
                           "first_divergent_token": int(probe["first_diff"])}
        result.update({"walk_state": int(stream_rows[0]["walk"]),
                       "forced_content_id": int(header["forced_content_id"]),
                       "forced_content_ids_sha256": header["forced_content_ids_sha256"],
                       "cross_binding_dispositions": dispositions,
                       "M_walk": walk_result,
                       "reason": "Progressed walk, cross-bind dispositions, and M-walk measured; instrument uncommissioned"})
    else:
        raise ValueError("unknown composition kind")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("screen")
    command.add_argument("input", type=Path)
    command.add_argument("output", type=Path)
    command = sub.add_parser("mutation")
    command.add_argument("input", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("target", type=Path)
    command.add_argument("donor", type=Path)
    command.add_argument("--screen", type=Path)
    command = sub.add_parser("summarize")
    command.add_argument("directory", type=Path)
    command = sub.add_parser("composition")
    command.add_argument("kind", choices=("adapter", "schema"))
    command.add_argument("input", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--walk", type=Path)
    args = parser.parse_args()
    if args.command == "screen":
        return screen(args.input, args.output)
    if args.command == "mutation":
        return mutation(args.input, args.output, args.target, args.donor, args.screen)
    if args.command == "summarize":
        return summarize(args.directory)
    if args.command == "composition":
        return composition(args.input, args.output, args.kind, args.walk)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
