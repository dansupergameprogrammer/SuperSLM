"""Independent T-3071 commissioning constructions for the T-3035 reducers.

The donor changes one pinned prompt token (index 19) and was selected without
using the test author's mutation population. Each invocation captures fresh
SSB6 and SSB5 observations through the shipped save/restore/decode probe, runs
the production reducer and summary loader, then checks independently fixed
observations. A reject cell exits 0 only after it has observed the specified
defect and the instrument's own refusal. The registry command inverts that
successful check to the nonzero must-reject convention.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path("D:/SuperSLM/.worktrees/T-3071")
ART = Path("D:/_artifacts/SuperSLM/t3071-commission")
PROBE = Path("D:/_scratch/T-3035/mutant-build/Release/t3035_mutant_probe.exe")
REDUCER = ROOT / "tools/t3035_revised_discrimination.py"
MODEL = Path("D:/_artifacts/SuperSLM/te442-qwen2.5-0.5b-instruct-dg/qwen2.5-0.5b-instruct-dg.sslm")
TARGET = ROOT / "tests/t3035-precache-refs/prompts/flat00.ids"
DONOR = ART / "donor-flat00-pos19.ids"
REFERENCE = Path("D:/_artifacts/SuperSLM/t3035-precache-refs/P1-real0p5-flat/flat00-run1.ref")
COMPOSITION = Path("D:/_scratch/T-3035/adapter-select-build/Release/t3035_composition_reference.exe")
G5 = Path("D:/_artifacts/SuperSLM/t2901-qwen2.5-1.5b-dg/qwen2.5-1.5b-instruct-dg-g5.sslm")
LORA = Path("D:/_artifacts/SuperSLM/t2901-qwen2.5-1.5b-dg/qwen2.5-1.5b-shopkeeper-lora-v2-t2901-dg-g5-runtime.sslm")
PROMPTS = ROOT / "tests/t3035-precache-refs/prompts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def line_fields(line):
    return dict(piece.split("=", 1) for piece in line.split() if "=" in piece)


def run(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"command failed {result.returncode}: {command}\n{result.stdout}\n{result.stderr}")


def capture(cell, fmt, reuse):
    folder = ART / cell / fmt
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "flat00.raw.txt"
    stream = folder / "flat00.stream.json"
    byte = folder / "flat00.byte.json"
    summary = folder / "summary.json"
    if not (reuse and raw.exists() and raw.stat().st_size):
        run([str(PROBE), str(MODEL), str(TARGET), str(DONOR), str(raw), "greedy", fmt, "all"])
    lines = raw.read_text(encoding="utf-8").splitlines()
    header = line_fields(lines[0])
    observed = {line.split()[0]: line_fields(line) for line in lines[1:]}
    run([sys.executable, str(REDUCER), "mutation", str(raw), str(REFERENCE),
         str(MODEL), str(TARGET), str(DONOR), str(stream), str(byte)])
    run([sys.executable, str(REDUCER), "summarize", str(folder), str(summary)])
    stream_data = json.loads(stream.read_text(encoding="utf-8"))
    byte_data = json.loads(byte.read_text(encoding="utf-8"))
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    return folder, header, observed, stream_data, byte_data, summary_data


def baseline(fmt, data):
    folder, h, rows, stream, byte, summary = data
    expected_format = "SSB6" if fmt == "budget" else "SSB5"
    require(h["format"] == expected_format and h["L"] == "159" and h["B"] == "16", "capture format/geometry")
    require(h["unedited_state_diff_bytes"] == "0" and h["baseline_tokens"] == "64", "unedited restore")
    ref_rows = [line_fields(line) for line in REFERENCE.read_text(encoding="utf-8").splitlines()[1:]
                if line.startswith("prefix_id=")]
    ref_tokens = [r["tokens"] for r in ref_rows if r["adopter_mode"] == fmt and r["decode_mode"] == "greedy"]
    require(len(ref_tokens) >= 1 and len(set(ref_tokens)) == 1 and
            h["tokens"] == ref_tokens[0], "unedited stream differs from prefill twin")
    require(stream["unedited_roundtrip_tokens"] == byte["unedited_roundtrip_tokens"] ==
            [int(x) for x in h["tokens"].split(",")], "reducer unedited stream")
    require(stream["reference_sha256"] == byte["reference_sha256"] == sha(REFERENCE), "reducer reference binding")
    require(stream["raw_sha256"] == byte["raw_sha256"] == sha(folder / "flat00.raw.txt"), "reducer raw binding")
    require(summary["prefixes"] == 1 and summary["unique_mutants"] == len(rows), "summary row population")
    return rows, {r["mutant"]: r for r in stream["mutants"]}, {r["mutant"]: r for r in byte["mutants"]}, summary


def check_stream_reject(rows, stream, byte, summary):
    # Fixed from the first fresh probe's raw output before reducer evaluation.
    for name, changed, first in (("M-page(4)", 72551, 6), ("M-page(8)", 72141, 6)):
        raw = rows[name]
        reduced = stream[name]
        require(raw["changed_bytes"] == str(changed) and raw["killed"] == "1" and
                raw["first_diff"] == str(first), f"independent {name} capture expectation")
        require(reduced["edit"] == "CHANGED" and reduced["changed_bytes"] == changed and
                reduced["stream_killed"] is True and reduced["first_divergent_token"] == first,
                f"stream instrument failed {name}")
    require(rows["M-page(0)"]["changed_bytes"] == "0" and
            stream["M-page(0)"]["edit"] == "IDENTITY" and
            stream["M-page(0)"]["stream_killed"] is None, "identity was counted")
    require(rows["M-res"]["changed_bytes"] == "752" and rows["M-res"]["killed"] == "0" and
            stream["M-res"]["stream_killed"] is False, "changed residual survivor was hidden")
    require(rows["M-swap(7,8)"]["changed_bytes"] == "186128" and
            stream["M-swap(7,8)"]["stream_killed"] is False and
            byte["M-swap(7,8)"]["exact_state_rejected"] is True,
            "stream-invisible complete-page swap misrouted")
    require(rows["M-row0-changed"]["changed_bytes"] == "1" and
            byte["M-row0-changed"]["exact_state_rejected"] is True,
            "changed-byte position-0 row misrouted")
    require(summary["classes"]["page:middle"]["killed"] == 1 and
            summary["classes"]["residual"]["survivors"][0]["mutant"] == "M-res",
            "per-stratum summary lost kill or survivor")


def check_byte_reject(rows, stream, byte, summary):
    expected = {"M-res": 752, "M-row0-changed": 1, "M-page(4)": 72551,
                "M-page(8)": 72141, "M-swap(7,8)": 186128, "M-scale": 1,
                **{f"M-count({offset})": 1 for offset in (92, 124, 132, 140, 148)}}
    for name, count in expected.items():
        raw = rows[name]
        reduced = byte[name]
        require(raw["changed_bytes"] == raw["restored_diff_bytes"] == str(count),
                f"independent restored-operand expectation for {name}")
        require(reduced["edit"] == "CHANGED" and reduced["changed_bytes"] == count and
                reduced["restored_operand_mismatched_bytes"] == count and
                reduced["exact_state_rejected"] is True,
                f"byte instrument failed {name}")
    require(byte["M-cur"]["scope"] == "SPX1-field-only" and
            byte["M-cur"]["exact_state_rejected"] is None,
            "sequence operand improperly graded nonnegative current_token")
    require(byte["M-page(0)"]["edit"] == "IDENTITY" and
            byte["M-page(0)"]["exact_state_rejected"] is None,
            "identity page received exact-state verdict")


def composition_controls(cell):
    folder = ART / cell / "composition"
    walk_out = folder / "walk"
    schema_out = folder / "schema"
    adapter_out = folder / "adapter"
    for path in (walk_out, schema_out, adapter_out):
        path.mkdir(parents=True, exist_ok=True)
    run([str(COMPOSITION), "walk", str(G5), str(PROMPTS), str(walk_out), "ids", "flat00"])
    walk = line_fields((walk_out / "schema0.walkprobe.txt").read_text(encoding="utf-8").strip())
    require(walk["original_walk"] == "2" and walk["mutant_walk"] == "0" and
            walk["restore_status"] == "0" and walk["killed"] == "1" and
            walk["first_diff"] == "0" and walk["baseline_tokens"] == "64",
            "progressed schema walk mutant was not a token-0 rejection")
    schema_record = ROOT / "tests/t3035-precache-refs/discrimination/real1p5-schema"
    run([str(COMPOSITION), "schema", str(G5), "-", str(PROMPTS), str(schema_out),
         str(schema_record), "run1", "1p5.ids", "TokenizerView::Encode(G5,4da4c4c)"])
    schema_lines = (schema_out / "schema0-run1.ref").read_text(encoding="utf-8").splitlines()
    schema_rows = [line_fields(line) for line in schema_lines[1:] if line.startswith("prefix_id=")]
    dispositions = [line_fields(line) for line in schema_lines[1:] if line.startswith("disposition ")]
    require(len(schema_rows) == 4 and all(row["walk"] == "2" and
            len(row["tokens"].split(",")) == 64 for row in schema_rows), "bound schema capture")
    require({row["bound_schema"]: row["status"] for row in dispositions} ==
            {"g5_minimal_one_field": "21", "none": "21"},
            "cross-bound and unbound schema refusal")
    adapter_record = ROOT / "tests/t3035-precache-refs/discrimination/real1p5-adapter"
    run([str(COMPOSITION), "adapter", str(G5), str(LORA), str(PROMPTS), str(adapter_out),
         str(adapter_record), "run1", "1p5.ids", "TokenizerView::Encode(G5,4da4c4c)", "1"])
    adapter_lines = (adapter_out / "adapter0-run1.ref").read_text(encoding="utf-8").splitlines()
    adapter_rows = [line_fields(line) for line in adapter_lines[1:] if line.startswith("prefix_id=")]
    require(len(adapter_rows) == 4, "adapter row count")
    for fmt in ("budget", "legacy"):
        on = [int(x) for x in next(row for row in adapter_rows if row["adopter_mode"] == fmt and
                            row["adapter"] == "bound")["tokens"].split(",")]
        off = [int(x) for x in next(row for row in adapter_rows if row["adopter_mode"] == fmt and
                             row["adapter"] == "off")["tokens"].split(",")]
        require(len(on) == len(off) == 64 and sum(a != b for a, b in zip(on, off)) == 34,
                f"{fmt} adapter on/off did not change 34 tokens")
    print(f"composition: walk_sha256={sha(walk_out / 'schema0.walkprobe.txt')}; "
          f"schema_sha256={sha(schema_out / 'schema0-run1.ref')}; "
          f"adapter_sha256={sha(adapter_out / 'adapter0-run1.ref')}; verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=("stream-accept", "stream-reject", "byte-accept", "byte-reject"))
    parser.add_argument("--reuse", action="store_true", help="preflight a previously captured raw file")
    args = parser.parse_args()
    for path, expected in (
        (PROBE, "f15d57215e935bdc19e596825ab05fb1a56e156e3f7763f1d8bddb44c0636f6c"),
        (COMPOSITION, "7eaa5f69595c235af1515425ac6d85d78633b6dd0d5cef63355e96abcf0ef76a"),
        (MODEL, "4b27e2d6e132d710a4b174d5ffc15e7705691ca37abf15eca65503803d0dba35"),
        (REFERENCE, "91dfbe91af5ef6bbf6f7e43189024a449e3a394899204ad7a374264f99d19df1"),
    ):
        require(sha(path) == expected, f"pinned input changed: {path}")
    require(sha(DONOR) == "1d5a0715d91e4df6e8bbb35f7380df678a0fb632e30c49d632985670e137c3f8", "donor changed")
    target_ids = [int(x) for x in TARGET.read_text(encoding="utf-8").split()]
    donor_ids = [int(x) for x in DONOR.read_text(encoding="utf-8").split()]
    require(len(target_ids) == len(donor_ids) == 159 and
            [(i, a, b) for i, (a, b) in enumerate(zip(target_ids, donor_ids)) if a != b] ==
            [(19, 9485, 12106)], "donor is not the independent one-token prefill")
    for fmt in ("budget", "legacy"):
        data = capture(args.cell, fmt, args.reuse)
        rows, stream, byte, summary = baseline(fmt, data)
        if args.cell == "stream-reject":
            check_stream_reject(rows, stream, byte, summary)
        elif args.cell == "byte-reject":
            check_byte_reject(rows, stream, byte, summary)
        print(f"{args.cell} {fmt}: normal capture/reducer/loader; raw_sha256={sha(data[0] / 'flat00.raw.txt')}; verified")
    if args.cell == "stream-reject":
        composition_controls(args.cell)
    print(f"{args.cell}: {'ACCEPTED correct restore' if args.cell.endswith('accept') else 'REJECTED specified defective restore'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CONSTRUCTION ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
