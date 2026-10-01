"""Quarantined T-3035 population discrimination instrument.

The screen subcommand reads raw token streams emitted by t3035_screen. The
mutation subcommand will reduce shipped-verb save/edit/restore observations.
Commissioning constructions are authored by an independent seat.
"""

import argparse
import json
from pathlib import Path


EOS = {151643, 151645}


def onset(tokens: list[int]) -> int:
    for start in range(len(tokens) + 1):
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
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("screen")
    command.add_argument("input", type=Path)
    command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "screen":
        return screen(args.input, args.output)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
