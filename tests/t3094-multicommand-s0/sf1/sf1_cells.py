"""Writes the harness cells file from sf1-fixture.json (format in sf1_harness.cpp)."""
import json
import sys


def main(fixture_path: str, out_path: str) -> None:
    fixture = json.loads(open(fixture_path, encoding="utf-8").read())
    lines = [f"CELLS {len(fixture['cells'])}"]
    for cell in fixture["cells"]:
        if len(cell["rows"]) != cell["capacity"] or len(cell["probes"]) != cell["capacity"]:
            raise SystemExit("fixture rows/probes do not match capacity")
        lines.append(f"CELL {cell['name']} {cell['dims']} {cell['capacity']}")
        lines += ["ROW " + " ".join(row) for row in cell["rows"]]
        lines += ["PROBE " + " ".join(probe) for probe in cell["probes"]]
    with open(out_path, "w", encoding="ascii", newline="\n") as out:
        out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
