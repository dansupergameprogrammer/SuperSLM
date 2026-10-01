"""Pin the raw, quarantined T-3035 instrument observations in scratch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PATTERNS = (
    "screen*.txt", "screen*.json", "core*.txt", "mutant0p5*.txt",
    "invisible*.txt", "swap*.txt", "row0-*.txt", "walk*/*.walkprobe.txt",
    "schema-smoke/out*/*.ref", "adapter-smoke/out/*.ref",
    "adapter-candidate01/out/*.ref",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted({path for pattern in PATTERNS for path in args.scratch.glob(pattern)})
    rows = []
    for path in files:
        data = path.read_bytes()
        rows.append({"path": str(path), "size": len(data),
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    result = {"status": "QUARANTINED", "files": rows, "count": len(rows),
              "purpose": "Raw screens, shipped-verb mutations, and provisional composition controls"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(f"{len(rows)} raw observations pinned")


if __name__ == "__main__":
    main()
