"""Audit and manifest the external T-3035 reference files without changing them."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    discrimination = {sha(path) for path in
                      (args.source_root / "tests/t3035-precache-refs/discrimination").rglob("*.json")}
    prompt_hashes = {sha(path) for directory in ("prompts", "synthetic-prompts")
                     for path in (args.source_root / "tests/t3035-precache-refs" / directory).glob("*")
                     if path.is_file() and path.suffix in {".txt", ".ids"}}
    results = []
    bad = []
    for run1 in sorted(args.artifact_root.rglob("*-run1.ref")):
        run2 = run1.with_name(run1.name.replace("-run1.ref", "-run2.ref"))
        if not run2.is_file():
            bad.append(f"missing second run: {run1}")
            continue
        pair_equal = run1.read_bytes() == run2.read_bytes()
        if not pair_equal:
            bad.append(f"byte mismatch: {run1}")
        for path in (run1, run2):
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines or not lines[0].startswith("# "):
                bad.append(f"missing header: {path}")
                continue
            header = fields(lines[0][2:])
            required = ("anchor", "harness", "model_sha256", "B", "cap",
                        "decode_modes_available", "prompt_source_sha256",
                        "prompt_ids_sha256", "tokenizer", "discrimination_record_sha256")
            for field in required:
                if field not in header:
                    bad.append(f"missing {field}: {path}")
            for field in ("anchor", "harness"):
                if len(header.get(field, "")) != 40:
                    bad.append(f"nonfull {field}: {path}")
            if header.get("discrimination_record_sha256") not in discrimination:
                bad.append(f"discrimination SHA not in committed source set: {path}")
            for field in ("prompt_source_sha256", "prompt_ids_sha256"):
                if header.get(field) not in prompt_hashes:
                    bad.append(f"{field} not in committed source set: {path}")
            streams = [fields(line) for line in lines[1:] if not line.startswith("disposition ")]
            for row in streams:
                if len(row.get("tokens", "").split(",")) != 64:
                    bad.append(f"stream not 64 tokens: {path}")
                for field in ("prefix_id", "L", "L_mod_B", "origin", "walk",
                              "adopter_mode", "decode_mode", "schema_name",
                              "schema_hash", "adapter", "terminal_status", "eos_index"):
                    if field not in row:
                        bad.append(f"missing row {field}: {path}")
            results.append({"path": str(path), "size": path.stat().st_size,
                            "sha256": sha(path),
                            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                            "pair_byte_equal": pair_equal, "stream_rows": len(streams),
                            "dispositions": sum(line.startswith("disposition ") for line in lines[1:]),
                            "header": header})
    summary = {"status": "QUARANTINED", "reference_files": len(results),
               "pairs": len(results) // 2, "audit_errors": bad,
               "files": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"{len(results)} files, {len(results)//2} pairs, {len(bad)} audit errors")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
