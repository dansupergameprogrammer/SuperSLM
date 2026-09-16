#!/usr/bin/env python3
"""T-2704's fail-closed release acceptance gate.

The production path validates every pinned identity before it invokes its
diagnostics.  Report arguments are a commissioning seam only: they let an
independent construction prove every rejection path without pretending that a
saved report is production acceptance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FIDELITY_FLOOR = 0.906603
RETRIEVAL_FLOOR = 148
RETRIEVAL_TOTAL = 239
REQUIRED_MODELS = {"Qwen2.5", "Qwen3"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reject(message: str) -> None:
    raise ValueError(message)


def validate_identities(manifest: dict, root: Path) -> dict[str, str]:
    identities = manifest.get("identities")
    if not isinstance(identities, list) or not identities:
        reject("manifest has no pinned identities")
    actual: dict[str, str] = {}
    for identity in identities:
        name, relative, expected = identity.get("name"), identity.get("path"), identity.get("sha256")
        if not all(isinstance(value, str) and value for value in (name, relative, expected)):
            reject("identity requires nonempty name, path, and sha256")
        path = root / relative
        if not path.is_file():
            reject(f"missing pinned identity: {name}: {path}")
        digest = sha256(path)
        if digest != expected:
            reject(f"changed pinned identity: {name}: expected {expected}, got {digest}")
        actual[name] = digest
    return actual


def invoke(command: list[str], label: str) -> dict:
    if not command or not all(isinstance(item, str) and item for item in command):
        reject(f"{label} diagnostic command is missing or malformed")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        reject(f"{label} diagnostic exited {completed.returncode}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        reject(f"{label} diagnostic did not emit a JSON result: {error}")


def validate_fidelity(report: dict) -> dict:
    rows = report.get("models")
    if not isinstance(rows, list):
        reject("fidelity report has no model rows")
    found: dict[str, float] = {}
    for row in rows:
        model, fidelity = row.get("model"), row.get("final_norm_fidelity")
        if model in found or model not in REQUIRED_MODELS or not isinstance(fidelity, (int, float)):
            reject("fidelity report has malformed or duplicate model row")
        found[model] = float(fidelity)
    if set(found) != REQUIRED_MODELS:
        reject("fidelity report omits Qwen2.5 or Qwen3")
    for model, fidelity in found.items():
        if fidelity < FIDELITY_FLOOR:
            reject(f"{model} fidelity {fidelity:.6f} < {FIDELITY_FLOOR:.6f}")
    return found


def validate_retrieval(report: dict) -> int:
    correct, total = report.get("retrieval_correct"), report.get("retrieval_total")
    if not isinstance(correct, int) or total != RETRIEVAL_TOTAL:
        reject("retrieval report must state an integer correct count over 239")
    if correct < RETRIEVAL_FLOOR:
        reject(f"retrieval {correct}/{total} < {RETRIEVAL_FLOOR}/{RETRIEVAL_TOTAL}")
    return correct


def validate_replay(report: dict) -> int:
    margins = report.get("margins")
    if not isinstance(margins, list) or not margins:
        reject("replay report has no element margins")
    bad = next((margin for margin in margins if not isinstance(margin, int) or margin <= 0), None)
    if bad is not None:
        reject(f"replayed element has non-positive INT64 margin: {bad}")
    return min(margins)


def gate(manifest_path: Path, report_path: Path, fidelity_path: Path | None,
         retrieval_path: Path | None, replay_path: Path | None) -> dict:
    manifest = load_json(manifest_path)
    root = manifest_path.parent.parent.parent
    identities = validate_identities(manifest, root)
    diagnostics = manifest.get("diagnostics", {})
    fidelity = load_json(fidelity_path) if fidelity_path else invoke(diagnostics.get("fidelity_command", []), "fidelity")
    retrieval = load_json(retrieval_path) if retrieval_path else invoke(diagnostics.get("retrieval_command", []), "retrieval")
    replay = load_json(replay_path) if replay_path else invoke(diagnostics.get("replay_command", []), "replay")
    result = {
        "manifest_sha256": sha256(manifest_path),
        "identities": identities,
        "fidelity": validate_fidelity(fidelity),
        "retrieval_correct": validate_retrieval(retrieval),
        "minimum_replay_margin": validate_replay(replay),
        "commissioning_seam": bool(fidelity_path or retrieval_path or replay_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def commission(receipt: Path) -> int:
    """Execute a healthy construction and every specified bad outcome."""
    with tempfile.TemporaryDirectory(prefix="t2704-release-gate-") as temporary:
        root = Path(temporary)
        artifact = root / "artifact.sslm"
        corpus = root / "corpus.jsonl"
        artifact.write_bytes(b"commissioned artifact\n")
        corpus.write_text("commissioned corpus\n", encoding="utf-8")
        manifest = root / "tests/data/t2704_release_manifest.json"
        manifest.parent.mkdir(parents=True)
        base = {"identities": [
            {"name": "artifact", "path": "artifact.sslm", "sha256": sha256(artifact)},
            {"name": "retrieval-corpus", "path": "corpus.jsonl", "sha256": sha256(corpus)},
        ], "diagnostics": {"fidelity_command": [], "retrieval_command": [], "replay_command": []}}
        manifest.write_text(json.dumps(base), encoding="utf-8")
        healthy = {"models": [{"model": "Qwen2.5", "final_norm_fidelity": FIDELITY_FLOOR},
                              {"model": "Qwen3", "final_norm_fidelity": FIDELITY_FLOOR}]}
        retrieval = {"retrieval_correct": RETRIEVAL_FLOOR, "retrieval_total": RETRIEVAL_TOTAL}
        replay = {"margins": [1, 9]}
        paths = {name: root / f"{name}.json" for name in ("fidelity", "retrieval", "replay")}
        for name, value in (("fidelity", healthy), ("retrieval", retrieval), ("replay", replay)):
            paths[name].write_text(json.dumps(value), encoding="utf-8")
        cases = {"healthy": ({}, 0), "fidelity_below_floor": ({"fidelity": {"models": [{"model": "Qwen2.5", "final_norm_fidelity": .9}, {"model": "Qwen3", "final_norm_fidelity": .91}]}}, 1),
                 "retrieval_147_239": ({"retrieval": {"retrieval_correct": 147, "retrieval_total": 239}}, 1),
                 "nonpositive_margin": ({"replay": {"margins": [1, 0]}}, 1),
                 "omitted_qwen25": ({"fidelity": {"models": [{"model": "Qwen3", "final_norm_fidelity": .91}]}}, 1),
                 "omitted_qwen3": ({"fidelity": {"models": [{"model": "Qwen2.5", "final_norm_fidelity": .91}]}}, 1)}
        results: dict[str, int] = {}
        for name, (changes, expected) in cases.items():
            for key, value in changes.items(): paths[key].write_text(json.dumps(value), encoding="utf-8")
            try: gate(manifest, root / f"{name}.out.json", paths["fidelity"], paths["retrieval"], paths["replay"]); exit_code = 0
            except ValueError: exit_code = 1
            results[name] = exit_code
            if exit_code != expected: return 2
            for key, value in (("fidelity", healthy), ("retrieval", retrieval), ("replay", replay)):
                paths[key].write_text(json.dumps(value), encoding="utf-8")
        corpus.write_text("changed corpus\n", encoding="utf-8")
        try: gate(manifest, root / "changed.out.json", paths["fidelity"], paths["retrieval"], paths["replay"]); results["changed_corpus_identity"] = 0
        except ValueError: results["changed_corpus_identity"] = 1
        if results["changed_corpus_identity"] != 1: return 2
        receipt.write_text(json.dumps({
            "id": "T-2713-release-gate",
            "instrument_sha256": sha256(Path(__file__)),
            "results": results,
            "status": "commissioned-seam-only; production release verdict quarantined pending pinned shipped inputs",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fidelity-report", type=Path)
    parser.add_argument("--retrieval-report", type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--commission", type=Path)
    args = parser.parse_args()
    if args.commission:
        return commission(args.commission)
    if not args.manifest or not args.report:
        parser.error("--manifest and --report are required outside --commission")
    try:
        print(json.dumps(gate(args.manifest, args.report, args.fidelity_report, args.retrieval_report, args.replay_report), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"T2704_RELEASE_GATE_REJECTED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
