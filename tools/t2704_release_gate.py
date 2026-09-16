#!/usr/bin/env python3
"""Fail-closed T-2704 release gate for the pinned real-model cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL_FIDELITY_FLOORS = {
    "Qwen3": {"value": .800623, "strictly_greater": True},
    "Qwen2.5": {"value": .938495, "strictly_greater": False},
}
QWEN3_FIDELITY_TARGET = .906603
RETRIEVAL_FLOOR, RETRIEVAL_TOTAL = 148, 239
REQUIRED_MODELS = {"Qwen2.5", "Qwen3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if member.is_file():
            digest.update(member.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256(member)))
    return digest.hexdigest()


def reject(message: str) -> None:
    raise ValueError(message)


def passes_fidelity_floor(model: str, fidelity: float) -> bool:
    rule = MODEL_FIDELITY_FLOORS[model]
    return fidelity > rule["value"] if rule["strictly_greater"] else fidelity >= rule["value"]


def fidelity_from_loss(loss: float) -> float:
    return round(1.0 - loss, 6)


def reject_fidelity_floor(model: str, fidelity: float) -> None:
    rule = MODEL_FIDELITY_FLOORS[model]
    comparator = "<=" if rule["strictly_greater"] else "<"
    reject(f"{model} fidelity {fidelity:.6f} {comparator} {rule['value']:.6f}")


def qwen3_target_report(fidelity: float) -> dict[str, float]:
    distance = round(abs(QWEN3_FIDELITY_TARGET - fidelity), 6)
    return {
        "final_norm_fidelity": fidelity,
        "target": QWEN3_FIDELITY_TARGET,
        "distance": distance,
        "shortfall": round(max(0.0, QWEN3_FIDELITY_TARGET - fidelity), 6),
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str, manifest: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else manifest.parent.parent.parent / candidate


def validate_models(manifest: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    rows = manifest.get("models")
    if not isinstance(rows, list):
        reject("manifest has no model rows")
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row["name"] in found:
            reject("model row is malformed or duplicate")
        artifact = row.get("artifact")
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or not isinstance(artifact.get("sha256"), str):
            reject(f"{row['name']} model row has no artifact path and SHA-256")
        file = resolve(artifact["path"], path)
        if not file.is_file():
            reject(f"missing pinned artifact: {row['name']}: {file}")
        digest = sha256(file)
        if digest != artifact["sha256"]:
            reject(f"changed pinned artifact: {row['name']}: expected {artifact['sha256']}, got {digest}")
        found[row["name"]] = {"row": row, "path": file, "sha256": digest}
    if set(found) != REQUIRED_MODELS:
        reject("manifest must have exactly Qwen2.5 and Qwen3 rows")
    return found


def validate_identities(manifest: dict[str, Any], path: Path) -> dict[str, str]:
    rows = manifest.get("identities")
    if not isinstance(rows, list) or not rows:
        reject("manifest has no pinned diagnostic identities")
    found = {}
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key] for key in ("name", "path", "sha256")):
            reject("diagnostic identity requires name, path, and SHA-256")
        file, kind = resolve(row["path"], path), row.get("kind", "file")
        if kind == "file" and file.is_file():
            digest = sha256(file)
        elif kind == "tree" and file.is_dir():
            digest = tree_sha256(file)
        else:
            reject(f"missing pinned {kind} identity: {row['name']}: {file}")
        if digest != row["sha256"]:
            reject(f"changed pinned identity: {row['name']}: expected {row['sha256']}, got {digest}")
        found[row["name"]] = digest
    return found


def run(command: list[str], label: str) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        reject(f"{label} diagnostic exited {completed.returncode}: {completed.stderr.strip()}")


def run_fidelity(config: dict[str, Any], models: dict[str, dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    runs = config.get("runs")
    if not isinstance(runs, list):
        reject("manifest has no fidelity runs")
    results, seen = [], set()
    for spec in runs:
        needed = ("model", "artifact_cache", "hf_model", "layer_trace")
        if not isinstance(spec, dict) or not all(isinstance(spec.get(key), str) for key in needed):
            reject("fidelity run is incomplete")
        model = spec["model"]
        if model not in models or model in seen:
            reject("fidelity runs must cover each model exactly once")
        output = work / ("fidelity-" + model.lower().replace(".", ""))
        command = ["python", "tools/t2703_fidelity_localization.py", "--artifact", str(models[model]["path"]),
                   "--artifact-cache", spec["artifact_cache"], "--hf-model", spec["hf_model"],
                   "--layer-trace", spec["layer_trace"], "--output", str(output)]
        if spec.get("control"):
            command.append("--control")
        run(command, f"{model} fidelity")
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            reject(f"{model} fidelity did not write summary.json")
        summary = load(summary_path)
        curves = summary.get("curves")
        if summary.get("artifact_sha256") != models[model]["sha256"] or not isinstance(curves, list) or not curves or curves[-1].get("pairing") != "final_norm":
            reject(f"{model} fidelity summary is not bound to its pinned artifact/final norm")
        loss = curves[-1].get("engine_vs_float", {}).get("median")
        if not isinstance(loss, (int, float)) or not 0 <= loss <= 1:
            reject(f"{model} final-norm loss is malformed")
        fidelity = fidelity_from_loss(float(loss))
        if not passes_fidelity_floor(model, fidelity):
            reject_fidelity_floor(model, fidelity)
        results.append({"model": model, "artifact_sha256": models[model]["sha256"], "final_norm_loss": loss,
                        "final_norm_fidelity": fidelity, "command": command, "summary": summary})
        seen.add(model)
    if seen != REQUIRED_MODELS:
        reject("fidelity runs omit Qwen2.5 or Qwen3")
    return results


def run_retrieval(config: dict[str, Any], models: dict[str, dict[str, Any]], work: Path) -> dict[str, Any]:
    needed = ("model", "corpus", "query_source", "hf_model", "probe", "provisional", "provisional_sha256")
    if not all(isinstance(config.get(key), str) for key in needed):
        reject("retrieval configuration is incomplete")
    output = work / "retrieval"
    arguments = ["--corpus", config["corpus"], "--query-source", config["query_source"], "--hf-model", config["hf_model"],
                 "--probe", config["probe"], "--candidate", str(models["Qwen3"]["path"]), "--candidate-sha256", models["Qwen3"]["sha256"],
                 "--provisional", config["provisional"], "--provisional-sha256", config["provisional_sha256"], "--output", str(output)]
    commands = [["python", "tools/t2703_retrieval_measure.py", phase, *arguments] for phase in ("float", "candidate", "provisional", "analyze")]
    for command in commands:
        run(command, f"retrieval {command[2]}")
    summary = load(output / "retrieval-summary.json")
    row = summary.get("arms", {}).get("candidate", {}).get("class_match_at_1", {})
    correct, total = row.get("count"), row.get("total")
    if not isinstance(correct, int) or total != RETRIEVAL_TOTAL:
        reject("retrieval report does not state candidate count over 239")
    if correct < RETRIEVAL_FLOOR:
        reject(f"retrieval {correct}/{total} < {RETRIEVAL_FLOOR}/{RETRIEVAL_TOTAL}")
    if summary.get("provenance", {}).get("candidate_artifact_sha256") != models["Qwen3"]["sha256"]:
        reject("retrieval candidate result is not bound to the pinned Qwen3 artifact")
    return {"model": config["model"], "retrieval_correct": correct, "retrieval_total": total, "commands": commands, "summary": summary}


def run_replay(config: dict[str, Any], models: dict[str, dict[str, Any]], work: Path) -> dict[str, Any]:
    if not all(isinstance(config.get(key), str) for key in ("qwen3_traces", "qwen2p5_traces")):
        reject("replay configuration is incomplete")
    output = work / "replay.json"
    command = ["python", "tools/t2723_t2704_production_refusal_replay.py", "--model", "both", "--qwen3-traces", config["qwen3_traces"],
               "--qwen3-artifact", str(models["Qwen3"]["path"]), "--qwen2p5-traces", config["qwen2p5_traces"],
               "--qwen2p5-artifact", str(models["Qwen2.5"]["path"]), "--output", str(output)]
    run(command, "real-trace replay")
    report, rows = load(output), load(output).get("models")
    if not isinstance(rows, dict) or set(rows) != {"qwen3", "qwen2p5"}:
        reject("replay report omits Qwen3 or Qwen2.5")
    for key, name in (("qwen3", "Qwen3"), ("qwen2p5", "Qwen2.5")):
        row = rows[key]
        if row.get("status") != "MEASURED" or row.get("artifact", {}).get("whole_file_sha256") != models[name]["sha256"]:
            reject(f"{name} replay is not measured against its pinned artifact")
        counts = row.get("refusals")
        if not isinstance(counts, dict) or any(value != 0 for value in counts.values()):
            reject(f"{name} replay has unexpected construction errors: {counts}")
    return {"command": command, "report": report}


def synthetic_fidelity(report: dict[str, Any]) -> dict[str, float]:
    rows, found = report.get("models"), {}
    if not isinstance(rows, list):
        reject("fidelity report has no model rows")
    for row in rows:
        name, fidelity = row.get("model"), row.get("final_norm_fidelity")
        if name not in REQUIRED_MODELS or name in found or not isinstance(fidelity, (int, float)):
            reject("fidelity report has malformed or duplicate row")
        found[name] = float(fidelity)
    if set(found) != REQUIRED_MODELS:
        reject("fidelity report omits a model or fails the threshold")
    for model, fidelity in found.items():
        if not passes_fidelity_floor(model, fidelity):
            reject_fidelity_floor(model, fidelity)
    return found


def gate(manifest_path: Path, report_path: Path, fidelity_path: Path | None = None, retrieval_path: Path | None = None, replay_path: Path | None = None) -> dict[str, Any]:
    manifest, models = load(manifest_path), None
    models = validate_models(manifest, manifest_path)
    identities = validate_identities(manifest, manifest_path)
    if fidelity_path or retrieval_path or replay_path:
        if not all((fidelity_path, retrieval_path, replay_path)):
            reject("commissioning requires all three synthetic reports")
        fidelity, retrieval, replay = synthetic_fidelity(load(fidelity_path)), load(retrieval_path), load(replay_path)
        correct, total, margins = retrieval.get("retrieval_correct"), retrieval.get("retrieval_total"), replay.get("margins")
        if not isinstance(correct, int) or total != RETRIEVAL_TOTAL or correct < RETRIEVAL_FLOOR or not isinstance(margins, list) or not margins or any(not isinstance(value, int) or value <= 0 for value in margins):
            reject("synthetic retrieval or replay report fails")
        result: dict[str, Any] = {"manifest_sha256": sha256(manifest_path), "identities": identities, "fidelity": fidelity,
                                  "qwen3_target": qwen3_target_report(fidelity["Qwen3"]), "retrieval_correct": correct,
                                  "minimum_replay_margin": min(margins), "commissioning_seam": True}
    else:
        diagnostics = manifest.get("diagnostics")
        if not isinstance(diagnostics, dict):
            reject("manifest has no diagnostics")
        work = report_path.parent / f".t2704-release-gate-work-{report_path.stem}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        fidelity = run_fidelity(diagnostics.get("fidelity", {}), models, work)
        qwen3_fidelity = next(row["final_norm_fidelity"] for row in fidelity if row["model"] == "Qwen3")
        result = {"manifest_sha256": sha256(manifest_path), "identities": identities,
                  "models": {name: {"path": str(value["path"]), "sha256": value["sha256"]} for name, value in models.items()},
                  "fidelity": fidelity, "qwen3_target": qwen3_target_report(qwen3_fidelity),
                  "retrieval": run_retrieval(diagnostics.get("retrieval", {}), models, work),
                  "replay": run_replay(diagnostics.get("replay", {}), models, work), "commissioning_seam": False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def commission(receipt: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="t2704-") as temporary:
        root = Path(temporary)
        models = []
        for name in sorted(REQUIRED_MODELS):
            artifact = root / f"{name}.sslm"; artifact.write_bytes(name.encode())
            models.append({"name": name, "artifact": {"path": artifact.name, "sha256": sha256(artifact)}})
        corpus = root / "corpus.jsonl"; corpus.write_text("corpus\n", encoding="utf-8")
        manifest = root / "tests/data/manifest.json"; manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"models": models, "identities": [{"name": "corpus", "path": corpus.name, "sha256": sha256(corpus)}], "diagnostics": {}}), encoding="utf-8")
        healthy = {"models": [{"model": "Qwen3", "final_norm_fidelity": .85},
                                {"model": "Qwen2.5", "final_norm_fidelity": .94}]}
        reports = {"fidelity": healthy, "retrieval": {"retrieval_correct": 148, "retrieval_total": 239}, "replay": {"margins": [1, 9]}}
        if fidelity_from_loss(.199377) != .800623:
            return 2
        paths = {name: root / f"{name}.json" for name in reports}
        for name, value in reports.items(): paths[name].write_text(json.dumps(value), encoding="utf-8")
        cases = {
            "healthy_qwen3_shortfall": ({}, 0),
            "qwen3_not_improved": ({"fidelity": {"models": [{"model": "Qwen3", "final_norm_fidelity": .800623}, {"model": "Qwen2.5", "final_norm_fidelity": .94}]}}, 1),
            "qwen25_regression": ({"fidelity": {"models": [{"model": "Qwen3", "final_norm_fidelity": .85}, {"model": "Qwen2.5", "final_norm_fidelity": .938494}]}}, 1),
            "retrieval_147_239": ({"retrieval": {"retrieval_correct": 147, "retrieval_total": 239}}, 1),
            "nonpositive_margin": ({"replay": {"margins": [1, 0]}}, 1),
            "omitted_qwen25": ({"fidelity": {"models": [{"model": "Qwen3", "final_norm_fidelity": .85}]}}, 1),
            "omitted_qwen3": ({"fidelity": {"models": [{"model": "Qwen2.5", "final_norm_fidelity": .94}]}}, 1),
        }
        results, healthy_qwen3_target = {}, None
        for name, (changes, expected) in cases.items():
            for key, value in changes.items(): paths[key].write_text(json.dumps(value), encoding="utf-8")
            try:
                report = gate(manifest, root / f"{name}.json.out", paths["fidelity"], paths["retrieval"], paths["replay"])
                if name == "healthy_qwen3_shortfall":
                    healthy_qwen3_target = report.get("qwen3_target")
                    if healthy_qwen3_target != {"final_norm_fidelity": .85, "target": .906603, "distance": .056603, "shortfall": .056603}:
                        return 2
                code = 0
            except ValueError: code = 1
            results[name] = code
            if code != expected: return 2
            for key, value in reports.items(): paths[key].write_text(json.dumps(value), encoding="utf-8")
        corpus.write_text("changed\n", encoding="utf-8")
        try: gate(manifest, root / "changed.json.out", paths["fidelity"], paths["retrieval"], paths["replay"]); results["changed_identity"] = 0
        except ValueError: results["changed_identity"] = 1
        if results["changed_identity"] != 1: return 2
        receipt.write_text(json.dumps({"id": "T-2713-release-gate", "instrument_sha256": sha256(Path(__file__)), "results": results,
                                      "healthy_qwen3_target": healthy_qwen3_target,
                                      "status": "commissioned-seam-only; production verdict requires real-data commissioning"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def write_rejection_report(manifest_path: Path | None, report_path: Path | None, error: Exception) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"verdict": "REJECTED", "decisive_witness": str(error)}
    if manifest_path is not None and manifest_path.is_file():
        result["manifest_sha256"] = sha256(manifest_path)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path); parser.add_argument("--report", type=Path)
    parser.add_argument("--fidelity-report", type=Path); parser.add_argument("--retrieval-report", type=Path); parser.add_argument("--replay-report", type=Path); parser.add_argument("--commission", type=Path)
    args = parser.parse_args()
    if args.commission: return commission(args.commission)
    if not args.manifest or not args.report: parser.error("--manifest and --report are required outside --commission")
    try: print(json.dumps(gate(args.manifest, args.report, args.fidelity_report, args.retrieval_report, args.replay_report), sort_keys=True)); return 0
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        write_rejection_report(args.manifest, args.report, error)
        print(f"T2704_RELEASE_GATE_REJECTED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
