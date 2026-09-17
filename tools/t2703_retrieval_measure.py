#!/usr/bin/env python3
"""T-2703 corpus-239 retrieval measurement through the compiled CPU engine.

The phases are deliberately resumable because each engine arm invokes the
release-tip CPU forward probe once per query/document.  ``float`` freezes and
validates the token IDs, runs the float32 anchor, and refuses to leave a valid
anchor unless class-match@1 is 146--147/239.  The two engine phases consume
those exact token IDs.  ``analyze`` writes the paired result and per-query
discordance without re-running an arm.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
END_TOKEN_ID = 151643
FINAL_HIDDEN_MAGIC = 0x54474D5331373032
EXPECTED_ROWS = 239
EXPECTED_HASHES = {
    "corpus": "275b0749a25cfd141e747111d4cbed66d7772ce58a50b5c13a394c5c0da222c0",
    "query_source": "0042a1e8d73f2648670be1e95c357c9388075fae278ea2b46605421da1af2ec4",
    "candidate": "87e58a201f4749f10c096ca2ef873e8af5840ac4432618d6b2240a686681c100",
    "provisional": "0d53db25017d49664f0b234515d5b4ea351d8dfce7b9a9005314141498a3ae5a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_rows(corpus: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"corpus has {len(rows)} rows, expected {EXPECTED_ROWS}")
    return rows


def verify_hash(path: Path, name: str, expected_override: str | None = None) -> str:
    actual = sha256(path)
    expected = expected_override or EXPECTED_HASHES.get(name)
    if expected is not None and actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: {actual} != {expected}")
    return actual


def source_prefix(query_source: Path) -> str:
    source = query_source.read_text(encoding="utf-8")
    definition = re.search(
        r'(?:constexpr|const)\s+char (?:semb::)?kQueryPrefix\[\]\s*=\s*'
        r'((?:"(?:\\.|[^"\\])*"\s*)+);',
        source,
    )
    if not definition:
        raise RuntimeError("could not locate kQueryPrefix in product query source")
    return "".join(
        json.loads(fragment)
        for fragment in re.findall(r'"(?:\\.|[^"\\])*"', definition.group(1))
    )


def class_match(query: np.ndarray, documents: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    # The shipping path promotes float32 components to double for cosine
    # accumulation.  Preserve that decision contract here for every arm.
    similarity = query.astype(np.float64, copy=False) @ documents.astype(np.float64, copy=False).T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    correct = classes[nearest] == classes
    return {
        "count": int(correct.sum()),
        "total": int(correct.size),
        "rate": float(correct.mean()),
        "nearest": nearest,
        "score": similarity[np.arange(correct.size), nearest],
        "correct": correct,
    }


def mcnemar_exact_p(left_only: int, right_only: int) -> float:
    # Same exact, two-sided binomial test implementation used by T-2657.
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise RuntimeError("zero or non-finite embedding norm")
    return matrix / norms


def load_token_rows(output: Path) -> list[dict[str, Any]]:
    path = output / "token-ids.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"frozen token file has {len(rows)} rows")
    return rows


def float_phase(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {
        "corpus": verify_hash(args.corpus, "corpus"),
        "query_source": verify_hash(args.query_source, "query_source"),
    }
    actual_prefix = source_prefix(args.query_source)
    if actual_prefix != QUERY_PREFIX:
        raise RuntimeError(f"query prefix differs from product source: {QUERY_PREFIX!r} != {actual_prefix!r}")

    rows = read_rows(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.hf_model), local_files_only=True, padding_side="left"
    )
    frozen: list[dict[str, Any]] = []
    mismatches: list[int] = []
    end_token_failures: list[int] = []
    for index, row in enumerate(rows):
        content = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        default = tokenizer(row["text"])["input_ids"]
        if content != row["content_ids"] or default != row["hf_default_ids"]:
            mismatches.append(index)
        if default != content + [END_TOKEN_ID] or default.count(END_TOKEN_ID) != 1:
            end_token_failures.append(index)
        query = tokenizer(QUERY_PREFIX + row["text"])["input_ids"]
        if not query or query[-1] != END_TOKEN_ID or query.count(END_TOKEN_ID) != 1:
            end_token_failures.append(index)
        frozen.append(
            {
                "index": index,
                "label": row["label"],
                "class": row["class"],
                "query_ids": query,
                "document_ids": list(row["hf_default_ids"]),
            }
        )
    if mismatches:
        raise RuntimeError(f"document tokenizer mismatch at rows {mismatches[:10]}")
    if end_token_failures:
        raise RuntimeError(f"terminal-token contract failed at rows {sorted(set(end_token_failures))[:10]}")

    token_path = args.output / "token-ids.jsonl"
    temporary = token_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in frozen),
        encoding="utf-8",
    )
    os.replace(temporary, token_path)
    hashes["token_ids"] = sha256(token_path)

    config = AutoConfig.from_pretrained(str(args.hf_model), local_files_only=True)
    hidden_size = int(config.hidden_size)
    all_ids = [row["query_ids"] for row in frozen] + [row["document_ids"] for row in frozen]
    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(all_ids), args.batch_size):
            batch_ids = all_ids[offset : offset + args.batch_size]
            width = max(len(ids) for ids in batch_ids)
            input_ids = torch.full(
                (len(batch_ids), width), int(tokenizer.pad_token_id), dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch_ids), width), dtype=torch.long, device=device
            )
            for row_index, ids in enumerate(batch_ids):
                input_ids[row_index, -len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row_index, -len(ids) :] = 1
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = output.last_hidden_state[:, -1, :]
            embeddings.append(normalize_rows(pooled.float().cpu().numpy()))
    wall_seconds = time.perf_counter() - start
    matrix = np.concatenate(embeddings).astype(np.float32, copy=False)
    if matrix.shape != (2 * EXPECTED_ROWS, hidden_size):
        raise RuntimeError(f"unexpected float matrix shape {matrix.shape}")
    np.save(args.output / "float-vectors.npy", matrix, allow_pickle=False)
    classes = np.asarray([row["class"] for row in frozen])
    result = class_match(matrix[:EXPECTED_ROWS], matrix[EXPECTED_ROWS:], classes)
    anchor_valid = result["count"] in (146, 147)
    float_result = {
        "anchor_valid": anchor_valid,
        "class_match_at_1": {key: result[key] for key in ("count", "total", "rate")},
        "device": str(device),
        "dtype": str(model.dtype),
        "hidden_size": hidden_size,
        "pooling": "last attended token, then L2 normalize",
        "query_prefix": QUERY_PREFIX,
        "terminal_token": {
            "id": END_TOKEN_ID,
            "policy": "one tokenizer-added terminal token in every query and document sequence",
        },
        "token_validation": {
            "document_content_and_default_ids_match_all_rows": True,
            "identical_query_and_document_ids_required_by_engine_phases": True,
            "rows": EXPECTED_ROWS,
        },
        "wall_seconds_including_model_load": wall_seconds,
        "hashes": hashes,
    }
    write_json(args.output / "float-result.json", float_result)
    if not anchor_valid:
        raise RuntimeError(
            f"float32 validity anchor failed: {result['count']}/{EXPECTED_ROWS}; refusing engine phases"
        )
    print(json.dumps(float_result, sort_keys=True))


def read_final_hidden(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 32:
        raise RuntimeError(f"final-hidden dump is truncated: {path}")
    magic, count, mantissa, exponent = struct.unpack_from("<QQqq", payload)
    if magic != FINAL_HIDDEN_MAGIC or len(payload) != 32 + count:
        raise RuntimeError(f"final-hidden dump has invalid framing: {path}")
    codes = np.frombuffer(payload, dtype=np.int8, offset=32, count=count).astype(np.float64)
    return codes * math.ldexp(float(mantissa), exponent)


def probe_equivalence_phase(args: argparse.Namespace) -> None:
    """Prove the retrieval fast path preserves the ordinary final-hidden bytes."""
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = load_token_rows(args.output)
    ids = frozen[0]["query_ids"]
    artifact_hash = verify_hash(args.candidate, "candidate", args.candidate_sha256)
    probe_hash = sha256(args.probe)
    copied_probe = args.output / args.probe.name
    shutil.copy2(args.probe, copied_probe)
    if sha256(copied_probe) != probe_hash:
        raise RuntimeError("copied equivalence probe hash mismatch")
    full_dump = args.output / "equivalence-full.bin"
    final_only_dump = args.output / "equivalence-final-only.bin"
    for dump in (full_dump, final_only_dump):
        if dump.exists():
            dump.unlink()

    def run_probe(dump: Path, final_hidden_only: bool) -> None:
        command = [
            str(args.probe), str(args.candidate), ",".join(map(str, ids)),
            "--dump-final-hidden", str(dump),
        ]
        if final_hidden_only:
            command.append("--final-hidden-only")
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"equivalence probe {'final-hidden-only' if final_hidden_only else 'ordinary'} "
                f"path exited {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if "forward_status=Ok" not in completed.stdout or not dump.is_file():
            raise RuntimeError(
                f"equivalence probe {'final-hidden-only' if final_hidden_only else 'ordinary'} "
                "path did not produce a successful final-hidden dump"
            )
        read_final_hidden(dump)

    run_probe(full_dump, False)
    run_probe(final_only_dump, True)
    equivalent = full_dump.read_bytes() == final_only_dump.read_bytes()
    result = {
        "artifact": str(args.candidate.resolve()),
        "artifact_sha256": artifact_hash,
        "comparison": "ordinary probe path versus --final-hidden-only on corpus row 0 query IDs",
        "equivalent": equivalent,
        "final_hidden_only_dump_sha256": sha256(final_only_dump),
        "full_path_dump_sha256": sha256(full_dump),
        "probe_sha256": probe_hash,
    }
    write_json(args.output / "probe-equivalence.json", result)
    if not equivalent:
        raise RuntimeError("ordinary and --final-hidden-only final-hidden dumps differ")
    print(json.dumps(result, sort_keys=True))


def validate_probe_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output / "probe-equivalence.json"
    if not path.is_file():
        raise RuntimeError("probe equivalence phase is required before analyze")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("equivalent") is not True:
        raise RuntimeError("probe equivalence phase did not establish byte identity")
    expected = {
        "artifact": str(args.candidate.resolve()),
        "artifact_sha256": verify_hash(args.candidate, "candidate", args.candidate_sha256),
        "probe_sha256": sha256(args.probe),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"probe equivalence {key} does not match current input")
    full_dump = args.output / "equivalence-full.bin"
    final_only_dump = args.output / "equivalence-final-only.bin"
    if not full_dump.is_file() or not final_only_dump.is_file():
        raise RuntimeError("probe equivalence dumps are missing")
    if record.get("full_path_dump_sha256") != sha256(full_dump):
        raise RuntimeError("ordinary probe equivalence dump hash mismatch")
    if record.get("final_hidden_only_dump_sha256") != sha256(final_only_dump):
        raise RuntimeError("final-hidden-only equivalence dump hash mismatch")
    if full_dump.read_bytes() != final_only_dump.read_bytes():
        raise RuntimeError("probe equivalence dumps differ")
    return record


def engine_phase(args: argparse.Namespace, arm: str) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    float_result = json.loads((args.output / "float-result.json").read_text(encoding="utf-8"))
    if not float_result.get("anchor_valid"):
        raise RuntimeError("valid float anchor is required before an engine phase")
    frozen = load_token_rows(args.output)
    expected_token_hash = float_result["hashes"]["token_ids"]
    if sha256(args.output / "token-ids.jsonl") != expected_token_hash:
        raise RuntimeError("frozen token IDs changed after the float phase")
    artifact = args.candidate if arm == "candidate" else args.provisional
    expected = args.candidate_sha256 if arm == "candidate" else args.provisional_sha256
    artifact_hash = verify_hash(artifact, arm, expected)
    probe_hash = sha256(args.probe)
    hidden_size = int(float_result["hidden_size"])
    sequences = [row["query_ids"] for row in frozen] + [row["document_ids"] for row in frozen]
    vector_path = args.output / f"{arm}-vectors.npy"
    progress_path = args.output / f"{arm}-progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["artifact_sha256"] != artifact_hash or progress["probe_sha256"] != probe_hash:
            raise RuntimeError(f"{arm} progress provenance does not match inputs")
        matrix = np.lib.format.open_memmap(vector_path, mode="r+")
    else:
        matrix = np.lib.format.open_memmap(
            vector_path, mode="w+", dtype=np.float64, shape=(len(sequences), hidden_size)
        )
        progress = {
            "arm": arm,
            "artifact": str(artifact.resolve()),
            "artifact_sha256": artifact_hash,
            "probe": str(args.probe.resolve()),
            "probe_sha256": probe_hash,
            "completed": 0,
            "done_indices": [],
            "total": len(sequences),
            "wall_seconds": 0.0,
            "probe_process_seconds": 0.0,
            "complete": False,
        }
        write_json(progress_path, progress)
    if tuple(matrix.shape) != (len(sequences), hidden_size):
        raise RuntimeError(f"{arm} vector matrix has unexpected shape {matrix.shape}")
    done = set(int(index) for index in progress.get("done_indices", range(int(progress["completed"]))))
    prior_wall = float(progress["wall_seconds"])
    phase_started = time.perf_counter()

    def run_one(index: int) -> tuple[int, np.ndarray, float]:
        dump = args.output / f".{arm}-final-hidden-{index:03d}.bin"
        if dump.exists():
            dump.unlink()
        command = [
            str(args.probe),
            str(artifact),
            ",".join(map(str, sequences[index])),
            "--dump-final-hidden",
            str(dump),
            "--final-hidden-only",
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(
                f"probe failed at {arm} sequence {index} ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if "forward_status=Ok" not in completed.stdout or not dump.is_file():
            raise RuntimeError(f"probe did not produce a successful final-hidden dump at {arm} sequence {index}")
        vector = read_final_hidden(dump)
        dump.unlink()
        if vector.size != hidden_size:
            raise RuntimeError(f"probe returned {vector.size} components, expected {hidden_size}")
        return index, normalize_rows(vector.reshape(1, -1))[0], elapsed

    pending = [index for index in range(len(sequences)) if index not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, index) for index in pending]
        for future in concurrent.futures.as_completed(futures):
            index, vector, elapsed = future.result()
            matrix[index] = vector
            matrix.flush()
            done.add(index)
            progress["done_indices"] = sorted(done)
            progress["completed"] = len(done)
            progress["wall_seconds"] = prior_wall + (time.perf_counter() - phase_started)
            progress["probe_process_seconds"] = float(progress.get("probe_process_seconds", 0.0)) + elapsed
            progress["complete"] = len(done) == len(sequences)
            write_json(progress_path, progress)
            if progress["completed"] % 10 == 0 or progress["complete"]:
                print(
                    f"{arm}: {progress['completed']}/{len(sequences)} "
                    f"({progress['wall_seconds']:.3f}s arm wall; {args.workers} workers)",
                    flush=True,
                )


def paired(engine: np.ndarray, floating: np.ndarray) -> dict[str, Any]:
    engine_only = int(np.count_nonzero(engine & ~floating))
    float_only = int(np.count_nonzero(~engine & floating))
    return {
        "engine_correct_float_wrong": engine_only,
        "engine_wrong_float_correct": float_only,
        "discordant_total": engine_only + float_only,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(engine_only, float_only),
    }


def analyze_phase(args: argparse.Namespace) -> None:
    equivalence = validate_probe_equivalence(args)
    frozen = load_token_rows(args.output)
    float_result = json.loads((args.output / "float-result.json").read_text(encoding="utf-8"))
    if not float_result.get("anchor_valid"):
        raise RuntimeError("cannot analyze without a valid float anchor")
    classes = np.asarray([row["class"] for row in frozen])
    vectors: dict[str, np.ndarray] = {
        "float": np.load(args.output / "float-vectors.npy", mmap_mode="r"),
    }
    progress: dict[str, dict[str, Any]] = {}
    for arm in ("candidate", "provisional"):
        progress[arm] = json.loads(
            (args.output / f"{arm}-progress.json").read_text(encoding="utf-8")
        )
        if not progress[arm].get("complete"):
            raise RuntimeError(f"{arm} engine arm is incomplete")
        vectors[arm] = np.load(args.output / f"{arm}-vectors.npy", mmap_mode="r")
    results = {
        name: class_match(matrix[:EXPECTED_ROWS], matrix[EXPECTED_ROWS:], classes)
        for name, matrix in vectors.items()
    }
    per_query_path = args.output / "per-query-discordance.jsonl"
    temporary = per_query_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(frozen):
            entry: dict[str, Any] = {
                "index": index,
                "label": row["label"],
                "class": row["class"],
                "arms": {},
            }
            for arm in ("float", "candidate", "provisional"):
                nearest = int(results[arm]["nearest"][index])
                entry["arms"][arm] = {
                    "nearest_index": nearest,
                    "nearest_label": frozen[nearest]["label"],
                    "nearest_class": frozen[nearest]["class"],
                    "cosine": float(results[arm]["score"][index]),
                    "correct": bool(results[arm]["correct"][index]),
                }
            for arm in ("candidate", "provisional"):
                engine_correct = bool(results[arm]["correct"][index])
                float_correct = bool(results["float"]["correct"][index])
                if engine_correct == float_correct:
                    category = "both_correct" if engine_correct else "both_wrong"
                else:
                    category = "engine_only" if engine_correct else "float_only"
                entry[f"{arm}_vs_float"] = category
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
    os.replace(temporary, per_query_path)

    arms_summary = {
        arm: {
            "class_match_at_1": {
                key: results[arm][key] for key in ("count", "total", "rate")
            },
            "wall_seconds": (
                float_result["wall_seconds_including_model_load"]
                if arm == "float"
                else progress[arm]["wall_seconds"]
            ),
        }
        for arm in ("float", "candidate", "provisional")
    }
    comparisons = {}
    for arm in ("candidate", "provisional"):
        comparisons[f"{arm}_vs_float"] = paired(
            results[arm]["correct"], results["float"]["correct"]
        )
        count = int(results[arm]["count"])
        comparisons[f"{arm}_vs_recorded"] = {
            "delta_from_shipped_engine_129": count - 129,
            "delta_from_simulated_fused_k_144": count - 144,
            "note": (
                "129/239 is the recorded shipped-engine result; 144/239 is a Python "
                "integer-reference simulation. This is the first compiled-engine reading "
                "for the 1.5.0 candidate."
            ),
        }
    summary = {
        "status": "complete",
        "contract": {
            "rows": EXPECTED_ROWS,
            "matching": "cosine query-to-document; diagonal excluded; nearest class equality",
            "pooling": "last attended token, then L2 normalize",
            "accumulation": "float64 dot product for ranking",
            "terminal_token_id": END_TOKEN_ID,
            "query_prefix": QUERY_PREFIX,
        },
        "arms": arms_summary,
        "paired_comparisons": comparisons,
        "power_limit": (
            "This 239-query population did not resolve the recorded 129-vs-144 difference: "
            "63 discordant pairs split 39/24, exact two-sided McNemar p=0.076926."
        ),
        "provenance": {
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": verify_hash(args.corpus, "corpus"),
            "query_source": str(args.query_source.resolve()),
            "query_source_sha256": verify_hash(args.query_source, "query_source"),
            "hf_model": str(args.hf_model.resolve()),
            "candidate_artifact": str(args.candidate.resolve()),
            "candidate_artifact_sha256": progress["candidate"]["artifact_sha256"],
            "provisional_artifact": str(args.provisional.resolve()),
            "provisional_artifact_sha256": progress["provisional"]["artifact_sha256"],
            "probe": str(args.probe.resolve()),
            "probe_sha256": progress["candidate"]["probe_sha256"],
            "token_ids_sha256": sha256(args.output / "token-ids.jsonl"),
            "probe_final_hidden_equivalence": equivalence,
        },
    }
    write_json(args.output / "retrieval-summary.json", summary)
    manifest_files = [
        "float-result.json",
        "float-vectors.npy",
        "token-ids.jsonl",
        "candidate-progress.json",
        "candidate-vectors.npy",
        "provisional-progress.json",
        "provisional-vectors.npy",
        "per-query-discordance.jsonl",
        "retrieval-summary.json",
        "probe-equivalence.json",
        "equivalence-full.bin",
        "equivalence-final-only.bin",
        args.probe.name,
    ]
    manifest = {
        "files": {
            name: {"sha256": sha256(args.output / name), "bytes": (args.output / name).stat().st_size}
            for name in manifest_files
        }
    }
    write_json(args.output / "sha256-manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("float", "candidate", "provisional", "equivalence", "analyze"))
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--query-source", required=True, type=Path)
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--provisional", required=True, type=Path)
    parser.add_argument("--candidate-sha256", help="pinned candidate identity from the release manifest")
    parser.add_argument("--provisional-sha256", help="pinned provisional control identity from the release manifest")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_files = (
        args.corpus,
        args.query_source,
        args.probe,
        args.candidate,
        args.provisional,
    )
    if not all(path.is_file() for path in required_files) or not args.hf_model.is_dir():
        raise SystemExit("corpus, source, probe, artifacts, and local HF model must exist")
    if args.batch_size < 1 or args.workers < 1:
        raise SystemExit("--batch-size and --workers must be positive")
    if args.phase == "float":
        float_phase(args)
    elif args.phase in ("candidate", "provisional"):
        engine_phase(args, args.phase)
    elif args.phase == "equivalence":
        probe_equivalence_phase(args)
    else:
        analyze_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
