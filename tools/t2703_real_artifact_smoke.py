#!/usr/bin/env python3
"""T-2703 real Qwen3 fused-QK smoke harness.

Build the required diagnostic from this repository before running:

    cmake -S . -B build -DSUPERSLM_BUILD_GPU=ON -DBUILD_TESTING=ON
    cmake --build build --target t2701_cpu_forward_probe

The probe's ``--smoke-attention`` mode asserts nonzero, varied Q31 scores and
non-uniform probabilities for every Q head at layers 0 and last. Its
``--dump-final-hidden`` payload is the production CPU forward's final RMSNorm
codes plus carried scale; this script decodes, last-token-pools, and L2
normalizes that vector before comparing it to the checkpoint's own float model
on exactly the same unpadded token IDs.

This is a structural smoke harness, not a calibration or release-quality
threshold deriver. It proves loading, the probe's CPU/GPU agreement path, and
finite non-degenerate final-hidden outputs. Cosine and rank agreement are
reported as observations only; the T-2704 Section 5 release gate grades
product quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


_FINAL_HIDDEN_MAGIC = 0x54474D5331373032
_SENTENCES = (
    "A small brown cat sleeps on a warm windowsill.",
    "A kitten is resting beside a sunny window.",
    "The spacecraft entered orbit around Mars yesterday.",
    "Engineers adjusted the satellite's orbit around the red planet.",
    "Fresh bread is cooling on the kitchen counter.",
    "The baker removed a warm loaf from the oven.",
    "A violinist practiced a difficult concerto after lunch.",
    "The musician rehearsed the violin passage carefully.",
    "Rainwater filled the narrow street after the storm.",
    "Heavy rain flooded the road during the thunderstorm.",
    "The database migration completed before midnight.",
    "Developers finished moving the production database overnight.",
    "A biologist measured coral growth near the reef.",
    "The team studied how quickly coral was growing in the ocean.",
    "The library closes at six o'clock on weekdays.",
    "Weekday visitors must leave the library before 6 PM.",
    "The chef seasoned the soup with smoked paprika.",
    "A cyclist repaired a flat tire beside the trail.",
    "The legal contract was signed by both companies.",
    "Mountain snow melted into the river during spring.",
    "A programmer optimized a matrix multiplication kernel.",
    "The concert audience applauded after the final song.",
    "The museum displayed a Roman bronze coin.",
    "A gardener planted tomatoes in rich soil.",
)
_RELATED_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15))
_UNRELATED_PAIRS = ((0, 18), (2, 16), (4, 20), (6, 22), (8, 14), (10, 23), (12, 17), (15, 19))


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                           f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed.stdout


def _read_final_hidden(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 32:
        raise ValueError(f"final-hidden dump is truncated: {path}")
    magic, count, m, e = struct.unpack_from("<QQqq", payload)
    if magic != _FINAL_HIDDEN_MAGIC or len(payload) != 32 + count:
        raise ValueError(f"final-hidden dump has invalid framing: {path}")
    codes = np.frombuffer(payload, dtype=np.int8, offset=32, count=count).astype(np.float64)
    return codes * math.ldexp(float(m), e)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("embedding has zero or non-finite L2 norm")
    return vector / norm


def _rank(values: list[float]) -> list[int]:
    return [index for index, _value in sorted(enumerate(values), key=lambda item: (-item[1], item[0]))]


def _spearman(left: list[float], right: list[float]) -> float:
    left_order = _rank(left)
    right_order = _rank(right)
    left_rank = {item: rank for rank, item in enumerate(left_order)}
    right_rank = {item: rank for rank, item in enumerate(right_order)}
    n = len(left)
    return 1.0 - 6.0 * sum((left_rank[i] - right_rank[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def _similarities(embeddings: list[np.ndarray], pairs: tuple[tuple[int, int], ...]) -> list[float]:
    return [float(np.dot(embeddings[left], embeddings[right])) for left, right in pairs]


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": float(min(values)), "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)), "mean": float(np.mean(values)), "max": float(max(values))}


def _float_embeddings(model_dir: Path, sentences: tuple[str, ...]):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    embeddings = []
    token_ids = []
    with torch.inference_mode():
        for sentence in sentences:
            encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
            ids = encoded["input_ids"][0].tolist()
            output = model(**{name: value.to(device) for name, value in encoded.items()})
            # README's Transformers example defines Qwen3-Embedding pooling as
            # the final non-padding token from outputs.last_hidden_state.
            float_hidden = output.last_hidden_state[0, -1].float().cpu().numpy()
            embeddings.append(_normalize(float_hidden))
            token_ids.append(ids)
    return embeddings, token_ids


_FORWARD_HASH_FIELDS = {"status", "sequence_state", "hidden", "kv_rows", "final_norm", "output_head", "aggregate"}


def _forward_hashes(output: str) -> dict[str, str]:
    fields = dict(re.findall(r"\b(status|sequence_state|hidden|kv_rows|final_norm|output_head|aggregate)=([0-9a-f]{64})\b", output))
    if set(fields) != _FORWARD_HASH_FIELDS:
        raise AssertionError("probe did not emit the complete forward byte-identity digest")
    return fields


def _assert_structural_smoke(verify_stdout: str, attention_passes: list[bool], cpu_gpu_agree: list[bool]) -> None:
    if "OK" not in verify_stdout:
        raise AssertionError("artifact verifier returned successfully without its acceptance marker")
    if not all(attention_passes):
        raise AssertionError("probe returned successfully without its attention PASS marker")
    if not all(cpu_gpu_agree):
        raise AssertionError("CPU and GPU forward byte-identity digests differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--verify", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=len(_SENTENCES),
                        help="provisional-only plumbing limit; real smoke uses the default 24")
    args = parser.parse_args()
    if not args.artifact.is_file() or not args.verify.is_file() or not args.probe.is_file():
        raise SystemExit("artifact, verifier, and probe must be existing files")
    if not args.hf_model.is_dir() or not 18 <= args.limit <= len(_SENTENCES):
        raise SystemExit(f"--hf-model must exist and --limit must be in [18,{len(_SENTENCES)}]")

    sentences = _SENTENCES[:args.limit]
    float_embeddings, token_ids = _float_embeddings(args.hf_model, sentences)
    with tempfile.TemporaryDirectory(prefix="t2703-real-smoke-") as temporary:
        root = Path(temporary)
        verify_output = root / "manifest.json"
        verify_stdout = _run([str(args.verify), str(args.artifact), str(verify_output)])
        engine_embeddings = []
        attention_passes = []
        cpu_gpu_agree = []
        for index, ids in enumerate(token_ids):
            dump = root / f"final-{index:02d}.bin"
            output = _run([str(args.probe), str(args.artifact), ",".join(map(str, ids)),
                           "--forward-hash", "--smoke-attention", "--dump-final-hidden", str(dump)])
            gpu_output = _run([str(args.probe), str(args.artifact), ",".join(map(str, ids)), "--forward-hash-gpu"])
            attention_passes.append("smoke_attention: PASS" in output)
            cpu_gpu_agree.append(_forward_hashes(output) == _forward_hashes(gpu_output))
            engine_embeddings.append(_normalize(_read_final_hidden(dump)))

    cosine = [float(np.dot(engine, floating)) for engine, floating in zip(engine_embeddings, float_embeddings)]
    _assert_structural_smoke(verify_stdout, attention_passes, cpu_gpu_agree)

    usable_related = tuple(pair for pair in _RELATED_PAIRS if max(pair) < len(sentences))
    usable_unrelated = tuple(pair for pair in _UNRELATED_PAIRS if max(pair) < len(sentences))
    if not usable_related or not usable_unrelated:
        raise AssertionError("sentence limit does not retain similarity controls")
    engine_related = _similarities(engine_embeddings, usable_related)
    engine_unrelated = _similarities(engine_embeddings, usable_unrelated)
    float_related = _similarities(float_embeddings, usable_related)
    float_unrelated = _similarities(float_embeddings, usable_unrelated)
    engine_pairs = engine_related + engine_unrelated
    float_pairs = float_related + float_unrelated
    rank_agreement = _spearman(engine_pairs, float_pairs)
    result = {
        "artifact": str(args.artifact),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "sslm_verify": "accepted" if "OK" in verify_stdout else "unexpected output",
        "sentence_count": len(sentences),
        "attention": "PASS",
        "cpu_gpu_byte_identity": "PASS",
        "quality_grading": "T-2704 Section 5 release gate",
        "cosine": _summary(cosine),
        "related_engine": engine_related,
        "unrelated_engine": engine_unrelated,
        "related_float": float_related,
        "unrelated_float": float_unrelated,
        "rank_agreement_spearman": rank_agreement,
        "engine_pair_rank": _rank(engine_pairs),
        "float_pair_rank": _rank(float_pairs),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
