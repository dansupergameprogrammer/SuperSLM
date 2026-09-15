#!/usr/bin/env python3
"""T-2703 all-position engine-state splice diagnostic.

This tool uses the existing read-only layer tracer on every causal prefix.  It
then feeds the dequantized engine residual vectors at every sequence position
into the artifact-weight int8 floor and continues with the floor's own blocks.
No production source or artifact is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_pipeline import artifact_cache  # noqa: E402
from t2703_fidelity_localization import (  # noqa: E402
    LONG_SENTENCES,
    SMOKE_SENTENCES,
    CaptureController,
    install_artifact_weights,
    patched_rope,
    read_engine_dump,
)


ENDPOINT_MARGIN = {
    "d_minus_1_max_item_cosine_loss": 1.0e-6,
    "d_minus_1_max_item_norm_ratio_error": 1.0e-6,
    "d_last_max_item_cosine_loss": 0.002,
    "d_last_max_item_norm_ratio_error": 0.002,
    "endpoint_statistic_absolute_error": 0.002,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denom) or denom == 0.0:
        raise ValueError("zero or non-finite norm")
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def norm_ratio(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64)) /
                 np.linalg.norm(np.asarray(right, dtype=np.float64)))


def distribution(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def final_curve(rows, floating) -> dict[str, dict[str, float]]:
    cosines = [cosine(rows[i], floating[i]) for i in range(len(rows))]
    ratios = [norm_ratio(rows[i], floating[i]) for i in range(len(rows))]
    return {"cosine": distribution(cosines), "norm_ratio": distribution(ratios)}


def run_trace(layer_trace: Path, artifact: Path, ids, dump: Path, stdout: Path,
              site_dump: Path | None = None, site_layer=None) -> np.ndarray:
    command = [str(layer_trace), str(artifact), "-", "--token-ids", ",".join(map(str, ids)),
               "--dump", str(dump), "--include-final-norm"]
    if site_dump is not None:
        command.extend(["--site-dump", str(site_dump)])
        if isinstance(site_layer, (tuple, list, set)):
            command.extend(["--site-layers", ",".join(map(str, site_layer))])
        else:
            command.extend(["--site-layer", str(site_layer)])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"layer trace failed ({completed.returncode}): {' '.join(command)}\n"
                           f"{completed.stdout}\n{completed.stderr}")
    return read_engine_dump(dump)


def capture_prefix_engine(layer_trace: Path, artifact: Path, token_ids, output: Path):
    root = output / "prefix-engine"
    root.mkdir(parents=True, exist_ok=True)
    all_items = []
    for item, ids in enumerate(token_ids):
        cache = root / f"item-{item:02d}.npz"
        if cache.is_file():
            with np.load(cache) as saved:
                if np.array_equal(saved["token_ids"], np.asarray(ids, dtype=np.int32)):
                    all_items.append(saved["states"])
                    continue
        item_root = root / f"item-{item:02d}"
        item_root.mkdir(parents=True, exist_ok=True)
        positions = []
        for position in range(len(ids)):
            stem = f"position-{position:03d}"
            rows = run_trace(layer_trace, artifact, ids[:position + 1],
                             item_root / f"{stem}.bin", item_root / f"{stem}.stdout.txt")
            positions.append(rows)
        states = np.stack(positions).astype(np.float32)
        np.savez_compressed(cache, states=states, token_ids=np.asarray(ids, dtype=np.int32))
        all_items.append(states)
        print(f"engine-prefix item={item + 1}/{len(token_ids)} positions={len(ids)}", flush=True)
    return all_items


def output_rows(output) -> np.ndarray:
    hidden = output.hidden_states
    rows = [value[0].detach().float().cpu().numpy() for value in hidden]
    rows.append(output.last_hidden_state[0].detach().float().cpu().numpy())
    return np.stack(rows, axis=1)


def capture_hf_rows(hf_path: Path, quantized_model, token_ids, floor: bool):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(hf_path), local_files_only=True, dtype=torch.float32,
                                      attn_implementation="eager").cuda().eval()
    if floor:
        install_artifact_weights(torch, model, quantized_model)
    controller = CaptureController(torch, quantized_model, floor)
    controller.install(model)
    rows = []
    try:
        with patched_rope(controller, model), torch.inference_mode():
            for ids in token_ids:
                controller.reset()
                input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                               output_hidden_states=True, use_cache=False, return_dict=True)
                rows.append(output_rows(output))
    finally:
        controller.close()
        del model
        torch.cuda.empty_cache()
    return rows


def _replace_input(torch, source):
    def hook(_module, inputs):
        value = torch.from_numpy(np.asarray(source, dtype=np.float32)).to(
            device=inputs[0].device, dtype=inputs[0].dtype).unsqueeze(0)
        return (value, *inputs[1:])
    return hook


def run_block_splices(hf_path: Path, quantized_model, token_ids, engine_positions,
                      floor_rows, floating_rows):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(hf_path), local_files_only=True, dtype=torch.float32,
                                      attn_implementation="eager").cuda().eval()
    install_artifact_weights(torch, model, quantized_model)
    controller = CaptureController(torch, quantized_model, True)
    controller.install(model)
    layers = quantized_model.config.num_hidden_layers
    final_vectors = np.empty((layers + 1, len(token_ids), quantized_model.config.hidden_size),
                             dtype=np.float32)
    try:
        with patched_rope(controller, model), torch.inference_mode():
            for splice_index, block in enumerate(range(-1, layers)):
                for item, ids in enumerate(token_ids):
                    controller.reset()
                    handle = None
                    if block >= 0:
                        target = model.norm if block == layers - 1 else model.layers[block + 1]
                        state = engine_positions[item][:, block + 1, :]
                        handle = target.register_forward_pre_hook(_replace_input(torch, state))
                    try:
                        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                       output_hidden_states=False, use_cache=False, return_dict=True)
                        final_vectors[splice_index, item] = (
                            output.last_hidden_state[0, -1].detach().float().cpu().numpy())
                    finally:
                        if handle is not None:
                            handle.remove()
                print(f"block-splice block={block} ({splice_index + 1}/{layers + 1})", flush=True)
    finally:
        controller.close()
        del model
        torch.cuda.empty_cache()

    floating_final = np.stack([rows[-1, -1] for rows in floating_rows])
    floor_final = np.stack([rows[-1, -1] for rows in floor_rows])
    engine_final = np.stack([rows[-1, -1] for rows in engine_positions])
    table = []
    previous_loss = None
    for index, block in enumerate(range(-1, layers)):
        curve = final_curve(final_vectors[index], floating_final)
        loss = 1.0 - curve["cosine"]["median"]
        table.append({
            "block": block,
            "final": curve,
            "median_cosine_loss": loss,
            "increment_median_cosine_loss": None if previous_loss is None else loss - previous_loss,
        })
        previous_loss = loss

    def endpoint(actual, expected):
        losses = [1.0 - cosine(actual[i], expected[i]) for i in range(len(actual))]
        ratios = [abs(norm_ratio(actual[i], expected[i]) - 1.0) for i in range(len(actual))]
        return {"cosine_loss": distribution(losses), "norm_ratio_absolute_error": distribution(ratios)}

    d_minus_1 = endpoint(final_vectors[0], floor_final)
    d_last = endpoint(final_vectors[-1], engine_final)
    floor_curve = final_curve(floor_final, floating_final)
    engine_curve = final_curve(engine_final, floating_final)
    endpoint_stat_error = {
        "d_minus_1_cosine": max(abs(table[0]["final"]["cosine"][key] - floor_curve["cosine"][key])
                                  for key in ("median", "p10", "p90")),
        "d_last_cosine": max(abs(table[-1]["final"]["cosine"][key] - engine_curve["cosine"][key])
                              for key in ("median", "p10", "p90")),
    }
    passed = (
        d_minus_1["cosine_loss"]["max"] <= ENDPOINT_MARGIN["d_minus_1_max_item_cosine_loss"] and
        d_minus_1["norm_ratio_absolute_error"]["max"] <=
        ENDPOINT_MARGIN["d_minus_1_max_item_norm_ratio_error"] and
        d_last["cosine_loss"]["max"] <= ENDPOINT_MARGIN["d_last_max_item_cosine_loss"] and
        d_last["norm_ratio_absolute_error"]["max"] <=
        ENDPOINT_MARGIN["d_last_max_item_norm_ratio_error"] and
        max(endpoint_stat_error.values()) <= ENDPOINT_MARGIN["endpoint_statistic_absolute_error"]
    )
    return final_vectors, table, {
        "declared_margin": ENDPOINT_MARGIN,
        "d_minus_1_vs_floor": d_minus_1,
        "d_last_vs_engine": d_last,
        "endpoint_statistic_absolute_error": endpoint_stat_error,
        "floor_curve": floor_curve,
        "engine_curve": engine_curve,
        "status": "PASS" if passed else "FAIL",
    }


def per_position_table(engine_positions, floating_rows, floor_rows):
    max_positions = max(len(rows) for rows in engine_positions)
    row_count = engine_positions[0].shape[1]
    table = []
    first_positive = None
    first_material = None
    for position in range(max_positions):
        members = [item for item, rows in enumerate(engine_positions) if position < len(rows)]
        position_rows = []
        for row in range(row_count):
            engine_loss = [1.0 - cosine(engine_positions[i][position, row], floating_rows[i][position, row])
                           for i in members]
            floor_loss = [1.0 - cosine(floor_rows[i][position, row], floating_rows[i][position, row])
                          for i in members]
            engine_ratio = [norm_ratio(engine_positions[i][position, row], floating_rows[i][position, row])
                            for i in members]
            floor_ratio = [norm_ratio(floor_rows[i][position, row], floating_rows[i][position, row])
                           for i in members]
            excess = float(np.median(engine_loss) - np.median(floor_loss))
            record = {
                "row": row,
                "pairing": "embedding" if row == 0 else (
                    "final_norm" if row == row_count - 1 else f"block_{row - 1}_residual"),
                "engine_cosine_loss": distribution(engine_loss),
                "floor_cosine_loss": distribution(floor_loss),
                "engine_norm_ratio": distribution(engine_ratio),
                "floor_norm_ratio": distribution(floor_ratio),
                "median_excess_cosine_loss": excess,
            }
            position_rows.append(record)
            if first_positive is None and excess > 0.0:
                first_positive = {"position": position, "row": row, "excess": excess}
            if first_material is None and excess >= 0.01:
                first_material = {"position": position, "row": row, "excess": excess}
        table.append({"position": position, "item_count": len(members), "rows": position_rows})
    return table, first_positive, first_material


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-cache", required=True, type=Path)
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--layer-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.artifact, args.layer_trace):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    for path in (args.artifact_cache, args.hf_model):
        if not path.is_dir():
            parser.error(f"missing directory: {path}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    sentences = SMOKE_SENTENCES[:6] if args.control else SMOKE_SENTENCES + LONG_SENTENCES
    token_ids = [tokenizer(sentence, add_special_tokens=True)["input_ids"] for sentence in sentences]
    quantized_model = artifact_cache.load_artifact(args.artifact_cache)
    if max(map(len, token_ids)) > quantized_model.config.context_cap:
        raise ValueError("input exceeds artifact context capacity")

    engine_positions = capture_prefix_engine(args.layer_trace.resolve(), args.artifact.resolve(),
                                             token_ids, args.output)
    floating_rows = capture_hf_rows(args.hf_model, quantized_model, token_ids, floor=False)
    floor_rows = capture_hf_rows(args.hf_model, quantized_model, token_ids, floor=True)
    table, first_positive, first_material = per_position_table(
        engine_positions, floating_rows, floor_rows)
    block_vectors, block_table, endpoint = run_block_splices(
        args.hf_model, quantized_model, token_ids, engine_positions, floor_rows, floating_rows)

    vector_path = args.output / "block-splice-vectors.npz"
    np.savez_compressed(vector_path, block_vectors=block_vectors,
                        floating_final=np.stack([rows[-1, -1] for rows in floating_rows]),
                        floor_final=np.stack([rows[-1, -1] for rows in floor_rows]),
                        engine_final=np.stack([rows[-1, -1] for rows in engine_positions]),
                        token_lengths=np.asarray(list(map(len, token_ids)), dtype=np.int32))
    first_ids = [ids[0] for ids in token_ids]
    position_zero = {
        "token_ids": first_ids,
        "tokens": tokenizer.convert_ids_to_tokens(first_ids),
        "decoded": [tokenizer.decode([token]) for token in first_ids],
        "all_same_token": len(set(first_ids)) == 1,
        "engine_embedding_max_absolute_pairwise_difference": float(max(
            np.max(np.abs(engine_positions[item][0, 0] - engine_positions[0][0, 0]))
            for item in range(len(engine_positions)))),
    }
    summary = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256(args.artifact),
        "hf_model": str(args.hf_model.resolve()),
        "item_count": len(token_ids),
        "token_lengths": distribution(list(map(float, map(len, token_ids)))),
        "position_zero": position_zero,
        "per_position": table,
        "first_positive_engine_excess": first_positive,
        "first_engine_excess_at_least_0p01": first_material,
        "block_splice": block_table,
        "endpoint_reproduction": endpoint,
        "evidence": {
            "block_vectors": str(vector_path.resolve()),
            "prefix_engine": str((args.output / "prefix-engine").resolve()),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path} endpoint={endpoint['status']} "
          f"first_material={first_material}", flush=True)
    if endpoint["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
