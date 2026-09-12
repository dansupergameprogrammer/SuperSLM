#!/usr/bin/env python3
"""T-2616: 28-layer local/propagated loss budget and deep-site localization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from reference_pipeline import pipeline
from reference_pipeline.artifact_cache import load_artifact


QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)
NUM_LAYERS = 28


def load_t2604():
    path = Path(__file__).parents[1] / "t2604" / "analyze.py"
    spec = importlib.util.spec_from_file_location("t2604_analyze", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_t2609():
    path = Path(__file__).parents[1] / "t2609" / "analyze.py"
    spec = importlib.util.spec_from_file_location("t2609_analyze", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scale_value(scale) -> float:
    return math.ldexp(float(scale[0]), int(scale[1]))


def cosine(candidate, reference) -> float:
    a = np.asarray(candidate, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / (na * nb))


def quantize_input_grid(values: np.ndarray):
    """Representation-only per-row dynamic grid used at a layer entry."""
    values = np.asarray(values, dtype=np.float64)
    codes = np.empty(values.shape, dtype=np.int8)
    scales = []
    for t, row in enumerate(values):
        target = max(float(np.max(np.abs(row), initial=0.0)) / 127.0,
                     np.finfo(np.float64).tiny)
        canonical = pipeline.canonical_scale(target)
        actual = scale_value(canonical)
        ratio = row / actual
        rounded = np.sign(ratio) * np.floor(np.abs(ratio) + 0.5)
        codes[t] = np.clip(rounded, -127, 127).astype(np.int8)
        scales.append(tuple(int(v) for v in canonical))
    return codes, scales


def write_local_input(path: Path, metadata: list[dict], arrays, first: int, last: int):
    count = len(metadata) * (last - first + 1)
    hidden = arrays[f"{metadata[0]['id']}__input__0"].shape[1]
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"T2616_LOCAL 1 {hidden} {count}\n")
        for item in metadata:
            for layer in range(first, last + 1):
                values = arrays[f"{item['id']}__input__{layer}"]
                codes, scales = quantize_input_grid(values)
                out.write(f"{item['id']} {layer} {len(values)}\n")
                for scale, row in zip(scales, codes):
                    out.write(f"{scale[0]} {scale[1]} " + " ".join(str(int(v)) for v in row) + "\n")


def capture(args) -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t2604 = load_t2604()
    specimens = t2604.selected_rows(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval().to(args.device)
    device = next(model.parameters()).device
    arrays = {}
    metadata = []
    with torch.no_grad():
        for path_kind in ("query", "document"):
            for sample_index, (_corpus_index, specimen) in enumerate(specimens):
                case_id = f"{path_kind}-{sample_index}"
                text = QUERY_PREFIX + specimen["text"] if path_kind == "query" else specimen["text"]
                ids = tokenizer(text)["input_ids"]
                token_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                steps = len(ids)
                position_ids = torch.arange(steps, device=device).unsqueeze(0)
                causal = torch.triu(
                    torch.full((steps, steps), float("-inf"), device=device), diagonal=1
                ).view(1, 1, steps, steps)
                hidden = model.embed_tokens(token_tensor)
                cos_table, sin_table = model.rotary_emb(hidden, position_ids)
                for layer_index, layer in enumerate(model.layers):
                    arrays[f"{case_id}__input__{layer_index}"] = hidden[0].float().cpu().numpy()
                    hidden = layer(
                        hidden, attention_mask=causal, position_ids=position_ids,
                        position_embeddings=(cos_table, sin_table), use_cache=False)
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                    arrays[f"{case_id}__output__{layer_index}"] = hidden[0].float().cpu().numpy()
                metadata.append({
                    "id": case_id, "path": path_kind, "sample_index": sample_index,
                    "label": specimen["label"], "class": specimen["class"],
                    "token_ids": [int(v) for v in ids], "token_count": steps,
                })
                print(f"captured {case_id}: {steps} tokens", flush=True)

    np.savez_compressed(args.out_dir / "float_layers.npz", **arrays)
    provenance = {
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "hf_model": str(args.hf_model), "float_dtype": "torch.float32",
        "population": "seven ASCII one-per-class T-2604 specimens per path",
        "input_grid": (
            "per-row absmax/127, canonical (m,e), round-half-away-from-zero, clamp [-127,127]"),
        "cases": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "propagated-input.txt").open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"T2616_PROP 1 {len(metadata)}\n")
        for item in metadata:
            out.write(f"{item['id']} {item['token_count']} " +
                      " ".join(str(v) for v in item["token_ids"]) + "\n")
    for first in range(0, NUM_LAYERS, 4):
        last = min(first + 3, NUM_LAYERS - 1)
        write_local_input(args.out_dir / f"local-{first:02d}-{last:02d}.txt",
                          metadata, arrays, first, last)
    print(json.dumps({"cases": len(metadata), "layers": NUM_LAYERS,
                      "elapsed_seconds": provenance["elapsed_seconds"]}, indent=2))
    return 0


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def decode_rows(rows, steps):
    ordered = sorted(rows, key=lambda row: int(row["token"]))
    if len(ordered) != steps:
        raise RuntimeError(f"expected {steps} output rows, got {len(ordered)}")
    return np.stack([
        np.asarray(row["codes"], dtype=np.float64) *
        math.ldexp(float(row["m_out"]), int(row["e_out"]))
        for row in ordered
    ])


def metric_pair(candidate, reference):
    return {
        "mean_cosine": cosine(candidate, reference),
        "terminal_cosine": cosine(candidate[-1], reference[-1]),
    }


def summarize(args) -> int:
    metadata_doc = json.loads((args.out_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = metadata_doc["cases"]
    arrays = np.load(args.out_dir / "float_layers.npz")
    local_rows = []
    for first in range(0, NUM_LAYERS, 4):
        last = min(first + 3, NUM_LAYERS - 1)
        local_rows.extend(read_jsonl(args.out_dir / f"local-{first:02d}-{last:02d}.jsonl"))
    propagated_rows = read_jsonl(args.out_dir / "propagated.jsonl")
    by_local = {}
    for row in local_rows:
        if row["kind"] == "output":
            by_local.setdefault((row["case"], int(row["layer"])), []).append(row)
    by_prop = {}
    for row in propagated_rows:
        if row["kind"] == "chain" and row["site"].endswith(".mlp_residual"):
            layer = int(row["site"].split(".", 1)[0][5:])
            by_prop.setdefault((row["case"], layer), []).append(row)

    specimens = []
    for item in metadata:
        for layer in range(NUM_LAYERS):
            reference = arrays[f"{item['id']}__output__{layer}"]
            local = decode_rows(by_local[(item["id"], layer)], item["token_count"])
            propagated = decode_rows(by_prop[(item["id"], layer)], item["token_count"])
            input_values = arrays[f"{item['id']}__input__{layer}"]
            input_codes, input_scales = quantize_input_grid(input_values)
            input_grid = np.stack([
                input_codes[t].astype(np.float64) * scale_value(input_scales[t])
                for t in range(item["token_count"])
            ])
            specimens.append({
                "case": item["id"], "path": item["path"], "layer": layer,
                "token_count": item["token_count"],
                "input_grid": metric_pair(input_grid, input_values),
                "local": metric_pair(local, reference),
                "propagated": metric_pair(propagated, reference),
            })

    aggregate = {kind: {} for kind in ("query", "document")}
    for kind in aggregate:
        for layer in range(NUM_LAYERS):
            cell = [r for r in specimens if r["path"] == kind and r["layer"] == layer]
            aggregate[kind][str(layer)] = {
                arm: {metric: float(np.mean([r[arm][metric] for r in cell]))
                      for metric in ("mean_cosine", "terminal_cosine")}
                for arm in ("input_grid", "local", "propagated")
            }
    result = {
        "provenance": metadata_doc | {
            "sslm_model": str(args.sslm_model),
            "sslm_sha256": hashlib.sha256(args.sslm_model.read_bytes()).hexdigest(),
            "cpp_probe": str(args.probe),
        },
        "aggregate": aggregate,
        "specimens": specimens,
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ranking = []
    for layer in range(NUM_LAYERS):
        loss = np.mean([1.0 - aggregate[k][str(layer)]["local"]["mean_cosine"]
                        for k in ("query", "document")])
        ranking.append((float(loss), layer))
    print(json.dumps({"worst_local_mean": sorted(ranking, reverse=True)[:8],
                      "result": str(args.result)}, indent=2))
    return 0


def make_site_input(args) -> int:
    metadata_doc = json.loads((args.out_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(args.out_dir / "float_layers.npz")
    write_local_input(args.output, metadata_doc["cases"], arrays,
                      args.layer, args.layer)
    return 0


def parity_key(row):
    return (row["kind"], row["site"], int(row["token"]), int(row["head"]))


def localize(args) -> int:
    import torch
    from transformers import AutoModel

    started = time.perf_counter()
    t2604 = load_t2604()
    metadata_doc = json.loads((args.out_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = metadata_doc["cases"]
    arrays = np.load(args.out_dir / "float_layers.npz")
    cpp_rows = read_jsonl(args.cpp_trace)
    cpp_by_case = {}
    for row in cpp_rows:
        cpp_by_case.setdefault(row["case"], []).append(row)
    propagated_by_case = {}
    if args.arm == "propagated":
        if args.layer == 0:
            raise ValueError("propagated site localization currently requires layer > 0")
        for row in read_jsonl(args.propagated_output):
            if (row["kind"] == "chain" and
                    row["site"] == f"layer{args.layer - 1}.mlp_residual"):
                propagated_by_case.setdefault(row["case"], []).append(row)

    model_int = load_artifact(args.integer_cache)
    model_fp32 = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval().to(args.device)
    site_rows = []
    exact_records = 0
    clamp_totals = {name: 0 for name in
                    ("kv_landing", "k_norm_landing", "rope_q", "rope_k")}
    for item in metadata:
        input_values = arrays[f"{item['id']}__input__{args.layer}"]
        if args.arm == "local":
            input_codes, input_scales = quantize_input_grid(input_values)
        else:
            source = sorted(propagated_by_case[item["id"]], key=lambda row: int(row["token"]))
            if len(source) != item["token_count"]:
                raise RuntimeError(f"missing propagated input rows for {item['id']}")
            input_codes = np.stack([np.asarray(row["codes"], dtype=np.int8) for row in source])
            input_scales = [(int(row["m_out"]), int(row["e_out"])) for row in source]
        ids = item["token_ids"]
        float_sites = t2604.float_block0(
            model_fp32, ids, layer_index=args.layer, input_hidden=input_values)
        integer_sites = t2604.integer_block0(
            model_int, ids, layer_index=args.layer,
            input_codes=input_codes, input_scales=input_scales)

        expected = {}
        for row in integer_sites["_cpp_expected"]:
            expected[parity_key(row)] = row
        actual = {}
        for row in cpp_by_case[item["id"]]:
            if row["kind"] in ("chain", "site"):
                comparable = {k: v for k, v in row.items() if k != "case"}
                actual[parity_key(comparable)] = comparable
            elif row["kind"] == "clamps":
                for name in clamp_totals:
                    clamp_totals[name] += int(row[name])
        if expected.keys() != actual.keys():
            missing = sorted(expected.keys() - actual.keys())[:5]
            extra = sorted(actual.keys() - expected.keys())[:5]
            raise RuntimeError(f"C++ parity record set differs for {item['id']}: {missing=} {extra=}")
        mismatches = [key for key in expected if expected[key] != actual[key]]
        if mismatches:
            raise RuntimeError(f"C++ differs from Python integer at {item['id']}: {mismatches[:5]}")
        exact_records += len(expected)

        for site in t2604.SITE_ORDER[1:]:
            metric = t2604.metrics(integer_sites[site], float_sites[site])
            terminal = t2604.metrics(
                t2604.terminal_site(integer_sites[site]),
                t2604.terminal_site(float_sites[site]))
            site_rows.append({
                "case": item["id"], "path": item["path"], "layer": args.layer,
                "site": site, "cosine": metric["cosine"],
                "terminal_cosine": terminal["cosine"],
                "max_abs_relative_error": metric["max_abs_relative_error"],
            })
        print(f"localized layer {args.layer} {item['id']}", flush=True)

    aggregate = {}
    for kind in ("query", "document"):
        aggregate[kind] = {}
        for site in t2604.SITE_ORDER[1:]:
            cell = [r for r in site_rows if r["path"] == kind and r["site"] == site]
            aggregate[kind][site] = {
                key: float(np.mean([r[key] for r in cell]))
                for key in ("cosine", "terminal_cosine", "max_abs_relative_error")
            }
    result = {
        "provenance": metadata_doc | {
            "layer": args.layer, "arm": args.arm, "integer_cache": str(args.integer_cache),
            "cpp_trace": str(args.cpp_trace), "float_dtype": "torch.float32",
        },
        "cpp_vs_python_integer": {"exact_records": exact_records, "mismatch_records": 0},
        "clamps": clamp_totals,
        "aggregate": aggregate,
        "specimens": site_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layer": args.layer, "exact_records": exact_records,
                      "aggregate": aggregate, "elapsed_seconds": result["elapsed_seconds"]},
                     indent=2))
    return 0


def input_for_arm(args, item, arrays, propagated_by_case):
    input_values = arrays[f"{item['id']}__input__{args.layer}"]
    if args.arm == "local":
        return input_values, *quantize_input_grid(input_values)
    source = sorted(propagated_by_case[item["id"]], key=lambda row: int(row["token"]))
    if len(source) != item["token_count"]:
        raise RuntimeError(f"missing propagated input rows for {item['id']}")
    codes = np.stack([np.asarray(row["codes"], dtype=np.int8) for row in source])
    scales = [(int(row["m_out"]), int(row["e_out"])) for row in source]
    return input_values, codes, scales


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def counterfactual(args) -> int:
    import torch
    from transformers import AutoModel
    from reference_pipeline import intmath

    started = time.perf_counter()
    t2604 = load_t2604()
    metadata_doc = json.loads((args.out_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = metadata_doc["cases"]
    arrays = np.load(args.out_dir / "float_layers.npz")
    propagated_by_case = {}
    if args.arm == "propagated":
        for row in read_jsonl(args.propagated_output):
            if (row["kind"] == "chain" and
                    row["site"] == f"layer{args.layer - 1}.mlp_residual"):
                propagated_by_case.setdefault(row["case"], []).append(row)
    model_int = load_artifact(args.integer_cache)
    model_fp32 = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval().to(args.device)
    requested = args.variants.split(",")
    rows = []
    token_diagnostics = []
    for item in metadata:
        input_values, input_codes, input_scales = input_for_arm(
            args, item, arrays, propagated_by_case)
        ids = item["token_ids"]
        float_sites = t2604.float_block0(
            model_fp32, ids, layer_index=args.layer, input_hidden=input_values)
        baseline = t2604.integer_block0(
            model_int, ids, layer_index=args.layer,
            input_codes=input_codes, input_scales=input_scales)
        variants = {"baseline": baseline["residual2"]}
        grids = {
            "attention_context": quantize_input_grid(
                np.asarray(float_sites["attention_context"]).reshape(item["token_count"], -1)),
            "mlp_activation": quantize_input_grid(float_sites["mlp_activation"]),
            "down_proj": quantize_input_grid(float_sites["down_proj"]),
            "residual1_at_final_add": quantize_input_grid(float_sites["residual1"]),
        }
        for name in requested:
            if name == "residual2_ceiling":
                codes, scales = quantize_input_grid(float_sites["residual2"])
                variants[name] = np.stack([
                    codes[t].astype(np.float64) * scale_value(scales[t])
                    for t in range(item["token_count"])
                ])
                continue
            if name == "softmax":
                stream = []
                for token in float_sites["softmax"]:
                    for row in np.asarray(token, dtype=np.float64):
                        stream.extend(max(1, int(round(float(p) * (1 << 50)))) for p in row)
                cursor = 0
                original = intmath.i_exp_from_constants

                def supplied(_q, _q_ln2, _q_b, _q_c):
                    nonlocal cursor
                    value = stream[cursor]
                    cursor += 1
                    return value

                intmath.i_exp_from_constants = supplied
                try:
                    walked = t2604.integer_block0(
                        model_int, ids, layer_index=args.layer,
                        input_codes=input_codes, input_scales=input_scales)
                finally:
                    intmath.i_exp_from_constants = original
                if cursor != len(stream):
                    raise RuntimeError(f"softmax override consumed {cursor}/{len(stream)} values")
                variants[name] = walked["residual2"]
                continue
            override_names = ("down_proj", "residual1_at_final_add") if name == "both_final_operands" else (name,)
            overrides = {key: grids[key] for key in override_names}
            walked = t2604.integer_block0(
                model_int, ids, layer_index=args.layer,
                input_codes=input_codes, input_scales=input_scales,
                site_overrides=overrides)
            variants[name] = walked["residual2"]
        reference = float_sites["residual2"]
        baseline_mean = cosine(variants["baseline"], reference)
        baseline_terminal = cosine(variants["baseline"][-1], reference[-1])
        metrics = {}
        for name, values in variants.items():
            mean_value = cosine(values, reference)
            terminal_value = cosine(values[-1], reference[-1])
            metrics[name] = {
                "mean_cosine": mean_value, "terminal_cosine": terminal_value,
                "mean_loss_recovered_fraction": ((mean_value - baseline_mean) /
                                                   max(1.0 - baseline_mean, np.finfo(float).tiny)),
                "terminal_loss_recovered_fraction": ((terminal_value - baseline_terminal) /
                                                       max(1.0 - baseline_terminal, np.finfo(float).tiny)),
            }
        rows.append({"case": item["id"], "path": item["path"], "metrics": metrics})

        float_r1 = np.asarray(float_sites["residual1"], dtype=np.float64)
        float_down = np.asarray(float_sites["down_proj"], dtype=np.float64)
        int_weighted = np.asarray(baseline["weighted_sum"], dtype=np.float64)
        float_weighted = np.asarray(float_sites["weighted_sum"], dtype=np.float64)
        for t in range(item["token_count"]):
            probs = np.asarray(float_sites["softmax"][t], dtype=np.float64)
            cancellation = (np.linalg.norm(float_r1[t] + float_down[t]) /
                            max(np.linalg.norm(float_r1[t]) + np.linalg.norm(float_down[t]),
                                np.finfo(float).tiny))
            token_diagnostics.append({
                "case": item["id"], "path": item["path"], "width": t + 1,
                "attention_sharpness_mean_max_probability": float(np.mean(np.max(probs, axis=1))),
                "weighted_sum_loss": 1.0 - cosine(int_weighted[t], float_weighted[t]),
                "residual2_loss": 1.0 - cosine(variants["baseline"][t], reference[t]),
                "float_final_residual_cancellation_ratio": float(cancellation),
            })
        print(f"counterfactual layer {args.layer} {args.arm} {item['id']}", flush=True)

    aggregate = {}
    for kind in ("query", "document"):
        aggregate[kind] = {}
        cell = [row for row in rows if row["path"] == kind]
        for name in ("baseline", *requested):
            aggregate[kind][name] = {
                key: float(np.mean([row["metrics"][name][key] for row in cell]))
                for key in ("mean_cosine", "terminal_cosine", "mean_loss_recovered_fraction",
                            "terminal_loss_recovered_fraction")
            }
    dependence = {}
    for kind in ("query", "document"):
        cell = [row for row in token_diagnostics if row["path"] == kind]
        dependence[kind] = {
            "tokens": len(cell),
            "width_vs_weighted_sum_loss_pearson": corr(
                [r["width"] for r in cell], [r["weighted_sum_loss"] for r in cell]),
            "sharpness_vs_weighted_sum_loss_pearson": corr(
                [r["attention_sharpness_mean_max_probability"] for r in cell],
                [r["weighted_sum_loss"] for r in cell]),
            "cancellation_ratio_vs_residual2_loss_pearson": corr(
                [r["float_final_residual_cancellation_ratio"] for r in cell],
                [r["residual2_loss"] for r in cell]),
            "terminal": {
                key: float(np.mean([r[key] for r in cell if r["width"] ==
                                    next(i["token_count"] for i in metadata if i["id"] == r["case"])]))
                for key in ("attention_sharpness_mean_max_probability", "weighted_sum_loss",
                            "residual2_loss", "float_final_residual_cancellation_ratio")
            },
        }
    result = {
        "provenance": metadata_doc | {"layer": args.layer, "arm": args.arm,
                                       "variants": requested},
        "aggregate": aggregate, "dependence": dependence,
        "specimens": rows, "token_diagnostics": token_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "dependence": dependence,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0


def diagnose_attention(args) -> int:
    import torch
    from transformers import AutoModel

    t2604 = load_t2604()
    t2609 = load_t2609()
    metadata_doc = json.loads((args.out_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = metadata_doc["cases"]
    arrays = np.load(args.out_dir / "float_layers.npz")
    model_int = load_artifact(args.integer_cache)
    model_fp32 = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval().to(args.device)
    rows = []
    for item in metadata:
        input_values = arrays[f"{item['id']}__input__{args.layer}"]
        input_codes, input_scales = quantize_input_grid(input_values)
        float_sites = t2604.float_block0(
            model_fp32, item["token_ids"], layer_index=args.layer, input_hidden=input_values)
        integer_sites = t2604.integer_block0(
            model_int, item["token_ids"], layer_index=args.layer,
            input_codes=input_codes, input_scales=input_scales)
        _scores, hybrid_probs = t2609.score_hybrids(
            model_int, integer_sites, float_sites, layer_index=args.layer)
        exact_integer = t2609.exact_probabilities(integer_sites["attention_scores"])
        kernel_float, _quant = t2609.kernel_probabilities_from_float_scores(
            model_int, integer_sites, float_sites["attention_scores"], layer_index=args.layer)
        metrics = {
            "baseline": t2609.cosine(integer_sites["softmax"], float_sites["softmax"]),
            "exact_on_integer_scores": t2609.cosine(exact_integer, float_sites["softmax"]),
            "integer_kernel_on_float_scores": t2609.cosine(kernel_float, float_sites["softmax"]),
            "float_q_integer_k": t2609.cosine(hybrid_probs["float_q_integer_k"], float_sites["softmax"]),
            "integer_q_float_k": t2609.cosine(hybrid_probs["integer_q_float_k"], float_sites["softmax"]),
            "integer_q_unlanded_integer_k": t2609.cosine(
                hybrid_probs["integer_q_unlanded_integer_k"], float_sites["softmax"]),
        }
        rows.append({"case": item["id"], "path": item["path"], "metrics": metrics})
        print(f"diagnosed attention layer {args.layer} {item['id']}", flush=True)
    aggregate = {
        kind: {key: float(np.mean([r["metrics"][key] for r in rows if r["path"] == kind]))
               for key in rows[0]["metrics"]}
        for kind in ("query", "document")
    }
    result = {"provenance": metadata_doc | {"layer": args.layer},
              "aggregate": aggregate, "specimens": rows}
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


def run_probe(args) -> int:
    command = [str(args.probe), args.mode, str(args.sslm_model), str(args.input), str(args.output)]
    if args.layer is not None:
        command.append(str(args.layer))
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capture_p = sub.add_parser("capture")
    capture_p.add_argument("--corpus", type=Path, required=True)
    capture_p.add_argument("--hf-model", type=Path, required=True)
    capture_p.add_argument("--out-dir", type=Path, required=True)
    capture_p.add_argument("--device", default="cuda")
    capture_p.set_defaults(func=capture)

    run_p = sub.add_parser("run-probe")
    run_p.add_argument("--probe", type=Path, required=True)
    run_p.add_argument("--mode", choices=("local", "sites", "propagated", "propagated-sites"), required=True)
    run_p.add_argument("--sslm-model", type=Path, required=True)
    run_p.add_argument("--input", type=Path, required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--layer", type=int)
    run_p.set_defaults(func=run_probe)

    summary_p = sub.add_parser("summarize")
    summary_p.add_argument("--out-dir", type=Path, required=True)
    summary_p.add_argument("--sslm-model", type=Path, required=True)
    summary_p.add_argument("--probe", type=Path, required=True)
    summary_p.add_argument("--result", type=Path, required=True)
    summary_p.set_defaults(func=summarize)

    input_p = sub.add_parser("make-site-input")
    input_p.add_argument("--out-dir", type=Path, required=True)
    input_p.add_argument("--layer", type=int, required=True)
    input_p.add_argument("--output", type=Path, required=True)
    input_p.set_defaults(func=make_site_input)

    localize_p = sub.add_parser("localize")
    localize_p.add_argument("--out-dir", type=Path, required=True)
    localize_p.add_argument("--layer", type=int, required=True)
    localize_p.add_argument("--integer-cache", type=Path, required=True)
    localize_p.add_argument("--hf-model", type=Path, required=True)
    localize_p.add_argument("--cpp-trace", type=Path, required=True)
    localize_p.add_argument("--result", type=Path, required=True)
    localize_p.add_argument("--arm", choices=("local", "propagated"), default="local")
    localize_p.add_argument("--propagated-output", type=Path)
    localize_p.add_argument("--device", default="cuda")
    localize_p.set_defaults(func=localize)

    cf_p = sub.add_parser("counterfactual")
    cf_p.add_argument("--out-dir", type=Path, required=True)
    cf_p.add_argument("--layer", type=int, required=True)
    cf_p.add_argument("--arm", choices=("local", "propagated"), required=True)
    cf_p.add_argument("--variants", required=True)
    cf_p.add_argument("--integer-cache", type=Path, required=True)
    cf_p.add_argument("--hf-model", type=Path, required=True)
    cf_p.add_argument("--propagated-output", type=Path)
    cf_p.add_argument("--result", type=Path, required=True)
    cf_p.add_argument("--device", default="cuda")
    cf_p.set_defaults(func=counterfactual)

    diag_p = sub.add_parser("diagnose-attention")
    diag_p.add_argument("--out-dir", type=Path, required=True)
    diag_p.add_argument("--layer", type=int, required=True)
    diag_p.add_argument("--integer-cache", type=Path, required=True)
    diag_p.add_argument("--hf-model", type=Path, required=True)
    diag_p.add_argument("--result", type=Path, required=True)
    diag_p.add_argument("--device", default="cuda")
    diag_p.set_defaults(func=diagnose_attention)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
