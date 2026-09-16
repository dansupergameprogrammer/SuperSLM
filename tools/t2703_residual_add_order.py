#!/usr/bin/env python3
"""T-2703 all-position residual-grid and add-order causal diagnostic."""

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
)
from t2703_fidelity_splice import (  # noqa: E402
    cosine,
    distribution,
    final_curve,
    norm_ratio,
    sha256,
)


MARGINS = {
    "blocks_20_27_maximum_median_loss_curve_absolute_error": 0.005,
    "blocks_20_27_maximum_increment_absolute_error": 0.003,
    "final_cosine_statistic_absolute_error": 0.005,
    "final_median_norm_ratio_absolute_error": 0.01,
    "qwen2p5_final_cosine_statistic_absolute_error": 0.005,
    "qwen2p5_final_median_norm_ratio_absolute_error": 0.01,
}


def physical(m: int, e: int) -> float:
    return math.ldexp(float(m), int(e))


def round_away_np(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


class GridAggregate:
    def __init__(self):
        self.quantum = []
        self.branch_abs = []
        self.zero = 0
        self.one = 0
        self.total = 0
        self.relative_l2 = []

    def add(self, quantum, branch, landed_codes):
        quantum = np.asarray(quantum, dtype=np.float64).reshape(-1)
        branch = np.asarray(branch, dtype=np.float64).reshape(-1, branch.shape[-1])
        landed_codes = np.asarray(landed_codes, dtype=np.int64).reshape(branch.shape)
        if quantum.size != branch.shape[0]:
            raise ValueError(f"quantum/row mismatch: {quantum.size} != {branch.shape[0]}")
        self.quantum.extend(quantum.tolist())
        self.branch_abs.append(np.abs(branch).astype(np.float32).reshape(-1))
        self.zero += int(np.count_nonzero(landed_codes == 0))
        self.one += int(np.count_nonzero(np.abs(landed_codes) == 1))
        self.total += int(landed_codes.size)
        survived = landed_codes * quantum[:, None]
        numer = np.linalg.norm(survived, axis=1)
        denom = np.linalg.norm(branch, axis=1)
        ratios = np.divide(numer, denom, out=np.ones_like(numer), where=denom != 0.0)
        self.relative_l2.extend(ratios.tolist())

    def summary(self):
        magnitude = np.concatenate(self.branch_abs) if self.branch_abs else np.asarray([0.0])
        return {
            "positions": len(self.quantum),
            "components": self.total,
            "stream_quantum": distribution(self.quantum),
            "branch_absolute_magnitude": distribution(magnitude),
            "zero_rounded_fraction": self.zero / self.total,
            "plus_or_minus_one_fraction": self.one / self.total,
            "surviving_relative_l2": distribution(self.relative_l2),
        }


def grid_table(aggregates, layers):
    return [{"block": layer,
             "attention": aggregates[(layer, "attention")].summary(),
             "mlp": aggregates[(layer, "mlp")].summary()}
            for layer in range(layers)]


class ResidualPolicy:
    def __init__(self, mode="standard", selected=(), observe=False):
        self.mode = mode
        self.selected = set(selected)
        self.observe = observe
        self.aggregates = {}

    @staticmethod
    def _round(torch, value):
        return torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))

    def quantize(self, torch, value, bits=8):
        limit = float((1 << (bits - 1)) - 1)
        wide = value if bits == 8 else value.double()
        peak = wide.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(peak > 0, peak / limit, torch.ones_like(peak))
        codes = self._round(torch, wide / scale).clamp(-limit, limit)
        return (codes * scale).to(value.dtype), codes.to(torch.int64), scale

    def __call__(self, torch, layer, site, stream, branch, nominal, standard_landed):
        stream_q, stream_codes, stream_scale = self.quantize(torch, stream, 8)
        _branch_q, branch_codes, branch_scale = self.quantize(torch, branch, 8)
        branch_on_stream = self._round(
            torch, branch_codes.double() * branch_scale / stream_scale).to(torch.int64)
        if self.observe:
            key = (layer, site)
            aggregate = self.aggregates.setdefault(key, GridAggregate())
            aggregate.add(stream_scale.detach().cpu().numpy(),
                          branch.detach().double().cpu().numpy(),
                          branch_on_stream.detach().cpu().numpy())

        if layer not in self.selected or self.mode == "standard":
            return standard_landed
        if self.mode == "engine_grid":
            wide = (stream_codes + branch_on_stream).double() * stream_scale
            return self.quantize(torch, wide.to(nominal.dtype), 8)[0]
        if self.mode == "common_finer":
            common = torch.minimum(stream_scale, branch_scale)
            stream_common = self._round(torch, stream_q.double() / common).to(torch.int64)
            branch_common = self._round(torch, branch.double() / common).to(torch.int64)
            wide = (stream_common + branch_common).double() * common
            return self.quantize(torch, wide.to(nominal.dtype), 8)[0]
        if self.mode in {"wide16", "wide32"}:
            bits = 16 if self.mode == "wide16" else 32
            stream_q, stream_codes, stream_scale = self.quantize(torch, stream, bits)
            branch_on_stream = self._round(
                torch, branch.double() / stream_scale).to(torch.int64)
            wide = (stream_codes + branch_on_stream).double() * stream_scale
            return self.quantize(torch, wide.to(nominal.dtype), bits)[0]
        raise ValueError(f"unknown residual policy mode: {self.mode}")


class ResidualController(CaptureController):
    def __init__(self, torch, quantized_model, policy):
        self.policy = policy
        super().__init__(torch, quantized_model, True)

    def reset(self):
        super().reset()
        self.mlp_streams = {}
        self.down_branches = {}

    def _dynamic_hook(self, name, scale_slot=None):
        base = super()._dynamic_hook(name, scale_slot)
        if not name.endswith(".down_proj"):
            return base
        layer = int(name.split(".", 1)[0][5:])

        def hook(module, inputs, output):
            result = base(module, inputs, output)
            landed = result[0] if isinstance(result, tuple) else result
            if landed is None:
                landed = output[0] if isinstance(output, tuple) else output
            self.down_branches[layer] = landed
            return result
        return hook

    def _attn_residual_pre(self, layer):
        name = f"layer{layer}.attn_residual"

        def hook(_module, inputs):
            nominal = inputs[0]
            standard_landed, _ = self.dynamic(nominal)
            landed = self.policy(self.torch, layer, "attention", self.block_inputs[layer],
                                 self.attn_branches[layer], nominal, standard_landed)
            self.mlp_streams[layer] = landed
            self._save(name, landed[0, -1])
            return (landed, *inputs[1:])
        return hook

    def _layer_hook(self, layer):
        def hook(_module, _inputs, output):
            nominal = output[0] if isinstance(output, tuple) else output
            standard_landed, _ = self.dynamic(nominal)
            landed = self.policy(self.torch, layer, "mlp", self.mlp_streams[layer],
                                 self.down_branches[layer], nominal, standard_landed)
            self.layer_rows[layer] = landed[0, -1].detach().float().cpu().numpy()
            self._save(f"layer{layer}.mlp_residual", landed[0, -1])
            return (landed, *output[1:]) if isinstance(output, tuple) else landed
        return hook


def capture_floor_variants(hf_path, model_data, token_ids, variants, observe_standard=False):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(hf_path), local_files_only=True, dtype=torch.float32,
                                      attn_implementation="eager").cuda().eval()
    install_artifact_weights(torch, model, model_data)
    policy = ResidualPolicy()
    controller = ResidualController(torch, model_data, policy)
    controller.install(model)
    output = {}
    try:
        with patched_rope(controller, model), torch.inference_mode():
            for name, mode, selected in variants:
                policy.mode = mode
                policy.selected = set(selected)
                policy.observe = observe_standard and name == "standard"
                if policy.observe:
                    policy.aggregates = {}
                rows = []
                for item, ids in enumerate(token_ids):
                    controller.reset()
                    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                          output_hidden_states=False, use_cache=False, return_dict=True)
                    rows.append(controller.rows(model_data.config.num_hidden_layers))
                output[name] = np.stack(rows)
                print(f"floor residual variant={name} items={len(token_ids)}", flush=True)
    finally:
        controller.close()
        del model
        torch.cuda.empty_cache()
    return output, policy.aggregates if observe_standard else {}


def run_engine_capture(layer_trace, artifact, token_ids, output, layers):
    root = output / "engine"
    root.mkdir(parents=True, exist_ok=True)
    selected = ",".join(map(str, range(layers)))
    for item, ids in enumerate(token_ids):
        site = root / f"item-{item:02d}.residual.jsonl"
        dump = root / f"item-{item:02d}.rows.bin"
        stdout = root / f"item-{item:02d}.stdout.txt"
        if site.is_file() and dump.is_file() and '"type":"stream"' in site.read_text(
                encoding="utf-8", errors="ignore")[:2000]:
            continue
        command = [str(layer_trace), str(artifact), "-", "--token-ids", ",".join(map(str, ids)),
                   "--dump", str(dump), "--include-final-norm", "--site-dump", str(site),
                   "--site-layers", selected, "--all-position-residual-only"]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        stdout.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"residual trace failed item {item}:\n{completed.stdout}\n{completed.stderr}")
        print(f"engine residual capture item={item + 1}/{len(token_ids)}", flush=True)
    return root


def engine_grid_metrics(root, item_count, layers):
    aggregates = {(layer, site): GridAggregate()
                  for layer in range(layers) for site in ("attention", "mlp")}
    expected = 0
    for item in range(item_count):
        rows = [json.loads(line) for line in
                (root / f"item-{item:02d}.residual.jsonl").read_text(encoding="utf-8").splitlines()]
        grouped = {(row.get("token"), row.get("layer")): {} for row in rows
                   if row["type"] == "stream"}
        for row in rows:
            if row["type"] == "stream":
                grouped[(row["token"], row["layer"])]["stream"] = row
            elif row["type"] == "chain":
                layer = int(row["site"].split(".", 1)[0][5:])
                grouped[(row["token"], layer)][row["site"].split(".", 1)[1]] = row
        for (_token, layer), records in grouped.items():
            required = {"stream", "o_proj.requant", "attn_residual",
                        "down_proj.requant", "mlp_residual"}
            if set(records) != required:
                raise ValueError(f"engine residual record set mismatch item={item} layer={layer}: "
                                 f"{sorted(records)}")
            stream = records["stream"]
            o = records["o_proj.requant"]
            attn = records["attn_residual"]
            stream_codes = np.asarray(stream["codes"], dtype=np.int64)
            attn_wide = np.asarray(attn["x_int"], dtype=np.int64)
            attn_landed = attn_wide - stream_codes
            o_physical = np.asarray(o["codes"], dtype=np.float64) * physical(o["m"], o["e"])
            aggregates[(layer, "attention")].add(
                np.asarray([physical(stream["m"], stream["e"])]), o_physical[None],
                attn_landed[None])

            down = records["down_proj.requant"]
            mlp = records["mlp_residual"]
            attn_codes = np.asarray(attn["codes"], dtype=np.int64)
            mlp_wide = np.asarray(mlp["x_int"], dtype=np.int64)
            mlp_landed = mlp_wide - attn_codes
            down_physical = (np.asarray(down["codes"], dtype=np.float64) *
                             physical(down["m"], down["e"]))
            aggregates[(layer, "mlp")].add(
                np.asarray([physical(attn["m"], attn["e"])]), down_physical[None],
                mlp_landed[None])
            expected += 1
    return aggregates, expected


def final_vectors(rows):
    rows = np.asarray(rows)
    return rows if rows.ndim == 2 else rows[:, -1, :]


def endpoint(actual, expected):
    return {
        "cosine_loss": distribution([1.0 - cosine(actual[i], expected[i])
                                     for i in range(len(actual))]),
        "norm_ratio_absolute_error": distribution(
            [abs(norm_ratio(actual[i], expected[i]) - 1.0) for i in range(len(actual))]),
    }


def reproduction(variants, baseline, block_summary, layers):
    floating = final_vectors(baseline["floating"])
    engine = final_vectors(baseline["engine"])
    floor = final_vectors(baseline["floor"])
    targets = {row["block"]: row for row in block_summary["block_splice"]}
    curve = []
    for layer in range(19, layers):
        vectors = final_vectors(variants[f"through_{layer}"])
        measured = 1.0 - final_curve(vectors, floating)["cosine"]["median"]
        previous = (1.0 - final_curve(floor, floating)["cosine"]["median"] if layer == 19
                    else curve[-1]["reproduced_median_loss"])
        target = targets[layer]["median_cosine_loss"]
        target_increment = targets[layer]["increment_median_cosine_loss"]
        curve.append({
            "block": layer,
            "measured_D_L_median_loss": target,
            "reproduced_median_loss": measured,
            "curve_absolute_error": abs(measured - target),
            "measured_D_L_increment": target_increment,
            "reproduced_increment": measured - previous,
            "increment_absolute_error": abs((measured - previous) - target_increment),
        })
    full = final_curve(final_vectors(variants[f"through_{layers - 1}"]), floating)
    engine_curve = final_curve(engine, floating)
    stat_error = {key: abs(full["cosine"][key] - engine_curve["cosine"][key])
                  for key in ("median", "p10", "p90")}
    norm_error = abs(full["norm_ratio"]["median"] - engine_curve["norm_ratio"]["median"])
    relevant = [row for row in curve if row["block"] >= 20]
    passed = (
        max(row["curve_absolute_error"] for row in relevant) <=
        MARGINS["blocks_20_27_maximum_median_loss_curve_absolute_error"] and
        max(row["increment_absolute_error"] for row in relevant) <=
        MARGINS["blocks_20_27_maximum_increment_absolute_error"] and
        max(stat_error.values()) <= MARGINS["final_cosine_statistic_absolute_error"] and
        norm_error <= MARGINS["final_median_norm_ratio_absolute_error"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "declared_margins": MARGINS,
        "curve": curve,
        "final_structural_order": full,
        "final_engine": engine_curve,
        "final_floor": final_curve(floor, floating),
        "final_cosine_statistic_absolute_error": stat_error,
        "final_median_norm_ratio_absolute_error": norm_error,
        "structural_final_vs_engine_endpoint": endpoint(
            final_vectors(variants[f"through_{layers - 1}"]), engine),
    }


def control_reproduction(variants, baseline):
    floating = final_vectors(baseline["floating"])
    engine = final_vectors(baseline["engine"])
    structural = final_vectors(variants["engine_order"])
    actual = final_curve(structural, floating)
    expected = final_curve(engine, floating)
    cosine_errors = {key: abs(actual["cosine"][key] - expected["cosine"][key])
                     for key in ("median", "p10", "p90")}
    norm_error = abs(actual["norm_ratio"]["median"] - expected["norm_ratio"]["median"])
    passed = (max(cosine_errors.values()) <= MARGINS["qwen2p5_final_cosine_statistic_absolute_error"]
              and norm_error <= MARGINS["qwen2p5_final_median_norm_ratio_absolute_error"])
    return {"status": "PASS" if passed else "FAIL", "structural_order": actual,
            "engine": expected,
            "floor": final_curve(final_vectors(baseline["floor"]), floating),
            "cosine_statistic_absolute_error": cosine_errors,
            "median_norm_ratio_absolute_error": norm_error,
            "structural_vs_engine_endpoint": endpoint(structural, engine)}


def price_table(variants, baseline, hidden_size):
    floating = final_vectors(baseline["floating"])
    engine_cos = final_curve(final_vectors(baseline["engine"]), floating)["cosine"]["median"]
    floor_cos = final_curve(final_vectors(baseline["floor"]), floating)["cosine"]["median"]
    gap = floor_cos - engine_cos
    result = []
    for name in ("common_finer", "wide16", "wide32"):
        curve = final_curve(final_vectors(variants[name]), floating)
        width = 1 if name == "common_finer" else (2 if name == "wide16" else 4)
        result.append({
            "candidate": name,
            "final": curve,
            "share_of_engine_to_floor_median_cosine_gap_closed":
                (curve["cosine"]["median"] - engine_cos) / gap,
            "artifact_bytes_added": 0,
            "runtime_vram_bytes_added_per_sequence": 0 if name == "common_finer" else
                hidden_size * (width - 1),
            "cost": ("extra operand-grid comparison and rescale in the existing int64 wide buffer"
                     if name == "common_finer" else
                     f"persistent {width * 8}-bit residual code stream; weights and artifact unchanged"),
        })
    return result


def trend(table, side):
    values = [table[layer][side]["zero_rounded_fraction"] for layer in range(20, 28)]
    return {"blocks": list(range(20, 28)), "zero_rounded_fraction": values,
            "block27_minus_block20": values[-1] - values[0],
            "nondecreasing": all(values[i + 1] >= values[i] for i in range(len(values) - 1))}


def load_baseline(path):
    source = path if path.is_file() else path / "vectors.npz"
    with np.load(source) as values:
        if "engine_final" in values:
            return {"engine": values["engine_final"], "floating": values["floating_final"],
                    "floor": values["floor_final"]}
        return {key: values[key] for key in ("engine", "floating", "floor")}


def tokens(hf, control):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(hf), local_files_only=True)
    sentences = SMOKE_SENTENCES[:6] if control else SMOKE_SENTENCES + LONG_SENTENCES
    return [tokenizer(sentence, add_special_tokens=True)["input_ids"] for sentence in sentences]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "artifact_cache", "hf_model", "block_summary", "baseline",
                 "control_artifact", "control_artifact_cache", "control_hf_model",
                 "control_block_summary", "control_baseline"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--layer-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    q3_model = artifact_cache.load_artifact(args.artifact_cache)
    q2_model = artifact_cache.load_artifact(args.control_artifact_cache)
    q3_ids = tokens(args.hf_model, False)
    q2_ids = tokens(args.control_hf_model, True)
    q3_root = run_engine_capture(args.layer_trace.resolve(), args.artifact.resolve(), q3_ids,
                                 args.output / "qwen3", q3_model.config.num_hidden_layers)
    q2_root = run_engine_capture(args.layer_trace.resolve(), args.control_artifact.resolve(), q2_ids,
                                 args.output / "qwen2p5", q2_model.config.num_hidden_layers)
    q3_engine_agg, q3_records = engine_grid_metrics(
        q3_root, len(q3_ids), q3_model.config.num_hidden_layers)
    q2_engine_agg, q2_records = engine_grid_metrics(
        q2_root, len(q2_ids), q2_model.config.num_hidden_layers)

    q3_variants = [("standard", "standard", ())] + [
        (f"through_{layer}", "engine_grid", range(layer + 1))
        for layer in range(19, q3_model.config.num_hidden_layers)]
    q3_rows, q3_floor_agg = capture_floor_variants(
        args.hf_model, q3_model, q3_ids, q3_variants, observe_standard=True)
    q2_rows, q2_floor_agg = capture_floor_variants(
        args.control_hf_model, q2_model, q2_ids,
        [("standard", "standard", ()),
         ("engine_order", "engine_grid", range(q2_model.config.num_hidden_layers))],
        observe_standard=True)

    q3_baseline = load_baseline(args.baseline)
    q2_baseline = load_baseline(args.control_baseline)
    q3_block = json.loads(args.block_summary.read_text(encoding="utf-8"))
    q3_repro = reproduction(q3_rows, q3_baseline, q3_block, q3_model.config.num_hidden_layers)
    q2_repro = control_reproduction(q2_rows, q2_baseline)
    q3_floor_endpoint = endpoint(final_vectors(q3_rows["standard"]),
                                 final_vectors(q3_baseline["floor"]))
    q2_floor_endpoint = endpoint(final_vectors(q2_rows["standard"]),
                                 final_vectors(q2_baseline["floor"]))
    if (q3_records != sum(map(len, q3_ids)) * q3_model.config.num_hidden_layers or
            q2_records != sum(map(len, q2_ids)) * q2_model.config.num_hidden_layers or
            q3_floor_endpoint["cosine_loss"]["max"] > 1.0e-6 or
            q3_floor_endpoint["norm_ratio_absolute_error"]["max"] > 1.0e-6 or
            q2_floor_endpoint["cosine_loss"]["max"] > 1.0e-6 or
            q2_floor_endpoint["norm_ratio_absolute_error"]["max"] > 1.0e-6):
        raise RuntimeError("residual-add apparatus endpoint or population validation failed")
    price_rows = {}
    prices = None
    if q3_repro["status"] == "PASS" and q2_repro["status"] == "PASS":
        price_variants = [(name, name, range(q3_model.config.num_hidden_layers))
                          for name in ("common_finer", "wide16", "wide32")]
        price_rows, _ = capture_floor_variants(
            args.hf_model, q3_model, q3_ids, price_variants, observe_standard=False)
        prices = price_table(price_rows, q3_baseline, q3_model.config.hidden_size)

    q3_engine_table = grid_table(q3_engine_agg, q3_model.config.num_hidden_layers)
    q3_floor_table = grid_table(q3_floor_agg, q3_model.config.num_hidden_layers)
    q2_engine_table = grid_table(q2_engine_agg, q2_model.config.num_hidden_layers)
    q2_floor_table = grid_table(q2_floor_agg, q2_model.config.num_hidden_layers)
    overall = "PASS" if q3_repro["status"] == q2_repro["status"] == "PASS" else "FAIL"
    vector_path = args.output / "residual-add-order-vectors.npz"
    saved = {f"qwen3.{key}": value for key, value in q3_rows.items()}
    saved.update({f"qwen2p5.{key}": value for key, value in q2_rows.items()})
    saved.update({f"price.{key}": value for key, value in price_rows.items()})
    np.savez_compressed(vector_path, **saved)
    result = {
        "status": overall,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256(args.artifact),
        "declared_margins": MARGINS,
        "qwen3": {
            "items": len(q3_ids), "positions": sum(map(len, q3_ids)),
            "engine_position_block_records": q3_records,
            "engine_grid_coarseness": q3_engine_table,
            "floor_grid_coarseness": q3_floor_table,
            "engine_zero_fraction_trend_20_27": {
                side: trend(q3_engine_table, side) for side in ("attention", "mlp")},
            "floor_zero_fraction_trend_20_27": {
                side: trend(q3_floor_table, side) for side in ("attention", "mlp")},
            "reproduction": q3_repro,
            "standard_floor_endpoint": q3_floor_endpoint,
        },
        "qwen2p5": {
            "items": len(q2_ids), "positions": sum(map(len, q2_ids)),
            "engine_position_block_records": q2_records,
            "engine_grid_coarseness": q2_engine_table,
            "floor_grid_coarseness": q2_floor_table,
            "reproduction": q2_repro,
            "standard_floor_endpoint": q2_floor_endpoint,
        },
        "repair_prices": prices,
        "smallest_production_fix_if_reproduced": (
            "src/forward/forward_sites.cpp:1109-1145: choose the finer operand grid for residual "
            "reconciliation, add in the existing int64 wide buffer, then requantize once"
            if prices is not None else None),
        "next_hypothesis_if_failed": ({
            "hypothesis": "the residual funnel's fixed-point RequantChainChecked scale carry, "
                          "combined with add order, creates the input-dependent late-block direction",
            "evidence": "add order recreates the aggregate median loss but fails block increments, "
                        "the Qwen3 p90 and the Qwen2.5 norm gate, while paired structural vectors "
                        "remain far from engine vectors; engine/floor zero-rounding profiles are "
                        "close, so zeroing alone is not specific to the observed direction",
            "next_test": "emulate the engine's canonical fixed-point output scale and reciprocal "
                         "carry at both residual funnels, then repeat the same frozen D(L) and "
                         "Qwen2.5 gates",
        } if overall == "FAIL" else None),
        "validation": {
            "qwen3_expected_and_observed_position_block_records": q3_records,
            "qwen2p5_expected_and_observed_position_block_records": q2_records,
            "standard_floor_endpoint_status": "PASS",
        },
        "evidence": {"vectors": str(vector_path.resolve()),
                     "qwen3_engine_raw": str(q3_root.resolve()),
                     "qwen2p5_engine_raw": str(q2_root.resolve())},
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path} status={overall} q3={q3_repro['status']} q2={q2_repro['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
