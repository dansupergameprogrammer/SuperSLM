#!/usr/bin/env python3
r"""T-2703 Qwen fidelity localization: engine, float32, and int8-only floor.

Build the diagnostic tracer first:

    cmd /c tools\build_layer_trace.bat

Example (Qwen3):

    python tools\t2703_fidelity_localization.py ^
      --artifact D:\_t2703conv\flow-final\qwen3-embedding-0.6b-1p5.sslm ^
      --artifact-cache D:\_t2703conv\flow-work\final-cache ^
      --hf-model D:\hf_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 ^
      --layer-trace out\sslm_layer_trace.exe --output D:\_t2703fid\qwen3

The floor changes only the shipping representation: artifact int8 weights are
dequantized with their artifact scales, and activations are landed to int8 at
the engine's boundaries. RMSNorm, RoPE, score reduction, softmax, context
accumulation, SiLU, residual addition, and all other arithmetic remain float32.
The named landing sites are recorded in ``summary.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_pipeline import artifact_cache, pipeline  # noqa: E402


SMOKE_SENTENCES = (
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

LONG_SENTENCES = (
    "After the overnight migration completed, the database team compared query latency, index integrity, replication lag, and application error rates before allowing production traffic back onto the primary cluster.",
    "During the spring survey, marine biologists photographed the reef, measured water temperature at several depths, catalogued bleaching patterns, and compared the new observations with records collected during the previous five years.",
    "The spacecraft crossed the planet's shadow while controllers tracked battery voltage, reaction-wheel momentum, thermal limits, and the delayed telemetry stream arriving through two ground stations.",
    "A senior engineer reviewed the matrix multiplication kernel from memory allocation through vectorized accumulation, checked the boundary tiles separately, and then repeated the benchmark with cold caches and a different thread count.",
)


def cosine_loss(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if not math.isfinite(denom) or denom == 0.0:
        raise ValueError("zero or non-finite vector norm")
    cosine = float(np.dot(left64, right64) / denom)
    return 1.0 - max(-1.0, min(1.0, cosine))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def read_engine_dump(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError(f"truncated engine dump: {path}")
    rows, hidden, _fingerprint, mode = struct.unpack_from("<QQQQ", data)
    row_bytes = 16 + hidden
    if mode != 1 or len(data) != 32 + rows * row_bytes:
        raise ValueError(f"invalid engine dump framing: {path}")
    result = np.empty((rows, hidden), dtype=np.float64)
    offset = 32
    for row in range(rows):
        m, e = struct.unpack_from("<qq", data, offset)
        offset += 16
        codes = np.frombuffer(data, dtype=np.int8, count=hidden, offset=offset).astype(np.float64)
        offset += hidden
        result[row] = codes * math.ldexp(float(m), int(e))
    return result


def run_engine_curves(layer_trace: Path, artifact: Path, token_ids: list[list[int]], output: Path,
                      site_layer=None) -> np.ndarray:
    rows: list[np.ndarray] = []
    dump_dir = output / "engine-dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for index, ids in enumerate(token_ids):
        dump = dump_dir / f"item-{index:02d}.bin"
        command = [str(layer_trace), str(artifact), "-", "--token-ids",
                   ",".join(map(str, ids)), "--dump", str(dump), "--include-final-norm"]
        if site_layer is not None:
            command.extend(["--site-dump", str(dump_dir / f"item-{index:02d}.sites.jsonl"),
                            "--site-layer", str(site_layer)])
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (dump_dir / f"item-{index:02d}.stdout.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"layer trace failed for item {index}: {completed.returncode}\n"
                               f"{completed.stdout}\n{completed.stderr}")
        rows.append(read_engine_dump(dump))
    shape = {row.shape for row in rows}
    if len(shape) != 1:
        raise ValueError(f"engine dump shapes disagree: {shape}")
    return np.stack(rows)


def _tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


class CaptureController:
    def __init__(self, torch, quantized_model, floor: bool, active_layers=None,
                 qkc_headroom: float = 1.0, mixed_precision: str | None = None,
                 emulate_attention_residual: bool = False,
                 injected_gains: dict[str, float] | None = None):
        self.torch = torch
        self.quantized_model = quantized_model
        self.floor = floor
        self.active_layers = active_layers
        self.qkc_headroom = qkc_headroom
        self.mixed_precision = mixed_precision
        self.emulate_attention_residual = emulate_attention_residual
        self.injected_gains = injected_gains or {}
        self.handles = []
        self.sites: dict[str, np.ndarray] = {}
        self.layer_rows: dict[int, np.ndarray] = {}
        self.embed_row = None
        self.final_row = None
        self.q_scales = {}
        self.k_scales = {}
        self.v_rows = {}
        self.rope_q = {}
        self.rope_k = {}
        self.rope_layer = 0
        self.block_inputs = {}
        self.attn_branches = {}

    def _exact_site(self, name: str) -> bool:
        if not self.floor or self.mixed_precision is None or not name.startswith("layer27."):
            return False
        leaf = name.split(".", 1)[1]
        if self.mixed_precision == "v":
            return leaf == "v_proj"
        if self.mixed_precision == "mlp":
            return leaf in {"gate_proj", "up_proj", "mlp_act", "down_proj"}
        if self.mixed_precision == "down_residual":
            return leaf in {"mlp_act", "down_proj", "mlp_residual"}
        raise ValueError(f"unknown mixed-precision candidate: {self.mixed_precision}")

    def _gain(self, name: str, value):
        return value * self.injected_gains.get(name, 1.0)

    def reset(self):
        self.sites = {}
        self.layer_rows = {}
        self.embed_row = None
        self.final_row = None
        self.q_scales = {}
        self.k_scales = {}
        self.v_rows = {}
        self.rope_q = {}
        self.rope_k = {}
        self.rope_layer = 0
        self.block_inputs = {}
        self.attn_branches = {}

    def _save(self, name, tensor):
        self.sites[name] = tensor.detach().float().cpu().numpy()

    def dynamic(self, value):
        peak = value.detach().abs().amax(dim=-1, keepdim=True)
        scale = self.torch.where(peak > 0, peak / 127.0, self.torch.ones_like(peak))
        scaled = value / scale
        codes = self.torch.where(scaled >= 0, self.torch.floor(scaled + 0.5),
                                 self.torch.ceil(scaled - 0.5)).clamp(-127, 127)
        return codes * scale, scale

    def fixed(self, value, scale):
        scaled = value / scale
        codes = self.torch.where(scaled >= 0, self.torch.floor(scaled + 0.5),
                                 self.torch.ceil(scaled - 0.5)).clamp(-127, 127)
        return codes * scale

    def _dynamic_hook(self, name, scale_slot=None):
        def hook(_module, _inputs, output):
            value = _tensor_output(output)
            landed, scale = (self.dynamic(value)
                             if self.floor and not self._exact_site(name) else (value, None))
            self._save(name, landed[0, -1])
            if scale_slot is not None and self.floor:
                scale_slot(landed, scale)
            if not self.floor or self._exact_site(name):
                return None
            if isinstance(output, tuple):
                return (landed, *output[1:])
            return landed
        return hook

    def _static_projection_hook(self, layer, kind):
        cfg = self.quantized_model.config
        heads = cfg.num_key_value_heads
        dim = cfg.head_dim
        physical = [
            math.ldexp(float(self.quantized_model.kv_landing_scales[
                f"layer{layer}.{kind}_head{head}"][0]),
                       int(self.quantized_model.kv_landing_scales[
                           f"layer{layer}.{kind}_head{head}"][1]))
            for head in range(heads)
        ]

        def hook(_module, _inputs, output):
            shaped = output.reshape(output.shape[0], output.shape[1], heads, dim)
            scale = self.torch.tensor(physical, dtype=output.dtype, device=output.device).reshape(1, 1, heads, 1)
            name = f"layer{layer}.{kind}_proj"
            landed = self.fixed(shaped, scale) if self.floor and not self._exact_site(name) else shaped
            self._save(f"layer{layer}.{kind}_proj", landed[0, -1])
            if kind == "k":
                self.k_scales[layer] = scale.transpose(1, 2)
            else:
                self.v_rows[layer] = landed
            return landed.reshape_as(output) if self.floor and not self._exact_site(name) else None
        return hook

    def _pre_dynamic(self, name):
        def hook(_module, inputs):
            value = inputs[0]
            landed, _scale = (self.dynamic(value)
                              if self.floor and not self._exact_site(name) else (value, None))
            self._save(name, landed[0, -1])
            if not self.floor or self._exact_site(name):
                return None
            return (landed, *inputs[1:])
        return hook

    def _layer_input_hook(self, layer):
        def hook(_module, inputs):
            self.block_inputs[layer] = inputs[0]
        return hook

    def _attn_residual_pre(self, layer):
        name = f"layer{layer}.attn_residual"
        def hook(_module, inputs):
            value = inputs[0]
            if self.floor and self.emulate_attention_residual:
                hidden = self.block_inputs[layer]
                branch = self.attn_branches[layer]
                hidden_peak = hidden.detach().abs().amax(dim=-1, keepdim=True)
                hidden_scale = self.torch.where(hidden_peak > 0, hidden_peak / 127.0,
                                                self.torch.ones_like(hidden_peak))
                hidden_code = self.torch.round(hidden / hidden_scale).clamp(-127, 127)
                # Reconcile the already-landed branch onto the residual
                # stream's integer grid, then funnel the wide sum.  This is
                # the float-host diagnostic analogue of ResidualReconcileSite.
                branch_on_hidden = self.torch.round(branch / hidden_scale)
                wide_physical = (hidden_code + branch_on_hidden) * hidden_scale
                landed, _scale = self.dynamic(wide_physical)
            else:
                landed, _scale = self.dynamic(value) if self.floor else (value, None)
            landed = self._gain(name, landed)
            self._save(name, landed[0, -1])
            return (landed, *inputs[1:]) if self.floor else None
        return hook

    def _layer_hook(self, layer):
        def hook(_module, _inputs, output):
            value = _tensor_output(output)
            name = f"layer{layer}.mlp_residual"
            landed, _scale = (self.dynamic(value)
                              if self.floor and not self._exact_site(name) else (value, None))
            landed = self._gain(name, landed)
            self.layer_rows[layer] = landed[0, -1].detach().float().cpu().numpy()
            self._save(f"layer{layer}.mlp_residual", landed[0, -1])
            if not self.floor or self._exact_site(name):
                return None
            if isinstance(output, tuple):
                return (landed, *output[1:])
            return landed
        return hook

    def install(self, model):
        if self.active_layers is None:
            self.handles.append(model.embed_tokens.register_forward_hook(self._embed_hook))
            self.handles.append(model.norm.register_forward_hook(self._final_hook))
        cfg = self.quantized_model.config
        for layer_index, layer in enumerate(model.layers):
            if self.active_layers is not None and layer_index not in self.active_layers:
                continue
            prefix = f"layer{layer_index}"
            self.handles.append(layer.register_forward_pre_hook(self._layer_input_hook(layer_index)))
            attn = layer.self_attn
            mlp = layer.mlp
            self.handles.append(layer.input_layernorm.register_forward_hook(
                self._dynamic_hook(f"{prefix}.attn_norm")))

            def q_scale_slot(_landed, scale, layer_index=layer_index):
                self.q_scales[layer_index] = scale.reshape(
                    scale.shape[0], scale.shape[1], 1, 1).transpose(1, 2)

            self.handles.append(attn.q_proj.register_forward_hook(
                self._dynamic_hook(f"{prefix}.q_proj", q_scale_slot)))
            self.handles.append(attn.k_proj.register_forward_hook(
                self._static_projection_hook(layer_index, "k")))
            self.handles.append(attn.v_proj.register_forward_hook(
                self._static_projection_hook(layer_index, "v")))
            if hasattr(attn, "q_norm"):
                def qnorm_scale_slot(_landed, scale, layer_index=layer_index):
                    # Qwen3 normalizes in [batch, sequence, heads, head_dim]
                    # and transposes to [batch, heads, sequence, head_dim]
                    # immediately before RoPE.
                    self.q_scales[layer_index] = scale.transpose(1, 2)
                self.handles.append(attn.q_norm.register_forward_hook(
                    self._dynamic_hook(f"{prefix}.q_norm", qnorm_scale_slot)))
            if hasattr(attn, "k_norm"):
                self.handles.append(attn.k_norm.register_forward_hook(
                    self._record_only_hook(f"{prefix}.k_norm")))
            self.handles.append(attn.o_proj.register_forward_pre_hook(
                self._pre_dynamic(f"{prefix}.attn_ctx")))
            def o_slot(landed, _scale, layer_index=layer_index):
                self.attn_branches[layer_index] = landed
            self.handles.append(attn.o_proj.register_forward_hook(
                self._dynamic_hook(f"{prefix}.o_proj", o_slot)))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                self._attn_residual_pre(layer_index)))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(
                self._dynamic_hook(f"{prefix}.mlp_norm")))
            self.handles.append(mlp.gate_proj.register_forward_hook(
                self._dynamic_hook(f"{prefix}.gate_proj")))
            self.handles.append(mlp.up_proj.register_forward_hook(
                self._dynamic_hook(f"{prefix}.up_proj")))
            self.handles.append(mlp.down_proj.register_forward_pre_hook(
                self._pre_dynamic(f"{prefix}.mlp_act")))
            self.handles.append(mlp.down_proj.register_forward_hook(
                self._dynamic_hook(f"{prefix}.down_proj")))
            self.handles.append(layer.register_forward_hook(self._layer_hook(layer_index)))

    def _record_only_hook(self, name):
        def hook(_module, _inputs, output):
            self._save(name, _tensor_output(output)[0, -1])
        return hook

    def _embed_hook(self, _module, _inputs, output):
        landed, _scale = self.dynamic(output) if self.floor else (output, None)
        self.embed_row = landed[0, -1].detach().float().cpu().numpy()
        self._save("embed", landed[0, -1])
        return landed if self.floor else None

    def _final_hook(self, _module, _inputs, output):
        landed, _scale = self.dynamic(output) if self.floor else (output, None)
        self.final_row = landed[0, -1].detach().float().cpu().numpy()
        self._save("final_norm", landed[0, -1])
        return landed if self.floor else None

    def rope(self, original):
        def wrapped(q, k, cos, sin, *args, **kwargs):
            q_rot, k_rot = original(q, k, cos, sin, *args, **kwargs)
            layer = self.rope_layer
            self.rope_layer += 1
            active = self.active_layers is None or layer in self.active_layers
            if self.floor and active:
                q_scale = self.q_scales[layer]
                q_rot = self.fixed(q_rot, q_scale)
                prefix = f"layer{layer}"
                self._save(f"{prefix}.k_rope_pre_qkc_all", k_rot[0])
                if prefix in self.quantized_model.qk_channel_peaks:
                    source = (np.asarray(self.quantized_model.qk_channel_peaks[prefix], dtype=np.float32)
                              * self.qkc_headroom / 127.0)
                    # QKC1 is stored in the engine's adjacent-pair RoPE order;
                    # this floor runs inside the upstream half-split HF layout.
                    order = pipeline._rope_pair_permutation(self.quantized_model.config.head_dim)
                    inverse = np.empty_like(order)
                    inverse[order] = np.arange(order.size)
                    source = source[:, inverse]
                    k_scale = self.torch.from_numpy(source).to(device=k_rot.device, dtype=k_rot.dtype).reshape(
                        1, source.shape[0], 1, source.shape[1])
                else:
                    k_scale = self.k_scales[layer]
                k_rot = self.fixed(k_rot, k_scale)
            if active:
                self.rope_q[layer] = q_rot.detach()
                self.rope_k[layer] = k_rot.detach()
                self._save(f"layer{layer}.q_rope", q_rot[0, :, -1])
                self._save(f"layer{layer}.k_rope", k_rot[0, :, -1])
            return q_rot, k_rot
        return wrapped

    def finish_attention_sites(self):
        cfg = self.quantized_model.config
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        for layer in range(cfg.num_hidden_layers):
            if self.active_layers is not None and layer not in self.active_layers:
                continue
            q = self.rope_q[layer][0]
            k = self.rope_k[layer][0]
            values = self.v_rows[layer][0].transpose(0, 1)
            score_rows = []
            probability_rows = []
            context_rows = []
            for head in range(cfg.num_attention_heads):
                kv_head = head // group
                scores = (q[head, -1].float() @ k[kv_head].float().T) / math.sqrt(cfg.head_dim)
                probs = self.torch.softmax(scores, dim=-1)
                context = probs @ values[kv_head].float()
                score_rows.append(scores)
                probability_rows.append(probs)
                context_rows.append(context)
            self._save(f"layer{layer}.scores", self.torch.stack(score_rows))
            self._save(f"layer{layer}.softmax", self.torch.stack(probability_rows))
            self._save(f"layer{layer}.context", self.torch.stack(context_rows))

    def rows(self, layers):
        return np.stack([self.embed_row] + [self.layer_rows[i] for i in range(layers)] + [self.final_row])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _inverse_permute(values: np.ndarray, cfg, heads: int) -> np.ndarray:
    order = pipeline._rope_pair_permutation(cfg.head_dim)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    shaped = values.reshape(heads, cfg.head_dim, *values.shape[1:])
    return shaped[:, inverse].reshape(values.shape)


def _dequantized_weight(model, name: str) -> np.ndarray:
    codes = np.asarray(model.weights[name], dtype=np.float32)
    scales = np.asarray(model.weight_scales[name], dtype=np.float32)
    if scales.size == 1:
        values = codes * scales.item()
    else:
        values = codes * scales.reshape((scales.size,) + (1,) * (codes.ndim - 1))
    leaf = name.split(".", 1)[-1]
    if leaf in ("q_proj", "q_proj.bias"):
        values = _inverse_permute(values, model.config, model.config.num_attention_heads)
    elif leaf in ("k_proj", "k_proj.bias"):
        values = _inverse_permute(values, model.config, model.config.num_key_value_heads)
    elif leaf in ("q_norm.gain", "k_norm.gain"):
        values = _inverse_permute(values, model.config, 1)
    return values


def install_artifact_weights(torch, hf_model, quantized_model, active_layers=None,
                             mixed_precision: str | None = None):
    assignments = [] if active_layers is not None else [
        (hf_model.embed_tokens.weight, "embed"),
        (hf_model.norm.weight, "final_norm.gain")]
    for layer_index, layer in enumerate(hf_model.layers):
        if active_layers is not None and layer_index not in active_layers:
            continue
        prefix = f"layer{layer_index}"
        assignments.extend([
            (layer.input_layernorm.weight, f"{prefix}.attn_norm.gain"),
            (layer.post_attention_layernorm.weight, f"{prefix}.mlp_norm.gain"),
            (layer.self_attn.q_proj.weight, f"{prefix}.q_proj"),
            (layer.self_attn.k_proj.weight, f"{prefix}.k_proj"),
            (layer.self_attn.v_proj.weight, f"{prefix}.v_proj"),
            (layer.self_attn.o_proj.weight, f"{prefix}.o_proj"),
            (layer.mlp.gate_proj.weight, f"{prefix}.gate_proj"),
            (layer.mlp.up_proj.weight, f"{prefix}.up_proj"),
            (layer.mlp.down_proj.weight, f"{prefix}.down_proj"),
        ])
        if hasattr(layer.self_attn, "q_norm"):
            assignments.append((layer.self_attn.q_norm.weight, f"{prefix}.q_norm.gain"))
        if hasattr(layer.self_attn, "k_norm"):
            assignments.append((layer.self_attn.k_norm.weight, f"{prefix}.k_norm.gain"))
    with torch.no_grad():
        for parameter, name in assignments:
            leaf = name.split(".", 1)[-1]
            preserve = name.startswith("layer27.") and (
                (mixed_precision == "v" and leaf == "v_proj") or
                (mixed_precision == "mlp" and leaf in {"gate_proj", "up_proj", "down_proj"}) or
                (mixed_precision == "down_residual" and leaf == "down_proj"))
            if preserve:
                # Diagnostic price point is FP16 storage/arithmetic.  The HF
                # graph stays float32, so round the source checkpoint weight to
                # FP16 and widen it back before execution.
                parameter.copy_(parameter.half().float())
                continue
            values = _dequantized_weight(quantized_model, name)
            if tuple(values.shape) != tuple(parameter.shape):
                raise ValueError(f"weight shape mismatch {name}: {values.shape} != {tuple(parameter.shape)}")
            parameter.copy_(torch.from_numpy(values).to(device=parameter.device, dtype=parameter.dtype))


@contextlib.contextmanager
def patched_rope(controller, hf_model):
    module_name = hf_model.__class__.__module__
    module = sys.modules[module_name]
    original = module.apply_rotary_pos_emb
    module.apply_rotary_pos_emb = controller.rope(original)
    try:
        yield
    finally:
        module.apply_rotary_pos_emb = original


def capture_hf(hf_path: Path, quantized_model, token_ids: list[list[int]], floor: bool,
               active_layers=None, qkc_headroom: float = 1.0,
               mixed_precision: str | None = None,
               emulate_attention_residual: bool = False,
               injected_gains: dict[str, float] | None = None):
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("T-2703 full-model localization requires CUDA")
    model = AutoModel.from_pretrained(str(hf_path), local_files_only=True, dtype=torch.float32,
                                      attn_implementation="eager").cuda().eval()
    if floor:
        install_artifact_weights(torch, model, quantized_model, active_layers, mixed_precision)
    controller = CaptureController(torch, quantized_model, floor, active_layers,
                                   qkc_headroom, mixed_precision, emulate_attention_residual,
                                   injected_gains)
    controller.install(model)
    all_rows = []
    all_sites = []
    try:
        with patched_rope(controller, model), torch.inference_mode():
            for ids in token_ids:
                controller.reset()
                input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                               output_hidden_states=True, use_cache=False, return_dict=True)
                if len(output.hidden_states) != quantized_model.config.num_hidden_layers + 1:
                    raise ValueError("unexpected Hugging Face hidden-state count")
                controller.finish_attention_sites()
                if active_layers is None:
                    all_rows.append(controller.rows(quantized_model.config.num_hidden_layers))
                all_sites.append(dict(controller.sites))
    finally:
        controller.close()
        del model
        torch.cuda.empty_cache()
    return (np.stack(all_rows) if active_layers is None else None), all_sites


def save_site_dumps(output: Path, kind: str, site_sets):
    directory = output / f"{kind}-site-dumps"
    directory.mkdir(parents=True, exist_ok=True)
    for index, sites in enumerate(site_sets):
        np.savez_compressed(directory / f"item-{index:02d}.npz", **sites)


def curve_table(engine: np.ndarray, floating: np.ndarray, floor: np.ndarray):
    if engine.shape != floating.shape or engine.shape != floor.shape:
        raise ValueError(f"curve shapes disagree: engine={engine.shape} float={floating.shape} floor={floor.shape}")
    table = []
    for row in range(engine.shape[1]):
        e_f = [cosine_loss(engine[i, row], floating[i, row]) for i in range(engine.shape[0])]
        r_f = [cosine_loss(floor[i, row], floating[i, row]) for i in range(engine.shape[0])]
        e_r = [cosine_loss(engine[i, row], floor[i, row]) for i in range(engine.shape[0])]
        label = "embedding" if row == 0 else (
            "final_norm" if row == engine.shape[1] - 1 else f"block_{row - 1}_residual")
        table.append({"row": row, "pairing": label,
                      "engine_vs_float": summarize(e_f),
                      "floor_vs_float": summarize(r_f),
                      "engine_vs_floor": summarize(e_r)})
    return table


def site_table(float_sites, floor_sites, layer: int):
    names = (
        "attn_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
        "q_rope", "k_rope", "scores", "softmax", "context", "attn_ctx",
        "o_proj", "attn_residual", "mlp_norm", "gate_proj", "up_proj",
        "mlp_act", "down_proj", "mlp_residual",
    )
    rows = []
    for leaf in names:
        key = f"layer{layer}.{leaf}"
        if key not in float_sites[0] or key not in floor_sites[0]:
            continue
        losses = [cosine_loss(floor_sites[i][key].reshape(-1), float_sites[i][key].reshape(-1))
                  for i in range(len(float_sites))]
        rows.append({"site": key, "floor_vs_float": summarize(losses)})
    return rows


def engine_site_table(output: Path, float_sites, floor_sites, quantized_model, layer: int):
    grouped_items = []
    for index in range(len(float_sites)):
        path = output / "engine-dumps" / f"item-{index:02d}.sites.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        grouped_items.append(rows)
    results = []
    chain_names = sorted({row["site"] for row in grouped_items[0] if row["type"] == "chain"})
    order = pipeline._rope_pair_permutation(quantized_model.config.head_dim)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    for engine_name in chain_names:
        reference_name = engine_name.replace(".requant", "")
        if reference_name not in float_sites[0] or reference_name not in floor_sites[0]:
            continue
        losses = ([], [], [])
        for index, rows in enumerate(grouped_items):
            matching = [row for row in rows if row["type"] == "chain" and row["site"] == engine_name]
            vector = np.concatenate([
                np.asarray(row["codes"], dtype=np.float64) *
                math.ldexp(float(row["m"]), int(row["e"])) for row in matching])
            if reference_name.endswith((".q_proj", ".q_norm")):
                vector = vector.reshape(quantized_model.config.num_attention_heads,
                                        quantized_model.config.head_dim)[:, inverse].reshape(-1)
            floating = float_sites[index][reference_name].reshape(-1)
            floor = floor_sites[index][reference_name].reshape(-1)
            losses[0].append(cosine_loss(vector, floating))
            losses[1].append(cosine_loss(floor, floating))
            losses[2].append(cosine_loss(vector, floor))
        results.append({"site": reference_name, "engine_vs_float": summarize(losses[0]),
                        "floor_vs_float": summarize(losses[1]),
                        "engine_vs_floor": summarize(losses[2])})

    # Fused-K callback is the only engine observation after wide K RMSNorm/RoPE
    # and QKC1 landing. Convert its adjacent-pair codes back to HF order.
    losses = ([], [], [])
    for index, rows in enumerate(grouped_items):
        fused = [row for row in rows if row["type"] == "fused_k"]
        codes = np.asarray([max(-127, min(127, row["landing_raw"])) for row in fused], dtype=np.float64)
        scales = np.asarray(quantized_model.qk_channel_peaks[f"layer{layer}"],
                            dtype=np.float64).reshape(-1) / 127.0
        vector = (codes * scales).reshape(quantized_model.config.num_key_value_heads,
                                         quantized_model.config.head_dim)[:, inverse].reshape(-1)
        floating = float_sites[index][f"layer{layer}.k_rope"].reshape(-1)
        floor = floor_sites[index][f"layer{layer}.k_rope"].reshape(-1)
        losses[0].append(cosine_loss(vector, floating))
        losses[1].append(cosine_loss(floor, floating))
        losses[2].append(cosine_loss(vector, floor))
    results.append({"site": f"layer{layer}.k_rope/QKC1", "engine_vs_float": summarize(losses[0]),
                    "floor_vs_float": summarize(losses[1]),
                    "engine_vs_floor": summarize(losses[2])})

    for reference_leaf, field in (("scores", "scores"), ("softmax", "probs"),
                                  ("context", "ctx_wide")):
        losses = ([], [], [])
        key = f"layer{layer}.{reference_leaf}"
        for index, rows in enumerate(grouped_items):
            attention = [row for row in rows if row["type"] == "attention"]
            per_head = ([], [], [])
            for row in attention:
                head = row["head"]
                vector = np.asarray(row[field], dtype=np.float64)
                floating = float_sites[index][key][head].reshape(-1)
                floor = floor_sites[index][key][head].reshape(-1)
                per_head[0].append(cosine_loss(vector, floating))
                per_head[1].append(cosine_loss(floor, floating))
                per_head[2].append(cosine_loss(vector, floor))
            for target, values in zip(losses, per_head):
                target.append(float(np.median(values)))
        results.append({"site": key, "engine_vs_float": summarize(losses[0]),
                        "floor_vs_float": summarize(losses[1]),
                        "engine_vs_floor": summarize(losses[2])})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-cache", required=True, type=Path)
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--layer-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", action="store_true", help="run six prompts rather than 24 + four long prompts")
    parser.add_argument("--site-layer", type=int,
                        help="capture engine interiors and a held-upstream-fixed floor for this layer")
    args = parser.parse_args()
    for path in (args.artifact, args.layer_trace):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    for path in (args.artifact_cache, args.hf_model):
        if not path.is_dir():
            parser.error(f"missing directory: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    sentences = SMOKE_SENTENCES[:6] if args.control else SMOKE_SENTENCES + LONG_SENTENCES
    token_ids = [tokenizer(sentence, add_special_tokens=True)["input_ids"] for sentence in sentences]
    quantized_model = artifact_cache.load_artifact(args.artifact_cache)
    if args.site_layer is not None and not 0 <= args.site_layer < quantized_model.config.num_hidden_layers:
        parser.error("--site-layer is outside the model")
    if max(map(len, token_ids)) > quantized_model.config.context_cap:
        raise ValueError("input exceeds artifact context capacity")

    floating, float_sites = capture_hf(args.hf_model, quantized_model, token_ids, floor=False)
    floor, floor_sites = capture_hf(args.hf_model, quantized_model, token_ids, floor=True)
    save_site_dumps(args.output, "float", float_sites)
    save_site_dumps(args.output, "floor", floor_sites)
    localized_sites = None
    if args.site_layer is not None:
        _, localized_sites = capture_hf(args.hf_model, quantized_model, token_ids, floor=True,
                                        active_layers={args.site_layer})
        save_site_dumps(args.output, f"layer{args.site_layer}-isolated-floor", localized_sites)
    engine = run_engine_curves(args.layer_trace.resolve(), args.artifact.resolve(), token_ids, args.output,
                               args.site_layer)

    table = curve_table(engine, floating, floor)
    excess = [row["engine_vs_float"]["median"] - row["floor_vs_float"]["median"]
              for row in table]
    departure = next((row for row in range(1, len(table) - 2)
                      if excess[row] > 0.01 and excess[row + 1] > 0.01), None)
    departure_layer = None if departure in (None, 0, len(table) - 1) else departure - 1
    sites = site_table(float_sites, floor_sites, args.site_layer) if args.site_layer is not None else []
    isolated = (site_table(float_sites, localized_sites, args.site_layer)
                if args.site_layer is not None else [])
    engine_sites = (engine_site_table(args.output, float_sites, floor_sites, quantized_model,
                                      args.site_layer) if args.site_layer is not None else [])
    np.savez_compressed(args.output / "vectors.npz", engine=engine, floating=floating, floor=floor,
                        token_lengths=np.asarray(list(map(len, token_ids)), dtype=np.int32))
    result = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "artifact_cache": str(args.artifact_cache.resolve()),
        "hf_model": str(args.hf_model.resolve()),
        "sentence_count": len(sentences),
        "token_length": summarize([float(len(ids)) for ids in token_ids]),
        "pairing": {
            "row_0": "embedding output before decoder block 0",
            "rows_1_through_L": "raw residual stream after decoder block row-1",
            "last_row": "post-final-RMSNorm output; Hugging Face last_hidden_state",
        },
        "floor_quantized_sites": [
            "embed", "layer*.attn_norm", "layer*.q_proj", "layer*.k_proj (static per KV head)",
            "layer*.v_proj (static per KV head)", "layer*.q_norm (when present)",
            "layer*.q_rope (same carried q scale)",
            "layer*.k_rope (QKC1 per-channel scale when present; otherwise static K scale)",
            "layer*.attn_ctx", "layer*.o_proj", "layer*.attn_residual", "layer*.mlp_norm",
            "layer*.gate_proj", "layer*.up_proj", "layer*.mlp_act", "layer*.down_proj",
            "layer*.mlp_residual", "final_norm",
        ],
        "floor_float_sites": ["RMSNorm arithmetic", "RoPE arithmetic", "scores", "softmax",
                              "context accumulation", "SiLU", "residual addition"],
        "curves": table,
        "departure_row": departure,
        "departure_layer": departure_layer,
        "departure_rule": "first two consecutive non-final rows where median(engine-float) minus "
                          "median(int8-floor-float) exceeds 0.01",
        "captured_site_layer": args.site_layer,
        "captured_layer_floor_sites": sites,
        "captured_layer_isolated_floor_sites": isolated,
        "captured_layer_engine_sites": engine_sites,
        "evidence": {
            "vectors": str((args.output / "vectors.npz").resolve()),
            "engine_dumps": str((args.output / "engine-dumps").resolve()),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"summary={args.output / 'summary.json'} departure_row={departure} "
          f"departure_layer={departure_layer} items={len(sentences)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
