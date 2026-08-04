#!/usr/bin/env python3
"""T-1697 -- offline weight-side outlier migration (SmoothQuant mechanism,
re-derived from this model's own weights and this project's own captured
activations, never an external calibration corpus).

Candidate 2 from `Claude/Feynman/t1691-activation-quantization-construction-
alternatives-2026-08-03.md`: for a chosen outlier intermediate channel `c`
(down_proj's own input channel), derive a per-layer scale `s_c > 1` and:

  * fold `1/s_c` into `up_proj`'s own per-channel weight-scale fold (`mult`,
    `shift`) at output channel `c` -- free, an existing scale constant
    re-derived, no int8 weight bytes touched, no shape change.
  * re-quantize `down_proj`'s int8 weight COLUMN `c` (every output channel's
    row, same column) by `round(old_int8[k, c] * s_c)`, clamped to
    `[-127, 127]` -- `down_proj` is quantized per OUTPUT channel (row), so a
    per-input-column factor cannot be absorbed into its existing fold and
    must be paid for by re-deriving the int8 codes against each row's own
    UNCHANGED `real_scale[k]`.

`SiLU` and `gate_proj` are never touched: only `up_proj`'s output channel `c`
and `down_proj`'s input column `c` change, which is exactly the pair the
elementwise product `h[c] = SiLU(gate[c]) * up[c]` carries the scale through
exactly (SiLU(gate[c]) is invariant, so up[c]/s_c times down_col[c]*s_c
composes back to the original product for every other operand held fixed).

`s_c` is SmoothQuant's own balance form: `s_c = max|X_c|^alpha /
max|W_c|^(1-alpha)`, floored at 1.0 (a computed value below 1.0 means this
layer's own channel is not activation-dominant, and shrinking a weight
column instead of growing it is a different, unrequested transform -- so
that layer is left unmigrated, `s_c = 1.0`, at that alpha).

`max|X_c|` is read from this project's OWN captured `mlp_act` site records
(the campaign's own established 9-prompt population,
`kv_saturation_report.POPULATION`) -- `down_proj`'s own real int8 input
codes, dequantized via that record's own row-level `d_prime` (the SAME
"codes * d_prime / 127" real-magnitude proxy `tools/t1695_downproj_
recoverable_ceiling.py`'s own Arm A already uses for this exact site).
`max|W_c|` is read from the artifact's own `down_proj` weight column,
dequantized via `shadow_layer_recompute.dequantize_weight_matrix`'s own
`real_scale` formula -- never estimated.

No calibration corpus outside this project's own artifact and this
project's own captured rows is read anywhere in this module.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
from sslm_artifact_reader import read_artifact, _read_container_sections  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
POP_DIR = os.path.join(REPO_ROOT, "out", "t1697_population")
NUM_LAYERS = 28


def quantize_multiplier(factor: float) -> tuple[int, int]:
    """The SAME gemmlowp (multiplier, shift) construction the real offline
    converter uses (`D:\\Wizard\\Tools\\superslm_spike\\pipeline.py`
    `quantize_multiplier`, read at source and transcribed here rather than
    cross-tree-imported -- this module stays runnable from a bare checkout,
    matching `tools/convert_model.py`'s own stated lazy-import convention for
    the same cross-tree dependency). `factor` must be in (0, 1): the real
    per-channel weight scales this project emits are always sub-unity
    fractional multipliers (§6.2's own requant form); a factor outside that
    range is a calibration defect, not a rounding question, and this
    function raises loudly rather than silently degrading, matching the
    source function's own contract."""
    if not 0.0 < factor < 1.0:
        raise ValueError(
            f"quantize_multiplier: factor {factor} outside (0, 1) -- the requant form "
            f"represents multipliers in (0, 1) only")
    fraction, exponent = math.frexp(factor)
    multiplier = int(round(fraction * (1 << 31)))
    if multiplier >= 1 << 31:
        multiplier >>= 1
        exponent += 1
    shift = -exponent
    if shift < 0:
        return (1 << 31) - 1, 0
    if shift > 31:
        raise ValueError(
            f"quantize_multiplier: factor {factor} needs shift {shift}, outside "
            f"RoundingDivideByPOT's defined range [0, 31]")
    return multiplier, shift


# =============================================================================
# Activation / weight magnitude statistics
# =============================================================================


def load_population_sitedumps(pop_dir: str = POP_DIR) -> dict[str, list]:
    """{label: [SiteRecord, ...]} for every prompt in the campaign's own
    9-prompt population, read from the site-dump files this task's own
    `t1697_capture_population_sitedumps.py` wrote against the BASELINE
    (unmigrated) artifact -- activation statistics are always read from the
    baseline capture, never re-captured against a migrated artifact, so the
    derivation is well-defined independent of which alpha is later tested."""
    out = {}
    for label, _question, _role in KSR.POPULATION:
        path = os.path.join(pop_dir, f"{label}.sitedump")
        out[label] = S.load_site_dump(path)
    return out


def max_abs_activation_per_layer(channel: int, records_by_label: dict) -> list[float]:
    """max|X_c| at each of the 28 layers, over the whole 9-prompt population,
    read from each layer's real `mlp_act` SiteRecord: `|codes[c]| * d_prime /
    127` is the real-magnitude proxy for channel `c`'s value BEFORE
    `down_proj.requant`'s own per-row quantization to int8 -- i.e. exactly
    `down_proj`'s own real input magnitude at that channel, in the fixed
    physical units `d_prime` (the row's own max-abs) already carries."""
    out = []
    for layer_idx in range(NUM_LAYERS):
        site = f"layer{layer_idx}.mlp_act"
        m = 0.0
        for label, recs in records_by_label.items():
            rec = next((r for r in recs if r.site == site), None)
            if rec is None:
                raise ValueError(f"missing site record {site!r} for prompt {label!r}")
            v = abs(int(rec.codes[channel])) * rec.d_prime / 127.0
            m = max(m, v)
        out.append(m)
    return out


def max_abs_weight_column_per_layer(artifact, channel: int) -> list[float]:
    """max|W_c| at each of the 28 layers -- the real dequantized magnitude of
    `down_proj`'s own input column `c`, over all 1536 output channels, via
    `real_scale`/`dequantize_weight_matrix`'s own formula (never estimated)."""
    out = []
    for layer_idx in range(NUM_LAYERS):
        layer = artifact.layers[layer_idx]
        real_scale_down = S.real_scale(layer.down_fold[:, 1], layer.down_fold[:, 2], layer.down_fold[:, 0])
        w_col = layer.down_weight[:, channel].astype(np.float64) * real_scale_down
        out.append(float(np.max(np.abs(w_col))))
    return out


def derive_layer_scales(max_x: list[float], max_w: list[float], alpha: float) -> list[float]:
    """SmoothQuant's own balance form per layer, floored at 1.0: `alpha=0`
    is EXACTLY the no-migration identity (`s_c = 1.0` at every layer, since
    `x^0 = 1`) -- this is this task's own mandatory no-op self-check, and it
    holds by construction of this formula, not by a special case in this
    function."""
    out = []
    for x, w in zip(max_x, max_w):
        s = (x ** alpha) / (w ** (1.0 - alpha))
        out.append(max(1.0, s))
    return out


# =============================================================================
# Phase 1 -- the float64 identity check (BEFORE any int8 is touched)
# =============================================================================


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@dataclass
class IdentityCheckResult:
    label: str
    layer_idx: int
    channel: int
    s_c: float
    baseline_y_norm: float
    max_abs_diff: float
    max_rel_diff: float


def float_identity_check(
    artifact, channel: int, layer_idx: int, s_c: float, label: str, records_by_label: dict,
) -> IdentityCheckResult:
    """Proves the transform is an identity in float64, using the REAL
    captured `mlp_norm` input codes for one (layer, prompt) cell as `x`:
    baseline float64 gate->up->SiLU->down composition versus the SAME
    composition with `up_proj`'s output channel `c` divided by `s_c` and
    `down_proj`'s input column `c` multiplied by `s_c`. Independent of any
    int8 mechanics -- pure float64 linear algebra, run BEFORE the int8
    artifact is touched at all, per this task's own mandated order of
    operations."""
    layer = artifact.layers[layer_idx]
    mlp_norm_rec = next(r for r in records_by_label[label] if r.site == f"layer{layer_idx}.mlp_norm")
    x = mlp_norm_rec.codes.astype(np.float64)

    gate_w = S.dequantize_weight_matrix(layer.gate_weight, layer.gate_fold[:, 1], layer.gate_fold[:, 2],
                                         layer.gate_fold[:, 0])
    up_w = S.dequantize_weight_matrix(layer.up_weight, layer.up_fold[:, 1], layer.up_fold[:, 2],
                                       layer.up_fold[:, 0])
    down_w = S.dequantize_weight_matrix(layer.down_weight, layer.down_fold[:, 1], layer.down_fold[:, 2],
                                         layer.down_fold[:, 0])

    gate = gate_w @ x
    up = up_w @ x
    h = _silu(gate) * up
    y = down_w @ h

    up_w2 = up_w.copy()
    up_w2[channel, :] = up_w2[channel, :] / s_c
    down_w2 = down_w.copy()
    down_w2[:, channel] = down_w2[:, channel] * s_c

    gate2 = gate_w @ x  # gate_proj is never touched
    up2 = up_w2 @ x
    h2 = _silu(gate2) * up2
    y2 = down_w2 @ h2

    diff = np.abs(y2 - y)
    baseline_norm = float(np.max(np.abs(y)))
    max_abs_diff = float(np.max(diff))
    max_rel_diff = float(np.max(diff / (np.abs(y) + 1e-300)))

    return IdentityCheckResult(label=label, layer_idx=layer_idx, channel=channel, s_c=s_c,
                                baseline_y_norm=baseline_norm, max_abs_diff=max_abs_diff,
                                max_rel_diff=max_rel_diff)


# =============================================================================
# Phase 2 -- apply to the int8 artifact: byte-patch a copy, in place
# =============================================================================

_HEADER_BYTES = 64
_SECTION_ENTRY_BYTES = 40
_SECTION_WEIGHTS = 2
_SECTION_WEIGHT_SCALES = 6


@dataclass
class LayerPatchStats:
    layer_idx: int
    s_c: float
    up_mult_before: int
    up_shift_before: int
    up_mult_after: int
    up_shift_after: int
    n_saturated: int  # count of down_proj int8 column elements clamped to +-127 by THIS migration
    n_elements: int
    max_abs_weight_quant_error: float  # max_k |new_int8[k]*real_scale[k] - old_int8[k]*s_c*real_scale[k]|
    rmse_weight_quant_error: float


@dataclass
class BuildResult:
    channel: int
    alpha: float
    out_path: str
    layer_stats: list[LayerPatchStats] = field(default_factory=list)
    bytes_changed: int = 0
    identical_to_baseline: bool = False


def _section_table(data: bytes) -> tuple[list[dict], int]:
    """Re-parses the section table into a mutable list of dicts (offset kept
    editable in place is not needed here -- offsets/sizes never change, only
    bytes WITHIN a section's existing byte range are overwritten) plus the
    table's own byte offset (always `_HEADER_BYTES`)."""
    (section_count,) = struct.unpack_from("<I", data, 12)
    entries = []
    for i in range(section_count):
        off = _HEADER_BYTES + i * _SECTION_ENTRY_BYTES
        (sec_type, dtype, offset, byte_size, elem_count, alignment, reserved) = struct.unpack_from(
            "<IIQQQII", data, off)
        entries.append(dict(type=sec_type, dtype=dtype, offset=offset, byte_size=byte_size,
                             elem_count=elem_count, alignment=alignment, table_off=off))
    return entries, _HEADER_BYTES


def _find_tensor_offset(section_bytes: bytes, name: str) -> tuple[int, int, tuple[int, ...]]:
    """Byte offset (ABSOLUTE within the section blob), element count, and
    shape of tensor `name` inside a WGT1/WSC1 manifest -- re-parses the same
    48-byte descriptor layout `sslm_artifact_reader._parse_tensor_manifest`
    reads, but returns the RAW offset/shape instead of a decoded ndarray, so
    the caller can overwrite bytes in the ORIGINAL buffer directly rather
    than re-serializing the whole manifest."""
    magic = section_bytes[0:4]
    (version, tensor_count, name_blob_len) = struct.unpack_from("<III", section_bytes, 4)
    desc_off = 16
    name_blob_off = desc_off + tensor_count * 48
    name_blob = section_bytes[name_blob_off:name_blob_off + name_blob_len]
    for i in range(tensor_count):
        d_off = desc_off + i * 48
        (name_off, name_len, rank) = struct.unpack_from("<III", section_bytes, d_off)
        shape = struct.unpack_from("<IIII", section_bytes, d_off + 12)
        (data_off, elem_count, reserved) = struct.unpack_from("<QQI", section_bytes, d_off + 28)
        n = name_blob[name_off:name_off + name_len].decode("utf-8")
        if n == name:
            return data_off, elem_count, tuple(shape[:rank])
    raise KeyError(f"tensor {name!r} not found in manifest (magic {magic!r})")


def build_migrated_artifact(
    baseline_path: str, out_path: str, channel: int, layer_scales: list[float],
) -> BuildResult:
    """Copies `baseline_path` byte-for-byte, then overwrites ONLY:
      - `up_proj`'s WSC1 fold row `channel` at each migrated layer (mult,
        shift re-derived for `real_scale[channel] / s_c`, identity untouched
        -- it is already 0 at every layer, confirmed by this task's own
        diagnostic pass).
      - `down_proj`'s WGT1 weight column `channel` at each migrated layer
        (every output-channel row's own byte at that column, re-quantized:
        `round_half_away_from_zero(old_int8 * s_c)`, clamped to [-127, 127]).
    Every other byte in the file -- every other tensor, every other channel,
    the header, the section table -- is untouched. The integrity SHA-256 is
    recomputed at the end over the whole file with the hash field zeroed,
    per `docs/sslm_format.md`'s own §"Load-bearing choices" item 2."""
    with open(baseline_path, "rb") as f:
        data = bytearray(f.read())

    sections, _table_off = _section_table(bytes(data))
    weights_entry = next(e for e in sections if e["type"] == _SECTION_WEIGHTS)
    fold_entry = next(e for e in sections if e["type"] == _SECTION_WEIGHT_SCALES)

    weights_blob = bytes(data[weights_entry["offset"]:weights_entry["offset"] + weights_entry["byte_size"]])
    fold_blob = bytes(data[fold_entry["offset"]:fold_entry["offset"] + fold_entry["byte_size"]])

    result = BuildResult(channel=channel, alpha=float("nan"), out_path=out_path)

    for layer_idx, s_c in enumerate(layer_scales):
        down_name = f"layer{layer_idx}.down_proj"
        up_name = f"layer{layer_idx}.up_proj"

        down_data_off, down_elem_count, down_shape = _find_tensor_offset(weights_blob, down_name)
        up_fold_data_off, up_fold_elem_count, up_fold_shape = _find_tensor_offset(fold_blob, up_name)

        out_channels_down, in_channels_down = down_shape  # [hidden_size, intermediate_size]
        assert channel < in_channels_down

        abs_down_data_off = weights_entry["offset"] + down_data_off
        abs_up_fold_data_off = fold_entry["offset"] + up_fold_data_off

        if s_c == 1.0:
            # alpha=0's own no-op cell, or a layer this channel's own
            # activation never dominates at this alpha: write nothing.
            result.layer_stats.append(LayerPatchStats(
                layer_idx=layer_idx, s_c=1.0, up_mult_before=0, up_shift_before=0,
                up_mult_after=0, up_shift_after=0, n_saturated=0, n_elements=out_channels_down,
                max_abs_weight_quant_error=0.0, rmse_weight_quant_error=0.0))
            continue

        # --- down_proj: re-quantize column `channel`, every output-channel row ---
        old_codes = np.empty(out_channels_down, dtype=np.int64)
        for k in range(out_channels_down):
            byte_off = abs_down_data_off + k * in_channels_down + channel
            old_codes[k] = struct.unpack_from("<b", data, byte_off)[0]

        raw = old_codes.astype(np.float64) * s_c
        new_codes = np.trunc(raw + np.where(raw >= 0, 0.5, -0.5)).astype(np.int64)  # round-half-away-from-zero
        n_saturated = int(np.sum(np.abs(new_codes) > 127))
        new_codes = np.clip(new_codes, -127, 127)

        down_fold_data_off, _de, down_fold_shape = _find_tensor_offset(fold_blob, down_name)
        abs_down_fold_data_off = fold_entry["offset"] + down_fold_data_off
        real_scale_down = np.empty(out_channels_down, dtype=np.float64)
        for k in range(out_channels_down):
            triple_off = abs_down_fold_data_off + k * 3 * 4
            identity_k, mult_k, shift_k = struct.unpack_from("<iii", data, triple_off)
            real_scale_down[k] = 1.0 if identity_k else mult_k * (2.0 ** (-31.0 - shift_k))

        w_old = old_codes.astype(np.float64) * real_scale_down
        w_target = w_old * s_c
        w_new = new_codes.astype(np.float64) * real_scale_down
        err = np.abs(w_new - w_target)
        max_err = float(np.max(err))
        rmse_err = float(np.sqrt(np.mean(err ** 2)))

        for k in range(out_channels_down):
            byte_off = abs_down_data_off + k * in_channels_down + channel
            struct.pack_into("<b", data, byte_off, int(new_codes[k]))

        # --- up_proj: re-derive (mult, shift) for real_scale[channel] / s_c ---
        triple_off = abs_up_fold_data_off + channel * 3 * 4
        identity_before, mult_before, shift_before = struct.unpack_from("<iii", data, triple_off)
        assert identity_before == 0, (
            f"layer {layer_idx}: up_proj channel {channel} has identity=1 (pass-through) -- "
            f"this construction assumes a real (mult, shift) fold, confirmed absent by this task's "
            f"own diagnostic pass; re-check before proceeding")
        old_real_scale_up = mult_before * (2.0 ** (-31.0 - shift_before))
        new_real_scale_up = old_real_scale_up / s_c
        new_mult, new_shift = quantize_multiplier(new_real_scale_up)
        struct.pack_into("<iii", data, triple_off, 0, new_mult, new_shift)

        result.layer_stats.append(LayerPatchStats(
            layer_idx=layer_idx, s_c=s_c, up_mult_before=mult_before, up_shift_before=shift_before,
            up_mult_after=new_mult, up_shift_after=new_shift, n_saturated=n_saturated,
            n_elements=out_channels_down, max_abs_weight_quant_error=max_err,
            rmse_weight_quant_error=rmse_err))

    # --- recompute integrity SHA-256 (header field zeroed while hashing) ---
    header = bytearray(data[0:_HEADER_BYTES])
    header[32:64] = b"\x00" * 32
    data[0:_HEADER_BYTES] = header
    digest = hashlib.sha256(bytes(data)).digest()
    data[32:64] = digest

    with open(out_path, "wb") as f:
        f.write(data)

    with open(baseline_path, "rb") as f:
        baseline_bytes = f.read()
    result.bytes_changed = sum(1 for a, b in zip(baseline_bytes, data) if a != b)
    result.identical_to_baseline = (bytes(data) == baseline_bytes)
    return result
