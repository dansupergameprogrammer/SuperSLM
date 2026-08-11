"""The simulated integer pipeline for the T-066 spike (SuperSLM_Plan.md §15).

§15: "offline Python; simulated integer pipeline in torch — static scales, pinned
requant semantics, i-exp/i-sqrt constructions, RoPE tables — no runtime code
needed." This module is the forward pass that wires `intmath.py`, `rope.py`, and
`constrain.py` into a Llama-family decode (§5 / D-SLM8: RMSNorm, RoPE,
grouped-query attention, SwiGLU MLP, tied embeddings).

Three properties are structural here rather than incidental:

**Every rounding on the reproducible path is `intmath.py`'s** (§15 / Forge W6). No
library rounding runs between token input and token output — the vectorized kernels
below reimplement the gemmlowp primitives in numpy and are held to §6.2's bar:
bit-equal to the scalar reference on the exhaustive edge corpus. `forward_scalar_reference`
is what the fast path is bit-equal *to*, and it is the normative one.

**float64 is an exact integer carrier, and the envelope is checked.** int8 x int8
over the §5 dot lengths peaks 6e7 times under 2^53, so every partial sum is exactly
representable, so the float accumulation is exact and — the property worth having —
order-independent. The headroom is operand-width-specific: int32 probabilities
against int8 values break at dot length 33,026, and Qwen2.5's own context cap of
32,768 sits at 99.2% of it. `assert_exact_accumulation` is therefore wired into
`int_matmul` at the true break point, and the carrier's exit is a **cast, not a
round**: a rounding step there hides the violation the guard exists to raise.

**Static scales are constants of the model** (D-SLM5). They are calibrated offline at model
construction — `fixture_model` or `load_model` — and read, never computed, by `forward`.
`compute_activation_scale` is §6.2's spike-gated dynamic fallback and the baseline never
calls it. Calibration reads the **whole** frozen §4 corpus (Dan, 2026-07-15), so §11's
reproducibility formula carries the calibration corpus's hash and that hash is sufficient:
it determines the scales on its own, with no draw for the formula to express.

**Calibration observes the prompt the run decodes.** §6.2 calibrates "on a task-representative
corpus", and the run does not decode a corpus record's utterance — it decodes
`baseline.SYSTEM_PROMPT` and `baseline.build_prompt(record)` through the checkpoint's chat
template. Encoding the bare utterance set every per-tensor activation scale from 5-17 tokens
while the run processes 463-475, so the ~450-token system prompt — the most plausible home for
a massive activation, and §6.2's known cost is precisely "quality on outlier-heavy
activations" — was invisible to the scales that quantize it. `run_token_ids` is the one
construction and `calibration_token_ids` is defined as it, so the two cannot drift apart; the
prompt itself is imported from `baseline.py` rather than restated, so there is one of it.

**`load_model` is what makes the spike's number mean anything.** `fixture_model` was the
only `QuantizedModel` constructor and its weights are a bit-mix of each tensor's **name
string** — right for CI, where determinism is a property of the arithmetic, and exactly wrong
for a measurement of task quality. The weight map is total: an unmapped or missing tensor is
a hard rejection (§11's N3), because a silently-skipped projection is a model that loads,
runs, generates fluent text, and is not the source model.

**The float reference is the unquantized model, streamed per tensor.** §13 item 2 measures
quantization *against* it, so a reference rebuilt by dequantizing the int8 weights would
compare the integer path to itself and report zero damage. It is the same fact as the memory
one: the originals are already float in the checkpoint, and materialising 1.5B of them in
float64 at once is 11.5 GiB.

**The KV cache and the RoPE hoist are determinism changes wearing performance clothes.** Both
are required to be bit-identical to the paths they replace, and that is asserted rather than
argued: a cache is a *different computation* the moment a reduction order, a requant, or a
position index moves, and one wrong bit at step k propagates to every token after it.

**Weight scales are per output channel; activation scales are per tensor.** §6.1 pins
"W8: symmetric int8, per-output-channel static scales"; §6.2 pins "static per-tensor
activation scales". So at a projection the composed requant multiplier inherits the
weight's granularity —

    acc[j]   = SUM_i x[i] * W[i][j]
    q_out[j] = requant(acc[j], M[j]),  M[j] = S_x * S_w[j] / S_out

— and both the multiplier and the shift are vectors of length `out_channels`. They are
carried as two parallel arrays rather than a list of pairs: §6.2 pins gemmlowp by
reference implementation and the reference's per-axis API is two `int32*` arrays, which
§6.8's "pin the reference" rule makes normative rather than a matter of taste.

**§6.8 PIN OWED rows are not invented here.** C7/C8 (i-exp coefficients and clip
bound), C9 (softmax construction), and C10 (SwiGLU SiLU construction) are owed at S2.
This module composes the constructions `intmath.py` already implements rather than
choosing numbers for them, and emits no golden hash: per §6.8, no conformance is
claimed against a row that is not a number. The spike answers format-lock, which is a
property of the pipeline class; the golden waits for S2.

Test-design record: Claude/Curie/superslm-t066-pipeline-test-design-2026-07-15.md
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import zlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from superslm_spike import intmath, rope, silu_lut
from superslm_spike.constrain import argmax_masked, greedy_decode

__all__ = [
    "ConfigError",
    "UnsupportedOpSet",
    "ExactnessEnvelopeExceeded",
    "DynamicScaleForbidden",
    "ModelConfig",
    "RequantSite",
    "StaticScales",
    "QuantizedModel",
    "KVCache",
    "CalibrationRecord",
    "EXACT_INT_LIMIT",
    "CALIBRATION_CORPUS_PATH",
    "CALIBRATION_CLASSES",
    "calibration_records",
    "calibrate",
    "run_prompt_messages",
    "run_token_ids",
    "calibration_token_ids",
    "load_config",
    "load_model",
    "new_kv_cache",
    "with_rope_tables",
    "forward_float_layers",
    "attention_group_size",
    "is_degenerate_grouping",
    "assert_exact_accumulation",
    "accumulator_width",
    "int_matmul",
    "quantize_weight_per_channel",
    "dequantize_weight_per_channel",
    "vec_rounding_divide_by_pot",
    "vec_saturating_rounding_doubling_high_mul",
    "vec_multiply_by_quantized_multiplier",
    "vec_i_sqrt",
    "vec_i_exp",
    "vec_rope_apply_pair",
    "quantize_multiplier",
    "compute_activation_scale",
    "scales_used_by_forward",
    "dynamic_scale_trace",
    "forward_dynamic_scale",
    "canonical_scale",
    "fold_projection_accumulator",
    "emit_dynamic_bias",
    "forward_dynamic",
    "fixture_model",
    "fixture_model_biased",
    "embedding_weight",
    "lm_head_weight",
    "forward",
    "forward_scalar_reference",
    "forward_batch",
    "forward_layers",
    "forward_layers_float_reference",
    "forward_float_reference",
    "select_greedy",
    "select_greedy_masked",
    "decode_greedy",
    "decode_greedy_float",
    "decode_constrained",
    "decode_constrained_dynamic",
]


class ConfigError(ValueError):
    """A checkpoint config the loader rejects rather than defaulting past (§6.8 C15)."""


class UnsupportedOpSet(ValueError):
    """A model shape outside §5's operator set. Rejected, never floor-divided past."""


class ExactnessEnvelopeExceeded(OverflowError):
    """An accumulation whose worst case is not exactly representable in float64."""


class DynamicScaleForbidden(RuntimeError):
    """The baseline forward reached for §6.2's spike-gated dynamic-scale fallback."""


INT32_MIN = intmath.INT32_MIN
INT32_MAX = intmath.INT32_MAX

# float64 represents every integer up to 2^53 exactly, and that bound is the entire
# licence for carrying integer accumulations in float64.
EXACT_INT_LIMIT = 2 ** 53

# Fixed-point widths for the sim's own intermediates. Not §6.8 rows: these are the
# carrier's precision, not the runtime's kernel constants.
NORM_FRAC_BITS = 16
PROB_FRAC_BITS = 15
SIGMOID_FRAC_BITS = 15

# The scale of a fixture tensor that is pinned directly as int8 codes rather than
# quantized from a float weight: the embedding lookup table and the elementwise RMSNorm
# gains. Neither is a §6.1 projection — the embedding is a lookup whose tied use as the
# lm_head exits int32 with no requant to carry a channel scale, and a gain is elementwise,
# where "per output channel" degenerates to per element.
_UNIT_INT8_SCALE = 1.0 / 127.0

# §6.1's W8 tensors: the int8 weight matrices whose output channels each carry their own
# static scale. Stored (out_features, in_features), so the output axis is 0.
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_PROJECTION_OUTPUT_AXIS = 0

_INT8_MIN = -127
_INT8_MAX = 127
_SCALE_HEADROOM = 1.0 + 2.0 ** -20

# §6.2's "task-representative corpus", frozen and hashed (§11). The corpus's own bytes are a
# term in §11's reproducibility formula, so the calibration input is pinned by hash rather
# than described.
CALIBRATION_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "Claude" / "Docs" / "spike" / "shopkeeper_corpus_v1.jsonl"
)

# §4's six adversarial classes. Calibration consumes the whole corpus (Dan, 2026-07-15), so
# every class arrives at its full corpus share and the classes are named rather than drawn
# from: the list is the corpus's own contents, and an unnamed class is a corpus this
# calibration was not written for.
#
# The classes still matter to the reading. §4 separates them because negation and correction
# are where the hard activations live, and §6.2's known cost of static scales is exactly
# "quality on outlier-heavy activations" — which a subsample that ignored class would have
# calibrated past.
CALIBRATION_CLASSES = ("plain", "multi_slot", "negation", "correction",
                       "underspecified", "out_of_domain")


# ==============================================================================
# Config — read from config.json, never through a library Config object
# ==============================================================================


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    context_cap: int


def _required(body, key, path):
    if key not in body:
        raise ConfigError(f"{path}: '{key}' is absent; the loader rejects rather than defaulting")
    value = body[key]
    if value is None:
        raise ConfigError(f"{path}: '{key}' is null; a default here produces a model that runs and is not the source model")
    return value


def load_config(config_json_path) -> ModelConfig:
    """Read a checkpoint's `config.json` into a `ModelConfig` (§6.8 C15 / D-SLM38).

    θ is read from the file, per model, and is never defaulted. The library's Config
    object is not a permitted source: at transformers 5.13.1 `cfg.rope_theta` returns
    None because 5.x moved it to `cfg.rope_parameters`, so the 4.x idiom emits §6.4's
    tables on a wrong θ and produces a model that loads, runs, generates fluent text,
    and is not the source model. This function reads the JSON itself.
    """
    path = Path(config_json_path)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot be read ({exc})") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: is not valid JSON ({exc})") from None

    theta = _required(body, "rope_theta", path)
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise ConfigError(f"{path}: 'rope_theta' is {theta!r}, not a number")

    hidden_size = int(_required(body, "hidden_size", path))
    num_attention_heads = int(_required(body, "num_attention_heads", path))
    head_dim = body.get("head_dim")
    if head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise ConfigError(
                f"{path}: hidden_size {hidden_size} is not a multiple of num_attention_heads "
                f"{num_attention_heads} and no explicit head_dim is present"
            )
        head_dim = hidden_size // num_attention_heads

    return ModelConfig(
        hidden_size=hidden_size,
        num_hidden_layers=int(_required(body, "num_hidden_layers", path)),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(_required(body, "num_key_value_heads", path)),
        head_dim=int(head_dim),
        intermediate_size=int(_required(body, "intermediate_size", path)),
        vocab_size=int(_required(body, "vocab_size", path)),
        rope_theta=float(theta),
        rms_norm_eps=float(_required(body, "rms_norm_eps", path)),
        tie_word_embeddings=bool(_required(body, "tie_word_embeddings", path)),
        context_cap=int(_required(body, "max_position_embeddings", path)),
    )


def attention_group_size(cfg: ModelConfig) -> int:
    """§5's grouped-query attention group size, `n_q // n_kv`.

    A head count that does not divide has no grouping; floor-dividing past it computes
    a different attention silently, so it is rejected (§11 reject-over-degrade).
    """
    if cfg.num_key_value_heads <= 0:
        raise UnsupportedOpSet(f"num_key_value_heads must be positive; got {cfg.num_key_value_heads}")
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
        raise UnsupportedOpSet(
            f"num_attention_heads {cfg.num_attention_heads} is not a whole multiple of "
            f"num_key_value_heads {cfg.num_key_value_heads}; §5's op set mandates grouped-query "
            f"attention and this shape has no grouping"
        )
    return cfg.num_attention_heads // cfg.num_key_value_heads


def is_degenerate_grouping(cfg: ModelConfig) -> bool:
    """Whether this config is full MHA — GQA at group size 1 (§5, 2026-07-14 bench).

    Named rather than tolerated: a pipeline that mandates GQA and silently accepts
    `n_kv == n_q` is passing a case it never decided to pass, and an accidental pass
    is indistinguishable from a missing check.
    """
    return attention_group_size(cfg) == 1


# ==============================================================================
# The exact integer carrier
# ==============================================================================


def assert_exact_accumulation(dot_length: int, max_abs_lhs: int, max_abs_rhs: int) -> None:
    """Raise unless every partial sum of the contraction is exactly representable.

    The bound is the true break point, not a conservative round number: an int32
    probability against an int8 value breaks at dot length 33,026 while Qwen2.5's own
    context cap is 32,768 — a 0.78% margin. A guard off by a factor of two either
    rejects a legal shape or admits an inexact one, and the inexact one is silent.
    """
    if dot_length < 0 or max_abs_lhs < 0 or max_abs_rhs < 0:
        raise ValueError(
            f"the envelope is stated in magnitudes; got dot_length={dot_length}, "
            f"max_abs_lhs={max_abs_lhs}, max_abs_rhs={max_abs_rhs}"
        )
    peak = int(dot_length) * int(max_abs_lhs) * int(max_abs_rhs)
    if peak > EXACT_INT_LIMIT:
        raise ExactnessEnvelopeExceeded(
            f"a contraction of length {dot_length} at magnitudes {max_abs_lhs} x {max_abs_rhs} "
            f"peaks at {peak}, past float64's exactly-representable bound {EXACT_INT_LIMIT}; "
            f"the accumulation would silently drop low bits and return a plausible number"
        )


def accumulator_width(dot_length: int, max_abs_lhs: int, max_abs_rhs: int) -> int:
    """§6.8 C17's per-layer accumulator width, derived from the contraction's worst case.

    C17: "int32, or int64 where dot-length x magnitude requires it — decided per layer at
    conversion from worst-case bounds and recorded in the artifact." The bound is the
    largest magnitude the accumulator can reach, `dot_length * max_abs_lhs * max_abs_rhs`,
    and the choice is whether signed int32 holds it.

    The sim accumulates exactly (`int_matmul`), which models an accumulator always wide
    enough. That models the runtime faithfully only while C17's choice actually covers the
    bound, so the derivation is a function rather than a claim. For every §5 shape the
    worst case is `D * 127 * 127`, which fits int32 with at least 4.1x headroom and first
    exceeds it at dot length 133,145.
    """
    if dot_length < 0 or max_abs_lhs < 0 or max_abs_rhs < 0:
        raise ValueError(
            f"the bound is stated in magnitudes; got dot_length={dot_length}, "
            f"max_abs_lhs={max_abs_lhs}, max_abs_rhs={max_abs_rhs}"
        )
    peak = int(dot_length) * int(max_abs_lhs) * int(max_abs_rhs)
    return 32 if peak <= INT32_MAX else 64


def _round_half_away_from_zero(values):
    """Offline rounding for the weight quantizer, in `RoundingDivideByPOT`'s direction.

    Spelled out rather than delegated: §15/W6 bars library rounding from the reproducible
    path, and `np.round`'s round-half-even would put a different tie rule in the converter
    than the one C3 pins for the runtime.
    """
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _channel_shape(ndim: int, axis: int):
    shape = [1] * ndim
    shape[axis] = -1
    return tuple(shape)


def _output_axis(weight, output_axis: int) -> int:
    values = np.asarray(weight)
    if not -values.ndim <= output_axis < values.ndim:
        raise ValueError(f"output_axis {output_axis} is outside a {values.ndim}-D weight")
    return output_axis % values.ndim


def quantize_weight_per_channel(weight, output_axis: int):
    """§6.1's W8: symmetric int8 codes plus one static scale per output channel.

    Symmetric means the grid is symmetric about zero — no zero point, and the negative end
    is -127 rather than -128, because -128 has no positive counterpart. Dequantization is
    therefore a bare multiply.

    Each channel's scale comes from that channel's own max-abs, which is the entire point:
    a single per-tensor scale is set by the widest channel and crushes the resolution of
    every other one. A channel whose columns differ from its neighbour's by 1270x gets a
    scale that differs by 1270x.
    """
    values = np.asarray(weight, dtype=np.float64)
    axis = _output_axis(values, output_axis)
    reduced = tuple(a for a in range(values.ndim) if a != axis)
    peaks = np.abs(values).max(axis=reduced) if reduced else np.abs(values)
    # An all-zero channel has no range to represent; the unit scale keeps zero at zero
    # and keeps the scale strictly positive, which the requant composition requires.
    scales = np.where(peaks > 0.0, peaks / _INT8_MAX, 1.0)
    divisor = scales.reshape(_channel_shape(values.ndim, axis))
    codes = np.clip(_round_half_away_from_zero(values / divisor), _INT8_MIN, _INT8_MAX)
    return codes.astype(np.int8), tuple(float(s) for s in np.atleast_1d(scales))


def dequantize_weight_per_channel(q, scales, output_axis: int):
    """The inverse of `quantize_weight_per_channel`: `w ~= code * S_w[j]`, no zero point."""
    codes = np.asarray(q, dtype=np.float64)
    axis = _output_axis(codes, output_axis)
    channel_scales = np.asarray(scales, dtype=np.float64)
    if channel_scales.shape != (codes.shape[axis],):
        raise ValueError(
            f"{channel_scales.shape[0] if channel_scales.ndim else 1} scales for "
            f"{codes.shape[axis]} output channels; §6.1 pins one per output channel"
        )
    return codes * channel_scales.reshape(_channel_shape(codes.ndim, axis))


def _max_abs(values):
    """An operand's largest magnitude, read without materialising a converted copy.

    `np.abs(x.astype(np.int64)).max()` is the obvious spelling and it allocates a full second
    copy of the operand on every call — at 1.5B weights per decoded token that is the
    dominant cost of the whole forward, and it is spent on a bound that two scalars answer.
    Reading `min`/`max` in the operand's own dtype and taking the magnitude in Python ints is
    exact for every input, and it is also *correct where `np.abs` is not*: `np.abs` on int8
    returns -128 for -128, which is the one input the envelope most needs to be right about.
    """
    array = np.asarray(values)
    if array.size == 0:
        return 0
    return max(abs(int(array.max())), abs(int(array.min())))


def int_matmul(a, b):
    """An exact integer matmul carried in float64.

    Every partial sum is an exactly-representable integer, so every float addition is
    exact, so any summation order — including a threaded BLAS GEMM's — yields identical
    bits. The envelope is checked on the actual operands rather than assumed.

    The exit is a **cast, not a round**. If the accumulation is exact a rounding step is
    a no-op; if it is not exact, the rounding step hides the violation
    `assert_exact_accumulation` exists to raise, so the guard would be disarmed by the
    operation that looks like safety.

    `np.asarray` rather than `astype` at the entry: an operand already carried as float64 is
    passed through instead of copied, which is what lets a caller hold the widened weight once
    rather than rebuild it per token. The values are identical either way — int8 to float64 is
    exact — so this is the cost changing and not the arithmetic.
    """
    left = np.asarray(a)
    right = np.asarray(b)
    assert_exact_accumulation(
        dot_length=int(left.shape[-1]),
        max_abs_lhs=_max_abs(left),
        max_abs_rhs=_max_abs(right),
    )
    product = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
    return product.astype(np.int64)


def _scalar_matmul(a, b):
    """`int_matmul`'s normative scalar reference: exact Python ints, same envelope."""
    rows = len(a)
    dot = len(a[0]) if rows else 0
    cols = len(b[0]) if b else 0
    assert_exact_accumulation(
        dot_length=dot,
        max_abs_lhs=max((abs(int(v)) for row in a for v in row), default=0),
        max_abs_rhs=max((abs(int(v)) for row in b for v in row), default=0),
    )
    out = []
    for row in a:
        out_row = []
        for column in range(cols):
            total = 0
            for k in range(dot):
                total += int(row[k]) * int(b[k][column])
            out_row.append(total)
        out.append(out_row)
    return out


# ==============================================================================
# Vectorized kernels — bit-equal to intmath.py / rope.py on the edge corpus
# ==============================================================================


def vec_rounding_divide_by_pot(x, exponent: int):
    """`intmath.rounding_divide_by_pot`, vectorized. Ties away from zero (§6.8 C3)."""
    values = np.asarray(x, dtype=np.int64)
    mask = np.int64((1 << exponent) - 1)
    remainder = values & mask
    threshold = (mask >> np.int64(1)) + (values < 0).astype(np.int64)
    return (values >> np.int64(exponent)) + (remainder > threshold).astype(np.int64)


def vec_saturating_rounding_doubling_high_mul(a, b):
    """`intmath.saturating_rounding_doubling_high_mul`, vectorized.

    Ties toward +infinity (§6.8 C2) — the direction that differs from its sibling, and
    the one a vectorized rewrite is most likely to homogenise. `(INT32_MIN, INT32_MIN)`
    is the only int32 pair whose defined result exceeds INT32_MAX; it saturates.
    """
    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    result = (left * right + np.int64(1 << 30)) >> np.int64(31)
    return np.where(result > np.int64(INT32_MAX), np.int64(INT32_MAX), result)


def vec_multiply_by_quantized_multiplier(x, quantized_multiplier: int, shift: int):
    """§6.8 C1's composed requant, vectorized: both primitives, in order, both roundings."""
    return vec_rounding_divide_by_pot(
        vec_saturating_rounding_doubling_high_mul(x, quantized_multiplier), shift
    )


def vec_i_sqrt(n):
    """`intmath.i_sqrt`, vectorized: the digit recurrence as 32 unconditional masked steps.

    The per-digit branch becomes a select, so the op count stays a constant of the type
    (§6.8 C5) rather than becoming a function of the radicand.
    """
    radicand = np.asarray(n, dtype=np.int64)
    if np.any(radicand < 0):
        raise ValueError("i_sqrt is defined on non-negative sums (n >= 0)")
    remainder = radicand.copy()
    root = np.zeros_like(radicand)
    bit = 1 << 62
    for _ in range(intmath.I_SQRT_ITERATIONS):
        trial = root + np.int64(bit)
        take = remainder >= trial
        remainder = np.where(take, remainder - trial, remainder)
        root = np.where(take, (root >> np.int64(1)) + np.int64(bit), root >> np.int64(1))
        bit >>= 2
    return root


def vec_i_exp(q, scale: float):
    """`intmath.i_exp`, vectorized: the I-BERT second-order polynomial on shifted logits.

    The `>> z` becomes a per-lane variable shift — the step most likely to diverge from
    the scalar form, and the one the edge cells drive across the branch boundary and
    into the clipped far tail.
    """
    values = np.asarray(q, dtype=np.int64)
    if np.any(values > 0):
        raise ValueError("i_exp is defined on max-shifted logits (q <= 0)")

    q_ln2 = intmath.i_exp_ln2_quantum(scale)
    if q_ln2 < 1:
        raise ValueError(
            f"i_exp is defined where the ln2 quantum floor(ln2/scale) is positive; got scale={scale}"
        )
    q_b = math.floor(intmath._POLY_B / scale)
    q_c = math.floor(intmath._POLY_C / (intmath._POLY_A * scale ** 2))
    out_scale = intmath._POLY_A * scale ** 2

    clipped = np.maximum(values, np.int64(-intmath.I_EXP_CLIP_N * q_ln2))
    z = (-clipped) // np.int64(q_ln2)
    q_p = clipped + z * np.int64(q_ln2)
    q_out = ((q_p + np.int64(q_b)) ** 2 + np.int64(q_c)) >> z
    return q_out, out_scale


def vec_rope_apply_pair(x, y, cos_q30, sin_q30):
    """`rope.rope_apply_pair`, vectorized: full-width combine, one rounding (§6.8 C13)."""
    xs = np.asarray(x, dtype=np.int64)
    ys = np.asarray(y, dtype=np.int64)
    cs = np.asarray(cos_q30, dtype=np.int64)
    ss = np.asarray(sin_q30, dtype=np.int64)
    return (
        vec_rounding_divide_by_pot(xs * cs - ys * ss, rope.ROPE_FRAC_BITS),
        vec_rounding_divide_by_pot(xs * ss + ys * cs, rope.ROPE_FRAC_BITS),
    )


# ==============================================================================
# Static scales — constants of the model, calibrated offline (D-SLM5)
# ==============================================================================


def quantize_multiplier(factor: float) -> tuple[int, int]:
    """A real multiplier in (0, 1) as gemmlowp's `(quantized_multiplier, shift)` pair.

    Offline only. §6.2's requant consumes the integer pair; the float never reaches the
    reproducible path.
    """
    if not 0.0 < factor < 1.0:
        raise ValueError(
            f"the §6.2 requant form represents multipliers in (0, 1); got {factor}. A scale "
            f"chain that needs a left shift is a calibration defect, not a rounding question"
        )
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
            f"multiplier {factor} needs a shift of {shift}, outside RoundingDivideByPOT's "
            f"defined range [0, 31]"
        )
    return multiplier, shift


def _round_half_even_ratio(numerator: int, denominator: int) -> int:
    """Offline half-even rounding of an exact rational (C14's precedent, cited by C26 for
    the offline canonical-scale emission). `numerator`/`denominator` are exact ints (from a
    `Fraction`); the tie rule matters only at an exact `.5`, which `Fraction` represents
    exactly rather than approximates."""
    if denominator <= 0:
        raise ValueError(f"denominator must be > 0; got {denominator}")
    q, r = divmod(numerator, denominator)
    twice = 2 * r
    if twice > denominator or (twice == denominator and q % 2 == 1):
        q += 1
    return q


def canonical_scale(value) -> tuple[int, int]:
    """C26's offline canonical-scale emission: a positive real scale as `(m, e)` with
    `value ~= m * 2**e`, mantissa `m` in `[2**30, 2**31)` (C21-normalized width), rounded
    HALF-EVEN (C14's offline-emission precedent). `value` is converted through `Fraction`
    (exact for an int, a `Fraction`, or a float — `Fraction(float)` is exact, so an offline
    float constant converts losslessly) so the whole derivation is exact rational arithmetic,
    never a float divide.
    """
    value = Fraction(value)
    if value <= 0:
        raise ValueError(f"canonical scale must be positive; got {value}")
    # e such that value / 2**e lands in [2**30, 2**31); seeded from the bit-length gap and
    # walked to the exact boundary (Fraction has no bit_length shortcut of its own).
    e = value.numerator.bit_length() - value.denominator.bit_length() - 31
    while value / Fraction(2) ** (e + 31) >= 1:
        e += 1
    while value / Fraction(2) ** (e + 30) < 1:
        e -= 1
    scaled = value / Fraction(2) ** e
    m = _round_half_even_ratio(scaled.numerator, scaled.denominator)
    if m == 1 << 31:                                    # rounded up to the ceiling
        m >>= 1
        e += 1
    assert (1 << 30) <= m < (1 << 31), f"C26 postcondition violated: m={m} for value={value}"
    return m, e


def emit_dynamic_bias(bias_floats, s_ref, q_b: int = 30):
    """C28's offline bias-storage emission: `B[j] = round_half_even(b[j] / S_ref · 2^q_B)`
    — the exact integers stored at the projection's fold reference `S_ref` in Q-format
    `q_B` (30 for this converter; the row leaves q_B per conversion). The rounding mode is
    ADOPTED from C14's offline half-even precedent (A-8 §17.1; folded into the C28 row's
    emission clause, commit 90c313f). Exact rational arithmetic throughout —
    `Fraction(float)` is exact, so the float biases convert losslessly before the one
    rounding.
    """
    if q_b < 0:
        raise ValueError(f"the C28 Q-format is non-negative; got q_b={q_b}")
    s_ref = Fraction(s_ref)
    if s_ref <= 0:
        raise ValueError(f"the C28 storage scale S_ref must be positive; got {s_ref}")
    codes = []
    for b in bias_floats:
        scaled = Fraction(float(b)) / s_ref * (1 << q_b)
        codes.append(_round_half_even_ratio(scaled.numerator, scaled.denominator))
    return tuple(codes)


def fold_projection_accumulator(acc_row, folds):
    """C24/C25: fold a §6.1 projection's wide accumulator row to the common reference
    channel `S_ref = max_j S_w[j]`, one rounding per non-identity channel.

    `folds[j] is None` marks the identity (max) channel — the widest weight channel is
    already at `S_ref`, so it is a TRUE PASS-THROUGH: no multiply, no shift (C24 — the
    pinned `(0, 1)` multiplier domain has no representable `1.0`, and the near-identity
    `(2**31 - 1, 0)` is off by one on `|x| in (2**30, 2**31]` and code-divergent). Every
    other channel takes one `intmath.multiply_by_quantized_multiplier` at its offline
    `(Mw[j], shw[j])` constant (C25) — the pinned C1 composite exactly (C2 ties toward
    +infinity on the high-mul, C3 ties away from zero on the divide).
    """
    if len(acc_row) != len(folds):
        raise ValueError(
            f"fold table must match the accumulator row: {len(folds)} folds for "
            f"{len(acc_row)} channels"
        )
    folded = []
    for a, fold in zip(acc_row, folds):
        if fold is None:
            folded.append(int(a))
        else:
            multiplier, shift = fold
            folded.append(intmath.multiply_by_quantized_multiplier(int(a), multiplier, shift))
    return folded


@dataclass(frozen=True)
class RequantSite:
    """A §6.1 projection's requant: an int8 weight matmul's output, back to int8.

    `multipliers` and `shifts` are two parallel arrays indexed by output channel, matching
    gemmlowp's per-axis `const int32*` pair, and `M[j] = S_x * S_w[j] / S_out` is why they
    carry the channel index at all: the activation scales enter as scalars (§6.2) and the
    weight scale carries `j` (§6.1).
    """

    name: str
    input_scale: float
    output_scale: float
    weight_scales: tuple[float, ...]
    multipliers: tuple[int, ...]
    shifts: tuple[int, ...]


@dataclass(frozen=True)
class StaticScales:
    """The model's offline-calibrated scales. Constants of the artifact, never computed
    at inference (D-SLM5).

    `requant` holds the §6.1 projection sites, where a per-output-channel weight scale
    composes into §6.2's requant. `rescale` holds the per-tensor sites that consume no
    weight — the norm outputs, the residual adds, the attention context, and the SwiGLU
    product — where there is no weight scale and so no channel index. `nonlinear` holds
    the offline float scales the §6.3 constructions are stated in, the same discipline
    `intmath.i_exp` already takes: the constant is offline, the arithmetic that consumes
    it is integer.
    """

    requant: tuple[RequantSite, ...]
    rescale: tuple[tuple[str, int, int], ...]
    nonlinear: tuple[tuple[str, float], ...]

    def requant_sites(self) -> tuple[str, ...]:
        return tuple(site.name for site in self.requant)

    def site(self, name: str) -> RequantSite:
        for site in self.requant:
            if site.name == name:
                return site
        raise KeyError(name)

    def input_scale(self, name: str) -> float:
        return self.site(name).input_scale

    def output_scale(self, name: str) -> float:
        return self.site(name).output_scale

    def weight_scales(self, name: str) -> tuple[float, ...]:
        return self.site(name).weight_scales

    def requant_for(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        site = self.site(name)
        return site.multipliers, site.shifts

    def rescale_sites(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.rescale)

    def rescale_for(self, name: str) -> tuple[int, int]:
        return {n: (m, s) for n, m, s in self.rescale}[name]

    def scale(self, name: str) -> float:
        return dict(self.nonlinear)[name]


def compute_activation_scale(x):
    """§6.2's **fallback**: a dynamic per-token scale from an integer max-abs reduction.

    Designed now, switched on only by spike evidence. The baseline forward never calls
    it: if it did, the spike would measure the fallback and report it as the baseline,
    and the format-lock answer would be for a pipeline we did not decide to ship.
    """
    values = np.asarray(x, dtype=np.int64)
    return int(np.abs(values).max(initial=0))


class _ScaleReader:
    """The forward's only access to the model's scales, and the record of what it read."""

    def __init__(self, scales: StaticScales):
        self._scales = scales
        self.read_requant: dict[str, RequantSite] = {}
        self.read_rescale: dict[str, tuple[int, int]] = {}
        self.read_nonlinear: dict[str, float] = {}

    def requant_for(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        site = self._scales.site(name)
        self.read_requant[name] = site
        return site.multipliers, site.shifts

    def rescale_for(self, name: str) -> tuple[int, int]:
        value = self._scales.rescale_for(name)
        self.read_rescale[name] = value
        return value

    def scale(self, name: str) -> float:
        value = self._scales.scale(name)
        self.read_nonlinear[name] = value
        return value

    def as_static_scales(self) -> StaticScales:
        return StaticScales(
            requant=tuple(sorted(self.read_requant.values(), key=lambda site: site.name)),
            rescale=tuple(sorted((n, m, s) for n, (m, s) in self.read_rescale.items())),
            nonlinear=tuple(sorted(self.read_nonlinear.items())),
        )


# ==============================================================================
# The model
# ==============================================================================


@dataclass(frozen=True)
class QuantizedModel:
    """The quantized artifact: int8 codes, their scales, and the constants of the forward.

    `rope_tables` is carried rather than rebuilt (§6.4 / C12). It was previously constructed
    inside every `forward()` — a pure-Python `context_cap x head_dim//2` `math.cos`/`math.sin`
    loop, ~1.3 s per token at Qwen2.5's 32768-row cap. It is a constant of the config, so it
    is built once at model construction and read from here. C12 still governs its shape:
    `context_cap` rows, indices `[0, context_cap)`, upper bound EXCLUSIVE.

    `float_source` is the **unquantized** model, one tensor at a time. §13 item 2's float
    reference is what quantization is measured *against*, so it cannot be the int8 weights
    dequantized — that compares the integer path to itself and reports zero damage, and it
    could never match upstream to G-2's bound. It is also why the access is per tensor: a
    whole-model float64 materialisation of 1.5B params is 11.5 GiB and does not run.

    `biases` are the int32 bias codes of the §6.1 projections that carry one (Qwen2.5's
    q/k/v). They are stated at the accumulator's own scale, `S_x * S_w[j]`, so they add into
    the int32 accumulator before the requant and no separate rounding is introduced.
    """

    config: ModelConfig
    scales: StaticScales
    weights: dict
    weight_scales: dict
    residual_scales: dict
    rope_tables: tuple
    biases: dict
    float_source: object
    calibration: CalibrationRecord
    gemm_weights: dict
    tokenize_prompt: object
    composition_constants: dict = dataclasses.field(default_factory=dict)
    kv_landing_scales: dict = dataclasses.field(default_factory=dict)
    kv_landing_reciprocals: dict = dataclasses.field(default_factory=dict)
    # C28's dynamic-arm bias storage (A-8 §17.1): `dynamic_biases[f"{prefix}.{name}"]
    # -> (q_B, codes)` — B[j] exact integers at the projection's fold reference S_ref,
    # Q-format q_B (30 for this converter). A frozen-dataclass FIELD (A-9 refinement):
    # twins are built via `dataclasses.replace`, never post-build mutation. Default
    # empty — the unbiased fixtures and the static arm are untouched by its existence.
    dynamic_biases: dict = dataclasses.field(default_factory=dict)
    # T-1894 (T-1822 design Sec31.2.1/Sec31.2.3, D-SLM2355/D-SLM2357): the Python-side
    # sibling of the compiled artifact's header `flags` bit
    # (`SslmArtifact.OptionGFusedKLandingEnabled()`, forward_sites.cpp) -- a property
    # of the loaded model, not a caller-supplied default. `dynamic_engine.py`'s own
    # `forward_dynamic_vec`/`_forward_dynamic_vec_layers`/`MicroStepState` all resolve
    # their own `option_g_fused_k_landing` parameter from here when it is not
    # explicitly overridden. Default False: no conversion path this pass writes sets
    # it, matching the compiled side's own "every existing shipped artifact has
    # flags == 0" baseline -- construct a twin with it set via
    # `with_option_g_fused_k_landing` (below), the same `dataclasses.replace`
    # convention `with_rope_tables` already establishes.
    option_g_fused_k_landing: bool = False

    def weight_names(self) -> list[str]:
        return list(self.weights)

    def weight(self, name: str):
        return self.weights[name]

    def float_weight(self, name: str):
        """The tensor's ORIGINAL float value — the unquantized model, one tensor at a time."""
        return self.float_source(name)


def _as_rope_tables(tables):
    """The carried tables as int64 arrays, whatever shape they arrive in."""
    cos_rows, sin_rows = tables
    return (np.asarray(cos_rows, dtype=np.int64), np.asarray(sin_rows, dtype=np.int64))


def _build_rope_tables(cfg: ModelConfig):
    """§6.4's tables for this config, built once. C12: `context_cap` rows, `[0, context_cap)`."""
    return _as_rope_tables(rope.rope_tables(cfg.head_dim, cfg.context_cap, cfg.rope_theta))


def with_rope_tables(model: QuantizedModel, tables) -> QuantizedModel:
    """The same model carrying different §6.4 tables.

    Exists so the hoist is testable at all: the failure mode of hoisting is a model that
    carries correct tables while the forward builds its own anyway, and nothing but
    perturbing the carried tables can see that.
    """
    return dataclasses.replace(model, rope_tables=_as_rope_tables(tables))


def with_option_g_fused_k_landing(model: QuantizedModel, enabled: bool) -> QuantizedModel:
    """T-1894 (design Sec31.2.1/Sec31.2.3): the same model with its own
    `option_g_fused_k_landing` field set -- the Python-side construction that stands
    in for setting the artifact header `flags` bit before load, since this converter
    pass does not itself write it. `dynamic_engine.py`'s own entry points resolve
    their default from `model.option_g_fused_k_landing`; this is how a caller (a test,
    or a future real converter path) produces a model that resolves to the fused
    order without threading an explicit override through every call.
    """
    return dataclasses.replace(model, option_g_fused_k_landing=bool(enabled))


def embedding_weight(model: QuantizedModel):
    return model.weights["embed"]


def lm_head_weight(model: QuantizedModel):
    """The embedding matrix itself when the config ties (§5 / D-SLM8).

    A pipeline that ignores the flag and keeps a separate head silently uses duplicated
    weights, and two of the three §16 nominees tie.
    """
    if model.config.tie_word_embeddings:
        return model.weights["embed"]
    return model.weights["lm_head"]


def _bit_mix(count: int, name: str):
    """A deterministic integer bit-mix over a flat index (§11: pinned, not sampled).

    A float RNG or an unseeded shuffle would make the gate's own reference
    nondeterministic.
    """
    index = np.arange(count, dtype=np.uint64)
    with np.errstate(over="ignore"):        # the bit-mix's wraparound is the construction
        seed = np.uint64(zlib.crc32(name.encode("utf-8"))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (index + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15) + seed
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def _pinned_int8(shape, name: str):
    """Deterministic int8 codes from the tensor's name (§11: pinned, not sampled)."""
    count = 1
    for extent in shape:
        count *= extent
    values = (_bit_mix(count, name) % np.uint64(255)).astype(np.int64) - 127
    return values.reshape(shape).astype(np.int8)


def _pinned_float_weight(shape, name: str, output_axis: int):
    """A §6.1 projection's float weight, before quantization, with channels that differ.

    The fixture's whole job at this seam is to exercise per-output-channel quantization,
    and a weight whose channels happen to share a dynamic range cannot tell a per-channel
    scale from a per-tensor one. So each output channel is pinned to its own magnitude —
    an exact power of two spanning 64x — over a pinned int8 pattern.

    The magnitudes run downward from 1.0 rather than outward around it: the widest channel
    then lands on the unit int8 scale, which keeps the activation-scale chain the
    calibration derives in the same range the rest of the fixture is stated in.
    """
    pattern = _pinned_int8(shape, name).astype(np.float64) / _INT8_MAX
    exponents = (_bit_mix(shape[output_axis], f"{name}::channel-range") % np.uint64(7))
    ranges = np.exp2(-exponents.astype(np.float64))
    return pattern * ranges.reshape(_channel_shape(len(shape), output_axis))


def _is_projection(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _PROJECTIONS


def _pinned_weights(cfg: ModelConfig):
    """The fixture's int8 codes, the scales that define them, and the floats behind them.

    The §6.1 projections are quantized through `quantize_weight_per_channel` — the
    converter's own path, so the fixture measures it rather than describing it. The
    embedding and the norm gains are pinned directly as codes at the unit scale.

    The float weights are returned rather than reconstructed: they are the fixture's
    *originals*, and §13 item 2's reference is the unquantized model. Dequantizing the codes
    back would give the reference the quantizer's own error and make the parity gate compare
    the integer path to itself.
    """
    codes: dict = {}
    weight_scales: dict = {}
    floats: dict = {}
    for name, shape in _weight_shapes(cfg):
        if _is_projection(name):
            floats[name] = _pinned_float_weight(shape, name, _PROJECTION_OUTPUT_AXIS)
            codes[name], weight_scales[name] = quantize_weight_per_channel(
                floats[name], output_axis=_PROJECTION_OUTPUT_AXIS)
        else:
            codes[name] = _pinned_int8(shape, name)
            weight_scales[name] = (_UNIT_INT8_SCALE,)
            # Pinned as codes, so the code IS the original and the unit scale is exact.
            floats[name] = np.asarray(codes[name], dtype=np.float64) * _UNIT_INT8_SCALE
    return codes, weight_scales, floats


def _weight_shapes(cfg: ModelConfig):
    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    shapes = [("embed", (cfg.vocab_size, cfg.hidden_size))]
    for layer in range(cfg.num_hidden_layers):
        shapes.extend([
            (f"layer{layer}.attn_norm.gain", (cfg.hidden_size,)),
            (f"layer{layer}.q_proj", (q_width, cfg.hidden_size)),
            (f"layer{layer}.k_proj", (kv_width, cfg.hidden_size)),
            (f"layer{layer}.v_proj", (kv_width, cfg.hidden_size)),
            (f"layer{layer}.o_proj", (cfg.hidden_size, q_width)),
            (f"layer{layer}.mlp_norm.gain", (cfg.hidden_size,)),
            (f"layer{layer}.gate_proj", (cfg.intermediate_size, cfg.hidden_size)),
            (f"layer{layer}.up_proj", (cfg.intermediate_size, cfg.hidden_size)),
            (f"layer{layer}.down_proj", (cfg.hidden_size, cfg.intermediate_size)),
        ])
    shapes.append(("final_norm.gain", (cfg.hidden_size,)))
    if not cfg.tie_word_embeddings:
        shapes.append(("lm_head", (cfg.vocab_size, cfg.hidden_size)))
    return shapes


def _calibration_record(tokenization: str) -> CalibrationRecord:
    return CalibrationRecord(
        corpus_sha256=_corpus_sha256(),
        tokenization=tokenization,
        classes=CALIBRATION_CLASSES,
    )


def _dict_float_source(floats: dict):
    """A `float_weight` over already-resident tensors, for a model small enough to hold."""
    def float_weight(name: str):
        return floats[name]
    return float_weight


def fixture_model(cfg: ModelConfig) -> QuantizedModel:
    """§11's fixture model: a few-layer toy transformer with pinned int8 weights.

    Determinism is a property of the arithmetic, not of whether the model is any good,
    so the weights are pinned integers and the scales are calibrated offline against the
    float reference — the converter's own discipline, at the fixture's scale.
    """
    attention_group_size(cfg)
    if cfg.head_dim % 2 != 0:
        raise UnsupportedOpSet(f"§6.4's rotation is pairwise; head_dim must be even, got {cfg.head_dim}")

    weights, weight_scales, floats = _pinned_weights(cfg)
    float_weight = _dict_float_source(floats)
    tokenize_prompt = _fixture_tokenize_prompt(cfg)
    records = calibration_records()
    maxima = _calibrate(cfg, float_weight, records,
                        lambda record: tokenize_prompt(run_prompt_messages(record)))
    scales, residual_scales, biases = _derive_scales(cfg, maxima, weight_scales, {})
    composition_constants, kv_landing_scales, kv_landing_reciprocals = _derive_composition_constants(
        cfg, weight_scales, scales)
    return QuantizedModel(config=cfg, scales=scales, weights=weights,
                          weight_scales=weight_scales, residual_scales=residual_scales,
                          rope_tables=_build_rope_tables(cfg), biases=biases,
                          float_source=float_weight, tokenize_prompt=tokenize_prompt,
                          calibration=_calibration_record(_FIXTURE_TOKENIZATION),
                          gemm_weights={}, composition_constants=composition_constants,
                          kv_landing_scales=kv_landing_scales,
                          kv_landing_reciprocals=kv_landing_reciprocals)


def fixture_model_biased(cfg: ModelConfig) -> QuantizedModel:
    """§11's fixture model plus the pinned pseudo-random C28 dynamic biases (A-8 §17.1):
    one entry per q/k/v projection per layer, integer codes drawn `random.Random(101)`
    within ±2^40 at q_B = 30.

    The magnitude is an EXECUTED pin, not a choice: at the fixture's scale regime ±2^22
    biases underflow to zero contribution (0/128 logits move), ±2^40 moves 128/128 logit
    elements, and ±2^50 trips C30's coarse-scale rejection — so ±2^40 is the decisive
    class. Built as a twin of `fixture_model` via `dataclasses.replace` (the frozen-field
    idiom, A-9): the plain fixture and every cell over it are untouched.
    """
    model = fixture_model(cfg)
    rng = random.Random(101)
    widths = {"q_proj": cfg.num_attention_heads * cfg.head_dim,
              "k_proj": cfg.num_key_value_heads * cfg.head_dim,
              "v_proj": cfg.num_key_value_heads * cfg.head_dim}
    dynamic_biases = {}
    for layer in range(cfg.num_hidden_layers):
        for name, width in widths.items():
            dynamic_biases[f"layer{layer}.{name}"] = (
                30, tuple(rng.randint(-(2 ** 40), 2 ** 40) for _ in range(width)))
    return dataclasses.replace(model, dynamic_biases=dynamic_biases)


# ==============================================================================
# Real weights — safetensors -> QuantizedModel
# ==============================================================================


class _SafeTensors:
    """A checkpoint's tensors, read one at a time, in numpy and without rounding.

    The format is read directly rather than through a framework binding, for two reasons that
    are both about this harness's discipline:

    **The widening is exact and has to be visibly so.** Qwen2.5-1.5B is stored bfloat16, which
    numpy has no dtype for. bfloat16 is float32's top 16 bits, so widening it is a left shift
    into a zeroed mantissa — a reinterpretation, not a conversion, and no rounding happens at
    all. Routing it through a framework's cast would put a library conversion at the one seam
    where the float reference's fidelity to the checkpoint is established (§15 / W6).

    **It streams.** The buffer is mapped and one tensor is materialised per call, which is what
    lets the float reference be the unquantized model at 1.5B params (11.5 GiB in float64 if
    held at once).
    """

    _DIRECT = {"F64": np.float64, "F32": np.float32, "F16": np.float16}

    def __init__(self, path):
        self._path = Path(path)
        with open(self._path, "rb") as handle:
            header_length = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_length))
        self._offset = 8 + header_length
        self._header = {k: v for k, v in header.items() if k != "__metadata__"}

    def keys(self):
        return set(self._header)

    def tensor(self, name: str):
        """One tensor, as float64. Raises KeyError if the checkpoint does not carry it."""
        if name not in self._header:
            raise KeyError(name)
        spec = self._header[name]
        start, end = spec["data_offsets"]
        raw = np.memmap(self._path, dtype=np.uint8, mode="r",
                        offset=self._offset + start, shape=(end - start,))
        dtype = spec["dtype"]
        if dtype == "BF16":
            words = raw.view(np.uint16).astype(np.uint32) << np.uint32(16)
            values = words.view(np.float32).astype(np.float64)
        elif dtype in self._DIRECT:
            values = raw.view(self._DIRECT[dtype]).astype(np.float64)
        else:
            raise UnsupportedOpSet(
                f"{self._path}: {name} is stored as {dtype}, which this loader does not widen "
                f"exactly; a lossy read here would be attributed to quantization"
            )
        return values.reshape(tuple(spec["shape"]))


def _upstream_names(cfg: ModelConfig):
    """The total map from a Qwen2.5 checkpoint's tensor names to this pipeline's.

    **Total is the point.** §11's N3 discipline — "an unrecognized ... constant in the config
    is a hard rejection, never a silent drop" — governs the weight map for the same reason it
    governs the config: a quietly-dropped projection is a model that loads, runs, generates
    fluent text, and is not Qwen. So this map is compared against the checkpoint's key set in
    both directions, and either difference is a rejection.
    """
    names = {
        "model.embed_tokens.weight": "embed",
        "model.norm.weight": "final_norm.gain",
    }
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        names[f"model.layers.{layer}.input_layernorm.weight"] = f"{prefix}.attn_norm.gain"
        names[f"model.layers.{layer}.post_attention_layernorm.weight"] = f"{prefix}.mlp_norm.gain"
        for upstream, ours in (("self_attn.q_proj", "q_proj"), ("self_attn.k_proj", "k_proj"),
                               ("self_attn.v_proj", "v_proj"), ("self_attn.o_proj", "o_proj"),
                               ("mlp.gate_proj", "gate_proj"), ("mlp.up_proj", "up_proj"),
                               ("mlp.down_proj", "down_proj")):
            names[f"model.layers.{layer}.{upstream}.weight"] = f"{prefix}.{ours}"
        # Qwen2.5 biases q/k/v and nothing else. Read from the checkpoint, not assumed:
        # an unmapped bias is a rejection, and a mapped-but-absent one is also a rejection.
        for upstream, ours in (("self_attn.q_proj", "q_proj"), ("self_attn.k_proj", "k_proj"),
                               ("self_attn.v_proj", "v_proj")):
            names[f"model.layers.{layer}.{upstream}.bias"] = f"{prefix}.{ours}.bias"
    if not cfg.tie_word_embeddings:
        names["lm_head.weight"] = "lm_head"
    return names


def _rope_pair_permutation(head_dim: int):
    """The row permutation that carries upstream's RoPE layout into §6.4's.

    Upstream rotates the pair `(i, i + head_dim/2)` — `rotate_half`, confirmed by reading the
    installed library rather than by recall. §6.4's tables and `rope.rope_apply_pair` rotate
    the **adjacent** pair `(2i, 2i+1)`, and that convention is pinned: C13 fixes the rotation's
    single rounding on it and the vectorized kernel is proven bit-equal to it across the edge
    corpus. Both conventions carry the same angles, so they differ by a permutation of the
    head dimension and nothing else.

    The permutation is applied to the q and k rows at load. That is exact rather than
    approximate: attention reads q and k only through `q . k` within a head, and a dot product
    is invariant under a permutation applied to both operands, so the scores — and therefore
    everything downstream — are the source model's. v and o are untouched, since v is not
    rotated.

    Reconciling here rather than in the rotation keeps the pinned convention pinned: the
    alternative changes `rope.py`'s and both forward paths' pair layout, which re-opens
    C13 and the bit-equality proof for a fact about a checkpoint's storage order.
    """
    half = head_dim // 2
    order = np.empty(head_dim, dtype=np.int64)
    order[0::2] = np.arange(half)
    order[1::2] = np.arange(half, head_dim)
    return order


def _permute_head_rows(tensor, cfg: ModelConfig, heads: int):
    """Apply the RoPE pair permutation to each head's rows of a q/k projection or bias."""
    order = _rope_pair_permutation(cfg.head_dim)
    values = np.asarray(tensor)
    reshaped = values.reshape(heads, cfg.head_dim, *values.shape[1:])
    return reshaped[:, order].reshape(values.shape)


def _quantize_tensor(name, values, cfg: ModelConfig):
    """One checkpoint tensor as int8 codes plus the scales that define it.

    A §6.1 projection takes per-output-channel scales from **that tensor's own** per-channel
    max-abs. The embedding and the gains take one scale for the tensor, and §6.5 requires it
    of the embedding rather than merely permitting it: the decode is an argmax over the raw
    int32 logits with no requant to fold a scale into, so it is correct only if every vocab
    channel shares one scale — measured, per-channel lm_head scales pick a different token in
    80% of draws. A gain is elementwise, where "per output channel" degenerates to per element.
    """
    if _is_projection(name):
        return quantize_weight_per_channel(values, output_axis=_PROJECTION_OUTPUT_AXIS)
    peak = float(np.abs(values).max(initial=0.0))
    scale = peak / _INT8_MAX if peak > 0.0 else 1.0
    codes = np.clip(_round_half_away_from_zero(values / scale), _INT8_MIN, _INT8_MAX)
    return codes.astype(np.int8), (scale,)


class _CheckpointFloatSource:
    """`float_weight` over the checkpoint, one tensor at a time.

    The originals are already float in the file, so the float reference reads them directly:
    there is nothing to dequantize, which is both the semantically correct thing (§13 item 2's
    reference is the unquantized model) and the only feasible one (1.5B params in float64 is
    11.5 GiB). safetensors keeps the file mapped and materialises one tensor per call.
    """

    def __init__(self, tensors: "_SafeTensors", names, cfg):
        self._tensors = tensors
        self._ours_to_upstream = {ours: upstream for upstream, ours in names.items()}
        self._cfg = cfg

    def __call__(self, name):
        upstream = self._ours_to_upstream.get(name)
        if upstream is None:
            raise KeyError(name)
        return _permuted_if_rope(name, self._tensors.tensor(upstream), self._cfg)


def _permuted_if_rope(name, values, cfg: ModelConfig):
    leaf = name.split(".", 1)[-1]
    if leaf in ("q_proj", "q_proj.bias"):
        return _permute_head_rows(values, cfg, cfg.num_attention_heads)
    if leaf in ("k_proj", "k_proj.bias"):
        return _permute_head_rows(values, cfg, cfg.num_key_value_heads)
    return values


def _checkpoint_tensor_file(checkpoint: Path) -> Path:
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise ConfigError(f"{checkpoint}: no .safetensors weights present")
    if len(files) > 1:
        raise UnsupportedOpSet(
            f"{checkpoint}: {len(files)} weight shards; the loader reads a single-file "
            f"checkpoint and a shard it did not read is a silently absent projection"
        )
    return files[0]


def _checkpoint_tokenize_prompt(checkpoint):
    """The checkpoint's own chat template and tokenizer over the run's prompt messages.

    **The same call `baseline.py` makes**, and that identity is the property: §6.2 calibrates
    on a task-representative corpus, and the tokens the run decodes are what the run is
    representative *of*. Encoding the bare utterance instead set every per-tensor activation
    scale from 5-17 tokens while the run processes 463-475 — ~1-2% of the token content, with
    the ~450-token system prompt invisible to it (record P-17).

    `transformers` is used to tokenize and for nothing else. D-SLM38's rule bars the library's
    **Config object** as a source of numeric constants — `load_config` reads `config.json`
    itself — and says nothing about tokenization, which has no constant to get wrong.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if tokenizer.chat_template is None:
        raise ConfigError(
            f"{checkpoint} carries no chat template; the run decodes the template's prompt "
            f"(baseline.py) and a calibration that cannot build it is not on the run's input"
        )

    def tokenize_prompt(messages):
        # `return_dict=True` and the explicit `input_ids` read are load-bearing at this
        # library version: `apply_chat_template` returns a `BatchEncoding`, so iterating the
        # result yields its KEYS — a plausible two-element "token list" that is not tokens and
        # that every length and equality check downstream would accept. Measured at
        # transformers 5.13.1; `baseline.py` reads `input_ids` off the same call.
        encoded = tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, return_dict=True)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):     # a batched conversation; this one is not
            raise ConfigError(
                f"the chat template returned {len(ids)} sequences for one conversation; the "
                f"run encodes one prompt per record"
            )
        if not all(isinstance(token, int) for token in ids):
            raise ConfigError(
                f"the chat template's input_ids are not integer token ids "
                f"({[type(t).__name__ for t in ids[:3]]}); a scale calibrated on anything "
                f"else is not calibrated on the run's tokens"
            )
        return list(ids)

    return tokenize_prompt


def load_model(checkpoint, extra_tensors=None, require_tensors=(),
               tokenize_prompt=None) -> QuantizedModel:
    """A real trained checkpoint as a `QuantizedModel` (§6.1 / §6.2 / §11).

    `fixture_model` was the only constructor, and its weights are a deterministic bit-mix of
    each tensor's **name string**. That fixture is right for CI — determinism is a property of
    the arithmetic, not of whether the model is any good — and it asks the opposite question
    from the spike: a run over it would emit arithmetic noise in exactly the right format to be
    quoted as T-066's answer.

    Every rejection here is §11's reject-over-degrade. An unmapped tensor, a missing tensor, or
    a shape that contradicts `config.json` fails the load; none of them is a silent skip.

    `extra_tensors` and `require_tensors` exist for the rejection cells: they add names to the
    observed and to the demanded key sets respectively, so both directions of the map's
    totality can be driven without a corrupt checkpoint on disk.

    `tokenize_prompt` defaults to the checkpoint's own chat template and tokenizer — the
    encoder the run decodes through. It is a parameter and not a choice: whichever encoder
    produces the scales is recorded in the artifact (`CalibrationRecord.tokenization`), because
    the same checkpoint and the same hashed corpus under two encoders are two artifacts.
    """
    checkpoint = Path(checkpoint)
    cfg = load_config(checkpoint / "config.json")
    attention_group_size(cfg)
    if cfg.head_dim % 2 != 0:
        raise UnsupportedOpSet(
            f"§6.4's rotation is pairwise; head_dim must be even, got {cfg.head_dim}")

    names = _upstream_names(cfg)
    tensors = _SafeTensors(_checkpoint_tensor_file(checkpoint))

    present = set(tensors.keys())
    present.update(extra_tensors or {})
    demanded = set(names) | set(require_tensors)

    unmapped = sorted(present - set(names))
    if unmapped:
        raise UnsupportedOpSet(
            f"{checkpoint}: {len(unmapped)} checkpoint tensors the map does not name, "
            f"starting with {unmapped[0]!r}. §11 makes this a hard rejection: an "
            f"unrecognized tensor is either a model we do not implement or a mapping bug, "
            f"and a silently-skipped projection is a model that runs and is not the "
            f"source model"
        )
    missing = sorted(demanded - present)
    if missing:
        raise KeyError(
            f"{checkpoint}: {len(missing)} tensors the map names and the checkpoint "
            f"lacks, starting with {missing[0]!r}. Loading fails rather than substituting "
            f"an identity or a zero"
        )

    weights: dict = {}
    weight_scales: dict = {}
    float_biases: dict = {}
    for upstream, ours in names.items():
        values = _permuted_if_rope(ours, tensors.tensor(upstream), cfg)
        if ours.endswith(".bias"):
            float_biases[ours[: -len(".bias")]] = values
            continue
        _check_shape(checkpoint, ours, values, cfg)
        weights[ours], weight_scales[ours] = _quantize_tensor(ours, values, cfg)

    float_weight = _CheckpointFloatSource(tensors, names, cfg)
    if tokenize_prompt is None:
        tokenize_prompt = _checkpoint_tokenize_prompt(checkpoint)
    records = calibration_records()
    maxima = _calibrate(cfg, float_weight, records,
                        lambda record: tokenize_prompt(run_prompt_messages(record)))
    scales, residual_scales, biases = _derive_scales(cfg, maxima, weight_scales, float_biases)
    composition_constants, kv_landing_scales, kv_landing_reciprocals = _derive_composition_constants(
        cfg, weight_scales, scales)
    # C28: when the checkpoint carries projection biases (Qwen2.5 biases q/k/v), the
    # converter emits the dynamic-arm storage too — B[j] at the projection's fold
    # reference S_ref = max_j S_w[j], q_B = 30, half-even (the same emission path
    # `fixture_model_biased` exercises at fixture scale).
    dynamic_biases = {
        name: (30, emit_dynamic_bias(
            np.asarray(values, dtype=np.float64).tolist(), max(weight_scales[name])))
        for name, values in float_biases.items()}
    return QuantizedModel(config=cfg, scales=scales, weights=weights,
                          weight_scales=weight_scales, residual_scales=residual_scales,
                          rope_tables=_build_rope_tables(cfg), biases=biases,
                          float_source=float_weight, tokenize_prompt=tokenize_prompt,
                          calibration=_calibration_record(_CHECKPOINT_TOKENIZATION),
                          gemm_weights={}, composition_constants=composition_constants,
                          kv_landing_scales=kv_landing_scales,
                          kv_landing_reciprocals=kv_landing_reciprocals,
                          dynamic_biases=dynamic_biases)


def _check_shape(checkpoint, name, values, cfg: ModelConfig):
    """§11: the converter "verifies the op set" against `config.json`.

    A tensor whose shape contradicts the config is a different model wearing the config's
    name, and every shape below is derived from the config rather than from the tensor.
    """
    expected = dict(_weight_shapes(cfg))[name]
    if tuple(values.shape) != tuple(expected):
        raise UnsupportedOpSet(
            f"{checkpoint}: {name} is {tuple(values.shape)}; config.json states {expected}"
        )


# ==============================================================================
# Offline calibration and scale derivation
# ==============================================================================


@dataclass(frozen=True)
class CalibrationRecord:
    """The calibration's terms, recorded in the artifact.

    §11's reproducibility formula is "converter version + checkpoint hash + config +
    calibration-corpus hash → identical bytes". With the whole corpus consumed the corpus hash
    is **sufficient** for the record set: it determines which records reached the scales on its
    own, and P-15's subsample rule and count are gone rather than retained and inert.

    `tokenization` is the term the hash cannot supply. The same checkpoint and the same hashed
    corpus under two encoders give two artifacts, so which encoding produced these scales is a
    reproducible-path constant and it is recorded here rather than left in the converter's
    source (record P-17).
    """

    corpus_sha256: str
    tokenization: str
    classes: tuple


def _corpus_sha256() -> str:
    import hashlib
    # The canonical LF form the corpus is published in, whatever the checkout did to line
    # endings — the hash is a term in §11's formula and must not be a property of the
    # platform that read it.
    return hashlib.sha256(
        CALIBRATION_CORPUS_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _calibration_corpus() -> list:
    text = CALIBRATION_CORPUS_PATH.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def calibration_records() -> list:
    """§6.2's calibration set: the **whole** frozen §4 corpus (Dan, 2026-07-15).

    Every class arrives at its full corpus share — 180/120/90/90/60/60, 600 records — because
    nothing selects. That is what dissolved P-15: a subsample needs a rule and a count, §11's
    formula carries a corpus *hash*, and a hash certifies the data while saying nothing about
    a draw. With no draw the hash is sufficient and the formula is true as written.

    An unnamed class is a hard rejection (§11's reject-over-degrade): a corpus carrying a class
    §4 does not name is a corpus this calibration was not written for, and a silent skip would
    calibrate on a subset while every count reads whole.
    """
    records = _calibration_corpus()
    unknown = sorted({record["class"] for record in records} - set(CALIBRATION_CLASSES))
    if unknown:
        raise ConfigError(
            f"the corpus carries classes §4 does not name: {unknown}; an unrecognized class "
            f"is a corpus this calibration was not written for"
        )
    return records


# The prompt the run decodes, in the two halves that compose it. `baseline.py` owns the text —
# it is imported rather than restated, so there is one system prompt and one turn rendering
# and they cannot drift (§6.6). The import is lazy because `baseline` pulls torch and
# transformers at module scope and this module's own path needs neither.


def run_prompt_messages(record) -> list:
    """The chat messages the run decodes for a §4 record — `baseline.py`'s own construction.

    §4's two record shapes (a flat `utterance`; the `correction` class's multi-turn `turns`)
    are handled by `baseline.build_prompt`, which is the point of reading it from there: the
    correction class is one of the two where §6.2's outlier-heavy activations live, and a
    second reader of the corpus is a second chance to drop it.
    """
    from superslm_spike.baseline import SYSTEM_PROMPT, build_prompt

    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(record)}]


def run_token_ids(model: QuantizedModel, record) -> list:
    """The token ids the run's prompt encodes to, under this model's own encoder.

    The prompt, not the utterance. The checkpoint is an **Instruct** model and §15's task is
    schema-constrained intent extraction driven through a system prompt and the chat template,
    so this is 463-475 tokens where the bare record is 5-17.
    """
    return list(model.tokenize_prompt(run_prompt_messages(record)))


def calibration_token_ids(model: QuantizedModel, record) -> list:
    """The token ids calibration observes — **defined as** the tokens the run decodes.

    §6.2 calibrates "on a task-representative corpus", and text the run never sees is not
    task-representative. This is one function rather than two agreeing ones on purpose: a
    scale is a reproducible-path constant, and two constructions that must agree are a
    construction that will eventually not (record P-17).
    """
    return run_token_ids(model, record)


_PROMPT_SHAPE = ("the run's prompt — baseline.SYSTEM_PROMPT as the system message and "
                 "baseline.build_prompt(record) as the user message, with "
                 "add_generation_prompt=True")

_FIXTURE_TOKENIZATION = (
    f"{_PROMPT_SHAPE} — rendered by the §11 fixture's pinned template and encoded byte-wise, "
    f"the fixture having no tokenizer of its own"
)
_CHECKPOINT_TOKENIZATION = (
    f"{_PROMPT_SHAPE} — applied through the checkpoint's own chat template and encoded by the "
    f"checkpoint's own tokenizer"
)

_FIXTURE_TURN = "<|{role}|>\n{content}\n"
_FIXTURE_GENERATION_PROMPT = "<|assistant|>\n"


def _fixture_render(messages) -> str:
    """The §11 fixture's stand-in for a chat template.

    The fixture has no chat template, and §11's framing governs: "determinism is a property of
    the arithmetic, not of whether the model is any good". So the render is pinned rather than
    good — what it has to be is a pure function of the messages, and the same messages the
    checkpoint's own template is handed.
    """
    body = "".join(_FIXTURE_TURN.format(role=m["role"], content=m["content"]) for m in messages)
    return body + _FIXTURE_GENERATION_PROMPT


def _fixture_tokenize_prompt(cfg: ModelConfig):
    """The §11 fixture's encoder: one token per byte of the rendered prompt.

    The fixture's vocabulary is arithmetic, so there is nothing to tokenize *with*, and a
    byte-level encoding is the honest floor of one — it is a pure function of the text, it is
    never a real checkpoint's path, and it does not truncate. Truncation to the model's context
    is the forward's business (`_calibrate`), and it is the run's own slice rather than this
    encoder's opinion.
    """
    def tokenize_prompt(messages):
        data = _fixture_render(messages).encode("utf-8") or b"\x00"
        return [byte % cfg.vocab_size for byte in data]

    return tokenize_prompt


def _calibrate(cfg: ModelConfig, float_weight, records, tokenize) -> dict:
    """Per-site max-abs over the calibration corpus, measured on the float reference.

    Max-abs is order-independent by construction, and that is the property rather than a
    convenience: §11's formula carries a corpus *hash*, not a corpus *order*, so a
    calibration with an order-dependent step (a running mean, an early stop, a top-k over a
    stream) could not be expressed by the terms the artifact records.

    Each sequence is truncated to the model's context cap. This is NOT what the run does with
    the same input: §6.8 C12 makes a position at or past the cap a rejection, so `forward`
    raises `UnsupportedOpSet` on an over-cap sequence where `_calibrate` returns from a silently
    truncated one. Measured 2026-07-15 on the §11 fixture (cap 16): a 1813-token render rejects
    through `forward` and calibrates through this function.

    Calibration therefore degrades where the run rejects, and §11/N3 is reject-over-degrade.
    Whether this function must reject instead of truncate is an open question with the planner
    (Curie P-18/F-1) — it extends N3 to a surface N3 does not name, and it is not decided here.
    The truncation is retained meanwhile because the §11 fixture calibrates all 600 records at
    construction, so a rejection would make the fixture unbuildable and ~40 cells build one
    (Curie P-18/F-2).

    For the §16 nominees the cap never fires: Qwen2.5's cap is 32,768 and the run's prompt is
    460-484 tokens. The divergence is reachable only in the fixture.
    """
    maxima: dict[str, float] = {}
    sequences = [tokens[: cfg.context_cap]
                 for tokens in (list(tokenize(record)) for record in records) if tokens]
    if sequences:
        _float_forward_many(cfg, float_weight, sequences, maxima=maxima, need_logits=False)
    return maxima


def calibrate(model: QuantizedModel, records, tokenize=None) -> StaticScales:
    """The §6.2 activation scales this record set calibrates, on the float reference.

    Offline, and a constant of the artifact once derived (D-SLM5): `forward` reads these and
    never computes one. The default encoding is the run's own, through
    `calibration_token_ids` — §6.2's task-representativeness is a property of the tokens, not
    of the corpus file.
    """
    if tokenize is None:
        def tokenize(record):
            return calibration_token_ids(model, record)
    maxima = _calibrate(model.config, model.float_weight, records, tokenize)
    return _derive_scales(model.config, maxima, model.weight_scales, {})[0]


def _observe(maxima, name, values):
    if maxima is None:
        return
    peak = float(np.abs(np.asarray(values, dtype=np.float64)).max(initial=0.0))
    if peak > maxima.get(name, 0.0):
        maxima[name] = peak


def _output_scale(maxima, name, *input_products):
    """The site's activation scale: the calibrated range over int8, never below its inputs.

    A requant's real multiplier is `input_product / output_scale`, and §6.2's form
    represents multipliers below 1 only — so an output scale that does not dominate its
    inputs is a calibration defect. Dominance is enforced here rather than discovered as
    a negative shift downstream.
    """
    scale = max(maxima.get(name, 0.0) / _INT8_MAX, 1e-12)
    for product in input_products:
        scale = max(scale, product * _SCALE_HEADROOM)
    return scale


def _derive_scales(cfg: ModelConfig, maxima, weight_scales, float_biases):
    """The model's scales, its bias codes, and the residual stream's float-domain scale.

    All come off one walk of the graph: the requant multipliers are ratios of adjacent
    scales, and a second derivation of the same chain would be a second quantization.

    A projection's multiplier is `S_x * S_w[j] / S_out` — the activation scales per tensor
    (§6.2), the weight scale per output channel (§6.1) — so the projection sites carry a
    vector and the weightless sites carry one number.

    A projection's bias is quantized to int32 at the accumulator's own scale `S_x * S_w[j]`,
    which is the only scale it can hold: the accumulator is already stated in those units, so
    the bias adds in exactly and the site's single requant carries the sum. Quantizing it at
    the output scale instead would need a second rounding, and §15 admits only `intmath.py`'s.
    """
    requant: list[RequantSite] = []
    rescale: list[tuple[str, int, int]] = []
    nonlinear: list[tuple[str, float]] = []
    residual_scales: dict[str, float] = {}
    bias_codes: dict[str, np.ndarray] = {}

    def add_rescale(name, factor):
        multiplier, shift = quantize_multiplier(factor)
        rescale.append((name, multiplier, shift))

    def add_requant(weight, input_scale, output_scale):
        channel_scales = weight_scales[weight]
        pairs = [quantize_multiplier(input_scale * scale / output_scale)
                 for scale in channel_scales]
        requant.append(RequantSite(
            name=f"{weight}.requant",
            input_scale=float(input_scale),
            output_scale=float(output_scale),
            weight_scales=channel_scales,
            multipliers=tuple(multiplier for multiplier, _ in pairs),
            shifts=tuple(shift for _, shift in pairs),
        ))
        bias = float_biases.get(weight)
        if bias is not None:
            quantum = float(input_scale) * np.asarray(channel_scales, dtype=np.float64)
            codes = _round_half_away_from_zero(np.asarray(bias, dtype=np.float64) / quantum)
            if np.abs(codes).max(initial=0.0) > INT32_MAX:
                raise ExactnessEnvelopeExceeded(
                    f"{weight}: a bias code does not fit the int32 accumulator it adds into"
                )
            # int32 IS the artifact format's bias dtype (artifact_cache saves int32; the
            # range check above guarantees the fit) — storing int64 here made the
            # in-memory fingerprint diverge from the reloaded artifact's (B-2's root
            # cause: same values, different tobytes width).
            bias_codes[weight] = codes.astype(np.int32)

    def projection_scale(site, weight, input_scale):
        """The site's per-tensor output scale, dominating its widest channel's product."""
        return _output_scale(maxima, site, input_scale * max(weight_scales[weight]))

    def gain_of(name):
        return weight_scales[name][0]

    # The residual stream enters in the embedding's own units, and each norm's requant is
    # driven by its own gain's scale. Both are read from the weights rather than pinned: a
    # real checkpoint's embedding and gains have their own ranges, and a constant here would
    # quantize Qwen against a number derived from the fixture.
    residual_scale = gain_of("embed")
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"

        norm_in = gain_of(f"{prefix}.attn_norm.gain") / (1 << NORM_FRAC_BITS)
        attn_norm_scale = _output_scale(maxima, f"{prefix}.attn_norm.out", norm_in)
        add_rescale(f"{prefix}.attn_norm.requant", norm_in / attn_norm_scale)

        q_scale = projection_scale(f"{prefix}.q", f"{prefix}.q_proj", attn_norm_scale)
        k_scale = projection_scale(f"{prefix}.k", f"{prefix}.k_proj", attn_norm_scale)
        v_scale = projection_scale(f"{prefix}.v", f"{prefix}.v_proj", attn_norm_scale)
        add_requant(f"{prefix}.q_proj", attn_norm_scale, q_scale)
        add_requant(f"{prefix}.k_proj", attn_norm_scale, k_scale)
        add_requant(f"{prefix}.v_proj", attn_norm_scale, v_scale)

        # C27's A-3-pinned per-head KV landing surface: static per-head scales as
        # nonlinear entries, constants of the artifact (D-SLM5's discipline on the
        # interior). Honestly stated: this calibration derives one scale per K/V
        # TENSOR, so every head of a layer carries the same value — the SURFACE is
        # per-head (the pin); the granularity of the fixture's calibration is not.
        for head in range(cfg.num_key_value_heads):
            nonlinear.append((f"{prefix}.k_head{head}.scale", k_scale))
            nonlinear.append((f"{prefix}.v_head{head}.scale", v_scale))

        nonlinear.append((f"{prefix}.softmax.input", q_scale * k_scale / math.sqrt(cfg.head_dim)))

        context_in = v_scale / (1 << PROB_FRAC_BITS)
        context_scale = _output_scale(maxima, f"{prefix}.attn_ctx", context_in)
        add_rescale(f"{prefix}.attn_ctx.requant", context_in / context_scale)

        attn_out_scale = projection_scale(f"{prefix}.attn_out", f"{prefix}.o_proj", context_scale)
        add_requant(f"{prefix}.o_proj", context_scale, attn_out_scale)

        attn_residual_scale = _output_scale(
            maxima, f"{prefix}.attn_residual", residual_scale, attn_out_scale)
        add_rescale(f"{prefix}.attn_residual.hidden", residual_scale / attn_residual_scale)
        add_rescale(f"{prefix}.attn_residual.branch", attn_out_scale / attn_residual_scale)
        residual_scale = attn_residual_scale
        residual_scales[f"{prefix}.attn_residual"] = residual_scale

        mlp_norm_in = gain_of(f"{prefix}.mlp_norm.gain") / (1 << NORM_FRAC_BITS)
        mlp_norm_scale = _output_scale(maxima, f"{prefix}.mlp_norm.out", mlp_norm_in)
        add_rescale(f"{prefix}.mlp_norm.requant", mlp_norm_in / mlp_norm_scale)

        gate_scale = projection_scale(f"{prefix}.gate", f"{prefix}.gate_proj", mlp_norm_scale)
        up_scale = projection_scale(f"{prefix}.up", f"{prefix}.up_proj", mlp_norm_scale)
        add_requant(f"{prefix}.gate_proj", mlp_norm_scale, gate_scale)
        add_requant(f"{prefix}.up_proj", mlp_norm_scale, up_scale)

        nonlinear.append((f"{prefix}.silu.input", gate_scale))

        activation_in = gate_scale * up_scale / (1 << SIGMOID_FRAC_BITS)
        activation_scale = _output_scale(maxima, f"{prefix}.mlp_act", activation_in)
        add_rescale(f"{prefix}.mlp_act.requant", activation_in / activation_scale)

        mlp_out_scale = projection_scale(
            f"{prefix}.mlp_out", f"{prefix}.down_proj", activation_scale)
        add_requant(f"{prefix}.down_proj", activation_scale, mlp_out_scale)

        mlp_residual_scale = _output_scale(
            maxima, f"{prefix}.mlp_residual", residual_scale, mlp_out_scale)
        add_rescale(f"{prefix}.mlp_residual.hidden", residual_scale / mlp_residual_scale)
        add_rescale(f"{prefix}.mlp_residual.branch", mlp_out_scale / mlp_residual_scale)
        residual_scale = mlp_residual_scale
        residual_scales[f"{prefix}.mlp_residual"] = residual_scale

    final_in = gain_of("final_norm.gain") / (1 << NORM_FRAC_BITS)
    final_scale = _output_scale(maxima, "final_norm.out", final_in)
    add_rescale("final_norm.requant", final_in / final_scale)

    scales = StaticScales(
        requant=tuple(sorted(requant, key=lambda site: site.name)),
        rescale=tuple(sorted(rescale)),
        nonlinear=tuple(sorted(nonlinear)),
    )
    return scales, residual_scales, bias_codes


# ==============================================================================
# §6.8 C23-C30: the eval forward's site-level scale composition (D-SLM56/D-SLM57)
# ==============================================================================

# C26: "per-site static factors (including each site's 1/127) are folded into offline
# constants at conversion". `_INV127_SCALE` is the universal (model-independent) part of
# that fold for the sites whose own static factor is otherwise 1 (the residual adds) —
# a fixed offline constant, computed once at import time.
_INV127_SCALE = canonical_scale(Fraction(1, 127))

# C30's i-exp integer coefficients, Q30, floor-emitted from the SAME I-BERT polynomial
# constants `intmath.py` already carries for the static path's float-scale i_exp
# (`_POLY_A/_POLY_B/_POLY_C`). §6.8 states the coefficient VALUES and Q-formats ride
# C7/C8's own S2 pin (owed separately, not decided here) — these are a PROVISIONAL Q30
# emission of the existing informal constants, positive per N2-5, sufficient to drive
# C30's derivation end to end; floor (not round) is the emission, matching the E7/E9
# fixture constants (744261117 / 1452772687 / 1030312935).
_IEXP_QFMT = 30
_IEXP_LN2_Q = math.floor(math.log(2) * (1 << _IEXP_QFMT))
_IEXP_B_Q = math.floor(intmath._POLY_B * (1 << _IEXP_QFMT))
_IEXP_CA_Q = math.floor((intmath._POLY_C / intmath._POLY_A) * (1 << _IEXP_QFMT))


def _reference_fold(channel_scales):
    """C24/C25's per-channel fold to the tensor's own reference `S_ref = max_j S_w[j]`:
    `None` (true pass-through) for every channel already at `S_ref`, an offline
    `(Mw[j], shw[j])` pair for every other channel."""
    s_ref = max(channel_scales)
    folds = [None if s >= s_ref else quantize_multiplier(s / s_ref) for s in channel_scales]
    return folds, s_ref


def _derive_composition_constants(cfg: ModelConfig, weight_scales, scales: StaticScales):
    """The §6.8 C23-C30 offline surface: `composition_constants[site] -> (m, e)` (each
    site's own folded static factor, C26's offline rule, `1/127` included);
    `kv_landing_scales[f"layer{L}.{k|v}_head{h}"] -> (m, e)` (C27's static per-head K/V
    target's real per-code value, D-SLM5's discipline retained on the interior); and
    `kv_landing_reciprocals[f"layer{L}.{k|v}_head{h}"] -> (m_t, e_t, R_t)` (C27 as
    corrected by D-SLM58: the landing composite's OFFLINE C19-class reciprocal, over the
    canonical `S_kh / S_ref` ratio mantissa — never a runtime reciprocal at this site).

    **Honestly stated simplification, not a pin:** the §11 fixture's calibration derives
    one scale per K/V *tensor* (`_derive_scales`'s existing `k_scale`/`v_scale`), not one
    per head — genuine per-head calibration is a `_calibrate` change this pass does not
    make. Every head of a layer's K (respectively V) is therefore landed at the SAME static
    scale here, which makes C27/D-SLM57's per-head attn_ctx pre-fold the all-identity fold
    on this artifact (every head already at `max_head S_v`) — sound, but degenerate; a
    real per-head calibration would exercise the fold's non-identity branch.
    """
    def gain_of(name):
        return weight_scales[name][0]

    # C23: embed is a scale-carrying site — its wide row is the embedding weight's int8
    # codes, so the incoming static factor is the embedding's own scale, folded with the
    # site's 1/127 into one offline constant (C26's rule).
    constants: dict[str, tuple[int, int]] = {
        "embed": canonical_scale(Fraction(gain_of("embed")) / 127)}
    kv_landing: dict[str, tuple[int, int]] = {}
    kv_reciprocals: dict[str, tuple[int, int, int]] = {}

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"

        for norm in ("attn_norm", "mlp_norm"):
            gain_scale = gain_of(f"{prefix}.{norm}.gain") / (1 << NORM_FRAC_BITS)
            constants[f"{prefix}.{norm}"] = canonical_scale(Fraction(gain_scale) / 127)

        for proj in ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            _, s_ref = _reference_fold(weight_scales[f"{prefix}.{proj}"])
            constants[f"{prefix}.{proj}"] = canonical_scale(Fraction(s_ref) / 127)

        k_scale = scales.output_scale(f"{prefix}.k_proj.requant")
        v_scale = scales.output_scale(f"{prefix}.v_proj.requant")
        # C27 as corrected by D-SLM58: the landing composite's reciprocal is OFFLINE, over
        # the canonical STATIC target mantissa (m_t, e_t) = canonical(S_kh / S_ref) — one
        # constant per (head, projection), k and v NOT sharing a target (their S_ref differ).
        # `kv_landing_scales[...]` stays the head's real per-code value canonical(S_kh) (the
        # trace's "m_out"/"e_out"); `kv_landing_reciprocals[...]` carries the ratio mantissa/
        # exponent/reciprocal the D-SLM58 composite actually multiplies by.
        _, k_s_ref = _reference_fold(weight_scales[f"{prefix}.k_proj"])
        _, v_s_ref = _reference_fold(weight_scales[f"{prefix}.v_proj"])
        m_t_k, e_t_k = canonical_scale(Fraction(k_scale) / Fraction(k_s_ref))
        m_t_v, e_t_v = canonical_scale(Fraction(v_scale) / Fraction(v_s_ref))
        r_t_k = intmath.dynamic_scale_reciprocal(m_t_k)
        r_t_v = intmath.dynamic_scale_reciprocal(m_t_v)
        for head in range(cfg.num_key_value_heads):
            kv_landing[f"{prefix}.k_head{head}"] = canonical_scale(Fraction(k_scale))
            kv_landing[f"{prefix}.v_head{head}"] = canonical_scale(Fraction(v_scale))
            kv_reciprocals[f"{prefix}.k_head{head}"] = (m_t_k, e_t_k, r_t_k)
            kv_reciprocals[f"{prefix}.v_head{head}"] = (m_t_v, e_t_v, r_t_v)
            # softmax.input's static half (C27/C30): S_k_head / sqrt(head_dim), one
            # offline canonical constant per kv head; the per-QUERY S_q(i) composes in
            # at runtime, incoming-first (D-SLM57).
            constants[f"{prefix}.softmax_khead{head}"] = canonical_scale(
                Fraction(k_scale) / Fraction(math.sqrt(cfg.head_dim)))

        # attn_ctx (C27/D-SLM57): after the per-head pre-fold to max_head S_v, the row's
        # single wide scale is 2**-PROB_FRAC_BITS * max_head_S_v; C23's chain applies from
        # there, with 1/127 folded into this offline constant like every other site.
        constants[f"{prefix}.attn_ctx"] = canonical_scale(
            Fraction(v_scale) / (1 << PROB_FRAC_BITS) / 127)
        # mlp_act (SwiGLU product, C23 scale-carrying): the sigmoid factor is Q(SIGMOID_
        # FRAC_BITS) with no scale of its own (a [0,1] fraction), so the site's static
        # factor is just the 2**-SIGMOID_FRAC_BITS shift and the folded 1/127.
        constants[f"{prefix}.mlp_act"] = canonical_scale(
            Fraction(1, (1 << SIGMOID_FRAC_BITS) * 127))
        constants[f"{prefix}.attn_residual"] = _INV127_SCALE
        constants[f"{prefix}.mlp_residual"] = _INV127_SCALE

    final_gain_scale = gain_of("final_norm.gain") / (1 << NORM_FRAC_BITS)
    constants["final_norm"] = canonical_scale(Fraction(final_gain_scale) / 127)
    return constants, kv_landing, kv_reciprocals


# ==============================================================================
# The float reference — an independent oracle, not the integer path
# ==============================================================================


def _float_rmsnorm(x, eps):
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(mean_square + eps)


def _float_rope(vectors, theta):
    """The float rotation, from the angles §6.4's tables quantize."""
    steps, heads, head_dim = vectors.shape
    pairs = head_dim // 2
    inv_freq = np.array([theta ** (-2.0 * index / head_dim) for index in range(pairs)])
    positions = np.arange(steps, dtype=np.float64).reshape(steps, 1)
    angles = positions * inv_freq.reshape(1, pairs)
    cos = np.cos(angles).reshape(steps, 1, pairs)
    sin = np.sin(angles).reshape(steps, 1, pairs)
    even = vectors[:, :, 0::2]
    odd = vectors[:, :, 1::2]
    rotated = np.empty_like(vectors)
    rotated[:, :, 0::2] = even * cos - odd * sin
    rotated[:, :, 1::2] = even * sin + odd * cos
    return rotated


def _float_bias(float_weight, name):
    """A §6.1 projection's float bias, or None where the architecture has none.

    Qwen2.5 biases q/k/v and nothing else; the fixture biases nothing. Absence is a fact
    about the model, so it is read from the weight source rather than configured.
    """
    try:
        return float_weight(f"{name}.bias")
    except KeyError:
        return None


def _float_project(float_weight, name, values):
    weight = float_weight(name)
    out = values @ weight.T
    bias = _float_bias(float_weight, name)
    if bias is not None:
        out = out + bias
    return out


def _float_layer(cfg, tensors, hidden, maxima, prefix):
    """One decoder layer in float64, for one sequence, over already-fetched tensors."""
    group = attention_group_size(cfg)
    steps = hidden.shape[0]
    fetch = tensors.__getitem__

    normed = _float_rmsnorm(hidden, cfg.rms_norm_eps) * fetch(f"{prefix}.attn_norm.gain")
    _observe(maxima, f"{prefix}.attn_norm.out", normed)

    q = _float_project(fetch, f"{prefix}.q_proj", normed).reshape(
        steps, cfg.num_attention_heads, cfg.head_dim)
    k = _float_project(fetch, f"{prefix}.k_proj", normed).reshape(
        steps, cfg.num_key_value_heads, cfg.head_dim)
    v = _float_project(fetch, f"{prefix}.v_proj", normed).reshape(
        steps, cfg.num_key_value_heads, cfg.head_dim)
    _observe(maxima, f"{prefix}.q", q)
    _observe(maxima, f"{prefix}.k", k)
    _observe(maxima, f"{prefix}.v", v)

    q = _float_rope(q, cfg.rope_theta)
    k = _float_rope(k, cfg.rope_theta)
    _observe(maxima, f"{prefix}.q", q)
    _observe(maxima, f"{prefix}.k", k)

    context = np.empty((steps, cfg.num_attention_heads, cfg.head_dim), dtype=np.float64)
    for head in range(cfg.num_attention_heads):
        kv_head = head // group
        scores = q[:, head, :] @ k[:, kv_head, :].T / math.sqrt(cfg.head_dim)
        scores = np.where(_causal_mask(steps), scores, -np.inf)
        shifted = scores - scores.max(axis=-1, keepdims=True)
        weights_row = np.exp(shifted)
        weights_row = weights_row / weights_row.sum(axis=-1, keepdims=True)
        context[:, head, :] = weights_row @ v[:, kv_head, :]
    _observe(maxima, f"{prefix}.attn_ctx", context)

    attention = _float_project(
        fetch, f"{prefix}.o_proj",
        context.reshape(steps, cfg.num_attention_heads * cfg.head_dim))
    _observe(maxima, f"{prefix}.attn_out", attention)
    hidden = hidden + attention
    _observe(maxima, f"{prefix}.attn_residual", hidden)

    normed = _float_rmsnorm(hidden, cfg.rms_norm_eps) * fetch(f"{prefix}.mlp_norm.gain")
    _observe(maxima, f"{prefix}.mlp_norm.out", normed)
    gate = _float_project(fetch, f"{prefix}.gate_proj", normed)
    up = _float_project(fetch, f"{prefix}.up_proj", normed)
    _observe(maxima, f"{prefix}.gate", gate)
    _observe(maxima, f"{prefix}.up", up)
    activation = (gate / (1.0 + np.exp(-gate))) * up
    _observe(maxima, f"{prefix}.mlp_act", activation)
    down = _float_project(fetch, f"{prefix}.down_proj", activation)
    _observe(maxima, f"{prefix}.mlp_out", down)
    hidden = hidden + down
    _observe(maxima, f"{prefix}.mlp_residual", hidden)
    return hidden


def _layer_tensors(float_weight, prefix):
    """One layer's tensors, fetched once. A `dict` rather than the source, so the layer body
    cannot re-fetch a weight per sequence."""
    tensors = {}
    for leaf in ("attn_norm.gain", "mlp_norm.gain") + _PROJECTIONS:
        name = f"{prefix}.{leaf}"
        tensors[name] = float_weight(name)
        bias = _float_bias(float_weight, name)
        if bias is not None:
            tensors[f"{name}.bias"] = bias
    return tensors


def _float_forward_many(cfg, float_weight, token_lists, maxima=None, layer_outputs=None,
                        trace=None, need_logits=True):
    """The float64 pipeline over several sequences, **layer-major**.

    Never calls the integer path — a parity gate whose reference is the thing under test
    proves nothing. And never dequantizes the int8 weights: §13 item 2's reference is the
    unquantized model, so a reference rebuilt from the codes would carry the quantizer's own
    error, report zero damage, and fail G-2's upstream bound for a reason that is not the
    architecture.

    **The loop order is a feasibility property, not a style.** Weights are streamed from the
    checkpoint, so a sequence-major loop re-reads all 1.5B of them per record: calibrating the
    600-record corpus that way moves ~7 TB. Layer-major fetches each tensor once for the
    whole calibration set and holds only one layer's weights at a time, which is what keeps
    the float reference the *unquantized* model without materialising it (11.5 GiB at once).

    Per-sequence results are unchanged by the reordering: no reduction crosses sequences, so
    each one's arithmetic is exactly what it would be alone, and `maxima` is a max-abs — which
    is order-independent by construction and is why §11 can carry a corpus hash rather than an
    order.
    """
    for tokens in token_lists:
        _check_positions(cfg, len(tokens))

    embed = float_weight("embed")
    hiddens = [embed[list(tokens), :] for tokens in token_lists]
    if trace is not None:
        trace["embed"] = hiddens[0].copy()
    del embed

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        tensors = _layer_tensors(float_weight, prefix)
        hiddens = [_float_layer(cfg, tensors, hidden, maxima, prefix) for hidden in hiddens]
        del tensors
        if layer_outputs is not None:
            layer_outputs.append(hiddens[0].copy())

    gain = float_weight("final_norm.gain")
    normed = [_float_rmsnorm(hidden, cfg.rms_norm_eps) * gain for hidden in hiddens]
    for row in normed:
        _observe(maxima, "final_norm.out", row)
    if trace is not None:
        trace["final_norm"] = normed[0].copy()
    if not need_logits:
        # Calibration observes no site on the logits, and the head is the model's widest
        # tensor: computing it would be the run's largest matmul and its largest allocation
        # (600 records x 466 positions x 151936 vocab in float64), for a number nothing reads.
        return None
    head = float_weight("embed") if cfg.tie_word_embeddings else float_weight("lm_head")
    return [row @ head.T for row in normed]


def _float_forward(cfg, float_weight, tokens, maxima=None, layer_outputs=None, trace=None):
    """The float64 pipeline's logits for one sequence."""
    return _float_forward_many(cfg, float_weight, [list(tokens)], maxima=maxima,
                               layer_outputs=layer_outputs, trace=trace)[0]


def forward_float_reference(model: QuantizedModel, tokens):
    """The float64 pipeline's logits — the parity gate's independent oracle (§13 item 2),
    and G-2's claim that the float path is the source model."""
    return _float_forward(model.config, model.float_weight, list(tokens))


def forward_layers_float_reference(model: QuantizedModel, tokens):
    """The float64 pipeline's residual stream after each layer."""
    outputs: list = []
    _float_forward(model.config, model.float_weight, list(tokens), layer_outputs=outputs)
    return outputs


def forward_float_layers(model: QuantizedModel, tokens):
    """The float path's hidden states in **upstream's** convention, for the G-2 per-layer cell.

    HF reports `num_hidden_layers + 1` states, and the shape is not the obvious one — it was
    read off the library rather than assumed:

        [embedding output, layer 0 out, ..., layer n-2 out, final_norm(layer n-1 out)]

    so the first entry is pre-layer and the **last entry is post-final-norm**, not the last
    layer's residual. Confirmed against the installed transformers by reconstructing the
    logits from the last state (matches to 4.9e-7) while the pre-norm reading misses by 0.60.

    This is the cell that turns "the model is wrong" into "layer 7 is wrong", which is why it
    asks for states rather than logits alone.
    """
    outputs: list = []
    trace: dict = {}
    _float_forward_many(model.config, model.float_weight, [list(tokens)],
                        layer_outputs=outputs, trace=trace, need_logits=False)
    return [trace["embed"]] + outputs[:-1] + [trace["final_norm"]]


# ==============================================================================
# The integer forward — vectorized
# ==============================================================================


def _causal_mask(steps, start=0, total=None):
    """The causal mask for `steps` tokens at positions `[start, start+steps)` over `total` keys.

    `start` is the cache's contribution and nothing else: a cached step computes one row whose
    query sits at an absolute position, against every key at or before it. With `start=0` and
    `total=steps` this is the uncached mask, identically — which is what makes the cached and
    uncached paths one computation rather than two.
    """
    if total is None:
        total = start + steps
    rows = start + np.arange(steps).reshape(steps, 1)
    columns = np.arange(total).reshape(1, total)
    return columns <= rows


def _check_positions(cfg, steps, start=0):
    if steps < 1:
        raise ValueError("a forward pass is defined on at least one token")
    if start + steps > cfg.context_cap:
        raise UnsupportedOpSet(
            f"{steps} tokens occupy positions {start}..{start + steps - 1}, past context_cap "
            f"{cfg.context_cap}; §6.8 C12 makes a position >= context_cap the documented "
            f"over-cap rejection, never a table read"
        )


class KVCache:
    """The decode's key/value cache — a determinism object, not a performance one.

    `decode_constrained` re-prefilled the whole prompt for every generated token, which
    projected the spike's 600-record run at ~150 hours. That is a real reason to cache and it
    is not a reason to accept different bits: the cache is required to be **bit-identical** to
    the uncached path at every step, because one bit of divergence at step k propagates to
    every token after it.

    The identity holds by construction rather than by tuning. Every row of the forward depends
    only on tokens at or before it — RMSNorm is per row, the projections are exact per-row dot
    products (§3's carrier is order-independent), RoPE is per position, and attention is
    causal — so a key computed at step k is bit-for-bit the key a full forward computes at
    position k. The cache stores those rows; it does not recompute them differently.
    """

    def __init__(self, model: QuantizedModel):
        self._layers = model.config.num_hidden_layers
        self.reset()

    def reset(self) -> None:
        """Clear the cache. Cleared, not rewound: a cache that merely resets its length would
        leave the previous sequence's keys readable and condition the next sequence on a
        prefix that is not in its prompt (§17 dimension 1's poison property)."""
        self._keys: list = [None] * self._layers
        self._values: list = [None] * self._layers

    @property
    def length(self) -> int:
        return 0 if self._keys[0] is None else int(self._keys[0].shape[0])

    def extend(self, layer: int, keys, values):
        """Append this step's keys/values and return the whole history for the layer."""
        if self._keys[layer] is None:
            self._keys[layer] = np.array(keys, dtype=np.int64, copy=True)
            self._values[layer] = np.array(values, dtype=np.int64, copy=True)
        else:
            self._keys[layer] = np.concatenate([self._keys[layer], keys], axis=0)
            self._values[layer] = np.concatenate([self._values[layer], values], axis=0)
        return self._keys[layer], self._values[layer]


def new_kv_cache(model: QuantizedModel) -> KVCache:
    """A cache for this model, holding no sequence."""
    return KVCache(model)


def _clamp_int8(values):
    return np.clip(values, _INT8_MIN, _INT8_MAX)


def _requant(values, reader, weight):
    """A §6.1 projection's requant, per output channel.

    The multiplier and shift arrays broadcast along the last axis because that axis IS the
    output channel: `M[j] = S_x * S_w[j] / S_out` carries `j` from the weight scale.
    """
    multipliers, shifts = reader.requant_for(f"{weight}.requant")
    return vec_multiply_by_quantized_multiplier(
        values,
        np.asarray(multipliers, dtype=np.int64),
        np.asarray(shifts, dtype=np.int64),
    )


def _rescale(values, reader, name):
    """A weightless site's requant: one multiplier for the tensor, no channel index."""
    multiplier, shift = reader.rescale_for(name)
    return vec_multiply_by_quantized_multiplier(values, multiplier, shift)


def _vec_rmsnorm(codes, hidden_size):
    """§6.3's RMSNorm numerator: the i-sqrt digit recurrence over the int64 sum.

    The activation scale cancels out of `x / rms(x)`, so the result is a scale-free
    Q(NORM_FRAC_BITS) value and the gain's scale alone drives the requant.
    """
    sums = (codes.astype(np.int64) ** 2).sum(axis=-1)
    root = vec_i_sqrt((sums << np.int64(2 * NORM_FRAC_BITS)) // np.int64(hidden_size))
    root = np.maximum(root, np.int64(1))
    return (codes.astype(np.int64) << np.int64(2 * NORM_FRAC_BITS)) // root.reshape(-1, 1)


def _vec_softmax(scores, mask, scale):
    """§6.3's softmax: max-subtraction (an integer op), i-exp, fixed-order summation.

    Masked positions are zeroed rather than driven to a large negative logit: i-exp's
    clip makes the far tail total, so a masked position would contribute the clip point's
    probability instead of none.
    """
    guarded = np.where(mask, scores, scores.min())
    shifted = np.where(mask, guarded - guarded.max(axis=-1, keepdims=True), 0)
    exponentials, _ = vec_i_exp(shifted, scale)
    exponentials = np.where(mask, exponentials, np.int64(0))
    _guard_probability_width(exponentials)
    totals = exponentials.sum(axis=-1, keepdims=True)
    return (exponentials << np.int64(PROB_FRAC_BITS)) // np.maximum(totals, np.int64(1))


# The largest i-exp output a softmax row may carry and still be normalizable at
# PROB_FRAC_BITS fractional bits without the numerator leaving int64. Named rather than
# written inline because the dynamic forward's own softmax (dynamic_engine.py) and the
# C++ port both refuse against this same ceiling, and three copies of one number is the
# drift class this tree keeps being bitten by (D-SLM367).
#
# It is one bit stricter than the arithmetic limit: the exact necessary condition is
# `peak << PROB_FRAC_BITS <= INT64_MAX`, i.e. `peak <= 2^48 - 1`. This ceiling is the
# conservative 2^62-based value this guard has always used, and D-SLM367 ratifies it as
# the shipped threshold across all three paths rather than widening it.
PROB_WIDTH_CEILING = (2 ** 62) >> PROB_FRAC_BITS


def _guard_probability_width(exponentials):
    peak = int(np.abs(exponentials).max(initial=0))
    if peak > PROB_WIDTH_CEILING:
        raise ExactnessEnvelopeExceeded(
            f"an i-exp output of {peak} cannot be normalized at {PROB_FRAC_BITS} fractional bits "
            f"without overflowing int64; the softmax input scale is too fine"
        )


def _vec_sigmoid(codes, scale):
    """§6.3's i-exp-based sigmoid, in Q(SIGMOID_FRAC_BITS).

    `intmath.i_exp` is defined on non-positive inputs, so the positive branch is taken
    through `sigmoid(x) = 1 - sigmoid(-x)` rather than through a second construction.
    """
    values = codes.astype(np.int64)
    unit, _ = vec_i_exp(np.zeros(1, dtype=np.int64), scale)
    exponentials, _ = vec_i_exp(-np.abs(values), scale)
    magnitude = (exponentials << np.int64(SIGMOID_FRAC_BITS)) // (unit[0] + exponentials)
    return np.where(values > 0, np.int64(1 << SIGMOID_FRAC_BITS) - magnitude, magnitude)


def _vec_rope(codes, cos_table, sin_table, steps, start=0):
    """§6.4's rotation over positions `[start, start+steps)`.

    The offset is the cache's only claim on RoPE: a cached step's token sits at its absolute
    position, so it reads the same table row a full forward would read for it.
    """
    even = codes[:, :, 0::2]
    odd = codes[:, :, 1::2]
    cos = cos_table[start:start + steps].reshape(steps, 1, -1)
    sin = sin_table[start:start + steps].reshape(steps, 1, -1)
    rotated_even, rotated_odd = vec_rope_apply_pair(even, odd, cos, sin)
    out = np.empty(codes.shape, dtype=np.int64)
    out[:, :, 0::2] = rotated_even
    out[:, :, 1::2] = rotated_odd
    return out


def _gemm_weight(model, name):
    """The projection's int8 codes, widened once and transposed, held for the model's life.

    int8 to float64 is an exact widening, so the value is the weight's and the arithmetic is
    unchanged; what changes is that it happens once instead of once per decoded token. Rebuilt
    per call it is the forward's dominant cost — a decode step reads every weight exactly once,
    so the conversion is pure overhead repeated 1.5B times a token.

    `int_matmul` still checks its envelope on the operand it is handed, and `_max_abs` reads
    the widened array as exactly as it read the codes.
    """
    cached = model.gemm_weights.get(name)
    if cached is None:
        cached = np.ascontiguousarray(model.weights[name].astype(np.float64).T)
        model.gemm_weights[name] = cached
    return cached


def _vec_project(model, name, values):
    """A §6.1 projection's int32 accumulator, bias included.

    The bias is already stated at `S_x * S_w[j]` — the accumulator's own units — so it adds
    in before the requant and the site's single rounding carries the sum.
    """
    accumulator = int_matmul(values, _gemm_weight(model, name))
    bias = model.biases.get(name)
    if bias is not None:
        accumulator = accumulator + bias
    return accumulator


def _vec_forward(model, tokens, reader, layer_outputs=None, cache=None):
    cfg = model.config
    group = attention_group_size(cfg)
    steps = len(tokens)
    # Read once, before any layer extends it: `extend` advances the cache, so a length read
    # inside the loop would give layer 0 one start and layer 1 another.
    start = 0 if cache is None else cache.length
    _check_positions(cfg, steps, start)

    cos_table, sin_table = model.rope_tables

    hidden = model.weights["embed"][list(tokens), :].astype(np.int64)
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        gain = model.weights[f"{prefix}.attn_norm.gain"].astype(np.int64)
        normed = _clamp_int8(_rescale(
            _vec_rmsnorm(hidden, cfg.hidden_size) * gain, reader, f"{prefix}.attn_norm.requant"))

        q = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.q_proj", normed),
            reader, f"{prefix}.q_proj")).reshape(steps, cfg.num_attention_heads, cfg.head_dim)
        k = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.k_proj", normed),
            reader, f"{prefix}.k_proj")).reshape(steps, cfg.num_key_value_heads, cfg.head_dim)
        v = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.v_proj", normed),
            reader, f"{prefix}.v_proj")).reshape(steps, cfg.num_key_value_heads, cfg.head_dim)

        q = _clamp_int8(_vec_rope(q, cos_table, sin_table, steps, start))
        k = _clamp_int8(_vec_rope(k, cos_table, sin_table, steps, start))

        if cache is not None:
            k, v = cache.extend(layer, k, v)

        softmax_scale = reader.scale(f"{prefix}.softmax.input")
        mask = _causal_mask(steps, start, k.shape[0])
        context = np.empty((steps, cfg.num_attention_heads, cfg.head_dim), dtype=np.int64)
        for head in range(cfg.num_attention_heads):
            kv_head = head // group
            scores = int_matmul(q[:, head, :], k[:, kv_head, :].T)
            probabilities = _vec_softmax(scores, mask, softmax_scale)
            context[:, head, :] = int_matmul(probabilities, v[:, kv_head, :])
        context = _clamp_int8(_rescale(context, reader, f"{prefix}.attn_ctx.requant"))

        attention = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.o_proj",
                         context.reshape(steps, cfg.num_attention_heads * cfg.head_dim)),
            reader, f"{prefix}.o_proj"))
        hidden = _clamp_int8(
            _rescale(hidden, reader, f"{prefix}.attn_residual.hidden")
            + _rescale(attention, reader, f"{prefix}.attn_residual.branch"))

        gain = model.weights[f"{prefix}.mlp_norm.gain"].astype(np.int64)
        normed = _clamp_int8(_rescale(
            _vec_rmsnorm(hidden, cfg.hidden_size) * gain, reader, f"{prefix}.mlp_norm.requant"))
        gate = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.gate_proj", normed), reader, f"{prefix}.gate_proj"))
        up = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.up_proj", normed), reader, f"{prefix}.up_proj"))
        sigmoid = _vec_sigmoid(gate, reader.scale(f"{prefix}.silu.input"))
        activation = _clamp_int8(_rescale(gate * sigmoid * up, reader, f"{prefix}.mlp_act.requant"))
        down = _clamp_int8(_requant(
            _vec_project(model, f"{prefix}.down_proj", activation), reader, f"{prefix}.down_proj"))
        hidden = _clamp_int8(
            _rescale(hidden, reader, f"{prefix}.mlp_residual.hidden")
            + _rescale(down, reader, f"{prefix}.mlp_residual.branch"))

        if layer_outputs is not None:
            layer_outputs.append(hidden.copy())

    gain = model.weights["final_norm.gain"].astype(np.int64)
    normed = _clamp_int8(_rescale(
        _vec_rmsnorm(hidden, cfg.hidden_size) * gain, reader, "final_norm.requant"))
    logits = int_matmul(normed, lm_head_weight(model).astype(np.int64).T)
    return _to_int32(logits)


def _to_int32(logits):
    peak = int(np.abs(logits).max(initial=0))
    if peak > INT32_MAX:
        raise ExactnessEnvelopeExceeded(
            f"a logit of magnitude {peak} does not fit the int32 the §6.5 selection reads"
        )
    return logits.astype(np.int32)


def forward(model: QuantizedModel, tokens, cache=None):
    """The pipeline's int32 logits, one row per token (§6.5: no final softmax exists).

    With a `cache`, `tokens` are the tokens **not yet fed** and they take the positions after
    the cache's contents; the logits returned are for those tokens. The result is required to
    be bit-identical to the uncached forward over the whole prefix, and that is the cache's
    entire contract — see `KVCache`.
    """
    return _vec_forward(model, list(tokens), _ScaleReader(model.scales), cache=cache)


def forward_batch(model: QuantizedModel, sequences):
    """Logits for each sequence.

    §8.2's batch invariance is by construction rather than by tuning: per-sequence
    computation is independent and no reduction crosses the batch dimension, so a
    sequence's bits cannot depend on what shares its batch.
    """
    return [forward(model, sequence) for sequence in sequences]


def forward_layers(model: QuantizedModel, tokens):
    """The integer pipeline's residual stream after each layer, in the float reference's
    units — the parity gate compares values, so the codes are dequantized by the static
    scale that defines them."""
    outputs: list = []
    _vec_forward(model, list(tokens), _ScaleReader(model.scales), layer_outputs=outputs)
    return [np.asarray(codes, dtype=np.float64) * model.residual_scales[f"layer{index}.mlp_residual"]
            for index, codes in enumerate(outputs)]


def scales_used_by_forward(model: QuantizedModel, tokens) -> StaticScales:
    """The scales a forward pass actually read.

    They are the model's, identically, for every input: D-SLM5 makes every scale a
    constant in the artifact and the runtime never computes one.
    """
    reader = _ScaleReader(model.scales)
    _vec_forward(model, list(tokens), reader)
    return reader.as_static_scales()


# ==============================================================================
# The integer forward — rung 1's dynamic per-token scale (§6.2 C19-C22, D-SLM48)
# ==============================================================================


def _token_activation_vector(model: QuantizedModel, token: int) -> tuple[int, ...]:
    """A token's own int32 activation vector (§6.2 C20): its embedding row, read as plain
    Python ints rather than through any numpy reduction.

    This is the dynamic-scale chain's entry point and the one place `forward_dynamic_scale`
    touches the model's weights. Reading it as a tuple of Python ints — rather than handing
    `intmath.max_abs_reduce` a live ndarray — is deliberate: the whole point of routing
    through `intmath`'s scalar constructions is that nothing on this path can reach for
    `np.amax`/`np.ndarray.max` by accident (§15/W6, S-1), and a tuple of ints forecloses it
    structurally rather than by discipline.
    """
    row = model.weights["embed"][int(token)]
    return tuple(int(v) for v in row)


def dynamic_scale_trace(model: QuantizedModel, tokens) -> list[dict]:
    """Rung 1's per-token dynamic-scale chain (§6.2 C19-C22), one dict per token actually
    processed by `forward_dynamic_scale` — the "record of what it read" pattern `_ScaleReader`
    already establishes for the static baseline (`scales_used_by_forward`).

    Every step is one of `intmath`'s pinned scalar constructions, in the order C19-C22
    compose them: `max_abs_reduce` (C20) reduces the token's own int32 activation vector to
    `D'`; `normalize_scale` (C21) maps it to `(Dn, s)`; `dynamic_scale_reciprocal` (C19) turns
    `Dn` into the fixed-point reciprocal `R`; `requant_token_code` (C22) — the "127 scale
    wrapper" — turns `(x_i, R, s)` into the token's own int8 code, per element. No
    `torch.amax`/`torch.Tensor.amax`/`np.amax`/`np.ndarray.max`/a float divide appears
    anywhere in this chain (§15/W6, S-1): every quantity above is a Python `int`, and
    `intmath`'s own constructions are pure integer arithmetic.

    Each record's keys are pinned exactly (§1.3, amendment A-2): `"x_int"` (the vector `D'`
    was reduced from), `"Dprime"`, `"Dn"`, `"s"`, `"R"` — plus `"codes"`, the per-element
    output `forward_dynamic_scale` composes into its result, carried here so the two
    functions compute the chain exactly once.
    """
    records: list = []
    for token in tokens:
        x_int = _token_activation_vector(model, token)
        d_prime = intmath.max_abs_reduce(x_int)
        dn, s = intmath.normalize_scale(d_prime)
        r = intmath.dynamic_scale_reciprocal(dn)
        codes = tuple(intmath.requant_token_code(x_i, r, s) for x_i in x_int)
        records.append({
            "x_int": x_int,
            "Dprime": d_prime,
            "Dn": dn,
            "s": s,
            "R": r,
            "codes": codes,
        })
    return records


def forward_dynamic_scale(model: QuantizedModel, tokens, cache=None):
    """The rung-1 analog of `forward()`: each token's own int8 codes, requantized from its
    raw int32 activation vector by a per-token DYNAMIC scale (§6.2 C19-C22) rather than the
    static per-tensor scale `forward()` reads from the model's calibrated `StaticScales`.

    Composes `dynamic_scale_trace`'s per-token records into one int8 array, one row per
    token — the same entry point, the same construction, nothing recomputed twice.

    Scope, stated rather than silently narrowed: this pass wires the dynamic-scale
    COMPUTATION (packet 8 / S-1's target — the failure mode being closed is a per-token scale
    that reaches `torch.amax` + a float divide instead of `intmath`'s pinned chain), not the
    dynamic-scale kernel set through every layer of the transformer. §6.2 itself scopes that
    wider wiring to S2 ("the fallback changes kernel signatures ... S2 builds the kernel set
    with the dynamic-scale signature from the first line"), and it could not be done here
    regardless: the static path's own envelope checks (`int_matmul`'s `_max_abs`, `_to_int32`,
    `_guard_probability_width`) all reduce through `np.ndarray.max`, so routing this function
    through `_vec_forward` would make it trip S-1's own sentinel on code the dynamic-scale
    change never touches. `cache` is accepted for signature parity with `forward()` (§1.3's
    pinned rung-1 contract) and is a no-op: there is no per-layer state here to cache.
    """
    del cache
    records = dynamic_scale_trace(model, tokens)
    return np.array([record["codes"] for record in records], dtype=np.int8)


# ==============================================================================
# forward_dynamic — the §15 full-stack W8A8-dynamic forward (C23-C30, D-SLM55/56/57)
# ==============================================================================


def _dynamic_quantize(wide_row):
    """The C19-C22 chain over an already-materialized wide int row: `(D', Dn, s, R, codes)`."""
    d_prime = intmath.max_abs_reduce(wide_row)
    dn, s = intmath.normalize_scale(d_prime)
    r = intmath.dynamic_scale_reciprocal(dn)
    codes = tuple(intmath.requant_token_code(int(x_i), r, s) for x_i in wide_row)
    return d_prime, dn, s, r, codes


def _chain_record(site, token_index, wide_row, incoming, trace):
    """Run the C19-C22 chain over `wide_row`, compose the carried scale through
    `intmath.carried_scale_product` (C26/D-SLM57: LEFT-ASSOCIATED in composition order —
    the incoming per-token factor(s) first, then this site's own offline static constant,
    then the exact `(Dn, -s)` D'-factor, C21's identity, no runtime rounding), append the
    pinned trace record, and return `(codes, (m_out, e_out))` for the caller to carry
    forward.

    `incoming`: the site's incoming per-token `(m, e)` factors in composition order —
    empty for a site with no incoming factor (embed, the scale-killing norms, attn_ctx
    post-fold), one pair for the chain sites, two (gate, up) at the SwiGLU product.
    """
    d_prime, dn, s, r, codes = _dynamic_quantize(wide_row)
    site_constant = None
    for candidate in (site, site.split(".requant")[0]):
        if candidate in _CURRENT_COMPOSITION_CONSTANTS[0]:
            site_constant = _CURRENT_COMPOSITION_CONSTANTS[0][candidate]
            break
    if site_constant is None:
        raise KeyError(f"forward_dynamic: no composition_constants entry for site {site!r}")
    m_out, e_out = intmath.carried_scale_product(
        [*incoming, site_constant, (dn, -s)])
    if trace is not None:
        trace.append({
            "site": site, "token_index": token_index, "x_int": tuple(int(v) for v in wide_row),
            "Dprime": d_prime, "Dn": dn, "s": s, "R": r, "codes": codes,
            "m_out": m_out, "e_out": e_out,
        })
    return codes, (m_out, e_out)


# A single-slot holder for the model's composition_constants table, read by `_chain_record`
# without threading it through every call in this already deep call chain. Set at the top
# of `forward_dynamic` for the duration of one call; not a module-global mutable default —
# it is scoped to the one active call the way `_ScaleReader` scopes a `forward()` call.
_CURRENT_COMPOSITION_CONSTANTS: list = [{}]


def forward_dynamic(model: QuantizedModel, tokens, cache=None, trace=None):
    """The W8A8-dynamic FULL-STACK integer forward -> int32 logits, one row per token —
    the §15 eval's measured arm (D-SLM48/D-SLM55), realizing §6.8 C23-C30's site-level
    composition over the C19-C22 per-token chain (§6.2). See `dynamic_scale_trace` for the
    entry-point chain this subsumes exactly at the "embed" site, and the plan's C23-C30
    rows (D-SLM56/D-SLM57) for the per-site composition this function implements.

    Scalar throughout (Python ints), matching `forward_scalar_reference`'s discipline
    rather than `_vec_forward`'s numpy one: a per-token dynamic scale has no shared
    per-tensor reduction to vectorize over, and pure Python ints keep every reduction
    off `np.amax`/`torch.amax` structurally (§15/W6, S-1) rather than by discipline.

    K/V per-head landing (C27, corrected by D-SLM58) lands each element by
    `intmath.residual_reconcile(acc'[j], m_a, R_t, e_a, e_t)` — the incoming carried
    mantissa `m_a` (shared by k and v) is the composite's MULTIPLIER, `R_t` the OFFLINE
    C19-class reciprocal of the canonical static target ratio `(m_t, e_t) =
    canonical(S_kh/S_ref)` (one constant per (head, projection), never recomputed at
    runtime — `model.kv_landing_reciprocals`). No runtime reciprocal exists at this site;
    the trace's `m_out`/`e_out` carry the head's real per-code landing value
    (`model.kv_landing_scales`), and `m_in`/`e_in` carry the per-key-token incoming
    mantissa/exponent the composite actually multiplied by (superseding an earlier
    `R_key`-shaped trace field this seat built against the row's original, unrealizable
    parenthetical, before D-SLM58 corrected it).
    """
    cfg = model.config
    group = attention_group_size(cfg)
    tokens = list(tokens)
    steps = len(tokens)
    start = 0 if cache is None else cache.length
    _check_positions(cfg, steps, start)

    _CURRENT_COMPOSITION_CONSTANTS[0] = model.composition_constants

    cos_rows, sin_rows = model.rope_tables
    cos_table = np.asarray(cos_rows).tolist()
    sin_table = np.asarray(sin_rows).tolist()
    embedding = model.weights["embed"].tolist()

    hidden = [[int(v) for v in embedding[token]] for token in tokens]
    hidden_scale = [None] * steps
    for t in range(steps):
        codes, scale = _chain_record("embed", t, hidden[t], [], trace)
        hidden[t] = list(codes)
        hidden_scale[t] = scale

    def projection_weight(name):
        weight = model.weights[name].tolist()
        return [[int(weight[o][i]) for o in range(len(weight))] for i in range(len(weight[0]))]

    def project(name, rows):
        transposed = projection_weight(name)
        return _scalar_matmul(rows, transposed)

    # C28's runtime reciprocal, one per (token, projection-input): memoized on the
    # incoming mantissa so the projections a norm output feeds (q/k/v; gate/up) share
    # one computation per token (A-8 §17.1's cost note), never one per projection.
    _bias_recip_cache: dict[int, int] = {}

    def _recip_a(m_a):
        r_a = _bias_recip_cache.get(m_a)
        if r_a is None:
            r_a = intmath.dynamic_scale_reciprocal(m_a)
            _bias_recip_cache[m_a] = r_a
        return r_a

    def biased_fold(site, folded, in_scale):
        """C28's runtime entry: the stored B[j] reconciled into the FOLDED accumulator
        (post-C25, pre-chain at dynamic projections, pre-landing at k/v) —
        b'[j] = bias_reconcile(B[j], q_B, R_a, e_a), R_a the C19 reciprocal over the
        incoming mantissa (RUNTIME, per token — the denominator is per-token here,
        unlike D-SLM58's landing), one rounding per element, C3 ties."""
        entry = model.dynamic_biases.get(site)
        if entry is None:
            return folded
        q_b, codes = entry
        m_a, e_a = in_scale
        r_a = _recip_a(m_a)
        return [f + intmath.bias_reconcile(int(b), q_b, r_a, e_a)
                for f, b in zip(folded, codes)]

    def rotate(vectors, positions):
        rotated = []
        for row, position in zip(vectors, positions):
            out_row = []
            for pair in range(len(row) // 2):
                x, y = rope.rope_apply_pair(
                    int(row[2 * pair]), int(row[2 * pair + 1]),
                    cos_table[position][pair], sin_table[position][pair])
                out_row.extend([x, y])
            rotated.append(out_row)
        return rotated

    positions = list(range(start, start + steps))

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"

        # --- attn_norm (C23 scale-killing) ---
        gain = [int(v) for v in model.weights[f"{prefix}.attn_norm.gain"].tolist()]
        normed_codes = []
        norm_scale = [None] * steps
        for t in range(steps):
            row = hidden[t]
            total = sum(v * v for v in row)
            root = max(intmath.i_sqrt((total << (2 * NORM_FRAC_BITS)) // cfg.hidden_size), 1)
            wide = [((row[i] << (2 * NORM_FRAC_BITS)) // root) * gain[i]
                    for i in range(cfg.hidden_size)]
            codes, scale = _chain_record(f"{prefix}.attn_norm", t, wide, [], trace)
            normed_codes.append([max(-127, min(127, c)) for c in codes])
            norm_scale[t] = scale

        # --- q/k/v projections ---
        q_raw = project(f"{prefix}.q_proj", normed_codes)
        k_raw = project(f"{prefix}.k_proj", normed_codes)
        v_raw = project(f"{prefix}.v_proj", normed_codes)

        q_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.q_proj"])
        q_codes_rows = []
        q_scale = [None] * steps
        for t in range(steps):
            folded = biased_fold(f"{prefix}.q_proj",
                                 fold_projection_accumulator(q_raw[t], q_folds),
                                 norm_scale[t])
            codes, scale = _chain_record(
                f"{prefix}.q_proj.requant", t, folded, [norm_scale[t]], trace)
            q_codes_rows.append([max(-127, min(127, c)) for c in codes])
            q_scale[t] = scale

        head_dim = cfg.head_dim
        q_heads = []
        for h in range(cfg.num_attention_heads):
            head_rows = [q_codes_rows[t][h * head_dim:(h + 1) * head_dim] for t in range(steps)]
            q_heads.append([[max(-127, min(127, c)) for c in row]
                            for row in rotate(head_rows, positions)])

        k_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.k_proj"])
        v_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.v_proj"])
        num_kv_heads = cfg.num_key_value_heads
        k_heads_codes = [[] for _ in range(num_kv_heads)]
        v_heads_codes = [[] for _ in range(num_kv_heads)]
        for t in range(steps):
            # C27 as corrected by D-SLM58: NO runtime reciprocal at this site. The
            # incoming carried mantissa m_a (shared by k and v) is the composite's
            # MULTIPLIER; the reciprocal is the OFFLINE R_t over the canonical static
            # target ratio (m_t, e_t) = canonical(S_kh/S_ref), one constant per
            # (head, projection) from `model.kv_landing_reciprocals`.
            m_a, e_a = norm_scale[t]
            k_folded = biased_fold(f"{prefix}.k_proj",
                                   fold_projection_accumulator(k_raw[t], k_folds),
                                   norm_scale[t])
            v_folded = biased_fold(f"{prefix}.v_proj",
                                   fold_projection_accumulator(v_raw[t], v_folds),
                                   norm_scale[t])
            for head in range(num_kv_heads):
                m_target, e_target = model.kv_landing_scales[f"{prefix}.k_head{head}"]
                m_t, e_t, r_t = model.kv_landing_reciprocals[f"{prefix}.k_head{head}"]
                seg = k_folded[head * head_dim:(head + 1) * head_dim]
                landed = [max(-127, min(127, intmath.residual_reconcile(
                    int(acc_j), m_a, r_t, e_a, e_t))) for acc_j in seg]
                k_heads_codes[head].append(landed)
                if trace is not None:
                    trace.append({
                        "site": f"{prefix}.k_proj.requant", "token_index": t, "head": head,
                        "x_int": tuple(seg), "m_in": m_a, "e_in": e_a, "codes": tuple(landed),
                        "m_out": m_target, "e_out": e_target,
                    })
                m_target_v, e_target_v = model.kv_landing_scales[f"{prefix}.v_head{head}"]
                m_t_v, e_t_v, r_t_v = model.kv_landing_reciprocals[f"{prefix}.v_head{head}"]
                seg_v = v_folded[head * head_dim:(head + 1) * head_dim]
                landed_v = [max(-127, min(127, intmath.residual_reconcile(
                    int(acc_j), m_a, r_t_v, e_a, e_t_v))) for acc_j in seg_v]
                v_heads_codes[head].append(landed_v)
                if trace is not None:
                    trace.append({
                        "site": f"{prefix}.v_proj.requant", "token_index": t, "head": head,
                        "x_int": tuple(seg_v), "m_in": m_a, "e_in": e_a, "codes": tuple(landed_v),
                        "m_out": m_target_v, "e_out": e_target_v,
                    })

        k_heads = [rotate(k_heads_codes[h], positions) for h in range(num_kv_heads)]
        k_heads = [[[max(-127, min(127, c)) for c in row] for row in head_rows]
                   for head_rows in k_heads]
        v_heads = v_heads_codes

        if cache is not None:
            k_arr = np.asarray(k_heads, dtype=np.int64).transpose(1, 0, 2)
            v_arr = np.asarray(v_heads, dtype=np.int64).transpose(1, 0, 2)
            k_full, v_full = cache.extend(layer, k_arr, v_arr)
            k_heads = [k_full[:, h, :].tolist() for h in range(num_kv_heads)]
            v_heads = [v_full[:, h, :].tolist() for h in range(num_kv_heads)]

        # softmax.input is per-query (C27: S_q(i) * S_k_head / sqrt(d), derived per C30).
        # The static half, canonical(S_k_head / sqrt(head_dim)), is an OFFLINE constant
        # per kv head (`composition_constants[f"{prefix}.softmax_khead{h}"]`); the
        # per-query S_q(i) composes in at runtime, incoming-first, left-associated
        # (C26/D-SLM57), through the pinned module-global product.
        context_rows = [[0] * (cfg.num_attention_heads * head_dim) for _ in range(steps)]
        for head in range(cfg.num_attention_heads):
            kv_head = head // group
            keys = k_heads[kv_head]
            values = v_heads[kv_head]
            softmax_static = model.composition_constants[f"{prefix}.softmax_khead{kv_head}"]
            transposed_keys = [[keys[j][d] for j in range(len(keys))] for d in range(head_dim)]
            scores = _scalar_matmul(q_heads[head], transposed_keys)
            for t in range(steps):
                sm_m, sm_e = intmath.carried_scale_product([q_scale[t], softmax_static])
                q_ln2, q_b, q_c = intmath.iexp_scale_constants(
                    sm_m, sm_e, _IEXP_LN2_Q, _IEXP_QFMT, _IEXP_B_Q, _IEXP_QFMT,
                    _IEXP_CA_Q, _IEXP_QFMT)
                width = t + start + 1
                present = scores[t][:width]
                shifted = intmath.shift_by_max(present)
                exponentials = [intmath.i_exp_from_constants(int(q), q_ln2, q_b, q_c)
                                for q in shifted]
                total = sum(exponentials)
                probabilities = [(e << PROB_FRAC_BITS) // max(total, 1) for e in exponentials]
                for d in range(head_dim):
                    acc = 0
                    for j in range(width):
                        acc += probabilities[j] * int(values[j][d])
                    context_rows[t][head * head_dim + d] = acc

        # attn_ctx per-head pre-fold (C27/D-SLM57): each head SEGMENT folds to the
        # common reference max_head S_v by an offline (0,1] constant through the pinned
        # C1 composite; the identity (max) head is a true pass-through (C24). The fold
        # constants are emitted offline from the artifact's raw per-head V scales (the
        # A-3-pinned nonlinear surface), exactly the C25 emission.
        s_v = [model.scales.scale(f"{prefix}.v_head{h}.scale") for h in range(num_kv_heads)]
        s_v_max = max(s_v)
        ctx_folds = []
        for h in range(cfg.num_attention_heads):
            f_v = s_v[h // group]
            entry = None if f_v == s_v_max else quantize_multiplier(f_v / s_v_max)
            ctx_folds.extend([entry] * head_dim)

        attn_out = []
        attn_scale = [None] * steps
        for t in range(steps):
            folded_ctx = fold_projection_accumulator(context_rows[t], ctx_folds)
            codes, scale = _chain_record(f"{prefix}.attn_ctx", t, folded_ctx, [], trace)
            attn_out.append([max(-127, min(127, c)) for c in codes])
            attn_scale[t] = scale

        o_raw = project(f"{prefix}.o_proj", attn_out)
        o_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.o_proj"])
        o_codes = []
        o_scale = [None] * steps
        for t in range(steps):
            folded = biased_fold(f"{prefix}.o_proj",
                                 fold_projection_accumulator(o_raw[t], o_folds),
                                 attn_scale[t])
            codes, scale = _chain_record(
                f"{prefix}.o_proj.requant", t, folded, [attn_scale[t]], trace)
            o_codes.append([max(-127, min(127, c)) for c in codes])
            o_scale[t] = scale

        # --- attn residual (C26) ---
        new_hidden = []
        new_hidden_scale = [None] * steps
        for t in range(steps):
            m_h, e_h = hidden_scale[t]
            m_b, e_b = o_scale[t]
            r_h = intmath.dynamic_scale_reciprocal(m_h)
            wide_sum = [hidden[t][i] + intmath.residual_reconcile(
                o_codes[t][i], m_b, r_h, e_b, e_h) for i in range(cfg.hidden_size)]
            codes, scale = _chain_record(f"{prefix}.attn_residual", t, wide_sum, [(m_h, e_h)], trace)
            new_hidden.append([max(-127, min(127, c)) for c in codes])
            new_hidden_scale[t] = scale
        hidden, hidden_scale = new_hidden, new_hidden_scale

        # --- mlp_norm (C23 scale-killing) ---
        gain = [int(v) for v in model.weights[f"{prefix}.mlp_norm.gain"].tolist()]
        mlp_normed = []
        mlp_norm_scale = [None] * steps
        for t in range(steps):
            row = hidden[t]
            total = sum(v * v for v in row)
            root = max(intmath.i_sqrt((total << (2 * NORM_FRAC_BITS)) // cfg.hidden_size), 1)
            wide = [((row[i] << (2 * NORM_FRAC_BITS)) // root) * gain[i]
                    for i in range(cfg.hidden_size)]
            codes, scale = _chain_record(f"{prefix}.mlp_norm", t, wide, [], trace)
            mlp_normed.append([max(-127, min(127, c)) for c in codes])
            mlp_norm_scale[t] = scale

        gate_raw = project(f"{prefix}.gate_proj", mlp_normed)
        up_raw = project(f"{prefix}.up_proj", mlp_normed)
        gate_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.gate_proj"])
        up_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.up_proj"])

        gate_codes, gate_scale = [], [None] * steps
        up_codes, up_scale = [], [None] * steps
        for t in range(steps):
            gf = biased_fold(f"{prefix}.gate_proj",
                             fold_projection_accumulator(gate_raw[t], gate_folds),
                             mlp_norm_scale[t])
            codes, scale = _chain_record(f"{prefix}.gate_proj.requant", t, gf,
                                         [mlp_norm_scale[t]], trace)
            gate_codes.append([max(-127, min(127, c)) for c in codes])
            gate_scale[t] = scale
            uf = biased_fold(f"{prefix}.up_proj",
                             fold_projection_accumulator(up_raw[t], up_folds),
                             mlp_norm_scale[t])
            codes, scale = _chain_record(f"{prefix}.up_proj.requant", t, uf,
                                         [mlp_norm_scale[t]], trace)
            up_codes.append([max(-127, min(127, c)) for c in codes])
            up_scale[t] = scale

        activation = []
        act_scale = [None] * steps
        for t in range(steps):
            m_g, e_g = gate_scale[t]
            wide = []
            for i in range(cfg.intermediate_size):
                value = gate_codes[t][i]
                sigmoid = silu_lut.silu_sigmoid_q15(value, m_g, e_g)
                wide.append(gate_codes[t][i] * sigmoid * up_codes[t][i])
            codes, scale = _chain_record(f"{prefix}.mlp_act", t, wide, [gate_scale[t], up_scale[t]], trace)
            activation.append([max(-127, min(127, c)) for c in codes])
            act_scale[t] = scale

        down_raw = project(f"{prefix}.down_proj", activation)
        down_folds, _ = _reference_fold(model.weight_scales[f"{prefix}.down_proj"])
        down_codes = []
        down_scale = [None] * steps
        for t in range(steps):
            folded = biased_fold(f"{prefix}.down_proj",
                                 fold_projection_accumulator(down_raw[t], down_folds),
                                 act_scale[t])
            codes, scale = _chain_record(
                f"{prefix}.down_proj.requant", t, folded, [act_scale[t]], trace)
            down_codes.append([max(-127, min(127, c)) for c in codes])
            down_scale[t] = scale

        # --- mlp residual (C26) ---
        new_hidden = []
        new_hidden_scale = [None] * steps
        for t in range(steps):
            m_h, e_h = hidden_scale[t]
            m_b, e_b = down_scale[t]
            r_h = intmath.dynamic_scale_reciprocal(m_h)
            wide_sum = [hidden[t][i] + intmath.residual_reconcile(
                down_codes[t][i], m_b, r_h, e_b, e_h) for i in range(cfg.hidden_size)]
            codes, scale = _chain_record(f"{prefix}.mlp_residual", t, wide_sum, [(m_h, e_h)], trace)
            new_hidden.append([max(-127, min(127, c)) for c in codes])
            new_hidden_scale[t] = scale
        hidden, hidden_scale = new_hidden, new_hidden_scale

    gain = [int(v) for v in model.weights["final_norm.gain"].tolist()]
    final_codes = []
    for t in range(steps):
        row = hidden[t]
        total = sum(v * v for v in row)
        root = max(intmath.i_sqrt((total << (2 * NORM_FRAC_BITS)) // cfg.hidden_size), 1)
        wide = [((row[i] << (2 * NORM_FRAC_BITS)) // root) * gain[i]
                for i in range(cfg.hidden_size)]
        codes, _ = _chain_record("final_norm", t, wide, [], trace)
        final_codes.append([max(-127, min(127, c)) for c in codes])

    head_weight = lm_head_weight(model).tolist()
    transposed = [[int(head_weight[o][i]) for o in range(len(head_weight))]
                  for i in range(len(head_weight[0]))]
    logits = np.asarray(_scalar_matmul(final_codes, transposed), dtype=np.int64)
    return _to_int32(logits)


# ==============================================================================
# The integer forward — the normative scalar reference
# ==============================================================================


def _scalar_requant(values, reader, weight):
    """The scalar form of a §6.1 projection's per-output-channel requant."""
    multipliers, shifts = reader.requant_for(f"{weight}.requant")
    return [[intmath.multiply_by_quantized_multiplier(int(v), multipliers[j], shifts[j])
             for j, v in enumerate(row)] for row in values]


def _scalar_rescale(values, reader, name):
    multiplier, shift = reader.rescale_for(name)
    return [[intmath.multiply_by_quantized_multiplier(int(v), multiplier, shift) for v in row]
            for row in values]


def _scalar_clamp(values):
    return [[min(max(int(v), _INT8_MIN), _INT8_MAX) for v in row] for row in values]


def _scalar_rmsnorm(codes, hidden_size):
    out = []
    for row in codes:
        total = sum(int(v) * int(v) for v in row)
        root = max(intmath.i_sqrt((total << (2 * NORM_FRAC_BITS)) // hidden_size), 1)
        out.append([(int(v) << (2 * NORM_FRAC_BITS)) // root for v in row])
    return out


def _scalar_softmax_row(scores, width, scale):
    present = [scores[j] for j in range(width)]
    shifted = intmath.shift_by_max(present)
    exponentials = [intmath.i_exp(int(q), scale)[0] for q in shifted]
    total = sum(exponentials)
    return [(e << PROB_FRAC_BITS) // max(total, 1) for e in exponentials]


def _scalar_sigmoid(value, scale, unit):
    exponential = intmath.i_exp(-abs(int(value)), scale)[0]
    magnitude = (exponential << SIGMOID_FRAC_BITS) // (unit + exponential)
    return (1 << SIGMOID_FRAC_BITS) - magnitude if value > 0 else magnitude


def _scalar_forward(model, tokens, reader):
    cfg = model.config
    group = attention_group_size(cfg)
    steps = len(tokens)
    _check_positions(cfg, steps)

    # The model's own hoisted tables, back to Python ints: the scalar path is normative and
    # `intmath.py`'s primitives are defined on exact Python integers, not on a fixed-width
    # array element that could wrap where an int would not.
    cos_rows, sin_rows = model.rope_tables
    cos_table = np.asarray(cos_rows).tolist()
    sin_table = np.asarray(sin_rows).tolist()
    embedding = model.weights["embed"].tolist()
    hidden = [[int(v) for v in embedding[token]] for token in tokens]

    def projection(name, values):
        weight = model.weights[name].tolist()
        transposed = [[int(weight[o][i]) for o in range(len(weight))] for i in range(len(weight[0]))]
        out = _scalar_matmul(values, transposed)
        bias = model.biases.get(name)
        if bias is not None:
            offsets = [int(b) for b in np.asarray(bias).tolist()]
            out = [[v + offsets[j] for j, v in enumerate(row)] for row in out]
        return out

    def rotate(vectors):
        rotated = []
        for position, row in enumerate(vectors):
            out_row = []
            for pair in range(len(row) // 2):
                x, y = rope.rope_apply_pair(
                    int(row[2 * pair]), int(row[2 * pair + 1]),
                    cos_table[position][pair], sin_table[position][pair])
                out_row.extend([x, y])
            rotated.append(out_row)
        return rotated

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        gain = [int(v) for v in model.weights[f"{prefix}.attn_norm.gain"].tolist()]
        normalized = _scalar_rmsnorm(hidden, cfg.hidden_size)
        scaled = [[normalized[t][i] * gain[i] for i in range(cfg.hidden_size)] for t in range(steps)]
        normed = _scalar_clamp(_scalar_rescale(scaled, reader, f"{prefix}.attn_norm.requant"))

        q = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.q_proj", normed), reader, f"{prefix}.q_proj"))
        k = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.k_proj", normed), reader, f"{prefix}.k_proj"))
        v = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.v_proj", normed), reader, f"{prefix}.v_proj"))

        head_dim = cfg.head_dim

        def split(values, head):
            return [row[head * head_dim:(head + 1) * head_dim] for row in values]

        q_heads = [_scalar_clamp(rotate(split(q, h))) for h in range(cfg.num_attention_heads)]
        k_heads = [_scalar_clamp(rotate(split(k, h))) for h in range(cfg.num_key_value_heads)]
        v_heads = [split(v, h) for h in range(cfg.num_key_value_heads)]

        softmax_scale = reader.scale(f"{prefix}.softmax.input")
        context = [[0] * (cfg.num_attention_heads * head_dim) for _ in range(steps)]
        for head in range(cfg.num_attention_heads):
            kv_head = head // group
            keys = k_heads[kv_head]
            transposed = [[keys[j][d] for j in range(steps)] for d in range(head_dim)]
            scores = _scalar_matmul(q_heads[head], transposed)
            for position in range(steps):
                width = position + 1
                probabilities = _scalar_softmax_row(scores[position], width, softmax_scale)
                for d in range(head_dim):
                    total = 0
                    for j in range(width):
                        total += probabilities[j] * int(v_heads[kv_head][j][d])
                    context[position][head * head_dim + d] = total
        context = _scalar_clamp(_scalar_rescale(context, reader, f"{prefix}.attn_ctx.requant"))

        attention = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.o_proj", context), reader, f"{prefix}.o_proj"))
        carried = _scalar_rescale(hidden, reader, f"{prefix}.attn_residual.hidden")
        branch = _scalar_rescale(attention, reader, f"{prefix}.attn_residual.branch")
        hidden = _scalar_clamp([[carried[t][i] + branch[t][i] for i in range(cfg.hidden_size)]
                                for t in range(steps)])

        gain = [int(v) for v in model.weights[f"{prefix}.mlp_norm.gain"].tolist()]
        normalized = _scalar_rmsnorm(hidden, cfg.hidden_size)
        scaled = [[normalized[t][i] * gain[i] for i in range(cfg.hidden_size)] for t in range(steps)]
        normed = _scalar_clamp(_scalar_rescale(scaled, reader, f"{prefix}.mlp_norm.requant"))
        gate = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.gate_proj", normed), reader, f"{prefix}.gate_proj"))
        up = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.up_proj", normed), reader, f"{prefix}.up_proj"))
        silu_scale = reader.scale(f"{prefix}.silu.input")
        unit = intmath.i_exp(0, silu_scale)[0]
        activation = [[gate[t][i] * _scalar_sigmoid(gate[t][i], silu_scale, unit) * up[t][i]
                       for i in range(cfg.intermediate_size)] for t in range(steps)]
        activation = _scalar_clamp(_scalar_rescale(activation, reader, f"{prefix}.mlp_act.requant"))
        down = _scalar_clamp(_scalar_requant(
            projection(f"{prefix}.down_proj", activation), reader, f"{prefix}.down_proj"))
        carried = _scalar_rescale(hidden, reader, f"{prefix}.mlp_residual.hidden")
        branch = _scalar_rescale(down, reader, f"{prefix}.mlp_residual.branch")
        hidden = _scalar_clamp([[carried[t][i] + branch[t][i] for i in range(cfg.hidden_size)]
                                for t in range(steps)])

    gain = [int(v) for v in model.weights["final_norm.gain"].tolist()]
    normalized = _scalar_rmsnorm(hidden, cfg.hidden_size)
    scaled = [[normalized[t][i] * gain[i] for i in range(cfg.hidden_size)] for t in range(steps)]
    normed = _scalar_clamp(_scalar_rescale(scaled, reader, "final_norm.requant"))
    head_weight = lm_head_weight(model).tolist()
    transposed = [[int(head_weight[o][i]) for o in range(len(head_weight))]
                  for i in range(len(head_weight[0]))]
    return _to_int32(np.asarray(_scalar_matmul(normed, transposed), dtype=np.int64))


def forward_scalar_reference(model: QuantizedModel, tokens):
    """The normative scalar pipeline: pure Python ints over `intmath.py`'s primitives.

    §6.2 makes the scalar reference normative and holds every specialization to it. This
    exists so the vectorized path has something to be bit-equal *to*, and the §11 fixture
    is small enough that both can run in one test.
    """
    return _scalar_forward(model, list(tokens), _ScaleReader(model.scales))


# ==============================================================================
# Selection and decode
# ==============================================================================


def select_greedy(logits) -> int:
    """§6.8 C16: argmax over int32 logits, lowest token index wins ties.

    Pinned rather than inherited: a reordered reduction or a torch path can disagree with
    numpy's incidental first-max behaviour, and nothing else would see it.
    """
    values = np.asarray(logits)
    best = values.max()
    return int(np.flatnonzero(values == best)[0])


def select_greedy_masked(logits, mask) -> int:
    """The §9 masked greedy step, through `constrain.argmax_masked`.

    One masker, so one tie-break: a second implementation of the same rule is a second
    contract.
    """
    return argmax_masked([int(v) for v in np.asarray(logits).ravel()], list(mask))


def _logits_fn(model, prompt_ids, use_cache=True):
    """The decode's `logits_fn`: the last row's int32 logits for `prompt + produced`.

    The cached form feeds only the tokens the cache has not seen. It is bit-identical to the
    uncached form by the cache's own contract, so `use_cache` selects the cost and never the
    answer. The fed prefix is tracked and verified rather than assumed: the decode loop
    happens to extend by one token at a time, and a cache silently reused across a prefix that
    changed underneath it is exactly the class of defect the cache cells exist to catch.
    """
    if not use_cache:
        def uncached(produced):
            return [int(v) for v in np.asarray(
                forward(model, list(prompt_ids) + list(produced))[-1])]
        return uncached

    cache = new_kv_cache(model)
    fed: list[int] = []

    def cached(produced):
        nonlocal fed
        full = list(prompt_ids) + list(produced)
        if full[:len(fed)] != fed:
            cache.reset()
            fed = []
        logits = forward(model, full[len(fed):], cache=cache)
        fed = full
        return [int(v) for v in np.asarray(logits[-1])]

    return cached


def decode_greedy(model: QuantizedModel, prompt_ids, max_new_tokens: int,
                  use_cache: bool = True) -> list[int]:
    """Unconstrained greedy decode — §6.5's v1 selection, run to `max_new_tokens`."""
    logits = _logits_fn(model, prompt_ids, use_cache)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        produced.append(select_greedy(logits(produced)))
    return produced


def decode_greedy_float(model: QuantizedModel, prompt_ids, max_new_tokens: int) -> list[int]:
    """The **float reference's** greedy decode — G-2's token-agreement oracle (§13 item 2a).

    §13 2a asks that the pipeline's greedy tokens agree with the upstream model's own
    framework, which is a claim about the *float* path: it is what proves the reimplementation
    is the source model, before the integer path is allowed to attribute anything to
    quantization. `decode_greedy` cannot answer it — §6.5 pins that one to an argmax over the
    integer path's raw int32 logits, and asserting the W8A8 decode reproduces an fp32 decode
    exactly would be asserting quantization does nothing.

    It is the secondary oracle. Measured: a wrong θ moves the logits 13% and flips no argmax
    on a short probe, so token agreement is an insensitive instrument and
    `forward_float_reference`'s logits carry the primary claim.
    """
    produced: list[int] = []
    for _ in range(max_new_tokens):
        logits = forward_float_reference(model, list(prompt_ids) + produced)
        produced.append(select_greedy(np.asarray(logits)[-1]))
    return produced


def decode_constrained(model: QuantizedModel, prompt_ids, dfa, max_new_tokens: int,
                       use_cache: bool = True) -> list[int]:
    """The §15 schema-constrained decode, through `constrain.greedy_decode`.

    The pipeline's job at this seam is to be a `logits_fn`, not a decoder: two decode
    loops is two stopping rules and two tie-breaks.
    """
    return greedy_decode(dfa, _logits_fn(model, prompt_ids, use_cache),
                         max_tokens=max_new_tokens)


def _logits_fn_dynamic(model, prompt_ids, use_cache=True, *, primed_cache=None, primed_fed=None):
    """`_logits_fn`'s twin over `forward_dynamic` — same cached/uncached shapes, same
    fed-prefix verification, only the forward differs (A-1 §10.1's decode seam).

    The forward is resolved through the module-global name at call time (pin P-1, the
    rung-1 clz64 pin class): the eval driver's arm sentinels intercept by patching
    `pipeline.forward_dynamic`, and an early-bound reference taken here would bypass the
    patch and let a mis-wired arm false-negative through the sentinel.

    `primed_cache`/`primed_fed` (C9 Unit 8a, T-522): the cached closure normally starts from
    an empty `KVCache` and an empty `fed` prefix, so the first call always forwards the whole
    prompt. A caller that has already primed a cache with a prefix shared across many prompts
    (a per-caller `deepcopy` of one cache built once from that prefix) passes it here along
    with the prefix it was built from; the closure seeds `fed` to that prefix so its first call
    forwards only the divergent tail beyond it, rather than resetting the cache and redoing the
    prefix's own forward. Behaviour is unchanged from before this parameter existed when both
    are left at their default of `None`.
    """
    if not use_cache:
        def uncached(produced):
            return [int(v) for v in np.asarray(
                forward_dynamic(model, list(prompt_ids) + list(produced))[-1])]
        return uncached

    cache = primed_cache if primed_cache is not None else new_kv_cache(model)
    fed: list[int] = list(primed_fed) if primed_fed is not None else []

    def cached(produced):
        nonlocal fed
        full = list(prompt_ids) + list(produced)
        if full[:len(fed)] != fed:
            cache.reset()
            fed = []
        logits = forward_dynamic(model, full[len(fed):], cache=cache)
        fed = full
        return [int(v) for v in np.asarray(logits[-1])]

    return cached


def decode_constrained_dynamic(model: QuantizedModel, prompt_ids, dfa, max_new_tokens: int,
                               use_cache: bool = True) -> list[int]:
    """`decode_constrained`'s TWIN over `forward_dynamic` (A-1 §10.1, D-SLM55) — the §15
    measured arm's schema-constrained decode. Same `constrain.greedy_decode` loop, same
    `logits_fn` shape, only the forward differs: a sibling, NOT a parameter grafted onto
    the green `decode_constrained` (whose signature does not change).
    """
    return greedy_decode(dfa, _logits_fn_dynamic(model, prompt_ids, use_cache),
                         max_tokens=max_new_tokens)
