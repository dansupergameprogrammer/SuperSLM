#!/usr/bin/env python3
"""T-1691 -- the weight-quantization/engine-arithmetic decomposition (design
`Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md`
Sec7, amended against Loki's fracture, D-SLM718).

Two independently-coded arms, sharing no function between them (design
Sec7 step 5's own "no shared apparatus" construction -- Sec11 dimension 8):

  * The **parity shadow** (`parity_shadow_layer`) reproduces every
    integer-precision mechanism the real int8 engine's own site sequence
    has: an int8 GEMM accumulate, the per-channel weight-scale fold
    (`MultiplyByQuantizedMultiplier`/`SaturatingRoundingDoublingHighMul`/
    `RoundingDivideByPOT`, bit-exact port of `src/intmath.cpp`), and the
    dynamic per-row max-abs requantization to 127 levels
    (`RequantChainChecked`'s own construction: `MaxAbsReduceWide` ->
    `NormalizeScale` -> `DynamicScaleReciprocal` -> `RequantTokenCodeWide`
    per element, plus its C29 domain check), composed once per site in
    the eleven-site real inventory. It is compared to the real engine by
    EXACT INTEGER-CODE EQUALITY -- this is the C1 test (design Sec9/Sec10).

  * The **precision shadow** (`precision_shadow_layer`) runs the SAME site
    sequence in float64: each projection's raw int8 weights are dequantized
    once via `real_scale` (design Sec2.4's own recovered per-channel
    multiplier) and the composition is a plain float64 matrix-vector
    product, with no intermediate int8 requantization, no `ClampRopeCode`
    saturation, and no `RequantChainChecked` domain rejection anywhere.
    It measures C2's own magnitude, trustworthy only once the parity arm
    has cleared a cell (design Sec7 step 6c).

Every primitive the parity shadow depends on
(`normalize_scale_reference`, `dynamic_scale_reciprocal_reference`,
`max_abs_reduce_wide_reference`, `requant_token_code_wide_reference`,
`clamp_rope_code_reference`, `requant_chain_reference`) is an
independently-coded reimplementation of the real compiled C++ function of
the same name (read at source, `include/superslm/intmath.h`,
`src/intmath.cpp`, `src/forward/checked_chain_funnel.cpp`,
`src/forward/forward_sites.cpp`, at `main`@`c6cfa03d1aa9a7fc6c671b811c1cef56fc7ca96b`),
never a call into the compiled library itself -- this module links against
nothing in `src/` or `include/superslm/`, by construction, per the design's
own "no shared apparatus" argument (design Sec7 step 1, step 5). Python's
arbitrary-precision integers are used to compute the SAME finite-width
result the real functions produce; every operation below is checked, at
this module's own primitive-level red-first proof (part 2), against the
real compiled function via `tests/t1691_primitive_probe.cpp` before it is
trusted, so python's unbounded-width arithmetic never silently diverges
from the real narrowing-cast/overflow convention at the value ranges this
module's own callers actually reach.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =============================================================================
# Constants
# =============================================================================

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

# The new K/V-context trailer's own magic (design Sec7 step 4) -- an 8-byte
# tag distinct from any prior trailer's own bytes, checked before any header
# field is parsed.
KV_CONTEXT_TRAILER_MAGIC = b"KVCTXT01"
assert len(KV_CONTEXT_TRAILER_MAGIC) == 8

_TRAILER_HEADER_FMT = "<QQQQ"  # num_target_layers, context_length, num_kv_heads, head_dim
_TRAILER_HEADER_SIZE = struct.calcsize(_TRAILER_HEADER_FMT)


class KvContextTrailerError(Exception):
    """Raised on any malformed K/V-context trailer -- wrong magic, a byte
    count that does not match the independently-derived expected count, or
    a truncated read. Loud and non-silent, per design Sec7 step 4's own two
    owed cells (D-SLM727)."""


# =============================================================================
# Primitives -- independently-coded reimplementations of the real compiled
# C++ functions, validated against them by tests/t1691_primitive_probe.cpp
# (red-first proof part 2) before any composed call trusts them.
# =============================================================================


def _to_int64(x: int) -> int:
    """Reinterpret an arbitrary Python int as its int64 two's-complement
    value (wraps exactly like a C++ int64_t narrowing cast)."""
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >= (1 << 63) else x


def _to_int32(x: int) -> int:
    x &= (1 << 32) - 1
    return x - (1 << 32) if x >= (1 << 31) else x


def clz64_reference(n: int) -> int:
    """Independently-coded `Clz64` -- count leading zeros over 64 bits.
    Domain n in [1, 2**64 - 1] (intmath.h's own stated domain)."""
    n &= (1 << 64) - 1
    if n == 0:
        raise ValueError("clz64_reference: n == 0 is out of contract")
    return 64 - n.bit_length()


def normalize_scale_reference(d_prime: int) -> tuple[int, int]:
    """Independently-coded `NormalizeScale` (src/intmath.cpp:288-292):
    p = 63 - Clz64(D'), s = 30 - p, Dn = D' << s (s >= 0) or D' >> 1 (s < 0).
    Returns (dn, s)."""
    d_prime = int(d_prime)
    p = 63 - clz64_reference(d_prime)
    s = 30 - p
    dn = (d_prime << s) if s >= 0 else (d_prime >> 1)
    return _to_int64(dn), s


def dynamic_scale_reciprocal_reference(dn: int) -> int:
    """Independently-coded `DynamicScaleReciprocal` (src/intmath.cpp:295-325):
    R = round_half_up(2**62 / Dn) via 3 Newton iterations plus 2 fixed
    branch-free correction steps -- the SAME construction, not the closed
    form, so this reference exercises the identical algorithm the real
    function runs (the closed-form equality is a PROVEN property of the
    real function's output, D-SLM730, not this reference's own
    construction)."""
    dn = int(dn)
    k_c32 = (2 * (48 << 31) + 17) // 34
    k_c32_2 = (2 * (32 << 31) + 17) // 34

    y = k_c32 - ((k_c32_2 * dn) >> 31)

    for _ in range(3):  # DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS
        dn_y = (dn * y) >> 31
        delta = (1 << 32) - dn_y
        y = (y * delta) >> 31

    for _ in range(2):  # DYNAMIC_RECIPROCAL_CORRECTION_STEPS
        residual_2x = 2 * ((1 << 62) - y * dn)
        if residual_2x >= dn:
            y += 1
        elif residual_2x < -dn:
            y -= 1
    return _to_int64(y)


def max_abs_reduce_wide_reference(x: np.ndarray) -> int:
    """Independently-coded `MaxAbsReduceWide` (src/intmath.cpp:419-441):
    unsigned magnitude max over the row, all-zero-row guard D' = max(D, 1),
    saturating narrow to int64 (clamped at INT64_MAX, reachable only from
    an INT64_MIN element whose true magnitude 2**63 has no int64
    representation)."""
    d = 0
    for xi in np.asarray(x).tolist():
        xi = int(xi)
        a = -xi if xi < 0 else xi
        if a > d:
            d = a
    if d < 1:
        d = 1
    return INT64_MAX if d > INT64_MAX else d


def requant_token_code_wide_reference(x_i: int, r: int, s: int) -> int:
    """Independently-coded `RequantTokenCodeWide` (src/intmath.cpp:468-480):
    q_i = clamp(round_half_away_from_zero((x_i * 127 * R) / 2**(62-s)),
    -127, 127)."""
    x_i, r, s = int(x_i), int(r), int(s)
    exponent = 62 - s
    abs_x = -x_i if x_i < 0 else x_i

    prod = abs_x * r * 127  # |x_i|*127*R, exact (Python arbitrary precision)
    numerator = 2 * prod + (1 << exponent)
    magnitude = numerator >> (exponent + 1)

    if magnitude > 127:
        magnitude = 127
    return -magnitude if x_i < 0 else magnitude


def clamp_rope_code_reference(raw: int) -> int:
    """Independently-coded `ClampRopeCode` (src/forward/forward_sites.cpp:443-449):
    clamp to the pinned CODE range [-127, 127]."""
    raw = int(raw)
    if raw > 127:
        return 127
    if raw < -127:
        return -127
    return raw


def requant_chain_reference(
    wide_row: np.ndarray, *, drop_rounding_bias: bool = False
) -> tuple[str, np.ndarray | None]:
    """Independently-coded `RequantChainChecked`'s own composition
    (src/forward/checked_chain_funnel.cpp:249-...): C29's domain check
    (D' > 2**31 -> `ChainInputOutOfDomain`), then NormalizeScale ->
    DynamicScaleReciprocal -> RequantTokenCodeWide per element. Returns
    (status, codes) -- codes is None on rejection.

    `drop_rounding_bias` is the guard-vitality/defect-injection knob
    (`parity_shadow_layer`'s own `inject_defect_site` uses this): drops the
    `+ 2**exponent` rounding-bias term from `requant_token_code_wide_reference`'s
    own numerator at every element in this one call, a deliberate one-line
    arithmetic departure (design Sec7's own "the rounding-bias term dropped
    from a single requant site")."""
    row = np.asarray(wide_row, dtype=object)
    d_prime = max_abs_reduce_wide_reference(row)
    if d_prime > (1 << 31):
        return "ChainInputOutOfDomain", None

    dn, s = normalize_scale_reference(d_prime)
    r = dynamic_scale_reciprocal_reference(dn)

    codes = np.empty(row.shape[0], dtype=np.int8)
    for i, x_i in enumerate(row.tolist()):
        if drop_rounding_bias:
            x_i, rr, ss = int(x_i), int(r), int(s)
            exponent = 62 - ss
            abs_x = -x_i if x_i < 0 else x_i
            prod = abs_x * rr * 127
            numerator = 2 * prod  # rounding-bias term dropped
            magnitude = numerator >> (exponent + 1)
            if magnitude > 127:
                magnitude = 127
            codes[i] = -magnitude if x_i < 0 else magnitude
        else:
            codes[i] = requant_token_code_wide_reference(int(x_i), r, s)
    return "Ok", codes


def saturating_rounding_doubling_high_mul_reference(a: int, b: int) -> int:
    """Independently-coded `SaturatingRoundingDoublingHighMul`
    (src/intmath.cpp:230-234): (a*b + 2**30) >> 31, ties toward +infinity,
    saturating at INT32_MAX."""
    a, b = _to_int32(int(a)), _to_int32(int(b))
    ab = a * b
    result = (ab + (1 << 30)) >> 31
    return INT32_MAX if result > INT32_MAX else result


def rounding_divide_by_pot_i32_reference(x: int, exponent: int) -> int:
    """Independently-coded int32 `RoundingDivideByPOT`
    (src/intmath.cpp:242-249): x / 2**exponent, ties away from zero."""
    x = _to_int32(int(x))
    exponent = int(exponent)
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    ux = x & 0xFFFFFFFF
    remainder = ux & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    q = x >> exponent  # Python's >> is an arithmetic (floor) shift, matching C++20's guarantee
    return q + (1 if remainder > threshold else 0)


def multiply_by_quantized_multiplier_reference(x: int, mult: int, shift: int) -> int:
    """Independently-coded `MultiplyByQuantizedMultiplier`
    (src/intmath.cpp:265-268): RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, mult), shift)."""
    return rounding_divide_by_pot_i32_reference(
        saturating_rounding_doubling_high_mul_reference(x, mult), shift
    )


def apply_weight_scale_fold_reference(acc: int, identity: int, mult: int, shift: int) -> int:
    """Independently-coded `ApplyWeightScaleFold` (forward_sites.cpp:260-267):
    a pass-through when identity != 0, else `MultiplyByQuantizedMultiplier`."""
    if identity:
        return _to_int32(int(acc))
    return multiply_by_quantized_multiplier_reference(acc, mult, shift)


# =============================================================================
# Part 1 (design Sec7 step 2): the precision shadow's own recovered
# per-channel real-valued multiplier and dequantized weight matrix.
# =============================================================================


def real_scale(mult: np.ndarray, shift: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Design Sec2.4's own recovered per-channel multiplier:
    `mult[k] * 2**(-31 - shift[k])` when `identity[k] == 0`, else `1.0` --
    the exact formula `ApplyWeightScaleFold`/`MultiplyByQuantizedMultiplier`
    implement, read at source, not estimated from output behavior."""
    mult = np.asarray(mult, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    identity = np.asarray(identity)
    scale = mult * np.exp2(-31.0 - shift)
    return np.where(identity != 0, 1.0, scale)


def dequantize_weight_matrix(
    w_int8: np.ndarray, mult: np.ndarray, shift: np.ndarray, identity: np.ndarray
) -> np.ndarray:
    """`w_float[k, i] = w_int8[k, i] * real_scale[k]` (design Sec7 step 2):
    `real_scale` is per OUTPUT channel -- `SiteFoldTriple`'s own
    `identity`/`mult`/`shift` shape `[out_channels]` -- broadcast across
    each output row `k` of the `[out_channels, in_channels]` weight
    matrix."""
    scale = real_scale(mult, shift, identity)  # shape [out_channels]
    return w_int8.astype(np.float64) * scale[:, None]


# =============================================================================
# Site-fold-triple / layer-shadow-result contract types (design Sec7 step 5,
# and tools/test_t1691_shadow_layer_recompute.py's own derived contract).
# =============================================================================


@dataclass
class SiteFoldTriple:
    kind: str
    w_int8: np.ndarray  # [out_channels, in_channels] int8
    identity: np.ndarray  # [out_channels] int64
    mult: np.ndarray  # [out_channels] int64
    shift: np.ndarray  # [out_channels] int64


@dataclass
class LayerShadowResult:
    codes: np.ndarray  # [hidden_size] int8
    status: str  # "Ok" or the SslmForwardStatus name of the first rejecting site
    rejected_at_site: int | None
    rope_landing_clamped: bool
    rope_apply_clamped: bool


def _site_step(
    codes: np.ndarray, site: SiteFoldTriple, *, drop_rounding_bias: bool = False
) -> tuple[str, np.ndarray | None]:
    """One site's own composition: an int8 GEMM accumulate against the
    site's raw int8 weight matrix, `ApplyWeightScaleFold` per output
    channel (the SAME `(identity, mult, shift)` fold triple the real
    engine's own `LayerWeights` carries), then `RequantChainChecked`'s own
    dynamic per-row max-abs requantization -- the SAME construction every
    one of the real eleven call sites uses (design Sec7 step 5a), reproduced
    once per site kind here rather than specialized per kind, because every
    one of the eleven real call sites folds through this identical
    primitive regardless of the site kind wrapping it."""
    out_channels = site.w_int8.shape[0]
    acc = site.w_int8.astype(np.int64) @ np.asarray(codes, dtype=np.int64)
    wide_row = np.empty(out_channels, dtype=object)
    for k in range(out_channels):
        wide_row[k] = apply_weight_scale_fold_reference(
            int(acc[k]), int(site.identity[k]), int(site.mult[k]), int(site.shift[k])
        )
    return requant_chain_reference(wide_row, drop_rounding_bias=drop_rounding_bias)


def parity_shadow_layer(
    seed_codes: np.ndarray,
    sites: list[SiteFoldTriple],
    rope_landing_raw: int,
    rope_apply_raw: int,
    inject_defect_site: int | None = None,
) -> LayerShadowResult:
    """Design Sec7 step 5a: composes `sites` in order through the SAME
    integer arithmetic `RequantChainChecked`'s own real call sites use --
    dynamic per-row max-abs requantization to 127 levels, the C29 domain
    check, and the int32/int64 narrowing-cast/overflow convention exactly
    as the real C++ functions (never Python's arbitrary-precision default,
    per every primitive above's own `_to_int32`/`_to_int64` narrowing).
    Reproduces `ClampRopeCode` at the K/V-landing site (`rope_landing_raw`)
    and the RopeApplySite call (`rope_apply_raw`) directly on the raw
    values supplied -- these are the current position's own freshly-computed
    landing/rotation values, already composed by the caller from the real
    engine's own K/V-landing/RoPE-apply formula (design Sec7 step 4/5a);
    this function's own job over them is exactly `ClampRopeCode`'s clamp,
    reproduced bit-exact (`clamp_rope_code_reference`) and reported as a
    saturation flag. `inject_defect_site` (a `SiteFoldTriple` index)
    deliberately departs from the correct arithmetic at that one site only
    (the rounding-bias term dropped from `RequantTokenCodeWide`'s own
    numerator) -- the red-first proof's own guard-vitality construction
    (design Sec7 red-first proof part 6)."""
    codes = np.asarray(seed_codes, dtype=np.int64)
    status = "Ok"
    rejected_at_site: int | None = None

    for idx, site in enumerate(sites):
        drop_bias = inject_defect_site is not None and idx == inject_defect_site
        site_status, site_codes = _site_step(codes, site, drop_rounding_bias=drop_bias)
        if site_status != "Ok":
            status = site_status
            rejected_at_site = idx
            break
        codes = site_codes.astype(np.int64)

    final_codes = codes.astype(np.int8) if status == "Ok" else np.zeros(0, dtype=np.int8)

    rope_landing_code = clamp_rope_code_reference(rope_landing_raw)
    rope_apply_code = clamp_rope_code_reference(rope_apply_raw)

    return LayerShadowResult(
        codes=final_codes,
        status=status,
        rejected_at_site=rejected_at_site,
        rope_landing_clamped=(rope_landing_code != int(rope_landing_raw)),
        rope_apply_clamped=(rope_apply_code != int(rope_apply_raw)),
    )


def precision_shadow_layer(seed_codes: np.ndarray, sites: list[SiteFoldTriple]) -> np.ndarray:
    """Design Sec7 step 5b, unchanged construction: the SAME site sequence
    in float64, no intermediate int8 requantization, no `ClampRopeCode`
    saturation, no `RequantChainChecked` domain rejection anywhere --
    independently coded from `parity_shadow_layer` (no shared function: a
    plain float64 matrix-vector product against `dequantize_weight_matrix`'s
    own dequantized weights, never `_site_step`/`requant_chain_reference`/
    `apply_weight_scale_fold_reference`)."""
    row = np.asarray(seed_codes, dtype=np.float64)
    for site in sites:
        w_float = dequantize_weight_matrix(site.w_int8, site.mult, site.shift, site.identity)
        row = w_float @ row
    return row


# =============================================================================
# Part 2 (design Sec7 step 6a): exact integer-code equality.
# =============================================================================


def compare_exact(int8_codes: np.ndarray, parity_codes: np.ndarray) -> tuple[int, list[int]]:
    """Design Sec7 step 6a: (count of differing positions, the differing
    indices) -- exact integer-code equality, never a rank statistic."""
    a = np.asarray(int8_codes)
    b = np.asarray(parity_codes)
    if a.shape != b.shape:
        raise ValueError(f"compare_exact: shape mismatch {a.shape} vs {b.shape}")
    diff = np.flatnonzero(a != b)
    return int(diff.size), diff.tolist()


# =============================================================================
# Part 3: attention-context-row reference (softmax-weighted sum), part of
# the K/V-context composition's own hand-derivable fixture.
# =============================================================================


def attention_context_row_reference(scores: np.ndarray, values: np.ndarray) -> float:
    """Softmax-weighted sum over `scores`/`values` -- the composed
    attention-context-row funnel's own hand-derivable float64 shape
    (design Sec7 red-first proof part 3), closing D-SLM503's width==1
    blindness at width > 1."""
    scores = np.asarray(scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    m = np.max(scores)
    exps = np.exp(scores - m)
    z = np.sum(exps)
    weights = exps / z
    return float(np.sum(weights * values))


# =============================================================================
# Part 4 (design Sec7 red-first proof part 4): execution-level sanity gates.
# =============================================================================


def execution_sanity_gate(stats_by_label: dict) -> int:
    """Returns 0 if every `LayerRowStats`-shaped value's own `spearman`,
    `pearson`, and `max_abs_z_diff` fields are finite across every label;
    otherwise the count of non-finite values found (nonzero, so the caller's
    own `!= 0` check reads it as a failure)."""
    bad = 0
    for stats_list in stats_by_label.values():
        for stats in stats_list:
            for value in (stats.spearman, stats.pearson, stats.max_abs_z_diff):
                if not np.isfinite(value):
                    bad += 1
    return bad


def exact_equality_sanity(n_diff: int, hidden_size: int) -> int:
    """Returns 0 if `n_diff` is in `[0, hidden_size]`; otherwise nonzero --
    a well-formedness gate distinct from the finiteness gate above, over
    the exact-equality count itself (design Sec7 red-first proof part 4)."""
    if 0 <= n_diff <= hidden_size:
        return 0
    return 1


# =============================================================================
# K/V-context trailer (design Sec7 step 4): reads the new dump section
# tools/sslm_layer_trace.cpp gains, appended after T-1689's own 28-uint64
# saturation-delta trailer.
# =============================================================================


@dataclass
class KvContextTrailer:
    num_target_layers: int
    context_length: int
    num_key_value_heads: int
    head_dim: int
    k_codes: np.ndarray  # [num_target_layers, context_length, num_key_value_heads, head_dim] int8
    v_codes: np.ndarray  # same shape


def load_kv_context_trailer(path, *, after_offset: int) -> KvContextTrailer:
    """Design Sec7 step 4: reads the K/V-context trailer appended after
    `after_offset` bytes of preceding dump content. Loudly rejects
    (`KvContextTrailerError`) a mismatched magic or a written element count
    that does not match the independently-derived expected count
    `num_target_layers * context_length * num_key_value_heads * head_dim * 2`
    (design Sec7 step 4's own two owed cells, (a)/(b))."""
    data = Path(path).read_bytes()
    if len(data) < after_offset + len(KV_CONTEXT_TRAILER_MAGIC):
        raise KvContextTrailerError(
            f"load_kv_context_trailer: file too short for magic at offset {after_offset}"
        )
    magic = data[after_offset : after_offset + len(KV_CONTEXT_TRAILER_MAGIC)]
    if magic != KV_CONTEXT_TRAILER_MAGIC:
        raise KvContextTrailerError(
            f"load_kv_context_trailer: bad magic {magic!r}, want {KV_CONTEXT_TRAILER_MAGIC!r}"
        )
    header_off = after_offset + len(KV_CONTEXT_TRAILER_MAGIC)
    if len(data) < header_off + _TRAILER_HEADER_SIZE:
        raise KvContextTrailerError("load_kv_context_trailer: file too short for header")
    num_target_layers, context_length, num_kv_heads, head_dim = struct.unpack_from(
        _TRAILER_HEADER_FMT, data, header_off
    )
    body_off = header_off + _TRAILER_HEADER_SIZE
    body = data[body_off:]

    expected_count = num_target_layers * context_length * num_kv_heads * head_dim * 2
    if len(body) != expected_count:
        raise KvContextTrailerError(
            f"load_kv_context_trailer: byte count {len(body)} != expected "
            f"{expected_count} (num_target_layers={num_target_layers}, "
            f"context_length={context_length}, num_key_value_heads={num_kv_heads}, "
            f"head_dim={head_dim})"
        )

    half = expected_count // 2
    shape = (num_target_layers, context_length, num_kv_heads, head_dim)
    k_codes = np.frombuffer(body[:half], dtype=np.int8).reshape(shape)
    v_codes = np.frombuffer(body[half:], dtype=np.int8).reshape(shape)

    return KvContextTrailer(
        num_target_layers=num_target_layers,
        context_length=context_length,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        k_codes=k_codes,
        v_codes=v_codes,
    )
