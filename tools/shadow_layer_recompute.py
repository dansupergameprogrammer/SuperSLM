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

# The K/V-context trailer's own fixed-width header, AFTER its own leading
# num_target_rows/target_rows[] fields (variable-length, read directly by
# `load_kv_context_trailer` below) -- context_length, num_kv_heads, head_dim.
# `WriteDump` (tools/sslm_layer_trace.cpp:222-259) carries no magic string
# anywhere in this trailer -- corrected 2026-08-03 from a prior version of
# this reader that assumed one no writer in this tree ever produces.
_TRAILER_HEADER_FMT = "<QQQ"  # context_length, num_kv_heads, head_dim
_TRAILER_HEADER_SIZE = struct.calcsize(_TRAILER_HEADER_FMT)


class KvContextTrailerError(Exception):
    """Raised on any malformed K/V-context trailer -- a byte
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


# =============================================================================
# RMSNorm site kind (T-1691 build-sequencing session, 2026-08-03) --
# independently-coded reproduction of RmsNormSite (src/forward/forward_sites.
# cpp:211-241), the first site kind ported under this session's own
# per-site-kind decomposition (build log: Claude/Brunel/superslm-t1691-
# site-comparison-build-2026-08-03.md).
# =============================================================================

_NORM_FRAC_BITS = 16  # forward_sites.cpp:38, kNormFracBits


def isqrt_reference(n: int) -> int:
    """Independently-coded floor(sqrt(n)) for n >= 0 -- the same
    mathematical function `ISqrt` (src/intmath.cpp:494-516, a radix-4
    restoring shift-and-subtract digit algorithm, 32 fixed iterations)
    computes, via Python's own arbitrary-precision `math.isqrt` (CPython's
    own long-integer square root, structurally unrelated to a fixed-digit
    restoring algorithm -- no shared apparatus, design Sec7 step 5's own
    construction rule applied to this primitive)."""
    import math
    if n < 0:
        raise ValueError("isqrt_reference: n < 0 is out of ISqrt's own stated domain")
    return math.isqrt(n)


def floor_div_i64_reference(a: int, b: int) -> int:
    """Independently-coded `FloorDivI64` (src/forward/forward_sites.cpp:
    211-219): the greatest integer q with q*b <= a, for b > 0
    (caller-ensures, matching the real function's own precondition).
    Python's own `//` operator is ALREADY floor division for arbitrary-sign
    numerators (rounds toward negative infinity), the identical mathematical
    operation the real function's own truncate-then-correct construction
    computes by a different route -- no shared code path, same function."""
    if b <= 0:
        raise ValueError("floor_div_i64_reference: b <= 0 is out of FloorDivI64's own caller-ensures domain")
    return a // b


_PROB_FRAC_BITS = 15  # intmath.h kProbFracBits
_ROPE_FRAC_BITS = 30  # intmath.h ROPE_FRAC_BITS
_BIAS_Q_FORMAT = 30  # intmath.h kBiasQFormat
_I_EXP_CLIP_N = 30  # intmath.h I_EXP_CLIP_N
_SOFTMAX_ROW_MAX_SAFE_EXPONENT = (1 << 62) >> _PROB_FRAC_BITS  # checked_chain_funnel.h

# C30's pinned, per-artifact-independent format-30 coefficient triple
# (intmath.h:512-521) -- computed here from the SAME pinned bit pattern the
# real header computes from at compile time, never hand-typed (StandardsDocument
# Sec5.4). `float.fromhex` reproduces the exact IEEE-754 double the header's own
# hex-float literal denotes; `int(x)` on a positive float truncates toward zero,
# matching C++'s `static_cast<int64_t>(double)` exactly for these positive values.
kIExpPinnedLn2 = float.fromhex("0x1.62e42fefa39efp-1")
kIExpPolyA = 0.3585
kIExpPolyB = 1.353
kIExpPolyC = 0.344
kIExpLn2Q = int(kIExpPinnedLn2 * float(1 << 30))
kIExpBQ = int(kIExpPolyB * float(1 << 30))
kIExpCaQ = int((kIExpPolyC / kIExpPolyA) * float(1 << 30))


def rmsnorm_shadow_codes(h: np.ndarray, g: np.ndarray) -> tuple[str, np.ndarray | None]:
    """Independently-coded `RmsNormSite`'s own real composition
    (src/forward/forward_sites.cpp:221-241), reproduced end to end:

        sumsq   = sum(h[i]^2)
        root    = max(isqrt(floor_div(sumsq << 2*NORM_FRAC_BITS, hidden_size)), 1)
        wide[i] = floor_div(h[i] << 2*NORM_FRAC_BITS, root) * g[i]
        (status, codes) = RequantChainChecked(wide)  -- `incoming` is empty and
            `site_constant` plays no part in `codes` (checked_chain_funnel.cpp:
            269-326: d_prime/r/s and the per-element codes loop read only
            `wide_row`; `incoming`/`site_constant` feed only the OUTPUT CARRIED
            SCALE, `running`, which this function does not compute or return --
            this session's own exact-equality comparison is on `codes` alone).

    `h` is the real, already-captured int8 hidden-state row entering this
    RMSNorm site (never dequantized: RmsNormSite's own `incoming_scale`
    parameter is accepted but "never folded in", forward_sites.cpp:224-226).
    `g` is the widened (int32, sign-extended from the artifact's own int8
    gain codes) per-channel gain array for this specific norm instance
    (attn_norm or mlp_norm)."""
    hidden_size = len(h)
    if len(g) != hidden_size:
        raise ValueError(f"rmsnorm_shadow_codes: h has {hidden_size} elements, g has {len(g)}")
    h64 = [int(x) for x in h]
    g64 = [int(x) for x in g]
    sumsq = sum(hi * hi for hi in h64)
    root = isqrt_reference(floor_div_i64_reference(sumsq << (2 * _NORM_FRAC_BITS), hidden_size))
    root = root if root > 1 else 1
    wide = np.array(
        [floor_div_i64_reference(hi << (2 * _NORM_FRAC_BITS), root) * gi for hi, gi in zip(h64, g64)],
        dtype=object)
    return requant_chain_reference(wide)


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
# Attention interior site kind (T-1691 site kind 2/4, build log Claude/Brunel/
# superslm-t1691-site-comparison-build2-2026-08-03.md) -- independently-coded
# reproductions of q_proj.requant/o_proj.requant (ProjectAndFunnel),
# attn_ctx (the inline funnel plus the fixed-point softmax/RoPE composition
# feeding it), and attn_residual (ResidualReconcileSite). Every new primitive
# below is validated against the REAL compiled function via
# tests/t1691_primitive_probe.cpp's own six new subcommands (red-first proof
# part 2, extended) before any composed call trusts it -- the same standard
# the RMSNorm-site primitives above were already held to.
# =============================================================================


def rounding_divide_by_pot_wide_reference(x: int, exponent: int) -> int:
    """Independently-coded `RoundingDivideByPOTWide` (src/intmath.cpp:357-370):
    C3's tie-away-from-zero rounding rule, generalized to an arbitrary-width
    numerator divided by 2**exponent (0 <= exponent <= 63). Python's own
    two's-complement bitwise `&`/`>>` on arbitrary-precision ints reproduce
    the real S128 facility's low-`exponent`-bit masking and floor shift
    exactly, for any `x` (positive or negative, any magnitude) -- no
    separate fixed-width tracking is needed for this primitive alone
    (`rope_apply_pair_reference`/`bias_reconcile_reference` below both mask
    their own wide intermediates to 128 bits BEFORE calling this, matching
    where the real U128/S128 facilities actually truncate)."""
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    shifted = x >> exponent  # Python's >> on a (possibly negative) int is an exact floor shift
    return shifted + 1 if remainder > threshold else shifted


def rope_apply_pair_reference(x: int, y: int, cos_q30: int, sin_q30: int) -> tuple[int, int]:
    """Independently-coded `RopeApplyPair` (src/intmath.cpp:774-784):
    (x*cos - y*sin, x*sin + y*cos), each rounded ONCE at ROPE_FRAC_BITS via
    the SAME C1/C3 tie-away-from-zero rule (`rounding_divide_by_pot_wide_
    reference`, matching `RoundingDivideByPOTImpl<int64_t>`). `x`/`y` are
    int32-range (widened int8 row values); `cos_q30`/`sin_q30` are Q2.30
    table entries -- every product stays comfortably inside int64 (design's
    own "each product <= 2^61" bound), so no 128-bit masking is needed here
    unlike the wider composites below."""
    xr = int(x) * int(cos_q30) - int(y) * int(sin_q30)
    yr = int(x) * int(sin_q30) + int(y) * int(cos_q30)
    return (
        rounding_divide_by_pot_wide_reference(xr, _ROPE_FRAC_BITS),
        rounding_divide_by_pot_wide_reference(yr, _ROPE_FRAC_BITS),
    )


def rope_apply_site_reference(
    row: np.ndarray, position: int, cos_table: np.ndarray, sin_table: np.ndarray
) -> np.ndarray:
    """Independently-coded `RopeApplySite` (forward_sites.cpp:451-549), the
    table-read/pairing/clamp composition around `rope_apply_pair_reference`:
    interleaved even/odd pairing (row[2i], row[2i+1]) against `cos_table`/
    `sin_table`'s own row `position`, then `ClampRopeCode` on each rotated
    component. Indexes `cos_table`/`sin_table` FLAT, at `position * pairs +
    i` -- matching the real function's own read (`ReadRopeTableEntryI64(cos-
    >data, row_offset + i)`, `row_offset = position * pairs`, over `cos`'s
    OWN `elem_count`, forward_sites.cpp:511-524), never a 2-D `[position,
    i]` index. Corrected 2026-08-04 from a prior version of this function
    that assumed `cos_table`/`sin_table` carry shape `[context_cap,
    head_dim/2]`: no writer in this tree emits ROP1 "cos"/"sin" tensors at
    rank 2 (`tools/kv_context_trailer_fixture.py`/`kv_saturation_fixture.py`
    both write a flat length-`context_cap*pairs` array; `sslm_artifact_
    reader.py`'s `_parse_tensor_manifest` reshapes to whatever rank the
    artifact's own tensor descriptor states, which is 1 for every artifact
    this module has read), and the real engine's own algorithm never assumes
    2-D either -- it derives `cos_rows = cos->elem_count / pairs` and reads
    by flat offset, agnostic to how many dimensions the tensor was declared
    with. Caught by `tools/test_t1691_attention_shadow.py`'s own real-engine
    fixture tier (an `IndexError`, not a value mismatch -- a defect in this
    port's own table-indexing plumbing, not a claim about the real engine's
    arithmetic) before any real-population comparison used this function.
    `CheckPositionOverCap`/`RopeTableExtentExceeded` are not reproduced here
    -- every call site in this module supplies a `position` already proven
    in-domain by the real engine's own successful run (the site-dump/K/V-
    context-trailer capture never exists for a rejected forward pass), so
    this function is total over the inputs it is actually driven with,
    matching this module's own established convention (`rmsnorm_shadow_
    codes` does not re-derive `RmsNormSite`'s own unreachable guards
    either)."""
    head_dim = len(row)
    pairs = head_dim // 2
    cos_flat = np.asarray(cos_table).reshape(-1)
    sin_flat = np.asarray(sin_table).reshape(-1)
    row_offset = position * pairs
    out = np.empty(head_dim, dtype=np.int8)
    for i in range(pairs):
        x = int(row[2 * i])
        y = int(row[2 * i + 1])
        cos_q30 = int(cos_flat[row_offset + i])
        sin_q30 = int(sin_flat[row_offset + i])
        xr, yr = rope_apply_pair_reference(x, y, cos_q30, sin_q30)
        out[2 * i] = clamp_rope_code_reference(xr)
        out[2 * i + 1] = clamp_rope_code_reference(yr)
    return out


def landing_rescale_reference(branch_code: int, m_a: int, r_t: int, e_a: int, e_t: int) -> tuple[int, bool]:
    """Independently-coded `LandingRescale` (forward_sites.cpp:301-441): C27's
    residual-reconcile composite, `round_half_up(|branch_code * m_a * r_t| /
    2**k)` (k = 62 - (e_a - e_t)) with the operands' combined sign reapplied,
    the SAME magnitude-then-sign construction the real function uses (never a
    signed intermediate, which would mishandle INT64_MIN). The magnitude is
    masked to 128 bits, matching where the real U128 facility itself would
    truncate (`ComposedExponent`'s own clamp keeps k inside a range this
    module's own reachable inputs -- real successful forward-pass state --
    never approaches; unmasked precision is otherwise exact, per this
    module's own "no shared apparatus" construction). Returns (raw,
    magnitude_exceeded_int64) -- `raw` is the int64-narrowed, sign-applied
    result (BEFORE `ClampRopeCode`, per the real call sites' own two-step
    `clamp(LandingRescale(...), -127, 127)` composition); `magnitude_
    exceeded_int64` is D-SLM457's own second, distinct overflow signal
    (`ResidualReconcileSite`'s own per-element gate; the K/V-landing call
    site does not read it, matching production)."""
    u128_mask = (1 << 128) - 1
    branch_negative = branch_code < 0
    m_a_negative = m_a < 0
    negative = branch_negative != m_a_negative
    abs_branch = -branch_code if branch_negative else branch_code
    abs_m_a = -m_a if m_a_negative else m_a
    magnitude = (abs_branch * abs_m_a * r_t) & u128_mask

    k = 62 - (e_a - e_t)
    if k >= 0:
        full = ((2 * magnitude + (1 << k)) & u128_mask) >> (k + 1) if k < 128 else 0
    else:
        shift = -k
        full = (magnitude << shift) & u128_mask if shift < 128 else 0
    magnitude_exceeded = full > INT64_MAX
    raw = _to_int64(full & ((1 << 64) - 1))
    if negative:
        raw = _to_int64(-raw)
    return raw, magnitude_exceeded


def saturating_add64_reference(a: int, b: int) -> int:
    """Independently-coded `SaturatingAdd64`/`SaturatingAdd64ForIExpScale`
    (checked_chain_funnel.cpp, intmath.cpp): int64 add, saturating at
    INT64_MIN/INT64_MAX -- Python's own arbitrary-precision `+` never
    overflows, so this is the clamp alone, not a wraparound-avoidance
    trick."""
    s = int(a) + int(b)
    if s > INT64_MAX:
        return INT64_MAX
    if s < INT64_MIN:
        return INT64_MIN
    return s


def combine_carried_scale_reference(a_m: int, a_e: int, b_m: int, b_e: int) -> tuple[int, int]:
    """Independently-coded `CombineCarriedScale` (checked_chain_funnel.cpp:
    149-172): one C1/C2 high-mul (`saturating_rounding_doubling_high_mul_
    reference`, already built and validated above), then an exact single-shift
    renormalization back into the canonical `[2**30, 2**31)` mantissa range
    when the high-mul result lands below it. Caller-ensures: both operands'
    mantissas fit int32_t's own range (this module's own callers check this
    before combining, matching `RequantChainChecked`'s own step-0 gate)."""
    ma = _to_int32(a_m)
    mb = _to_int32(b_m)
    e = saturating_add64_reference(saturating_add64_reference(a_e, b_e), 31)
    m = saturating_rounding_doubling_high_mul_reference(ma, mb)
    if m < (1 << 30):
        m <<= 1
        e -= 1
    return m, e


def bias_reconcile_reference(b: int, q_b: int, r_a: int, e_a: int) -> int:
    """Independently-coded `BiasReconcile`/`BiasReconcileWide` (forward_sites.
    cpp:271-299, src/intmath.cpp:389-410): `round_half_away_from_zero(B * R_a
    / 2**exponent)`, exponent = q_b + 62 + e_a. Returns 0 when the exponent
    is outside [0, 63] -- `BiasReconcile`'s own out-of-domain convention,
    discarding `BiasReconcileWide`'s representability flag exactly as the
    real function does; otherwise the low 64 bits of the exact rounded
    product, two's-complement reinterpreted (`b * r_a` is exact in Python's
    arbitrary precision, matching the real S128 `SMul`'s own exactness)."""
    exponent = int(q_b) + 62 + int(e_a)
    if not (0 <= exponent <= 63):
        return 0
    wide = int(b) * int(r_a)
    rounded = rounding_divide_by_pot_wide_reference(wide, exponent)
    return _to_int64(rounded)


def apply_bias_reconcile_row_reference(
    acc: list[int], bias: np.ndarray, in_scale_m: int, in_scale_e: int
) -> tuple[str, list[int] | None]:
    """Independently-coded `ApplyBiasReconcileRow` (forward_sites.cpp:803-824):
    the per-channel bias insertion `ProjectAndFunnel`'s q_proj call (and the
    K/V-landing composite) apply between the weight-scale fold and the
    funnel. Only called by this module's own callers when `bias is not
    None` -- matching the real call sites' own `if (bias != nullptr)` guard,
    reproduced at the caller rather than inside this function (ProjectAndFunnel's
    own shape)."""
    exponent = _BIAS_Q_FORMAT + 62 + int(in_scale_e)
    if not (0 <= exponent <= 63):
        return "BiasReconcileProductOutOfDomain", None
    r_a = dynamic_scale_reciprocal_reference(in_scale_m)
    out = []
    for i, a in enumerate(acc):
        term = bias_reconcile_reference(int(bias[i]), _BIAS_Q_FORMAT, r_a, in_scale_e)
        total = int(a) + term
        if total < INT64_MIN or total > INT64_MAX:
            return "BiasReconcileProductOutOfDomain", None
        out.append(_to_int64(total))
    return "Ok", out


def gemm_int8_dot_reference(a, b) -> int:
    """Independently-coded `DotRowScalar`/`GemmInt8AccumulateRow` (src/matmul.
    cpp:33-41,138-147): int64 accumulate of `a[k] * b[k]`, exact -- no
    saturation, no narrowing. `a`/`b` are equal-length int8-valued
    sequences."""
    return int(sum(int(x) * int(y) for x, y in zip(a, b)))


def gemm_prob_q15_accumulate_reference(probs, values_rows, head_dim: int) -> list[int]:
    """Independently-coded `GemmProbQ15Accumulate` (src/matmul.cpp:175-191):
    `out_ctx[d] = Sum_k probs[k] * values_rows[k][d]`, exact int64
    accumulation, no saturation, no rounding."""
    out = [0] * head_dim
    for p, row in zip(probs, values_rows):
        p = int(p)
        for d in range(head_dim):
            out[d] += p * int(row[d])
    return out


def project_and_funnel_shadow_codes(
    in_codes, in_scale_m: int, in_scale_e: int, w_int8: np.ndarray,
    identity: np.ndarray, mult: np.ndarray, shift: np.ndarray, bias: np.ndarray | None,
) -> tuple[str, np.ndarray | None]:
    """Independently-coded `ProjectAndFunnel` (forward_sites.cpp:832-860): an
    int8 GEMM accumulate against the raw int8 weight matrix, the per-channel
    WSC1 fold (`apply_weight_scale_fold_reference`, already built above), the
    optional C28 bias reconciliation (`apply_bias_reconcile_row_reference`),
    then the funnel (`requant_chain_reference`) -- the same construction used
    for q_proj (`bias` supplied) and o_proj (`bias=None`), matching the real
    engine's own single shared function for both call sites."""
    out_channels = w_int8.shape[0]
    acc = [gemm_int8_dot_reference(in_codes, w_int8[j]) for j in range(out_channels)]
    acc = [
        apply_weight_scale_fold_reference(acc[j], int(identity[j]), int(mult[j]), int(shift[j]))
        for j in range(out_channels)
    ]
    if bias is not None:
        status, acc = apply_bias_reconcile_row_reference(acc, bias, in_scale_m, in_scale_e)
        if status != "Ok":
            return status, None
    wide = np.array(acc, dtype=object)
    return requant_chain_reference(wide)


def kv_landing_shadow_codes(
    in_codes, in_scale_m: int, in_scale_e: int, weight: np.ndarray,
    fold_identity, fold_mult, fold_shift, bias: np.ndarray | None,
    landing_r_t, landing_e_t, num_kv_heads: int, head_dim: int,
) -> np.ndarray:
    """Independently-coded K/V-landing composite (forward_sites.cpp:1209-1267):
    GEMM -> per-channel WSC1 fold -> optional C28 bias reconcile -> per-
    (head, dim) `LandingRescale` -> `ClampRopeCode`. Reproduces the CURRENT
    position's own K/V landing -- the mechanism D-SLM745 found unobservable
    from the real engine directly (no `SslmEmitKvLandingTrace` call site
    exists in production, and `RopeApplySite` has no trace-hook parameter),
    so this function's own correctness rests entirely on red-first proof
    (every primitive it calls validated against the real compiled function)
    and never on a real-engine trace of its OWN output -- named explicitly as
    an uncompared part of this pass (build log, "scope boundary")."""
    kv_hidden = num_kv_heads * head_dim
    acc = [gemm_int8_dot_reference(in_codes, weight[j]) for j in range(kv_hidden)]
    acc = [
        apply_weight_scale_fold_reference(
            acc[j], int(fold_identity[j]), int(fold_mult[j]), int(fold_shift[j]))
        for j in range(kv_hidden)
    ]
    if bias is not None:
        status, acc2 = apply_bias_reconcile_row_reference(acc, bias, in_scale_m, in_scale_e)
        if status != "Ok":
            raise ValueError(f"kv_landing_shadow_codes: bias reconcile rejected: {status}")
        acc = acc2
    out = np.empty((num_kv_heads, head_dim), dtype=np.int8)
    for h in range(num_kv_heads):
        for d in range(head_dim):
            i = h * head_dim + d
            raw, _ = landing_rescale_reference(
                acc[i], in_scale_m, int(landing_r_t[h]), in_scale_e, int(landing_e_t[h]))
            out[h, d] = clamp_rope_code_reference(raw)
    return out


def iexp_construct_reference(q: int, q_ln2: int, q_b: int, q_c: int) -> tuple[str, int, int]:
    """Independently-coded `IExpConstruct` (src/intmath.cpp:572-660): the
    range-reduction step feeding `IExpEvaluate`. Returns (domain, z, base);
    `z`/`base` are meaningful whenever domain is `kOk` or `kNotRepresentable`
    (IExpConstruct's own doc: a well-formed-but-unrepresentable construction
    is still reported, matching the real function's own total-over-its-
    domain construction)."""
    if q > 0:
        return "kBadQ", 0, 0
    max_q_ln2 = INT64_MAX // _I_EXP_CLIP_N
    if q_ln2 < 1 or q_ln2 > max_q_ln2:
        return "kBadQLn2", 0, 0
    clip_lo = -_I_EXP_CLIP_N * q_ln2
    clipped = clip_lo if q < clip_lo else q
    z = (-clipped) // q_ln2
    q_p = clipped + z * q_ln2
    if q_b < INT64_MIN - q_p:
        return "kBadQB", 0, 0
    base = q_p + q_b
    v = base * base + q_c
    limit = 1 << (63 + z)  # 2**63 * 2**z, z in [0, I_EXP_CLIP_N]
    upper = limit - 1
    lower = -limit
    if upper >= v >= lower:
        return "kOk", z, base
    return "kNotRepresentable", z, base


def iexp_evaluate_reference(z: int, base: int, q_c: int) -> int:
    """Independently-coded `IExpEvaluate` (src/intmath.cpp:532-559):
    `(base**2 + q_c) >> z`, an arithmetic (floor) shift, then int64-narrowed
    (two's-complement low 64 bits) -- matching the real function's own
    behaviour even on a `kNotRepresentable` construction, which is still
    evaluated and can be negative (D-SLM78's own pinned behaviour)."""
    v = base * base + q_c
    return _to_int64(v >> z)


def iexp_scale_constants_reference(
    m: int, e: int, ln2_q: int, b_q: int, ca_q: int
) -> tuple[str, tuple[int, int, int] | None]:
    """Independently-coded `IExpScaleConstants` (src/intmath.cpp:694-770),
    format 30 pinned for all three coefficients (this module's own only real
    call shape, matching `RunLayerLoop`'s own production call site). Returns
    (domain, (q_ln2, q_b, q_c)) -- the triple is `None` unless domain is
    `kOk`."""
    ln2_fmt = b_fmt = ca_fmt = 30
    if ln2_q <= 0 or b_q <= 0 or ca_q <= 0:
        return "kBadCoefficient", None
    shift_ln2 = saturating_add64_reference(saturating_add64_reference(ln2_fmt, 62), e)
    shift_b = saturating_add64_reference(saturating_add64_reference(b_fmt, 62), e)
    two_e = saturating_add64_reference(e, e)
    shift_c = saturating_add64_reference(saturating_add64_reference(ca_fmt, 124), two_e)
    if min(shift_ln2, shift_b, shift_c) < 0:
        return "kNegativeShift", None

    r_m = dynamic_scale_reciprocal_reference(m)  # non-negative over this function's reachable
                                                  # domain (design Sec4.2 step 6's own established
                                                  # execution result), so no unsigned reinterpretation
                                                  # is needed for r_m to equal its own true value.
    num_ln2 = int(ln2_q) * r_m
    num_b = int(b_q) * r_m
    num_c = (r_m * r_m) * int(ca_q)

    def shr(v: int, s: int) -> int:
        return 0 if s >= 128 else (v >> s)

    u_q_ln2 = shr(num_ln2, shift_ln2)
    u_q_b = shr(num_b, shift_b)
    u_q_c = shr(num_c, shift_c)
    if not (INT64_MIN <= u_q_ln2 <= INT64_MAX and INT64_MIN <= u_q_b <= INT64_MAX and
            INT64_MIN <= u_q_c <= INT64_MAX):
        return "kNotRepresentable", None
    return "kOk", (u_q_ln2, u_q_b, u_q_c)


def check_softmax_row_width_domain_reference(q_b: int, q_c: int, width: int) -> str:
    """Independently-coded `CheckSoftmaxRowWidthDomain` (checked_chain_funnel.
    cpp:512-575): the closed-form gate `SoftmaxRowQ15`'s own production call
    site runs before invoking the kernel."""
    if q_c < 0:
        return "SoftmaxRowWidthOutOfDomain"
    if width == 0:
        return "SoftmaxRowWidthOutOfDomain"
    m128 = q_b * q_b + q_c
    if not (INT64_MIN <= m128 <= INT64_MAX) or m128 > _SOFTMAX_ROW_MAX_SAFE_EXPONENT:
        return "SoftmaxRowWidthOutOfDomain"
    if m128 != 0 and width > (INT64_MAX // m128):
        return "SoftmaxRowWidthOutOfDomain"
    return "Ok"


def softmax_row_q15_reference(
    scores: list[int], width: int, q_ln2: int, q_b: int, q_c: int
) -> tuple[bool, list[int]]:
    """Independently-coded `SoftmaxRowQ15` (src/intmath.cpp:788-893): C32's
    per-row fixed-point softmax -- shift-by-max, per-element `IExpConstruct`/
    `IExpEvaluate`, the per-element real-value ceiling `M = q_b**2 + q_c`
    (Popper 2026-07-28 Null 1's own closed-form-independent check), sum, then
    the Q15 divide. Returns (well_formed, probs)."""
    if width == 0:
        return True, []
    m128 = q_b * q_b + q_c
    m_usable = (1 <= m128 <= INT64_MAX) and (m128 <= _SOFTMAX_ROW_MAX_SAFE_EXPONENT)
    m = m128 if m_usable else 0

    peak = max(scores)
    shifted = [s - peak for s in scores]

    exps = []
    total = 0
    all_well_formed = True
    for sh in shifted:
        domain, z, base = iexp_construct_reference(sh, q_ln2, q_b, q_c)
        if domain in ("kBadQ", "kBadQLn2", "kBadQB"):
            all_well_formed = False
            exps.append(0)
            continue
        value = iexp_evaluate_reference(z, base, q_c)
        if not m_usable or value < 0 or value > m:
            all_well_formed = False
            exps.append(0)
            continue
        exps.append(value)
        total += value

    denom = total if total > 1 else 1
    probs = [(e << _PROB_FRAC_BITS) // denom for e in exps]
    return all_well_formed, probs


def attn_ctx_shadow_codes(
    q_codes_real: np.ndarray, q_scale_m: int, q_scale_e: int,
    attn_norm_codes_real: np.ndarray, attn_norm_scale_m: int, attn_norm_scale_e: int,
    position: int, width: int, layer, rope_cos: np.ndarray, rope_sin: np.ndarray,
    real_k_prior: np.ndarray, real_v_prior: np.ndarray,
    num_heads: int, num_kv_heads: int, head_dim: int,
) -> tuple[str, np.ndarray | None]:
    """Composes the attn_ctx site's own real construction (forward_sites.cpp:
    1269-1424) end to end from real, already-captured inputs: the real
    q_proj.requant codes/scale, the real attn_norm codes/scale (this layer's
    K/V-landing GEMM input, identical to q_proj's own GEMM input), the real
    prior-position K/V codes (`real_k_prior`/`real_v_prior`, the K/V-context
    trailer, shape `[context_length, num_kv_heads, head_dim]`), and the
    artifact's own real weights/folds/constants (`layer`, a
    `LayerParityWeights`). The CURRENT position's own K/V landing and BOTH
    RoPE rotations (Q and the current K) are computed by THIS function's own
    independently-coded primitives -- never traced from the real engine,
    which has no accessor for either (D-SLM745) -- so a mismatch this
    function reports could originate in this function's own RoPE/landing
    reproduction as well as in the real engine's arithmetic; the build log
    states this as an explicit, named limit on what an attn_ctx disagreement
    would mean."""
    group = num_heads // num_kv_heads

    k_current = kv_landing_shadow_codes(
        attn_norm_codes_real, attn_norm_scale_m, attn_norm_scale_e, layer.k_weight,
        layer.k_fold[:, 0], layer.k_fold[:, 1], layer.k_fold[:, 2], layer.k_bias,
        layer.kv_landing_r_t_k, layer.kv_landing_e_t_k, num_kv_heads, head_dim)
    v_current = kv_landing_shadow_codes(
        attn_norm_codes_real, attn_norm_scale_m, attn_norm_scale_e, layer.v_weight,
        layer.v_fold[:, 0], layer.v_fold[:, 1], layer.v_fold[:, 2], layer.v_bias,
        layer.kv_landing_r_t_v, layer.kv_landing_e_t_v, num_kv_heads, head_dim)

    k_current_rot = np.empty_like(k_current)
    for kv_head in range(num_kv_heads):
        k_current_rot[kv_head] = rope_apply_site_reference(k_current[kv_head], position, rope_cos, rope_sin)

    ctx_wide = np.empty(num_heads * head_dim, dtype=object)
    khead_cache: dict[int, tuple[int, int, int]] = {}

    for h in range(num_heads):
        kv_head = h // group
        q_rot = rope_apply_site_reference(
            q_codes_real[h * head_dim:(h + 1) * head_dim], position, rope_cos, rope_sin)

        if kv_head not in khead_cache:
            if not (INT32_MIN <= q_scale_m <= INT32_MAX):
                return "CarriedScaleMantissaOutOfDomain", None
            sm_khead_m = int(layer.iexp_softmax_khead_m[kv_head])
            sm_khead_e = int(layer.iexp_softmax_khead_e[kv_head])
            if not (INT32_MIN <= sm_khead_m <= INT32_MAX):
                return "CarriedScaleMantissaOutOfDomain", None
            sm_m, sm_e = combine_carried_scale_reference(q_scale_m, q_scale_e, sm_khead_m, sm_khead_e)
            if not (INT32_MIN <= sm_m <= INT32_MAX):
                return "CarriedScaleMantissaOutOfDomain", None
            domain, triple = iexp_scale_constants_reference(
                sm_m, sm_e, kIExpLn2Q, kIExpBQ, kIExpCaQ)
            if domain != "kOk":
                return "IExpScaleDerivationOutOfDomain", None
            khead_cache[kv_head] = triple
        q_ln2, q_b, q_c = khead_cache[kv_head]

        gate = check_softmax_row_width_domain_reference(q_b, q_c, width)
        if gate != "Ok":
            return gate, None

        k_rows = [real_k_prior[pos, kv_head] for pos in range(width - 1)] + [k_current_rot[kv_head]]
        scores = [gemm_int8_dot_reference(q_rot, k_row) for k_row in k_rows]

        well_formed, probs = softmax_row_q15_reference(scores, width, q_ln2, q_b, q_c)
        if not well_formed:
            return "SoftmaxKernelRefusedAfterGateAccepted", None

        v_rows = [real_v_prior[pos, kv_head] for pos in range(width - 1)] + [v_current[kv_head]]
        ctx_acc = gemm_prob_q15_accumulate_reference(probs, v_rows, head_dim)

        for d in range(head_dim):
            ctx_wide[h * head_dim + d] = apply_weight_scale_fold_reference(
                ctx_acc[d], int(layer.ctx_fold[h, 0]), int(layer.ctx_fold[h, 1]), int(layer.ctx_fold[h, 2]))

    return requant_chain_reference(ctx_wide)


def residual_reconcile_shadow_codes(
    branch_codes, branch_scale_m: int, branch_scale_e: int,
    stream_codes, stream_scale_m: int, stream_scale_e: int, hidden_size: int,
) -> tuple[str, np.ndarray | None]:
    """Independently-coded `ResidualReconcileSite` (forward_sites.cpp:659-777):
    `r_h = CarriedScaleReciprocal(stream_scale.m)`, then per element
    `LandingRescale(branch_code[i], branch_scale.m, r_h, branch_scale.e,
    stream_scale.e) + stream_code[i]`, rejecting the whole row if any
    element's `magnitude_exceeded_int64` flag fires (D-SLM457), then the
    funnel. Used for both attn_residual and mlp_residual (this pass drives
    attn_residual only; mlp_residual is site kind 3/4's own scope)."""
    if not (INT32_MIN <= stream_scale_m <= INT32_MAX):
        return "CarriedScaleMantissaOutOfDomain", None
    r_h = dynamic_scale_reciprocal_reference(stream_scale_m)
    wide = np.empty(hidden_size, dtype=object)
    any_exceeded = False
    for i in range(hidden_size):
        reconciled, exceeded = landing_rescale_reference(
            int(branch_codes[i]), branch_scale_m, r_h, branch_scale_e, stream_scale_e)
        if exceeded:
            any_exceeded = True
            continue
        wide[i] = reconciled + int(stream_codes[i])
    if any_exceeded:
        return "ResidualReconciliationMagnitudeOutOfDomain", None
    return requant_chain_reference(wide)


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
    target_rows: list[int]  # the captured dump ROW numbers (1-indexed, `row = layer_index + 1`),
                             # in the order `WriteDump` wrote them -- ascending layer order
    context_length: int
    num_key_value_heads: int
    head_dim: int
    k_codes: np.ndarray  # [len(target_rows), context_length, num_key_value_heads, head_dim] int8
    v_codes: np.ndarray  # same shape

    @property
    def num_target_layers(self) -> int:
        """Kept as a read-only alias -- `len(target_rows)`, the count this
        module's own count-reconciliation formula (design Sec7 step 4's own
        `num_target_layers * context_length * num_key_value_heads * head_dim
        * 2`) names."""
        return len(self.target_rows)


def load_kv_context_trailer(path, *, after_offset: int) -> KvContextTrailer:
    """Reads `WriteDump`'s own real K/V-context trailer byte layout
    (`tools/sslm_layer_trace.cpp:222-259`, read at source and corrected
    2026-08-03 against a PRIOR version of this reader that assumed a
    magic-prefixed, K-block-then-V-block layout no writer in this tree ever
    produces -- an oracle validated only against its own test-authored
    fixture bytes, the same shape as the BIA1 parser defect this campaign
    already found once, D-SLM742). The real layout, appended after
    `after_offset` bytes of preceding dump content (the 29-row body plus
    T-1689's own saturation-delta trailer), carries NO magic string:
    `num_target_rows` (u64), `target_rows[num_target_rows]` (u64 each, the
    dump ROW numbers captured, ascending), `context_length` (u64),
    `num_key_value_heads` (u64), `head_dim` (u64), then the codes -- laid
    out per target row, per position (0..context_length-1), K for every kv
    head THEN V for every kv head (interleaved per position, never a
    whole-K-block followed by a whole-V-block). Loudly rejects
    (`KvContextTrailerError`) a truncated file or a written element count
    that does not match the independently-derived expected count
    `len(target_rows) * context_length * num_key_value_heads * head_dim * 2`
    (design Sec7 step 4's own two owed cells, (a)/(b))."""
    data = Path(path).read_bytes()
    if len(data) < after_offset + 8:
        raise KvContextTrailerError(
            f"load_kv_context_trailer: file too short for the num_target_rows header at offset "
            f"{after_offset}"
        )
    (num_target_rows,) = struct.unpack_from("<Q", data, after_offset)
    off = after_offset + 8
    if len(data) < off + 8 * num_target_rows:
        raise KvContextTrailerError("load_kv_context_trailer: file too short for target_rows")
    target_rows = list(struct.unpack_from(f"<{num_target_rows}Q", data, off))
    off += 8 * num_target_rows

    if len(data) < off + _TRAILER_HEADER_SIZE:
        raise KvContextTrailerError("load_kv_context_trailer: file too short for the fixed header")
    context_length, num_kv_heads, head_dim = struct.unpack_from(_TRAILER_HEADER_FMT, data, off)
    off += _TRAILER_HEADER_SIZE

    body = data[off:]
    expected_count = num_target_rows * context_length * num_kv_heads * head_dim * 2
    if len(body) != expected_count:
        raise KvContextTrailerError(
            f"load_kv_context_trailer: byte count {len(body)} != expected "
            f"{expected_count} (num_target_rows={num_target_rows}, "
            f"context_length={context_length}, num_key_value_heads={num_kv_heads}, "
            f"head_dim={head_dim})"
        )

    # Per row, per position: K (num_kv_heads*head_dim bytes), then V
    # (num_kv_heads*head_dim bytes) -- WriteDump's own real interleaving,
    # never a K-block-then-V-block split at the row level.
    per_row_shape = (context_length, 2, num_kv_heads, head_dim)
    all_rows = np.frombuffer(body, dtype=np.int8).reshape((num_target_rows,) + per_row_shape)
    k_codes = np.ascontiguousarray(all_rows[:, :, 0, :, :])
    v_codes = np.ascontiguousarray(all_rows[:, :, 1, :, :])

    return KvContextTrailer(
        target_rows=target_rows,
        context_length=context_length,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        k_codes=k_codes,
        v_codes=v_codes,
    )


# =============================================================================
# Site-dump (T-1691 build-sequencing session, 2026-08-03): reads the SEPARATE
# file `sslm_layer_trace.cpp --site-dump` writes -- one record per
# RequantChainChecked invocation at each target row (tools/sslm_layer_trace.
# cpp's own `WriteSiteDump`, this session's own build log). Never the same
# file as the K/V-context trailer above; no shared parsing code with it.
# =============================================================================


@dataclass
class SiteRecord:
    site: str  # e.g. "layer26.attn_norm" -- 0-indexed layer, matching KeyRow's own `layer` parameter
    x_int: np.ndarray  # int64, the wide row as read at the site (input)
    codes: np.ndarray  # int8, the site's own real output codes
    d_prime: int
    dn: int
    s: int
    r: int
    m_out: int
    e_out: int


def load_site_dump(path) -> list[SiteRecord]:
    """Reads `WriteSiteDump`'s own format (tools/sslm_layer_trace.cpp): uint64
    num_records, then per record: uint64 name_len, the name's own bytes,
    uint64 n_x, n_x int64 x_int values, uint64 n_codes, n_codes int8 codes,
    int64 d_prime, int64 dn, int32 s, int64 r, int64 m_out, int64 e_out."""
    data = Path(path).read_bytes()
    off = 0
    (n,) = struct.unpack_from("<Q", data, off)
    off += 8
    records = []
    for _ in range(n):
        (name_len,) = struct.unpack_from("<Q", data, off)
        off += 8
        site = data[off : off + name_len].decode()
        off += name_len
        (n_x,) = struct.unpack_from("<Q", data, off)
        off += 8
        x_int = np.array(struct.unpack_from(f"<{n_x}q", data, off), dtype=np.int64)
        off += 8 * n_x
        (n_codes,) = struct.unpack_from("<Q", data, off)
        off += 8
        codes = np.frombuffer(data, dtype=np.int8, count=n_codes, offset=off).copy()
        off += n_codes
        d_prime, dn, s, r, m_out, e_out = struct.unpack_from("<qqiqqq", data, off)
        off += 8 + 8 + 4 + 8 + 8 + 8
        records.append(SiteRecord(site=site, x_int=x_int, codes=codes, d_prime=d_prime, dn=dn, s=s, r=r,
                                   m_out=m_out, e_out=e_out))
    if off != len(data):
        raise ValueError(f"load_site_dump: {len(data) - off} trailing bytes beyond the declared records")
    return records
