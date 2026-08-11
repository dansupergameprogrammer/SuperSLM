"""C10 -- the SwiGLU sigmoid fixed-point piecewise-linear LUT (S3.0, F-S3-1).

Ported per `Claude/Vitruvius/SuperSLM_S2.4_SiLU_LUT_Design-2026-07-19.md` Section 4 (the
table), Section 5 (the runtime index derivation), Section 6 (the runtime interpolation):
the pinned parameters (N=1024 nodes over x in [-16, 16], Q_idx=12 fractional bits, signed
Q15 table entries) and the two tie rules -- C14's round-half-even for the offline
table-generation step, C3's round-half-away-from-zero for both runtime roundings (the
index-placement shift and the interpolation).

`SuperSLM_S3a_WalkingSkeleton_Plan.md` Section 11 S3.0 (F-S3-1, Section 4.1) requires the
Python reference forward to compute this construction in place of the excluded
i-exp-sigmoid form (`intmath.i_exp_from_constants`-based). Both `pipeline.forward_dynamic`
and `dynamic_engine.forward_dynamic_vec` call `silu_sigmoid_q15` from this one module --
bit-equality between the two forwards' `mlp_act` sites depends on both calling the same
construction rather than two independently-written copies.

The offline table (`SIGMOID_Q15_TABLE`) is generated once, at import, in high-precision
decimal arithmetic -- the offline half of Section 3's offline/runtime split (the RoPE
`ROP1` table precedent this design follows). The runtime path below is integer-only,
division-free except for the two `intmath.rounding_divide_by_pot` roundings, and never
touches a float.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Sequence

from superslm_spike import intmath

__all__ = [
    "TABLE_N",
    "TABLE_X",
    "TABLE_Q_IDX",
    "TABLE_FRAC_BITS",
    "TABLE_K_SHIFT",
    "SIGMOID_Q15_TABLE",
    "silu_lut_index",
    "silu_lut_interpolate",
    "silu_sigmoid_q15",
]

# Section 4 -- PINNED for S2.4's initial ship.
TABLE_N = 1024
TABLE_X = 16
# Section 5 -- Q_idx = 12 fractional bits for the sub-node position.
TABLE_Q_IDX = 12
# Section 4 -- table entries are signed Q15 (SIGMOID_FRAC_BITS, pipeline.py:192).
TABLE_FRAC_BITS = 15

if TABLE_N % (2 * TABLE_X) != 0:
    raise AssertionError("Section 5 requires K = N / (2X) to be an exact integer")
_K = TABLE_N // (2 * TABLE_X)
if _K & (_K - 1) != 0:
    raise AssertionError("Section 5 requires K = N / (2X) to be a power of two")
# Section 5 -- "K = 2^k exactly ... the index scale is itself a power of two".
TABLE_K_SHIFT = _K.bit_length() - 1


def _round_half_even_to_int(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def _build_sigmoid_q15_table(n: int = TABLE_N, x_bound: int = TABLE_X,
                              frac_bits: int = TABLE_FRAC_BITS) -> tuple[int, ...]:
    """Section 4's offline table, entry i (0..n) =
    round_half_even(sigmoid(-X + i*(2X/N)) * 2**frac_bits), at high (60-digit decimal)
    precision -- the one place this construction touches a non-integer type, exactly as
    RoPE's sin/cos table generation does (Section 3). Full-domain table, no
    mirror-and-negate (Section 4): n+1 entries, node n included as the closed top end.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        step = Decimal(2 * x_bound) / Decimal(n)
        scale = Decimal(1 << frac_bits)
        one = Decimal(1)
        entries = []
        for i in range(n + 1):
            x = Decimal(-x_bound) + Decimal(i) * step
            sigmoid = one / (one + (-x).exp())
            entries.append(_round_half_even_to_int(sigmoid * scale))
    return tuple(entries)


SIGMOID_Q15_TABLE: tuple[int, ...] = _build_sigmoid_q15_table()


def silu_lut_index(code: int, m: int, e: int, n: int = TABLE_N,
                    q_idx: int = TABLE_Q_IDX, k_shift: int = TABLE_K_SHIFT
                    ) -> tuple[int, int]:
    """Section 5's index derivation: `(i0, frac)`, where `i0` is the CLAMPED node index --
    clamped first -- and `frac` is measured against that same clamped `i0` (the
    2026-07-19 frac/index-order correction). `frac` lands in the closed range
    `[0, 2**q_idx]`; it equals `2**q_idx` exactly (not 0) when `pos_fixed` saturates to
    the table's top end, which is what makes Section 6's saturation resolve to
    `table[n]` rather than `table[n-1]`.

    `code`: the int8 SwiGLU gate value in [-127, 127]. `(m, e)`: the gate's per-token
    carried scale, canonical form (`m` in [2**30, 2**31), integer `e`).
    """
    if not -127 <= code <= 127:
        raise ValueError(f"silu_lut_index: code out of int8 domain [-127,127]: {code}")
    if not (1 << 30) <= m < (1 << 31):
        raise ValueError(f"silu_lut_index: m out of canonical domain [2**30, 2**31): {m}")
    term = code * m
    shift = e + k_shift + q_idx
    if shift >= 0:
        pos_fixed = term << shift
    else:
        pos_fixed = intmath.rounding_divide_by_pot(term, -shift)
    pos_fixed += (n << q_idx) // 2
    pos_fixed = max(0, min(pos_fixed, n << q_idx))
    i0 = min(pos_fixed >> q_idx, n - 1)
    frac = pos_fixed - (i0 << q_idx)
    return i0, frac


def silu_lut_interpolate(table: Sequence[int], i0: int, frac: int,
                          q_idx: int = TABLE_Q_IDX) -> int:
    """Section 6's interpolation: `lo + RoundingDivideByPOT(frac * (hi - lo), q_idx)` --
    the C3 tie rule, the only rounding besides Section 5's own. `table` must have at
    least `i0 + 2` entries (the clamp in `silu_lut_index` guarantees `i0 <= n - 1`)."""
    lo, hi = table[i0], table[i0 + 1]
    diff = hi - lo
    product = frac * diff
    delta = intmath.rounding_divide_by_pot(product, q_idx)
    return lo + delta


def silu_sigmoid_q15(code: int, m: int, e: int, table: Sequence[int] = SIGMOID_Q15_TABLE,
                      n: int = TABLE_N, q_idx: int = TABLE_Q_IDX,
                      k_shift: int = TABLE_K_SHIFT) -> int:
    """C10's full runtime construction, composed: index derivation then interpolation
    (Section 5, Section 6) -- the reference forward's `mlp_act` conformance target
    (`SuperSLM_S3a_WalkingSkeleton_Plan.md` Section 11 S3.0)."""
    i0, frac = silu_lut_index(code, m, e, n=n, q_idx=q_idx, k_shift=k_shift)
    return silu_lut_interpolate(table, i0, frac, q_idx=q_idx)
