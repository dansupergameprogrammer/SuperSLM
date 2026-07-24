"""RoPE for the T-066 spike harness.

Realizes SuperSLM_Plan.md §6.4: per-position Q2.30 sin/cos tables in int32 over
positions ``[0, context_cap)`` (§6.8 C12: upper bound exclusive), and the
pairwise rotation
``(x*cos - y*sin, x*sin + y*cos)`` in int64 intermediates with a single pinned
rounding to the activation format.

Table generation is the one place double precision runs — §6.4 states it, and
the runtime reads only the emitted integers. The rotation is integer throughout.

The single rounding is §6.2's `rounding_divide_by_pot`, the plan's only
normative rounding; §6.4 pins that there is exactly one rounding without naming
it (test-design record finding C-6), and the plan carries two different tie
rules, so the primitive is imported rather than restated.

Test-design record: Claude/Curie/superslm-t066-harness-test-design-2026-07-14.md
"""

from __future__ import annotations

import math

from superslm_spike.intmath import rounding_divide_by_pot

__all__ = [
    "ROPE_FRAC_BITS",
    "ROPE_ONE",
    "rope_tables",
    "rope_apply_pair",
]

ROPE_FRAC_BITS = 30
ROPE_ONE = 1 << ROPE_FRAC_BITS


def rope_tables(head_dim: int, context_cap: int, theta: float) -> tuple[list[list[int]], list[list[int]]]:
    """The Q2.30 cos and sin tables for positions ``[0, context_cap)``.

    ``context_cap`` rows by ``head_dim // 2`` columns — one column per rotated
    pair. The angle is ``position * theta ** (-2*pair_index/head_dim)``,
    evaluated per position rather than accumulated, so the last row carries the
    same error as the first.

    §6.8 C12 pins the upper bound EXCLUSIVE: a sequence of ``context_cap``
    tokens occupies positions ``0 .. context_cap-1``. A position
    ``>= context_cap`` is the documented over-cap rejection (§17 dim 5), never
    a table read, so this function must not emit a row for it.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"the rotation is pairwise; head_dim must be even, got {head_dim}")

    pairs = head_dim // 2
    inv_freq = [theta ** (-2.0 * pair_index / head_dim) for pair_index in range(pairs)]

    cos_table: list[list[int]] = []
    sin_table: list[list[int]] = []
    for position in range(context_cap):
        cos_row: list[int] = []
        sin_row: list[int] = []
        for frequency in inv_freq:
            angle = position * frequency
            cos_row.append(round(math.cos(angle) * ROPE_ONE))
            sin_row.append(round(math.sin(angle) * ROPE_ONE))
        cos_table.append(cos_row)
        sin_table.append(sin_row)
    return cos_table, sin_table


def rope_apply_pair(x: int, y: int, cos_q30: int, sin_q30: int) -> tuple[int, int]:
    """Rotate one pair. Each component is combined at full width, then rounded once."""
    return (
        rounding_divide_by_pot(x * cos_q30 - y * sin_q30, ROPE_FRAC_BITS),
        rounding_divide_by_pot(x * sin_q30 + y * cos_q30, ROPE_FRAC_BITS),
    )
