"""Arbitrary-precision reference oracle for C10 -- the SwiGLU sigmoid fixed-point
piecewise-linear LUT (SuperSLM_Plan.md S6.8's C10 pin; the runtime construction pinned by
`Claude/Vitruvius/SuperSLM_S2.4_SiLU_LUT_Design-2026-07-19.md` SS4 (the table), SS5 (the
index derivation), SS6 (the interpolation)).

Sibling to `rung1_ref.py` and `composition_ref.py`: exact arithmetic only (`decimal.Decimal`
at high precision for the one offline, double-precision-by-design table-generation step --
S2.4 SS4's own "offline-derived exception", the RoPE-table precedent; exact Python `int` for
every runtime-path step), and this module is written from the C10 row and the S2.4 design's
own prose -- NEVER a recode of `D:/SuperSLM/src/silu_lut.cpp` or
`D:/SuperSLM/include/superslm/silu_lut.h`. `SuperSLM_S3a_WalkingSkeleton_Plan.md` SS11 S3.0
names the reason: every C++ parity cell this construction eventually gates compares the C++
LUT against a Python one, so a Python LUT transcribed from the C++ source would make that
parity oracle correlated with the thing it certifies. This module and the eventual C++
construction are two independent readings of the same pinned design.

Every rounding reuses `rung1_ref.round_half_away_from_zero_ratio` -- C3's tie rule
(`RoundingDivideByPOT`), already an independent oracle in this suite -- rather than a second
reimplementation of the same primitive (oracle building on oracle, `composition_ref.py`'s own
precedent). This module never imports `intmath` or `pipeline`.

Test-design record:
Claude/Curie/superslm-s3.0-silu-lut-reference-reconciliation-test-design-2026-07-27.md
"""

from __future__ import annotations

import re
import struct
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Sequence

from rung1_ref import round_half_away_from_zero_ratio

__all__ = [
    "TABLE_N",
    "TABLE_X",
    "TABLE_Q_IDX",
    "TABLE_K_SHIFT",
    "TABLE_FRAC_BITS",
    "build_sigmoid_q15_table_oracle",
    "silu_lut_index_oracle",
    "silu_lut_interpolate_oracle",
    "silu_sigmoid_q15_oracle",
    "load_golden_table_witness",
    "SiluLutRealVectors",
    "load_silu_lut_real_vectors",
]

# S2.4 SS4: "PINNED for S2.4's initial ship" -- N=1024 nodes over x in [-16, 16].
TABLE_N = 1024
TABLE_X = 16
# S2.4 SS5: Q_idx = 12 fractional bits for the sub-node position.
TABLE_Q_IDX = 12
# S2.4 SS4: table entries are signed Q15.
TABLE_FRAC_BITS = 15

if TABLE_N % (2 * TABLE_X) != 0:
    raise AssertionError("S2.4 SS5 requires K = N / (2X) to be an exact integer")
_K = TABLE_N // (2 * TABLE_X)
if _K & (_K - 1) != 0:
    raise AssertionError("S2.4 SS5 requires K = N / (2X) to be a power of two")
# S2.4 SS5: "K = 2^k exactly ... the index scale is itself a power of two".
TABLE_K_SHIFT = _K.bit_length() - 1


def _round_half_even_to_int(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def build_sigmoid_q15_table_oracle(n: int = TABLE_N, x_bound: int = TABLE_X,
                                    frac_bits: int = TABLE_FRAC_BITS) -> tuple[int, ...]:
    """S2.4 SS4's offline table, entry i (0..n) =
    round_half_even(sigmoid(-X + i*(2X/N)) * 2**frac_bits), at arbitrary (60-digit decimal)
    precision. Round-half-even is C14's offline-emission tie rule (the RoPE-table
    precedent S2.4 SS4 cites) -- distinct from C3's runtime round-half-away-from-zero,
    which governs SS5/SS6 below, never this table-generation step.

    Full-domain table, no mirror-and-negate (S2.4 SS4): n+1 entries, node n included as
    the table's closed top end.
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


def silu_lut_index_oracle(code: int, m: int, e: int, n: int = TABLE_N,
                           q_idx: int = TABLE_Q_IDX, k_shift: int = TABLE_K_SHIFT
                           ) -> tuple[int, int]:
    """S2.4 SS5's index derivation, under the 2026-07-19 frac/index-order correction
    (the same section's own amendment log item 4): (i0, frac), where i0 is the CLAMPED
    node index -- clamped first -- and frac is measured AGAINST that same clamped i0, so
    the two are never in different reference frames. frac lands in the closed range
    [0, 2**q_idx]; it equals 2**q_idx exactly (not 0) when pos_fixed saturates to the
    table's top end, which is what makes SS6's saturation resolve to table[n] rather than
    table[n-1].

    `code`: the int8 SwiGLU gate value in [-127, 127]. `(m, e)`: the gate's per-token
    carried scale, canonical form (m in [2**30, 2**31), integer e).
    """
    if not -127 <= code <= 127:
        raise ValueError(f"silu_lut_index_oracle: code out of int8 domain [-127,127]: {code}")
    if not (1 << 30) <= m < (1 << 31):
        raise ValueError(
            f"silu_lut_index_oracle: m out of canonical domain [2**30, 2**31): {m}")
    term = code * m
    shift = e + k_shift + q_idx
    if shift >= 0:
        pos_fixed = term << shift
    else:
        pos_fixed = round_half_away_from_zero_ratio(term, 1 << (-shift))
    pos_fixed += (n << q_idx) // 2
    pos_fixed = max(0, min(pos_fixed, n << q_idx))
    i0 = min(pos_fixed >> q_idx, n - 1)
    frac = pos_fixed - (i0 << q_idx)
    return i0, frac


def silu_lut_interpolate_oracle(table: Sequence[int], i0: int, frac: int,
                                 q_idx: int = TABLE_Q_IDX) -> int:
    """S2.4 SS6's interpolation: lo + RoundingDivideByPOT(frac * (hi - lo), q_idx) --
    the C3 tie rule (round-half-away-from-zero), the only rounding besides SS5's own
    (index-placement) rounding. `table` must have at least `i0 + 2` entries."""
    lo, hi = table[i0], table[i0 + 1]
    diff = hi - lo
    product = frac * diff
    delta = round_half_away_from_zero_ratio(product, 1 << q_idx)
    return lo + delta


def silu_sigmoid_q15_oracle(table: Sequence[int], code: int, m: int, e: int,
                             n: int = TABLE_N, q_idx: int = TABLE_Q_IDX,
                             k_shift: int = TABLE_K_SHIFT) -> int:
    """C10's full runtime construction, composed: index derivation then interpolation
    (S2.4 SS5, SS6) -- the "arbitrary-precision recomputation of C10" that
    `SuperSLM_S3a_WalkingSkeleton_Plan.md` SS11 S3.0 names as the reference's `mlp_act`
    conformance target. Division-free on the runtime side; the one Decimal computation
    in this module is `build_sigmoid_q15_table_oracle`'s offline table, per SS3's
    offline/runtime split."""
    i0, frac = silu_lut_index_oracle(code, m, e, n=n, q_idx=q_idx, k_shift=k_shift)
    return silu_lut_interpolate_oracle(table, i0, frac, q_idx=q_idx)


_GOLDEN_TABLE_ARRAY_RE = re.compile(r"kSiluLutGoldenTable\[\d+\]\s*=\s*\{(.*?)\};", re.S)


def load_golden_table_witness(path) -> tuple[int, ...]:
    """The CANONICAL SIL1 table, parsed as DATA out of the shared witness header
    (`tests/silu_lut_golden_table.h`, D:/SuperSLM) -- not a recode of any construction,
    just the fixed integers `SuperSLM_S3a_WalkingSkeleton_Plan.md` SS11 S3.0 names as
    part of the shared witness set both sides must reproduce."""
    text = Path(path).read_text(encoding="utf-8")
    match = _GOLDEN_TABLE_ARRAY_RE.search(text)
    if match is None:
        raise AssertionError(f"kSiluLutGoldenTable array not found in {path}")
    body = match.group(1)
    tokens = [tok for tok in re.split(r"[,\s]+", body.strip()) if tok]
    return tuple(int(tok) for tok in tokens)


class SiluLutRealVectors:
    """The Laplace-derived real-activation SwiGLU rows, loaded from the shared witness
    binary (`tests/fixtures/silu_lut_real_vectors.bin`, D:/SuperSLM) per the format
    `tests/sslm_silu_lut_real_vectors_fixtures.h` (D:/SuperSLM) documents."""

    __slots__ = ("row_count", "width", "m", "e", "g_codes", "u_codes")

    def __init__(self, row_count, width, m, e, g_codes, u_codes):
        self.row_count = row_count
        self.width = width
        self.m = m
        self.e = e
        self.g_codes = g_codes
        self.u_codes = u_codes

    def row(self, index: int):
        """(m, e, g_codes, u_codes) for one row -- codes as plain Python int lists."""
        start = index * self.width
        end = start + self.width
        return (self.m[index], self.e[index],
                self.g_codes[start:end], self.u_codes[start:end])


def load_silu_lut_real_vectors(path) -> SiluLutRealVectors:
    """Parses 'SLV1': u32 version(=1), u32 row_count, u32 width, u32 reserved(=0),
    int64[row_count] m (LE), int32[row_count] e (LE), int8[row_count*width] g_codes,
    int8[row_count*width] u_codes -- read directly from the shared witness binary, not
    through the C++ loader header (this parser is an independent reading of the same
    documented format, as the format's own comment invites: "this file's format ...
    sufficient to regenerate")."""
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise AssertionError(f"silu-lut real-vectors fixture truncated before header: {path}")
    if data[0:4] != b"SLV1":
        raise AssertionError(f"silu-lut real-vectors fixture bad magic (want SLV1): {path}")
    version, row_count, width, _reserved = struct.unpack_from("<IIII", data, 4)
    if version != 1:
        raise AssertionError(f"silu-lut real-vectors fixture unsupported version: {path}")
    pos = 20
    m = list(struct.unpack_from(f"<{row_count}q", data, pos))
    pos += 8 * row_count
    e = list(struct.unpack_from(f"<{row_count}i", data, pos))
    pos += 4 * row_count
    elems = row_count * width
    g_codes = list(struct.unpack_from(f"<{elems}b", data, pos))
    pos += elems
    u_codes = list(struct.unpack_from(f"<{elems}b", data, pos))
    pos += elems
    return SiluLutRealVectors(row_count, width, m, e, g_codes, u_codes)
