#!/usr/bin/env python3
"""Generates sslm_matmul_fixtures.h — the S2.5 matmul Coverage Model red suite's
fixtures (Claude/Curie/superslm-s2.5-matmul-test-design-2026-07-20.md).

Every golden here is computed by ARBITRARY-PRECISION PYTHON INTEGER ARITHMETIC
(Python's native `int` is unbounded) reproducing the scalar reference construction
pinned in SuperSLM_matmul_subslot_design-2026-07-20.md S5 — never by calling any
C++ implementation (there is none yet: src/matmul.cpp is a red-phase stub) and
never hand-computed. This is the independent oracle S5 requires. Composition/
pipeline goldens additionally cross the shipped Python reference `intmath.py`
(the same pinned reference gen_intmath_fixtures.py uses), so the composition-
regression cell (S12 dim 8) is checked against a SECOND independent computation,
not only the matmul oracle agreeing with itself.

Re-running this script must reproduce sslm_matmul_fixtures.h byte-for-byte.

Test-design record: Claude/Curie/superslm-s2.5-matmul-test-design-2026-07-20.md
"""

from __future__ import annotations

import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPIKE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "Wizard", ".claude", "worktrees",
                 "superslm-dev-continue-d0b08e", "Tools", "superslm_spike")
)
_TOOLS_DIR = os.path.dirname(_SPIKE_DIR)
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _TOOLS_DIR)

import intmath as im  # noqa: E402

OUT_PATH = os.path.join(_THIS_DIR, "sslm_matmul_fixtures.h")

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1

_rng = random.Random(20260720)


def rand_activation() -> int:
    # RequantTokenCode's certified output range (design S3): never -128.
    return _rng.randint(-127, 127)


def rand_weight() -> int:
    return _rng.randint(INT8_MIN, INT8_MAX)


def oracle_row(activations: list[int], weights_flat: list[int], in_channels: int,
               out_channels: int) -> list[int]:
    """The design S5 scalar reference, in arbitrary-precision Python ints —
    the independent oracle. Widens nothing (Python ints are already unbounded);
    every multiply-add is exact."""
    assert len(activations) == in_channels
    assert len(weights_flat) == out_channels * in_channels
    out = []
    for j in range(out_channels):
        acc = 0
        row = weights_flat[j * in_channels:(j + 1) * in_channels]
        for a, w in zip(activations, row):
            assert INT8_MIN <= a <= INT8_MAX and INT8_MIN <= w <= INT8_MAX
            acc += a * w
        out.append(acc)
    return out


def fits_int64(v: int) -> bool:
    return INT64_MIN <= v <= INT64_MAX


def wrap_to_int32(v: int) -> int:
    """Two's-complement 32-bit wraparound of an exact integer — the value a
    naive int32 accumulator would (wrongly) produce. Used only to DEMONSTRATE
    that a boundary case is a genuine overflow (wrap != true value), never as
    a golden for the kernel itself (the kernel's own accumulator is int64 and
    must equal the true, unwrapped value)."""
    m = v % (1 << 32)
    return m - (1 << 32) if m >= (1 << 31) else m


# ---------------------------------------------------------------------------
# A — small exactness + int8 extremes on both operands (S12 dim 1/6, S11 item 1)
# ---------------------------------------------------------------------------

row_cases: list[dict] = []


def add_row_case(label: str, in_channels: int, out_channels: int,
                  activations: list[int], weights: list[int]) -> dict:
    expected = oracle_row(activations, weights, in_channels, out_channels)
    for v in expected:
        assert fits_int64(v), f"{label}: oracle sum {v} does not fit int64"
    case = {
        "label": label, "in_channels": in_channels, "out_channels": out_channels,
        "activations": activations, "weights": weights, "expected": expected,
    }
    row_cases.append(case)
    return case


# Small random.
_small_acts = [rand_activation() for _ in range(8)]
_small_wgts = [rand_weight() for _ in range(3 * 8)]
add_row_case("small_random_in8_out3", 8, 3, _small_acts, _small_wgts)

# Extremes crossed: every combination of {+127,-127} x {+127,-128} across a
# short vector, one output channel per weight pattern.
_ext_acts = [127, -127, 127, -127, 127, -127, 127, -127]
_ext_weight_rows = [
    [127, 127, 127, 127, 127, 127, 127, 127],       # act extremes x +127
    [-128, -128, -128, -128, -128, -128, -128, -128],  # act extremes x -128
    [127, -128, 127, -128, 127, -128, 127, -128],   # alternating
    [-128, 127, -128, 127, -128, 127, -128, 127],   # alternating, opposite phase
]
add_row_case("int8_extremes_both_operands", 8, 4, _ext_acts,
             [w for row in _ext_weight_rows for w in row])

# Single-element floor (S12 dim 4 — in_channels/out_channels architectural floor = 1).
add_row_case("floor_in1_out1", 1, 1, [42], [-17])
add_row_case("floor_in1_out1_extremes", 1, 1, [-127], [-128])

# ---------------------------------------------------------------------------
# B — realistic hidden_size, int32-safe regime + composition + float-parity
#     source data (S12 dim 1/4/8/10, S11 items 2/3/5)
# ---------------------------------------------------------------------------

composition_cases: list[dict] = []


def add_composition_case(label: str, in_channels: int, out_channels: int) -> None:
    activations = [rand_activation() for _ in range(in_channels)]
    weights = [rand_weight() for _ in range(out_channels * in_channels)]
    expected_i64 = oracle_row(activations, weights, in_channels, out_channels)
    for v in expected_i64:
        assert INT32_MIN <= v <= INT32_MAX, (
            f"{label}: row value {v} does not fit int32 — this case must stay in the "
            "declared-Int32 regime (design S8, in_channels <= 131,071)"
        )
    golden_i32 = list(expected_i64)  # independently-constructed int32 row (S11 item 3)

    # Full C19-C22 pipeline over the golden int32 row, via the SAME pinned Python
    # reference gen_intmath_fixtures.py uses — a second independent oracle for the
    # composition-regression cell (not merely the matmul oracle agreeing with itself).
    d_prime = im.max_abs_reduce(golden_i32)
    dn, s = im.normalize_scale(d_prime)
    r = im.dynamic_scale_reciprocal(dn)
    codes = [im.requant_token_code(x, r, s) for x in golden_i32]

    composition_cases.append({
        "label": label, "in_channels": in_channels, "out_channels": out_channels,
        "activations": activations, "weights": weights, "golden_i32": golden_i32,
        "d_prime": d_prime, "dn": dn, "s": s, "r": r, "codes": codes,
    })


add_composition_case("hidden_size_1536_qwen2_5_1_5b", 1536, 8)
add_composition_case("hidden_size_960_smollm2_360m", 960, 8)

# ---------------------------------------------------------------------------
# C — accumulator-overflow boundary: the ATTAINABLE transition, forced with the
#     jointly-attainable worst case (S8, S11 item 2, S12 dim 4/6). NOT authored
#     at the conservative width-selection number 131,072 — see design S11 item 2.
# ---------------------------------------------------------------------------

uniform_cases: list[dict] = []


def add_uniform_case(label: str, in_channels: int, act_value: int, wgt_value: int) -> dict:
    assert INT8_MIN <= act_value <= INT8_MAX and INT8_MIN <= wgt_value <= INT8_MAX
    product = act_value * wgt_value
    expected = product * in_channels
    assert fits_int64(expected), f"{label}: {expected} does not fit int64"
    wrapped_i32 = wrap_to_int32(expected)
    case = {
        "label": label, "in_channels": in_channels, "act_value": act_value,
        "wgt_value": wgt_value, "expected": expected, "wrapped_i32": wrapped_i32,
        "overflows_i32": not (INT32_MIN <= expected <= INT32_MAX),
    }
    uniform_cases.append(case)
    return case


# Maximal ATTAINABLE product magnitude: activation in [-127,127], weight in
# [-128,127] -> |127 * -128| = 16256 (design S8's attainable-transition derivation).
_LAST_SAFE = 132104   # 132,104 * 16,256 = 2,147,482,624 <= INT32_MAX
_FIRST_OVERFLOW = 132105  # 132,105 * 16,256 = 2,147,498,880 > INT32_MAX

c1 = add_uniform_case("overflow_boundary_last_safe_positive", _LAST_SAFE, -127, -128)  # product +16256
c2 = add_uniform_case("overflow_boundary_last_safe_negative", _LAST_SAFE, 127, -128)   # product -16256
c3 = add_uniform_case("overflow_boundary_first_overflow_positive", _FIRST_OVERFLOW, -127, -128)
c4 = add_uniform_case("overflow_boundary_first_overflow_negative", _FIRST_OVERFLOW, 127, -128)

assert not c1["overflows_i32"] and not c2["overflows_i32"], (
    "132,104 must NOT overflow int32 under the attainable product — the design's own "
    "'last-safe' claim (S8) failed its own arithmetic"
)
assert c3["overflows_i32"] and c4["overflows_i32"], (
    "132,105 must overflow int32 under the attainable product — the design's own "
    "'first-overflow' claim (S8) failed its own arithmetic"
)
assert c3["wrapped_i32"] != c3["expected"] and c4["wrapped_i32"] != c4["expected"], (
    "the first-overflow cases must have a wrapped-int32 value that DIFFERS from the true "
    "sum -- otherwise this boundary is not actually forcing a wrap and is silently too weak "
    "(design S11 item 2's own warning about the 131,072 hole)"
)

# The width-SELECTION number (131,071/131,072) is explicitly NOT the overflow-test
# boundary (design S11 item 2) -- documented here, not authored as a wrap-forcing
# cell, so a future reader does not reintroduce the too-weak cell at this number.
_conservative_would_be_safe = add_uniform_case(
    "width_selection_number_131072_does_not_wrap_documentation_only", 131072, -127, -128)
assert not _conservative_would_be_safe["overflows_i32"], (
    "131,072 with the ATTAINABLE product must NOT overflow int32 -- this is the exact "
    "gap the delta-audit found (design S14 amendment 2); if this ever fails, the design's "
    "own width-selection/overflow-test decoupling claim is wrong"
)

# D — deep int64 exactness, well beyond the overflow transition (S11 item 2 "at
# least one deep in_channels ... for int64 exactness").
add_uniform_case("deep_int64_exactness_ordinary_product_1e6", 1_000_000, 37, -53)
add_uniform_case("deep_int64_exactness_extreme_product_5e6", 5_000_000, -127, -128)

# ---------------------------------------------------------------------------
# F — SIMD-saturation hazard: a per-group event independent of S8's regime
#     (S7, S12 dim 6). Symmetric 127x127=16129 matches the design's own
#     illustrative derivation (127x127x3 = 48,387 > INT16_MAX).
# ---------------------------------------------------------------------------

add_uniform_case("simd_saturation_hazard_single_group_3elem", 3, 127, 127)
add_uniform_case("simd_saturation_hazard_ordinary_dimension_1536", 1536, 127, 127)

# ---------------------------------------------------------------------------
# E — tail-length / shape matrix (S12 dim 4, S11 item 4): real intermediate_size
#     (defensive, block-aligned) + deliberately non-block-aligned synthetic
#     lengths (load-bearing).
# ---------------------------------------------------------------------------

add_composition_case("intermediate_size_8960_qwen2_5_1_5b", 8960, 4)
add_composition_case("intermediate_size_2560_smollm2_360m", 2560, 4)
add_composition_case("synthetic_unaligned_length_777", 777, 3)
add_composition_case("synthetic_unaligned_length_4095", 4095, 3)

# ---------------------------------------------------------------------------
# G — row independence + prefill/decode multi-row (S12 dim 7, S11 item 3's
#     discrimination cell; S3's row-major out_acc layout).
# ---------------------------------------------------------------------------

multi_row_cases: list[dict] = []


def make_multi_row_case(label: str, num_tokens: int, in_channels: int, out_channels: int,
                         weights: list[int], row_activations: list[list[int]]) -> dict:
    assert len(row_activations) == num_tokens
    activations_flat: list[int] = []
    expected_flat: list[int] = []
    for row in row_activations:
        activations_flat.extend(row)
        expected_flat.extend(oracle_row(row, weights, in_channels, out_channels))
    case = {
        "label": label, "num_tokens": num_tokens, "in_channels": in_channels,
        "out_channels": out_channels, "weights": weights,
        "activations": activations_flat, "expected": expected_flat,
    }
    multi_row_cases.append(case)
    return case


_indep_weights = [rand_weight() for _ in range(5 * 16)]
_indep_rows_base = [[rand_activation() for _ in range(16)] for _ in range(4)]
make_multi_row_case("row_independence_base", 4, 16, 5, _indep_weights, _indep_rows_base)

_indep_rows_mutated = [list(r) for r in _indep_rows_base]
_indep_rows_mutated[2] = [rand_activation() for _ in range(16)]  # mutate ONLY row 2
assert _indep_rows_mutated[2] != _indep_rows_base[2]
make_multi_row_case("row_independence_row2_mutated", 4, 16, 5, _indep_weights, _indep_rows_mutated)

# ---------------------------------------------------------------------------
# I — warm-object: many independent activation rows against the SAME weight
#     buffer across many calls (S12 dim 1's warm-object requirement).
# ---------------------------------------------------------------------------

_warm_weights = [rand_weight() for _ in range(4 * 16)]
_warm_rows = [[rand_activation() for _ in range(16)] for _ in range(20)]
_warm_expected = [oracle_row(r, _warm_weights, 16, 4) for r in _warm_rows]

# ---------------------------------------------------------------------------
# J — scratch-buffer reuse across a shape change, no stale-byte carryover
#     (S12 dim 1's reuse-across-shape-change requirement).
# ---------------------------------------------------------------------------

_reuse_weights = [rand_weight() for _ in range(4 * 16)]
_reuse_call1_rows = [[rand_activation() for _ in range(16)] for _ in range(6)]
_reuse_call2_rows = [[rand_activation() for _ in range(16)] for _ in range(2)]
# Force rows 0/1 to differ between call1 and call2 so stale carryover is detectable.
while _reuse_call2_rows[0] == _reuse_call1_rows[0]:
    _reuse_call2_rows[0] = [rand_activation() for _ in range(16)]
while _reuse_call2_rows[1] == _reuse_call1_rows[1]:
    _reuse_call2_rows[1] = [rand_activation() for _ in range(16)]
make_multi_row_case("scratch_reuse_call1_num_tokens_6", 6, 16, 4, _reuse_weights, _reuse_call1_rows)
make_multi_row_case("scratch_reuse_call2_num_tokens_2", 2, 16, 4, _reuse_weights, _reuse_call2_rows)

# ---------------------------------------------------------------------------
# H — order/lane-regrouping associativity (S12 dim 7's "any lane order is
#     safe" contract claim; S4).
# ---------------------------------------------------------------------------

_perm_acts = [rand_activation() for _ in range(16)]
_perm_wgts = [rand_weight() for _ in range(16)]
_perm_index = list(range(16))
_rng.shuffle(_perm_index)
_perm_expected = oracle_row(_perm_acts, _perm_wgts, 16, 1)[0]
_perm_acts_reordered = [_perm_acts[i] for i in _perm_index]
_perm_wgts_reordered = [_perm_wgts[i] for i in _perm_index]
_perm_expected_reordered = oracle_row(_perm_acts_reordered, _perm_wgts_reordered, 16, 1)[0]
assert _perm_expected == _perm_expected_reordered, (
    "re-labeling k alongside its paired weight must not change the dot product -- if this "
    "assertion ever fails the permutation was built wrong, not a matmul property"
)

# ---------------------------------------------------------------------------
# Emit the C++ header.
# ---------------------------------------------------------------------------


def cxx_i8(v: int) -> str:
    return f"INT8_C({v})"


def cxx_i32(v: int) -> str:
    return f"INT32_C({v})"


def cxx_i64(v: int) -> str:
    return f"INT64_C({v})"


def cxx_str(s: str) -> str:
    return f'"{s}"'


lines: list[str] = []
emit = lines.append

emit("// GENERATED FILE. Do not hand-edit.")
emit("//")
emit("// Produced by tests/gen_matmul_fixtures.py. Every golden here is computed by")
emit("// arbitrary-precision Python integer arithmetic reproducing the design's scalar")
emit("// reference (SuperSLM_matmul_subslot_design-2026-07-20.md S5), independent of any")
emit("// C++ implementation -- src/matmul.cpp has none yet (red-phase stub). Composition")
emit("// cases additionally cross the pinned Python intmath.py reference. Re-running the")
emit("// generator must reproduce this file byte-for-byte.")
emit("//")
emit("// Test-design record:")
emit("// Claude/Curie/superslm-s2.5-matmul-test-design-2026-07-20.md")
emit("#ifndef SUPERSLM_TESTS_SSLM_MATMUL_FIXTURES_H")
emit("#define SUPERSLM_TESTS_SSLM_MATMUL_FIXTURES_H")
emit("")
emit('#include "superslm/matmul.h"')
emit("")
emit("#include <cstddef>")
emit("#include <cstdint>")
emit("")
emit("namespace superslm_test {")
emit("")

# --- A/B/E: RowCase ---
emit("// --- Row cases: activations/weights -> expected int64 row (design S5 oracle). ---")
emit("")
emit("struct RowCase {")
emit("\tconst char* label;")
emit("\tsize_t in_channels;")
emit("\tsize_t out_channels;")
emit("\tconst int8_t* activations;")
emit("\tconst int8_t* weights;      // out_channels * in_channels, row-major")
emit("\tconst int64_t* expected;    // out_channels")
emit("};")
emit("")
for idx, c in enumerate(row_cases):
    aarr = ", ".join(cxx_i8(v) for v in c["activations"])
    warr = ", ".join(cxx_i8(v) for v in c["weights"])
    earr = ", ".join(cxx_i64(v) for v in c["expected"])
    emit(f"inline constexpr int8_t kRowActs{idx}[] = {{{aarr}}};  // {c['label']}")
    emit(f"inline constexpr int8_t kRowWgts{idx}[] = {{{warr}}};")
    emit(f"inline constexpr int64_t kRowExpected{idx}[] = {{{earr}}};")
emit("")
emit("inline constexpr RowCase kRowCases[] = {")
for idx, c in enumerate(row_cases):
    emit(f"\t{{{cxx_str(c['label'])}, {c['in_channels']}, {c['out_channels']}, "
         f"kRowActs{idx}, kRowWgts{idx}, kRowExpected{idx}}},")
emit("};")
emit(f"inline constexpr size_t kRowCasesCount = {len(row_cases)};")
emit("")

# --- Composition cases ---
emit("// --- Composition cases: matmul row -> already-shipped C19-C22 chain, both the")
emit("//     matmul oracle AND the independent Python intmath.py pipeline oracle")
emit("//     (S12 dim 8, S11 items 2/3/5; also load-bearing for the tail-length S12 dim 4")
emit("//     cells, which reuse this shape). ---")
emit("")
emit("struct CompositionCase {")
emit("\tconst char* label;")
emit("\tsize_t in_channels;")
emit("\tsize_t out_channels;")
emit("\tconst int8_t* activations;")
emit("\tconst int8_t* weights;        // out_channels * in_channels, row-major")
emit("\tconst int32_t* golden_i32;    // out_channels -- independently-constructed row")
emit("\tint64_t expected_d_prime;")
emit("\tint64_t expected_dn;")
emit("\tint expected_s;")
emit("\tint64_t expected_r;")
emit("\tconst int8_t* expected_codes;  // out_channels")
emit("};")
emit("")
for idx, c in enumerate(composition_cases):
    aarr = ", ".join(cxx_i8(v) for v in c["activations"])
    warr = ", ".join(cxx_i8(v) for v in c["weights"])
    garr = ", ".join(cxx_i32(v) for v in c["golden_i32"])
    codesarr = ", ".join(cxx_i8(v) for v in c["codes"])
    emit(f"inline constexpr int8_t kCompActs{idx}[] = {{{aarr}}};  // {c['label']}")
    emit(f"inline constexpr int8_t kCompWgts{idx}[] = {{{warr}}};")
    emit(f"inline constexpr int32_t kCompGolden{idx}[] = {{{garr}}};")
    emit(f"inline constexpr int8_t kCompCodes{idx}[] = {{{codesarr}}};")
emit("")
emit("inline constexpr CompositionCase kCompositionCases[] = {")
for idx, c in enumerate(composition_cases):
    emit(f"\t{{{cxx_str(c['label'])}, {c['in_channels']}, {c['out_channels']}, "
         f"kCompActs{idx}, kCompWgts{idx}, kCompGolden{idx}, "
         f"{cxx_i64(c['d_prime'])}, {cxx_i64(c['dn'])}, {c['s']}, {cxx_i64(c['r'])}, "
         f"kCompCodes{idx}}},")
emit("};")
emit(f"inline constexpr size_t kCompositionCasesCount = {len(composition_cases)};")
emit("")

# --- Uniform cases (overflow boundary, deep, saturation hazard) ---
emit("// --- Uniform cases: in_channels elements, all activation==act_value,")
emit("//     all weight==wgt_value (single output channel). Overflow boundary")
emit("//     (S8/S11 item 2), deep int64 exactness, SIMD-saturation-hazard cells. ---")
emit("")
emit("struct UniformCase {")
emit("\tconst char* label;")
emit("\tsize_t in_channels;")
emit("\tint8_t act_value;")
emit("\tint8_t wgt_value;")
emit("\tint64_t expected;      // the TRUE, unwrapped int64 sum -- always the kernel's contract")
emit("\tint32_t wrapped_i32;   // what a (WRONG) naive int32 accumulator would produce")
emit("\tbool overflows_i32;    // true iff expected does not fit int32 (documentation)")
emit("};")
emit("")
emit("inline constexpr UniformCase kUniformCases[] = {")
for c in uniform_cases:
    emit(f"\t{{{cxx_str(c['label'])}, {c['in_channels']}, {cxx_i8(c['act_value'])}, "
         f"{cxx_i8(c['wgt_value'])}, {cxx_i64(c['expected'])}, {cxx_i32(c['wrapped_i32'])}, "
         f"{'true' if c['overflows_i32'] else 'false'}}},")
emit("};")
emit(f"inline constexpr size_t kUniformCasesCount = {len(uniform_cases)};")
emit("")

# --- Multi-row cases ---
emit("// --- Multi-row cases: num_tokens independent activation rows against ONE weight")
emit("//     matrix (S12 dim 7 row independence; dim 1 scratch-buffer reuse). ---")
emit("")
emit("struct MultiRowCase {")
emit("\tconst char* label;")
emit("\tsize_t num_tokens;")
emit("\tsize_t in_channels;")
emit("\tsize_t out_channels;")
emit("\tconst int8_t* weights;      // out_channels * in_channels")
emit("\tconst int8_t* activations;  // num_tokens * in_channels")
emit("\tconst int64_t* expected;    // num_tokens * out_channels, row-major")
emit("};")
emit("")
for idx, c in enumerate(multi_row_cases):
    warr = ", ".join(cxx_i8(v) for v in c["weights"])
    aarr = ", ".join(cxx_i8(v) for v in c["activations"])
    earr = ", ".join(cxx_i64(v) for v in c["expected"])
    emit(f"inline constexpr int8_t kMultiRowWgts{idx}[] = {{{warr}}};  // {c['label']}")
    emit(f"inline constexpr int8_t kMultiRowActs{idx}[] = {{{aarr}}};")
    emit(f"inline constexpr int64_t kMultiRowExpected{idx}[] = {{{earr}}};")
emit("")
emit("inline constexpr MultiRowCase kMultiRowCases[] = {")
for idx, c in enumerate(multi_row_cases):
    emit(f"\t{{{cxx_str(c['label'])}, {c['num_tokens']}, {c['in_channels']}, {c['out_channels']}, "
         f"kMultiRowWgts{idx}, kMultiRowActs{idx}, kMultiRowExpected{idx}}},")
emit("};")
emit(f"inline constexpr size_t kMultiRowCasesCount = {len(multi_row_cases)};")
emit("")

# --- Warm-object case ---
emit("// --- Warm-object: ONE weight buffer, many independent activation rows called")
emit("//     one-at-a-time (simulating many tokens/resets against a loaded artifact,")
emit("//     S12 dim 1). ---")
emit("")
warr = ", ".join(cxx_i8(v) for v in _warm_weights)
emit(f"inline constexpr int8_t kWarmObjectWeights[] = {{{warr}}};")
emit(f"inline constexpr size_t kWarmObjectInChannels = 16;")
emit(f"inline constexpr size_t kWarmObjectOutChannels = 4;")
emit(f"inline constexpr size_t kWarmObjectRowCount = {len(_warm_rows)};")
emit("")
for idx, row in enumerate(_warm_rows):
    aarr = ", ".join(cxx_i8(v) for v in row)
    emit(f"inline constexpr int8_t kWarmObjectActs{idx}[] = {{{aarr}}};")
emit(f"inline constexpr const int8_t* kWarmObjectActRows[] = {{{', '.join(f'kWarmObjectActs{i}' for i in range(len(_warm_rows)))}}};")
emit("")
for idx, exp in enumerate(_warm_expected):
    earr = ", ".join(cxx_i64(v) for v in exp)
    emit(f"inline constexpr int64_t kWarmObjectExpected{idx}[] = {{{earr}}};")
emit(f"inline constexpr const int64_t* kWarmObjectExpectedRows[] = {{{', '.join(f'kWarmObjectExpected{i}' for i in range(len(_warm_expected)))}}};")
emit("")

# --- Permutation case ---
emit("// --- Order/lane-regrouping associativity (S12 dim 7's contract claim). ---")
emit("")
aarr = ", ".join(cxx_i8(v) for v in _perm_acts)
warr = ", ".join(cxx_i8(v) for v in _perm_wgts)
parr = ", ".join(str(v) for v in _perm_index)
emit(f"inline constexpr int8_t kPermActs[] = {{{aarr}}};")
emit(f"inline constexpr int8_t kPermWgts[] = {{{warr}}};")
emit(f"inline constexpr size_t kPermIndex[] = {{{parr}}};")
emit(f"inline constexpr size_t kPermInChannels = 16;")
emit(f"inline constexpr int64_t kPermExpected = {cxx_i64(_perm_expected)};")
emit("")

emit("}  // namespace superslm_test")
emit("")
emit("#endif  // SUPERSLM_TESTS_SSLM_MATMUL_FIXTURES_H")
emit("")

with open(OUT_PATH, "w", newline="\n") as f:
    f.write("\n".join(lines))

print(f"Wrote {OUT_PATH}")
print(f"Row cases: {len(row_cases)}")
print(f"Composition cases: {len(composition_cases)}")
print(f"Uniform cases: {len(uniform_cases)}")
print(f"Multi-row cases: {len(multi_row_cases)}")
print(f"Warm-object rows: {len(_warm_rows)}")
