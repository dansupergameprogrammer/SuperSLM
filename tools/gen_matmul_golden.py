"""Emit tests/matmul_golden_pin.h — the S2.5 matmul cross-platform golden hash and the
canonical input set it is taken over (design SuperSLM_matmul_subslot_design-2026-07-20.md
§10 step 5 / §11 item 6).

The hash pinned by this script is computed here, in ARBITRARY-PRECISION PYTHON INTEGER
ARITHMETIC reproducing the design's §5 scalar reference — never by running the C++ kernel
and recording what it printed. The golden is therefore generated FROM the reference, the
same discipline gen_matmul_fixtures.py and gen_intmath_fixtures.py apply, and a C++ path
that agrees with it agrees with an independent computation rather than with itself.

WHY THE INPUTS ARE GENERATED, NOT PINNED AS DATA
The S2.4 golden pinned its input (the SIL1 sigmoid table) as a C++ literal, because that
input was a converter-emitted float64 table and regenerating it with libm at test time
would have made a cross-platform std::exp difference a FALSE determinism break. Matmul has
no such hazard: its inputs are int8 activation and weight codes, and the canonical set
includes a 132,105-element case whose vectors would be megabytes as literals. The inputs
are therefore produced by a PINNED 64-bit LCG — pure uint64 arithmetic, exactly defined by
the standard on every toolchain, calling nothing — mirrored bit-for-bit by the C++ test.
kMatmulGoldenLcgProbe pins the generator's own first bytes, so a generator drift fails as a
distinct, named check instead of masquerading as a kernel determinism break.

WHAT THE CANONICAL SET COVERS (design §11 item 4)
Both real hidden_size values (1536, 960) and both real intermediate_size values (8960,
2560) — all block-aligned at the common SIMD widths, so on their own they never exercise a
remainder — plus two deliberately non-block-aligned synthetic lengths (tails of 5 and 7),
the int8-extremes row at both operands' limits, the in_channels=1 architectural floor, and
one 132,105-deep case. The deep case is the only one that drives the SSE2 accumulator flush
window (src/matmul.cpp's kFlushBlocks trips at 131,072 elements) and the only one carrying
sums outside int32, so without it the golden would certify neither.

The hash covers, per case: every int64 accumulator GemmInt8Accumulate produces (the shipping
dispatch — SSE2 on x64, the scalar reference on arm64), the DotRowScalarRef value for row 0
(so the normative scalar construction is inside the golden even on x64, where the shipping
dispatch never selects it), and NarrowAccumulatorToI32's int32 output for the cases whose
declared MatmulAccumWidth is Int32 per §8's 131,071 bound.

Run from tools/:  python gen_matmul_golden.py
"""

from __future__ import annotations

import hashlib
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "tests" / "matmul_golden_pin.h"

# --- The pinned generator (mirrored exactly in tests/test_main.cpp) -------------------
#
# A 64-bit LCG with the PCG multiplier/increment. All arithmetic is mod 2^64 and every
# step is exactly defined by the C++ standard for uint64_t, so the sequence is identical
# on every toolchain and ISA. Bytes are taken from the HIGH end of the state: an LCG's low
# bits have short periods, and a weak input distribution would weaken what the golden
# exercises.
_MULT = 6364136223846793005
_INCR = 1442695040888963407
_MASK = (1 << 64) - 1

KIND_LCG = 0
KIND_EXTREMES = 1
# Single-signed attainable extremes: every product at the maximal ATTAINABLE magnitude
# 16,256 (activation +/-127 against weight -128) and all of one sign, so the sum grows
# monotonically and leaves int32 at the deep lengths. An LCG fill cannot reach here --
# its mixed signs cancel, which is precisely why the first version of this golden
# certified nothing about the int64 accumulator (Poirot finding 1).
KIND_EXTREME_NEG = 2  # activations +127, weights -128 -> product -16,256
KIND_EXTREME_POS = 3  # activations -127, weights -128 -> product +16,256


class Lcg:
    def __init__(self, seed: int) -> None:
        self.x = seed & _MASK

    def next_byte(self) -> int:
        self.x = (self.x * _MULT + _INCR) & _MASK
        return (self.x >> 56) & 0xFF


def act_code(b: int) -> int:
    """int8 activation code in [-127, 127] — C22's RequantTokenCode never emits -128, and
    matmul.h's contract states that range, so the golden must not feed a code the runtime
    cannot produce."""
    return (b % 255) - 127


def wgt_code(b: int) -> int:
    """int8 weight code over the full [-128, 127] — WGT1 weights carry no such exclusion."""
    return b - 128


# --- The canonical cases -------------------------------------------------------------
#
# (label, seed, in_channels, out_channels, num_tokens, kind, int32_safe)
CASES = [
    ("real_hidden_1536",        1, 1536,   1536, 2, KIND_LCG,      True),
    ("real_hidden_960",         2,  960,    960, 1, KIND_LCG,      True),
    ("real_intermediate_8960",  3, 8960,     64, 1, KIND_LCG,      True),
    ("real_intermediate_2560",  4, 2560,     64, 1, KIND_LCG,      True),
    ("synthetic_tail5_1533",    5, 1533,      8, 2, KIND_LCG,      True),
    ("synthetic_tail7_1535",    9, 1535,      8, 1, KIND_LCG,      True),
    ("floor_in1_out1",          6,    1,      1, 1, KIND_LCG,      True),
    ("int8_extremes_1024",      0, 1024,      4, 1, KIND_EXTREMES, True),
    ("deep_flush_lcg_132105",   8, 132105,    2, 1, KIND_LCG,      False),
    ("deep_beyond_i32_neg",     0, 132105,    1, 1, KIND_EXTREME_NEG, False),
    ("deep_beyond_i32_pos",     0, 132105,    1, 1, KIND_EXTREME_POS, False),
]

# The two single-signed deep cases are the ONLY ones whose sums leave int32:
# 132,105 x 16,256 = 2,147,498,880, which is 15,233 past INT32_MAX and 15,232 past
# INT32_MIN in the other direction. main() asserts this rather than trusting the
# arithmetic here, so the property cannot silently lapse if a length is edited.
_I32_ESCAPE_CASES = {"deep_beyond_i32_neg", "deep_beyond_i32_pos"}


def build_inputs(seed, in_channels, out_channels, num_tokens, kind):
    """Materialize one case's activations and weights. Mirrored exactly in C++."""
    if kind == KIND_EXTREMES:
        acts = [[127 if (k % 2 == 0) else -127 for k in range(in_channels)]
                for _ in range(num_tokens)]
        wgts = [[127 if ((j + k) % 2 == 0) else -128 for k in range(in_channels)]
                for j in range(out_channels)]
        return acts, wgts
    if kind in (KIND_EXTREME_NEG, KIND_EXTREME_POS):
        a = 127 if kind == KIND_EXTREME_NEG else -127
        acts = [[a] * in_channels for _ in range(num_tokens)]
        wgts = [[-128] * in_channels for _ in range(out_channels)]
        return acts, wgts
    # Fill order is load-bearing: all activation rows first (token-major), then all
    # weight rows (output-channel-major), one LCG stream per case.
    g = Lcg(seed)
    acts = [[act_code(g.next_byte()) for _ in range(in_channels)] for _ in range(num_tokens)]
    wgts = [[wgt_code(g.next_byte()) for _ in range(in_channels)] for _ in range(out_channels)]
    return acts, wgts


def dot(a, w):
    """The design §5 scalar reference in arbitrary precision — the independent oracle."""
    return sum(int(x) * int(y) for x, y in zip(a, w))


def i64_le(v: int) -> bytes:
    return (v & ((1 << 64) - 1)).to_bytes(8, "little")


def i32_le(v: int) -> bytes:
    return (v & ((1 << 32) - 1)).to_bytes(4, "little")


def main() -> None:
    h = hashlib.sha256()
    total = 0
    per_case = []
    for label, seed, in_ch, out_ch, tokens, kind, int32_safe in CASES:
        acts, wgts = build_inputs(seed, in_ch, out_ch, tokens, kind)
        blob = bytearray()
        # 1. Every int64 accumulator, row-major [num_tokens, out_channels].
        wide_row0 = None
        for t in range(tokens):
            row = [dot(acts[t], wgts[j]) for j in range(out_ch)]
            if t == 0:
                wide_row0 = row
            for v in row:
                blob += i64_le(v)
        # 2. The normative scalar reference on row 0 / output channel 0.
        blob += i64_le(dot(acts[0], wgts[0]))
        # 3. The narrowed int32 row, for the cases §8's bound declares Int32.
        if int32_safe:
            for v in wide_row0:
                assert -(1 << 31) <= v < (1 << 31), (
                    f"{label}: accumulator {v} does not fit int32 — the case is mislabelled "
                    f"int32_safe and NarrowAccumulatorToI32 would be UB")
                blob += i32_le(v)
        # The int64-range property, asserted rather than trusted: a golden whose every
        # accumulator fits int32 is reproduced bit-for-bit by a kernel that narrows
        # mid-reduction, so it certifies nothing about the int64 accumulate. Checking it
        # here means a future edit to a length or a fill cannot silently take it away.
        if label in _I32_ESCAPE_CASES:
            assert any(v > (1 << 31) - 1 or v < -(1 << 31) for v in wide_row0), (
                f"{label}: every accumulator fits int32, so this case no longer proves the "
                f"int64 range it exists to prove")
        h.update(blob)
        total += len(blob)
        per_case.append((label, len(blob)))

    # The same property stated over the whole set: at least one case must leave int32,
    # or the golden as a whole is passed by a wrapping-int32 accumulator.
    assert _I32_ESCAPE_CASES, "the canonical set has no case that leaves int32"

    digest = h.hexdigest()

    probe = Lcg(1)
    probe_bytes = [probe.next_byte() for _ in range(16)]

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tools/gen_matmul_golden.py. The pinned hash below is computed by that",
        "// script in arbitrary-precision Python integer arithmetic reproducing the design's S5",
        "// scalar reference (SuperSLM_matmul_subslot_design-2026-07-20.md), NOT by recording",
        "// what a C++ build printed -- the golden is generated from the reference, so a matching",
        "// C++ hash is agreement with an independent computation.",
        "//",
        "// Design record: SuperSLM_matmul_subslot_design-2026-07-20.md S10 step 5 / S11 item 6.",
        "// Re-running the generator must reproduce this file byte-for-byte.",
        "#ifndef SUPERSLM_TESTS_MATMUL_GOLDEN_PIN_H",
        "#define SUPERSLM_TESTS_MATMUL_GOLDEN_PIN_H",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "",
        "// kind: 0 = pinned-LCG fill, 1 = alternating int8-extremes pattern, 2 = single-signed",
        "// attainable extremes (act +127, wgt -128), 3 = single-signed the other direction (act",
        "// -127, wgt -128). All four are mirrored in test_main.cpp. Kinds 2 and 3 are the only",
        "// fills whose sums leave int32: an LCG's mixed signs cancel, so a golden without them is",
        "// reproduced bit-for-bit by a kernel that narrows mid-reduction.",
        "struct MatmulGoldenCase {",
        "\tconst char* label;",
        "\tuint64_t seed;",
        "\tsize_t in_channels;",
        "\tsize_t out_channels;",
        "\tsize_t num_tokens;",
        "\tint kind;",
        "\tbool int32_safe;  // S8's 131,071 bound declares MatmulAccumWidth::Int32 for this case",
        "};",
        "",
        "inline constexpr MatmulGoldenCase kMatmulGoldenCases[] = {",
    ]
    for label, seed, in_ch, out_ch, tokens, kind, int32_safe in CASES:
        lines.append(
            f'\t{{"{label}", {seed}ull, {in_ch}, {out_ch}, {tokens}, {kind}, '
            f'{"true" if int32_safe else "false"}}},')
    lines += [
        "};",
        "",
        "// The generator's own first 16 bytes from seed 1. Pinned separately so a drift in the",
        "// LCG mirror fails as its own named check rather than as a kernel determinism break.",
        "inline constexpr uint8_t kMatmulGoldenLcgProbe[16] = {",
        "\t" + ", ".join(str(b) for b in probe_bytes) + ",",
        "};",
        "",
        f"inline constexpr size_t kMatmulGoldenTotalBytes = {total};",
        "",
        "// PINNED golden. A mismatch is a cross-platform/toolchain/ISA determinism break -- or an",
        "// intended construction change, in which case re-run tools/gen_matmul_golden.py",
        "// deliberately and review the diff.",
        f'inline constexpr char kMatmulGoldenHash[] =',
        f'    "{digest}";',
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_MATMUL_GOLDEN_PIN_H",
    ]
    text = "\n".join(lines) + "\n"
    # The emitted header is pure ASCII and is written as UTF-8 with an explicit newline.
    # Both halves are load-bearing for "re-running the generator reproduces this file
    # byte-for-byte": write_text without encoding= uses the platform's locale encoding, so
    # a non-ASCII character would serialize to different bytes on a Windows host (cp1252)
    # than on a Linux one (UTF-8) for an unchanged hash, and would raise outright under a
    # C/POSIX locale. Asserting ASCII here means the claim cannot lapse by someone adding a
    # typographic character to a comment above.
    text.encode("ascii")  # raises UnicodeEncodeError if a non-ASCII character crept in
    OUT.write_text(text, newline="\n", encoding="utf-8")

    print(f"wrote {OUT}")
    for label, n in per_case:
        print(f"  {label:<26} {n:>9} bytes")
    print(f"  {'TOTAL':<26} {total:>9} bytes")
    print(f"pinned hash: {digest}")


if __name__ == "__main__":
    main()
