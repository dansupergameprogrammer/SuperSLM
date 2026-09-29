"""Emit tests/matmul_tiled_golden_pin.h -- the tiled GEMM's own golden (tiled-matmul plan, slice 1,
coverage cell 6.1).

The existing matmul golden (tools/gen_matmul_golden.py) has num_tokens <= 2 in every case, so it never
reaches the tiled kernel, which starts at 8 tokens. This set does: every case has num_tokens >= 8.
Like that generator, this one computes the hash FROM THE REFERENCE -- the scalar construction's exact
integer sum -- never by recording what a C++ build printed, so a matching C++ hash is agreement with an
independent computation. Uniform cases use the closed form K * a * w; LCG-filled cases use numpy int64
dot products, which are exact here (every |product| <= 16,384 and every |sum| < 2^37, far inside int64).
The LCG and the activation/weight code maps are gen_matmul_golden.py's, mirrored in
tests/test_tiled_gemm.cpp.

What the set is for (plan §9 6.1 and §10):
  - uniform single-signed extremes at K = 132,105 (activation +127 or -127 against weight -128), where a
    tiled kernel with its int32 flush deleted overflows (mutant X1);
  - uniform extremes at K = 262,147, past two flush windows, and one deep case at K = 5,000,000
    (M = 8, N = 16 only), where a flush window widened past 66,052 pairs overflows (X2);
  - LCG fills at K in {17, 1023} (tails, pair order, panel order: X3a-c, X4c, X4e);
  - LCG fills past the first flush window, K in {65,539; 132,105} at M in {8, 9}, N = 33: the only
    cases that kill a lane permutation at the in-loop flush (X9), which every uniform and shallow
    case survives.
M in {8, 9, 33} and N in {16, 33} cross the AVX2 (4 x 16) and AVX-512 (8 x 32) tiles' token and panel
tails.

Run from tools/:  python gen_matmul_tiled_golden.py
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from gen_matmul_golden import (
    KIND_EXTREME_NEG,
    KIND_EXTREME_POS,
    KIND_LCG,
    Lcg,
    act_code,
    wgt_code,
)

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "tests" / "matmul_tiled_golden_pin.h"


def build_cases():
    cases = []  # (label, seed, K, N, M, kind)
    for m in (8, 9, 33):
        for n in (16, 33):
            cases.append((f"uniform_neg_k132105_m{m}_n{n}", 0, 132105, n, m, KIND_EXTREME_NEG))
            cases.append((f"uniform_pos_k132105_m{m}_n{n}", 0, 132105, n, m, KIND_EXTREME_POS))
            cases.append((f"uniform_neg_k262147_m{m}_n{n}", 0, 262147, n, m, KIND_EXTREME_NEG))
            cases.append((f"uniform_pos_k262147_m{m}_n{n}", 0, 262147, n, m, KIND_EXTREME_POS))
    cases.append(("uniform_neg_k5000000_m8_n16", 0, 5000000, 16, 8, KIND_EXTREME_NEG))
    seed = 100
    for m in (8, 9, 33):
        for n in (16, 33):
            for k in (17, 1023):
                seed += 1
                cases.append((f"lcg_k{k}_m{m}_n{n}", seed, k, n, m, KIND_LCG))
    for m in (8, 9):
        for k in (65539, 132105):
            seed += 1
            cases.append((f"lcg_past_window_k{k}_m{m}_n33", seed, k, 33, m, KIND_LCG))
    return cases


CASES = build_cases()


def lcg_fill(seed, count, code):
    g = Lcg(seed)
    return [code(g.next_byte()) for _ in range(count)]


def case_outputs(seed, k, n, m, kind):
    """Row-major [m][n] exact int64 sums, as Python ints."""
    if kind in (KIND_EXTREME_NEG, KIND_EXTREME_POS):
        a = 127 if kind == KIND_EXTREME_NEG else -127
        v = k * a * -128
        return [v] * (m * n)
    # One LCG stream per case: all activation rows first (token-major), then all weight rows.
    g = Lcg(seed)
    acts = np.array([act_code(g.next_byte()) for _ in range(m * k)], dtype=np.int64).reshape(m, k)
    wgts = np.array([wgt_code(g.next_byte()) for _ in range(n * k)], dtype=np.int64).reshape(n, k)
    out = acts @ wgts.T
    return [int(x) for x in out.reshape(-1)]


def i64_le(v: int) -> bytes:
    return (v & ((1 << 64) - 1)).to_bytes(8, "little")


def main() -> None:
    h = hashlib.sha256()
    total = 0
    escaped = 0
    for label, seed, k, n, m, kind in CASES:
        vals = case_outputs(seed, k, n, m, kind)
        if any(v > (1 << 31) - 1 or v < -(1 << 31) for v in vals):
            escaped += 1
        blob = b"".join(i64_le(v) for v in vals)
        h.update(blob)
        total += len(blob)
        print(f"  {label:<34} {len(blob):>7} bytes")
    # The uniform deep cases must really leave int32, or they certify nothing about the int64 flush.
    assert escaped >= 25, f"only {escaped} cases leave int32"
    digest = h.hexdigest()

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tools/gen_matmul_tiled_golden.py from the scalar reference's exact integer sums",
        "// (closed form for uniform fills, exact int64 dot products for LCG fills), never from a C++",
        "// build's output. Tiled-matmul plan slice 1, coverage cell 6.1. Re-running the generator must",
        "// reproduce this file byte-for-byte.",
        "#ifndef SUPERSLM_TESTS_MATMUL_TILED_GOLDEN_PIN_H",
        "#define SUPERSLM_TESTS_MATMUL_TILED_GOLDEN_PIN_H",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "",
        "// kind: 0 = pinned-LCG fill (activation rows, then weight rows, one stream), 2 = activation +127",
        "// against weight -128 everywhere, 3 = activation -127 against weight -128 everywhere -- the kinds",
        "// of matmul_golden_pin.h, mirrored in tests/test_tiled_gemm.cpp.",
        "struct MatmulTiledGoldenCase {",
        "\tconst char* label;",
        "\tuint64_t seed;",
        "\tsize_t in_channels;",
        "\tsize_t out_channels;",
        "\tsize_t num_tokens;",
        "\tint kind;",
        "};",
        "",
        "inline constexpr MatmulTiledGoldenCase kMatmulTiledGoldenCases[] = {",
    ]
    for label, seed, k, n, m, kind in CASES:
        lines.append(f'\t{{"{label}", {seed}ull, {k}, {n}, {m}, {kind}}},')
    lines += [
        "};",
        "",
        f"inline constexpr size_t kMatmulTiledGoldenTotalBytes = {total};",
        "",
        "// PINNED golden: SHA-256 over every case's int64 outputs, row-major, little-endian, in case order.",
        "inline constexpr char kMatmulTiledGoldenHash[] =",
        f'    "{digest}";',
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_MATMUL_TILED_GOLDEN_PIN_H",
    ]
    text = "\n".join(lines) + "\n"
    text.encode("ascii")
    OUT.write_text(text, newline="\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(CASES)} cases, {total} bytes, {escaped} leave int32")
    print(f"pinned hash: {digest}")


if __name__ == "__main__":
    main()
