#!/usr/bin/env python3
"""Tiled-matmul plan slice 1, cell 11.3: the tiled kernel's linkage check.

Why: the tiled kernel's functions carry per-function target attributes (AVX2, AVX-512BW). If one of them
(or a template instantiation of one) had external, weak or COMDAT linkage, the linker could fold it with
a copy compiled without that attribute, or with a different one, in another object -- the silent-SIGILL
class the forced-tier design exists to prevent. So every tiled and packer symbol must be LOCAL.

The population is every symbol of the given matmul.cpp objects whose demangled name contains `Tiled`
(functions and data alike: the kernels, the packer, the activation prep, the constants) or `ProbV` /
`ProbQ15AccumulateInto` (the attention and per-row sites plan's slice S2: the AVX2 and AVX-512BW prob·V
bodies, their guard and the accumulate-into core, cell 11.3 of that plan), plus the
build-configuration record `superslm_build_config_record`. Slice S3 of the same plan gives src/intmath.cpp
its first target-attributed functions, the requant row bodies `RequantRowAvx2` and `RequantRowAvx512`
(`RequantRowAvx` in the population; the exported leaf `RequantRowWide` that dispatches to them carries no
target attribute and is outside it), so the intmath.cpp objects are passed too. Slice S4 adds the softmax
fast path's bodies there: `SoftmaxRowAvx2` / `SoftmaxRowAvx512` and the per-step helpers they inline,
`SoftmaxExpAvx2` / `SoftmaxExpAvx512` and `SoftmaxProbAvx2` / `SoftmaxProbAvx512` (`SoftmaxRowAvx`,
`SoftmaxExpAvx` and `SoftmaxProbAvx` in the population), and their shared, unattributed guard and row
set-up (`SoftmaxFastGuard`, `MakeSoftmaxFastRow`, `SoftmaxProbReciprocal`; `SoftmaxFast` and
`SoftmaxProbReciprocal`); the exported `SoftmaxRowQ15` that dispatches to them is outside it. The engine's public API in the same object
is outside it (including the `superslm::detail::` entries the header declares, which carry no target
attribute), and so are the test seam's own `superslm_test::` variables, which exist only in seam
builds and are shared with the test translation unit on purpose.

Rules:
  1. every population symbol except the record is local: never global, weak, unique or COMDAT;
  2. the record is present exactly once per matmul.cpp object and is external (the single allowed
     exception); any other object (intmath.cpp's) holds no record;
  3. the name list below is not stale: each listed function resolves to a symbol in at least one of the
     given objects, or is proven inlined by its signature instruction in its caller's disassembly (at -O3
     the packer, the micro-kernels and the activation prep are all inlined).

Vitality (run by hand; the plan's plants): a tiled function given external linkage, and a non-static
packer table, must each turn rule 1 red.

Usage: check_tiled_matmul_linkage.py OBJECT [OBJECT...]   (Linux/ELF: nm, readelf, objdump)
       (the matmul.cpp objects are recognized by "matmul" in the object's file name)
Windows (dumpbin /symbols) is not implemented here; see the plan's §11.3 Windows leg.
"""

from __future__ import annotations

import re
import subprocess
import sys

RECORD = "superslm_build_config_record"
POPULATION = re.compile(
    r"Tiled|ProbV|ProbQ15AccumulateInto|RequantRowAvx|SoftmaxRowAvx|SoftmaxExpAvx|SoftmaxProbAvx|SoftmaxFast|"
    r"SoftmaxProbReciprocal|" + RECORD)
SEAM = "superslm_test::"
DETAIL_API = "superslm::detail::"  # declared in include/superslm/matmul.h; never target-attributed

# name -> (caller whose disassembly holds it when inlined, a regex its inlined body must contribute)
EXPECTED = {
    "TiledGemmAvx2": ("GemmInt8AccumulateCols", r"call.*TiledGemmAvx2"),
    "TiledGemmAvx512": ("GemmInt8AccumulateCols", r"call.*TiledGemmAvx512"),
    "TiledMicroAvx2": ("TiledGemmAvx2", r"vpmaddwd\s.*%ymm"),
    "TiledMicroAvx512": ("TiledGemmAvx512", r"vpmaddwd\s.*%zmm"),
    "TiledPackPanel16": ("TiledGemmAvx", r"punpck[lh]qdq"),
    "TiledTranspose8x8Epi16": ("TiledGemmAvx", r"punpck[lh]wd"),
    "TiledWidenActivations": ("GemmInt8AccumulateCols", r"(movsbw|pmovsxbw)"),
    "RunTiledGemm": ("GemmInt8AccumulateCols", r"call.*TiledGemmAvx"),
    # Attention and per-row sites plan, slice S2 (cell 11.3).
    "ProbVAccumulateIntoAvx2": ("GemmProbQ15Accumulate", r"(call|jmp).*ProbVAccumulateIntoAvx2"),
    "ProbVAccumulateIntoAvx512": ("GemmProbQ15Accumulate", r"(call|jmp).*ProbVAccumulateIntoAvx512"),
    "ProbVBlockAvx2": ("ProbVAccumulateIntoAvx2", r"vpmaddwd\s.*%ymm"),
    "ProbVBlockAvx512": ("ProbVAccumulateIntoAvx512", r"vpmaddwd\s.*%zmm"),
    "ProbVTail16Avx512": ("ProbVAccumulateIntoAvx512", r"vinserti128"),
    "ProbQ15AccumulateInto": ("GemmProbQ15Accumulate", r"(call|jmp).*ProbVAccumulateIntoAvx"),
    # Attention and per-row sites plan, slice S3 (cell 11.3): src/intmath.cpp's requant row bodies.
    "RequantRowAvx2": ("RequantRowWide", r"vpmuludq\s.*%ymm"),
    "RequantRowAvx512": ("RequantRowWide", r"vpmuludq\s.*%zmm"),
    # Attention and per-row sites plan, slice S4 (cell 11.3): src/intmath.cpp's softmax fast path.
    "SoftmaxRowAvx2": ("SoftmaxRowQ15", r"(call|jmp).*SoftmaxRowAvx2"),
    "SoftmaxRowAvx512": ("SoftmaxRowQ15", r"(call|jmp).*SoftmaxRowAvx512"),
    "SoftmaxExpAvx2": ("SoftmaxRowAvx2", r"vpsrlvq\s.*%ymm"),
    "SoftmaxExpAvx512": ("SoftmaxRowAvx512", r"vpsrlvq\s.*%zmm"),
    "SoftmaxProbAvx2": ("SoftmaxRowAvx2", r"vpsllq\s+\$0xf,.*%ymm"),
    "SoftmaxProbAvx512": ("SoftmaxRowAvx512", r"vpsllq\s+\$0xf,.*%zmm"),
    "SoftmaxFastGuard": ("SoftmaxRowQ15", r"(0x2000000000000000|0xe000000000000000|0x4000000000000000)"),  # +-2^61
    "MakeSoftmaxFastRow": ("SoftmaxRowQ15", r"(lzcnt|bsr)"),
    "SoftmaxProbReciprocal": ("SoftmaxRowAvx", r"\bdiv"),
}


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def symbols(obj: str) -> list[tuple[str, str]]:
    out = []
    for line in run(["nm", "-C", obj]).splitlines():
        m = re.match(r"^(?:[0-9a-fA-F]+)?\s+([A-Za-z?])\s+(.*)$", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def comdat_signatures(obj: str) -> set[str]:
    sigs = set()
    for line in run(["readelf", "-gW", "-C", obj]).splitlines():
        m = re.search(r"COMDAT group section \[\s*\d+\] `[^']*' \[(.*)\] contains", line)
        if m:
            sigs.add(m.group(1))
    return sigs


def disassembly_by_function(obj: str) -> dict[str, str]:
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in run(["objdump", "-d", "-C", "--no-show-raw-insn", obj]).splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
        elif cur is not None:
            funcs[cur].append(line)
    return {k: "\n".join(v) for k, v in funcs.items()}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    failures = []
    resolved = {name: None for name in EXPECTED}
    for obj in argv[1:]:
        syms = symbols(obj)
        comdat = comdat_signatures(obj)
        records = 0
        pop = 0
        for kind, name in syms:
            if not POPULATION.search(name) or name.startswith(SEAM) or name.startswith(DETAIL_API):
                continue
            pop += 1
            if name == RECORD:
                records += 1
                if kind not in "RDBG":
                    failures.append(f"{obj}: the record has nm type '{kind}', want external data (R/D)")
                continue
            if kind == "U":
                continue  # a reference, not a definition
            if not kind.islower() or kind in "uvw":
                failures.append(f"{obj}: '{name}' is not local (nm type '{kind}')")
            if name in comdat:
                failures.append(f"{obj}: '{name}' is a COMDAT group signature")
            for exp in EXPECTED:
                if re.search(r"(^|::)" + exp + r"\b", name) and resolved[exp] is None:
                    resolved[exp] = f"symbol in {obj}"
        want_records = 1 if "matmul" in obj.rsplit("/", 1)[-1] else 0
        if records != want_records:
            failures.append(f"{obj}: the record appears {records} times, want exactly {want_records}")
        dis = disassembly_by_function(obj)
        for exp, (caller, sig) in EXPECTED.items():
            if resolved[exp] is not None:
                continue
            for fname, body in dis.items():
                if caller in fname and re.search(sig, body):
                    resolved[exp] = f"inlined into {fname.replace('(anonymous namespace)::', '').split('(')[0]} in {obj}"
                    break
        print(f"{obj}: {pop} population symbols, record x{records}")
    for exp, how in resolved.items():
        if how is None:
            failures.append(f"listed name '{exp}' resolves to no symbol and no inlined body (stale list?)")
        else:
            print(f"  {exp}: {how}")
    if failures:
        print("check_tiled_matmul_linkage: FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("check_tiled_matmul_linkage: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
