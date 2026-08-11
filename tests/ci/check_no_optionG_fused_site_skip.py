"""T-1899 (Curie, red suite for T-1894) -- structural-absence checker for
T-1822 design Sec12's "-60 load floor" cell, mechanism (2) of 2:

  "a tree-wide source scan of src/ for the retired symbol name and for any
  conditional wrapping the dynamic-gate call sites, asserted in the build's
  code review, PLUS the T-1898 probe re-run" (design Sec31.2.2's own text).

WHAT THIS CHECKS, AND WHY IT IS TEXT-SCAN, NOT AST. Two independent
population/assertion pairs, on the same "text scan, no AST" discipline this
tree's own check_no_forward_leaf_calls.py already establishes (that module's
own docstring: mirrors tests/test_check_no_pow_operator.py's convention):

  1. RETIRED SYMBOL ABSENCE. No file under `src/` may declare a symbol named
     `kOptionGFusedKvLandingExponentMin` -- T-1891's own spike-only pre-filter
     constant, explicitly retired (not repaired) by the design's own round-3
     fix (D-SLM2384/D-SLM2385/D-SLM2388): "no new production symbol
     comparable to the spike's own kOptionGFusedKvLandingExponentMin exists."
     A build that reintroduces a fused-site-specific floor constant under
     ANY name containing this exact pattern is caught.

  2. NO CONDITIONAL AHEAD OF THE DYNAMIC GATE. `out_magnitude_exceeded_int64`
     (LandingRescale's own dynamic overflow signal, threaded at the fused
     K-landing call sites per design Sec31.2.2) must be checked
     UNCONDITIONALLY -- "on every element, with no condition on
     kv_landing_e_t_k[h] or any other static exponent" (design's own text).
     Flags any `if` whose condition mentions an exponent-shaped identifier
     (`e_t`, `kv_landing_e_t`, `ExponentMin`, or a numeric literal in
     [-128,0), the legal e_t range) within `_SKIP_WINDOW_LINES` lines above a
     line mentioning `out_magnitude_exceeded_int64` -- the shape a
     resurrected early-exit clause (T-1898's own fracture, design Sec31.2.2)
     would take.

MUTATION VITALITY (StandardsDocument.md Sec4: a new structure is validated
against an independently-found population before it is trusted; a check
shown able to fail on an INJECTED fault, not merely to pass on unchanged
input). This module's own test
(tests/ci/test_check_no_optionG_fused_site_skip.py) constructs scratch
fixtures reproducing both faults this checker exists to catch -- never edits
the real src/ tree -- mirroring check_no_forward_leaf_calls.py's own
established convention exactly.

Test-design record: Claude/Curie/t1899-optionG-red-suite-2026-08-11.md.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

_RETIRED_SYMBOL_PATTERN = re.compile(r"\bkOptionGFusedKvLandingExponentMin\b")
_DYNAMIC_GATE_MARKER = "out_magnitude_exceeded_int64"
_EXPONENT_SHAPED = re.compile(
    r"\be_t\b|\bkv_landing_e_t\b|ExponentMin|(?<![\w.])-(?:1[01]\d|12[0-7]|[1-9]?\d)(?![\w.])"
)
_SKIP_WINDOW_LINES = 6  # lines of look-back from an out_magnitude_exceeded_int64 mention


@dataclass(frozen=True)
class RetiredSymbolHit:
    path: str
    line_no: int
    line: str


@dataclass(frozen=True)
class SkipConditionHit:
    path: str
    if_line_no: int
    if_line: str
    gate_line_no: int
    gate_line: str


def find_retired_symbol_uses(path: str) -> List[RetiredSymbolHit]:
    hits: List[RetiredSymbolHit] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if _RETIRED_SYMBOL_PATTERN.search(line):
                hits.append(RetiredSymbolHit(path, i, line.rstrip("\n")))
    return hits


def find_dynamic_gate_skip_conditions(path: str) -> List[SkipConditionHit]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    hits: List[SkipConditionHit] = []
    for i, line in enumerate(lines):
        if _DYNAMIC_GATE_MARKER not in line:
            continue
        gate_line_no = i + 1
        window_start = max(0, i - _SKIP_WINDOW_LINES)
        for j in range(window_start, i):
            candidate = lines[j]
            stripped = candidate.strip()
            if not stripped.startswith("if") and "if (" not in stripped and "if(" not in stripped:
                continue
            if _EXPONENT_SHAPED.search(candidate):
                hits.append(
                    SkipConditionHit(
                        path=path,
                        if_line_no=j + 1,
                        if_line=candidate.rstrip("\n"),
                        gate_line_no=gate_line_no,
                        gate_line=line.rstrip("\n"),
                    )
                )
    return hits


def _default_population(repo_root: str) -> List[str]:
    patterns = [
        os.path.join(repo_root, "src", "**", "*.cpp"),
        os.path.join(repo_root, "src", "**", "*.h"),
        os.path.join(repo_root, "include", "**", "*.h"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def check(repo_root: str, population: Sequence[str] = None) -> Tuple[List[RetiredSymbolHit],
                                                                       List[SkipConditionHit]]:
    files = list(population) if population is not None else _default_population(repo_root)
    retired_hits: List[RetiredSymbolHit] = []
    skip_hits: List[SkipConditionHit] = []
    for path in files:
        retired_hits.extend(find_retired_symbol_uses(path))
        skip_hits.extend(find_dynamic_gate_skip_conditions(path))
    return retired_hits, skip_hits


def main(repo_root: str, population: Sequence[str] = None) -> int:
    retired_hits, skip_hits = check(repo_root, population)
    if not retired_hits and not skip_hits:
        return 0
    for h in retired_hits:
        print(f"{h.path}:{h.line_no}: retired symbol kOptionGFusedKvLandingExponentMin found: "
              f"{h.line}")
    for h in skip_hits:
        print(f"{h.path}:{h.gate_line_no}: out_magnitude_exceeded_int64 mention preceded (line "
              f"{h.if_line_no}) by an exponent-shaped conditional: {h.if_line!r} -- possible "
              f"resurrected early-exit")
    return 1


if __name__ == "__main__":
    import sys

    _repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                os.pardir, os.pardir))
    sys.exit(main(_repo_root))
