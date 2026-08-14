"""CI source check: the GPU-serial port's guard-count "structural closure"
claim (`include/superslm/gpu_layer_loop_guards.def`, `gpu_port.h:91-103`,
`superslm_gpu.cpp:544-553`) is itself checked against `forward_sites.cpp`,
not merely restated three ways.

WHY THIS EXISTS (T-2055, Claude/Poirot/db73b22-gpu-serial-port-final-
confirmation-review.md, P1). T-2052 built `gpu_layer_loop_guards.def` plus a
`static_assert(GpuLayerLoopGuard::kCount == 9)` on both the `.def`-generated
enum and the pin round's own table-walk cell, and FOUR records then claimed
this closed CPU-guard-count drift as a class rather than a fourth hand-count
(the `.def` file's own header, `gpu_port.h:91-103`, `superslm_gpu.cpp:544-
553`, and build log §17.2). It does not: **nothing in that apparatus reads
`forward_sites.cpp`.** The `.def` file is a hand-written mirror of one
reading of CPU's own ladder; `kCount` is generated from the mirror, not from
CPU; both `static_assert`s compare `kCount` against the literal `9`; the
table-walk cell hand-writes nine fixtures whose labels were matched to the
`.def`'s row order by eye. Falsified by execution (the review's own
"mutation A"): adding a tenth guard to BOTH `RunLayerLoopImpl` and
`RunLayerLoopGpu`, with no matching `.def` row, left the build green and the
suite at 33870/3, unchanged in every count -- neither `static_assert` fired,
no cell moved.

WHAT THIS CHECK ACTUALLY DOES. Extracts the DISTINCT `SslmForwardStatus`
values returned from three places -- (a) `RunLayerLoopImpl`'s own entry-
guard range in `forward_sites.cpp` (from the function's opening brace to the
first occurrence of `const int64_t position = seq.context_length;`, the
real, named point in that source where the guards end and the per-token
forward computation begins -- not a line-number range, so a guard inserted
anywhere in that span is still seen), (b) `RunLayerLoopGpu`'s own guard
ladder in `superslm_gpu.cpp` (from the function's opening brace to its own
`static_assert(static_cast<int>(superslm_gpu::GpuLayerLoopGuard::kCount)`
line), and (c) the `status_name` column of every `SSLM_GPU_LAYER_LOOP_GUARD`
row in `gpu_layer_loop_guards.def` -- and asserts all three SETS are equal.
Verified at source, today: all three are the same nine members
(`InvalidLayerBudget`, `InvalidContextCap`, `HeadDimGeometryMismatch`,
`KvHeadGeometryMismatch`, `WorkspaceTooSmall`, `InvalidHiddenCodes`,
`SequenceAlreadyComplete`, `PositionOverCap`, `KvCapacityExhausted`), so this
check is live and green from its first run, not a check that starts by
skipping itself.

A second, independent check pairs with the first: every `.def` row's own
`cpp_citation` (a `"file.cpp:line"` or `"file.cpp:line-line"` string, the
`.def` file's own third macro argument) must still resolve, at the cited
line(s), to a rejecting `return SslmForwardStatus::<status_name>` statement
in the cited file -- catching a citation left stale by an unrelated edit
that shifts line numbers, independent of whether the STATUS SET check above
would also have caught the same drift.

HONEST RESIDUAL, stated here rather than left to a fifth review to
rediscover the shape of (the failure this finding is about is the
overclaim, not the apparatus, and restating that overclaim about THIS check
would be the same mistake with a fifth author): **a tenth CPU guard that
returns a status ALREADY in the nine-member set would not change the set,
and this check would stay green.** The set-equality comparison this module
performs cannot distinguish "CPU's guard range has exactly these nine
returns" from "CPU's guard range has these nine returns, plus a tenth that
happens to reuse one of them" -- reusing an existing `SslmForwardStatus`
across two logically distinct guards is legal C++ and produces no
observable difference to this check. What this check DOES close, exactly as
`RunLayerLoopGpu`'s own comment claims for the `static_assert` it replaces
in that role: a guard whose status is new to CPU's guard range but missing
from the `.def` (the review's own mutation A, run with a status outside the
nine, e.g. `RopeTableTensorMissing`) now fails the build here, where nothing
previously caught it. This module's own test file (`test_check_gpu_guard_
status_parity.py`) reproduces both cases -- reddens on a new-status tenth
guard, stays green on a same-status one -- specifically so the residual
above is a measured property of this check, not a claim about it.

Modelled on this tree's own established CI-source-check convention
(`tests/ci/check_no_forward_leaf_calls.py`, `tests/ci/check_checked_chain_
funnel_position_cap_not_a_stub.py`): a text scan over the real source with
comments stripped first (unlike a leaf-name ban, this module counts DISTINCT
NAMES rather than merely detecting presence, so a status name mentioned only
in a comment must not inflate the derived set -- verified by this module's
own test file), two-step CI wiring (`python -m pytest tests/ci/test_check_
gpu_guard_status_parity.py -v` then `python tests/ci/check_gpu_guard_status_
parity.py`, `.github/workflows/tests.yml`), no line-number-hardcoded region
(the leaf-ban check's own "glob the whole tree" answer to the identical
class of fragility, applied here as "anchor on real source text instead of
a line count").
"""
from __future__ import annotations

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

FORWARD_SITES_CPP = os.path.join(_REPO_ROOT, "src", "forward", "forward_sites.cpp")
SUPERSLM_GPU_CPP = os.path.join(_REPO_ROOT, "src", "gpu", "superslm_gpu.cpp")
GUARDS_DEF = os.path.join(_REPO_ROOT, "include", "superslm", "gpu_layer_loop_guards.def")

# --- Real, named anchors -- source TEXT, never a line number. ---
CPU_FUNC_SIGNATURE = "static SslmForwardStatus RunLayerLoopImpl("
CPU_GUARD_REGION_END_MARKER = "const int64_t position = seq.context_length;"
GPU_FUNC_SIGNATURE = "superslm::SslmForwardStatus RunLayerLoopGpu("
GPU_GUARD_REGION_END_MARKER = "static_assert(static_cast<int>(superslm_gpu::GpuLayerLoopGuard::kCount)"

_STATUS_RETURN_RE = re.compile(r"return\s+(?:superslm::)?SslmForwardStatus::([A-Za-z_][A-Za-z0-9_]*)")
_DEF_ROW_RE = re.compile(
    r"SSLM_GPU_LAYER_LOOP_GUARD\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\"([^\"]*)\"\s*\)"
)
# A `.def` citation's leading "file.ext:line" or "file.ext:line-line" prefix
# -- the parenthetical annotation some rows carry after it (WorkspaceSizeOr
# Overflow's own dual-status row) is not part of the location.
_CITATION_RE = re.compile(r"^([A-Za-z0-9_./\\]+):(\d+)(?:-(\d+))?")


def strip_comments(text: str) -> str:
    """Removes `//...` and `/*...*/` C++ comments. Not a tokenizer -- this
    tree's guard regions and ladder contain no string or char literal that
    itself carries `//` or `/*` (verified by inspection of both real
    regions this module scans), so a naive strip is exact here, not merely
    an accepted over-approximation the way `check_no_forward_leaf_calls.py`'s
    ban-not-classifier scan is. Kept anyway, deliberately, because this
    module counts DISTINCT NAMES: an uncommented false positive would
    silently enlarge a derived set and could hide a real mismatch behind an
    accidental match, which is a soundness gap for a comparison in a way it
    is not for a presence ban."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i + 2] == "//":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if text[i:i + 2] == "/*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def extract_region(text: str, start_marker: str, end_marker: str, *, label: str) -> str:
    """The substring of `text` from the END of `start_marker`'s first
    occurrence to the START of `end_marker`'s first occurrence AFTER it --
    real source text delimits both ends, so a guard inserted anywhere in
    between is inside the extracted region regardless of which line it
    lands on."""
    start_at = text.find(start_marker)
    if start_at == -1:
        raise ValueError(f"{label}: start marker not found: {start_marker!r}")
    start_at += len(start_marker)
    end_at = text.find(end_marker, start_at)
    if end_at == -1:
        raise ValueError(f"{label}: end marker not found after start: {end_marker!r}")
    return text[start_at:end_at]


def extract_status_set(region_text: str) -> set[str]:
    """Every DISTINCT `SslmForwardStatus` name a `return` statement in
    `region_text` names, comments stripped first."""
    stripped = strip_comments(region_text)
    return set(_STATUS_RETURN_RE.findall(stripped))


def cpu_guard_region_text(text: str) -> str:
    return extract_region(text, CPU_FUNC_SIGNATURE, CPU_GUARD_REGION_END_MARKER, label="forward_sites.cpp")


def gpu_ladder_region_text(text: str) -> str:
    return extract_region(text, GPU_FUNC_SIGNATURE, GPU_GUARD_REGION_END_MARKER, label="superslm_gpu.cpp")


class DefRow:
    __slots__ = ("enum_name", "status_name", "citation")

    def __init__(self, enum_name: str, status_name: str, citation: str) -> None:
        self.enum_name = enum_name
        self.status_name = status_name
        self.citation = citation

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics only
        return f"DefRow({self.enum_name!r}, {self.status_name!r}, {self.citation!r})"


def parse_def_rows(def_text: str) -> list[DefRow]:
    """Every `SSLM_GPU_LAYER_LOOP_GUARD(enum_name, status_name, "citation")`
    invocation in `def_text`, in file order -- the `#define`/`#error`/
    `#undef` lines around them do not match this pattern and are ignored."""
    stripped = strip_comments(def_text)
    return [DefRow(m.group(1), m.group(2), m.group(3)) for m in _DEF_ROW_RE.finditer(stripped)]


def def_status_set(rows: list[DefRow]) -> set[str]:
    return {row.status_name for row in rows}


def check_citation(row: DefRow, repo_root: str = _REPO_ROOT) -> str | None:
    """None if `row.citation`'s cited file:line(s) hold a rejecting `return
    SslmForwardStatus::<row.status_name>` somewhere in the cited (inclusive)
    line range; else a one-line failure description. A relative citation
    path is resolved against `src/forward/` first (every citation in the
    real `.def` file today names `forward_sites.cpp` bare, not a path), then
    against `repo_root` itself, so this stays correct if a future guard
    cites a file elsewhere in the tree."""
    m = _CITATION_RE.match(row.citation)
    if not m:
        return f"{row.enum_name}: citation {row.citation!r} does not start with a file:line prefix"
    cited_file, lo_s, hi_s = m.group(1), m.group(2), m.group(3)
    lo = int(lo_s)
    hi = int(hi_s) if hi_s else lo
    candidates = [
        os.path.join(repo_root, "src", "forward", cited_file),
        os.path.join(repo_root, cited_file),
    ]
    real_path = next((p for p in candidates if os.path.isfile(p)), None)
    if real_path is None:
        return f"{row.enum_name}: cited file {cited_file!r} not found under {repo_root}"
    with open(real_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if hi > len(lines):
        return f"{row.enum_name}: cited range {lo}-{hi} exceeds {cited_file}'s own {len(lines)} lines"
    window = "".join(lines[lo - 1:hi])
    needle_re = re.compile(r"return\s+(?:superslm::)?SslmForwardStatus::" + re.escape(row.status_name) + r"\b")
    if not needle_re.search(strip_comments(window)):
        return (
            f"{row.enum_name}: {cited_file}:{lo}"
            + (f"-{hi}" if hi != lo else "")
            + f" does not hold a rejecting 'return SslmForwardStatus::{row.status_name}'"
        )
    return None


def run_all_checks(
    forward_sites_path: str = FORWARD_SITES_CPP,
    superslm_gpu_path: str = SUPERSLM_GPU_CPP,
    guards_def_path: str = GUARDS_DEF,
    repo_root: str = _REPO_ROOT,
) -> list[str]:
    """Every failure this module can report against the three real files, or
    an empty list if all three status sets agree and every `.def` citation
    resolves. Raises `ValueError` (not a failure list) if a source anchor
    itself is missing -- that is an environment/drift problem this check's
    own extraction logic cannot route around, matching `check_no_forward_
    leaf_calls.py`'s own ClangUnavailable precedent of raising rather than
    reporting a false pass."""
    with open(forward_sites_path, "r", encoding="utf-8") as f:
        cpu_text = f.read()
    with open(superslm_gpu_path, "r", encoding="utf-8") as f:
        gpu_text = f.read()
    with open(guards_def_path, "r", encoding="utf-8") as f:
        def_text = f.read()

    cpu_set = extract_status_set(cpu_guard_region_text(cpu_text))
    gpu_set = extract_status_set(gpu_ladder_region_text(gpu_text))
    def_rows = parse_def_rows(def_text)
    def_set = def_status_set(def_rows)

    failures: list[str] = []
    if cpu_set != gpu_set or cpu_set != def_set or gpu_set != def_set:
        failures.append(
            "guard status sets disagree:\n"
            f"    CPU (forward_sites.cpp guard range): {sorted(cpu_set)}\n"
            f"    GPU (RunLayerLoopGpu ladder):         {sorted(gpu_set)}\n"
            f"    .def (gpu_layer_loop_guards.def):     {sorted(def_set)}\n"
            f"    CPU - GPU: {sorted(cpu_set - gpu_set)}   GPU - CPU: {sorted(gpu_set - cpu_set)}\n"
            f"    CPU - .def: {sorted(cpu_set - def_set)}   .def - CPU: {sorted(def_set - cpu_set)}"
        )
    for row in def_rows:
        failure = check_citation(row, repo_root)
        if failure is not None:
            failures.append(failure)
    return failures


def main() -> int:
    try:
        failures = run_all_checks()
    except ValueError as e:
        print(f"check_gpu_guard_status_parity.py: FAILED (source anchor missing) -- {e}", file=sys.stderr)
        return 1
    if failures:
        print("check_gpu_guard_status_parity.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "check_gpu_guard_status_parity.py: OK -- CPU guard range, GPU ladder, and "
        "gpu_layer_loop_guards.def all name the identical status set, and every "
        ".def citation resolves to a rejecting return at its cited source location. "
        "(A tenth guard reusing an existing status would not be caught -- see this "
        "module's own docstring.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
