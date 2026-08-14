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
values returned from three places -- (a) `RunLayerLoopImpl`'s own FULL body
in `forward_sites.cpp` (from the function's own opening brace to its own
real, brace-matched closing brace -- `extract_function_body`/`cpu_guard_
status_set` below -- with the six per-site arithmetic-rejection statuses
that legitimately live below the entry guards, plus `Ok`, subtracted back
out; corrected 2026-08-14, T-2062, S2, from an earlier cut that stopped at
the first occurrence of `const int64_t position = seq.context_length;`,
a text marker a guard placed below it could silently exit past), (b)
`RunLayerLoopGpu`'s own FULL body in `superslm_gpu.cpp` (the symmetric twin
of (a) -- `extract_function_body`/`gpu_ladder_status_set` below -- with the
one device-capability rejection status that legitimately lives below the
nine-guard ladder subtracted back out; corrected 2026-08-14, T-2069, S3,
from an earlier cut that stopped at the ladder's own `static_assert`
line, the identical class of defect as (a)'s own T-2062 correction, found
one review later by aiming the SAME falsifying mutation at this side
instead), and (c) the `status_name` column of every `SSLM_GPU_LAYER_LOOP_GUARD`
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

CORRECTED 2026-08-14 (T-2062, Claude/Poirot/a3d44e7-gpu-serial-port-ship-
confirmation-review.md, S2; D-SLM3195, superseding whichever prior decision
carried the "closes the class" claim about this module -- left as written
above, not rewritten, per this tree's own append-only discipline): the
paragraph above's own residual statement was honest and incomplete. It
stated one mutation the prior review had actually run and none other. A
SECOND residual existed the day this module shipped and was falsified by
execution one review later: the CPU end anchor,
`const int64_t position = seq.context_length;` (`forward_sites.cpp:1365`),
is the real point where today's nine guards end, but it is a claim about
CODE THAT EXISTS, not a boundary the language enforces -- a tenth guard
placed one line PAST it, with a status new to the set and no `.def` row,
was invisible to this check (`OK`, exit 0), because everything after the
marker was simply never scanned. **Fixed** (the review's own first, class-
closing remedy, chosen over the cheaper "document a second residual"
alternative): the CPU region is no longer marker-truncated. It is now
`RunLayerLoopImpl`'s own FULL body, found by real brace-depth counting from
the function's own opening `{` to its own matching closing `}` (skipping
comments and string/char literals, so no line of source can fool the
depth count) -- never a second text marker guessing where new code will or
will not land. The six per-site arithmetic-rejection statuses (plus `Ok`)
that legitimately live below the old marker --
`CPU_BELOW_GUARD_ARITHMETIC_STATUSES` below, enumerated at the reviewing
seat's own source read and stable across this arc's every round to date --
are subtracted back out, so today's real nine-member set is unchanged and
this check stays green on the unmutated tree. A NEW status appearing
ANYWHERE in the function -- guard region, arithmetic region, or a region
that does not exist yet -- now surfaces in the raw set and is caught,
UNLESS it happens to reuse one of the six subtracted names (folded into the
honest residual below, widened rather than left the same size the fix that
found the gap left it).

CORRECTED 2026-08-14 (T-2069, Claude/Poirot/b543abe-gpu-serial-port-ship-
reverdict-review.md, S3; superseding whichever prior record called the
paragraph above's own fix "class-closing" without qualification -- left as
written above, not rewritten, per this tree's own append-only discipline):
"class-closing" was true of the CPU side and false of the module as a
whole. Part (b) above -- `RunLayerLoopGpu`'s own guard ladder -- carried the
IDENTICAL defect, marker-truncated at its own `static_assert` line, for one
more round: falsified by execution, a new-status GPU-only guard placed past
that marker stayed green (`OK`, exit 0) while the identical guard placed
before it reddened -- the marker, not the guard's own domain, decided the
outcome, exactly the shape S2 was about. **Not a contrived placement**:
`RunLayerLoopGpu` already returns a rejecting status twice below its own
marker (`superslm_gpu.cpp:643, 655`, both device-capability rejections,
`KvPrecisionUnsupported`), and T-2059 added two new enumerators
(`GpuAllocationFailed`/`GpuDeviceRemoved`) for exactly that region three
commits before this fix -- a third device-capability rejection carrying a
new status is the natural next addition, and the marker-truncated check
could not see it. **Fixed, the symmetric twin of the CPU-side remedy**:
`gpu_ladder_status_set` (below) replaces the marker cut with the same real,
brace-matched full-body extraction `cpu_guard_status_set` already uses,
subtracting `GPU_BELOW_LADDER_STATUSES` (one member today,
`KvPrecisionUnsupported`) back out. Verified at source: unchanged on the
unmutated tree. Verified by execution, both directions, against the real
files: the falsifying mutation now reddens; the same guard placed above the
old marker (already caught before this fix, and still caught after it)
confirms the fix changes nothing about the case that already worked. Both
directions are now permanent cells in this module's own test file
(`test_mutation_e_*`, mirroring `test_mutation_d_*`'s own CPU-side pair).

HONEST RESIDUAL, stated here for BOTH sides rather than left to a seventh
review to rediscover the shape of (the failure this finding is about is the
overclaim, not the apparatus, and restating that overclaim about THIS check
would be the same mistake with a seventh author): **a tenth CPU guard, or a
new GPU-side rejection, that returns a status ALREADY in the derived set --
including any of the six `CPU_BELOW_GUARD_ARITHMETIC_STATUSES` names or the
one `GPU_BELOW_LADDER_STATUSES` name, now that both are subtracted rather
than out of scan range entirely -- would not change the set, and this check
would stay green.** The set-equality comparison this module performs cannot
distinguish "this function's own real body has exactly these returns" from
"...plus one more that happens to reuse an existing name" -- reusing an
existing `SslmForwardStatus` across two logically distinct rejections, on
either side, is legal C++ and produces no observable difference to this
check. **Also on both sides**: a status returned through anything other than
a literal `return SslmForwardStatus::X;` (a local variable, a ternary --
`GpuAllocationFailed`/`GpuDeviceRemoved`, T-2059, are returned through
exactly such a ternary at `superslm_gpu.cpp:1276-1277` today) is invisible
to `_STATUS_RETURN_RE` and therefore to this whole module, independent of
which side it is on (recorded, not fixed -- every guard in the tree today
uses the literal form, so the likelihood is low). What this check DOES
close, exactly as `RunLayerLoopGpu`'s own comment claims for the
`static_assert` it replaces in that role: a guard or rejection whose status
is new to the derived set but missing from the `.def` (the review's own
mutation A, run with a status outside the nine, e.g. `RopeTableTensorMissing`)
now fails the build here, wherever in EITHER function's own body it lands --
guard region or arithmetic region, ladder or device-capability check, before
or after either old marker. This module's own test file (`test_check_gpu_
guard_status_parity.py`) reproduces every case named above on both sides --
reddens on a new-status rejection anywhere in either function, stays green
on a same-status one anywhere in either function -- specifically so the
residual above is a measured property of this check, not a claim about it.

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
# T-2062 (S2): retained as a citation of the real, named point in
# `forward_sites.cpp` where the nine entry guards end and the per-token
# forward computation begins -- no longer used as an EXTRACTION boundary
# (see `cpu_guard_status_set` below, which scans the function's own real
# closing brace instead).
CPU_GUARD_REGION_END_MARKER = "const int64_t position = seq.context_length;"
# The distinct `SslmForwardStatus` values `RunLayerLoopImpl` legitimately
# returns BELOW `CPU_GUARD_REGION_END_MARKER` -- one status per per-site
# arithmetic/domain rejection inside the per-layer loop body (never a host-
# visible entry guard), plus `Ok` itself at the function's own end. Named at
# source, not derived: `OptionGWideRopeMagnitudeOutOfDomain` (RoPE rotation
# overflow), `OptionGFusedLandingExponentOutOfDomain` (post-rotation landing
# magnitude), `CarriedScaleMantissaOutOfDomain` (two sites, C26's own
# combine step), `IExpScaleDerivationOutOfDomain` (the per-kv-head i-exp
# derivation), `SoftmaxKernelRefusedAfterGateAccepted` (the softmax kernel's
# own post-gate refusal). Stable across this arc's every round to date
# (T-2062, Claude/Poirot/a3d44e7-gpu-serial-port-ship-confirmation-
# review.md, S2's own remedy) -- a name added to CPU's per-layer body that
# is NOT on this list surfaces as a real set disagreement, exactly like a
# guard would; only these six are treated as "known, non-guard, below-the-
# guards" statuses.
CPU_BELOW_GUARD_ARITHMETIC_STATUSES = frozenset({
    "OptionGWideRopeMagnitudeOutOfDomain",
    "OptionGFusedLandingExponentOutOfDomain",
    "CarriedScaleMantissaOutOfDomain",
    "IExpScaleDerivationOutOfDomain",
    "SoftmaxKernelRefusedAfterGateAccepted",
    "Ok",
})
GPU_FUNC_SIGNATURE = "superslm::SslmForwardStatus RunLayerLoopGpu("
# T-2069 (Claude/Poirot/b543abe-gpu-serial-port-ship-reverdict-review.md,
# S3): retained as a citation of the real, named point in `superslm_gpu.cpp`
# where the nine-guard ladder ends and the two device-capability rejections
# begin -- no longer used as an EXTRACTION boundary (see
# `gpu_ladder_status_set` below, which scans the function's own real closing
# brace instead, exactly as `cpu_guard_status_set` already does for the CPU
# side -- S3 is the symmetric twin of S2, one file over).
GPU_GUARD_REGION_END_MARKER = "static_assert(static_cast<int>(superslm_gpu::GpuLayerLoopGuard::kCount)"
# The distinct `SslmForwardStatus` values `RunLayerLoopGpu` legitimately
# returns BELOW `GPU_GUARD_REGION_END_MARKER` -- both device-capability
# rejections (`!dev.available` at `superslm_gpu.cpp:643`; the sub-Tier-3
# `MapModelGpuResidencyTierCheck` check at `:654-656`) return the SAME
# status, so this is a one-member set today. Named at source, not derived
# (T-2069, S3's own remedy, symmetric with `CPU_BELOW_GUARD_ARITHMETIC_
# STATUSES` above): a name added below the ladder that is NOT
# `KvPrecisionUnsupported` surfaces as a real set disagreement, exactly like
# a guard would. `GpuAllocationFailed`/`GpuDeviceRemoved` (T-2059) are NOT on
# this list and do not need to be: both are returned through a ternary
# (`superslm_gpu.cpp:1276-1277`), a shape `_STATUS_RETURN_RE` below does not
# match at all (O19, the same casebook) -- neither appears in the raw
# extracted set in the first place, so there is nothing to subtract for
# them. Recorded here rather than silently relied on: a future guard
# returned through a ternary, a local variable, or any shape other than a
# literal `return SslmForwardStatus::X;` is invisible to this WHOLE module,
# on both sides, independent of this constant.
GPU_BELOW_LADDER_STATUSES = frozenset({
    "KvPrecisionUnsupported",
})

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


def find_matching_close_brace(text: str, open_brace_index: int) -> int:
    """The index of the `}` that matches the `{` at `open_brace_index` in
    `text`, found by real depth counting over CODE only -- a `//` line
    comment, a `/* */` block comment, and a `"..."`/`'...'` literal
    (backslash-escaped quotes honored) are all skipped without affecting
    depth, so a brace character inside a comment or a string cannot desync
    the count (T-2062, S2: the defect class this replaces was a text-marker
    boundary that could not see past itself at all; a naive brace counter
    that trusted every `{`/`}` byte would trade that gap for a narrower one
    a stray brace in a comment could still open)."""
    if text[open_brace_index] != "{":
        raise ValueError(f"find_matching_close_brace: text[{open_brace_index}] is not '{{'")
    depth = 0
    i = open_brace_index
    n = len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if two == "/*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            if depth == 0:
                return i
            i += 1
            continue
        i += 1
    raise ValueError(f"find_matching_close_brace: no matching '}}' found for '{{' at index {open_brace_index}")


def extract_function_body(text: str, signature: str, *, label: str) -> str:
    """The full body of the function whose signature (a unique source-text
    anchor) is `signature` -- from immediately after its own opening `{` to
    immediately before its own matching closing `}`, found by real brace-
    depth counting from the signature's own first `{` (`find_matching_close_
    brace` above), never a second text marker. T-2062 (S2): this is what
    replaced the CPU side's own marker-truncated extraction -- a marker
    describes where the guards end IN CODE THAT EXISTS; a function's own
    closing brace is where the language says the function ends, and nothing
    added below a marker and above the real close can go unseen."""
    start_at = text.find(signature)
    if start_at == -1:
        raise ValueError(f"{label}: signature not found: {signature!r}")
    open_brace = text.find("{", start_at)
    if open_brace == -1:
        raise ValueError(f"{label}: no opening brace found after signature {signature!r}")
    close_brace = find_matching_close_brace(text, open_brace)
    return text[open_brace + 1:close_brace]


def cpu_guard_status_set(
    cpu_text: str,
    below_guard_arithmetic_statuses: frozenset[str] = CPU_BELOW_GUARD_ARITHMETIC_STATUSES,
) -> set[str]:
    """The CPU guard range's own comparable status set (T-2062, S2): every
    distinct `SslmForwardStatus` `RunLayerLoopImpl` returns ANYWHERE in its
    own real body (`extract_function_body`, real brace boundaries -- not the
    marker `cpu_guard_region_text` used to truncate at), with
    `below_guard_arithmetic_statuses` subtracted back out. A name that is
    new to the function and not on that named, source-derived list surfaces
    in the returned set regardless of where in the function it appears --
    the property `cpu_guard_region_text`'s own marker truncation did not
    have."""
    body = extract_function_body(cpu_text, CPU_FUNC_SIGNATURE, label="forward_sites.cpp")
    return extract_status_set(body) - below_guard_arithmetic_statuses


def gpu_ladder_region_text(text: str) -> str:
    """Retained for the test file's own generic `extract_region` coverage
    (O21's own observation about `cpu_guard_region_text` applies here too --
    this function is no longer used by `run_all_checks`, only by tests that
    exercise `extract_region` itself as a generic utility)."""
    return extract_region(text, GPU_FUNC_SIGNATURE, GPU_GUARD_REGION_END_MARKER, label="superslm_gpu.cpp")


def gpu_ladder_status_set(
    gpu_text: str,
    below_ladder_statuses: frozenset[str] = GPU_BELOW_LADDER_STATUSES,
) -> set[str]:
    """The GPU ladder's own comparable status set (T-2069, S3 -- the
    symmetric twin of `cpu_guard_status_set`/S2): every distinct
    `SslmForwardStatus` `RunLayerLoopGpu` returns ANYWHERE in its own real
    body (`extract_function_body`, real brace boundaries -- not the marker
    `gpu_ladder_region_text` used to truncate at), with
    `below_ladder_statuses` subtracted back out. A name that is new to the
    function and not on that named, source-derived list surfaces in the
    returned set regardless of where in the function it appears -- the
    property the marker-truncated extraction did not have, falsified by the
    reviewing seat's own executed mutation (a new-status guard placed past
    the old marker stayed green; the identical guard placed before it
    reddened)."""
    body = extract_function_body(gpu_text, GPU_FUNC_SIGNATURE, label="superslm_gpu.cpp")
    return extract_status_set(body) - below_ladder_statuses


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

    cpu_set = cpu_guard_status_set(cpu_text)
    gpu_set = gpu_ladder_status_set(gpu_text)
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
            f"    CPU - .def: {sorted(cpu_set - def_set)}   .def - CPU: {sorted(def_set - cpu_set)}\n"
            # T-2069 (M3): the check's own most likely red is a LEGITIMATE new
            # per-site arithmetic rejection in RunLayerLoopImpl, which reads
            # exactly like guard-count drift above -- named here so the fix
            # is not "invent a .def guard row for something that is not a
            # guard." A status that legitimately lives below the guards/ladder
            # belongs in CPU_BELOW_GUARD_ARITHMETIC_STATUSES or
            # GPU_BELOW_LADDER_STATUSES (tests/ci/check_gpu_guard_status_
            # parity.py), not in gpu_layer_loop_guards.def.
            "    (if the new name above is a legitimate per-site arithmetic/device-capability\n"
            "     rejection, not a guard: add it to CPU_BELOW_GUARD_ARITHMETIC_STATUSES or\n"
            "     GPU_BELOW_LADDER_STATUSES in this module, not to gpu_layer_loop_guards.def)"
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
