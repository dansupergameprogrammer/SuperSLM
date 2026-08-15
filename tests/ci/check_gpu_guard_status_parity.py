"""CI source check: the GPU-serial port's guard-count "structural closure"
claim (`include/superslm/gpu_layer_loop_guards.def`, `gpu_port.h`'s own
`GpuLayerLoopGuard` enum, `superslm_gpu.cpp`'s own `static_assert` beside its
guard ladder) is itself checked against `forward_sites.cpp`,
not merely restated three ways.

WHY THIS EXISTS (T-2055, Claude/Poirot/db73b22-gpu-serial-port-final-
confirmation-review.md, P1). T-2052 built `gpu_layer_loop_guards.def` plus a
`static_assert(GpuLayerLoopGuard::kCount == 9)` on both the `.def`-generated
enum and the pin round's own table-walk cell, and FOUR records then claimed
this closed CPU-guard-count drift as a class rather than a fourth hand-count
(the `.def` file's own header, `gpu_port.h`'s own `GpuLayerLoopGuard` enum,
`superslm_gpu.cpp`'s own guard-ladder `static_assert`, and build log §17.2).
It does not: **nothing in that apparatus reads
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
`const int64_t position = seq.context_length;` (`forward_sites.cpp`,
`RunLayerLoopImpl`'s own guard/arithmetic boundary),
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
marker (`dev.available` (`superslm_gpu.cpp`) and
`MapModelGpuResidencyTierCheck` (`superslm_gpu.cpp`), both device-capability
rejections, `KvPrecisionUnsupported`), and T-2059 added two new enumerators
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
exactly such a ternary, `device_removed_reason` (`superslm_gpu.cpp`),
today) is invisible
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

CORRECTED 2026-08-14 (T-2080, Claude/Poirot/94ebee3-gpu-serial-port-closing-
review.md, M1; D-SLM3241): a SEPARATE residual existed in the citation
machinery itself, orthogonal to the two named above -- `GPU_PORT_H_LWUWS_
CITATIONS` (below) was a hand-transcribed MIRROR of `gpu_port.h`'s own
citations, held in this file, and nothing checked the mirror against the
original: corrupting every citation in the header's own text left this
whole module green, since nothing here ever read `gpu_port.h` at all.
Fixed: `parse_gpu_port_h_citation_lines`/`check_gpu_port_h_citations_
match_table` (below) derive the citation POPULATION from `gpu_port.h`'s own
text and assert it equals the table -- an edit to either side that is not
mirrored in the other now fails CI. Proven by execution, on the real
files: corrupting all eleven ladder citations in `gpu_port.h` while leaving
this module's own table untouched -- `OK`, 48 passed before this fix;
`FAILED`, naming the exact eleven-line divergence, after it.

CORRECTED 2026-08-14 (T-2083, Claude/Poirot/42ecf79-gpu-serial-port-
round9-review.md, O34/O35): two further, narrower residuals inside the
citation machinery itself. O35 CLOSED: `GPU_PORT_H_LWUWS_CITATIONS`'s own
`DecodeStickyTag` range citation -- which read `superslm_gpu.cpp:583-601` at
T-2083, quoted here as the historical text of a citation T-2094 has since
DELETED along with the whole table, never as a live location to resolve --
checked two
ENDPOINTS' own content, not what was inside the range -- the range-end
needle, a bare `"}"`, was satisfied by 177 of the file's own 2,023 lines
(executed at T-2083), so a shift onto the wrong closing brace would pass it, and
deleting an interior `case` while preserving the file's own total line
count (executed: `case 13` removed) left both endpoints individually valid
and this module green, 54 passed, while the citation's own claim ("fourteen
statuses, THIRTEEN of them rejecting") became false. Fixed:
`decode_sticky_tag_status_set`/`check_decode_sticky_tag_range` (below)
extract `DecodeStickyTag`'s own real, brace-matched body and count DISTINCT
statuses directly, pinning the exact claim the range citation exists to
protect rather than two points near where it happens to sit.

O34 -- RETIRED as a residual, 2026-08-14 (T-2098,
Claude/Poirot/152035b-gpu-serial-port-substrate-removal-review.md, M2;
D-SLM3291). The paragraph that stood here reasoned, in the present tense,
about `GPU_PORT_H_LWUWS_CITATIONS`, `check_gpu_port_h_citations_match_table`
and a comparison of "SETS of line numbers" -- machinery T-2094 DELETED. Its
own body was a same-position SWAP of two citations' line numbers in
`gpu_port.h`, which left the set unchanged and this module `OK`; there are
no line-number citations in `gpu_port.h` any more and no set to compare, so
the defect it described cannot occur and the specification it carried
(a parser associating each citation with its own dated-correction
sub-section) is work nobody now owes. Left as a dated retirement rather
than deleted, per this tree's own append-only discipline: a reader who finds
O34 cited in an older record needs to land somewhere that says what became
of it.

THE SUCCESSOR RESIDUAL, which is O34's own defect generalized and is NOT
closed. `check_marked_citations` below checks a citation for EXISTENCE and
never for correspondence, so it cannot distinguish a correct citation from a
citation to a DIFFERENT REAL THING. Executed (T-2098, MUT-O): in
`gpu_port.h`'s own ladder enumeration, `InvalidLayerBudget`
(`superslm_gpu.cpp`) swapped for `WorkspaceTooSmall` (`superslm_gpu.cpp`) --
a citation now naming the wrong guard, both symbols real -- left this module
`OK`, exit 0. O34 was "a swap of two POSITIONS is invisible"; this is "a
swap of two LIVE SYMBOLS is invisible," the same class one level out, and it
is the price the existence-not-position convention is bought at. It is named
here, beside `check_marked_citations`'s own true claim that there is no
boundary a citation can sit outside of, because that claim and this residual
are about different axes and reading the first as covering the second is
what would make this module look more closed than it is: the SCAN is
complete and the RESOLUTION is shallow.

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
GPU_PORT_H = os.path.join(_REPO_ROOT, "include", "superslm", "gpu_port.h")
BUILD_BAT = os.path.join(_REPO_ROOT, "build.bat")

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
# rejections (`dev.available` (`superslm_gpu.cpp`); the sub-Tier-3
# `MapModelGpuResidencyTierCheck` (`superslm_gpu.cpp`) check) return the SAME
# status, so this is a one-member set today. Named at source, not derived
# (T-2069, S3's own remedy, symmetric with `CPU_BELOW_GUARD_ARITHMETIC_
# STATUSES` above): a name added below the ladder that is NOT
# `KvPrecisionUnsupported` surfaces as a real set disagreement, exactly like
# a guard would. `GpuAllocationFailed`/`GpuDeviceRemoved` (T-2059) are NOT on
# this list and do not need to be: both are returned through a ternary
# (`device_removed_reason` (`superslm_gpu.cpp`)), a shape `_STATUS_RETURN_RE`
# below does not
# match at all (O19, the same casebook) -- neither appears in the raw
# extracted set in the first place, so there is nothing to subtract for
# them. Recorded here rather than silently relied on: a future guard
# returned through a ternary, a local variable, or any shape other than a
# literal `return SslmForwardStatus::X;` is invisible to this WHOLE module,
# on both sides, independent of this constant.
GPU_BELOW_LADDER_STATUSES = frozenset({
    "KvPrecisionUnsupported",
    # T-2101 (S3-prime, code review 6d9e04e-t2101-gpu-throughput-review.md,
    # confirmation pass @ f7026db): the multi-group GEMM dispatch's own
    # standing group-arithmetic guard (`ComputeGpuGemmSiteGroupPlan`,
    # `superslm_gpu.cpp`) -- a per-site arithmetic rejection checked well below
    # the nine-guard ladder's own closing `static_assert`, on the SAME per-site
    # dimensions every real call carries, not a tenth pre-flight guard. Returned
    # via a literal `return superslm::SslmForwardStatus::
    # GpuGemmGroupArithmeticInvalid;` inside a dedicated catch clause (S4), so
    # unlike `GpuAllocationFailed`/`GpuDeviceRemoved` (returned through a
    # ternary, invisible to `_STATUS_RETURN_RE` and needing no entry here) this
    # ONE does match the literal-return pattern and must be named explicitly or
    # it reads as an undocumented tenth guard.
    "GpuGemmGroupArithmeticInvalid",
})

# T-2083 (O35, Claude/Poirot/42ecf79-gpu-serial-port-round9-review.md): the
# real, named anchor for `DecodeStickyTag`'s own function body -- unlike
# every other function this module extracts from, `DecodeStickyTag` is not
# scanned by `_STATUS_RETURN_RE` at all (it uses a function-local `using S =
# superslm::SslmForwardStatus;` alias, `_DECODE_STICKY_TAG_RETURN_RE` below
# matches `S::` specifically, scoped to this one extraction so it cannot
# collide with an unrelated `S` elsewhere in the tree). Executed: this
# signature occurs exactly once in `superslm_gpu.cpp` at HEAD.
DECODE_STICKY_TAG_FUNC_SIGNATURE = "superslm::SslmForwardStatus DecodeStickyTag(int64_t tag) {"
_DECODE_STICKY_TAG_RETURN_RE = re.compile(r"return\s+S::([A-Za-z_][A-Za-z0-9_]*)")
# `gpu_port.h`'s own citation claims this function "maps the device's own
# sticky tag to FOURTEEN statuses, THIRTEEN of them rejecting" -- the number
# this module now pins directly against the function's own real body,
# instead of via the two-endpoint range citation T-2083 retired (it read
# `superslm_gpu.cpp:583`, `:601` then -- quoted as history, not a live
# location; T-2094 deleted the table that held it) whose own range-end
# needle, a bare `"}"`, was satisfied by 177 of the file's 2,023 lines
# (executed at T-2083) and therefore covered where the function
# ENDS, not what is inside it -- deleting an interior `case` while
# preserving the file's own total line count left that citation green.
DECODE_STICKY_TAG_EXPECTED_TOTAL = 14
DECODE_STICKY_TAG_EXPECTED_REJECTING = 13

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


def decode_sticky_tag_status_set(gpu_text: str) -> set[str]:
    """T-2083 (O35): every DISTINCT status name `DecodeStickyTag` returns
    ANYWHERE in its own real, brace-matched body -- content, not the two
    range-endpoint citations `gpu_port.h`'s own prose uses. Comments and
    string/char literals stripped first (`strip_comments`, same convention
    as every other extraction in this module), so a status name mentioned
    only in a comment cannot inflate the set -- mirroring `extract_status_
    set`'s own guarantee for the ladder extractions, applied to this
    function's own `S::` alias instead of the `SslmForwardStatus::`/
    `superslm::SslmForwardStatus::` forms those use."""
    body = extract_function_body(gpu_text, DECODE_STICKY_TAG_FUNC_SIGNATURE, label="superslm_gpu.cpp")
    stripped = strip_comments(body)
    return set(_DECODE_STICKY_TAG_RETURN_RE.findall(stripped))


def check_decode_sticky_tag_range(gpu_text: str) -> list[str]:
    """T-2083 (O35): pins the EXACT claim `gpu_port.h`'s own citation exists
    to protect -- "fourteen statuses, THIRTEEN of them rejecting" -- against
    `DecodeStickyTag`'s own real body, rather than the two-endpoint range
    citation whose own range-end needle (a bare `"}"`) is satisfied by a great
    many of `superslm_gpu.cpp`'s own lines and therefore proves only that the
    function ends somewhere near the cited line, not what is inside it.
    Falsified by execution: deleting one interior `case` while preserving
    the file's own total line count left the two-endpoint citation green;
    this check reads the function's own real content instead, so the same
    deletion changes what it counts regardless of where anything else in the
    file shifted to compensate."""
    names = decode_sticky_tag_status_set(gpu_text)
    total = len(names)
    rejecting = len(names - {"Ok"})
    if total == DECODE_STICKY_TAG_EXPECTED_TOTAL and rejecting == DECODE_STICKY_TAG_EXPECTED_REJECTING:
        return []
    return [
        f"DecodeStickyTag's own real body returns {total} distinct statuses ({rejecting} rejecting), "
        f"not the {DECODE_STICKY_TAG_EXPECTED_TOTAL} ({DECODE_STICKY_TAG_EXPECTED_REJECTING} rejecting) "
        f"gpu_port.h's own citation claims -- names found: {sorted(names)}"
    ]


# T-2088 (S1, Claude/Poirot/8420005-gpu-serial-port-round11-review.md; D-SLM3261): `gpu_port.h`'s
# own paragraph claimed "Not defined by `build.bat` -- the default build never sees these two
# declarations at all," true from T-2070 to T-2071 and false since -- `build.bat`'s test-binary
# compile line has defined this macro for five rounds and nothing in the tree asserted the
# opposite. Executed: removing the flag in scratch leaves gate-on and gate-off reporting the
# IDENTICAL 3-failure count (16 checks and both O11 discrimination cells gone, no test failure of
# any kind) -- the prose alone is the discipline that already failed five times; this is the
# structure. The real anchor is TEXT POSITION, not a line number: `build.bat` issues exactly two
# `cl /nologo` invocations, and only the FIRST (the test binary) may define this macro -- the
# second (the C5 harness) must never see it, by design, so scanning the whole file would be the
# wrong check.
_CL_INVOCATION_MARKER = "cl /nologo"
BUILD_BAT_O11_GATE_FLAG = "/DSUPERSLM_O11_ALLOC_INJECTION"


def check_build_bat_defines_o11_gate(build_bat_text: str) -> list[str]:
    """None (empty list) iff `build.bat`'s own FIRST `cl /nologo` invocation -- the test-binary
    compile line -- contains `BUILD_BAT_O11_GATE_FLAG`. The C5-harness invocation (the second, or
    everything after the first if there is only one) is deliberately excluded: it must never define
    this macro, so its own absence there is correct and not what this check verifies."""
    first_at = build_bat_text.find(_CL_INVOCATION_MARKER)
    if first_at == -1:
        return ["build.bat: no 'cl /nologo' invocation found -- cannot check the O11 gate flag"]
    second_at = build_bat_text.find(_CL_INVOCATION_MARKER, first_at + len(_CL_INVOCATION_MARKER))
    block = build_bat_text[first_at:second_at] if second_at != -1 else build_bat_text[first_at:]
    if BUILD_BAT_O11_GATE_FLAG in block:
        return []
    return [
        f"build.bat's own test-binary compile line no longer defines {BUILD_BAT_O11_GATE_FLAG} -- "
        "the default build silently drops both O11 cells (gate-on and gate-off then report the "
        "IDENTICAL check/failure count, D-SLM3261 S1's own executed measurement, with no test "
        "failure of any kind to flag it)"
    ]


# T-2091 (class B / O30, Dan's own scope-change dispatch, build log §27): a PIN protecting a cell that
# a given pipeline never runs is hollow in that pipeline -- O30/O3 (rounds 9-12's own honestly
# disclosed residual) found ONE instance by hand: `build.bat` ran no Python at all (CORRECTED this
# same round, below -- it now does), so `check_build_bat_defines_o11_gate`'s own pin never fired
# on the LOCAL build path, only in GitHub Actions -- and Actions' CMake job compiles `test_main.cpp`
# linked only against `superslm_test_injection` (`CMakeLists.txt`'s own `SUPERSLM_CORE_SOURCES`),
# which never lists `src/gpu/superslm_gpu.cpp` and never defines `SUPERSLM_O11_ALLOC_INJECTION` --
# so the two O11 discrimination cells this pin exists to protect never ran THERE either. This
# sweep (T-2091) widened the radius: `test_main.cpp` calls `superslm_gpu::*` symbols with NO
# preprocessor gate wider than the narrow O11 one, so on INSPECTION (not executed by this pass --
# Actions is out of the standing project stance's scope, `Claude/CLAUDE.md`'s own "Hardening isn't
# the phase," and chasing its own green is explicitly not this arc's priority) the CMake
# `superslm_tests` target should not even LINK, meaning EVERY GPU-calling cell in `test_main.cpp`
# is unconfirmed there, not only the O11-gated subset.
#
# CLOSED, the local half (T-2091, same round): `build.bat` now invokes this module directly
# (guarded by `where python`, non-fatal if absent -- this script's own contract is a C++-only
# build, and not chasing Actions' own green is not a license to skip a check that IS installed and
# IS the real gate this arc's own ship decisions already run against). The Actions/CMake half
# remains hollow and remains named in `EXECUTION_SCOPE_WAIVERS` below -- this correction narrows
# the residual, it does not close it.
#
# Derived here from real source (never asserted from memory), and checked against a NAMED,
# git-dated waiver registry rather than silently trusted: an execution scope this module discovers
# to be hollow and NOT in `EXECUTION_SCOPE_WAIVERS` fails the build, forcing a future change that
# introduces a NEW hollow scope to name it explicitly rather than let the class re-accrete quietly
# the way the O11 gate flag's own false "not defined" claim did for five rounds.
CMAKE_LISTS = os.path.join(_REPO_ROOT, "CMakeLists.txt")

# T-2094 (S4, Claude/Poirot/2a937f5-gpu-serial-port-class-closure-review.md; D-SLM3277): the
# original manifest hardcoded ONE CMake target by name (`_CMAKE_SUPERSLM_TESTS_LINK_RE` matched
# only `target_link_libraries(superslm_tests ...)`) -- a NEW `add_executable` compiling
# `test_main.cpp` (the exact case the module's own docstring claimed it prevents) landed green
# (T-2094's own MUT-J reproduction). Generic now: every `add_executable`, its own direct source
# list, and its own `target_link_libraries` -> `add_library` -> source-list chain, all derived by
# NAME from the real text, no target hardcoded.
_CMAKE_ADD_EXECUTABLE_RE = re.compile(r"add_executable\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^)]*)\)")
_CMAKE_ADD_LIBRARY_RE = re.compile(
    r"add_library\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(?:STATIC|SHARED|MODULE)?\s*([^)]*)\)"
)
_CMAKE_TARGET_LINK_LIBRARIES_RE = re.compile(
    r"target_link_libraries\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(?:PUBLIC|PRIVATE|INTERFACE)?\s*([^)]*)\)"
)
_CMAKE_VAR_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _cmake_resolve_sources(raw_args: str, cmake_text: str) -> str:
    """`raw_args` (an `add_executable`/`add_library`'s own literal argument text) with every
    `${VAR}` reference expanded against the file's own `set(VAR ...)` definitions -- a flat string
    checked by substring, not a real CMake source-list parse, but enough to answer "does this
    target's own source graph include `superslm_gpu.cpp`" without hardcoding a variable name."""
    resolved = raw_args
    for var_m in _CMAKE_VAR_REF_RE.finditer(raw_args):
        var_name = var_m.group(1)
        set_m = re.search(r"set\(\s*" + re.escape(var_name) + r"\s(.*?)\)", cmake_text, re.DOTALL)
        if set_m is not None:
            resolved += " " + set_m.group(1)
    return resolved


class ExecutionScope:
    __slots__ = ("name", "compiles_test_main", "compiles_gpu_source", "defines_o11_macro")

    def __init__(self, name: str, compiles_test_main: bool, compiles_gpu_source: bool, defines_o11_macro: bool) -> None:
        self.name = name
        self.compiles_test_main = compiles_test_main
        self.compiles_gpu_source = compiles_gpu_source
        self.defines_o11_macro = defines_o11_macro

    @property
    def is_hollow(self) -> bool:
        """A scope is hollow iff it compiles `test_main.cpp` (so `superslm_gpu::*` calls are real
        code the linker must resolve) WITHOUT also compiling `superslm_gpu.cpp` into the SAME
        binary -- every GPU-calling cell there is then unconfirmed to even link, independent of
        whether the O11 macro is defined (a narrower hollowness, "the macro-gated cells specifically
        can't run," is a SUBSET of this one and not tracked separately: a scope that cannot link at
        all cannot usefully be described as "partially" hollow)."""
        return self.compiles_test_main and not self.compiles_gpu_source

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics only
        return (
            f"ExecutionScope({self.name!r}, compiles_test_main={self.compiles_test_main}, "
            f"compiles_gpu_source={self.compiles_gpu_source}, defines_o11_macro={self.defines_o11_macro})"
        )


def derive_execution_scope(build_bat_text: str, cmake_text: str) -> tuple[ExecutionScope, ...]:
    """Every real build target this tree's own two build entry points (`build.bat`, `CMakeLists.
    txt`) define that could compile `tests/test_main.cpp` -- derived from the TEXT ITSELF (which
    `cl`/`add_executable` invocations exist, which source lists they draw from, which `/D`/
    `target_compile_definitions` macros they carry), never hardcoded as a claim about what those
    files say."""
    scopes: list[ExecutionScope] = []
    # build.bat: two `cl /nologo` invocations, the test binary (first) and the C5 harness
    # (second) -- both compile `src\gpu\superslm_gpu.cpp` directly in the SAME command (verified
    # by substring presence in each block), so neither is hollow; only the first must carry the
    # O11 macro (check_build_bat_defines_o11_gate's own job, not restated here).
    first_at = build_bat_text.find(_CL_INVOCATION_MARKER)
    if first_at != -1:
        second_at = build_bat_text.find(_CL_INVOCATION_MARKER, first_at + len(_CL_INVOCATION_MARKER))
        first_block = build_bat_text[first_at:second_at] if second_at != -1 else build_bat_text[first_at:]
        scopes.append(ExecutionScope(
            "build.bat: test binary (out\\superslm_tests.exe)",
            compiles_test_main="test_main.cpp" in first_block,
            compiles_gpu_source="superslm_gpu.cpp" in first_block,
            defines_o11_macro=BUILD_BAT_O11_GATE_FLAG in first_block,
        ))
        if second_at != -1:
            second_block = build_bat_text[second_at:]
            scopes.append(ExecutionScope(
                "build.bat: C5 harness (out\\t2039_c5_harness.exe)",
                compiles_test_main="test_main.cpp" in second_block,
                compiles_gpu_source="superslm_gpu.cpp" in second_block,
                defines_o11_macro=BUILD_BAT_O11_GATE_FLAG in second_block,
            ))
    # CMakeLists.txt: EVERY `add_executable` target, generically -- not one hardcoded name (S4,
    # T-2094). A target is in this manifest's own scope at all only if IT ITSELF compiles
    # `test_main.cpp` (a tool with no test_main.cpp in its own source graph cannot be hollow in
    # the sense this manifest tracks, whatever else it links). `compiles_gpu_source` is true if
    # `superslm_gpu.cpp` appears in the executable's OWN direct sources OR anywhere in its own
    # `target_link_libraries` -> `add_library` chain (one level; this tree's own real graph never
    # nests a library inside another).
    libraries: dict[str, str] = {}
    for lib_m in _CMAKE_ADD_LIBRARY_RE.finditer(cmake_text):
        libraries[lib_m.group(1)] = _cmake_resolve_sources(lib_m.group(2), cmake_text)
    links: dict[str, list[str]] = {}
    for link_m in _CMAKE_TARGET_LINK_LIBRARIES_RE.finditer(cmake_text):
        links.setdefault(link_m.group(1), []).extend(link_m.group(2).split())
    o11_macro_anywhere = "SUPERSLM_O11_ALLOC_INJECTION" in cmake_text
    for exe_m in _CMAKE_ADD_EXECUTABLE_RE.finditer(cmake_text):
        exe_name = exe_m.group(1)
        own_sources = _cmake_resolve_sources(exe_m.group(2), cmake_text)
        if "test_main.cpp" not in own_sources:
            continue
        compiles_gpu_source = "superslm_gpu.cpp" in own_sources
        for lib_name in links.get(exe_name, []):
            if "superslm_gpu.cpp" in libraries.get(lib_name, ""):
                compiles_gpu_source = True
        scopes.append(ExecutionScope(
            f"CMake: {exe_name} (GitHub Actions)",
            compiles_test_main=True,
            compiles_gpu_source=compiles_gpu_source,
            defines_o11_macro=o11_macro_anywhere,
        ))
    return tuple(scopes)


# Named, dated waivers for a hollow scope this sweep found and is NOT fixing this round -- per
# `Claude/CLAUDE.md`'s own standing project rule ("Hardening isn't the phase... red/aborted CI is
# expected and fine") and Dan's own explicit stance against chasing Actions/CI green at this
# project phase. Adding GPU compilation to the CMake target is a real, larger engineering task
# (conditional D3D12/Windows-only linkage, `dxc`-compiled shaders CMake does not currently invoke
# at all) -- named here as owed, not silently absorbed into "CI is out of scope" as a blanket
# excuse for every future gap.
EXECUTION_SCOPE_WAIVERS = {
    "CMake: superslm_tests (GitHub Actions)": (
        "T-2091 (build log §27), 2026-08-14: does not compile src/gpu/superslm_gpu.cpp at all "
        "(SUPERSLM_CORE_SOURCES never lists it, confirmed by direct read of CMakeLists.txt) and "
        "never defines SUPERSLM_O11_ALLOC_INJECTION -- test_main.cpp calls superslm_gpu::* symbols "
        "unconditionally (no preprocessor gate wider than the narrow O11 one), so on inspection "
        "this target should not even link. NOT EXECUTED by this pass (Actions is out of scope per "
        "the standing project stance; the spending limit is also already hit) -- a derived finding "
        "from source inspection, not a confirmed CI failure. The LOCAL build.bat path (real GPU "
        "hardware, C5 bit-identical every round of this arc) is the gate that governs ship "
        "decisions; this waiver does not relax that gate in any way."
    ),
}


def check_execution_scope_waivers(
    build_bat_path: str | None = BUILD_BAT,
    cmake_path: str | None = CMAKE_LISTS,
    waivers: dict[str, str] | None = None,
) -> list[str]:
    """None (empty list) iff every hollow `ExecutionScope` `derive_execution_scope` finds is a KEY
    in `waivers`, AND every key in `waivers` names a scope that currently EXISTS and is currently
    HOLLOW -- an unnamed hollow scope fails the build (a future change that introduces one, a new
    CMake target compiling `test_main.cpp`, cannot land silently the way the O11 gate flag's own
    false "not defined" claim did for five rounds); a STALE waiver -- naming a scope that no longer
    exists, or that a later fix made non-hollow -- ALSO fails the build (T-2094, S4: a waiver
    registry that never expires is the identical class one level up, D-SLM3277's own finding, MUT-H).
    Pass either path as `None` to skip (no real `build.bat`/`CMakeLists.txt` in a fixture tree).
    `waivers` defaults to the CURRENT `EXECUTION_SCOPE_WAIVERS` at CALL time, not at function-
    definition time (`None` resolved inside the body, deliberately, not a `= EXECUTION_SCOPE_
    WAIVERS` default) -- a default bound at definition time would not see a test's own monkeypatch
    of the module-level dict, which is exactly the kind of self-exemption this whole round exists
    to close."""
    if waivers is None:
        waivers = EXECUTION_SCOPE_WAIVERS
    if build_bat_path is None or cmake_path is None:
        return []
    with open(build_bat_path, "r", encoding="utf-8") as f:
        build_bat_text = f.read()
    with open(cmake_path, "r", encoding="utf-8") as f:
        cmake_text = f.read()
    scopes = derive_execution_scope(build_bat_text, cmake_text)
    scopes_by_name = {s.name: s for s in scopes}
    failures: list[str] = []
    for scope in scopes:
        if scope.is_hollow and scope.name not in waivers:
            failures.append(
                f"{scope.name}: hollow execution scope (compiles test_main.cpp without compiling "
                "superslm_gpu.cpp, so its own GPU-calling cells cannot link) with no named waiver "
                "in EXECUTION_SCOPE_WAIVERS -- either fix it or waive it by name, dated, with a reason"
            )
    for waived_name in waivers:
        current = scopes_by_name.get(waived_name)
        if current is None:
            failures.append(
                f"{waived_name}: named in EXECUTION_SCOPE_WAIVERS but no longer exists as a derived "
                "execution scope -- stale waiver, remove it"
            )
        elif not current.is_hollow:
            failures.append(
                f"{waived_name}: named in EXECUTION_SCOPE_WAIVERS as hollow, but it is not hollow "
                "any more -- the waiver's own claim is now false, remove it"
            )
    return failures


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


# --- T-2094 (Claude/Poirot/2a937f5-gpu-serial-port-class-closure-review.md, S1/S2; D-SLM3277;
# Dan's own "prefer removing the defect class's SUBSTRATE over checking it more carefully" order):
# this module used to carry TWO separate line-number-citation systems -- `ProseCitation`
# (`(file, line, needle)`, checked by POSITION) for `gpu_port.h`'s own paragraph, and a
# four-hand-picked-chunk mirror of the identical shape (`SELF_CITATIONS`) for this module's own
# prose about `superslm_gpu.cpp`. Both were brittle by construction: ANY unrelated line inserted
# above a cited line moves every citation below it, which is what made T-2084's own S2 restoration
# (nineteen inserted lines) break twenty-one citations in one edit, and a chunk boundary is a place
# a citation can silently sit OUTSIDE the checked population, which is what let M2's exact defect
# (four stale citations, three stale line counts) hide outside `SELF_CITATIONS`' own four chunks in
# T-2091's own round (D-SLM3277's own S1 finding).
#
# The substrate is removed, not scanned more carefully. A citation in this tree's own prose is now
# ALWAYS a MARKED ANCHOR reference -- `` `Name` (file.ext) `` in prose, a backtick-quoted
# symbol/constant/member-access name immediately followed by its target file in parens -- checked
# for EXISTENCE (does `Name` still appear as real code in `file.ext`), never for POSITION.
# Inserting a comment anywhere in `superslm_gpu.cpp` cannot break an anchor citation, because
# nothing here reads WHERE `Name` sits, only whether it is still there. Where a symbol name would
# collide between two distinct sites prose needs to distinguish (`g_last_weight_upload_was_skipped`'s
# own four write sites, two of them byte-identical statements), a named `// ANCHOR:<name>` comment
# is placed at the target site instead -- `gpu_port.h`'s own citations name
# `lwuws_write_function_entry`/`lwuws_write_catch` this way (`src/gpu/superslm_gpu.cpp`).
#
# ONE mechanism now covers what were five: `find_marked_citations`/`check_marked_citations` below
# replace `ProseCitation`, `GPU_PORT_H_LWUWS_CITATIONS`, `SELF_CITATIONS`,
# `parse_gpu_port_h_citation_lines`, `check_gpu_port_h_citations_match_table`,
# `parse_self_citation_population`, `check_self_citation_population_matches`, `check_prose_citation`,
# and `check_all_prose_citations` in one pass -- scanning the WHOLE tree's own prose for
# marked citations directly, never a maintained Python-side population that could itself drift from
# what the prose actually states, and never a chunk list a citation could sit outside of. There is
# no longer a table to keep in sync with `gpu_port.h`'s own text, because the table IS `gpu_port.h`'s
# own text, read fresh every run. The old line-count-claim constant and its own check function are
# removed outright, not replaced: `check_decode_sticky_tag_range` (above) already pins the real
# claim that figure existed to argue for, directly at source, so the figure itself was never
# load-bearing (round 12's own O1, the cheaper alternative this round takes instead of building a
# third mechanism).
#
# Split-identifier detection (T-2091's own S2, `find_split_backtick_identifiers`/`check_symbol_
# integrity` below) is UNCHANGED in mechanism and EXTENDED in population: it already scans every
# backtick-quoted identifier split across a line-wrap, across the whole tree, with no chunk
# boundary -- exactly the shape S1's own remedy needed, generalized to marked citations too rather
# than duplicated. A marked citation split by a line-wrap is caught by `check_symbol_integrity`'s
# own split-shape match (the identifier half); `check_marked_citations` catches the wider harm --
# a NAME that is not split at all but simply wrong, renamed, or removed.


def _join_wrapped_comment_lines(text: str, comment_marker: str) -> str:
    """Joins `comment_marker`-continued lines with ZERO inserted characters -- a marked citation's
    own `` `Name` (file.ext) `` shape, if line-wrapped between the name and the parenthesized file,
    has zero characters between the two halves in the source a reader sees once the wrap is
    mentally removed, the same reasoning `find_split_backtick_identifiers` already applies to a
    split identifier."""
    pattern = r"\n[ \t]*" + re.escape(comment_marker) + r"[ \t]?"
    return re.sub(pattern, "", text)


# T-2094 (item 1's own scope): the marked-citation form is a new convention this round introduces
# specifically for `gpu_port.h`'s and `superslm_gpu.cpp`'s own prose (and this module's own
# docstrings about them) -- NOT a claim that `tests/test_main.cpp`'s own pre-existing "Name (file)"
# -shaped parentheticals (a different, older, unrelated convention Curie's own file uses for its
# own citations to headers this module does not track) are wrong. Scoped narrower than `SYMBOL_
# INTEGRITY_SCANNED_FILES` on purpose: `check_symbol_integrity`'s own split-identifier scan still
# covers all six files (that class is general); the marked-citation-resolution scan below covers
# only the three files this round's conversion actually touched.
MARKED_CITATION_SCANNED_FILES = (
    ("src/gpu/superslm_gpu.cpp", "//"),
    ("include/superslm/gpu_port.h", "//"),
    ("tests/ci/check_gpu_guard_status_parity.py", "#"),
)
# The RESOLUTION population is wider than the SCAN population: `gpu_port.h`'s own T-2076 note
# legitimately cites a real cell in `tests/test_main.cpp` by name (`TestT2063_S1Mb_WorkScratchUav
# AllocationThrow_ReturnsGpuAllocationFailed_SkippedFalse`) -- a real target this module must be
# able to resolve INTO, even though `test_main.cpp`'s own prose is not itself scanned FOR new
# marked citations (its own pre-existing "Name (file)" convention is unrelated and out of this
# round's scope, named above).
#
# T-2098 (S1, Claude/Poirot/152035b-gpu-serial-port-substrate-removal-review.md; D-SLM3291): the
# four entries below `test_main.cpp` are the round-15 repair. T-2094 converted
# `superslm_gpu.cpp`'s own twelve `file:LINE` citations to prose but could not write ONE of them
# in the marked form, because the files they name were not on this list -- so
# `check_marked_citation` returned "not a recognized citation target" for the correct form and
# `OK` for the unchecked prose form, and the conversion produced nine citations nothing reads
# (executed, the reviewing seat's own MUT-C). The resolvable list is the load-bearing half: until
# a target file is on it, the convention FORBIDS the citation instead of checking it, and prose is
# the only remaining option. `forward_sites.cpp` carries seven of the nine; `forward_sites.h`,
# `checked_chain_funnel.cpp` and `d3d12_harness.h` are the three surviving `file:LINE` citations'
# own targets, converted in the same round so no citation in this arc's own prose is left in a
# form nothing resolves.
MARKED_CITATION_RESOLVABLE_FILES = MARKED_CITATION_SCANNED_FILES + (
    ("tests/test_main.cpp", "//"),
    ("src/forward/forward_sites.cpp", "//"),
    ("include/superslm/forward_sites.h", "//"),
    ("src/forward/checked_chain_funnel.cpp", "//"),
    ("src/gpu/d3d12_harness.h", "//"),
)

# A marked citation: a backtick-quoted, identifier-shaped anchor (letters/digits/underscore, one
# embedded dot allowed for a member-access anchor such as one naming a `dev` field) immediately
# followed by its target file in parens, backticks around the filename optional -- REQUIRED to end
# in one of this arc's own five real extensions (`.cpp`/`.h`/`.py`/`.def`/`.bat`), which is what
# keeps this pattern from firing on ordinary English parentheticals ("(below)", "(inclusive)") or a
# hex literal ("(0xC0000005)") that happen to sit after a backtick-quoted word for unrelated
# reasons -- those are not filenames and never were the shape this class was about. The ONLY
# citation form the files this module scans use to reference another file's content by name.
_MARKED_CITATION_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)`\s*"
    r"\(`?([A-Za-z0-9_./\\-]+\.(?:cpp|h|py|def|bat))`?\)"
)


def find_marked_citations(text: str, comment_marker: str) -> list[tuple[str, str]]:
    """Every `` `Name` (some.ext) `` marked citation in `text`, in order, duplicates included."""
    joined = _join_wrapped_comment_lines(text, comment_marker)
    return _MARKED_CITATION_RE.findall(joined)


def _resolve_citation_target(target_file: str) -> tuple[str, str] | None:
    """`(rel_path, comment_marker)` for `target_file` against `MARKED_CITATION_RESOLVABLE_FILES`
    (below) -- matched by exact relative path first, then by bare basename, so prose can cite the
    short (bare-filename) form without repeating the full repo-relative path. `None` if
    `target_file` names something this population does not include."""
    for rel, marker in MARKED_CITATION_RESOLVABLE_FILES:
        if rel == target_file or os.path.basename(rel) == target_file:
            return rel, marker
    return None


def check_marked_citation(name: str, target_file: str, repo_root: str = _REPO_ROOT) -> str | None:
    """None iff `name` resolves against `target_file` -- as a real CODE identifier (comments
    stripped), as a literal substring of the comment-stripped code (for a member-access anchor like
    `dev.available`, which is not identifier-shaped), or as a `// ANCHOR:<name>` marker comment
    placed deliberately at a target site (checked against the RAW text, since it lives inside a
    comment on purpose) -- else a one-line failure description."""
    resolved = _resolve_citation_target(target_file)
    if resolved is None:
        return f"`{name}` ({target_file}): not a recognized citation target"
    rel_path, marker = resolved
    path = os.path.join(repo_root, rel_path)
    if not os.path.isfile(path):
        return f"`{name}` ({target_file}): file not found: {rel_path}"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = strip_comments(text) if marker == "//" else strip_python_comments(text)
    if name in collect_code_identifiers(text, marker):
        return None
    if "." in name and name in stripped:
        return None
    if f"ANCHOR:{name}" in text:
        return None
    return f"`{name}` ({target_file}): does not resolve -- citation is stale, refresh it"


def check_marked_citation_population_is_live(
    scanned_files: tuple[tuple[str, str], ...] | None = None,
    repo_root: str = _REPO_ROOT,
) -> list[str]:
    """None (empty list) iff EVERY file named in `scanned_files` actually yields at least one
    marked citation. T-2098 (S1, Claude/Poirot/152035b-gpu-serial-port-substrate-removal-
    review.md; D-SLM3291) -- the structural half of that finding, and the cheaper half by a wide
    margin.

    A scan population with a member that contributes NOTHING reads exactly like one that is
    working: `check_marked_citations` iterates three files, finds citations in one, resolves all
    of them, and returns green. That is the state round 15 shipped in and the state three
    consecutive rounds commissioned to close citation staleness each read as covered --
    `MARKED_CITATION_SCANNED_FILES` named `superslm_gpu.cpp` and the checker module, and
    `find_marked_citations` returned 21, 0 and 0 over them. The reviewing seat's own distinguishing
    test was one command: run the mechanism over its own declared population and count what it
    returns. This function IS that command, wired into the build so it cannot go unrun.

    It is deliberately a floor of ONE rather than a pinned count: a count would be a hardcoded
    number of exactly the kind T-2094 removed the substrate for, and it would redden on any honest
    prose edit. One is the number that distinguishes "this file participates" from "this file is
    listed and silent," which is the whole failure."""
    if scanned_files is None:
        scanned_files = MARKED_CITATION_SCANNED_FILES
    failures: list[str] = []
    for rel_path, marker in scanned_files:
        path = os.path.join(repo_root, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if not find_marked_citations(text, marker):
            failures.append(
                f"{rel_path}: named in MARKED_CITATION_SCANNED_FILES but yields ZERO marked "
                "citations -- it is being scanned and contributing nothing, which is "
                "indistinguishable from being covered. Either its cross-file references are "
                "written in a form this scan cannot see (the T-2094 conversion's own defect: "
                "prose `Name`s and parenthesized files that are not adjacent), or the file no "
                "longer cites anything and belongs off the list"
            )
    return failures


def check_marked_citations(
    scanned_files: tuple[tuple[str, str], ...] | None = None,
    repo_root: str = _REPO_ROOT,
) -> list[str]:
    """None (empty list) iff every marked citation found in ANY of `scanned_files`' own prose
    resolves (`check_marked_citation`). Scans the WHOLE of each file -- no chunk, no bounded
    paragraph -- so a citation added anywhere, by anyone, in any of these files is checked; there is
    no boundary for one to sit outside of (T-2094, closing S1's own class: "no chunk list to be
    outside of"). `scanned_files` defaults to `MARKED_CITATION_SCANNED_FILES` (above) at CALL time,
    not at function-definition time, for the same monkeypatch-visibility reason `check_execution_
    scope_waivers` already established for `EXECUTION_SCOPE_WAIVERS`.

    THE COMPLETENESS CLAIM ABOVE IS ABOUT THE SCAN, AND THE RESOLUTION IS SHALLOW -- the two are
    separate axes and this docstring states both, because stating only the first is what reads as
    more closure than was bought (T-2098, M2; D-SLM3291). A citation is checked for EXISTENCE, so a
    citation naming a DIFFERENT REAL SYMBOL passes: executed, MUT-O, two real anchor names swapped
    in `gpu_port.h`'s own ladder enumeration left this check `OK`, exit 0. See this module's own
    docstring, "THE SUCCESSOR RESIDUAL", for the full statement and for why the position-based
    predecessor (O34) is retired rather than carried.

    That the scan population is not merely complete but LIVE -- every listed file actually
    contributing citations -- is `check_marked_citation_population_is_live`'s job, not this one's,
    and it runs first."""
    if scanned_files is None:
        scanned_files = MARKED_CITATION_SCANNED_FILES
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()
    for rel_path, marker in scanned_files:
        path = os.path.join(repo_root, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for name, target_file in find_marked_citations(text, marker):
            if (name, target_file) in seen:
                continue
            seen.add((name, target_file))
            failure = check_marked_citation(name, target_file, repo_root)
            if failure is not None:
                failures.append(f"{rel_path}: {failure}")
    return failures


# T-2091 (S2, Claude/Poirot/2aceac3-gpu-serial-port-ship-candidate-review.md; build log §27): round
# 11's own M1 routed a structural remedy -- "a scan for the old family's fragments with
# `//`-continuation joining" -- and round 12 (T-2088) built a manual fragment-grep re-run instead,
# with no record stating the routed half was dropped. The class stayed live and minted an ELEVENTH
# carrier the same round: `gpu_port.h`'s own dated correction split
# `` `check_build_bat_defines_o11_gate` `` -- the symbol T-2088 itself created -- across the same
# `//`-continuation shape, so an exact-string `grep` for it returns nothing at that site. Built for
# real here: unlike a citation-LIST joiner (where inserting a space between physical lines is
# harmless), a split IDENTIFIER has ZERO characters between its own two halves in the source the
# compiler/linker sees -- the regex below matches the split shape directly in the RAW text
# (backtick, identifier prefix, newline, comment marker, identifier suffix, backtick), which is
# also narrower and less noisy than joining first and then hunting for backtick spans: a whole-file
# join-then-scan matched 307 distinct backtick-quoted terms in this tree, the overwhelming majority
# ordinary prose/local variable names never meant to resolve against a symbol table; matching the
# SPLIT shape directly finds exactly the eleven candidates this class is actually about.
_SPLIT_BACKTICK_IDENTIFIER_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _split_backtick_identifier_re(comment_marker: str) -> "re.Pattern[str]":
    if comment_marker not in _SPLIT_BACKTICK_IDENTIFIER_RE_CACHE:
        _SPLIT_BACKTICK_IDENTIFIER_RE_CACHE[comment_marker] = re.compile(
            r"`([A-Za-z_][A-Za-z0-9_]*)\n[ \t]*" + re.escape(comment_marker) + r"[ \t]?([A-Za-z0-9_]+)`"
        )
    return _SPLIT_BACKTICK_IDENTIFIER_RE_CACHE[comment_marker]


_CODE_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")

# The six files this arc's own O11-rename history has touched -- the same population M1's own
# fragment-grep swept, generalized from "one hardcoded name list per rename" to "every backtick
# identifier split by a line-wrap, in any of these files, forever," and now also the population
# `check_marked_citations` scans for the whole-file anchor-citation class. `(relpath, comment_marker)`.
SYMBOL_INTEGRITY_SCANNED_FILES = (
    ("src/gpu/superslm_gpu.cpp", "//"),
    ("include/superslm/gpu_port.h", "//"),
    ("tests/test_main.cpp", "//"),
    ("tests/ci/check_gpu_guard_status_parity.py", "#"),
    ("tests/ci/test_check_gpu_guard_status_parity.py", "#"),
    ("include/superslm/gpu_layer_loop_guards.def", "//"),
)
# Names split by a line-wrap that do NOT resolve to a current symbol, on purpose: both live inside
# `superslm_gpu.cpp`'s own marked T-2071 verbatim historical quote (build log §25.1, restored by
# T-2084's own S2) -- a quote of history is accurate only if it still says what history actually
# said, confirmed against `git show 72a9b0d` for both names. `TestT2063_S1Mb_WeightAllocationThrow_
# ReturnsGpuAllocationFailed_SkippedFalse` was the real cell name at `72a9b0d` (renamed by `34b3a78`,
# T-2076); `SUPERSLM_O11_WEIGHT_ALLOC_INJECTION` was the real macro name at `72a9b0d` (renamed by
# `07ffe63`, T-2084's own M2). Any other unresolved name is a real defect, not an allowlist
# candidate -- adding one here is a claim that should be checked against `git show`, not a reflex.
SYMBOL_INTEGRITY_ALLOWLIST = frozenset({
    "TestT2063_S1Mb_WeightAllocationThrow_ReturnsGpuAllocationFailed_SkippedFalse",
    "SUPERSLM_O11_WEIGHT_ALLOC_INJECTION",
})


def strip_python_comments(text: str) -> str:
    """Removes `#...` Python comments line by line -- `strip_comments` above only knows the C++
    `//`/`/* */` forms; the two Python files in `SYMBOL_INTEGRITY_SCANNED_FILES` need this instead.
    Not a tokenizer (a `#` inside a string literal would be misread as a comment start), which is
    an accepted over-approximation here exactly like `check_no_forward_leaf_calls.py`'s own
    ban-not-classifier scan: this function's only use is building the CODE-IDENTIFIER universe,
    and under-stripping (leaving a `#`-comment's own text in) can only ADD false-negative resolve
    targets, never hide a real split identifier."""
    return "\n".join(line.split("#", 1)[0] for line in text.split("\n"))


def find_split_backtick_identifiers(text: str, comment_marker: str) -> list[str]:
    """Every backtick-quoted identifier in `text` that is split across a `//`- or `#`-continued
    line -- matched directly against the RAW (unjoined) text via `_split_backtick_identifier_re`,
    concatenating the two halves with ZERO inserted characters (the shape a genuine mid-word
    line-wrap has in source; unlike a joined citation LIST, no space belongs between the halves).
    Order of appearance, duplicates included -- the caller decides how to dedupe/report."""
    pattern = _split_backtick_identifier_re(comment_marker)
    return [m.group(1) + m.group(2) for m in pattern.finditer(text)]


def collect_code_identifiers(text: str, comment_marker: str) -> set[str]:
    """Every identifier-shaped token appearing in `text` OUTSIDE a comment -- the real symbol
    universe a split-backtick candidate is checked against. `comment_marker` selects the stripping
    convention (`strip_comments` for `//`/`/* */`, `strip_python_comments` for `#`)."""
    stripped = strip_comments(text) if comment_marker == "//" else strip_python_comments(text)
    return set(_CODE_IDENTIFIER_RE.findall(stripped))


def check_symbol_integrity(
    scanned_files: tuple[tuple[str, str], ...] | None = None,
    allowlist: frozenset[str] | None = None,
    repo_root: str = _REPO_ROOT,
) -> list[str]:
    """None (empty list) iff every backtick-quoted identifier split across a line-wrap, anywhere in
    `scanned_files`, reconstructs to a real symbol appearing as CODE somewhere in `scanned_files` --
    or is in `allowlist`. T-2091's own red-proof, corrected by T-2094's own M1 (D-SLM3277: the
    original claim was false on the real tree, true only against a fixture that could never
    reproduce the production path): a split of a symbol that is STILL LIVE resolves by design and
    must not redden; this check catches a split whose reconstructed name does NOT resolve --
    typically the moment a rename sweeps every UNSPLIT occurrence and misses the split one, which a
    naive exact-string grep cannot see to sweep in the first place. `scanned_files`/`allowlist`
    default to `SYMBOL_INTEGRITY_SCANNED_FILES`/`SYMBOL_INTEGRITY_ALLOWLIST` at CALL time, not at
    function-definition time, for the same monkeypatch-visibility reason `check_execution_scope_
    waivers` already established."""
    if scanned_files is None:
        scanned_files = SYMBOL_INTEGRITY_SCANNED_FILES
    if allowlist is None:
        allowlist = SYMBOL_INTEGRITY_ALLOWLIST
    all_identifiers: set[str] = set()
    per_file_splits: list[tuple[str, str]] = []
    for rel_path, marker in scanned_files:
        path = os.path.join(repo_root, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_identifiers |= collect_code_identifiers(text, marker)
        for name in find_split_backtick_identifiers(text, marker):
            per_file_splits.append((name, rel_path))
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()
    for name, rel_path in per_file_splits:
        if (name, rel_path) in seen:
            continue
        seen.add((name, rel_path))
        if name in all_identifiers or name in allowlist:
            continue
        failures.append(
            f"{rel_path}: backtick-quoted `{name}` is split across a line-wrap and does not "
            "resolve to any real identifier across SYMBOL_INTEGRITY_SCANNED_FILES, and is not in "
            "SYMBOL_INTEGRITY_ALLOWLIST -- either a rename missed this comment or the allowlist "
            "needs a new, git-show-verified entry"
        )
    return failures


# T-2094 (S3, Claude/Poirot/2a937f5-gpu-serial-port-class-closure-review.md; D-SLM3277): the
# "15-path count saga" -- `gpu_port.h`'s own "fourteen paths in all... fifteen paths' own
# destination in total" claim, hand-re-derived wrong four times across five rounds (T-2055's
# "twelve," corrected wrong again, T-2069's "fourteen," T-2075's "fifteen") -- closed for real.
# T-2091's own attempt compared two HARDCODED constants against a citation table's own label
# counts; neither the constants nor the table read the SENTENCE, so the English words themselves
# could be edited to anything and the check stayed green (T-2091's own MUT-D, D-SLM3277's S3).
# Fixed here by reading BOTH sides fresh: the two number-WORDS parsed directly from `gpu_port.h`'s
# own sentence (`parse_lwuws_path_counts`), and the real structural count derived directly from
# `RunLayerLoopGpu`'s own body (`derive_lwuws_before_decision_count`/`derive_lwuws_after_decision_
# count`) -- no citation table, no hardcoded constant, in either direction.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_LWUWS_BEFORE_COUNT_RE = re.compile(r"catch, (\w+)\s*paths in all")
_LWUWS_TOTAL_COUNT_RE = re.compile(r"alike, (\w+) paths' own destination in\s*total")


def parse_lwuws_path_counts(gpu_port_h_text: str) -> tuple[int, int]:
    """The two number-words `gpu_port.h`'s own `LastWeightUploadWasSkipped` paragraph states in
    English -- "...catch, **fourteen** paths in all" (before the residency decision) and "...alike,
    **fifteen** paths' own destination in total" (the grand total, before + after) -- parsed
    directly from the prose via `_NUMBER_WORDS`, never a hardcoded constant re-typing what the
    sentence already says. Raises `ValueError` if either anchor phrase is missing or its number
    word is not recognized -- the same loud-failure discipline this module has used since T-2080's
    own marker-based parsing: a check that cannot find the sentence it verifies must not pass
    silently."""
    joined = re.sub(r"\n[ \t]*//[ \t]?", " ", gpu_port_h_text)
    before_m = _LWUWS_BEFORE_COUNT_RE.search(joined)
    total_m = _LWUWS_TOTAL_COUNT_RE.search(joined)
    if before_m is None:
        raise ValueError("gpu_port.h: 'catch, <N> paths in all' sentence not found")
    if total_m is None:
        raise ValueError("gpu_port.h: '<N> paths' own destination in total' sentence not found")
    before_word = before_m.group(1).lower()
    total_word = total_m.group(1).lower()
    if before_word not in _NUMBER_WORDS:
        raise ValueError(f"gpu_port.h: unrecognized number word {before_word!r} in path-count sentence")
    if total_word not in _NUMBER_WORDS:
        raise ValueError(f"gpu_port.h: unrecognized number word {total_word!r} in path-count sentence")
    return _NUMBER_WORDS[before_word], _NUMBER_WORDS[total_word]


def count_status_return_statements(region_text: str) -> int:
    """Every literal `return (superslm::)?SslmForwardStatus::X;` OCCURRENCE (not distinct name,
    unlike `extract_status_set`) in `region_text`, comments stripped first."""
    return len(_STATUS_RETURN_RE.findall(strip_comments(region_text)))


# T-2098 (S3/M1, Claude/Poirot/152035b-gpu-serial-port-substrate-removal-review.md; D-SLM3291).
#
# The CUT. The complete residency assignment statement, matched against the COMMENT-STRIPPED body.
# Round 15 cut on `body.find("weights_resident;")` over UN-stripped text, so the boundary landed on
# a T-2080 comment 24 lines above the write it named, and the outcome of a structural derivation
# was decided by the wording of an unrelated comment: executed both ways by the reviewing seat -- a
# genuinely new rejecting return added before the residency decision left the count at 14 and the
# checker `OK` (MUT-N), while rewording that comment so it no longer contained the literal made the
# IDENTICAL return counted (MUT-N2, "real body has 15"), and a comment merely MENTIONING the
# fragment above the ladder collapsed the count to 1 and failed a correct tree (MUT-D). Stripping
# comments removes the comment sensitivity; matching the whole statement rather than the bare
# fragment removes the substring ambiguity. This is the marker-truncation class this module has now
# repaired three times (T-2062 S2 on the CPU side, T-2069 S3 on the GPU side, here on the count).
#
# Stated plainly, because the honest form is what the two prior repairs each had to add later: this
# is STILL A TEXT MARKER, not a language boundary. `find_matching_close_brace` is what this module
# uses when the boundary must be one the language enforces; a "returns above the residency write"
# region has no such boundary, because it is a claim about statement ORDER inside one body rather
# than about nesting. What the repair buys is exactness of the marker (a complete statement, in
# code, not in a comment) -- not immunity to a future edit that moves the statement itself.
_LWUWS_RESIDENCY_WRITE_STATEMENT = "g_last_weight_upload_was_skipped = weights_resident;"
# The CATCH TERNARY. Counted by matching the ternary's own return SHAPE, once per occurrence --
# round 15 asked whether the string `device_removed_reason` existed anywhere in the body and added
# exactly 1, so a SECOND ternary-returned rejection (the shape T-2059 added once already, and the
# shape `_STATUS_RETURN_RE` is documented as blind to) was free: derived count 14, real count 15.
# An existence flag standing in for a count is only ever right at a count of one.
_STATUS_RETURN_TERNARY_RE = re.compile(
    r"return\s+[A-Za-z_][A-Za-z0-9_]*\s*!=\s*S_OK\s*\?", re.DOTALL
)


def count_status_return_ternaries(region_text: str) -> int:
    """Every `return <expr> != S_OK ? ... : ...;` OCCURRENCE in `region_text`, comments stripped
    first -- the one return shape this arc's own guards use that `_STATUS_RETURN_RE` cannot see,
    counted rather than merely detected (T-2098, M1)."""
    return len(_STATUS_RETURN_TERNARY_RE.findall(strip_comments(region_text)))


def derive_lwuws_before_decision_count(gpu_text: str) -> int:
    """The real, structural count of `RunLayerLoopGpu`'s own rejecting return paths that return
    BEFORE the weight-residency decision runs -- the nine-guard ladder's own eleven literal return
    statements plus the two device-capability rejections' own two literal return statements, plus
    one per recording-window catch ternary (a `device_removed_reason` `?:` expression, invisible to
    `_STATUS_RETURN_RE` since it is not a literal `return SslmForwardStatus::X;` -- counted by
    `count_status_return_ternaries`, per occurrence, not by asking whether one exists).

    THE TWO TERMS HAVE DIFFERENT REGIONS, and that is a property of the claim rather than an
    oversight -- stated here because it is the one thing about this derivation a reader would
    otherwise have to re-derive from `gpu_port.h`'s own paragraph.

    The LITERAL-RETURN term is cut positionally: everything in the function's own COMMENT-STRIPPED
    body above the complete residency assignment statement `_LWUWS_RESIDENCY_WRITE_STATEMENT` -- a
    text marker, named as one here rather than described as "the write", and exact in the two ways
    round 15's was not (see that constant's own note). Thirteen today: the nine-guard ladder's own
    eleven, plus the two device-capability rejections.

    The TERNARY term is counted over the WHOLE body, and must be. The number this derivation
    verifies is `gpu_port.h`'s own "fourteen paths in all," and that sentence enumerates
    "the nine-guard ladder, the two device-capability rejections, AND THE RECORDING-WINDOW CATCH."
    The catch's own ternary return sits textually BELOW the residency write, so a purely positional
    cut excludes it and the derivation reads 13 against a prose 14 -- executed, and the reason this
    function does not simply cut both terms at the same index. The catch belongs to the counted set
    on the property the paragraph is actually about: it writes the observable back to `false`
    (`lwuws_write_catch` (`superslm_gpu.cpp`)) before returning, so it reads `false` exactly like
    the thirteen above it, despite having run past the decision. "Before the decision" is the
    paragraph's shorthand for that class, not a claim about statement order, and the derivation
    matches the class.

    Round 15's whole-body scope for this term was therefore RIGHT and its arithmetic was wrong: it
    asked whether `device_removed_reason` appeared anywhere and added exactly 1, so a second
    ternary-returned rejection would have been free. `count_status_return_ternaries` keeps the
    scope and counts the occurrences.

    Raises `ValueError` if the residency marker is absent: a derivation whose region silently
    collapses to the whole body would report a number computed over the wrong region, and this
    module's standing discipline is that a check which cannot find the anchor it verifies must not
    pass silently."""
    body = extract_function_body(gpu_text, GPU_FUNC_SIGNATURE, label="superslm_gpu.cpp")
    stripped_body = strip_comments(body)
    residency_write_at = stripped_body.find(_LWUWS_RESIDENCY_WRITE_STATEMENT)
    if residency_write_at == -1:
        raise ValueError(
            "superslm_gpu.cpp: RunLayerLoopGpu's own residency assignment "
            f"{_LWUWS_RESIDENCY_WRITE_STATEMENT!r} not found in its comment-stripped body -- the "
            "before/after path-count region has no boundary to cut on"
        )
    before_region = stripped_body[:residency_write_at]
    return count_status_return_statements(before_region) + count_status_return_ternaries(stripped_body)


def derive_lwuws_after_decision_count(gpu_text: str) -> int:
    """ONE -- `RunLayerLoopGpu`'s own single terminal call-based return, `return DecodeStickyTag(
    sticky_tag);`, the only path structure that returns AFTER the residency decision (the success
    path and the sticky-tag-decoded path share this one return statement; `DecodeStickyTag`'s own
    internal branching is a separate function, not a second return site in `RunLayerLoopGpu`
    itself)."""
    body = extract_function_body(gpu_text, GPU_FUNC_SIGNATURE, label="superslm_gpu.cpp")
    return len(re.findall(r"return\s+DecodeStickyTag\(", strip_comments(body)))


def check_lwuws_path_count_claim(gpu_port_h_text: str, gpu_text: str) -> list[str]:
    """None iff the two number-words `gpu_port.h`'s own prose states (`parse_lwuws_path_counts`)
    equal the real, structural counts `RunLayerLoopGpu`'s own body has
    (`derive_lwuws_before_decision_count`/`derive_lwuws_after_decision_count`) -- reads the actual
    sentence it governs, both directions, rather than comparing two numbers neither of which was
    the sentence itself (T-2091's own gap, D-SLM3277's S3)."""
    before_word, total_word = parse_lwuws_path_counts(gpu_port_h_text)
    before_real = derive_lwuws_before_decision_count(gpu_text)
    after_real = derive_lwuws_after_decision_count(gpu_text)
    total_real = before_real + after_real
    failures: list[str] = []
    if before_word != before_real:
        failures.append(
            f"gpu_port.h's own prose says '{before_word}' paths in all before the residency "
            f"decision; RunLayerLoopGpu's own real body has {before_real}"
        )
    if total_word != total_real:
        failures.append(
            f"gpu_port.h's own prose says '{total_word}' paths' own destination in total; "
            f"RunLayerLoopGpu's own real body has {total_real} ({before_real} before + {after_real} after)"
        )
    return failures


def run_all_checks(
    forward_sites_path: str = FORWARD_SITES_CPP,
    superslm_gpu_path: str = SUPERSLM_GPU_CPP,
    guards_def_path: str = GUARDS_DEF,
    repo_root: str = _REPO_ROOT,
    gpu_port_h_path: str | None = GPU_PORT_H,
    check_decode_sticky_tag: bool = True,
    build_bat_path: str | None = BUILD_BAT,
    check_marked_citation_scan: bool = True,
    run_symbol_integrity_scan: bool = True,
    check_lwuws_path_count: bool = True,
    cmake_path: str | None = CMAKE_LISTS,
    marked_citation_scanned_files: tuple[tuple[str, str], ...] | None = None,
    symbol_integrity_scanned_files: tuple[tuple[str, str], ...] | None = None,
) -> list[str]:
    """Every failure this module can report against the real tree, or an empty list if all three
    guard-status sets agree, every `.def` citation resolves, and every marked/split citation and
    derived count checks out. Raises `ValueError` (not a failure list) if a source anchor itself is
    missing -- an environment/drift problem this check's own extraction logic cannot route around,
    matching `check_no_forward_leaf_calls.py`'s own `ClangUnavailable` precedent of raising rather
    than reporting a false pass.

    T-2094 (Claude/Poirot/2a937f5-gpu-serial-port-class-closure-review.md; D-SLM3277) removed the
    line-number-citation substrate entirely -- `prose_citations`/`self_citations`/`GPU_PORT_H_
    LWUWS_CITATIONS`/`SELF_CITATIONS`/`SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM` and their own checks no
    longer exist. `check_marked_citation_scan` (below) is what replaces ALL of them at once: a
    whole-tree scan for `` `Name` (file.ext) `` marked citations, checked for existence, never
    position. Nothing here needs a real `gpu_port.h`/`superslm_gpu.cpp` fixture path threaded
    through separately any more, because the scan reads whichever files `marked_citation_scanned_
    files` names directly.

    `gpu_port_h_path` (T-2080, M1's own structural remedy) -- retained ONLY for `check_lwuws_path_
    count` below, which reads `gpu_port.h`'s own real prose to parse its two number-words. Pass
    `None` to skip both (fixture-driven tests with no real `gpu_port.h`).

    `check_decode_sticky_tag` (T-2083, O35) -- pass `False` to skip: this module's own small
    synthetic GPU fixture (`_GPU_FIXTURE`) ends with a CALL to `DecodeStickyTag`, not a definition
    of it, so `extract_function_body` would raise for every fixture-driven mechanism test the same
    way `gpu_port_h_path=None` exists to avoid; every REAL-tree call (`main()` included) gets it
    for free.

    `build_bat_path` (T-2088, S1's own structural remedy, build log §26) -- pass `None` to skip:
    fixture-driven tests have no real `build.bat` to scan. Every REAL-tree call gets the
    O11-gate-flag pin for free.

    `check_marked_citation_scan`/`marked_citation_scanned_files` (T-2094, S1's own structural
    remedy) -- `check_marked_citations` reads EVERY marked citation in EVERY scanned file's own
    prose, whole-file, no chunk boundary: this is what makes M2's own defect class (a stale
    citation sitting outside a hand-picked chunk) structurally impossible rather than merely
    covered more chunks. Pass `check_marked_citation_scan=False`, or an alternate `marked_citation_
    scanned_files` tuple, for a fixture tree with no real files to scan.

    `run_symbol_integrity_scan` (T-2091, S2's own structural remedy, build log §27) -- every
    backtick-quoted identifier split across a line-wrap, across `SYMBOL_INTEGRITY_SCANNED_FILES`,
    must resolve to a real symbol or sit in `SYMBOL_INTEGRITY_ALLOWLIST`. Reads six real files
    directly, so it too needs no fixture-path threading -- pass `False` only to isolate another
    check's own failure in a test.

    `check_lwuws_path_count` (T-2094, S3's own structural remedy) -- parses `gpu_port.h`'s own two
    number-words directly (`parse_lwuws_path_counts`) and compares them against `RunLayerLoopGpu`'s
    own real, structurally-derived path counts (`derive_lwuws_before_decision_count`/`derive_lwuws_
    after_decision_count`) -- no citation table, no hardcoded constant, on either side. Reads
    `gpu_port_h_path` and `superslm_gpu_path`; pass `False` (or `gpu_port_h_path=None`) to skip for
    a fixture tree.

    `cmake_path` (T-2091, class-B sweep, the execution-scope manifest; generalized by T-2094's S4
    to enumerate every `add_executable` target rather than one hardcoded name) -- `None` skips,
    same reasoning as `build_bat_path=None`. Every real-tree call gets `check_execution_scope_
    waivers` for free: any hollow execution scope not named in `EXECUTION_SCOPE_WAIVERS`, or any
    waiver naming a scope that is no longer hollow or no longer exists, fails the build."""
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
    # T-2083 (O35): DecodeStickyTag's own real content, pinned against the
    # exact count gpu_port.h's own citation claims -- see
    # check_decode_sticky_tag_range's own docstring for why the two-endpoint
    # range citation alone cannot do this.
    if check_decode_sticky_tag:
        failures += check_decode_sticky_tag_range(gpu_text)
    # T-2088 (S1, build log §26): build.bat's own test-binary compile line must still define the
    # O11 gate flag -- see check_build_bat_defines_o11_gate's own docstring for the silent-shrink
    # this closes (removing the flag leaves gate-on and gate-off reporting the identical count).
    if build_bat_path is not None:
        with open(build_bat_path, "r", encoding="utf-8") as f:
            build_bat_text = f.read()
        failures += check_build_bat_defines_o11_gate(build_bat_text)
    # T-2094 (S1, D-SLM3277): every marked `` `Name` (file.ext) `` citation, in every scanned
    # file's own prose, whole-file -- the class-closing remedy for what SELF_CITATIONS/GPU_PORT_H_
    # LWUWS_CITATIONS/check_gpu_port_h_citations_match_table used to do with a chunk or a paragraph
    # boundary something could sit outside of.
    if check_marked_citation_scan:
        # T-2098 (S1, D-SLM3291): the population's own vitality FIRST -- a listed file that yields
        # zero citations makes the resolution pass below green for the wrong reason.
        failures += check_marked_citation_population_is_live(
            scanned_files=marked_citation_scanned_files, repo_root=repo_root
        )
        failures += check_marked_citations(scanned_files=marked_citation_scanned_files, repo_root=repo_root)
    # T-2091 (S2, build log §27): the line-wrap symbol-integrity scan round 11's own M1 routed and
    # round 12 did not build -- every backtick-quoted identifier split across a line-wrap, across
    # the arc's own six files, must resolve to a real symbol or sit in the named allowlist.
    if run_symbol_integrity_scan:
        failures += check_symbol_integrity(
            scanned_files=symbol_integrity_scanned_files, repo_root=repo_root
        )
    # T-2094 (S3, D-SLM3277): the path-count sum is derived from gpu_port.h's own real sentence AND
    # RunLayerLoopGpu's own real body -- neither side is a hardcoded/re-typed number any more.
    if check_lwuws_path_count and gpu_port_h_path is not None:
        with open(gpu_port_h_path, "r", encoding="utf-8") as f:
            gpu_port_h_text = f.read()
        failures += check_lwuws_path_count_claim(gpu_port_h_text, gpu_text)
    # T-2091/T-2094 (class B, S4): a hollow scope (a target that compiles test_main.cpp's GPU calls
    # without compiling superslm_gpu.cpp) must be a named, dated, CURRENT waiver -- a stale one
    # (naming a scope that no longer exists or is no longer hollow) fails just as loudly.
    failures += check_execution_scope_waivers(build_bat_path=build_bat_path, cmake_path=cmake_path)
    return failures


def main() -> int:
    try:
        failures = run_all_checks()
    except ValueError as e:
        print(f"check_gpu_guard_status_parity.py: FAILED (source anchor missing) -- {e}", file=sys.stderr)
        return 1
    except OSError as e:
        # T-2098 (O3, D-SLM3291): a scanned or resolvable file that has been renamed or removed
        # raises `FileNotFoundError` out of `check_marked_citations`/`check_symbol_integrity`, which
        # `ValueError` above does not catch -- the module used to answer that with a traceback
        # instead of its own failure line. Loud either way, so this is a conventions fix rather
        # than a soundness one: the build now reads one message shape for every way this check can
        # fail.
        print(f"check_gpu_guard_status_parity.py: FAILED (source file unreadable) -- {e}", file=sys.stderr)
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
