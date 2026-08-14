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
`DecodeStickyTag` range citation (`superslm_gpu.cpp:583-601`) checks two
ENDPOINTS' own content, not what is inside the range -- the range-end
needle, a bare `"}"`, is satisfied by 177 of the file's own 2,023 lines
(executed), so a shift onto the wrong closing brace would pass it, and
deleting an interior `case` while preserving the file's own total line
count (executed: `case 13` removed) left both endpoints individually valid
and this module green, 54 passed, while the citation's own claim ("fourteen
statuses, THIRTEEN of them rejecting") became false. Fixed:
`decode_sticky_tag_status_set`/`check_decode_sticky_tag_range` (below)
extract `DecodeStickyTag`'s own real, brace-matched body and count DISTINCT
statuses directly, pinning the exact claim the range citation exists to
protect rather than two points near where it happens to sit.

O34 NOT CLOSED, and named rather than silently dropped. The T-2080 remedy
above compares SETS of line numbers, so a same-position SWAP of two
citations' own numbers in `gpu_port.h`'s own text (executed: `:716`
`!dev.available` swapped with `:728` the sub-Tier-3 check) left the set
unchanged and this module `OK`. A SEQUENCE/ORDER-preserving comparison was
built and tried -- and FALSIFIED against the real header, not just the
synthetic fixture it first passed against: `gpu_port.h`'s own paragraph
legitimately re-cites some of the same line numbers a second and third time
across its own four dated corrections (`:903`, `:716`, `:728` each appear
twice at HEAD), and the corrections' own citation order, accreted
chronologically over four rounds, does not match `GPU_PORT_H_LWUWS_
CITATIONS`'s fixed definition order even after deduplication. Wiring the
order comparison into `check_gpu_port_h_citations_match_table` turned the
check RED on the real, unmutated tree -- caught before landing by running
the mechanism against real content rather than trusting the fixture alone
(`StandardsDocument.md` §5.4), and reverted. Closing O34 for real needs a
parser that associates each citation with WHICH of the paragraph's own
dated-correction sub-sections it belongs to and compares within each
section, not across the whole accreted paragraph as one sequence -- new
machinery, specified here rather than forced into a check that would be
wrong.

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
# instead of via the two-endpoint range citation (`superslm_gpu.cpp:583`,
# `:601`) whose own range-end needle, a bare `"}"`, is satisfied by 177 of
# the file's 2,023 lines (executed) and therefore covers where the function
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
    citation whose own range-end needle (a bare `"}"`) is satisfied by 177 of
    `superslm_gpu.cpp`'s own 2,023 lines and therefore proves only that the
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


# T-2091 (class B / O30, Dan's own scope-change dispatch, D-SLM3271): a PIN protecting a cell that
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
_CMAKE_CORE_SOURCES_RE = re.compile(r"set\(SUPERSLM_CORE_SOURCES(.*?)\)", re.DOTALL)
_CMAKE_SUPERSLM_TESTS_LINK_RE = re.compile(r"target_link_libraries\(superslm_tests\s+PRIVATE\s+([A-Za-z_]+)\)")


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
    # CMakeLists.txt: `superslm_tests` (`add_executable(superslm_tests tests/test_main.cpp)`,
    # linked against whichever library `target_link_libraries(superslm_tests PRIVATE ...)` names)
    # -- that library's own `set(<NAME> ...)` source list is checked for the GPU translation unit,
    # and the whole file is checked for the O11 macro (`target_compile_definitions`), by NAME, not
    # assumed present or absent.
    link_m = _CMAKE_SUPERSLM_TESTS_LINK_RE.search(cmake_text)
    if link_m is not None:
        linked_lib = link_m.group(1)
        sources_re = re.compile(r"add_library\(" + re.escape(linked_lib) + r"\s+STATIC\s+\$\{([A-Za-z_]+)\}\)")
        sources_m = sources_re.search(cmake_text)
        gpu_in_sources = False
        if sources_m is not None:
            var_name = sources_m.group(1)
            var_re = re.compile(r"set\(" + re.escape(var_name) + r"(.*?)\)", re.DOTALL)
            var_m = var_re.search(cmake_text)
            if var_m is not None:
                gpu_in_sources = "gpu/superslm_gpu.cpp" in var_m.group(1) or "superslm_gpu.cpp" in var_m.group(1)
        scopes.append(ExecutionScope(
            "CMake: superslm_tests (GitHub Actions)",
            compiles_test_main=True,  # add_executable(superslm_tests tests/test_main.cpp) is unconditional
            compiles_gpu_source=gpu_in_sources,
            defines_o11_macro="SUPERSLM_O11_ALLOC_INJECTION" in cmake_text,
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
        "T-2091 (D-SLM3271), 2026-08-14: does not compile src/gpu/superslm_gpu.cpp at all "
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
    in `waivers` -- an UNNAMED hollow scope fails the build, so a future change that introduces one
    (a new CMake target compiling `test_main.cpp`, a new macro-gated cell family) cannot land
    silently the way the O11 gate flag's own false "not defined" claim did for five rounds. Pass
    either path as `None` to skip (no real `build.bat`/`CMakeLists.txt` in a fixture tree).
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
    failures: list[str] = []
    for scope in scopes:
        if scope.is_hollow and scope.name not in waivers:
            failures.append(
                f"{scope.name}: hollow execution scope (compiles test_main.cpp without compiling "
                "superslm_gpu.cpp, so its own GPU-calling cells cannot link) with no named waiver "
                "in EXECUTION_SCOPE_WAIVERS -- either fix it or waive it by name, dated, with a reason"
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


# --- T-2075 (Claude/Poirot/72a9b0d-gpu-serial-port-final-review.md,
# Structural remedy routed by S2/M1): prose-citation resolution, extending
# `check_citation`'s own "a file:line citation must resolve against the
# statement it claims to hold" discipline from the `.def`'s own citations to
# the PROSE citations this module and `gpu_port.h` carry -- the class M1
# found (seventeen of eighteen line citations stale, broken by a sibling
# commit eight minutes after they were written, in the same worktree) is
# exactly what `check_citation` already prevents for `.def` rows; nothing
# extended that same machinery to a hand-written paragraph's own citations,
# which is why they went stale silently while the `.def`'s never have.
#
# Each entry names ONE claim about CURRENT source -- a `superslm_gpu.cpp`
# line that must still contain a given substring. Deliberately does NOT
# cover every citation in this tree: this module's own "WHY THIS EXISTS"
# docstring cites `gpu_port.h:91-103`/`superslm_gpu.cpp:544-553` as a
# description of what the code looked like AT THE TIME T-2055 found the
# guard-count "structural closure" claim false -- a historical record of a
# past state, not a claim resolvable against HEAD, and checking it against
# HEAD would produce a false positive on every unrelated line shift forever
# after. The population below is `gpu_port.h`'s own `LastWeightUploadWasSkipped`
# paragraph -- the ONE paragraph that has been re-corrected four consecutive
# rounds (T-2055, T-2062, T-2069, T-2075) and is therefore the citation
# surface most likely to drift again.
class ProseCitation:
    __slots__ = ("label", "file_relpath", "line", "needle")

    def __init__(self, label: str, file_relpath: str, line: int, needle: str) -> None:
        self.label = label
        self.file_relpath = file_relpath
        self.line = line
        self.needle = needle

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics only
        return f"ProseCitation({self.label!r}, {self.file_relpath!r}, {self.line!r}, {self.needle!r})"


# `gpu_port.h`'s own `LastWeightUploadWasSkipped` paragraph, T-2075's own
# fresh re-derivation: eleven ladder returns, two device-capability
# rejections, the catch's own ternary, the sticky-tag terminal call, and the
# observable's own four write sites -- nineteen citations, all fifteen
# rejecting-return-path members plus the four write sites S2/M2 both name.
#
# CORRECTED 2026-08-14 (T-2084, Claude/Poirot/42ecf79-gpu-serial-port-round9-
# review.md; D-SLM3247): every citation below shifted when S2's restoration
# (superslm_gpu.cpp) reinserted the 19 lines T-2080's own correction had
# actually deleted -- re-derived at source, one grep per anchor, not by
# assuming a uniform per-line delta (the shift is NOT uniform: 51 lines
# ahead of `DecodeStickyTag`, 58 lines ahead of the write sites below it,
# because T-2084's own M1/M2/M3 fixes add lines of their own between them).
GPU_PORT_H_LWUWS_CITATIONS = (
    ProseCitation("ladder return 1 (InvalidLayerBudget)", "src/gpu/superslm_gpu.cpp", 712, "InvalidLayerBudget"),
    ProseCitation("ladder return 2 (InvalidContextCap)", "src/gpu/superslm_gpu.cpp", 713, "InvalidContextCap"),
    ProseCitation("ladder return 3 (HeadDimGeometryMismatch)", "src/gpu/superslm_gpu.cpp", 722, "HeadDimGeometryMismatch"),
    ProseCitation("ladder return 4 (KvHeadGeometryMismatch)", "src/gpu/superslm_gpu.cpp", 726, "KvHeadGeometryMismatch"),
    ProseCitation("ladder return 5a (InvalidContextCap overflow)", "src/gpu/superslm_gpu.cpp", 741, "InvalidContextCap"),
    ProseCitation("ladder return 5b (WorkspaceTooSmall null)", "src/gpu/superslm_gpu.cpp", 745, "WorkspaceTooSmall"),
    ProseCitation("ladder return 5c (WorkspaceTooSmall undersized)", "src/gpu/superslm_gpu.cpp", 746, "WorkspaceTooSmall"),
    ProseCitation("ladder return 6 (InvalidHiddenCodes)", "src/gpu/superslm_gpu.cpp", 756, "InvalidHiddenCodes"),
    ProseCitation("ladder return 7 (SequenceAlreadyComplete)", "src/gpu/superslm_gpu.cpp", 759, "SequenceAlreadyComplete"),
    ProseCitation("ladder return 8 (PositionOverCap)", "src/gpu/superslm_gpu.cpp", 765, "PositionOverCap"),
    ProseCitation("ladder return 9 (KvCapacityExhausted)", "src/gpu/superslm_gpu.cpp", 767, "KvCapacityExhausted"),
    ProseCitation("device-capability 1 (!dev.available)", "src/gpu/superslm_gpu.cpp", 774, "dev.available"),
    ProseCitation("device-capability 2 (Tier-3 check)", "src/gpu/superslm_gpu.cpp", 786, "KvPrecisionUnsupported"),
    ProseCitation("catch's own ternary", "src/gpu/superslm_gpu.cpp", 1468, "device_removed_reason"),
    ProseCitation("sticky-tag terminal return", "src/gpu/superslm_gpu.cpp", 1519, "DecodeStickyTag"),
    ProseCitation("write site 1 (static init)", "src/gpu/superslm_gpu.cpp", 299, "g_last_weight_upload_was_skipped"),
    ProseCitation("write site 2 (function entry)", "src/gpu/superslm_gpu.cpp", 672, "g_last_weight_upload_was_skipped"),
    ProseCitation("write site 3 (residency decision)", "src/gpu/superslm_gpu.cpp", 961, "g_last_weight_upload_was_skipped"),
    ProseCitation("write site 4 (the catch)", "src/gpu/superslm_gpu.cpp", 1458, "g_last_weight_upload_was_skipped"),
    # T-2080 (Claude/Poirot/94ebee3-gpu-serial-port-closing-review.md, M2):
    # the paragraph's own `DecodeStickyTag` citation is a RANGE
    # (`superslm_gpu.cpp:583-601`), which `ProseCitation`'s single-`line`
    # shape cannot represent as one entry -- covered here as its own two
    # endpoints, the review's own named cheap fix, rather than widening
    # `ProseCitation` itself for a population of one range.
    ProseCitation("DecodeStickyTag range start", "src/gpu/superslm_gpu.cpp", 583, "DecodeStickyTag"),
    ProseCitation("DecodeStickyTag range end", "src/gpu/superslm_gpu.cpp", 601, "}"),
)

# T-2088 (M2, Claude/Poirot/8420005-gpu-serial-port-round11-review.md; D-SLM3261): this module's
# own docstrings state `superslm_gpu.cpp`'s total line count three times (to argue the
# DecodeStickyTag range-end needle's own false-positive rate) and cite its DecodeStickyTag range
# four times (to describe what `check_decode_sticky_tag_range` above replaces) -- both went stale
# when T-2084's own S2 restoration shifted the file, because that round's citation sweep refreshed
# `GPU_PORT_H_LWUWS_CITATIONS` and `gpu_port.h` and never checked this module's OWN prose about a
# file it does not own. "The module whose purpose is stopping stale citations carried four stale
# citations and three stale line counts about its own subject" (the review's own words). Fixed the
# same way every other self-inflicted staleness in this arc has been: the class dies by joining the
# checked population, not by one more manual refresh. `SELF_CITATIONS` reuses `ProseCitation`/
# `check_all_prose_citations` exactly as `GPU_PORT_H_LWUWS_CITATIONS` does; the line-count claim
# gets its own pin below since a total-line-count is not a `(file, line, needle)` shape.
SELF_CITATIONS = (
    ProseCitation("this module's own DecodeStickyTag range start (self-citation)",
                  "src/gpu/superslm_gpu.cpp", 583, "DecodeStickyTag"),
    ProseCitation("this module's own DecodeStickyTag range end (self-citation)",
                  "src/gpu/superslm_gpu.cpp", 601, "}"),
)

SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM = 2023


def check_superslm_gpu_cpp_line_count_claim(gpu_text: str) -> list[str]:
    """None iff `superslm_gpu.cpp`'s own real line count matches `SUPERSLM_GPU_CPP_LINE_COUNT_
    CLAIM`, the figure this module's own docstrings state (three places) about a file this module
    does not own. Not load-bearing by itself -- `check_decode_sticky_tag_range` already pins the
    real claim the citation exists to protect, directly at source -- but the denominator is stated
    as fact three times and D-SLM3261 M2 found it off by 58 (1,965 claimed, 2,023 real) after this
    round's own S2 restoration alone, so it is checked exactly like every other prose claim this
    module makes about a file it does not own."""
    actual = len(gpu_text.splitlines())
    if actual == SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM:
        return []
    return [
        f"superslm_gpu.cpp has {actual} lines; this module's own docstrings (three places) claim "
        f"{SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM} -- refresh both the prose and this constant together"
    ]


# T-2091 (S1, Claude/Poirot/2aceac3-gpu-serial-port-ship-candidate-review.md; D-SLM3271):
# SELF_CITATIONS/SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM (T-2088) are fixed values checked AGAINST the
# real superslm_gpu.cpp -- sound as far as it goes, but nothing derives either one FROM this
# module's own prose, so restoring the exact stale text D-SLM3261's own M2 found (four
# `:532-550` citations, three "1,965 lines" claims) left every check in this module green, exit 0
# (MUT-3, the review's own falsifying case): three records called this "closing the class... the
# same way gpu_port.h's prose is checked," and it was not the same way -- gpu_port.h's own
# citations get a SECOND check, `check_gpu_port_h_citations_match_table`, that derives the
# citation POPULATION from the header's own text; this module's citations about ITSELF had only
# the first (resolution) half. Built here: the population is parsed from this module's own
# docstrings/comments, at the four chunks that currently state either fact -- not the whole file,
# because a whole-file scan would also match this module's own DELIBERATELY historical citations
# (the top docstring's `:544-553`, `:643, 655`, `:1276-1277`, none of them claims about HEAD) and
# false-positive on every one of them, which is exactly the false-positive risk `GPU_PORT_H_LWUWS_
# CITATIONS`'s own docstring names for the identical reason.
_SELF_CITATION_TEXT_CHUNKS = (
    ("O35 CLOSED: `GPU_PORT_H_LWUWS_CITATIONS`'s own",
     "O34 NOT CLOSED, and named rather than silently dropped."),
    ("# `gpu_port.h`'s own citation claims this function \"maps the device's own",
     "DECODE_STICKY_TAG_EXPECTED_TOTAL = 14"),
    ('"""T-2083 (O35): pins the EXACT claim',
     'file shifted to compensate."""'),
    ("the paragraph's own `DecodeStickyTag` citation is a RANGE",
     'ProseCitation("DecodeStickyTag range end"'),
)
_LINE_COUNT_FIGURE_RE = re.compile(r"([\d,]+) lines")


def parse_self_citation_population(this_module_text: str) -> tuple[set[int], list[int]]:
    """Every `superslm_gpu.cpp:<line>` citation and every `N,NNN lines` figure THIS module's own
    docstrings/comments state about `superslm_gpu.cpp` -- parsed from the TEXT ITSELF at the four
    named chunks in `_SELF_CITATION_TEXT_CHUNKS`, reusing `_GPU_PORT_H_CITATION_RE` (the identical
    full-form/short-form/range matcher `parse_gpu_port_h_citation_lines` uses for `gpu_port.h`'s
    own paragraph) and the SAME `//`-continuation-joining transformation, applied per chunk rather
    than to one contiguous span. Returns (the set of distinct cited lines, every line-count figure
    found, in encounter order) -- the caller compares each against its own real population."""
    citation_lines: set[int] = set()
    line_count_claims: list[int] = []
    for start_marker, end_marker in _SELF_CITATION_TEXT_CHUNKS:
        start = this_module_text.find(start_marker)
        if start == -1:
            raise ValueError(
                f"check_gpu_guard_status_parity.py: self-citation chunk start marker not found: {start_marker!r}"
            )
        end = this_module_text.find(end_marker, start)
        if end == -1:
            raise ValueError(
                f"check_gpu_guard_status_parity.py: self-citation chunk end marker not found after start: {end_marker!r}"
            )
        chunk = this_module_text[start:end]
        joined = re.sub(r"\n[ \t]*#?[ \t]?", " ", chunk)
        for m in _GPU_PORT_H_CITATION_RE.finditer(joined):
            for part in m.group(1).split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-")
                    citation_lines.add(int(a))
                    citation_lines.add(int(b))
                else:
                    citation_lines.add(int(part))
        for m in _LINE_COUNT_FIGURE_RE.finditer(joined):
            line_count_claims.append(int(m.group(1).replace(",", "")))
    return citation_lines, line_count_claims


def check_self_citation_population_matches(
    this_module_text: str,
    self_citations: tuple[ProseCitation, ...] = SELF_CITATIONS,
    line_count_claim: int = SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM,
) -> list[str]:
    """The structural half of S1's remedy (T-2091, D-SLM3271): derives the citation/line-count
    POPULATION from this module's own docstrings/comments (`parse_self_citation_population`
    above) and asserts it equals `self_citations`'s own line set and `line_count_claim` -- the
    exact shape `check_gpu_port_h_citations_match_table` already implements against `gpu_port.h`,
    now applied to this module's OWN claims about a file it does not own. Catches restoring the
    M2 defect verbatim even when `SELF_CITATIONS`/`SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM` themselves
    are untouched and correct -- MUT-3, the review's own falsifying case, which T-2088's
    fixed-value-only checks could not see. `self_citations`/`line_count_claim` default to the real
    module-level constants for every production call; a test passes an alternate tuple/int to
    prove the comparison catches drift in EITHER direction (prose stale, or the constant itself
    moved without the prose following)."""
    parsed_lines, parsed_line_counts = parse_self_citation_population(this_module_text)
    failures: list[str] = []
    table_lines = {c.line for c in self_citations}
    if parsed_lines != table_lines:
        failures.append(
            "check_gpu_guard_status_parity.py's own docstrings cite a different set of "
            "superslm_gpu.cpp lines than SELF_CITATIONS does -- the constant is a copy that has "
            "drifted from this module's own prose:\n"
            f"    this module's own prose cites: {sorted(parsed_lines)}\n"
            f"    SELF_CITATIONS states:         {sorted(table_lines)}\n"
            f"    prose has, constant lacks: {sorted(parsed_lines - table_lines)}   "
            f"constant has, prose lacks: {sorted(table_lines - parsed_lines)}"
        )
    wrong_counts = sorted({n for n in parsed_line_counts if n != line_count_claim})
    if wrong_counts:
        failures.append(
            "check_gpu_guard_status_parity.py's own docstrings state a superslm_gpu.cpp line "
            f"count ({wrong_counts}) that disagrees with SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM "
            f"({line_count_claim}) -- the prose and the constant must state the same figure"
        )
    return failures


# T-2091 (S2, Claude/Poirot/2aceac3-gpu-serial-port-ship-candidate-review.md; D-SLM3271): round
# 11's own M1 routed a structural remedy -- "a scan for the old family's fragments with
# `//`-continuation joining, which is machinery `parse_gpu_port_h_citation_lines` already
# implements for exactly this header style" -- and round 12 (T-2088) built a manual fragment-grep
# re-run instead, with no record stating the routed half was dropped. The class stayed live and
# minted an ELEVENTH carrier the same round: `gpu_port.h`'s own dated correction split
# `` `check_build_bat_defines_o11_gate` `` -- the symbol T-2088 itself created -- across the same
# `//`-continuation shape, so an exact-string `grep` for it returns nothing at that site. Built for
# real here: unlike `parse_gpu_port_h_citation_lines`'s own joiner (built for comma-separated
# citation LISTS, where inserting a space between physical lines is harmless), a split IDENTIFIER
# has ZERO characters between its own two halves in the source the compiler/linker sees -- the
# regex below matches the split shape directly in the RAW text (backtick, identifier prefix,
# newline, comment marker, identifier suffix, backtick), which is also narrower and less noisy
# than joining first and then hunting for backtick spans: a whole-file join-then-scan matched 307
# distinct backtick-quoted terms in this tree, the overwhelming majority ordinary prose/local
# variable names never meant to resolve against a symbol table; matching the SPLIT shape directly
# finds exactly the eleven candidates this class is actually about.
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
# identifier split by a line-wrap, in any of these files, forever." `(relpath, comment_marker)`.
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
    scanned_files: tuple[tuple[str, str], ...] = SYMBOL_INTEGRITY_SCANNED_FILES,
    allowlist: frozenset[str] = SYMBOL_INTEGRITY_ALLOWLIST,
    repo_root: str = _REPO_ROOT,
) -> list[str]:
    """None (empty list) iff every backtick-quoted identifier split across a line-wrap, anywhere in
    `scanned_files`, reconstructs to a real symbol appearing as CODE somewhere in `scanned_files` --
    or is in `allowlist`. T-2091's own red-proof: restoring `gpu_port.h`'s own pre-fix split,
    `` `check_build_bat_defines_o11_\\n// gate` ``, must redden this check by the reconstructed
    name; the fixed carrier (one line, no mid-identifier wrap) must not."""
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


# T-2091 (class A, Dan's own scope-change dispatch -- the "15-path count saga" instance): the
# "fourteen paths... fifteen paths' own destination in total" claim `gpu_port.h`'s own
# LastWeightUploadWasSkipped paragraph makes is a SUM over labelled groups within
# `GPU_PORT_H_LWUWS_CITATIONS` -- a claim that was re-derived by hand four times (T-2055's
# "twelve," T-2062's "twelve" corrected, T-2069's "fourteen," T-2075's "fifteen," each landing
# only after a review found the prior count wrong) and never pinned as a property of the SAME
# table `check_gpu_port_h_citations_match_table` already derives from the header's own text.
# Pinned here structurally instead of narratively: the paragraph's own labels ("ladder return N",
# "device-capability N", "catch's own ternary" -- fourteen BEFORE-decision destinations; "sticky-
# tag terminal return" -- the one AFTER-decision destination that fans into Ok/thirteen rejecting
# sub-statuses but is ONE return statement) are counted directly from the table, not re-typed as a
# second copy of the number.
_LWUWS_BEFORE_DECISION_LABEL_PREFIXES = ("ladder return", "device-capability", "catch's own ternary")
_LWUWS_AFTER_DECISION_LABEL_PREFIXES = ("sticky-tag terminal return",)
LWUWS_EXPECTED_BEFORE_DECISION_COUNT = 14
LWUWS_EXPECTED_AFTER_DECISION_COUNT = 1


def check_lwuws_path_count_claim(
    citations: tuple[ProseCitation, ...] = GPU_PORT_H_LWUWS_CITATIONS,
) -> list[str]:
    """None iff `citations` (`gpu_port.h`'s own citation population, already checked against the
    header's real text by `check_gpu_port_h_citations_match_table`) has exactly `LWUWS_EXPECTED_
    BEFORE_DECISION_COUNT` labels naming a before-the-residency-decision destination and exactly
    `LWUWS_EXPECTED_AFTER_DECISION_COUNT` naming an after-the-decision one -- the sum ("fourteen ...
    fifteen paths' own destination in total") `gpu_port.h`'s own prose states in English, now
    derived from the SAME table rather than re-typed as an independent number four rounds have
    each gotten wrong at least once before the next review caught it."""
    before = [c.label for c in citations if c.label.startswith(_LWUWS_BEFORE_DECISION_LABEL_PREFIXES)]
    after = [c.label for c in citations if c.label.startswith(_LWUWS_AFTER_DECISION_LABEL_PREFIXES)]
    failures: list[str] = []
    if len(before) != LWUWS_EXPECTED_BEFORE_DECISION_COUNT:
        failures.append(
            f"GPU_PORT_H_LWUWS_CITATIONS has {len(before)} before-the-decision labels "
            f"(ladder/device-capability/catch), not the {LWUWS_EXPECTED_BEFORE_DECISION_COUNT} "
            "gpu_port.h's own prose claims ('fourteen paths in all') -- found: "
            f"{before}"
        )
    if len(after) != LWUWS_EXPECTED_AFTER_DECISION_COUNT:
        failures.append(
            f"GPU_PORT_H_LWUWS_CITATIONS has {len(after)} after-the-decision labels "
            f"(sticky-tag terminal), not the {LWUWS_EXPECTED_AFTER_DECISION_COUNT} gpu_port.h's "
            f"own prose structure implies -- found: {after}"
        )
    return failures


# T-2080 (Claude/Poirot/94ebee3-gpu-serial-port-closing-review.md, M1): the
# real anchors bounding `gpu_port.h`'s own `LastWeightUploadWasSkipped`
# paragraph -- the SAME two markers used throughout this ticket's own
# citation refresh -- so `parse_gpu_port_h_citation_lines` below reads
# exactly the span `GPU_PORT_H_LWUWS_CITATIONS` claims to describe, never
# the whole file.
GPU_PORT_H_LWUWS_PARAGRAPH_START = (
    "T-2052 (item 3, Claude/Curie/t2019-gpu-serial-red-suite-2026-08-13.md §13.2):"
)
GPU_PORT_H_LWUWS_PARAGRAPH_END = "bool LastWeightUploadWasSkipped();"

_GPU_PORT_H_CITATION_RE = re.compile(r"(?:superslm_gpu\.cpp:|`:)(\d+(?:-\d+)?(?:,\s*\d+)*)")


def parse_gpu_port_h_citation_lines(gpu_port_h_text: str) -> set[int]:
    """Every line number `gpu_port.h`'s own `LastWeightUploadWasSkipped`
    paragraph cites -- parsed from the TEXT ITSELF, not a hand-transcribed
    copy of it (T-2080, M1: `GPU_PORT_H_LWUWS_CITATIONS` above WAS such a
    copy, held in this Python module, and nothing checked it against the
    header it claims to describe -- corrupting the header's own citations
    left this whole module green, since nothing here ever read `gpu_port.h`
    at all). Both citation shapes the paragraph actually uses are matched:
    the first-mention full form (`superslm_gpu.cpp:654, 655, 664, ...`, a
    comma list after ONE prefix) and every subsequent short form
    (`` `:716` ``, backtick-wrapped, no prefix) -- including RANGE forms of
    either (`` `:583-601` ``), expanded to both endpoints, matching how
    `GPU_PORT_H_LWUWS_CITATIONS` represents `DecodeStickyTag`'s own range as
    two entries. Line-wrapped `//`-continued comment prose (this header's own
    house style -- a citation list routinely spans two physical lines) is
    joined into one continuous span first, so a citation split across a line
    break by ordinary word-wrap is not missed.

    T-2083 (O34) tried a SEQUENCE/ORDER-preserving variant of this parser,
    reasoning that a same-position SWAP of two citations' own numbers (the
    review's own executed falsifying case, `:716`/`:728`) would change
    encounter order even though it does not change the SET -- true on the
    review's own small fixture, and FALSIFIED against the real header:
    `gpu_port.h`'s own paragraph legitimately re-cites some of the same line
    numbers a second and third time across its own four dated corrections
    (`:903`, `:716`, `:728` each appear twice; the corrections' own citation
    order does not match `GPU_PORT_H_LWUWS_CITATIONS`'s fixed definition
    order even after deduplication, since later corrections were appended
    chronologically, not re-sorted into the table's own logical order).
    Wiring a sequence comparison into `check_gpu_port_h_citations_match_
    table` below turned the check RED on the real, unmutated tree -- caught
    before landing by running against real content rather than trusting the
    fixture, `StandardsDocument.md` §5.4's own rule. O34 is filed as a named
    residual instead (see this module's own top docstring): closing it needs
    a parser that associates each citation with WHICH of the paragraph's own
    dated-correction sub-sections it belongs to, and compares within each
    section rather than across the whole accreted paragraph -- new
    machinery, not a set-to-list type change."""
    start = gpu_port_h_text.find(GPU_PORT_H_LWUWS_PARAGRAPH_START)
    if start == -1:
        raise ValueError(f"gpu_port.h: paragraph start marker not found: {GPU_PORT_H_LWUWS_PARAGRAPH_START!r}")
    end = gpu_port_h_text.find(GPU_PORT_H_LWUWS_PARAGRAPH_END, start)
    if end == -1:
        raise ValueError(f"gpu_port.h: paragraph end marker not found after start: {GPU_PORT_H_LWUWS_PARAGRAPH_END!r}")
    paragraph = gpu_port_h_text[start:end]
    joined = re.sub(r"\n[ \t]*//[ \t]?", " ", paragraph)
    lines: set[int] = set()
    for m in _GPU_PORT_H_CITATION_RE.finditer(joined):
        for part in m.group(1).split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                lines.add(int(a))
                lines.add(int(b))
            else:
                lines.add(int(part))
    return lines


def check_gpu_port_h_citations_match_table(
    gpu_port_h_text: str,
    citations: tuple[ProseCitation, ...] = GPU_PORT_H_LWUWS_CITATIONS,
) -> list[str]:
    """The structural half of M1's remedy: derives the citation POPULATION
    from `gpu_port.h`'s own text and asserts it equals `GPU_PORT_H_LWUWS_
    CITATIONS`'s own line-number set. `check_prose_citation` (below) still
    verifies each of THIS module's citations resolves against `superslm_gpu.
    cpp` -- that half was already sound. What was missing is this one: that
    the table's own line numbers are the SAME ones the header currently
    cites, so an edit to `gpu_port.h`'s own citations (a correct refresh OR
    an accidental corruption) that is not mirrored in this table is caught,
    instead of leaving this module checking a frozen copy of a fact the
    header no longer states.

    T-2083 (O34): still a SET comparison, deliberately -- see `parse_gpu_
    port_h_citation_lines`'s own docstring for the order-preserving variant
    that was tried, found to false-positive against the real header's own
    legitimate repeated citations, and reverted before landing. A same-
    position swap of two citations' own numbers is a named, open residual,
    not silently closed by a check that would have been wrong."""
    parsed = parse_gpu_port_h_citation_lines(gpu_port_h_text)
    table = {c.line for c in citations}
    if parsed == table:
        return []
    return [
        "gpu_port.h's own LastWeightUploadWasSkipped paragraph cites a different set of lines than "
        "GPU_PORT_H_LWUWS_CITATIONS does -- the table is a copy that has drifted from the header it "
        "describes:\n"
        f"    gpu_port.h's own citations: {sorted(parsed)}\n"
        f"    this module's own table:    {sorted(table)}\n"
        f"    header has, table lacks: {sorted(parsed - table)}   table has, header lacks: {sorted(table - parsed)}"
    ]


def check_prose_citation(citation: ProseCitation, repo_root: str = _REPO_ROOT) -> str | None:
    """None if `citation.needle` appears on line `citation.line` of
    `citation.file_relpath` (resolved against `repo_root`); else a one-line
    failure description naming the citation and what is actually there --
    the exact shape `check_citation` already gives `.def` rows, applied to a
    hand-written paragraph's own citations instead of a generated table's."""
    path = os.path.join(repo_root, citation.file_relpath)
    if not os.path.isfile(path):
        return f"{citation.label}: file not found: {citation.file_relpath}"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if citation.line < 1 or citation.line > len(lines):
        return (
            f"{citation.label}: {citation.file_relpath}:{citation.line} out of range "
            f"({len(lines)} lines in file) -- citation is stale, refresh it"
        )
    actual = lines[citation.line - 1]
    if citation.needle not in actual:
        return (
            f"{citation.label}: {citation.file_relpath}:{citation.line} does not contain "
            f"{citation.needle!r} -- actual line: {actual.strip()!r} -- citation is stale, refresh it"
        )
    return None


def check_all_prose_citations(
    citations: tuple[ProseCitation, ...] = GPU_PORT_H_LWUWS_CITATIONS,
    repo_root: str = _REPO_ROOT,
) -> list[str]:
    failures: list[str] = []
    for c in citations:
        failure = check_prose_citation(c, repo_root)
        if failure is not None:
            failures.append(failure)
    return failures


def run_all_checks(
    forward_sites_path: str = FORWARD_SITES_CPP,
    superslm_gpu_path: str = SUPERSLM_GPU_CPP,
    guards_def_path: str = GUARDS_DEF,
    repo_root: str = _REPO_ROOT,
    prose_citations: tuple[ProseCitation, ...] = GPU_PORT_H_LWUWS_CITATIONS,
    gpu_port_h_path: str | None = GPU_PORT_H,
    check_decode_sticky_tag: bool = True,
    build_bat_path: str | None = BUILD_BAT,
    self_citations: tuple[ProseCitation, ...] = SELF_CITATIONS,
    check_self_line_count: bool = True,
    check_self_citation_population: bool = True,
    run_symbol_integrity_scan: bool = True,
    check_lwuws_path_count: bool = True,
    cmake_path: str | None = CMAKE_LISTS,
) -> list[str]:
    """Every failure this module can report against the three real files, or
    an empty list if all three status sets agree and every `.def` citation
    resolves. Raises `ValueError` (not a failure list) if a source anchor
    itself is missing -- that is an environment/drift problem this check's
    own extraction logic cannot route around, matching `check_no_forward_
    leaf_calls.py`'s own ClangUnavailable precedent of raising rather than
    reporting a false pass.

    `prose_citations` defaults to the real population (`gpu_port.h`'s own
    `LastWeightUploadWasSkipped` paragraph, `GPU_PORT_H_LWUWS_CITATIONS`) --
    every REAL-tree call site (`main()` included) gets this module's own
    structural citation check for free. A caller driving a small SYNTHETIC
    `superslm_gpu.cpp` fixture (this module's own mechanism/mutation tests)
    passes `prose_citations=()` explicitly: the real population cites real
    production line numbers no tiny fixture file has, and checking it
    against a fixture would fail for a reason that has nothing to do with
    what that test is exercising.

    `gpu_port_h_path` (T-2080, M1's own structural remedy) is where the
    citation POPULATION is derived FROM, checked against `prose_citations`
    for agreement -- pass `None` to skip this half (fixture-driven tests
    with no real `gpu_port.h` to parse do this, same reasoning as
    `prose_citations=()`).

    `check_decode_sticky_tag` (T-2083, O35) -- pass `False` to skip: this
    module's own small synthetic GPU fixture (`_GPU_FIXTURE`) ends with a
    CALL to `DecodeStickyTag`, not a definition of it, so `extract_function_
    body` would raise for every fixture-driven mechanism test the same way
    `gpu_port_h_path=None` exists to avoid for the citation-population check
    above; every REAL-tree call (`main()` included) gets it for free.

    `build_bat_path` (T-2088, S1's own structural remedy, D-SLM3261) -- pass
    `None` to skip: fixture-driven tests have no real `build.bat` to scan,
    same reasoning as `gpu_port_h_path=None`. Every REAL-tree call gets the
    O11-gate-flag pin for free.

    `self_citations`/`check_self_line_count` (T-2088, M2's own structural
    remedy) -- this module's OWN prose makes claims about `superslm_gpu.cpp`
    (the DecodeStickyTag range, cited four times; the file's total line
    count, stated three times) exactly like `gpu_port.h`'s paragraph does,
    and D-SLM3261 M2 found both classes stale here after a line shift that
    was swept everywhere else. Checked the same way `prose_citations` is:
    pass `self_citations=()` / `check_self_line_count=False` for a fixture
    tree with no real `superslm_gpu.cpp` to check its own citations against.

    `check_self_citation_population` (T-2091, S1's own structural remedy, D-SLM3271) -- the
    POPULATION half `self_citations`/`check_self_line_count` were missing: derives the citation/
    line-count claims from THIS MODULE'S OWN docstrings (`parse_self_citation_population`) and
    asserts they equal `SELF_CITATIONS`/`SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM` -- catches restoring
    the exact M2 defect verbatim even when those two constants are untouched and correct (MUT-3,
    the review's own falsifying case). Reads this module's OWN source text, not a fixture, so it
    needs no path parameter and no fixture-test gating beyond the bool itself.

    `run_symbol_integrity_scan` (T-2091, S2's own structural remedy, D-SLM3271) -- round 11's own
    M1 routed a line-wrap-aware symbol-integrity scan and round 12 built a manual fragment-grep
    instead, silently. `check_symbol_integrity` closes it for real: every backtick-quoted
    identifier split across a line-wrap, across `SYMBOL_INTEGRITY_SCANNED_FILES`, must resolve to
    a real symbol or sit in `SYMBOL_INTEGRITY_ALLOWLIST`. Reads six real files directly (its own
    parameters default from `SYMBOL_INTEGRITY_SCANNED_FILES`/`repo_root`), so it too needs no
    fixture-path threading -- pass `False` only to isolate another check's own failure in a test.

    `check_lwuws_path_count` (T-2091, class-A sweep) -- reuses `prose_citations` (the SAME
    already-checked table) to derive the "fourteen ... fifteen paths' own destination" sum
    `gpu_port.h`'s own prose states in English, closing the "15-path count saga" class (re-derived
    by hand four times across five rounds, each landing a different wrong number before the next
    review caught it) as a structural property instead of a fifth hand-count.

    `cmake_path` (T-2091, class-B sweep, the execution-scope manifest) -- `None` skips, same
    reasoning as `build_bat_path=None`. Every real-tree call gets `check_execution_scope_waivers`
    for free: any hollow execution scope (a target that compiles `test_main.cpp` without compiling
    `superslm_gpu.cpp`) not named in `EXECUTION_SCOPE_WAIVERS` fails the build."""
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
    # T-2075 (Structural): the prose-citation population is checked against
    # `repo_root` too -- resolved relative to it exactly like the `.def`
    # citations above, so a scratch-tree mutation test can exercise this the
    # same way it exercises everything else in this function.
    failures += check_all_prose_citations(citations=prose_citations, repo_root=repo_root)
    # T-2080 (M1's own structural remedy): the population itself is checked
    # against `gpu_port.h`'s own text -- editing the header without mirroring
    # the edit into `prose_citations` (a correct refresh OR a corruption) is
    # what M1 found this module blind to.
    if gpu_port_h_path is not None:
        with open(gpu_port_h_path, "r", encoding="utf-8") as f:
            gpu_port_h_text = f.read()
        failures += check_gpu_port_h_citations_match_table(gpu_port_h_text, citations=prose_citations)
    # T-2083 (O35): DecodeStickyTag's own real content, pinned against the
    # exact count gpu_port.h's own citation claims -- see
    # check_decode_sticky_tag_range's own docstring for why the two-endpoint
    # range citation alone cannot do this.
    if check_decode_sticky_tag:
        failures += check_decode_sticky_tag_range(gpu_text)
    # T-2088 (S1, D-SLM3261): build.bat's own test-binary compile line must still define the O11
    # gate flag -- see check_build_bat_defines_o11_gate's own docstring for the silent-shrink this
    # closes (removing the flag leaves gate-on and gate-off reporting the identical failure count).
    if build_bat_path is not None:
        with open(build_bat_path, "r", encoding="utf-8") as f:
            build_bat_text = f.read()
        failures += check_build_bat_defines_o11_gate(build_bat_text)
    # T-2088 (M2, D-SLM3261): this module's own prose about superslm_gpu.cpp -- the DecodeStickyTag
    # range citations, the file's total line count -- is now checked the same way gpu_port.h's own
    # prose about the same file already is, closing the class where this module's own citations
    # were the one artifact its own machinery did not govern.
    if self_citations:
        failures += check_all_prose_citations(citations=self_citations, repo_root=repo_root)
    if check_self_line_count:
        failures += check_superslm_gpu_cpp_line_count_claim(gpu_text)
    # T-2091 (S1, D-SLM3271): the population half M2's own "structural remedy" was missing --
    # this module's own docstring claims about superslm_gpu.cpp, derived from the TEXT ITSELF and
    # checked against SELF_CITATIONS/SUPERSLM_GPU_CPP_LINE_COUNT_CLAIM, the way gpu_port.h's own
    # citation population already is against GPU_PORT_H_LWUWS_CITATIONS.
    if check_self_citation_population:
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:
            this_module_text = f.read()
        failures += check_self_citation_population_matches(this_module_text)
    # T-2091 (S2, D-SLM3271): the line-wrap symbol-integrity scan round 11's own M1 routed and
    # round 12 did not build -- every backtick-quoted identifier split across a line-wrap, across
    # the arc's own six files, must resolve to a real symbol or sit in the named allowlist.
    if run_symbol_integrity_scan:
        failures += check_symbol_integrity(repo_root=repo_root)
    # T-2091 (class A, the "15-path count saga" instance): the path-count sum is derived from the
    # SAME citations table already checked against gpu_port.h's own text, not re-typed.
    if check_lwuws_path_count:
        failures += check_lwuws_path_count_claim(citations=prose_citations)
    # T-2091 (class B, the execution-scope manifest): a hollow scope (a target that compiles
    # test_main.cpp's GPU calls without compiling superslm_gpu.cpp) must be a named, dated waiver.
    failures += check_execution_scope_waivers(build_bat_path=build_bat_path, cmake_path=cmake_path)
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
