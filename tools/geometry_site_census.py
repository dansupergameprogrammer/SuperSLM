"""geometry_site_census.py -- T-2432 (design Sec2.5): the structural closure for the
class of geometry-assumption defects four independent passes (grounding, temper, audit,
fold round 1) each found more instances of. Converts "was every site found" from a reading
question into a checkable one, in three parts:

  1. Registry <-> marker symmetry. Every registry entry with status "fixed" or
     "confirmed-correct" has exactly its own `expected_marker_count` occurrences of
     `SSLM-GEOMETRY-SITE: GS-NN` in the tree (T-2441, Poirot 327ee29-t2438-ask5-tracka-
     review.md, Significant 4, D-SLM5438: the registry's own per-site count, not a blanket
     "at least one" -- several sites legitimately touch more than one code location, so
     the count itself varies by site); every marker in the tree has a matching registry
     entry. A site moved, deleted, or renamed without updating the registry fails here; a
     marker copy-pasted with the wrong id, or landed on the wrong declaration among a
     site's own several, fails here.
  2. Pattern <-> marker coverage -- the check that catches a site nobody has registered
     yet. An independent, deliberately broad regex family per assumption family, run
     across every source class the four grounding passes ranged over (*.cpp, *.h, *.hlsl,
     *.py), and every raw hit not covered by a marker within the same line window fails
     the census.

     KNOWN LIMITATION (T-2468 F2, Claude/Mendeleev/t2468-census-recommissioning-2026-08-31.md
     Sec4a case 4, dated 2026-08-31, pre-existing -- not introduced or touched by T-2475's own
     exclusion-matching fix): `_r1_multiply_hit`/`_qow_hit` co-occurrence-match one PHYSICAL
     LINE at a time (`readlines()`, per-line). A genuinely new R1/QOW site whose co-occurring
     tokens land on different physical lines -- plausible under ordinary formatting of a long
     conditional or a wrapped return -- evades Part 2 entirely, regardless of markers or
     exclusions. A `PASS` from this census does not cover that shape; it covers only defects
     whose co-occurring tokens share one physical line. Restated at `main()`'s own PASS line so
     a reader of a passing run sees it without opening this file.

     DEAD EXCLUSION check (T-2475, Claude/Poirot/6597903-t2472-ask5-tracka-confirmation.md
     Observation, D-SLM5617): every `_PART2_EXCLUDED_TEXT` entry must match at least one
     non-comment line in its own registered file somewhere during the walk. An entry that
     matches nothing -- its excused statement deleted, moved, or reworded -- previously
     produced no diagnostic at all, indistinguishable from a healthy tree; it now fails the
     census by name (`DEAD EXCLUSION: ...`) so a stale, silently-inert exclusion is surfaced
     rather than left to describe a residual that no longer exists.
  3. Per-site regression check (T-2441, Poirot 327ee29-t2438-ask5-tracka-review.md,
     Significant 2, D-SLM5436; scope bounding corrected T-2445, Poirot
     ddbc57a-t2443-ask5-tracka-confirmation.md, Significant 2, D-SLM5436 superseded; keying
     corrected T-2475; keying corrected again T-2481 -- occurrence identification is now
     CONTENT-ADDRESSED rather than position-keyed, see below).
     Parts 1 and 2 both read whether a marker EXISTS; neither reads what the code AT a
     marked location actually says, so a "fixed" site whose own fix is reverted in place,
     marker left untouched, satisfies both. For every `fixed` site that carries a
     `required_tokens` list (the registry's own field, one substring set per site), at
     least one of those tokens must appear in that OCCURRENCE's own bounded scope -- the
     registry's own `required_token_scopes` map, keyed by FILE and matched to a specific
     physical marker by CONTENT (an `anchor` fragment expected near that occurrence's own
     code) rather than by that occurrence's position among its site's own other occurrences
     (see the registry's own header comment for how each record is derived). A revert that
     restores the pre-fix code removes the token the landed fix itself introduced, so this
     part fails where parts 1 and 2 do not. Demonstrated by construction: GS-14's fix
     reverted to `plan.out_channels = hidden_size;`, marker untouched, passes parts 1 and 2
     and fails here.

     T-2445 correction: scope used to run from a marker to the NEXT marker anywhere in the
     same file, which (a) swept unrelated code between two distant, unrelated sites'
     markers into the search -- a required token appearing there by coincidence masked a
     revert of the actual site -- and (b) for a multi-line site, let an untouched sibling
     line elsewhere in that same wide scope mask a revert of the one line a specific
     mutation touched. Executed and found doing exactly that: three independent single-line
     reverts (GS-12's o_proj in-width, GS-10's packed q_weight byte extent, GS-18's
     LayerScratch q_codes width) all passed under the old rule. The registry's own
     per-occurrence scope-end field closed both, at the time, by keying each occurrence to
     its marker's absolute line and storing the absolute end line: the bounded scope was
     exactly as wide as that occurrence's own governed code.

     T-2475 correction (Claude/Poirot/6597903-t2472-ask5-tracka-confirmation.md, Significant
     1): that absolute keying carried its own drift. The LOOKUP half (`marker_line`) is
     recomputed from a fresh scan every run, which a prior round read as making the whole
     structure self-correcting; the STORED half (the registry's own key and its end line)
     is a literal frozen at authoring time and is not recomputed by anything. An edit
     anywhere ELSE in the same file -- one comment line inserted far above every marker, not
     touching any site's own code -- shifts every marker below it without shifting the
     registry to match, and the census reports a false `MISSING ... ENTRY` on untouched,
     correctly-fixed code. GS-01's own registry note already records this exact shape being
     hit and hand-repaired once (T-2450's 8-line insert); the review reproduced it as the
     general case on demand. The registry now keys each occurrence by ORDINAL (this site's
     Nth marker found in this file, which an edit elsewhere cannot reorder) and stores an
     OFFSET from the marker to its own scope end (a local distance, invariant outside that
     occurrence's own span) rather than an absolute line -- removing the term that drifted
     rather than adding a rule to remember it. An edit INSIDE one occurrence's own governed
     span (between its marker and its own required token) can still desync the offset; this
     is unchanged from the prior scheme and is not the class either scheme targets.

Scope (T-2432): this census covers families R1 and QOW, the two families Track A owns.
Family KLP (the Option-G fused-K-landing assumption, GS-05) is Track B's own scope
(D-SLM5243) and is not built by this ticket -- GS-05 is registered with status
"not-yet-built" and is exempt from all three parts of this census until Track B's own
build lands it; running the KLP pattern sweep here would either find nothing (the code
does not exist yet) or, if Track B's own code already exists unmarked elsewhere,
incorrectly attribute Track B's own obligation to this census. Track B's own build is
expected to extend this same registry/script when it lands GS-05, not to duplicate the
mechanism.

Usage: python geometry_site_census.py [--repo-root PATH]
Exit 0 on a clean census (every fixed/confirmed-correct site has exactly its own registered
`expected_marker_count` of markers, no unmarked pattern hit exists for R1/QOW, and no fixed
site's own required token is missing from its marker's scope); exit 1 and a printed report
otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_REGISTRY_PATH = os.path.join(_THIS_DIR, "geometry_site_registry.json")

_MARKER_RE = re.compile(r"SSLM-GEOMETRY-SITE:\s*(GS-\d+)")

# T-2481: Part 3's content-addressed occurrence identification (see the Part 3 header comment,
# below) searches an EXEMPT record's own anchor within this many lines of its marker -- an
# exempt occurrence carries no offset of its own (there is no required-token check to bound), so
# this window exists only to decide WHICH registered record an exempt marker matches, never to
# bound a correctness check. Generous relative to every anchor actually registered today (the
# widest is GS-19's own 9 lines) and cheap to widen if a future exempt occurrence needs more.
_ANCHOR_SEARCH_WINDOW_WHEN_EXEMPT = 20

# Family R1: hidden_size == num_attention_heads * head_dim, re-derived or enforced.
# The division form (uncast): `hidden_size / head_dim` or `hidden_size\head_dim` in any
# spacing. The multiply form is matched by same-line co-occurrence of head_dim, a
# heads-count token, and a `*` -- loosened from strict adjacency because the founding
# site (CheckConfigGeometry) wraps both operands in static_cast<uint64_t>(...).
_R1_DIVISION_RE = re.compile(r"hidden_size\s*/\s*head_dim")


def _r1_multiply_hit(line: str) -> bool:
    if "head_dim" not in line:
        return False
    if not (("num_attention_heads" in line) or re.search(r"\bheads\b", line)):
        return False
    return "*" in line


# Family QOW: hidden_size used as a stand-in for q_proj's/o_proj's true width, without
# checking anything -- a same-line co-occurrence of `hidden_size`/`g_hidden_size` and one
# of the projection/weight/fold/channel-count tokens the design's own §2.5 names.
_QOW_TOKENS = ("q_proj", "o_proj", "q_weight", "o_weight", "q_fold", "o_fold", "QProj",
               "OProj", "out_channels")


def _qow_hit(line: str) -> bool:
    if "hidden_size" not in line and "g_hidden_size" not in line:
        return False
    return any(tok in line for tok in _QOW_TOKENS)


_SOURCE_GLOBS = (".cpp", ".h", ".hlsl", ".py")
# Directories this census does not sweep -- generated/vendored/build output, never
# hand-authored geometry logic.
_SKIP_DIR_NAMES = {".git", "out", "build", "__pycache__", "node_modules"}

# T-2432's own narrowing, stated rather than silently applied: the pattern-coverage half
# (part 2) sweeps PRODUCTION source only -- src/, include/, and the shipped conversion
# pipeline's own three Python modules (T-2441 Minor 4 fix, D-SLM5449: this comment said
# "four" against a three-entry _PART2_ALLOWED_FILES below it, and the build log that shipped
# alongside it repeated "four" -- checked rather than assumed a typo, per the review's own
# framing that the count decides which files get swept: tools/convert_tokenizer.py, the one
# other Python conversion-adjacent tool in this tree, is never called by convert_model.py and
# converts a DIFFERENT artifact, the tokenizer, which carries no hidden_size/q_width geometry
# at all -- it is not a fourth member of this pipeline, and no other candidate was found, so
# the count itself was the error, not the file list) -- not the whole tree the design's own
# fold-round-1 text describes ("*.cpp, *.h, *.hlsl, *.py, and the test-fixture tree"). A blunt
# same-line-token-co-occurrence regex, run over test assertions, docstrings, and the
# ALREADY-audited (design Sec2.3, "needed no change for geometry") reference_pipeline.py's
# own correct q_width arithmetic, produced dozens of false positives this session's own
# remaining time could not hand-tune away without risking a regex precise enough to also
# miss a real future defect. Narrowing to production source keeps the check's actual job --
# catching an unmarked geometry defect before it ships -- while the registry <-> marker
# symmetry half (part 1) still runs tree-wide, unnarrowed. Flagged here as an owed
# broadening, not silently scoped down.
_PART2_ALLOWED_PREFIXES = (
    os.path.join("src", ""),
    os.path.join("include", ""),
    os.path.join("src", "gpu", "shaders", ""),
)
_PART2_ALLOWED_FILES = (
    os.path.join("tools", "sslm_convert_validate.py"),
    os.path.join("tools", "convert_model.py"),
    os.path.join("tools", "calibrate_checkpoint.py"),
)

# T-2432's own named exclusions from Part 2, each citing why the hit is not a fresh finding:
_PART2_EXCLUDED_FILES = {
    # design Sec2.3: "the converter is already geometry-correct... needed no change" --
    # already-audited, not a defect this census exists to (re-)find.
    os.path.join("tools", "reference_pipeline", "pipeline.py"),
}
# (relative path, distinctive non-comment source-text fragment) pairs excluded individually,
# each with its own reason in the comment beside it -- never a whole-file exclusion for a
# single documented residual.
#
# T-2467 (Claude/Poirot/665f430-t2462-ask5-tracka-confirmation.md, Significant 1, D-SLM5557):
# this set used to be keyed by (relative path, 1-based line number) -- a decorative identifier
# with no window/scope semantics of its own, the same shape the bad-alloc membership oracle's
# line column had (T-2458, D-SLM5537). Executed both directions on the line-keyed form: a
# single comment line inserted above `proof_manifest.h:54` turned the census red on an
# untouched, already-excused enumerator declaration (false negative -- the excluded line's own
# text just moved down one line and the stale line number no longer covered it); a NEW,
# unregistered q_proj-width-from-hidden_size site written AT `adapter_marshal.h:188` was
# silently absorbed, exit 0, with the unaudited site sitting in the tree (false positive -- the
# stale line number covered whatever code happened to be there, not the excused code itself).
# Re-keyed on the excused code's own text so the exclusion follows its subject instead of a
# coordinate that can drift away from it.
#
# T-2468 (Claude/Mendeleev/t2468-census-recommissioning-2026-08-31.md, F1, D-SLM5603): re-keying
# on text closed the position defeat and opened a content-shaped one -- the re-key's own match,
# `text in line`, excused the WHOLE physical line the instant the fragment appeared anywhere in
# it, so a genuinely new, unregistered geometry-defect statement appended onto the SAME line as
# an already-excused fragment was silently absorbed too, exit 0, unaudited site in the tree --
# demonstrated on 2 of these 3 exclusions, mechanism identical for the third. The header comment
# this replaces claimed widening was "bounded to another literal copy of this exact excused
# statement, not whatever lands on this line number next"; that held for a DIFFERENT line
# reusing the fragment (still true today, each fragment below verified unique among its file's
# own non-comment lines) and did not hold for the excused line's OWN remainder, which is exactly
# what T-2468 found and this correction states.
#
# T-2475: the property this set is now built to is that an exclusion excuses exactly the
# statement it names and nothing else sharing its line. `_part2_excise_excluded_text` (below)
# removes each matched fragment's own text from the line before the pattern regexes ever see
# it, rather than skipping the whole line -- new content before, after, or instead of the
# fragment is left in the remainder and scanned exactly like ordinary code. The `model.h` and
# `proof_manifest.h` fragments below now carry their own full stripped line (the enumerator
# declaration plus the trailing comment that is the actual reason Part 2's regexes fire on that
# line at all) rather than a shorter uniqueness-only prefix -- excising a partial prefix would
# have left that trailing comment's own hidden_size/head_dim/num_attention_heads/`*` text in the
# remainder, re-triggering Part 2 on the untouched tree. `adapter_marshal.h`'s fragment already
# spans its whole line and is unchanged.
_PART2_EXCLUDED_TEXT = {
    # design Sec2.1 closing paragraph: "One item explicitly NOT touched, flagged rather than
    # silently left alone" -- AdapterOutChannelsFor's own identical hidden_size-for-q_proj/
    # o_proj convention, out of scope for Ask 5 (no adapter conversion requested for a
    # non-square base model). A residual for whichever design next asks for one.
    (os.path.join("include", "superslm", "adapter_marshal.h"),
     'if (proj == "q_proj" || proj == "o_proj" || proj == "down_proj") return hidden_size;'),
    # T-2432's own named residual, not fixed by this build: CheckConfigGeometry's own
    # ConfigGeometryStatus::HiddenSizeGeometryMismatch enumerator (and its C-ABI mirror,
    # SslmModelStatus::ConfigGeometryHiddenSizeMismatch) is now UNREACHABLE dead code -- Track
    # A step 1 removed the return path that ever produced it, but the enumerator itself is
    # additive-only (D-SLM3526) and this build does not remove enum values, only what
    # populates them. The comment beside each declaration still describes the value's
    # original, now-stale meaning. Flagged for the planner rather than silently left, per
    # StandardsDocument.md Sec5.6's "every deferral is surfaced loudly" -- not fixed here
    # because removing or renaming an ABI-additive enumerator is a design-level call, not a
    # build-time one.
    #
    # T-2475: fragment widened from the enumerator's own text alone ("ConfigGeometryHiddenSize
    # Mismatch,") to the full stripped line, including the trailing "// R1: ..." comment that is
    # itself what trips _r1_multiply_hit -- the enumerator text alone matches no R1/QOW pattern.
    (os.path.join("include", "superslm", "model.h"),
     "ConfigGeometryHiddenSizeMismatch,    // R1: hidden_size != num_attention_heads * head_dim"),
    # T-2475: same widening as model.h above, and for the same reason -- the trailing comment,
    # not the enumerator name, is what the R1 regex actually matches.
    (os.path.join("include", "superslm", "proof_manifest.h"),
     "HiddenSizeGeometryMismatch, // hidden_size != num_attention_heads * head_dim -- R1, REMOVED"),
}


def _part2_excise_excluded_text(rel_path: str, line: str, matched=None) -> str:
    """Returns `line` with every `_PART2_EXCLUDED_TEXT` fragment registered for `rel_path`
    removed from it -- each fragment's own exact text, at most once per fragment PER LINE. T-2475:
    this replaces the prior `_part2_excluded_text_hit`, which matched a fragment's presence and
    then skipped the ENTIRE line -- excusing whatever else shared it. This function excuses only
    the fragment's own text, wherever it sits on the line; the caller scans whatever remains
    exactly like ordinary code, so new content sharing the excused line -- appended after the
    fragment, prepended before it, or on a second, unrelated statement -- is not swept in for
    free. A line an exclusion does not touch is returned unchanged.

    `matched`, if given a dict, is incremented once (`matched[(f, text)] += 1`) for every
    NON-COMMENT LINE on which a fragment is actually found and excised -- the caller's own tally
    of how many times each `_PART2_EXCLUDED_TEXT` entry fired across the whole tree walk. An
    entry with a count of ZERO is DEAD: its excused statement was deleted, moved, or reworded, so
    this fragment now matches nothing (T-2475 fold-in, Poirot 6597903-t2472-ask5-tracka-
    confirmation.md Observation, D-SLM5617). An entry with a count ABOVE ONE is AMBIGUOUS
    (T-2479/T-2481, Poirot f363c2a-t2479-census-class-confirmation.md Significant 1, D-SLM5651/
    D-SLM5670): the exclusion's own uniqueness bound -- "each fragment verified unique among its
    file's own non-comment lines" -- is a hand-checked, present-tense fact about today's tree, not
    a property the mechanism enforces on its own; a genuinely new, unregistered site written as a
    literal copy of an excused fragment is excised for free by a set-membership check exactly as
    readily as the real excused statement is, so uniqueness holding is not uniqueness enforced. A
    count in place of a set-or-boolean check turns that hand-verified assumption into something
    the walk itself confirms every run, on the same data it already visits -- run_census's own
    caller reports both a dead entry and an ambiguous one as findings rather than leaving either
    silently inert."""
    for f, text in _PART2_EXCLUDED_TEXT:
        if rel_path == f and text in line:
            line = line.replace(text, "", 1)
            if matched is not None:
                matched[(f, text)] = matched.get((f, text), 0) + 1
    return line


def _iter_source_files(repo_root: str, *, production_only: bool = False):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for fn in filenames:
            if not fn.endswith(_SOURCE_GLOBS):
                continue
            full = os.path.join(dirpath, fn)
            if production_only:
                rel = os.path.relpath(full, repo_root)
                if not (rel.startswith(_PART2_ALLOWED_PREFIXES) or rel in _PART2_ALLOWED_FILES):
                    continue
            yield full


def _load_registry():
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["sites"]


def run_census(repo_root: str) -> list[str]:
    """Returns a list of failure lines; empty means a clean census."""
    failures: list[str] = []
    registry = _load_registry()
    registry_by_id = {s["id"]: s for s in registry}

    # Sites this census is not yet responsible for (Track B's own future obligation).
    exempt_ids = {s["id"] for s in registry if s["status"] == "not-yet-built"}
    marker_required_ids = {s["id"] for s in registry if s["status"] in ("fixed", "confirmed-correct")}

    # --- Part 1: registry <-> marker symmetry. ---
    found_markers: dict[str, list[tuple[str, int]]] = {}
    # T-2441 (S2 fix, D-SLM5436): markers grouped BY FILE, sorted by line, plus the file's own
    # lines cached -- Part 3 (below) needs "every marker in this file, in order" to bound each
    # occurrence's own scope, and re-reads would otherwise repeat this same file walk a third
    # time (Part 2 already repeats it once, production-source-only).
    markers_by_file: dict[str, list[tuple[int, str]]] = {}
    file_lines_cache: dict[str, list[str]] = {}
    for path in _iter_source_files(repo_root):
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel_path = os.path.relpath(path, repo_root)
        file_markers: list[tuple[int, str]] = []
        for i, line in enumerate(lines, start=1):
            m = _MARKER_RE.search(line)
            if m:
                gs_id = m.group(1)
                found_markers.setdefault(gs_id, []).append((rel_path, i))
                file_markers.append((i, gs_id))
        if file_markers:
            markers_by_file[rel_path] = file_markers
            file_lines_cache[rel_path] = lines

    for gs_id in marker_required_ids:
        locs = found_markers.get(gs_id, [])
        # T-2441 (Poirot 327ee29-t2438-ask5-tracka-review.md, Significant 4, D-SLM5438): the
        # design's own text (Sec2.5) states "exactly one marker" as this check's contract, but
        # the code here used to accept ANY count >= 1 -- a docstring describing a check the
        # code did not perform, and the gap this section closes. Several registered sites
        # (GS-09..GS-12, GS-19, GS-26, GS-27 among them) legitimately touch more than one code
        # location, so "exactly one" is not the right invariant EITHER -- the fix is the
        # registry's own `expected_marker_count` field (one count per site, derived by reading
        # that site's own diff/current tree, not a blanket relaxation to "at least one"): a
        # site now reports both too few AND too many markers, closing what M5 in the same
        # review names as the first thing the old, weaker form let through (a marker sitting on
        # the wrong declaration is a count mismatch, not silently absorbed as "at least one").
        expected = registry_by_id[gs_id].get("expected_marker_count")
        if expected is None:
            failures.append(f"MISSING expected_marker_count: {gs_id} has status="
                             f"{registry_by_id[gs_id]['status']!r} (requires a marker) but the "
                             f"registry names no expected_marker_count for it")
        elif len(locs) != expected:
            failures.append(f"MARKER COUNT MISMATCH: {gs_id} ({registry_by_id[gs_id]['function']}) "
                             f"has {len(locs)} marker(s) at {locs}, want exactly {expected} per the "
                             f"registry's own expected_marker_count")
    for gs_id, locs in found_markers.items():
        if gs_id not in registry_by_id:
            failures.append(f"ORPHAN MARKER: {gs_id} at {locs} has no registry entry")
        elif gs_id in exempt_ids:
            failures.append(f"UNEXPECTED MARKER: {gs_id} is registered status=not-yet-built "
                             f"(Track B's own scope) but a marker already exists at {locs}")

    # --- Part 2: pattern <-> marker coverage (families R1, QOW only, production source
    #     only -- see module docstring and _PART2_ALLOWED_PREFIXES's own comment). ---
    # T-2475 fold-in (Poirot 6597903-t2472-ask5-tracka-confirmation.md Observation, D-SLM5617):
    # tracks which `_PART2_EXCLUDED_TEXT` entries actually matched something during this walk, so
    # a dead entry (its excused statement deleted, moved, or reworded elsewhere) is reported
    # rather than left silently inert -- see the check right after this loop. T-2481 (D-SLM5651/
    # D-SLM5670): a `dict` counting non-comment-line matches, not a `set` recording presence, so a
    # LITERAL DUPLICATE of an excused fragment elsewhere in its own file -- which a presence check
    # excises for free, exit 0, unaudited site in the tree -- is caught by the same tally rather
    # than requiring a second structure.
    matched_exclusions: dict[tuple[str, str], int] = {}
    for path in _iter_source_files(repo_root, production_only=True):
        if os.path.basename(path) in ("geometry_site_census.py", "geometry_site_registry.json"):
            continue
        rel_path = os.path.relpath(path, repo_root)
        if rel_path in _PART2_EXCLUDED_FILES:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, OSError):
            continue
        # Which lines in THIS file carry a marker, for the "within a fixed line window" test.
        marker_lines = [i for i, line in enumerate(lines, start=1) if _MARKER_RE.search(line)]
        ext = os.path.splitext(path)[1]
        comment_prefixes = ("#",) if ext == ".py" else ("//", "*", "/*")

        def _covered(line_no: int, window: int = 25) -> bool:
            return any(abs(line_no - ml) <= window for ml in marker_lines)

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            # T-2432's own narrowing: a pure-comment/prose line is exempt from Part 2 -- the
            # census's job is to catch CODE that misuses hidden_size, not prose that names the
            # relation while explaining a fix (every marked site's own explanatory comment
            # mentions hidden_size and q_proj/head_dim by construction). A comment line that
            # hides a genuine unmarked defect is still caught the moment the defect becomes
            # code, since code and its explaining comment are never the ONLY two lines in a
            # diff -- but this narrowing is stated, not silently applied: see this ticket's own
            # build log for the same disclosure the production-only file scope above carries.
            if stripped.startswith(comment_prefixes):
                continue
            # T-2475: excise the excused fragment's own text (if any) rather than skipping the
            # whole line -- see `_part2_excise_excluded_text`'s own docstring and the header
            # comment above `_PART2_EXCLUDED_TEXT` for why. `scan_line` is what the pattern
            # regexes see; failure messages below still quote the real, un-excised `line` so a
            # human reading a finding sees the actual source text.
            scan_line = _part2_excise_excluded_text(rel_path, line, matched_exclusions)
            hit = None
            if _R1_DIVISION_RE.search(scan_line) or _r1_multiply_hit(scan_line):
                hit = "R1"
            elif _qow_hit(scan_line):
                hit = "QOW"
            if hit and not _covered(i):
                failures.append(f"UNMARKED {hit} PATTERN HIT: {os.path.relpath(path, repo_root)}:{i}: "
                                 f"{line.strip()}")

    for f, text in _PART2_EXCLUDED_TEXT:
        count = matched_exclusions.get((f, text), 0)
        if count == 0:
            failures.append(
                f"DEAD EXCLUSION: _PART2_EXCLUDED_TEXT entry for {f!r} ({text!r}) matched no "
                f"non-comment line in that file during this walk -- the excused statement may "
                f"have been deleted, moved, or reworded; remove this entry or update its text to "
                f"match the current tree")
        elif count > 1:
            # T-2481 (D-SLM5651/D-SLM5670): a literal copy of an excused fragment anywhere else
            # in its own file is excised for free by a presence check, hiding a genuinely new,
            # unregistered site at exit 0 -- executed, `include/superslm/adapter_marshal.h`. An
            # exclusion excuses exactly the ONE statement it names; a second literal copy is a
            # second, unaudited candidate this exclusion would otherwise hide.
            failures.append(
                f"AMBIGUOUS EXCLUSION: _PART2_EXCLUDED_TEXT entry for {f!r} ({text!r}) matched "
                f"{count} non-comment lines in that file during this walk, not exactly one -- an "
                f"exclusion excuses exactly the statement it names; re-derive the fragment so it "
                f"is unique in this file, or register each additional occurrence as its own "
                f"reviewed site")

    # --- Part 3: per-site regression check (T-2441, S2 fix, D-SLM5436). ---
    # Part 1 proves a marker exists somewhere in the tree; Part 2 proves no UNMARKED pattern
    # hit exists. Neither reads what the code AT a marked location actually says -- a "fixed"
    # site whose own fix is reverted in place, with its marker left untouched, satisfies both:
    # the marker is still there (Part 1), and the reverted line is within its own marker's
    # exemption window (Part 2). Demonstrated by construction (Poirot 327ee29-t2438-ask5-
    # tracka-review.md Significant 2): GS-14's fix reverted to `plan.out_channels =
    # hidden_size;`, marker untouched, both parts PASS.
    #
    # This part closes that gap for every `fixed` site that carries a `required_tokens` list
    # (registry's own field, one substring set per site, derived once from that site's own
    # diff against v1.3.0): for each marker OCCURRENCE (a site can have more than one, per
    # Part 1's own "several registered sites legitimately touch more than one call site"), a
    # bounded scope is derived and at least one of the site's own required tokens must appear
    # somewhere in it. A revert that keeps the marker but restores the pre-fix code removes the
    # token that scope would have contained (every landed fix introduces its own named quantity
    # -- effective_q_width, g_q_width, QWIDTH, or similar -- exactly because that is what
    # distinguishes the fix from what it replaced), so the site fails here instead of passing
    # silently. `confirmed-correct` sites carry no `required_tokens` (their own governing
    # quantity is correctly UNCHANGED by this ask, so a presence check would be backwards for
    # them) and are not checked by this part.
    #
    # T-2481 (Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md F1, D-SLM5647/5659):
    # every STORED-COORDINATE keying this scope has ever used has been defeated by an edit its
    # own author did not imagine -- (path, line) by a new site written AT the excluded line
    # (T-2462); (path, text) substring by a statement appended to an excused line (T-2468,
    # that was Part 2's own class); occurrence-ORDINAL plus offset-from-marker (T-2475) by
    # REORDERING two of a multi-occurrence site's own existing occurrences in their shared file
    # -- nothing added or removed, marker count unchanged, Part 1 blind to it -- which desyncs
    # the ordinal-to-offset lookup from the occurrence it was meant to bound: the relocated
    # occurrence borrows a NEIGHBOR's own offset, which can spill past its own governed code
    # into unrelated downstream context that coincidentally satisfies the required-token check,
    # silently absorbing a genuine revert while citing an untouched line (demonstrated on GS-10;
    # reproduced here on GS-11, which the prior commissioning's own exposure list omitted --
    # see the registry's own header comment and this ticket's fix log). A fourth ORDINAL-shaped
    # keying was ruled out in advance (D-SLM5659): the defect is not which position is stored,
    # it is that ANY position (a proxy for where an occurrence's own code sits) drifts from that
    # code the moment the file's own layout changes around it, independent of the code's own
    # correctness.
    #
    # The replacement is CONTENT-ADDRESSED rather than position-keyed: `required_token_scopes`
    # (registry, below) maps each file to a LIST of scope records, `{"anchor": <str-or-null>,
    # "offset": <int-or-null>}`. A record's `anchor`, when not null, is an exact, literal
    # substring of that occurrence's own local code -- chosen, by hand, to be present whether
    # the occurrence's own fix is intact or reverted (never itself one of the site's own
    # `required_tokens`, so a revert cannot make the anchor disappear along with the thing it
    # is supposed to help detect) -- expected to appear somewhere within THAT record's own
    # claimed window, `[marker_line, marker_line + offset]`. For each physical marker occurrence
    # found in a file (order irrelevant), every one of that gs_id's own registered records is
    # tested against that marker's own local text; a record whose anchor is null applies only
    # when this gs_id has exactly ONE physical occurrence in this file (nothing to disambiguate
    # against -- ordinal is a degenerate, always-unambiguous identifier for a population of one,
    # which is why it is kept, unchanged, for the 17 of 20 `fixed`-with-`required_tokens` sites
    # that are single-occurrence-per-file; see the registry's own header comment for the
    # boundary this draws). Reordering two occurrences moves each one's own code -- anchor
    # included -- as one physical unit, so content-based lookup finds the SAME occurrence's own
    # correct offset regardless of which one now comes first in the file: there is no position
    # left in the key for a reorder to desync. Two registered anchors may legitimately match the
    # SAME set of candidates when two occurrences share literally identical local code AND an
    # identical offset (GS-12's two `ctx_wide` declarations); that is harmless by construction --
    # misassigning between them changes nothing, since both records agree on what to check. A
    # marker matching zero registered anchors, or matching anchors that disagree on offset, is
    # reported by name rather than silently guessed at (`MISSING SCOPE ANCHOR` /
    # `AMBIGUOUS SCOPE ANCHOR`, below) -- the same fail-closed posture Part 2's own dead/
    # ambiguous-exclusion checks hold.
    for rel_path, occurrences in markers_by_file.items():
        lines = file_lines_cache[rel_path]
        rel_path_fwd = rel_path.replace(os.sep, "/")
        ext = os.path.splitext(rel_path)[1]
        # T-2441: deliberately NOT Part 2's own `("//", "*", "/*")` tuple -- this codebase's
        # own pervasive `/*name=*/value` inline-argument-annotation idiom (e.g.
        # `/*layer_budget=*/num_hidden_layers`, seen throughout forward_sites.cpp/
        # superslm_gpu.cpp) opens a block comment that CLOSES on the same line, followed by
        # real code; a bare `/*` prefix check treats that whole line as comment-only and
        # excludes it, which for Part 2 only means "one fewer place a hit could fire"
        # (safe-direction over-exclusion) but for Part 3 turns a line carrying the required
        # token into an invisible one -- executed and found doing exactly that: GS-26's own
        # `/*q_width=*/num_attention_heads * head_dim);` line was excluded and produced a
        # false REGRESSED SITE on an unmodified, correct tree. `*` alone is kept (still
        # correctly excludes a `/* ... */` block's own continuation lines, this codebase's
        # multi-line-comment convention elsewhere).
        comment_prefixes = ("#",) if ext == ".py" else ("//", "*")

        # Group this file's own markers by gs_id -- content-addressing (T-2481) matches each
        # gs_id's own physical occurrences against that gs_id's own registered scope records,
        # independent of any other site sharing the file.
        marker_lines_by_gs_id: dict[str, list[int]] = {}
        for marker_line, gs_id in occurrences:
            marker_lines_by_gs_id.setdefault(gs_id, []).append(marker_line)

        for gs_id, marker_lines in marker_lines_by_gs_id.items():
            site = registry_by_id.get(gs_id)
            if site is None or site["status"] != "fixed":
                continue
            required = site.get("required_tokens")
            if not required:
                continue
            records = site.get("required_token_scopes", {}).get(rel_path_fwd, [])
            if not records:
                failures.append(
                    f"MISSING required_token_scopes ENTRY: {gs_id} has required_tokens but no "
                    f"registry scope record for {rel_path_fwd!r} ({len(marker_lines)} marker(s) "
                    f"found for this site in this file at lines {sorted(marker_lines)}) -- add "
                    f"one per physical occurrence (an `anchor` -- null only if this gs_id has "
                    f"exactly one occurrence in this file -- plus the OFFSET, in lines, scanning "
                    f"forward from the marker on the correct tree, at which the site's own "
                    f"required token is found; null offset if this occurrence has no "
                    f"independently-revertible code of its own to check)")
                continue

            for marker_line in sorted(marker_lines):
                # Which of this gs_id's own registered records apply to THIS physical marker,
                # decided by content rather than position: a null-anchor record applies only
                # when this gs_id has exactly one occurrence in this file (nothing to
                # disambiguate against); any other record applies when its own `anchor` text is
                # found somewhere within ITS OWN claimed window, [marker_line, marker_line +
                # offset]. An exempt record (offset null) has no natural offset of its own to
                # bound the search, so its anchor is searched within a generous fixed window
                # instead (`_ANCHOR_SEARCH_WINDOW_WHEN_EXEMPT`) -- this window only decides
                # WHICH record an exempt occurrence matches, never a required-token check (an
                # exempt occurrence has none), so a generous bound costs nothing in precision.
                candidates = []
                for rec in records:
                    anchor = rec.get("anchor")
                    offset = rec.get("offset")
                    if anchor is None:
                        if len(marker_lines) == 1:
                            candidates.append(rec)
                        continue
                    window_end = (marker_line + _ANCHOR_SEARCH_WINDOW_WHEN_EXEMPT if offset is None
                                  else marker_line + offset)
                    window = lines[marker_line - 1:window_end]
                    if any(anchor in ln for ln in window):
                        candidates.append(rec)

                if not candidates:
                    failures.append(
                        f"MISSING SCOPE ANCHOR: {gs_id} at {rel_path}:{marker_line} -- none of "
                        f"this site's registered scope anchors for {rel_path_fwd!r} were found "
                        f"near this occurrence; its own governing code may have moved, been "
                        f"reverted past recognition, or the registry needs re-deriving")
                    continue
                distinct_offsets = {rec.get("offset") for rec in candidates}
                if len(distinct_offsets) > 1:
                    failures.append(
                        f"AMBIGUOUS SCOPE ANCHOR: {gs_id} at {rel_path}:{marker_line} -- "
                        f"{len(candidates)} registered scope anchors matched with disagreeing "
                        f"offsets {sorted(distinct_offsets, key=lambda x: (x is None, x))!r}; "
                        f"re-derive the anchors so each physical occurrence matches unambiguously")
                    continue

                offset = candidates[0].get("offset")
                if offset is None:
                    # Explicit exemption (registry header comment documents when this is
                    # correct): this occurrence has no code of its own whose regression Part 3
                    # could detect.
                    continue
                scope_end = marker_line + offset
                # 0-indexed slice: lines[marker_line - 1 : scope_end] covers 1-based lines
                # [marker_line, scope_end], i.e. the marker's own line through this occurrence's
                # own content-addressed end line.
                #
                # Comment-only lines are excluded from the search -- executed and found load-
                # bearing, not a narrowing carried over by assumption from Part 2: every fix's
                # own explanatory comment (placed immediately below its marker, by this same
                # ticket's own authoring convention) names the exact quantity the fix introduces
                # in prose ("q_proj's real output width is q_width, not hidden_size"), so a
                # mutant that reverts the CODE line while leaving the marker AND its comment
                # block untouched -- precisely the construction Significant 2's own must-reject
                # uses -- would read the token out of the comment and report a false PASS if
                # comments were included. A first version of this part searched the whole scope,
                # comments included, and was shown by this exact construction to miss GS-14's
                # own reverted mutant before this exclusion was added.
                scope_lines = [
                    ln for ln in lines[marker_line - 1:scope_end] if not ln.strip().startswith(comment_prefixes)
                ]
                scope_text = "".join(scope_lines)
                # T-2445 (Significant 2 remedy, GS-18): most sites use OR semantics -- different
                # occurrences of the SAME site spell its own quantity differently (GS-10's own
                # two occurrences use "effective_q_width" and "QW" respectively), so any ONE
                # listed token satisfies. A site whose ONE occurrence governs several
                # independent sub-assignments that all share the same generic token text
                # (GS-18's q_codes/q_rot/ctx_codes, each `codes_block(effective_q_width)`) needs
                # the opposite: reverting just one sub-assignment leaves the bare token
                # "effective_q_width" present via the others (or via this function's own shared
                # derivation line), so a presence-of-any check never fires. `require_all_tokens`
                # (registry field, default false) switches that site to AND semantics over
                # exact, per-assignment fragments (`required_tokens` then holds one fragment per
                # sub-assignment, not alternates).
                require_all = site.get("require_all_tokens", False)
                if require_all:
                    missing_toks = [tok for tok in required if tok not in scope_text]
                    ok = not missing_toks
                else:
                    missing_toks = required
                    ok = any(tok in scope_text for tok in required)
                if not ok:
                    verb = "missing" if require_all else "none of"
                    shown = missing_toks if require_all else required
                    failures.append(
                        f"REGRESSED SITE: {gs_id} at {rel_path}:{marker_line} -- {verb} "
                        f"{shown!r} found within its own bounded scope; the fix may have been "
                        f"reverted with its marker left in place")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=_DEFAULT_REPO_ROOT)
    args = ap.parse_args()
    failures = run_census(args.repo_root)
    if failures:
        print(f"geometry_site_census: FAIL -- {len(failures)} finding(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    registry = _load_registry()
    marked = sum(1 for s in registry if s["status"] in ("fixed", "confirmed-correct"))
    print(f"geometry_site_census: PASS -- {marked} site(s) fixed-and-marked or "
          f"confirmed-correct-and-marked, 0 unmarked R1/QOW pattern hits "
          f"({len(registry) - marked} site(s) exempt, Track B's own not-yet-built scope)")
    print("  KNOWN LIMITATION (T-2468 F2, dated 2026-08-31, pre-existing): Part 2 matches R1/QOW "
          "co-occurrence one physical line at a time -- a genuinely new site whose co-occurring "
          "tokens land on different physical lines is not covered by this PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
