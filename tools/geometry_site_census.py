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
  3. Per-site regression check (T-2441, Poirot 327ee29-t2438-ask5-tracka-review.md,
     Significant 2, D-SLM5436; scope bounding corrected T-2445, Poirot
     ddbc57a-t2443-ask5-tracka-confirmation.md, Significant 2, D-SLM5436 superseded).
     Parts 1 and 2 both read whether a marker EXISTS; neither reads what the code AT a
     marked location actually says, so a "fixed" site whose own fix is reverted in place,
     marker left untouched, satisfies both. For every `fixed` site that carries a
     `required_tokens` list (the registry's own field, one substring set per site), at
     least one of those tokens must appear in that OCCURRENCE's own bounded scope -- the
     registry's own `required_token_scope_ends` map, keyed by exact marker location, one
     entry per occurrence (see the registry's own header comment for how each value is
     derived). A revert that restores the pre-fix code removes the token the landed fix
     itself introduced, so this part fails where parts 1 and 2 do not. Demonstrated by
     construction: GS-14's fix reverted to `plan.out_channels = hidden_size;`, marker
     untouched, passes parts 1 and 2 and fails here.

     T-2445 correction: scope used to run from a marker to the NEXT marker anywhere in the
     same file, which (a) swept unrelated code between two distant, unrelated sites'
     markers into the search -- a required token appearing there by coincidence masked a
     revert of the actual site -- and (b) for a multi-line site, let an untouched sibling
     line elsewhere in that same wide scope mask a revert of the one line a specific
     mutation touched. Executed and found doing exactly that: three independent single-line
     reverts (GS-12's o_proj in-width, GS-10's packed q_weight byte extent, GS-18's
     LayerScratch q_codes width) all passed under the old rule. The registry's own
     per-occurrence `required_token_scope_ends` closes both: each value is the minimal line
     (scanning forward from the marker on the correct tree) at which the site's own
     required token is actually found, so the bounded scope is exactly as wide as that
     occurrence's own governed code.

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
# coordinate that can drift away from it -- each fragment below is verified unique among this
# file's own non-comment lines (`_part2_excluded_text_hit`'s whole match surface), so widening
# match is bounded to "another literal copy of this exact excused statement," not "whatever
# lands on this line number next."
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
    (os.path.join("include", "superslm", "model.h"), "ConfigGeometryHiddenSizeMismatch,"),
    (os.path.join("include", "superslm", "proof_manifest.h"), "HiddenSizeGeometryMismatch,"),
}


def _part2_excluded_text_hit(rel_path: str, line: str) -> bool:
    """True when `line` (from `rel_path`) carries one of `_PART2_EXCLUDED_TEXT`'s own excused
    fragments -- a substring match on the fragment's exact text, not the line's position, so an
    unrelated line above it shifting the excused code down the file changes nothing here."""
    return any(rel_path == f and text in line for f, text in _PART2_EXCLUDED_TEXT)


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
            if _part2_excluded_text_hit(rel_path, line):
                continue
            hit = None
            if _R1_DIVISION_RE.search(line) or _r1_multiply_hit(line):
                hit = "R1"
            elif _qow_hit(line):
                hit = "QOW"
            if hit and not _covered(i):
                failures.append(f"UNMARKED {hit} PATTERN HIT: {os.path.relpath(path, repo_root)}:{i}: "
                                 f"{line.strip()}")

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
    # Part 1's own "several registered sites legitimately touch more than one call site"), the
    # scope is [marker_line, required_token_scope_ends[file:marker_line]] -- the registry's own
    # per-occurrence bound (T-2445; see the registry's own header comment and this module's
    # docstring for why "next marker in the file" was replaced) -- and at least one of the
    # site's own required tokens must appear somewhere in that scope. A revert that keeps the
    # marker but restores the pre-fix code removes the token that scope would have contained
    # (every landed fix introduces its own named quantity -- effective_q_width, g_q_width,
    # QWIDTH, or similar -- exactly because that is what distinguishes the fix from what it
    # replaced), so the site fails here instead of passing silently. `confirmed-correct` sites
    # carry no `required_tokens` (their own governing quantity is correctly UNCHANGED by this
    # ask, so a presence check would be backwards for them) and are not checked by this part.
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
        sorted_occ = sorted(occurrences, key=lambda p: p[0])
        for idx, (marker_line, gs_id) in enumerate(sorted_occ):
            site = registry_by_id.get(gs_id)
            if site is None or site["status"] != "fixed":
                continue
            required = site.get("required_tokens")
            if not required:
                continue
            scope_ends = site.get("required_token_scope_ends", {})
            occ_key = f"{rel_path_fwd}:{marker_line}"
            if occ_key not in scope_ends:
                failures.append(
                    f"MISSING required_token_scope_ends ENTRY: {gs_id} has required_tokens but "
                    f"no registry scope-end for occurrence {occ_key!r} -- add one (the minimal "
                    f"line, scanning forward from the marker on the correct tree, at which the "
                    f"site's own required token is found), or null if this occurrence has no "
                    f"independently-revertible code of its own to check")
                continue
            scope_end = scope_ends[occ_key]
            if scope_end is None:
                # Explicit exemption (registry header comment documents when this is correct):
                # this occurrence has no code of its own whose regression Part 3 could detect.
                continue
            # 0-indexed slice: lines[marker_line - 1 : scope_end] covers 1-based lines
            # [marker_line, scope_end], i.e. the marker's own line through the registry's own
            # recorded end line for this specific occurrence.
            #
            # Comment-only lines are excluded from the search -- executed and found load-
            # bearing, not a narrowing carried over by assumption from Part 2: every fix's own
            # explanatory comment (placed immediately below its marker, by this same ticket's
            # own authoring convention) names the exact quantity the fix introduces in prose
            # ("q_proj's real output width is q_width, not hidden_size"), so a mutant that
            # reverts the CODE line while leaving the marker AND its comment block untouched --
            # precisely the construction Significant 2's own must-reject uses -- would read the
            # token out of the comment and report a false PASS if comments were included. A
            # first version of this part searched the whole scope, comments included, and was
            # shown by this exact construction to miss GS-14's own reverted mutant before this
            # exclusion was added.
            scope_lines = [
                ln for ln in lines[marker_line - 1:scope_end] if not ln.strip().startswith(comment_prefixes)
            ]
            scope_text = "".join(scope_lines)
            # T-2445 (Significant 2 remedy, GS-18): most sites use OR semantics -- different
            # occurrences of the SAME site spell its own quantity differently (GS-10's own two
            # occurrences use "effective_q_width" and "QW" respectively), so any ONE listed
            # token satisfies. A site whose ONE occurrence governs several independent
            # sub-assignments that all share the same generic token text (GS-18's q_codes/
            # q_rot/ctx_codes, each `codes_block(effective_q_width)`) needs the opposite:
            # reverting just one sub-assignment leaves the bare token "effective_q_width"
            # present via the others (or via this function's own shared derivation line), so a
            # presence-of-any check never fires. `require_all_tokens` (registry field, default
            # false) switches that site to AND semantics over exact, per-assignment fragments
            # (`required_tokens` then holds one fragment per sub-assignment, not alternates).
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
