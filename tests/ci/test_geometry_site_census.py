"""Red suite for tools/geometry_site_census.py, T-2475: closes the census-exclusion class
Claude/Mendeleev/t2468-census-recommissioning-2026-08-31.md found NOT COMMISSIONED (F1,
D-SLM5603) -- `_part2_excluded_text_hit`'s substring match (`text in line`) excused the WHOLE
physical line the instant an excluded fragment's text appeared anywhere in it, so a genuinely
new, unregistered geometry-defect statement appended onto the SAME line as an already-excused
fragment was silently absorbed: PASS, exit 0, unaudited site in the tree. This is the second
instance of one class -- the first (line-keyed, T-2467) excused a whole line on a stale line
number; the second (text-keyed, this file) excused a whole line on a partial text match. Both
share the same shape: the exclusion excuses more than the thing it names.

T-2475's own property: an exclusion excuses exactly the statement it names and nothing else on
that line. `_part2_excise_excluded_text` removes the excused fragment's own text from a line
before the pattern regexes see it, rather than skipping the whole line -- new content sharing
the line, whichever side of the fragment it lands on, is left in the remainder and scanned like
ordinary code. `model.h`'s and `proof_manifest.h`'s own `_PART2_EXCLUDED_TEXT` fragments were
widened from a uniqueness-only prefix (the enumerator name alone) to their own full stripped
line, because their trailing explanatory comment -- not the enumerator declaration -- is what
actually trips `_r1_multiply_hit`; excising only the prefix would have left that comment's own
hidden_size/head_dim/num_attention_heads/`*` text in the remainder and turned the untouched
baseline red.

Structure, in the order `StandardsDocument.md` Sec4/Sec5.4 requires (population derived and
verified at source before the mechanism was built, mechanism built, population reproduced
against the built mechanism):

  - Mechanism cells (`test_excise_*`): `_part2_excise_excluded_text` in isolation, synthetic
    lines, no file I/O -- proves the excision itself, independent of the real tree.
  - Real-tree population cells, against the REAL checkout via `_mutated` (mutate one real
    production file for the duration of one test, restore its exact original bytes after --
    this module imports `census` ONCE, at collection time, so `census._REGISTRY_PATH` is bound
    to this repo's own `tools/` directory for every call in this file; calling
    `census.run_census(tmp_root)` against a synthetic tmp tree would still load THIS repo's real
    registry, mismatched against the tmp tree's own paths, not a copy of it -- a fresh subprocess
    invoking a copy of `tools/` alongside a copy of `src/`/`include/` genuinely does carry its own
    registry (Claude/Poirot/dcefab3-t2486-census-content-keying-confirmation.md's own lab does
    exactly this), which this suite does not use because every cell here calls the already-
    imported module in-process; mirrors T-2468's own mutate/run/revert discipline):
      * `test_real_tree_baseline_is_clean_today` -- cell zero, unmutated.
      * `test_must_accept_*` (T-2468 Sec1, D-SLM5601) -- comment inserted above each of the
        three excluded lines; must stay PASS (position-independence, unaffected by this fix).
      * `test_must_reject_*` (T-2468 Sec2, D-SLM5602) -- a genuinely new, unregistered site in
        a zero-marker production header; must FAIL and cite the injected line.
      * `test_defeats_the_prior_mechanism_*` (T-2468 Sec3, D-SLM5603 -- THE finding this ticket
        closes) -- a genuinely new statement appended onto the SAME physical line as an
        already-excused fragment, on each of the three exclusions; must now FAIL (was PASS
        under `_part2_excluded_text_hit`).
      * `test_prepended_content_on_an_excused_line_is_also_caught` -- this ticket's own added
        construction: new content BEFORE the fragment, not just after -- the excise-and-scan
        property is symmetric, unlike a hypothetical "only widen at the end" remedy.
      * `test_safe_direction_rename_and_reformat_*` (T-2468 Sec4a cases 1-2, D-SLM5604) -- the
        brief's own named safe-direction Note: renaming or reformatting the excused line itself
        still re-triggers the census (a real cost, not a hole a defect can hide in). Confirms
        this fix does not make the exclusion stickier.
      * `test_wrap_comment_split_is_now_reported_as_a_dead_exclusion_not_a_silent_pass` (T-2468
        Sec4a case 3, reclassified by T-2475's own fragment-widening + dead-entry check) --
        splitting the excused declaration from its own comment removes the pattern-triggering
        co-occurrence entirely AND breaks the widened fragment's exact-text match; the case is
        no longer a silent, uninformative PASS -- it is now a named DEAD EXCLUSION finding.
      * `test_two_line_split_is_the_documented_f2_limitation_still_open` (T-2468 Sec4a case 4,
        F2) -- a genuinely new QOW site split across two physical lines still evades Part 2;
        pre-existing, not this ticket's own diff (Part 2's per-line hit detection is unchanged),
        reproduced here so the PASS this case still returns is a proven, not assumed, fact, and
        cross-checked against the KNOWN LIMITATION text `main()` now prints.

T-2481 fold-in (Claude/Bach/briefs/t2481.md; Claude/Mendeleev/t2480-census-recommissioning-
2026-08-31.md F1, D-SLM5647/D-SLM5659; Claude/Poirot/f363c2a-t2479-census-class-confirmation.md
Significant 1/2, D-SLM5651/D-SLM5652): four sources bundled into one round --

  - Part 3's occurrence-ordinal keying is defeated by REORDERING two existing occurrences of a
    same-file, multi-occurrence site -- nothing added or removed, marker count unchanged, Part 1
    blind to it -- which desyncs the ordinal-to-offset lookup and silently absorbs a genuine
    revert while citing an untouched line (`test_part3_reorder_*`, below). Replaced with
    CONTENT-ADDRESSED occurrence identification (`required_token_scopes`, registry) -- see
    `tools/geometry_site_census.py`'s own Part 3 header comment and the registry's own header
    comment for the full account.
  - Part 2's exclusion mechanism excuses EVERY literal copy of an excused fragment in its own
    file, not only the one statement it names -- a genuinely new, unregistered site written as a
    literal copy of an excused statement is excised for free (`test_part2_ambiguous_exclusion_*`,
    below). Closed by counting non-comment-line matches instead of recording mere presence.
  - Two of the four cells pinning T-2475's own headline remedy exercised text a compiler never
    sees (append AFTER a fragment's own trailing `//` comment); the two prior cells renamed to
    state precisely what they do prove (a dead exclusion, not a caught defect statement). T-2481
    also added two cells over what it believed was the producible shape for these two files
    (insert BEFORE the trailing comment) -- corrected below, T-2491: it is not producible.
  - The S2 red-check comment block's own cell count ("exactly four") is corrected to what a
    full, unfiltered run of this file actually reddens under the same reversion.

T-2491 fold-in (Claude/Bach/briefs/t2491.md; Claude/Poirot/dcefab3-t2486-census-content-keying-
confirmation.md Significant 1/2, D-SLM5716/D-SLM5717/D-SLM5718; D-SLM5732): fix round on T-2486's
confirmation of T-2481's diff --

  - T-2481's own `test_defeats_the_prior_mechanism_same_line_insert_before_comment_model_h`/
    `_proof_manifest_h` were believed to pin the producible same-line shape for `model.h` and
    `proof_manifest.h`. Both excused lines are enumerators inside an `enum class` body
    (`SslmModelStatus`, `ConfigGeometryStatus`); an enum body admits enumerators, not statements,
    so the construction does not compile -- executed with the CI's own pinned compiler, clang
    18.1.8, `-fsyntax-only -std=c++20`: `error: missing ',' between enumerators` on both headers.
    The cells also have zero discrimination, executed both with the QOW family forced false and
    with the injected identifier replaced by an inert statement: the assertion still holds either
    way, because what actually fires is the trailing prose comment's own R1 co-occurrence once
    the exclusion's exact-text match goes dead, not recognition of the injected statement. Deleted
    rather than replaced with a third attempt at the SAME shape.

    T-2491's own replacement text (D-SLM5732, D-SLM5740) went further than the compile finding
    supports: "there is no producible same-line geometry defect an enum body can host." Refuted
    by construction (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Significant 1,
    D-SLM5756/D-SLM5757): an enum body admits enumerators, not statements, but a same-line
    ENUMERATOR prepend is both legal C++ and enough. Because the excused fragment is the whole
    stripped line, a prepend leaves it intact -- the exclusion still fires and the injected text
    is scanned as ordinary remainder. Executed at the CI's pinned compiler, clang 18.1.8,
    `-fsyntax-only -std=c++20`: `q_proj_out_channels_hidden_size, ` prepended to `model.h`'s
    fragment and `o_proj_out_channels_hidden_size, ` prepended to `proof_manifest.h`'s both
    compile clean and fire `UNMARKED QOW PATTERN HIT`, discriminating both ways (silent with the
    QOW family disabled, silent with an inert enumerator in place of the defect). T-2497 adds one
    cell per header (`test_defeats_the_prior_mechanism_same_line_enumerator_prepend_model_h`/
    `_proof_manifest_h`, below) pinning exactly this. The corrected, narrower fact: a same-line
    *statement* is not producible inside an `enum class` body; a same-line *enumerator* is, and
    the census catches it. `adapter_marshal.h`'s excused line sits inside a function body, where
    both statement forms (`..._append_adapter_marshal`/
    `test_prepended_content_on_an_excused_line_is_also_caught`, below) are real.
  - `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_all`
    restored `tools/geometry_site_registry.json` via `open(path, "w", encoding="utf-8")`, which
    translates `\\n` to `\\r\\n` on this platform; the file is `attr/text eol=lf`, so a clean
    checkout came back `w/crlf` and ` M` after one run. Fixed with `newline=""` on both the write
    and the restore. `_mutated`, below, is hardened the same round -- not with a blanket
    `newline=""` (every `.h`/`.cpp`/`.hlsl` file it mutates is `attr/text`, checked out CRLF, and
    an unconditional `newline=""` would restore those as LF instead, the same defect in the other
    direction), but by detecting each target file's own newline convention from its raw bytes
    before writing, so the next `eol=lf` file it is pointed at inherits the correct behavior.
  - `AMBIGUOUS SCOPE ANCHOR` (Part 3's third fail-closed diagnostic) had no cell -- pinned by
    `test_part3_ambiguous_scope_anchor_fires_when_two_records_own_anchors_collide_in_one_window`,
    below, reproducing the reviewer's own construction.
  - GS-12's own nine occurrences are anchored on their own explanatory COMMENT text (this file's
    own Part 3 header comment explains why: a bare code prefix collided with unrelated, unmarked
    code once a reorder moved it), so the match key includes comment prose for all nine of GS-12's
    occurrences, through eight registered records (two occurrences, `occ2`/`occ7`, share one
    record -- see the registry's own header comment) -- a developer who rewords one of those
    comments, no code touched, reddens the
    census with `MISSING SCOPE ANCHOR` (Claude/Poirot/dcefab3-t2486-census-content-keying-
    confirmation.md Sec7 O2, D-SLM5726, executed: "the normed" to "the normalised" in one GS-12
    comment). This was disclosed nowhere a reader of the tool would find it; disclosed now in
    `tools/geometry_site_census.py`'s own Part 3 header comment and the registry's own header
    comment, both of which also say the correct response to that red: re-derive the occurrence's
    own anchor from the new comment text, not treat the finding as a caught regression.
  - This module's own docstring (below) and the S2 red-check comment block (further down this
    file) each carried a stale claim untouched by T-2481's diff -- corrected in place rather than
    superseded a third time; see each site's own comment for what was wrong and how it was
    checked.

T-2497 fold-in (Claude/Bach/briefs/t2497.md; Claude/Poirot/ba29de4-t2496-census-fixes-
confirmation.md Significant 1/2, D-SLM5756/D-SLM5757/D-SLM5758): fix round on T-2496's
confirmation of T-2491's diff --

  - T-2491's own replacement fact for the two deleted pins -- "there is no producible same-line
    geometry defect an enum body can host" -- overclaimed. Refuted by construction: an enum body
    admits enumerators, not statements, but a same-line ENUMERATOR prepend is legal C++, compiles
    clean at the CI's pinned compiler, and fires the census from the excused fragment's own
    remainder. Two cells restore what the deletion left uncovered
    (`test_defeats_the_prior_mechanism_same_line_enumerator_prepend_model_h`/
    `_proof_manifest_h`, below); the corrected, narrower fact -- a same-line statement is not
    producible inside an enum class body, a same-line enumerator is, and is caught -- replaces
    the overclaim at every site it reached (this docstring, the S2 red-check comment block, both
    T-2481/T-2491 build logs, and D-SLM5732/D-SLM5740).
  - Neither production change T-2491 made is detectable by any cell in this file: reverting
    `newline=""` on `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_
    registry_record_at_all`'s own registry write still passes while the registry's own bytes
    change underneath it, and reverting `_mutated`'s convention-detection hardening leaves this
    whole file green. `_mutated_targets_and_registry_are_byte_identical_after_the_module_runs`,
    below, is a module-scoped, autouse fixture that snapshots the seven `_mutated` targets plus
    the registry before this module's own suite runs and asserts them byte-identical after --
    closing both halves of the dirty-checkout class at once, and every future cell that touches
    the real tree, not just the two named above.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import geometry_site_census as census  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_ADAPTER_H = os.path.join("include", "superslm", "adapter_marshal.h")
_MODEL_H = os.path.join("include", "superslm", "model.h")
_PROOF_H = os.path.join("include", "superslm", "proof_manifest.h")
_MATMUL_H = os.path.join("include", "superslm", "matmul.h")
_FORWARD_SITES_CPP_T2481 = os.path.join("src", "forward", "forward_sites.cpp")

_ADAPTER_FRAGMENT = 'if (proj == "q_proj" || proj == "o_proj" || proj == "down_proj") return hidden_size;'
_MODEL_FRAGMENT = "ConfigGeometryHiddenSizeMismatch,    // R1: hidden_size != num_attention_heads * head_dim"
_PROOF_FRAGMENT = "HiddenSizeGeometryMismatch, // hidden_size != num_attention_heads * head_dim -- R1, REMOVED"


@contextlib.contextmanager
def _mutated(rel_path: str, transform):
    """Mutates a real production file, at `rel_path` under the repo root, for the duration of
    the `with` block via `transform(original_text) -> new_text`, then restores the original
    content byte-for-byte. See this module's own docstring for why the real tree is mutated
    in place rather than exercised against a synthetic `tmp` repo root.

    T-2491 (Poirot dcefab3-t2486-census-content-keying-confirmation.md Significant 2, D-SLM5718):
    the write-back explicitly picks the target file's own newline convention rather than trusting
    the platform default. `.h`/`.cpp`/`.hlsl` files are `attr/text` (tool-native, `w/crlf` on this
    checkout) and the platform-default write already reproduced that -- the defect this closes is
    an `attr/text eol=lf` file (e.g. the JSON registry): the platform-default write translates
    every `\\n` to `\\r\\n` regardless of the file's own pinned convention, so a byte-for-byte
    restore of an eol=lf file silently comes back CRLF. Detected from the file's own raw bytes
    (not assumed from its extension), so a future `eol=lf` file this helper is pointed at inherits
    the correct behavior automatically rather than a second instance of this same defect."""
    full = os.path.join(_REPO_ROOT, rel_path)
    with open(full, "rb") as f:
        _uses_crlf = b"\r\n" in f.read()
    _write_newline = "\r\n" if _uses_crlf else ""
    with open(full, "r", encoding="utf-8") as f:
        original = f.read()
    try:
        with open(full, "w", encoding="utf-8", newline=_write_newline) as f:
            f.write(transform(original))
        yield
    finally:
        with open(full, "w", encoding="utf-8", newline=_write_newline) as f:
            f.write(original)


# T-2497 (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Significant 2, D-SLM5758):
# every real path `_mutated` is pointed at in this module, plus the registry -- the two files
# T-2491's own S2 remedy touches (`_mutated` itself, and the registry write in
# `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_all`)
# and the five more `_mutated` also restores. Executed by the reviewer: reverting `newline=""`
# on the registry cell's own writes still leaves that cell `1 passed` while the registry's own
# sha256 changes underneath it; reverting `_mutated`'s convention-detection hardening leaves
# this whole file green (every file it touches is pure CRLF today, so the hardening and the
# platform default agree on every input this suite has -- Claude/Poirot/ba29de4-t2496-census-
# fixes-confirmation.md Sec7). The dirty-checkout class this fixture closes can return with the
# suite green.
_MUTATED_TARGETS_AND_REGISTRY_PATHS = (
    _ADAPTER_H,
    _MATMUL_H,
    _MODEL_H,
    _PROOF_H,
    _FORWARD_SITES_CPP_T2481,
    os.path.join("src", "proof_manifest.cpp"),
    os.path.join("src", "gpu", "superslm_gpu.cpp"),
    os.path.join("tools", "geometry_site_registry.json"),
)


@pytest.fixture(scope="module", autouse=True)
def _mutated_targets_and_registry_are_byte_identical_after_the_module_runs():
    """Session-scoped in spirit, module-scoped in fact (this module's own `census` import is
    already bound to this repo's `tools/`, per this file's own docstring above) -- snapshots the
    raw bytes of every path in `_MUTATED_TARGETS_AND_REGISTRY_PATHS` BEFORE the first cell in
    this module runs, and asserts them byte-identical AFTER the last one has, whatever mix of
    `_mutated()` blocks and direct registry writes ran in between. Closes T-2496's Significant 2
    (D-SLM5758) at the root rather than per-cell: a fix that touches one of these paths and
    leaves it modified fails HERE regardless of what that fix's own cell asserts, so the next
    dirty-checkout regression cannot ship with this suite green the way this round's own did."""
    paths = [os.path.join(_REPO_ROOT, rel) for rel in _MUTATED_TARGETS_AND_REGISTRY_PATHS]
    before = {}
    for p in paths:
        with open(p, "rb") as f:
            before[p] = f.read()
    yield
    changed = []
    for p in paths:
        with open(p, "rb") as f:
            after = f.read()
        if after != before[p]:
            changed.append(os.path.relpath(p, _REPO_ROOT))
    assert not changed, (
        "this module's own suite left the checkout modified -- byte mismatch after the run on: "
        + ", ".join(changed)
    )


# --- Mechanism cells: _part2_excise_excluded_text in isolation. ---

def test_excise_removes_the_adapter_marshal_fragment_entirely():
    line = _ADAPTER_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(_ADAPTER_H, line)
    assert out.strip() == ""


def test_excise_leaves_appended_new_content_in_the_remainder():
    line = (_ADAPTER_FRAGMENT + ' if (proj == "k_proj") { uint64_t out_channels = hidden_size;'
            " return out_channels; }\n")
    out = census._part2_excise_excluded_text(_ADAPTER_H, line)
    assert "k_proj" in out
    assert "q_proj" not in out, "the excused fragment's own q_proj text must be gone from the remainder"


def test_excise_leaves_prepended_new_content_in_the_remainder():
    line = 'uint64_t sneaky_new_site = hidden_size; /* q_proj_alias */ ' + _ADAPTER_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(_ADAPTER_H, line)
    assert "sneaky_new_site" in out


def test_excise_does_not_touch_a_line_in_an_unregistered_file():
    line = _ADAPTER_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(os.path.join("include", "superslm", "other.h"), line)
    assert out == line


def test_excise_does_not_touch_an_unrelated_line_in_a_registered_file():
    line = "uint64_t unrelated_thing = 1;\n"
    out = census._part2_excise_excluded_text(_ADAPTER_H, line)
    assert out == line


def test_excise_removes_the_full_widened_model_h_fragment_including_its_own_comment():
    line = "\t" + _MODEL_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(_MODEL_H, line)
    assert "head_dim" not in out
    assert "hidden_size" not in out


def test_excise_removes_the_full_widened_proof_manifest_h_fragment_including_its_own_comment():
    line = "\t" + _PROOF_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(_PROOF_H, line)
    assert "head_dim" not in out
    assert "hidden_size" not in out


# --- Cell zero: the real, unmutated tree is clean today. ---

def test_real_tree_baseline_is_clean_today():
    assert census.run_census(_REPO_ROOT) == []


# --- Must-accept (T-2468 Sec1, D-SLM5601): comment inserted above each excluded line --
# position-independence, the property text-keying was built for, unaffected by this fix. ---

def test_must_accept_comment_above_adapter_marshal_exclusion():
    def _t(text):
        return text.replace(_ADAPTER_FRAGMENT, "// a harmless comment, no defect\n\t" + _ADAPTER_FRAGMENT, 1)
    with _mutated(_ADAPTER_H, _t):
        assert census.run_census(_REPO_ROOT) == []


def test_must_accept_comment_above_model_h_exclusion():
    def _t(text):
        return text.replace("\t" + _MODEL_FRAGMENT, "\t// a harmless comment, no defect\n\t" + _MODEL_FRAGMENT, 1)
    with _mutated(_MODEL_H, _t):
        assert census.run_census(_REPO_ROOT) == []


def test_must_accept_comment_above_proof_manifest_h_exclusion():
    def _t(text):
        return text.replace("\t" + _PROOF_FRAGMENT, "\t// a harmless comment, no defect\n\t" + _PROOF_FRAGMENT, 1)
    with _mutated(_PROOF_H, _t):
        assert census.run_census(_REPO_ROOT) == []


# --- Must-reject (T-2468 Sec2, D-SLM5602): a genuinely new, unregistered site in a
# zero-marker production header -- the obvious candidate a defeated exclusion would let hide. ---

def test_must_reject_new_unmarked_site_in_zero_marker_header():
    assert census._MARKER_RE.search(open(os.path.join(_REPO_ROOT, _MATMUL_H), encoding="utf-8").read()) is None, (
        "this cell's premise is a zero-marker file -- no marker-proximity window to absorb the new site"
    )
    injected = (
        "\ninline uint64_t T2475ProbeOutWidthFor(const std::string& proj, uint64_t hidden_size) {\n"
        '\tif (proj == "q_proj" || proj == "o_proj") return hidden_size;\n'
        "\treturn 0;\n"
        "}\n"
    )

    def _t(text):
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    with _mutated(_MATMUL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a new, unmarked QOW site in a zero-marker header must fail the census"
    assert any("matmul.h" in f and "q_proj" in f for f in failures)


# --- THE finding this ticket closes (T-2468 Sec3, D-SLM5603): a genuinely new statement
# appended onto the SAME physical line as an already-excused fragment. Reproduced on all
# three exclusions -- the mechanism is one function applied identically to each. ---

def test_defeats_the_prior_mechanism_same_line_append_adapter_marshal():
    def _t(text):
        return text.replace(
            _ADAPTER_FRAGMENT,
            _ADAPTER_FRAGMENT + ' if (proj == "k_proj") { uint64_t out_channels = hidden_size;'
                                 " return out_channels; }",
            1,
        )
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a new QOW statement appended to the same line as the excused fragment must now FAIL"
    assert any("adapter_marshal.h" in f and "k_proj" in f for f in failures)


def test_appending_after_model_h_s_trailing_comment_fires_on_apparatus_unsound_grounds():
    # T-2479 Significant 2 (Poirot f363c2a-t2479-census-class-confirmation.md Sec5, D-SLM5652):
    # this construction's own injected text lands AFTER the fragment's trailing `//` comment, so
    # from a real compiler's own point of view the ENTIRE remainder of the physical line --
    # including the appended `uint64_t o_proj_out_channels = hidden_size;` -- is commented-out
    # prose, never compiled, and cannot be a geometry site. Census.py does not know that: once
    # the fragment (enumerator PLUS its own trailing `// R1: ...` comment) is excised as one
    # exact-text match, only the appended text remains in the scanned line, and that text
    # independently looks QOW-shaped (`hidden_size` plus the `out_channels` substring inside its
    # own identifier) regardless of ever having sat inside a comment. Executed: this construction
    # DOES fire, on `UNMARKED QOW PATTERN HIT`, not merely a `DEAD EXCLUSION` -- but the
    # right-verdict-for-the-wrong-reason gap D-SLM5652 names still holds: nothing here proves the
    # census recognizes commented-out text as inert, only that this SPECIFIC probe's own
    # appended identifier happens to independently trip a pattern. T-2481 believed inserting new
    # code BEFORE the trailing comment (rather than after it) was the producible, genuinely
    # compiled defeat for these two files, and pinned it as a third cell; T-2491 (Poirot dcefab3-
    # t2486-census-content-keying-confirmation.md Sec4, D-SLM5716/D-SLM5717, D-SLM5732) found
    # that construction does not compile either -- `ConfigGeometryHiddenSizeMismatch` and
    # `HiddenSizeGeometryMismatch` are enumerators inside an `enum class` body, which admits
    # enumerators, not statements, so an insert between the enumerator and its own trailing
    # comment is exactly as uncompilable as an append after it. There is no producible same-line
    # STATEMENT for `model.h`/`proof_manifest.h`: the same-line-append class T-2468 found is
    # vacuous, for statements, on these two files and real only on `adapter_marshal.h` (whose
    # excused line sits inside a function body, not an enum) -- `test_defeats_the_prior_mechanism_
    # same_line_append_adapter_marshal` and `test_prepended_content_on_an_excused_line_is_also_
    # caught`, above, are that class's whole population. A same-line ENUMERATOR, by contrast, IS
    # producible and caught -- see `test_defeats_the_prior_mechanism_same_line_enumerator_prepend_
    # model_h`/`_proof_manifest_h`, below (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md
    # Significant 1, D-SLM5756/D-SLM5757). T-2491's own broader claim, "there is no producible
    # same-line geometry defect an enum body can host," is false and corrected here.
    def _t(text):
        return text.replace(
            "\t" + _MODEL_FRAGMENT,
            "\t" + _MODEL_FRAGMENT + "  uint64_t o_proj_out_channels = hidden_size; // T-2475 probe",
            1,
        )
    with _mutated(_MODEL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "appending after model.h's own trailing comment must still redden the census"
    assert any("UNMARKED QOW PATTERN HIT" in f and "model.h" in f and "out_channels" in f
               for f in failures), (
        "the appended identifier's own out_channels substring, not a recognition of commented-"
        "out prose, is what fires here -- see this cell's own docstring"
    )


def test_appending_after_proof_manifest_h_s_trailing_comment_fires_on_apparatus_unsound_grounds():
    # Same shape and same caveat as the model.h cell above (T-2479 Significant 2, D-SLM5652).
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\t" + _PROOF_FRAGMENT + "  uint64_t o_proj_out_channels = hidden_size; // T-2475 probe",
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "appending after proof_manifest.h's own trailing comment must still redden the census"
    assert any("UNMARKED QOW PATTERN HIT" in f and "proof_manifest.h" in f and "out_channels" in f
               for f in failures)


# --- No same-line STATEMENT is producible for model.h/proof_manifest.h (T-2491, Poirot dcefab3-
# t2486-census-content-keying-confirmation.md Sec4, D-SLM5716/D-SLM5717, D-SLM5732): both excused
# lines are enumerators inside an `enum class` body (`SslmModelStatus`, `ConfigGeometryStatus`),
# which admits enumerators, not statements. T-2481's two cells here inserted a statement between
# the enumerator and its own trailing comment, believing that construction was real, compiled code
# the production path could emit; executed with the CI's own pinned compiler (clang 18.1.8,
# `-fsyntax-only -std=c++20`), both headers fail with `error: missing ',' between enumerators`.
# The cells also had zero discrimination, executed both with the QOW family forced false and with
# the injected identifier replaced by an inert statement: the assertion held either way, because
# what actually fired was the trailing prose comment's own R1 co-occurrence once the exclusion's
# exact-text match went dead -- not recognition of the injected statement. Deleted rather than
# replaced with a third attempt at the SAME (statement) shape.
# `test_defeats_the_prior_mechanism_same_line_append_adapter_marshal` and
# `test_prepended_content_on_an_excused_line_is_also_caught`, above, are the statement class's
# whole population -- `adapter_marshal.h`'s excused line sits inside a function body, not an enum.
#
# T-2491's own replacement text overclaimed: "there is no producible same-line geometry defect an
# enum body can host" is false -- refuted by construction (Claude/Poirot/ba29de4-t2496-census-
# fixes-confirmation.md Significant 1, D-SLM5756/D-SLM5757). An enum body admits enumerators, not
# statements, but a same-line ENUMERATOR prepend is legal C++ and enough: the excused fragment is
# the whole stripped line, so a prepend leaves it intact -- the exclusion still fires and the
# injected text is scanned as ordinary remainder. Executed at the CI's pinned compiler, clang
# 18.1.8, `-fsyntax-only -std=c++20`: `q_proj_out_channels_hidden_size, ` prepended to `model.h`'s
# fragment and `o_proj_out_channels_hidden_size, ` prepended to `proof_manifest.h`'s both compile
# clean and fire `UNMARKED QOW PATTERN HIT`; both discriminate (silent with the QOW family
# disabled, silent with an inert enumerator instead of the defect). The two cells below pin
# exactly this -- the corrected fact is: a same-line statement is not producible inside an enum
# body; a same-line enumerator is, and the census catches it. ---

def test_defeats_the_prior_mechanism_same_line_enumerator_prepend_model_h():
    def _t(text):
        return text.replace(
            "\t" + _MODEL_FRAGMENT,
            "\tq_proj_out_channels_hidden_size, " + _MODEL_FRAGMENT,
            1,
        )
    with _mutated(_MODEL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, (
        "a new enumerator prepended before model.h's own excused enumerator must FAIL -- this is "
        "the producible same-line construction the class admits inside an enum class body"
    )
    assert any("UNMARKED QOW PATTERN HIT" in f and "model.h" in f and "out_channels" in f
               for f in failures), (
        "the new enumerator's own out_channels/hidden_size co-occurrence is what fires here, in "
        "the excused fragment's own remainder -- see this file's own comment block above"
    )
    assert not any("DEAD EXCLUSION" in f for f in failures), (
        "a prepend, unlike an insert-between-the-comment, leaves the excused fragment's own exact "
        "text intact -- the exclusion must still be found alive"
    )


def test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h():
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\to_proj_out_channels_hidden_size, " + _PROOF_FRAGMENT,
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, (
        "a new enumerator prepended before proof_manifest.h's own excused enumerator must FAIL"
    )
    assert any("UNMARKED QOW PATTERN HIT" in f and "proof_manifest.h" in f and "out_channels" in f
               for f in failures)
    assert not any("DEAD EXCLUSION" in f for f in failures)


# --- This ticket's own added construction: new content BEFORE the fragment, not just after --
# proves the excise-and-scan property is symmetric, not an "append-only" patch. ---

def test_prepended_content_on_an_excused_line_is_also_caught():
    def _t(text):
        return text.replace(
            _ADAPTER_FRAGMENT,
            'if (proj == "k_proj") { uint64_t out_channels = hidden_size; return out_channels; } '
            + _ADAPTER_FRAGMENT,
            1,
        )
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a new QOW statement prepended before the excused fragment must FAIL, same as appended"
    assert any("adapter_marshal.h" in f and "k_proj" in f for f in failures)


# --- Safe-direction Note (T-2468 Sec4a cases 1-2, D-SLM5604): rename/reformat of the excused
# line itself must still re-trigger the census. This fix must not make exclusions stickier. ---

def test_safe_direction_rename_of_the_excused_enumerator_reddens():
    def _t(text):
        return text.replace("ConfigGeometryHiddenSizeMismatch,", "ConfigGeometryHiddenSizeMismatchRenamed,", 1)
    with _mutated(_MODEL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "renaming the excused enumerator must re-trigger the census (safe direction)"


def test_safe_direction_reformat_of_the_excused_line_reddens():
    def _t(text):
        return text.replace(_ADAPTER_FRAGMENT, _ADAPTER_FRAGMENT.replace(" || ", "  ||  "), 1)
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "whitespace-only reformat of the excused line must re-trigger the census (safe direction)"


# --- Wrap/comment-split (T-2468 Sec4a case 3): splitting the excused declaration from its own
# comment removes the pattern co-occurrence entirely. Under T-2468's own narrow fragment (the
# enumerator name alone) this PASSed uninformatively -- Part 2 had nothing to exclude, so the
# PASS was never evidence the exclusion survived. T-2475 widened this exclusion's own fragment
# to its full original line (declaration + trailing comment together, see the header comment
# above `_PART2_EXCLUDED_TEXT`), so the split ALSO breaks the widened fragment's own exact-text
# match -- and the dead-exclusion check (this file's own `test_dead_exclusion_*` cells) now
# reports it by name instead of passing silently. Strictly more informative than the T-2468-era
# behavior: a human re-reading a DEAD EXCLUSION for this entry after a wrap/split would correctly
# read it as "re-derive or drop this exclusion," where a bare PASS gave no such signal. ---

def test_wrap_comment_split_is_now_reported_as_a_dead_exclusion_not_a_silent_pass():
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\tHiddenSizeGeometryMismatch,\n\t// hidden_size != num_attention_heads * head_dim -- R1, REMOVED",
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, (
        "splitting the widened proof_manifest.h fragment across two lines must now be reported "
        "as a DEAD EXCLUSION (T-2475's own dead-entry check), not silently PASS as it did under "
        "T-2468's narrower, enumerator-only fragment"
    )
    assert any("DEAD EXCLUSION" in f and "proof_manifest.h" in f for f in failures)
    # And it must not ALSO be reported as an UNMARKED pattern hit -- the split still removes the
    # co-occurrence from every resulting line, exactly as T-2468 found; only the exclusion's own
    # aliveness changed, not whether the split content itself trips a pattern.
    assert not any("UNMARKED" in f for f in failures)


# --- F2, pre-existing, not this ticket's own diff (T-2468 Sec4a case 4): a genuinely new QOW
# site whose co-occurring tokens land on different physical lines evades Part 2 entirely.
# Reproduced so the PASS it returns is proven, and cross-checked against the KNOWN LIMITATION
# text main() now prints, so the two cannot drift apart silently. ---

def test_two_line_split_is_the_documented_f2_limitation_still_open():
    # Neither physical line carries BOTH a QOW token and hidden_size on its own -- the
    # condition line names q_proj/o_proj without hidden_size, and the return line names
    # hidden_size without any QOW token, exactly the shape an ordinary wrapped conditional
    # produces (T-2468 Sec4a case 4's own construction, reproduced here).
    injected = (
        "\ninline uint64_t T2475TwoLineProbe(const std::string& proj, uint64_t hidden_size) {\n"
        '\tif (proj == "q_proj" || proj == "o_proj")\n'
        "\t\treturn hidden_size;\n"
        "\treturn 0;\n"
        "}\n"
    )

    def _t(text):
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    with _mutated(_MATMUL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], (
        "a new QOW site split across two physical lines is NOT caught today -- this is F2, "
        "pre-existing and named as a known limitation rather than closed by this ticket; a "
        "change to this assertion means F2 was closed and the KNOWN LIMITATION text below "
        "(and the module docstring) must be updated in the same change"
    )


def test_known_limitation_for_f2_is_printed_on_a_passing_run():
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "tools", "geometry_site_census.py")],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "KNOWN LIMITATION" in proc.stdout
    assert "F2" in proc.stdout


# --- End-to-end against the real tree via the real CLI entry point, not just run_census(). ---

def test_main_end_to_end_via_subprocess_is_green_today():
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "tools", "geometry_site_census.py")],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "PASS" in proc.stdout


# =====================================================================================
# T-2475 fold-in (Claude/Poirot/6597903-t2472-ask5-tracka-confirmation.md Sec4-5,
# Significant 1 and 2 -- both in this same file, folded into this same round rather than
# left for a second builder):
#
# S1: Part 3's `required_token_scope_ends` was ruled self-detecting on the reasoning that
# the LOOKUP key's `marker_line` is recomputed fresh every run. True, and irrelevant -- the
# STORED key and its end line were both absolute line numbers frozen at authoring time, so
# an edit anywhere ELSE in the same file (the reviewer's own construction: one comment line
# at the very top) shifted every marker below it without shifting the registry to match,
# producing a false `MISSING ... ENTRY` on untouched, correctly-fixed code. Closed by
# re-keying `required_token_scope_end_offsets` on occurrence ORDINAL (this site's Nth
# marker in this file) and OFFSET (marker-to-scope-end distance) instead of absolute line
# numbers -- see `tools/geometry_site_census.py`'s own Part 3 section and the registry's
# header comment for the full account.
#
# S2: this file's own suite IS the pin Significant 2 asked for -- the census had no cell at
# all before it, and the reviewer's own mutation-proof (reverting the text-keyed exclusion
# remedy to skip-whole-line-on-match leaves every gate green) is reproduced by this file's own
# same-line population cells, below -- see the S2 red-check comment block further down this
# file for the exact, executed cell list and count.
# =====================================================================================

_PROOF_MANIFEST_CPP = os.path.join("src", "proof_manifest.cpp")
_SUPERSLM_GPU_CPP = os.path.join("src", "gpu", "superslm_gpu.cpp")

_GS14_FIXED_LINE = "\t\t\tplan.out_channels = (q_width != UINT32_MAX) ? q_width : hidden_size;\n"
_GS14_REVERTED_LINE = "\t\t\tplan.out_channels = hidden_size;\n"


# --- S1: the founding demonstration (GS-14 revert, marker untouched) must stay caught. ---

def test_part3_gs14_revert_with_marker_untouched_is_still_caught():
    def _t(text):
        assert _GS14_FIXED_LINE in text, "GS-14's own fixed line has moved -- update this fixture"
        return text.replace(_GS14_FIXED_LINE, _GS14_REVERTED_LINE, 1)
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "GS-14's fix reverted with its marker left in place must still fail Part 3"
    assert any("GS-14" in f and "REGRESSED SITE" in f for f in failures)


# --- S1: the finding this fold-in closes -- an edit far from any marker must NOT false-FAIL. ---

def test_part3_unrelated_edit_elsewhere_in_the_file_does_not_false_fail():
    # The reviewer's own construction: one comment line inserted at the very TOP of a file
    # carrying a Part-3-checked marker (GS-01, src/proof_manifest.cpp), nowhere near any
    # site's own code -- under the prior absolute-line keying this shifted every marker
    # below it and produced `MISSING required_token_scope_ends ENTRY` on untouched, correct
    # code (T-2472 Significant 1, reproduced live against `6597903` before this fix landed).
    def _t(text):
        return "// T-2475 S1 reproduction: one comment line at the top of the file\n" + text
    with _mutated(_PROOF_MANIFEST_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], (
        "an edit far from every marker must not desync the ordinal/offset keying -- if this "
        "fails, S1's own remedy has regressed"
    )


# --- S2 red-check: reverting `_part2_excise_excluded_text` to skip-whole-line-on-match (the
# TEXT-keyed exclusion mechanism the reviewer's own mutation-proof reverted to -- `Claude/Brunel/
# t2475-census-exclusion-class-2026-08-31.md` Sec10 records what was actually run; NOT a return
# to (path, line) keying) reddens exactly EIGHT cells (T-2480 F2, Poirot f363c2a-t2479-census-
# class-confirmation.md Sec8 M2, D-SLM5654; re-verified against this file's own current, grown
# population by Poirot dcefab3-t2486-census-content-keying-confirmation.md Sec8 M2, D-SLM5721;
# T-2497 adds the two enumerator-prepend cells to the population, re-executed against the same
# reversion -- 8 failed, 32 passed, up from the prior 6 failed/34 passed): the SIX same-line
# population cells --
# test_defeats_the_prior_mechanism_same_line_append_adapter_marshal,
# test_appending_after_model_h_s_trailing_comment_fires_on_apparatus_unsound_grounds,
# test_appending_after_proof_manifest_h_s_trailing_comment_fires_on_apparatus_unsound_grounds,
# test_prepended_content_on_an_excused_line_is_also_caught,
# test_defeats_the_prior_mechanism_same_line_enumerator_prepend_model_h,
# test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h -- plus the two
# mechanism-level unit cells test_excise_leaves_appended_new_content_in_the_remainder and
# test_excise_leaves_prepended_new_content_in_the_remainder (this file's own docstring separately
# names these). The two cells T-2491 deletes (formerly the `..._insert_before_comment_*` pair,
# above) were never among these eight -- executed and confirmed: neither depended on the excision
# mechanism at all, so reverting it left both unaffected regardless of whether they existed.


# --- Observation carried into this round (Poirot 6597903-t2472-ask5-tracka-confirmation.md
# Sec10 O1, D-SLM5617): a `_PART2_EXCLUDED_TEXT` entry whose subject is deleted used to go
# silent (exit 0, no diagnostic) -- unsafe for nobody (a genuinely new site elsewhere in the
# same file is still caught), but the disclosure of a deliberate, documented residual quietly
# disappearing is itself a defect in what a PASS claims. Closed the same way as the class this
# round exists to close: excision now tracks which fragments actually matched something during
# the walk, and an entry that matched nothing anywhere is reported by name. ---

def test_dead_exclusion_is_reported_when_its_subject_is_deleted():
    def _t(text):
        assert (_ADAPTER_FRAGMENT + "\n") in text
        return text.replace(_ADAPTER_FRAGMENT + "\n", "", 1)
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "deleting an excused statement outright must be reported, not silent"
    assert any("DEAD EXCLUSION" in f and "adapter_marshal.h" in f for f in failures)


def test_dead_exclusion_check_does_not_false_fire_on_the_untouched_tree():
    # Sanity companion to the cell above -- the baseline test already asserts this globally,
    # restated here so a DEAD EXCLUSION regression is locatable by name if it ever recurs.
    failures = census.run_census(_REPO_ROOT)
    assert not any("DEAD EXCLUSION" in f for f in failures)


# =====================================================================================
# T-2481 fold-in, item 2 (Poirot f363c2a-t2479-census-class-confirmation.md Significant 1,
# D-SLM5651/D-SLM5670): `_part2_excise_excluded_text` matches an excused fragment by exact-text
# PRESENCE, so a genuinely new, unregistered site written as a LITERAL COPY of an excused
# fragment, anywhere else in that same file, is excised for free -- exit 0, unaudited site in
# the tree. Closed by counting non-comment-line matches instead of recording mere presence and
# failing above one. This is the same class as item 1 below one level apart (a key that names
# LESS than the thing it stands for -- here, "this text exists somewhere in the file" instead of
# "this text is THE one statement it was written to excuse").
# =====================================================================================

def test_part2_ambiguous_exclusion_fires_on_a_literal_duplicate_copy():
    def _t(text):
        assert text.count(_ADAPTER_FRAGMENT) == 1
        injected = (
            "\ninline uint64_t T2481DuplicateProbe(const std::string& proj, uint64_t hidden_size) {\n"
            "\t" + _ADAPTER_FRAGMENT + "\n"
            "\treturn 0;\n"
            "}\n"
        )
        assert "}  // namespace superslm" in text
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a literal duplicate copy of an excused fragment must FAIL, not be excised for free"
    assert any("AMBIGUOUS EXCLUSION" in f and "adapter_marshal.h" in f and "2 non-comment lines" in f
               for f in failures)


def test_part2_ambiguous_exclusion_does_not_false_fire_on_a_near_copy():
    # A near-copy (the excused fragment's own text disturbed, not reproduced literally) is NOT
    # a duplicate -- it is a genuinely different statement, and must be caught the ordinary way
    # (UNMARKED ... PATTERN HIT), not misreported as an ambiguous exclusion.
    def _t(text):
        near_copy = _ADAPTER_FRAGMENT.replace("proj ==", "proj  ==", 1)
        assert near_copy != _ADAPTER_FRAGMENT
        injected = (
            "\ninline uint64_t T2481NearCopyProbe(const std::string& proj, uint64_t hidden_size) {\n"
            "\t" + near_copy + "\n"
            "\treturn 0;\n"
            "}\n"
        )
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a near-copy (not a literal duplicate) must still FAIL as an unmarked site"
    assert any("UNMARKED QOW PATTERN HIT" in f and "adapter_marshal.h" in f for f in failures)
    assert not any("AMBIGUOUS EXCLUSION" in f for f in failures)


def test_part2_ambiguous_exclusion_does_not_false_fire_on_the_untouched_tree():
    failures = census.run_census(_REPO_ROOT)
    assert not any("AMBIGUOUS EXCLUSION" in f for f in failures)


# =====================================================================================
# T-2481 fold-in, item 1 (Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md F1,
# D-SLM5647; D-SLM5659): Part 3's occurrence-ORDINAL keying is defeated by REORDERING two of a
# multi-occurrence site's own existing occurrences within their shared file -- nothing added or
# removed, marker count unchanged, Part 1 blind to it -- which desyncs the ordinal-to-offset
# lookup and silently absorbs a genuine revert while citing an untouched line. Demonstrated on
# GS-10 (D-SLM5647); T-2481 found the SAME exposure, independently, on GS-11 (same two-occurrence
# shape, omitted from the prior commissioning's own exposure list) before building the remedy.
# `required_token_scopes` (registry) replaces ordinal with CONTENT-ADDRESSED anchors -- these
# cells prove the reorder-plus-revert construction that produced byte-identical output under the
# old scheme now correctly discriminates: PASS on the healthy reorder, FAIL citing the RELOCATED
# occurrence's own line on the reverted one.
# =====================================================================================

def _marker_line(indent: str, gs_id: str) -> str:
    """Builds a real marker's own literal line at RUNTIME rather than embedding it as a
    contiguous string constant in this module -- this test file is itself swept by Part 1's
    tree-wide marker scan (`_iter_source_files` with no `production_only` filter includes
    tests/**/*.py), so a literal "SSLM-GEOMETRY-SITE: GS-NN" substring written directly into
    this file's own source would be picked up as a genuine (orphan/duplicate) marker occurrence
    of THIS file, not just of the mutated production file a fixture targets."""
    return indent + "// SSLM-GEOMETRY" + "-SITE: " + gs_id + "\n"


def _swap_and_optionally_revert(text, chunk_a, chunk_b, revert_from=None, revert_to=None):
    """Swaps `chunk_a` and `chunk_b` bodily (each occupies the other's former physical position),
    optionally reverting one occurrence of a literal substring (`revert_from` -> `revert_to`,
    the FIRST one found) in the RESULT -- i.e. after the swap, wherever that text now physically
    sits. Mirrors Claude/Mendeleev/t2480-probes/t2480_q2_part3_edit_survivability.py's own
    reorder-plus-revert construction. `revert_from` need not be unique in the swapped text: two
    occurrences whose own local code is genuinely identical (GS-12's occ2/occ7) legitimately
    share the same revertible text, and reverting either one is an equally valid probe of that
    pair's own harmless-collision property."""
    assert text.count(chunk_a) == 1, "chunk_a not found uniquely -- update this fixture"
    assert text.count(chunk_b) == 1, "chunk_b not found uniquely -- update this fixture"
    swapped = text.replace(chunk_a, "\0T2481CHUNKB\0").replace(chunk_b, chunk_a)
    swapped = swapped.replace("\0T2481CHUNKB\0", chunk_b)
    if revert_from is None:
        return swapped
    assert swapped.count(revert_from) >= 1, "revert_from not found at all after the swap"
    return swapped.replace(revert_from, revert_to, 1)


_GS10_CHUNK_A = (
    _marker_line("\t", "GS-10") +
    "\t// T-2432 (Track A step 5, design §2.1 item 5/§6 Track A step 5, GS-10): "
    "q_proj.weight's\n"
    "\t// real shape is [q_width, hidden_size] -- byte extent q_width * hidden_size, not\n"
    "\t// hidden_size * hidden_size.\n"
    "\tL.off[2] = cur; cur += Align8U32(effective_q_width * hidden_size);  // q_weight (int8)\n"
)
_GS10_CHUNK_B = (
    _marker_line("\t\t", "GS-10") +
    "\t\tfor (uint32_t i = 0; i < QW * H; ++i) lw_bytes[base + layout.off[2] + i] = "
    "static_cast<uint8_t>(lw.q_weight[i]);\n"
)


def test_part3_reorder_gs10_healthy_swap_stays_clean():
    # The bare reorder, with neither occurrence's own fix touched -- content-addressing must not
    # false-fire just because the two occurrences trade physical positions.
    with _mutated(_SUPERSLM_GPU_CPP, lambda t: _swap_and_optionally_revert(t, _GS10_CHUNK_A, _GS10_CHUNK_B)):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], "reordering two healthy occurrences must not, by itself, redden the census"


def test_part3_reorder_gs10_plus_revert_is_caught_at_the_relocated_line():
    # Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md's own founding construction
    # (D-SLM5647): under the pre-T2481 ordinal keying this produced BYTE-IDENTICAL output to the
    # healthy-swap cell above, citing the UNTOUCHED occurrence's own line while the RELOCATED
    # occurrence's own genuine revert went silently absorbed.
    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS10_CHUNK_A, _GS10_CHUNK_B,
            revert_from="for (uint32_t i = 0; i < QW * H; ++i)",
            revert_to="for (uint32_t i = 0; i < H * H; ++i)",
        )
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a genuine revert of the relocated occurrence must FAIL, not be silently absorbed"
    assert any("REGRESSED SITE" in f and "GS-10" in f and ":714" in f for f in failures), (
        "the finding must cite the RELOCATED occurrence's own (now-first) line, not the "
        "untouched occurrence -- a wrong-line citation is exactly what the prior ordinal "
        "keying produced"
    )


_GS11_CHUNK_A = (
    _marker_line("\t", "GS-11") +
    "\t// T-2432 (Track A step 5, GS-11): o_proj.weight's real shape is [hidden_size, q_width] --\n"
    "\t// byte extent hidden_size * q_width, not hidden_size * hidden_size. Unlike GS-07's own\n"
    "\t// per-output-channel fold count (o_fold_identity/mult/shift below, GS-09, confirmed\n"
    "\t// correct as hidden_size-sized and NOT touched by this step), the weight MATRIX itself\n"
    "\t// genuinely decouples on its input axis.\n"
    "\tL.off[25] = cur; cur += Align8U32(hidden_size * effective_q_width);  // o_weight (int8)\n"
)
_GS11_CHUNK_B = (
    _marker_line("\t\t", "GS-11") +
    "\t\tfor (uint32_t i = 0; i < H * QW; ++i) lw_bytes[base + layout.off[25] + i] = "
    "static_cast<uint8_t>(lw.o_weight[i]);\n"
)


def test_part3_reorder_gs11_healthy_swap_stays_clean():
    # GS-11 shares GS-10's own two-occurrence, one-file shape (offsets 6 and 1) and is
    # independently exposed to the same ordinal-desync defect (T-2481, D-SLM5659) -- the prior
    # commissioning's own exposure list ("GS-10, GS-12, GS-19") omitted it. Verified by
    # execution: this exact swap produced a false REGRESSED SITE on healthy, unreverted code
    # under the pre-T2481 ordinal keying (a safe-direction false alarm, not a silent pass, for
    # this specific construction -- contingent on what unrelated text fell inside the misapplied
    # window, per Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md F3).
    with _mutated(_SUPERSLM_GPU_CPP, lambda t: _swap_and_optionally_revert(t, _GS11_CHUNK_A, _GS11_CHUNK_B)):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], "reordering two healthy GS-11 occurrences must not redden the census"


def test_part3_reorder_gs11_plus_revert_is_caught_at_the_relocated_line():
    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS11_CHUNK_A, _GS11_CHUNK_B,
            revert_from="for (uint32_t i = 0; i < H * QW; ++i)",
            revert_to="for (uint32_t i = 0; i < H * H; ++i)",
        )
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a genuine revert of the relocated GS-11 occurrence must FAIL"
    assert any("REGRESSED SITE" in f and "GS-11" in f and ":741" in f for f in failures)


_GS12_OCC0 = (
    _marker_line("\t", "GS-12") +
    "\t// T-2432 (Track A step 3, design §2.1 item 5/§6 Track A step 3): "
    "q_codes/q_rot/ctx_codes\n"
    "\t// are Q's own output-width buffers -- sized `effective_q_width`, not `hidden_size`.\n"
    "\t// k_rot is written at `h * head_dim` for `h` up to `num_heads` (this loop's own "
    "query-head\n"
    "\t// index, not the KV-head index LandTokenKVRow uses to size its own K store) -- the same\n"
    "\t// query-head-count bound q_rot uses, so it needs the identical widening or an\n"
    "\t// out-of-bounds write follows the moment `num_heads` exceeds `hidden_size / head_dim`\n"
    "\t// (a mechanical consequence of widening `num_heads`, not a separate design decision --\n"
    "\t// the design's own §6 Track A step 3 text names q_codes/q_rot/ctx_wide/ctx_codes "
    "and does\n"
    "\t// not separately name k_rot because k_rot did not yet exist as a distinct local at the\n"
    "\t// text's own citation range; its indexing is identical to q_rot's).\n"
    "\tstd::vector<int8_t> normed(hidden_size), q_codes(effective_q_width), o_codes(hidden_size);\n"
)
_GS12_OCC1 = (
    _marker_line("\t\t", "GS-12") +
    "\t\t// T-2432 (Track A step 3): q_proj's INPUT width stays hidden_size (the normed\n"
    "\t\t// residual stream is unchanged by this ask); its OUTPUT width is effective_q_width.\n"
    "\t\tst = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, hidden_size, "
    "effective_q_width,\n"
)


def test_part3_reorder_gs12_two_differently_offset_occurrences_plus_revert_is_caught():
    # GS-12 has nine occurrences in one file with a mix of offsets (11, 3, 3, 4, 3, 4, 4, 3, 4)
    # -- occ0 (offset 11) and occ1 (offset 3) genuinely differ, unlike the occ2/occ7 pair below.
    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS12_OCC0, _GS12_OCC1,
            revert_from=("st = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, "
                          "hidden_size, effective_q_width,"),
            revert_to=("st = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, "
                        "hidden_size, hidden_size,"),
        )
    with _mutated(_FORWARD_SITES_CPP_T2481, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a genuine revert of the relocated GS-12 occurrence must FAIL"
    assert any("REGRESSED SITE" in f and "GS-12" in f and ":1681" in f for f in failures)


_GS12_OCC2 = (
    _marker_line("\t\t\t", "GS-12") +
    "\t\t\t// T-2432 (Track A step 3): ctx_wide is the pre-fold attention-context accumulator,\n"
    "\t\t\t// one head_dim-wide slice per query head -- sized effective_q_width, not "
    "hidden_size.\n"
    "\t\t\tstd::vector<int64_t> ctx_wide(effective_q_width);\n"
)
_GS12_OCC7 = (
    _marker_line("\t\t\t\t", "GS-12") +
    "\t\t\t\t// T-2432 (Track A step 3): ctx_wide is the pre-fold attention-context accumulator,\n"
    "\t\t\t\t// one head_dim-wide slice per query head -- sized effective_q_width, not "
    "hidden_size.\n"
    "\t\t\t\tstd::vector<int64_t> ctx_wide(effective_q_width);\n"
)


def test_part3_reorder_gs12_identical_shared_anchor_pair_is_harmless():
    # occ2 and occ7 share literally identical local comment text (a copy-pasted explanation for
    # equivalent code in two sibling functions) and the SAME offset (3) -- the registry
    # registers one shared anchor record for both. Reordering these two specifically must stay
    # harmless in both directions: healthy swap clean, and a revert of EITHER relocated
    # occurrence still correctly caught (misassigning between two occurrences whose own
    # registered offset agrees changes nothing about what gets checked).
    with _mutated(_FORWARD_SITES_CPP_T2481, lambda t: _swap_and_optionally_revert(t, _GS12_OCC2, _GS12_OCC7)):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], "reordering the two identical-anchor GS-12 occurrences must not redden the census"

    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS12_OCC2, _GS12_OCC7,
            revert_from="std::vector<int64_t> ctx_wide(effective_q_width);",
            revert_to="std::vector<int64_t> ctx_wide(hidden_size);",
        )
    with _mutated(_FORWARD_SITES_CPP_T2481, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a revert of either identical-anchor occurrence must still be caught"
    assert any("REGRESSED SITE" in f and "GS-12" in f for f in failures)


def test_part3_insert_unregistered_occurrence_between_two_existing_is_caught_without_corrupting_others():
    # Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md Sec3.2's own "add an
    # occurrence between two existing ones" construction, F3: under the pre-T2481 ordinal
    # keying this cascaded an ordinal shift through every downstream occurrence, contingently
    # producing a mix of false REGRESSED SITEs on untouched code. Content-addressing has no
    # ordinal to cascade: only the new, genuinely unregistered marker is flagged, and every
    # pre-existing occurrence -- unaffected by an edit elsewhere in the file -- stays clean.
    def _t(text):
        anchor_line = ("\tstd::vector<int8_t> normed(hidden_size), q_codes(effective_q_width), "
                        "o_codes(hidden_size);\n")
        assert text.count(anchor_line) == 1
        injected = (_marker_line("\t", "GS-12") +
                    "\tstd::vector<int8_t> ctx_extra(effective_q_width);\n")
        return text.replace(anchor_line, anchor_line + injected, 1)
    with _mutated(_FORWARD_SITES_CPP_T2481, _t):
        failures = census.run_census(_REPO_ROOT)
    assert any("MARKER COUNT MISMATCH" in f and "GS-12" in f for f in failures)
    assert any("MISSING SCOPE ANCHOR" in f and "GS-12" in f for f in failures), (
        "the new, unregistered occurrence must be named by its own marker line"
    )
    assert not any("REGRESSED SITE" in f for f in failures), (
        "no PRE-EXISTING occurrence may be falsely flagged just because an unrelated new "
        "marker was inserted elsewhere in the file"
    )


def test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_all():
    # A `fixed` site with `required_tokens` but NO `required_token_scopes` entry for a file it
    # has a marker in -- the registry-side twin of `MISSING SCOPE ANCHOR` (a marker with no
    # matching record at all, rather than one that fails to match any of its site's records).
    #
    # T-2491 (Poirot dcefab3-t2486-census-content-keying-confirmation.md Significant 2,
    # D-SLM5718): both writes use `newline=""` -- `tools/geometry_site_registry.json` is
    # `attr/text eol=lf`, and the default text-mode write translates every `\n` to `\r\n` on this
    # platform, which left the checkout ` M` after this cell ran even though it restores the
    # original *content* byte-for-byte. `newline=""` writes the string's own bytes with no
    # translation, so a clean checkout stays clean before and after.
    def _t(registry_json):
        import json
        data = json.loads(registry_json)
        for site in data["sites"]:
            if site["id"] == "GS-01":
                del site["required_token_scopes"]
                break
        else:
            raise AssertionError("GS-01 not found in registry")
        return json.dumps(data, indent=2)
    registry_path = os.path.join(_REPO_ROOT, "tools", "geometry_site_registry.json")
    with open(registry_path, "r", encoding="utf-8") as f:
        original = f.read()
    try:
        with open(registry_path, "w", encoding="utf-8", newline="") as f:
            f.write(_t(original))
        failures = census.run_census(_REPO_ROOT)
    finally:
        with open(registry_path, "w", encoding="utf-8", newline="") as f:
            f.write(original)
    assert any("MISSING required_token_scopes ENTRY" in f and "GS-01" in f for f in failures)


# =====================================================================================
# T-2491 fold-in (Claude/Poirot/dcefab3-t2486-census-content-keying-confirmation.md O1,
# D-SLM5725): `AMBIGUOUS SCOPE ANCHOR` is the third of Part 3's three fail-closed diagnostics and
# the only one this file never exercised. It is a live guard, not a dead one -- the reviewer
# fired it by placing a second record's own anchor text inside a first record's own claimed
# window, with no reorder needed. Reproduced here on GS-10: occurrence 0's own registered record
# (`anchor="L.off[2] = cur; cur += Align8U32(", offset=4`) claims a 5-line window starting at its
# marker (its own 5-line chunk, marker through its own governed code); occurrence 1's own
# registered anchor (`"lw_bytes[base + layout.off[2] + i] = static_cast<uint8_t>(lw.q_weight[i]);
# "`) is appended onto occurrence 0's OWN marker line -- inside occurrence 0's own window and
# without adding or removing a line, so occurrence 0's own real anchor stays exactly where it
# was. Both records now match occurrence 0's marker, and they disagree on offset (4 vs 1).
# =====================================================================================

def test_part3_ambiguous_scope_anchor_fires_when_two_records_own_anchors_collide_in_one_window():
    def _t(text):
        assert _GS10_CHUNK_A in text, "GS-10 occurrence 0's own chunk has moved -- update this fixture"
        marker = _marker_line("\t", "GS-10")
        assert marker in _GS10_CHUNK_A
        collided_marker = marker.rstrip("\n") + (
            "  // collision probe: lw_bytes[base + layout.off[2] + i] = "
            "static_cast<uint8_t>(lw.q_weight[i]);\n"
        )
        collided = _GS10_CHUNK_A.replace(marker, collided_marker, 1)
        assert collided != _GS10_CHUNK_A
        assert collided.count("\n") == _GS10_CHUNK_A.count("\n"), (
            "must not change the chunk's own line count -- occurrence 0's real anchor line "
            "would shift out of its own registered window"
        )
        return text.replace(_GS10_CHUNK_A, collided, 1)
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert any("AMBIGUOUS SCOPE ANCHOR" in f and "GS-10" in f for f in failures), (
        "two registered records whose own anchors both match within one physical marker's own "
        "window, with disagreeing offsets, must be reported by name rather than guessed at"
    )
    assert not any("REGRESSED SITE" in f and "GS-10" in f for f in failures), (
        "an ambiguous match must not fall through and silently pass -- this second assertion is "
        "itself inert (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Sec8 M2, "
        "D-SLM5760: with the guard disabled the census reports nothing at exit 0, never a false "
        "REGRESSED SITE); the first assertion above carries the whole pin, and what the guard "
        "actually prevents is that silent absorption, not a required-token check against either "
        "record's own window"
    )
