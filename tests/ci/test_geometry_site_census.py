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
    `run_census`'s own registry path is fixed relative to the script and is not parameterized
    by `repo_root`, so a synthetic tmp tree cannot carry the 26-site registry this census
    checks Part 1/Part 3 against; mirrors T-2468's own mutate/run/revert discipline):
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
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import geometry_site_census as census  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_ADAPTER_H = os.path.join("include", "superslm", "adapter_marshal.h")
_MODEL_H = os.path.join("include", "superslm", "model.h")
_PROOF_H = os.path.join("include", "superslm", "proof_manifest.h")
_MATMUL_H = os.path.join("include", "superslm", "matmul.h")

_ADAPTER_FRAGMENT = 'if (proj == "q_proj" || proj == "o_proj" || proj == "down_proj") return hidden_size;'
_MODEL_FRAGMENT = "ConfigGeometryHiddenSizeMismatch,    // R1: hidden_size != num_attention_heads * head_dim"
_PROOF_FRAGMENT = "HiddenSizeGeometryMismatch, // hidden_size != num_attention_heads * head_dim -- R1, REMOVED"


@contextlib.contextmanager
def _mutated(rel_path: str, transform):
    """Mutates a real production file, at `rel_path` under the repo root, for the duration of
    the `with` block via `transform(original_text) -> new_text`, then restores the original
    content byte-for-byte. See this module's own docstring for why the real tree is mutated
    in place rather than exercised against a synthetic `tmp` repo root."""
    full = os.path.join(_REPO_ROOT, rel_path)
    with open(full, "r", encoding="utf-8") as f:
        original = f.read()
    try:
        with open(full, "w", encoding="utf-8") as f:
            f.write(transform(original))
        yield
    finally:
        with open(full, "w", encoding="utf-8") as f:
            f.write(original)


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


def test_defeats_the_prior_mechanism_same_line_append_model_h():
    def _t(text):
        return text.replace(
            "\t" + _MODEL_FRAGMENT,
            "\t" + _MODEL_FRAGMENT + "  uint64_t o_proj_out_channels = hidden_size; // T-2475 probe",
            1,
        )
    with _mutated(_MODEL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "new content appended after model.h's own excused line must now FAIL"
    assert any("model.h" in f for f in failures)


def test_defeats_the_prior_mechanism_same_line_append_proof_manifest_h():
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\t" + _PROOF_FRAGMENT + "  uint64_t o_proj_out_channels = hidden_size; // T-2475 probe",
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "new content appended after proof_manifest.h's own excused line must now FAIL"
    assert any("proof_manifest.h" in f for f in failures)


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
# all before it, and the reviewer's own mutation-proof (reverting the text-keyed remedy to
# the exact defective `(path, line)` form leaves every gate green) is reproduced here as
# `test_red_check_*` below, which independently confirms the population above catches the
# exact regression the reviewer used to prove the gap existed.
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


# --- S2 red-check: this suite's own same-line-append population (test_defeats_the_prior_
# mechanism_* and test_prepended_content_on_an_excused_line_is_also_caught, above) IS the
# cell the reviewer's Significant 2 asked for. Confirmed by construction (recorded in
# Claude/Brunel/t2475-census-exclusion-class-2026-08-31.md, not reproduced as a monkeypatch
# test here to avoid duplicating run_census's own source under a second name): with
# `_part2_excise_excluded_text` textually reverted to the reviewer's own defective
# `(rel_path, line_number)` skip-whole-line form, exactly those four cells fail and every
# other cell in this file (including the must-accept/must-reject/safe-direction population)
# stays green -- proving the pin is neither vacuous nor over-broad, then the fix was
# restored and this file re-run clean (22/22) before landing.


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
