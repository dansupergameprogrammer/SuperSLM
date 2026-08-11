"""T-1899 (Curie) -- Curie's own red suite for
check_no_optionG_fused_site_skip.py (T-1822 design Sec12's "-60 load floor"
cell, structural-absence half). Mechanism 2 RE-ANCHORED and its own vitality
suite REWRITTEN 2026-08-11 round 4 (T-1901 Critical 2/D-SLM2417, D-SLM2425).

Mechanism 1 (retired-symbol absence) still mirrors
tests/ci/test_check_no_forward_leaf_calls.py's own convention: driven against
CONSTRUCTED SCRATCH files under `tempfile.TemporaryDirectory()`.

Mechanism 2 (no conditional ahead of the dynamic gate) is now driven against
REAL-FILE mutations -- a COPY of the actual, real
`src/forward/forward_sites.cpp`, never a synthetic fixture. T-1901's own
finding (Critical 2): the prior version of this test suite validated the
checker's mechanism 2 against a scratch fixture carrying a marker line,
`(void) out_magnitude_exceeded_int64;`, that appears NOWHERE in real code --
"a population of one chosen by the rule's author", the exact failure
StandardsDocument.md Sec4 names ("a check shown able to fail on an injected
fault... says nothing about whether the rule set covers the defects that
actually exist"). Executed against copies of the real file, the checker's
PRIOR anchor (the literal string `out_magnitude_exceeded_int64`, six-line
backward look-back) reported CLEAN on both an injected exponent-conditioned
skip and an outright deletion of the refusal, because that literal occurs at
the real call site only inside a comment. This suite now reproduces both of
T-1901's own executed mutations against a real-file copy and requires both to
report RED, per StandardsDocument.md Sec4's own population-validation
requirement -- and confirms the real, unmutated tree still reports CLEAN, so
the re-anchored checker is not simply oversensitive.

Mechanism 1's own scratch-fixture tests are unaffected by this round's
re-anchor (mechanism 1's own anchor -- the retired symbol's literal name --
was never the defect T-1901 found) and are kept as they were.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_REAL_FORWARD_SITES_CPP = os.path.join(_REPO_ROOT, "src", "forward", "forward_sites.cpp")

import check_no_optionG_fused_site_skip as chk  # noqa: E402


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


def _copy_real_forward_sites(tmp: str) -> str:
    """A COPY of the real, unmutated `forward_sites.cpp` under a scratch
    directory -- the real `src/` tree is never edited in place. The copy is
    then mutated by each test below, independently."""
    dest = os.path.join(tmp, "forward_sites.cpp")
    shutil.copyfile(_REAL_FORWARD_SITES_CPP, dest)
    return dest


def _replace_once(path: str, old: str, new: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    count = text.count(old)
    assert count == 1, (
        f"expected exactly one occurrence of the real refusal block in {path}, found {count} "
        f"-- the real file's own shape may have moved; this fixture needs updating, not the "
        f"checker"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))


# The real, unmutated refusal block this suite mutates -- transcribed from
# `src/forward/forward_sites.cpp`'s own fused K-landing loop. If a future
# edit changes this block's own text, `_replace_once`'s count assertion
# fails loudly (not silently vacuous) rather than mutating nothing.
_REAL_REFUSAL_BLOCK = (
    "\t\t\t\t\t\tif (exceeded0 || exceeded1) {\n"
    "\t\t\t\t\t\t\treturn SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain;\n"
    "\t\t\t\t\t\t}"
)


# --- Mechanism 1: retired symbol absence (unaffected by the round-4 re-anchor) ---


def test_a_file_declaring_the_retired_symbol_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/forward_sites.cpp",
            "constexpr int64_t kOptionGFusedKvLandingExponentMin = -25;\n",
        )
        hits = chk.find_retired_symbol_uses(path)
        assert len(hits) == 1 and hits[0].line_no == 1


def test_a_file_not_mentioning_the_retired_symbol_reports_no_hits():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/forward_sites.cpp",
            "constexpr int64_t kKvLandingExponentMin = -60;\n",
        )
        assert chk.find_retired_symbol_uses(path) == []


def test_a_substring_match_is_not_falsely_flagged():
    """kOptionGFusedKvLandingExponentMinX is a DIFFERENT identifier -- the word-
    boundary regex must not flag it as the retired symbol."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/forward_sites.cpp",
            "constexpr int64_t kOptionGFusedKvLandingExponentMinX = -25;\n",
        )
        assert chk.find_retired_symbol_uses(path) == []


# --- Mechanism 2, RE-ANCHORED: real-file vitality (T-1901 Critical 2) -------


def test_real_file_copy_unmutated_reports_clean():
    """The baseline: a COPY of the real file, unmutated, must report clean --
    confirms the re-anchored checker is not oversensitive before the
    mutation tests below rely on a clean baseline to contrast against."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _copy_real_forward_sites(tmp)
        skip_hits = chk.find_dynamic_gate_skip_conditions(path)
        missing_hits = chk.find_missing_anchors([path])
        assert skip_hits == [], f"unmutated real file must report no skip hits: {skip_hits}"
        assert missing_hits == [], (
            f"unmutated real file must carry both refusal statements: {missing_hits}"
        )


def test_real_file_copy_with_injected_exponent_skip_is_caught():
    """T-1901's own first executed mutation, reproduced exactly: an `if
    (lw.kv_landing_e_t_k[h] < -40)` wrapped around the real refusal block --
    "any conditional wrapping the dynamic-gate call sites" (design's own
    text). The PRIOR checker (comment-anchored, six-line backward window)
    reported this clean; the re-anchored one must not."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _copy_real_forward_sites(tmp)
        mutated = (
            "\t\t\t\t\t\tif (lw.kv_landing_e_t_k[h] < -40) {\n"
            "\t\t\t\t\t\t\tif (exceeded0 || exceeded1) {\n"
            "\t\t\t\t\t\t\t\treturn SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain;\n"
            "\t\t\t\t\t\t\t}\n"
            "\t\t\t\t\t\t}"
        )
        _replace_once(path, _REAL_REFUSAL_BLOCK, mutated)
        skip_hits = chk.find_dynamic_gate_skip_conditions(path)
        assert len(skip_hits) >= 1, (
            "an exponent-conditioned skip wrapping the real refusal block must be caught"
        )


def test_real_file_copy_with_refusal_deleted_outright_is_caught():
    """T-1901's own second executed mutation, reproduced exactly: the
    refusal replaced with a bare discard, `(void)exceeded0; (void)exceeded1;`
    -- the shape "no skip condition found" alone cannot catch, since deleting
    the refusal trivially satisfies it. Caught here by the PRESENCE
    assertion (`find_missing_anchors`), not the wrapping one."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _copy_real_forward_sites(tmp)
        _replace_once(path, _REAL_REFUSAL_BLOCK, "\t\t\t\t\t\t(void)exceeded0; (void)exceeded1;")
        skip_hits = chk.find_dynamic_gate_skip_conditions(path)
        missing_hits = chk.find_missing_anchors([path])
        assert skip_hits == [], (
            "no skip-shaped conditional exists after outright deletion -- the wrapping check "
            "alone is expected to stay silent here"
        )
        assert any(h.token == "OptionGFusedLandingExponentOutOfDomain" for h in missing_hits), (
            f"deleting the refusal outright must be caught by the presence assertion -- got "
            f"{missing_hits}"
        )


def test_an_unrelated_enclosing_block_is_not_flagged():
    """A real enclosing ancestor of the refusal (the `for` loop over pairs,
    the `if (option_g_fused_k_landing)` branch, the K/V-landing block's own
    bare `{`, the outer `while`, the function signature) must not be
    mistaken for an exponent-shaped conditional -- confirms the re-anchored
    checker walks the REAL enclosing-block stack without producing false
    positives on the real file's own genuine nesting."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _copy_real_forward_sites(tmp)
        skip_hits = chk.find_dynamic_gate_skip_conditions(path)
        assert skip_hits == [], (
            f"the real file's own genuine nesting (for/if(option_g_fused_k_landing)/while/"
            f"function) must not be flagged -- got {skip_hits}"
        )


def test_a_conditional_in_a_comment_or_string_is_not_flagged():
    """The exponent-shaped-and-if-shaped text this suite's own header
    comments and diagnostic strings contain (e.g. "e_t", "-59", "ExponentMin"
    appear throughout this file's own prose) must never be mistaken for a
    real wrapping conditional -- comments and string contents are stripped
    before matching."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/forward_sites.cpp",
            "\n".join([
                "// if (e_t < -40) -- this text describes the fractured early-exit, not code",
                'const char* msg = "if (kv_landing_e_t_k[h] < -40) return ExponentMin";',
                "if (exceeded0 || exceeded1) {",
                "    return SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain;",
                "}",
                "",
            ]),
        )
        skip_hits = chk.find_dynamic_gate_skip_conditions(path)
        assert skip_hits == [], (
            f"a comment and a string literal describing the fractured shape must not be read "
            f"as real wrapping code -- got {skip_hits}"
        )


# --- Population coverage: the default population reaches both src/ and include/ ---


def test_default_population_covers_a_file_under_src_and_under_include():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward/forward_sites.cpp",
               "constexpr int64_t kOptionGFusedKvLandingExponentMin = -25;\n")
        _write(tmp, "include/superslm/forward_sites.h",
               "constexpr int64_t kOptionGFusedKvLandingExponentMin = -25;\n")
        retired_hits, _skip_hits, _missing_hits = chk.check(tmp)
        matched_dirs = {os.path.relpath(h.path, tmp).split(os.sep)[0] for h in retired_hits}
        assert matched_dirs == {"src", "include"}, (
            f"the default population must scan BOTH src/ and include/ -- got hits from "
            f"{matched_dirs}"
        )


# --- The wiring cell: the real, unmodified tree, this session. --------------


def test_real_tree_is_currently_clean():
    """The real src/+include/ tree, unmodified by this session, reports
    clean under the re-anchored mechanism -- the two fused K-landing call
    sites this build created (`src/forward/forward_sites.cpp`) carry the
    unconditional refusal the design specifies, with no wrapping conditional
    and no missing anchor. Kept separate from the mechanism cells above per
    this module's own docstring: a clean real tree passing alone proves
    nothing about whether the checker can catch a fault; the real-file
    mutation cells above are what prove that."""
    retired_hits, skip_hits, missing_hits = chk.check(_REPO_ROOT)
    assert retired_hits == [], f"retired symbol found in the real tree: {retired_hits}"
    assert skip_hits == [], f"skip-shaped conditional found in the real tree: {skip_hits}"
    assert missing_hits == [], f"a required refusal statement is missing from the real tree: {missing_hits}"
