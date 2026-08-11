"""T-1899 (Curie) -- Curie's own red suite for
check_no_optionG_fused_site_skip.py (T-1822 design Sec12's "-60 load floor"
cell, structural-absence half).

Mirrors tests/ci/test_check_no_forward_leaf_calls.py's own convention
exactly: every mechanism cell below is driven against CONSTRUCTED SCRATCH
files under `tempfile.TemporaryDirectory()`, never against the real `src/`
tree edited in place -- StandardsDocument.md Sec4's population-validation
requirement (a check shown able to FAIL on an injected fault, not merely
shown to pass on unchanged input). The one cell that DOES read the real tree
(`test_real_tree_is_currently_clean`) is kept separate for the same reason
that module's own docstring gives: it is the WIRING cell (does the real glob,
unmodified, currently report clean), not a mechanism cell, and a real,
unmodified tree passing alone could never demonstrate the checker catches
anything.
"""
from __future__ import annotations

import os
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir))

import check_no_optionG_fused_site_skip as chk  # noqa: E402


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


# --- Mechanism 1: retired symbol absence ------------------------------------


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


# --- Mechanism 2: no conditional ahead of the dynamic gate -------------------


def test_a_resurrected_early_exit_ahead_of_the_dynamic_gate_is_caught():
    """Reproduces the EXACT shape of T-1898's own fracture (design
    Sec31.2.2): an `if` testing `kv_landing_e_t_k[h]` against a static
    threshold, guarding the code that reads out_magnitude_exceeded_int64."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/forward_sites.cpp",
            "\n".join([
                "for (size_t h = 0; h < num_key_value_heads; ++h) {",
                "    if (lw.kv_landing_e_t_k[h] < kOptionGFusedKvLandingExponentMin) {",
                "        const int64_t raw0 = LandingRescale(rotated.x, m, r_t, e_a, e_t,",
                "            &seq.kv_saturation_count, &exceeded);",
                "        if (exceeded) return OptionGFusedLandingExponentOutOfDomain;",
                "        (void)out_magnitude_exceeded_int64;",
                "    }",
                "}",
                "",
            ]),
        )
        hits = chk.find_dynamic_gate_skip_conditions(path)
        assert len(hits) >= 1, "the resurrected early-exit must be caught"


def test_the_unconditional_construction_reports_no_skip_hit():
    """The DESIGN's own repaired construction: out_magnitude_exceeded_int64
    is checked with no exponent-shaped conditional anywhere in its own
    look-back window."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/forward_sites.cpp",
            "\n".join([
                "for (size_t h = 0; h < num_key_value_heads; ++h) {",
                "    bool exceeded = false;",
                "    const int64_t raw0 = LandingRescale(rotated.x, m, r_t, e_a, e_t,",
                "        &seq.kv_saturation_count, &exceeded);",
                "    if (exceeded) return OptionGFusedLandingExponentOutOfDomain;",
                "    (void)out_magnitude_exceeded_int64;",
                "}",
                "",
            ]),
        )
        hits = chk.find_dynamic_gate_skip_conditions(path)
        assert hits == [], (
            f"an ordinary `if (exceeded)` refusal test, with no exponent-shaped condition "
            f"gating whether the check RUNS AT ALL, must not be flagged -- got {hits}"
        )


def test_an_unrelated_conditional_far_from_the_gate_is_not_flagged():
    """A conditional outside the look-back window must not produce a false
    positive -- the window is deliberately small (Sec12's own text: 'on
    every element, with no condition ... ahead of the check', a LOCAL
    property, not a whole-function one)."""
    with tempfile.TemporaryDirectory() as tmp:
        lines = ["if (e_t < -60) return SomeUnrelatedStatus;"]
        lines += [f"// filler line {i}" for i in range(20)]
        lines += ["(void)out_magnitude_exceeded_int64;"]
        path = _write(tmp, "src/forward/forward_sites.cpp", "\n".join(lines) + "\n")
        hits = chk.find_dynamic_gate_skip_conditions(path)
        assert hits == [], f"a conditional 21 lines above the gate mention is out of window: {hits}"


# --- Population coverage: the default population reaches both src/ and include/ ---


def test_default_population_covers_a_file_under_src_and_under_include():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward/forward_sites.cpp",
               "constexpr int64_t kOptionGFusedKvLandingExponentMin = -25;\n")
        _write(tmp, "include/superslm/forward_sites.h",
               "constexpr int64_t kOptionGFusedKvLandingExponentMin = -25;\n")
        retired_hits, _ = chk.check(tmp)
        matched_dirs = {os.path.relpath(h.path, tmp).split(os.sep)[0] for h in retired_hits}
        assert matched_dirs == {"src", "include"}, (
            f"the default population must scan BOTH src/ and include/ -- got hits from "
            f"{matched_dirs}"
        )


# --- The wiring cell: the real, unmodified tree, this session. --------------


def test_real_tree_is_currently_clean():
    """The real src/+include/ tree, unmodified by this session's build (only
    by this session's own header DECLARATIONS -- forward_sites.h/artifact.h/
    checked_chain_funnel.h -- none of which names the retired symbol or adds
    a conditional ahead of a dynamic-gate mention, since no dynamic-gate CALL
    SITE exists in production code yet at all). Kept separate from the
    mechanism cells above per this module's own docstring: a clean real tree
    passing alone proves nothing about whether the checker can catch a
    fault; the scratch cells above are what prove that."""
    retired_hits, skip_hits = chk.check(_REPO_ROOT)
    assert retired_hits == [], f"retired symbol found in the real tree: {retired_hits}"
    assert skip_hits == [], f"skip-shaped conditional found in the real tree: {skip_hits}"
