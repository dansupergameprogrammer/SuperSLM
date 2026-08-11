"""T-1899 -- Curie's red suite for T-1894 (T-1822 design Sec12 "Option G's
own coverage"): the Python-reference halves of two cells --

  - "Engine/reference bit-parity": the fused K-landing output vs. a
    re-derived `dynamic_engine.py` reference computed at the coordinated
    (post-Option-G, land-after-RoPE) ordering.
  - "Microstep/whole-token parity, Python": `dynamic_engine.py`'s own
    `begin_token`/`decode_step` path and `forward_dynamic_vec` agree on
    which ordering runs for the same loaded model.

WHY THIS FILE IS RED BY IMPORT, NOT BY ASSERTION. `dynamic_engine.py` does
not exist on main@c6cfa03 at all -- verified this session
(`git log --all --oneline -- tests/reference/superslm_spike/dynamic_engine.py`
shows it introduced on the T-1519/T-1520 lineage and present today only on
branches descending from T-1891's own spike work, never merged to `main`).
The design's own Sec31.2's text reads it directly from that path ("Verified
at source this session, tests/reference/superslm_spike/dynamic_engine.py"),
naming it as an existing asset without stating that main lacks it -- porting
or authoring this module for production is T-1894's own build obligation
(the same "engine/tooling-integration tier, gated on infrastructure that
does not exist yet" disposition Claude/Curie/t1832-...-red-suite-test-
design-2026-08-08.md Sec1 already establishes for this suite's C++-side
sibling cells), not a gap in the Coverage Model itself.

This module is therefore authored to the SAME red-cell convention this
suite's C++ side uses for a declared-but-undefined symbol (`RopeApplyPairWide`
et al. in test_main.cpp's own T-1899 section): it imports the real reference
module by its real, final name and location, and is RED BY IMPORT FAILURE
until T-1894's build lands it. `pytest.importorskip` is deliberately NOT used
here -- that would SKIP the cell, not fail it, and a skipped cell reads as
"not yet run" rather than "red for a missing mechanism", the exact
distinction StandardsDocument.md Sec4's structural-enforcement law exists to
preserve (a check that can be silently skipped is not a check).

Test-design record: Claude/Curie/t1899-optionG-red-suite-2026-08-11.md
(records worktree, D:\\Wizard).
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_SPIKE_DIR = os.path.join(_TESTS_DIR, "reference", "superslm_spike")
if _SPIKE_DIR not in sys.path:
    sys.path.insert(0, _SPIKE_DIR)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


def _import_dynamic_engine():
    """Imports the real production reference module by its real name.

    Deliberately not wrapped in try/except -- an ImportError here IS this
    cell's own red state, and pytest reports it as a collection/test error
    (a hard failure), not a pass. This function exists only so both test
    functions below share one import point and one failure message.
    """
    import dynamic_engine  # noqa: F401 (import-for-its-own-sake; ImportError is the cell)

    return dynamic_engine


def test_option_g_fused_k_landing_matches_dynamic_engine_reference():
    """Sec12 "Engine/reference bit-parity" -- Python-reference half.

    Once `dynamic_engine.py` exists on `main` (T-1894's own build), this
    cell drives `forward_dynamic_vec` at the coordinated (post-Option-G,
    land-after-RoPE) ordering on a small fixture and asserts its K-landing
    output equals the compiled engine's own output on the identical
    fixture (test_main.cpp's own
    TestOptionGFusedKLanding_NonNullConfiguration_DivergesFromLegacy,
    kOptionGFusedK_Pos1_NonNullConfiguration=[127,57],
    tests/sslm_t1899_optionG_fixtures.h) -- mutation-pinned by reverting
    EITHER side alone (design's own text: "a cell that only moves when both
    change together is not a parity cell"). Not yet authorable as a running
    assertion: the module this cell's own reference half depends on does
    not exist on `main`.
    """
    dynamic_engine = _import_dynamic_engine()
    pytest.fail(
        "dynamic_engine module imported successfully, but this cell's own cross-"
        "language differential is not yet authored against it (T-1894's build "
        "must land forward_dynamic_vec's coordinated post-Option-G ordering "
        "first) -- reaching this line at all means dynamic_engine.py now exists "
        "on main and this test needs completing, not merely un-skipping: "
        f"{dynamic_engine!r}"
    )


def test_dynamic_engine_microstep_and_whole_token_paths_agree_under_flag():
    """Sec12 "Microstep/whole-token parity, Python".

    Once `dynamic_engine.py` exists on `main`, this cell resumes mid-token
    under a flags=1 model (`MicroStepState`/`begin_token`/`decode_step`) and
    asserts fused output, matching a whole-token run
    (`forward_dynamic_vec`) of the identical model -- design Sec31.2.3's own
    text: "both `forward_dynamic_vec` and `_forward_dynamic_vec_layers`...
    read the identical, single source of truth" (the artifact's own `flags`
    bit, not a caller-supplied boolean). Not yet authorable: same import gap
    as the cell above.
    """
    dynamic_engine = _import_dynamic_engine()
    pytest.fail(
        "dynamic_engine module imported successfully, but this cell's own "
        "microstep/whole-token resumable-parity differential is not yet "
        "authored against it -- reaching this line means dynamic_engine.py "
        f"now exists on main: {dynamic_engine!r}"
    )
