"""T-2371 (Brunel), D-SLM5018 S2/M2 -- pins for two of Poirot's build-round
findings against `tests/ci/scan_build_output.py` and its symbol-table reader.

S2: the module's own docstring asserted as settled the checks-(A)/(B)
fail-open property `Claude/Decisions/DecisionLog.md` D-SLM5009 files as
OPEN, waiting on Dan, and contained a self-contradictory sentence
(`StandardsDocument.md` §7). The remedy is a wording correction; this file
pins the corrected text's own load-bearing properties (states the true
narrower claim, cites D-SLM5009, and drops the self-contradiction) rather
than re-asserting the false claim never returns, which no test can prove
about future prose.

M2: `scan_build_output.py` imported `run_fp_free_scan_real_corpus` -- a
module the design calls "retired... no longer load-bearing" -- for
`_read_function_symbol_names`, making that file a hard dependency of the
ship gate. The remedy moves the function into `check_fp_free_scan.py`
(the gate's own production module) and makes the retired driver delegate
to it rather than keep a second, independent copy. This file pins both the
new location and the absence of the retired-module import.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_HERE)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")
_ENGINE_ROOT = os.path.dirname(_TESTS_ROOT)
_SRC = os.path.join(_ENGINE_ROOT, "src")

sys.path.insert(0, _CI_DIR)


def _read_scan_build_output_source():
    with open(os.path.join(_CI_DIR, "scan_build_output.py"), encoding="utf-8") as f:
        return f.read()


def test_scan_build_output_docstring_states_the_narrower_true_claim():
    """S2: the module docstring must state check (A)'s open, named fail-open
    axis on `p`/`vp`-prefixed mnemonics -- citing D-SLM5009 as the open
    question it is -- rather than asserting fail-closed behaviour that a
    fabricated `p`/`vp` mnemonic (`vpneverheardof`) is measured to defeat.
    """
    text = _read_scan_build_output_source()
    assert "D-SLM5009" in text, (
        "scan_build_output.py's own module docstring no longer cites "
        "D-SLM5009 -- the open, unresolved question about check (A)'s "
        "p/vp fail-open behaviour must be named where a reader of this "
        "driver's own claims would look for it"
    )
    assert "fail-OPEN" in text or "fail-open" in text, (
        "scan_build_output.py's own module docstring no longer states "
        "that check (A) is fail-open on the p/vp structural-only class -- "
        "the property the strike and this ticket's own review measured "
        "false against the previous, settled-sounding wording"
    )


def test_scan_build_output_docstring_is_not_self_contradictory():
    """S2's second half: the old text claimed check (C)'s retirement
    "narrows what can fail the gate, it does not widen what can pass it"
    in the same breath as "the gate" -- narrowing what fails and widening
    what passes are the same fact stated twice, and 304 symbols that
    failed the pre-fold-39 gate pass the current one. The corrected text
    must not assert both halves of that contradiction about the same
    subject.
    """
    text = _read_scan_build_output_source()
    assert "does not widen what can pass it" not in text, (
        "scan_build_output.py's own module docstring still asserts check "
        "(C)'s retirement does not widen what can pass THE GATE -- self-"
        "contradictory with narrowing what can fail it, and false against "
        "the measured fact that symbols which failed the pre-fold-39 gate "
        "now pass"
    )


def test_read_function_symbol_names_lives_on_check_fp_free_scan():
    """M2: the ship gate's own production module now owns
    `_read_function_symbol_names` directly -- not through an import of the
    retired `run_fp_free_scan_real_corpus.py` driver.
    """
    import check_fp_free_scan as scan  # noqa: E402
    assert hasattr(scan, "_read_function_symbol_names"), (
        "check_fp_free_scan.py does not define _read_function_symbol_names "
        "-- M2's remedy (moving it out of the retired driver) did not land"
    )


def test_scan_build_output_no_longer_imports_the_retired_driver():
    """M2: `scan_build_output.py` -- the ship gate's own driver -- must not
    contain an `import` statement for `run_fp_free_scan_real_corpus`. The
    design calls that module "no longer load-bearing"; an import of it from
    the gate's own driver makes that claim false. (Prose mentioning the
    retired module's own name, e.g. explaining why it is no longer
    imported, is not the defect this pins.)
    """
    import re as _re
    text = _read_scan_build_output_source()
    assert not _re.search(r"^\s*(?:import|from)\s+run_fp_free_scan_real_corpus\b",
                           text, _re.MULTILINE), (
        "scan_build_output.py still contains an import statement for "
        "run_fp_free_scan_real_corpus -- the retired driver the design "
        "calls no longer load-bearing is still a hard dependency of the "
        "ship gate's own production driver"
    )


def test_run_fp_free_scan_real_corpus_delegates_rather_than_redefines():
    """M2: the retired driver must not carry its own independent
    definition of `_read_function_symbol_names` any more -- two copies of
    the same reader is exactly the drift M2 exists to close (one changes,
    the other silently does not). It may still expose the name (existing
    internal callers in that file use it), but only as a delegation to the
    production module's own definition.
    """
    path = os.path.join(_CI_DIR, "run_fp_free_scan_real_corpus.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "def _read_function_symbol_names" not in text, (
        "run_fp_free_scan_real_corpus.py still defines its own "
        "_read_function_symbol_names -- two independent copies of the "
        "same symbol-table reader is the drift hazard M2 exists to close"
    )
    assert "_read_function_symbol_names" in text, (
        "run_fp_free_scan_real_corpus.py no longer references "
        "_read_function_symbol_names at all -- its own internal callers "
        "(main()) need it delegated from check_fp_free_scan, not removed"
    )


def test_read_function_symbol_names_reads_real_object_symbols():
    """Must-accept, behavioural: the moved function still does its one job
    -- reading every function-typed symbol name out of a real compiled
    object's own symbol table -- unchanged by the move. Synthesizes a real
    COFF object (this suite's own pop14_make_objects.py constructor, which
    every population-14 cell in test_check_fp_free_scan.py already uses)
    with two named function symbols and one non-function symbol, and
    confirms the two function names come back and the non-function symbol
    does not.
    """
    import check_fp_free_scan as scan  # noqa: E402
    import tempfile

    fixtures_dir = os.path.join(_HERE, "fp_scan_fixtures")
    sys.path.insert(0, fixtures_dir)
    import pop14_make_objects as mk  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        obj_path = os.path.join(tmp, "t2371_hygiene.obj")
        mk.build_coff(
            obj_path, mk.COFF_X86_64, mk.X86_TEXT,
            symbols=[
                ("AlphaFn", mk.X86_HASH[0], True),
                ("BetaFn", mk.X86_BODY[0], True),
                ("SomeDataSymbol", mk.X86_HASH[0], False),
            ],
        )
        names = scan._read_function_symbol_names(obj_path)
    assert {"AlphaFn", "BetaFn"} <= names, (
        "check_fp_free_scan._read_function_symbol_names did not read back "
        "the function symbols a real, freshly-synthesized COFF object "
        "carries -- found {}".format(sorted(names))
    )
    assert "SomeDataSymbol" not in names, (
        "check_fp_free_scan._read_function_symbol_names returned a "
        "non-function symbol -- it must read only function-typed symbols"
    )
