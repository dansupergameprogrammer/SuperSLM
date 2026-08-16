"""S2 fix (Poirot casebook f83afe0-t2137-vendoring-review.md): the port dropped a
corpus-driven cell in the ported `test_constrained_decode.py` silently.

`test_constrained_decode.py:395`'s own `CORPUS_PATH` was left with its pre-move
`parents[3]/Claude/Docs/spike/shopkeeper_corpus_v1.jsonl` derivation -- at the source
address that resolved into `D:\\Wizard`; at the vendored address it resolves into a path
this repository does not have, `_corpus_golds()` returns `[]`, and the cell that runs the
constrained decoder against real corpus golds is skipped rather than run. The skip is the
suite's only skip, and the build log recorded it under the wrong reason (a symlink
limitation that does not apply to this file at all).

Red: before the fix, `CORPUS_GOLDS` is empty and the cell's own `@pytest.mark.skipif` fires.
Green: `CORPUS_PATH` reads from `reference_pipeline.pipeline.CALIBRATION_CORPUS_PATH` (B3's
own vendored location) rather than re-deriving a second path off `__file__`, `CORPUS_GOLDS`
is the real 600-record corpus, and the cell runs rather than skips.
"""

import importlib
import os
import sys

_TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_pipeline", "tests")


def _load_tcd():
    if _TESTS_DIR not in sys.path:
        sys.path.insert(0, _TESTS_DIR)
    import test_constrained_decode as tcd
    importlib.reload(tcd)
    return tcd


def test_ported_constrained_decode_corpus_path_resolves_inside_reference_pipeline():
    tcd = _load_tcd()

    assert "reference_pipeline" in str(tcd.CORPUS_PATH).replace("\\", "/"), (
        f"test_constrained_decode.py's CORPUS_PATH ({tcd.CORPUS_PATH}) still derives a path "
        f"outside tools/reference_pipeline/ -- the stale parents[3] walk was not fixed"
    )
    assert tcd.CORPUS_PATH.is_file(), f"CORPUS_PATH ({tcd.CORPUS_PATH}) does not exist on disk"


def test_ported_constrained_decode_corpus_golds_are_not_empty():
    tcd = _load_tcd()

    assert len(tcd.CORPUS_GOLDS) == 600, (
        f"CORPUS_GOLDS has {len(tcd.CORPUS_GOLDS)} records, expected 600 -- the real-corpus "
        f"gold cell is still running against an empty or partial corpus"
    )
