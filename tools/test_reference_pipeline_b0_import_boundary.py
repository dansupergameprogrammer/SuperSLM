"""B0 guard-vitality test (T-2123/T-2137): `reference_pipeline` must be importable, and must
resolve `pipeline`/`artifact_cache`/`intmath`/`rope`/`silu_lut`/`constrain`, with no `D:\\Wizard`
path present anywhere on `sys.path` at import time.

Before B0 lands, this package does not exist at all (`ModuleNotFoundError: No module named
'reference_pipeline'`) -- confirmed red by running this file against the pre-B0 tree. After B0,
the six Closure-A files (design `Claude/Vitruvius/t2123-converter-vendoring-design-2026-08-16.md`
§2.1) live under `tools/reference_pipeline/` as a real package (an `__init__.py`, closing the
PEP 420 implicit-namespace fragility §2.1/§3.4 named) and every internal `from superslm_spike
import ...` is rewritten to `from reference_pipeline import ...`.
"""

import sys

import pytest


def test_no_wizard_path_present_before_import():
    # A hostile/misconfigured sys.path would defeat the point of this test by making the
    # cross-tree import succeed even though this package's own code no longer needs it.
    assert not any("wizard" in p.lower() for p in sys.path), (
        "D:\\Wizard is present on sys.path -- this test cannot prove reference_pipeline is "
        "self-sufficient while the old cross-tree path is still importable"
    )


@pytest.mark.parametrize(
    "module_name",
    ["reference_pipeline", "reference_pipeline.pipeline", "reference_pipeline.artifact_cache",
     "reference_pipeline.intmath", "reference_pipeline.rope", "reference_pipeline.silu_lut",
     "reference_pipeline.constrain"],
)
def test_reference_pipeline_submodule_imports_with_no_wizard_on_path(module_name):
    assert not any("wizard" in p.lower() for p in sys.path)
    __import__(module_name)


def test_reference_pipeline_has_no_leftover_superslm_spike_reference():
    import reference_pipeline.pipeline as pl
    import reference_pipeline.artifact_cache as ac
    import reference_pipeline.rope as rope_mod
    import reference_pipeline.silu_lut as silu_mod

    import inspect

    for mod in (pl, ac, rope_mod, silu_mod):
        src = inspect.getsource(mod)
        assert "superslm_spike" not in src, (
            f"{mod.__name__} still references superslm_spike -- B0's rewrite is incomplete"
        )
