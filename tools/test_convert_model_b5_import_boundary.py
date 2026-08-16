"""B5 guard-vitality test (T-2123/T-2137, design §6 B5): `convert_model.py` drops
`_SPIKE_ROOT`/`sys.path` insertion and imports `reference_pipeline` directly.

Red: today `convert_model._load_spike()` requires `D:\\Wizard\\Tools` on `sys.path` (its
`_SPIKE_ROOT` hardcode) -- false to prove with `D:\\Wizard` absent from `sys.path` and the
module reloaded fresh.

Green: `_load_spike()` resolves to `from reference_pipeline import artifact_cache, pipeline`
with no `_SPIKE_ROOT` constant and no `sys.path` mutation anywhere in the module.
"""

import importlib
import inspect
import sys


def test_convert_model_has_no_spike_root_constant():
    import convert_model

    assert not hasattr(convert_model, "_SPIKE_ROOT"), (
        "convert_model._SPIKE_ROOT still exists -- B5's cross-tree hardcode was not removed"
    )


def test_convert_model_load_spike_does_not_mutate_sys_path():
    import convert_model

    before = list(sys.path)
    convert_model._load_spike()
    assert sys.path == before, (
        "convert_model._load_spike() mutated sys.path -- the sys.path.insert(0, _SPIKE_ROOT) "
        "pattern was not removed"
    )


def test_convert_model_load_spike_succeeds_with_no_wizard_on_path():
    assert not any("wizard" in p.lower() for p in sys.path)

    import convert_model
    importlib.reload(convert_model)

    artifact_cache, pipeline = convert_model._load_spike()
    assert artifact_cache.__name__ == "reference_pipeline.artifact_cache"
    assert pipeline.__name__ == "reference_pipeline.pipeline"


def test_convert_model_source_has_no_superslm_spike_reference():
    import convert_model

    src = inspect.getsource(convert_model)
    assert "superslm_spike" not in src, "convert_model.py still references superslm_spike"
    assert r"D:\Wizard" not in src, "convert_model.py still hardcodes a D:\\Wizard path"
