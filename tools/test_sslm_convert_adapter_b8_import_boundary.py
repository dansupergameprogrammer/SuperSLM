"""B8 guard-vitality test (T-2123/T-2137, design §6 B8, D-SLM3449, §3.6 class member #2):
`sslm_convert_adapter.py` drops `_SPIKE_ROOT`/`sys.path` insertion and imports
`reference_pipeline` directly, the identical shape B5 fixes for `convert_model.py`.

Red: today `sslm_convert_adapter._load_spike()` requires `D:\\Wizard\\Tools` on `sys.path`
(its own `_SPIKE_ROOT` hardcode, confirmed at source identical in shape to `convert_model.py`'s
own pre-B5 defect) -- false to prove with `D:\\Wizard` absent from `sys.path`.

Green: `_load_spike()` resolves to `from reference_pipeline import pipeline` with no
`_SPIKE_ROOT` constant and no `sys.path` mutation anywhere in the module. §3.6's ruling scopes
this build item to the import boundary only -- the delta-fold derivation, the B3 statistical
gate, and the artifact-format logic this file also carries are untouched.
"""

import importlib
import inspect
import sys


def test_sslm_convert_adapter_has_no_spike_root_constant():
    import sslm_convert_adapter

    assert not hasattr(sslm_convert_adapter, "_SPIKE_ROOT"), (
        "sslm_convert_adapter._SPIKE_ROOT still exists -- B8's cross-tree hardcode was not "
        "removed"
    )


def test_sslm_convert_adapter_load_spike_does_not_mutate_sys_path():
    import sslm_convert_adapter

    before = list(sys.path)
    sslm_convert_adapter._load_spike()
    assert sys.path == before, (
        "sslm_convert_adapter._load_spike() mutated sys.path -- the "
        "sys.path.insert(0, _SPIKE_ROOT) pattern was not removed"
    )


def test_sslm_convert_adapter_load_spike_succeeds_with_no_wizard_on_path():
    assert not any("wizard" in p.lower() for p in sys.path)

    import sslm_convert_adapter
    importlib.reload(sslm_convert_adapter)

    pipeline = sslm_convert_adapter._load_spike()
    assert pipeline.__name__ == "reference_pipeline.pipeline"


def test_sslm_convert_adapter_source_has_no_superslm_spike_reference():
    import sslm_convert_adapter

    src = inspect.getsource(sslm_convert_adapter)
    assert "superslm_spike" not in src, (
        "sslm_convert_adapter.py still references superslm_spike"
    )
    assert r"D:\Wizard" not in src, (
        "sslm_convert_adapter.py still hardcodes a D:\\Wizard path"
    )
