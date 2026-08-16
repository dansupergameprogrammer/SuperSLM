"""B2 guard-vitality + coverage test (T-2123/T-2137, design §6 B2).

Red: `reference_pipeline.calibration_prompt` does not exist until B2 lands, and before B2
`reference_pipeline.pipeline`'s calibration path lazily imports the five needed symbols from
`superslm_spike.baseline` -- unreachable once `D:\\Wizard` is off `sys.path`.

Green: `SYSTEM_PROMPT`, `USER_TEMPLATE`, `TURN_PREFIX`, `record_turns`, `build_prompt`
extracted verbatim from `baseline.py` (the fp16-CUDA eval driver, not vendored -- only these
five module-scope symbols are). Asserts, per the design's §2.2 finding, that importing the
extract pulls in neither `torch` nor `transformers` (checked via `sys.modules` before/after,
since neither symbol's own construction needs them), unlike importing `baseline.py` itself
would (that module imports `torch`/`transformers.AutoModelForCausalLM` at module scope for
its own unrelated `main()` eval driver).

Coverage: the existing `record_turns`/`build_prompt` behavior (flat `utterance` vs.
multi-turn `correction` records) has no dedicated unit test in the source spike (checked:
`tests/` there holds none testing `baseline.py`'s prompt functions directly) -- a coverage
gap inherited from the source, named per the design rather than silently carried forward.
The two cases below (flat record, multi-turn record) are new coverage this build item adds.
"""

import sys


def test_calibration_prompt_module_imports_with_no_wizard_on_path():
    assert not any("wizard" in p.lower() for p in sys.path)
    import reference_pipeline.calibration_prompt as cp  # noqa: F401


def test_calibration_prompt_import_pulls_in_neither_torch_nor_transformers():
    for mod in ("torch", "transformers"):
        sys.modules.pop(mod, None)
    import reference_pipeline.calibration_prompt  # noqa: F401
    assert "torch" not in sys.modules, (
        "importing reference_pipeline.calibration_prompt pulled in torch -- the whole point "
        "of extracting five symbols rather than importing baseline.py wholesale is to avoid "
        "this"
    )


def test_system_prompt_and_templates_match_the_original_source():
    from reference_pipeline.calibration_prompt import SYSTEM_PROMPT, USER_TEMPLATE, TURN_PREFIX

    assert SYSTEM_PROMPT.startswith(
        "You are an intent-extraction component for a tattoo shop's front desk."
    )
    assert USER_TEMPLATE == "{turns}"
    assert TURN_PREFIX == "Customer: "


def test_record_turns_flat_utterance_record():
    from reference_pipeline.calibration_prompt import record_turns

    assert record_turns({"utterance": "book me for tuesday"}) == ["book me for tuesday"]


def test_record_turns_multi_turn_correction_record():
    from reference_pipeline.calibration_prompt import record_turns

    turns = ["actually make that wednesday", "no wait, thursday"]
    assert record_turns({"turns": turns}) == turns


def test_build_prompt_renders_turns_with_the_customer_prefix():
    from reference_pipeline.calibration_prompt import build_prompt

    assert build_prompt({"utterance": "book me for tuesday"}) == "Customer: book me for tuesday"
    assert build_prompt({"turns": ["a", "b"]}) == "Customer: a\nCustomer: b"


def test_pipeline_run_prompt_messages_imports_from_calibration_prompt_not_baseline():
    import inspect

    import reference_pipeline.pipeline as pl

    src = inspect.getsource(pl.run_prompt_messages)
    assert "calibration_prompt" in src, (
        "run_prompt_messages no longer imports from reference_pipeline.calibration_prompt"
    )
    assert "baseline" not in src, (
        "run_prompt_messages still references baseline.py -- the lazy import was not repointed"
    )
