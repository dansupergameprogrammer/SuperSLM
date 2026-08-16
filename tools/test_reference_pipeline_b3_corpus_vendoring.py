"""B3 guard-vitality + coverage test (T-2123/T-2137, design §6 B3).

Red: before B3, `reference_pipeline.pipeline.CALIBRATION_CORPUS_PATH` is derived from
`Path(__file__).resolve().parents[2]`, which after B0's move resolves to this repo's own
root, not a `D:\\Wizard` sibling -- `<repo>/Claude/Docs/spike/shopkeeper_corpus_v1.jsonl`
does not exist, so `pipeline.calibration_records()` raises `FileNotFoundError`.

Green: the corpus is vendored to `tools/reference_pipeline/data/shopkeeper_corpus_v1.jsonl`
and `CALIBRATION_CORPUS_PATH`'s derivation is rewritten to a path relative to
`reference_pipeline/`'s own `__file__` rather than walking `parents[2]` into a sibling
`Claude/Docs/` tree that has no reason to exist inside this repo. `_corpus_sha256()`
(pinned in the original spike's own `smoke_eval_run.py:69`) confirms the moved file is
byte-identical to the one read from Wizard, not a stale or partial copy.
"""

import hashlib
from pathlib import Path

import pytest

# Pinned in smoke_eval_run.py:69 (D:\Wizard\Tools\superslm_spike\smoke_eval_run.py) --
# the known-good hash of Claude/Docs/spike/shopkeeper_corpus_v1.jsonl's LF-normalized bytes.
KNOWN_CORPUS_SHA256 = "640a5770da9e093446e2aece8b3b7e2869476963e54039af1e897ec750e4d9ae"


def test_calibration_corpus_path_points_inside_reference_pipeline():
    import reference_pipeline.pipeline as pl

    assert "reference_pipeline" in str(pl.CALIBRATION_CORPUS_PATH).replace("\\", "/"), (
        f"CALIBRATION_CORPUS_PATH ({pl.CALIBRATION_CORPUS_PATH}) does not resolve inside "
        f"tools/reference_pipeline/ -- the parents[2] walk into a sibling Claude/Docs/ tree "
        f"was not rewritten"
    )
    assert pl.CALIBRATION_CORPUS_PATH.exists(), (
        f"CALIBRATION_CORPUS_PATH ({pl.CALIBRATION_CORPUS_PATH}) does not exist on disk"
    )


def test_calibration_records_loads_all_600_records_with_no_wizard_on_path():
    import sys
    assert not any("wizard" in p.lower() for p in sys.path)

    import reference_pipeline.pipeline as pl

    records = pl.calibration_records()
    assert len(records) == 600, f"expected 600 calibration records, got {len(records)}"


def test_vendored_corpus_is_byte_identical_to_the_original_wizard_copy():
    import reference_pipeline.pipeline as pl

    data = pl.CALIBRATION_CORPUS_PATH.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(data).hexdigest()
    assert digest == KNOWN_CORPUS_SHA256, (
        f"vendored corpus sha256 {digest} does not match the known-good pin "
        f"{KNOWN_CORPUS_SHA256} -- the moved file is stale or partial"
    )
