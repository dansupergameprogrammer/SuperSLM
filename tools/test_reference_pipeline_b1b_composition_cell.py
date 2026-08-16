"""B1b guard-vitality test (T-2123/T-2137, design §6 B1b): a CI-runnable, fixture-scale
composition cell between B1's output (calibration) and Closure A's input (conversion).

§5's full gate is the only place Closure B (calibration) and Closure A (conversion) are
proven to compose, and it is real-checkpoint, ~105-minute, manual, non-CI -- correctly
labeled as such, but not a substitute for a cheap regression check on the seam between them.

Red: no cell before B1b runs B1's write path and `convert_model.py`'s read path back to
back.

Green: using B1's new on-disk fixture checkpoint, run `calibrate_checkpoint.py` end to end
into a fixture-scale artifact directory, then feed that directory straight into
`convert_model.py`'s existing `artifact_cache.load_artifact` + `V.validate_model` +
`build_sections` path, and assert the composed run completes with no format-mismatch error.
Explicitly a mechanism check (StandardsDocument §5.4) -- it proves the seam does not silently
drift, not that the product works at real scale, which remains §5's job alone.
"""

import calibrate_checkpoint as CLI
import convert_model as CM
import reference_pipeline.pipeline as pl
import sslm_convert_validate as V
from _calibrate_checkpoint_fixture import build_fixture_checkpoint


def test_calibrate_then_convert_model_composes_at_fixture_scale(tmp_path, monkeypatch):
    orig = pl.calibration_records
    monkeypatch.setattr(pl, "calibration_records", lambda: orig()[:2])

    checkpoint = build_fixture_checkpoint(tmp_path / "checkpoint")
    artifact_dir = tmp_path / "artifact"

    rc = CLI.main(["--checkpoint", str(checkpoint), "--out", str(artifact_dir)])
    assert rc == 0

    artifact_cache, _pipeline = CM._load_spike()
    model = artifact_cache.load_artifact(str(artifact_dir))

    # Phase 1: validate (the same reject-over-degrade phase convert_model.main() runs).
    V.validate_model(
        model, fold_ops_tensor=CM._fold_ops_tensor, ctx_fold_tensor=CM._ctx_fold_tensor,
        unicode_major=V.PINNED_UNICODE_VERSION[0],
        unicode_minor=V.PINNED_UNICODE_VERSION[1],
        unicode_patch=V.PINNED_UNICODE_VERSION[2],
    )

    # Phase 2: serialize -- the composed run completes with no format-mismatch error.
    sections, fold_approximation_error = CM.build_sections(model)
    assert len(sections) > 0
    assert fold_approximation_error is not None
