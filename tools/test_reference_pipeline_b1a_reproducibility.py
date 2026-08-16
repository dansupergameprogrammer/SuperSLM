"""B1a guard-vitality test (T-2123/T-2137, design §6 B1a): the plan's §11 artifact-
reproducibility formula, re-asserted post-vendoring.

Red: no cell before B1a proves `converter version + checkpoint hash + config + calibration-
corpus hash -> identical artifact bytes` survives the vendoring move -- B0-B3 prove
import-boundary self-sufficiency and corpus byte-identity to the pre-move original, neither
of which is the plan's own reproducibility claim.

Green: run `calibrate_checkpoint.py` (B1) twice against the same small on-disk fixture
checkpoint and assert byte-identical `artifact_cache._compute_fingerprint` output across the
two runs -- the mutation-provable form of the plan's §11 claim, checked from the vendored
location rather than inherited by assertion. CI-runnable (fixture-scale, no real checkpoint
download).
"""

import calibrate_checkpoint as CLI
import reference_pipeline.pipeline as pl
from _calibrate_checkpoint_fixture import build_fixture_checkpoint


def test_two_calibration_runs_against_the_same_fixture_checkpoint_are_byte_identical(
    tmp_path, monkeypatch, capsys
):
    orig = pl.calibration_records
    monkeypatch.setattr(pl, "calibration_records", lambda: orig()[:2])

    checkpoint = build_fixture_checkpoint(tmp_path / "checkpoint")

    out1 = tmp_path / "artifact1"
    out2 = tmp_path / "artifact2"
    rc1 = CLI.main(["--checkpoint", str(checkpoint), "--out", str(out1)])
    fp1_line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("written fingerprint")][0]
    rc2 = CLI.main(["--checkpoint", str(checkpoint), "--out", str(out2)])
    fp2_line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("written fingerprint")][0]

    assert rc1 == 0 and rc2 == 0
    assert fp1_line == fp2_line, (
        f"two calibration runs against the same fixture checkpoint produced different "
        f"fingerprints: {fp1_line!r} vs {fp2_line!r} -- §11's reproducibility formula does "
        f"not hold post-vendoring"
    )

    # And the two artifact directories' own weight/scale arrays agree bit-for-bit, not just
    # the summary fingerprint -- the fingerprint is a hash of exactly these arrays, so this is
    # a second, independent read of the same claim.
    import numpy as np

    arr1 = np.load(out1 / "arrays.npz")
    arr2 = np.load(out2 / "arrays.npz")
    assert set(arr1.files) == set(arr2.files)
    for name in arr1.files:
        assert np.array_equal(arr1[name], arr2[name]), f"array {name!r} differs across runs"
