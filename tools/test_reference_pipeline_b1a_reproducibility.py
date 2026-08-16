"""B1a guard-vitality test (T-2123/T-2137, design §6 B1a): the plan's §11 artifact-
reproducibility formula, re-asserted post-vendoring.

Red: no cell before B1a proves `converter version + checkpoint hash + config + calibration-
corpus hash -> identical artifact bytes` survives the vendoring move -- B0-B3 prove
import-boundary self-sufficiency and corpus byte-identity to the pre-move original, neither
of which is the plan's own reproducibility claim.

Green: run `calibrate_checkpoint.py` (B1) twice against the same small on-disk fixture
checkpoint and assert byte-identical `artifact_cache._compute_fingerprint` output across the
two runs -- the mutation-provable form of the plan's §11 claim, checked from the vendored
location rather than inherited by assertion. CI-runnable, at TWO reduced cells stacked (named
here explicitly, per Poirot casebook f83afe0-t2137-vendoring-review.md M5, which found the
prior docstring's "fixture-scale" named only the checkpoint and left the corpus reduction
unstated): the on-disk checkpoint is a tiny fixture (not the real 1.5B model), AND
calibration is reduced to the first 2 of the real 600-record corpus
(`pl.calibration_records` monkeypatched below). §11's own formula names the
calibration-corpus hash as one of its inputs; this cell exercises reproducibility over 2 of
those 600 records, not the full corpus -- narrower than the full release-gate claim, stated
rather than left implicit.
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
    # the summary fingerprint. This is NOT an independent second witness (Poirot casebook
    # f83afe0-t2137-vendoring-review.md, M5): the fingerprint IS a hash of exactly these
    # arrays (artifact_cache._compute_fingerprint), so a hash of X and X itself is one
    # reading, not two -- StandardsDocument §5.4's rule that a reference sharing its input
    # with the thing it grades is not a reference. What this second check DOES establish,
    # independently of the fingerprint comparison above: it LOCALIZES a mismatch to a named
    # array rather than only reporting "the hash differs" -- useful for diagnosis, not a
    # second proof that reproducibility holds.
    import numpy as np

    arr1 = np.load(out1 / "arrays.npz")
    arr2 = np.load(out2 / "arrays.npz")
    assert set(arr1.files) == set(arr2.files)
    for name in arr1.files:
        assert np.array_equal(arr1[name], arr2[name]), f"array {name!r} differs across runs"
