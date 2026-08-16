"""B1 guard-vitality + coverage test (T-2123/T-2137, design §1/§6 B1): `calibrate_checkpoint.py`,
the missing general-purpose "calibrate this checkpoint" CLI.

Red: before B1, no such file exists (`ModuleNotFoundError`/`FileNotFoundError` depending on how
it is invoked) -- §5's clean-checkout gate step 5 has no target.

Green: an `argparse` CLI taking `--checkpoint <dir>` and `--out <dir>`, calling
`reference_pipeline.pipeline.load_model(checkpoint)` then
`reference_pipeline.artifact_cache.save_artifact(model, out, checkpoint)`, printing the written
fingerprint. No hardcoded paths -- `--checkpoint`/`--out` are required, never defaulted.

Coverage, using the new on-disk fixture checkpoint (`tools/_calibrate_checkpoint_fixture.py`,
not the pre-existing in-memory fixtures -- design §6 B1's own note: neither
`sslm_pinned_calibration_fixture.py` nor `pipeline.fixture_model` ever calls `load_model`, which
is the function this CLI actually calls and which requires a real checkpoint directory on
disk): the CLI's own argument-surface trust boundary (a file instead of a directory, a
nonexistent path, a nonexistent --out parent), required-argument guards, and the three content
rejections `load_model` already names (missing config.json, missing chat template, an unmapped
tensor), exercised through this CLI's own surface -- plus that it writes a `manifest.json`
`artifact_cache.load_artifact` can read back.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from _calibrate_checkpoint_fixture import build_fixture_checkpoint

import calibrate_checkpoint as CLI
import reference_pipeline.pipeline as pl


@pytest.fixture(scope="module")
def fixture_checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b1_fixture_ckpt")
    return build_fixture_checkpoint(tmp / "checkpoint")


@pytest.fixture(autouse=True)
def _reduce_calibration_to_two_records(monkeypatch):
    # Fixture-scale (design §6 B1a/B1b's own reduction shape): the calibration MATH runs for
    # real over the real fixture weights; only the record COUNT is reduced, so this test file
    # runs in a fraction of a second rather than iterating the full 600-record corpus.
    orig = pl.calibration_records
    monkeypatch.setattr(pl, "calibration_records", lambda: orig()[:2])
    yield


# --- argument-surface trust boundary (design §6 B1, folding T-2129 G1) ---------------------


def test_missing_required_checkpoint_argument_is_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        CLI.main(["--out", "somewhere"])
    assert excinfo.value.code != 0
    assert "--checkpoint" in capsys.readouterr().err


def test_missing_required_out_argument_is_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        CLI.main(["--checkpoint", "somewhere"])
    assert excinfo.value.code != 0
    assert "--out" in capsys.readouterr().err


def test_checkpoint_pointing_at_a_file_not_a_directory_is_rejected(tmp_path, capsys):
    a_file = tmp_path / "not_a_dir"
    a_file.write_text("x")
    rc = CLI.main(["--checkpoint", str(a_file), "--out", str(tmp_path / "out")])
    assert rc != 0
    assert "not a directory" in capsys.readouterr().err


def test_nonexistent_checkpoint_path_is_rejected(tmp_path, capsys):
    rc = CLI.main(["--checkpoint", str(tmp_path / "does_not_exist"), "--out", str(tmp_path / "out")])
    assert rc != 0
    assert "not a directory" in capsys.readouterr().err


def test_nonexistent_out_parent_is_rejected(fixture_checkpoint, tmp_path, capsys):
    rc = CLI.main([
        "--checkpoint", str(fixture_checkpoint),
        "--out", str(tmp_path / "no" / "such" / "parent" / "out"),
    ])
    assert rc != 0
    assert "does not exist" in capsys.readouterr().err


# --- content rejections inherited from pipeline.load_model, exercised through the CLI ------


def test_checkpoint_missing_config_json_is_rejected(fixture_checkpoint, tmp_path):
    broken = tmp_path / "no_config"
    shutil.copytree(fixture_checkpoint, broken)
    (broken / "config.json").unlink()
    with pytest.raises(pl.ConfigError):
        CLI.main(["--checkpoint", str(broken), "--out", str(tmp_path / "out1")])


def test_checkpoint_missing_chat_template_is_rejected(fixture_checkpoint, tmp_path):
    broken = tmp_path / "no_chat_template"
    shutil.copytree(fixture_checkpoint, broken)
    cfg = json.loads((broken / "tokenizer_config.json").read_text())
    del cfg["chat_template"]
    (broken / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(pl.ConfigError):
        CLI.main(["--checkpoint", str(broken), "--out", str(tmp_path / "out2")])


def test_checkpoint_with_an_unmapped_tensor_is_rejected(fixture_checkpoint, tmp_path):
    import numpy as np

    broken = tmp_path / "unmapped_tensor"
    shutil.copytree(fixture_checkpoint, broken)
    header = {}
    payload = bytearray()
    extra = np.zeros((2, 2), dtype=np.float32)
    header["model.some_unmapped_tensor.weight"] = {
        "dtype": "F32", "shape": [2, 2], "data_offsets": [0, extra.nbytes],
    }
    payload += extra.tobytes()
    header["__metadata__"] = {}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(broken / "model.safetensors", "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(bytes(payload))
    with pytest.raises(pl.UnsupportedOpSet):
        CLI.main(["--checkpoint", str(broken), "--out", str(tmp_path / "out3")])


# --- the successful CLI path, end to end ----------------------------------------------------


def test_cli_writes_a_manifest_that_load_artifact_can_read_back(fixture_checkpoint, tmp_path, capsys):
    import reference_pipeline.artifact_cache as ac

    out_dir = tmp_path / "artifact"
    rc = CLI.main(["--checkpoint", str(fixture_checkpoint), "--out", str(out_dir)])
    assert rc == 0

    stdout = capsys.readouterr().out
    assert "written fingerprint:" in stdout

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert "source_fingerprint" in manifest

    reloaded = ac.load_artifact(str(out_dir))
    assert reloaded is not None
