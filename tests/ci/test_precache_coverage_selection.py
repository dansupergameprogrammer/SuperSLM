"""The precache coverage driver runs exactly its allowlist and fails on stale or empty selections."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("precache_suites", ROOT / "tools/ci/run_precache_suites.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_selection_keeps_registration_order_and_leaves_out_unlisted_cells():
    assert driver.select_cells(["a", "box", "b"], ["b", "a"]) == ["a", "b"]


@pytest.mark.parametrize("registered,listed", [([], ["x"]), (["x", "x"], ["x"]), (["x"], []), (["x"], ["x", "x"]),
                                               (["x"], ["gone"]), (["bad id"], ["bad id"])])
def test_invalid_selection_fails(registered, listed):
    with pytest.raises(ValueError):
        driver.select_cells(registered, listed)


def test_allowlist_names_the_two_suites_and_no_synthetic_or_box_cell():
    manifest = json.loads(driver.CELLS.read_text())
    assert {key for key in manifest if not key.startswith("_")} == set(driver.SUITES)
    for suite in driver.SUITES:
        cells = manifest[suite]
        assert cells and len(set(cells)) == len(cells)
        assert not [c for c in cells if "synthetic" in c or "real-gpu" in c or ":reference" in c]


def test_failed_suite_fails_driver_and_selection_is_recorded(tmp_path, monkeypatch):
    for name in ("pkv_def", "pkv_qk", "pkv_odd", "pkv_32k"):
        (tmp_path / f"{name}.sslm").touch()
    manifest = tmp_path / "cells.json"
    manifest.write_text(json.dumps({suite: ["kept"] for suite in driver.SUITES}))
    monkeypatch.setattr(driver, "CELLS", manifest)
    commands = []

    def run(command, **kwargs):
        if command[-1] == "--list":
            return SimpleNamespace(stdout="kept\nbox\n", returncode=0)
        commands.append((command, kwargs["env"].get("LLVM_PROFILE_FILE")))
        return SimpleNamespace(returncode=3 if "unpinned" in command[0] else 0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr("sys.argv", ["run_precache_suites.py", "--binary-dir", str(tmp_path), "--fixture-dir",
                                     str(tmp_path), "--profile-dir", str(tmp_path / "p"), "--receipt", str(receipt)])
    assert driver.main() == 1
    assert [c[0][1:] for c in commands] == [["=kept"], ["=kept"]]
    assert [Path(c[1]).name for c in commands] == ["precache_e1.profraw", "precache_unpinned.profraw"]
    recorded = json.loads(receipt.read_text())["suites"]
    assert [s["omitted"] for s in recorded] == [["box"], ["box"]]
    assert [s["exit_code"] for s in recorded] == [0, 3]
