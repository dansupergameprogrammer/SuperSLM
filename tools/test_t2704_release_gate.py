"""Focused vitality checks for the T-2704 release-gate instrument."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_GATE_PATH = Path(__file__).with_name("t2704_release_gate.py")
_MANIFEST_PATH = _GATE_PATH.parents[1] / "tests" / "data" / "t2704_release_manifest.json"
_SPEC = importlib.util.spec_from_file_location("t2704_release_gate", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def test_commission_exercises_each_provenance_refusal_and_undecodable_child(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    assert gate.commission(receipt) == 0
    results = json.loads(receipt.read_text(encoding="utf-8"))["results"]
    assert results["provenance_healthy"] == 0
    assert results["provenance_nonancestor"] == 1
    assert results["provenance_changed_source"] == 1
    assert results["provenance_changed_executable"] == 1
    assert results["undecodable_child_rejection"] == 1


def test_commission_exercises_the_exact_paired_retrieval_rule(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    assert gate.commission(receipt) == 0
    report = json.loads(receipt.read_text(encoding="utf-8"))
    results = report["results"]
    assert all(results[name] == 0 for name in (
        "retrieval_observed_19_11", "retrieval_boundary_accept_19_11", "retrieval_zero_discordant",
    ))
    assert all(results[name] == 1 for name in (
        "retrieval_worse_25_5", "retrieval_boundary_reject_20_10", "retrieval_missing_b",
        "retrieval_missing_c", "retrieval_malformed_paired_count", "retrieval_float_anchor_outside",
    ))
    assert report["observed_retrieval_significance"] == {
        "b": 19,
        "c": 11,
        "discordant_total": 30,
        "one_sided_exact_p": 107636402 / 1073741824,
        "one_sided_exact_p_fraction": "107636402/1073741824",
    }


def test_byte_capture_preserves_invalid_utf8_as_a_decisive_witness():
    assert gate.captured_text(None) == "<not captured>"
    assert gate.captured_text(b"\x8f") == "\\x8f"
    with pytest.raises(ValueError, match=r"stderr: \\x8f"):
        gate.run([gate.sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'\\x8f'); sys.exit(1)"],
                 "invalid-byte child")


def test_current_manifest_accepts_the_rebuilt_compiled_diagnostics_before_execution():
    # Dev-box compiled diagnostics: present on the release machine, absent on CI runners. Skip rather than
    # fail when any is missing; when all are present, the check runs unchanged.
    for required in (Path("D:/_artifacts/superslm/t3011-v190-diag-b/layer-tracer/sslm_layer_trace.exe"),
                     Path("D:/_artifacts/superslm/t3011-v190-diag-b/retrieval-probe/build/Release/t2701_cpu_forward_probe.exe")):
        if not required.is_file():
            pytest.skip(f"dev-box diagnostic absent: {required}")
    manifest = gate.load(_MANIFEST_PATH)
    records = gate.validate_compiled_diagnostic_provenance(manifest, _MANIFEST_PATH)
    assert records == {
        "baseline layer tracer": {
            "path": "D:\\_artifacts\\superslm\\t3011-v190-diag-b\\layer-tracer\\sslm_layer_trace.exe",
            "source_commit": "4e8d3de0e3674e84af7216fcb9bd46d7b13c04f4",
            "build_command": ["cmd /c tools\\build_layer_trace.bat"],
            "sha256": "5e00110f12bd6fbd684d393c87b25267b3d6a7f69bcf354649e69656c600a558",
        },
        "retrieval CPU probe": {
            "path": "D:\\_artifacts\\superslm\\t3011-v190-diag-b\\retrieval-probe\\build\\Release\\t2701_cpu_forward_probe.exe",
            "source_commit": "4e8d3de0e3674e84af7216fcb9bd46d7b13c04f4",
            "build_command": [
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe -S . -B D:/_artifacts/superslm/t3011-v190-diag-b/retrieval-probe/build -DSUPERSLM_BUILD_GPU=ON -DBUILD_TESTING=OFF",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe --build D:/_artifacts/superslm/t3011-v190-diag-b/retrieval-probe/build --config Release --target t2701_cpu_forward_probe",
            ],
            "sha256": "d5309590acf64aa84d5e2a6276d29f42cd1c9bdd9d1aa0dffe2aa3f24efc3462",
        },
    }
