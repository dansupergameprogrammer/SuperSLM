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


def test_byte_capture_preserves_invalid_utf8_as_a_decisive_witness():
    assert gate.captured_text(None) == "<not captured>"
    assert gate.captured_text(b"\x8f") == "\\x8f"
    with pytest.raises(ValueError, match=r"stderr: \\x8f"):
        gate.run([gate.sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'\\x8f'); sys.exit(1)"],
                 "invalid-byte child")


def test_current_manifest_refuses_the_nonancestor_compiled_diagnostic_before_execution():
    manifest = gate.load(_MANIFEST_PATH)
    with pytest.raises(ValueError, match="source is not an ancestor of gate HEAD"):
        gate.validate_compiled_diagnostic_provenance(manifest, _MANIFEST_PATH)
