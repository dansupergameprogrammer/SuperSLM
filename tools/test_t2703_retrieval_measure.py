"""Vitality tests for T-2703 compiled-probe retrieval measurement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_MEASURE_PATH = Path(__file__).with_name("t2703_retrieval_measure.py")
_SPEC = importlib.util.spec_from_file_location("t2703_retrieval_measure", _MEASURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
measure = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(measure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyze_refuses_distinct_equivalence_dumps_even_when_each_hash_is_honest(tmp_path: Path):
    """The analyze gate checks the actual two payloads, not only a producer claim."""
    candidate = tmp_path / "candidate.sslm"
    probe = tmp_path / "probe.exe"
    candidate.write_bytes(b"candidate")
    probe.write_bytes(b"probe")
    full = tmp_path / "equivalence-full.bin"
    final_only = tmp_path / "equivalence-final-only.bin"
    full.write_bytes(b"ordinary final hidden")
    final_only.write_bytes(b"different final hidden")
    (tmp_path / "probe-equivalence.json").write_text(json.dumps({
        "artifact": str(candidate.resolve()),
        "artifact_sha256": _sha256(candidate),
        "probe_sha256": _sha256(probe),
        "full_path_dump_sha256": _sha256(full),
        "final_hidden_only_dump_sha256": _sha256(final_only),
        "equivalent": True,
    }), encoding="utf-8")
    args = SimpleNamespace(
        output=tmp_path,
        candidate=candidate,
        candidate_sha256=_sha256(candidate),
        probe=probe,
    )
    with pytest.raises(RuntimeError, match="probe equivalence dumps differ"):
        measure.analyze_phase(args)
