"""Provenance hashing round-trip: `write_provenance` then `check_provenance` must agree on a clean
tree, and must detect a mismatch when a checked file changes on disk (mirrors this repo's existing
`tests/reference/check_provenance.py` behavior)."""

import os

from ..provenance import check_provenance, write_provenance


def test_write_then_check_round_trips_clean(tmp_path):
    (tmp_path / "a.json").write_text('{"x": 1}\n', encoding="utf-8", newline="\n")
    (tmp_path / "b.json").write_text('{"y": 2}\n', encoding="utf-8", newline="\n")
    prov_path = str(tmp_path / "PROVENANCE.md")

    write_provenance(prov_path, str(tmp_path), ["a.json", "b.json"], corpus_version="test-v1")
    mismatches = check_provenance(prov_path, str(tmp_path))
    assert mismatches == []


def test_check_detects_content_drift(tmp_path):
    (tmp_path / "a.json").write_text('{"x": 1}\n', encoding="utf-8", newline="\n")
    prov_path = str(tmp_path / "PROVENANCE.md")
    write_provenance(prov_path, str(tmp_path), ["a.json"], corpus_version="test-v1")

    (tmp_path / "a.json").write_text('{"x": 2}\n', encoding="utf-8", newline="\n")
    mismatches = check_provenance(prov_path, str(tmp_path))
    assert len(mismatches) == 1
    assert "a.json" in mismatches[0]


def test_check_detects_missing_file(tmp_path):
    (tmp_path / "a.json").write_text('{"x": 1}\n', encoding="utf-8", newline="\n")
    prov_path = str(tmp_path / "PROVENANCE.md")
    write_provenance(prov_path, str(tmp_path), ["a.json"], corpus_version="test-v1")

    os.remove(tmp_path / "a.json")
    mismatches = check_provenance(prov_path, str(tmp_path))
    assert len(mismatches) == 1
    assert "not found" in mismatches[0]
