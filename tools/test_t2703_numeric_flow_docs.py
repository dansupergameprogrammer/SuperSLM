"""T-2703 R4 — release-facing documents state one calibration-flow contract."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_DOCUMENTS = (
    _ROOT / "docs" / "fused_k_numeric_spec.md",
    _ROOT / "docs" / "quickstart.md",
    _ROOT / "CHANGELOG.md",
)
_REQUIRED = (
    "A/B/C",
    "max(float, A)",
    "max(float, A, B)",
    "pass C",
    "one per million",
    "5,226 s",
    "1,495 s",
    "30-minute",
)


def test_release_documents_agree_on_the_fused_k_capture_flow():
    for document in _DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        missing = [phrase for phrase in _REQUIRED if phrase not in text]
        assert not missing, f"{document.relative_to(_ROOT)} missing {missing}"
