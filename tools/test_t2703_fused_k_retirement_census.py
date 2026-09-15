"""T-2703 R3 — anchored source census for the retired tensor-wide fused-K family.

The census is intentionally repository-wide over executable sources and normative
documentation. A legacy spelling is legal only at an explicitly named refusal,
coverage, no-QK compatibility, or immutable-history location; no broad directory
exemption exists. The active oracle reader has a separate zero-hit rule because a
renamed second landing is still a reader even if no serialized key survives.
"""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SEARCH_ROOTS = ("src", "include", "tools", "tests", "docs")
_SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".hlsl", ".hlsli", ".py", ".md"}
_TERMS = (
    "k_normed" + "_head", ".k_normed", "k_norm" + ".requant",
    "softmax" + ".input", "k_norm" + "_landing", "QK_NORM_K_" + "STAGE",
    "QK_NORM_K_" + "SCALE", "work_qk_norm_k_" + "stage_off",
    "work_qk_norm_k_" + "scale_off", "k_normed" + "_landing_saturation_count",
)
_ACTIVE_READER_TERMS = ("land" + "_" + "retired" + "_k", "retired" + "_k")
_REFUSAL_ANCHOR = ("src/model.cpp", "k_normed" + "_head")
_HISTORICAL_REFERENCE = "tests/reference/superslm_spike/pipeline.py"
_COVERAGE_TEXT = {
    "tools/reference_pipeline/tests/test_armd_arme_kv_calibration.py",
    "tools/reference_pipeline/tests/test_pipeline.py",
    "tools/reference_pipeline/tests/test_t2606_calibration_key_domains.py",
}


def _sources():
    return {
        path.relative_to(_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for directory in _SEARCH_ROOTS
        for path in (_ROOT / directory).rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    }


def _allowlisted(relative_path: str, term: str) -> bool:
    """The design's named refusal/coverage/history dispositions, not a path glob."""
    if relative_path == "src/model.cpp":
        return term in {"k_normed" + "_head", ".k_normed"}  # exact KLR1 refusal recognizer
    if relative_path == "tools/convert_model.py":
        return term in {"k_normed" + "_head", "k_norm" + ".requant", "softmax" + ".input"}
    if relative_path == "tools/reference_pipeline/pipeline.py":
        return term == "softmax" + ".input"  # live no-QK compatibility branch
    if relative_path == "tools/reference_pipeline/tests/composition_ref.py":
        return term == "softmax" + ".input"  # same no-QK oracle branch
    if relative_path == "docs/fused_k_numeric_spec.md":
        return True  # normative retirement disposition, not an executable reader
    if relative_path == _HISTORICAL_REFERENCE:
        return True  # frozen pre-repair historical reference
    if relative_path in _COVERAGE_TEXT:
        return True  # named coverage text; active-reader rule remains independent
    if relative_path == "tools/test_t2703_qkc_relation_loader.py":
        return True  # exact restored-key coverage
    if relative_path == "tools/test_t2703_fused_k_retirement_census.py":
        return True  # the census's dynamically assembled search vocabulary
    if relative_path == "tests/test_main.cpp":
        return term == "k_norm" + "_landing"  # named test-fixture backing only
    if relative_path == "src/gpu/superslm_gpu.cpp":
        return term == "k_norm" + "_landing"  # historical diagnostic comparison
    if relative_path == "include/superslm/checked_chain_funnel.h":
        return term == "k_norm" + "_landing"  # historical diagnostic comparison
    return False


def _unclassified_legacy_hits(sources, allowlisted=_allowlisted):
    hits = []
    for relative_path, text in sources.items():
        for line_number, line in enumerate(text.splitlines(), start=1):
            for term in _TERMS:
                if term in line and not allowlisted(relative_path, term):
                    hits.append(f"{relative_path}:{line_number}:{term}")
    return hits


def _active_reader_hits(sources):
    return [f"{relative_path}:{line_number}:{term}"
            for relative_path, text in sources.items()
            for line_number, line in enumerate(text.splitlines(), start=1)
            for term in _ACTIVE_READER_TERMS if term in line]


def test_retired_fused_k_source_census_is_exhaustive_and_classified():
    sources = _sources()
    assert set(_SEARCH_ROOTS).issubset({Path(path).parts[0] for path in sources})
    assert not _unclassified_legacy_hits(sources)
    assert not _active_reader_hits(sources)


def test_retirement_census_allowlist_control_is_live():
    sources = _sources()

    def without_refusal_anchor(relative_path, term):
        return (relative_path, term) != _REFUSAL_ANCHOR and _allowlisted(relative_path, term)

    assert any(hit.startswith("src/model.cpp:") and hit.endswith(":" + _REFUSAL_ANCHOR[1])
               for hit in _unclassified_legacy_hits(sources, without_refusal_anchor))


def test_retirement_census_rejects_a_restored_oracle_reader():
    sources = _sources()
    oracle = "tools/reference_pipeline/tests/composition_ref.py"
    sources[oracle] += "\n" + "def land" + "_" + "retired" + "_k(rows):\n    return rows\n"
    assert any(hit.startswith(f"{oracle}:") and hit.endswith(":" + _ACTIVE_READER_TERMS[0])
               for hit in _active_reader_hits(sources))
