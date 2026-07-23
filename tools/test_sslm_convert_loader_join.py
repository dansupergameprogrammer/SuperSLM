"""S-HARDEN-3, §17.3 cell 2: "a clean-checkout integration cell that parses the
emitted section-type set BEFORE invoking the C++ loader, so a shared omission
cannot pass twice." Specification filed by Mendeleev's 2026-07-21 coverage
audit §3.2; authored here.

Requires the compiled `sslm_verify` binary (CMake target `sslm_verify`) --
skipped, not failed, if it is not present, since building it is a separate CI
step from running the Python suite (see this slot's build log for the CI
wiring). Run locally with:
    cmake --build build --config Release --target sslm_verify
    python -m pytest tools/test_sslm_convert_loader_join.py

Two independent parsers own every artifact byte in this test, and NEITHER
reads the other's code path:
  1. `_independent_parse_section_types` below -- transcribed directly from
     docs/sslm_format.md's own byte-layout table (cell 1's own oracle), not
     from sslm_format.py (the writer under test) and not from any C++ header.
  2. The compiled `sslm_verify` binary -- SslmArtifact::OpenFromFile +
     SslmModel::Load, the production C++ loader.
A Python REIMPLEMENTATION of "which sections are required," never a
subprocess call into the loader's OWN logic wrapped a second time -- calling
sslm_verify a second time as the "independent" side would silently become a
second instance of the correlated-oracle failure this whole addendum exists
to close (§3.2's own text).
"""

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import sslm_format as F  # noqa: E402
import sslm_model_writer as W  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Required sections per format_version -- transcribed from docs/sslm_format.md
# section-types table ("required" / "required from v2"), independent of both
# the Python writer and the C++ loader's own MissingSection logic.
_REQUIRED_SECTIONS_BY_VERSION = {
    1: {0},       # Config
    2: {0, 12},   # Config, SigmoidLut
}


def _find_verifier():
    for rel in ("build/Release/sslm_verify.exe", "build/Release/sslm_verify",
                "build/sslm_verify.exe", "build/sslm_verify"):
        p = os.path.join(_REPO_ROOT, rel)
        if os.path.isfile(p):
            return p
    return None


_VERIFIER = _find_verifier()
pytestmark = pytest.mark.skipif(_VERIFIER is None, reason="sslm_verify not built -- see module docstring")


def _independent_parse_header_and_sections(data):
    """Transcribed directly from docs/sslm_format.md "Byte layout" -- a
    64-byte header then `section_count` x 40-byte descriptors. Returns
    (format_version, {section_type, ...}) or raises AssertionError on any
    structural deviation (this test's own artifacts are always well-formed;
    a raise here means the writer under test regressed the byte layout
    itself, not a hostile-input case -- that population is the C++ suite's).
    """
    assert len(data) >= 64, "file shorter than the 64-byte header"
    magic = data[0:4]
    assert magic == b"SSLM", f"bad magic {magic!r}"
    format_version, header_bytes, section_count, flags, reserved0 = struct.unpack_from("<IIIII", data, 4)
    assert header_bytes == 64
    assert flags == 0
    assert reserved0 == 0
    (file_bytes,) = struct.unpack_from("<Q", data, 24)
    assert file_bytes == len(data)

    types = set()
    for i in range(section_count):
        row = 64 + i * 40
        (sec_type,) = struct.unpack_from("<I", data, row + 0)
        types.add(sec_type)
    return format_version, types


def _independent_required_sections_ok(format_version, types):
    required = _REQUIRED_SECTIONS_BY_VERSION.get(format_version)
    assert required is not None, f"no independent schema entry for format_version {format_version}"
    return required.issubset(types)


# --- a tiny valid synthetic v2 artifact (Config + SigmoidLut only -- the
#     minimum the required-section schema demands), built via the real
#     Python emission path (sslm_format.write_artifact / sslm_model_writer),
#     never via a hand-rolled byte buffer. ---

def _build_valid_v2_artifact_bytes():
    cfg_bytes = W.write_cfg1(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, intermediate_size=8, vocab_size=4, context_cap=4, tie_word_embeddings=False,
        kv_precision=W.KV_PRECISION_INT8, kv_block_size=16, unicode_major=15, unicode_minor=1,
        unicode_patch=0, rope_theta=10000.0, rms_norm_eps=1e-6)
    sections = [
        F.Section(F.SectionType.CONFIG, cfg_bytes),
        F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()),
    ]
    data, fingerprint = F.build_artifact(sections)
    return data, fingerprint


def _build_invalid_v2_artifact_missing_sigmoid_lut_bytes():
    """The exact correlated-oracle shape S-HARDEN-1 (F1) closed: a v2
    artifact carrying only Config, missing the required SigmoidLut."""
    cfg_bytes = W.write_cfg1(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, intermediate_size=8, vocab_size=4, context_cap=4, tie_word_embeddings=False,
        kv_precision=W.KV_PRECISION_INT8, kv_block_size=16, unicode_major=15, unicode_minor=1,
        unicode_patch=0, rope_theta=10000.0, rms_norm_eps=1e-6)
    sections = [F.Section(F.SectionType.CONFIG, cfg_bytes)]
    data, fingerprint = F.build_artifact(sections)
    return data, fingerprint


def _run_verifier(tmp_path, data, name="artifact.sslm"):
    artifact_path = tmp_path / name
    artifact_path.write_bytes(data)
    manifest_path = tmp_path / (name + ".manifest.json")
    proc = subprocess.run([_VERIFIER, str(artifact_path), str(manifest_path)],
                          capture_output=True, text=True)
    manifest = json.loads(manifest_path.read_text())
    return proc.returncode, manifest


def test_valid_artifact_both_independent_oracles_agree_it_is_well_formed(tmp_path):
    data, _fp = _build_valid_v2_artifact_bytes()

    # (b) independent Python-side schema check, before any C++ code sees the bytes.
    format_version, types = _independent_parse_header_and_sections(data)
    assert format_version == 2
    python_verdict_ok = _independent_required_sections_ok(format_version, types)
    assert python_verdict_ok, f"Python schema check: types {types} missing required {_REQUIRED_SECTIONS_BY_VERSION[2]}"

    # (c) the same bytes through the real C++ loader.
    returncode, manifest = _run_verifier(tmp_path, data)
    cpp_verdict_ok = (manifest.get("status") != "REJECTED") and returncode == 0

    # (d) both verdicts agree.
    assert cpp_verdict_ok, f"C++ verifier rejected a Python-schema-valid artifact: {manifest}"
    assert python_verdict_ok == cpp_verdict_ok


def test_artifact_missing_required_sigmoid_lut_both_independent_oracles_reject_it(tmp_path):
    # The correlated-oracle shape this whole addendum exists to close: BOTH
    # sides must independently catch the omission, so neither can bless the
    # other's identical blind spot (§17.3's own preamble: "a converter and a
    # loader that both omit the same required section so each blesses the
    # other").
    data, _fp = _build_invalid_v2_artifact_missing_sigmoid_lut_bytes()

    format_version, types = _independent_parse_header_and_sections(data)
    python_verdict_ok = _independent_required_sections_ok(format_version, types)
    assert not python_verdict_ok, "Python schema check must flag the missing SigmoidLut section"

    returncode, manifest = _run_verifier(tmp_path, data)
    cpp_verdict_ok = (manifest.get("status") != "REJECTED") and returncode == 0
    assert not cpp_verdict_ok, f"C++ verifier must also reject the missing-SigmoidLut artifact: {manifest}"
    assert manifest.get("status") == "REJECTED"

    assert python_verdict_ok == cpp_verdict_ok  # both False -- neither blesses the other


def test_verifier_manifest_section_type_set_matches_independent_python_parse(tmp_path):
    # A second, stronger form of agreement: not just "both accept/reject" but
    # the C++ manifest's OWN reported section list matches the independently
    # parsed type set exactly (name-mapped, since the manifest reports names).
    data, _fp = _build_valid_v2_artifact_bytes()
    _format_version, types = _independent_parse_header_and_sections(data)

    _returncode, manifest = _run_verifier(tmp_path, data)
    reported_names = {s["type"] for s in manifest["sections"]}
    # 0 -> "Config", 12 -> "SigmoidLut" per docs/sslm_format.md's own table --
    # transcribed here too, independent of proof_manifest.cpp's SectionTypeName.
    expected_names = {{0: "Config", 12: "SigmoidLut"}[t] for t in types}
    assert reported_names == expected_names
