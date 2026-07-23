"""Curie's red suite for sslm_convert_manifest.py (S-HARDEN-3, F13).

Uses a small fake-verifier stand-in script (invoked as `sys.executable
fake_verifier.py <artifact> <manifest>`) instead of the real compiled
sslm_verify binary, so these tests run on a bare checkout with no C++ build
step -- the real binary's behavior is proven separately by the C++ suite
(test_main.cpp) and by tools/test_sslm_convert_loader_join.py's integration
cell, which DOES invoke the compiled binary as the independent oracle
S-HARDEN-3's cell 2 requires.
"""

import json
import sys
import textwrap

import pytest

import sslm_convert_manifest as M


def _write_fake_verifier(tmp_path, manifest_body, returncode=0):
    """Writes a fake verifier that ignores its artifact-path argument and
    writes `manifest_body` (a dict) to the given manifest-path argument,
    exiting with `returncode`."""
    script = tmp_path / "fake_verifier.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        manifest = {manifest_body!r}
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        sys.exit({returncode})
    """))
    return [sys.executable, str(script)]


def _accepting_manifest():
    return {
        "schema": "sslm_proof_manifest_v1",
        "artifact_hash": "deadbeef",
        "config_geometry": {"ok": True, "status": "Ok", "diagnostic": ""},
        "sections": [],
    }


def test_verify_and_merge_accepts_and_writes_combined_manifest(tmp_path):
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"fake artifact bytes")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    (calib_dir / "weights.bin").write_bytes(b"calibration data")

    cmd = _write_fake_verifier(tmp_path, _accepting_manifest())
    manifest_out = tmp_path / "model.sslm.manifest.json"
    combined = M.verify_and_merge(str(tmp_path), str(artifact), str(calib_dir), verifier_cmd=cmd,
                                  manifest_out_path=str(manifest_out))

    assert combined["artifact_hash"] == "deadbeef"
    assert combined["source_hashes"]["aggregate"] != ""
    assert "weights.bin" in combined["source_hashes"]["files"]
    assert combined["quantization_error_percentiles"] is None
    assert combined["fold_approximation_error"] is None
    on_disk = json.loads(manifest_out.read_text())
    assert on_disk == combined


def test_verify_and_merge_raises_on_rejected_status(tmp_path):
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    rejected = {"schema": "sslm_proof_manifest_v1", "status": "REJECTED",
                "reject_status": "ArtifactRejected", "diagnostic": "bad magic"}
    cmd = _write_fake_verifier(tmp_path, rejected, returncode=1)

    with pytest.raises(M.VerifierFailure) as exc:
        M.verify_and_merge(str(tmp_path), str(artifact), str(tmp_path), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"))
    assert "ArtifactRejected" in str(exc.value)


def test_verify_and_merge_raises_on_geometry_mismatch_even_with_zero_exit_code(tmp_path):
    # A verifier that (hypothetically) forgot to fold its own geometry check
    # into its process exit code must still be caught here -- this cell
    # proves the merge layer does not simply trust the subprocess return
    # code, closing exactly the correlated-oracle shape this whole family
    # exists to prevent (a manifest field nobody's control flow depends on).
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    mismatched = {
        "schema": "sslm_proof_manifest_v1",
        "artifact_hash": "abc123",
        "config_geometry": {"ok": False, "status": "HiddenSizeGeometryMismatch", "diagnostic": "4097 != 4096"},
        "sections": [],
    }
    cmd = _write_fake_verifier(tmp_path, mismatched, returncode=0)

    with pytest.raises(M.VerifierFailure) as exc:
        M.verify_and_merge(str(tmp_path), str(artifact), str(tmp_path), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"))
    assert "HiddenSizeGeometryMismatch" in str(exc.value)


def test_verify_and_merge_raises_on_nonzero_exit_with_otherwise_ok_manifest(tmp_path):
    artifact = tmp_path / "model.sslm"
    artifact.write_bytes(b"x")
    cmd = _write_fake_verifier(tmp_path, _accepting_manifest(), returncode=3)

    with pytest.raises(M.VerifierFailure):
        M.verify_and_merge(str(tmp_path), str(artifact), str(tmp_path), verifier_cmd=cmd,
                           manifest_out_path=str(tmp_path / "out.json"))


def test_sha256_directory_is_deterministic_and_platform_independent_paths(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.bin").write_bytes(b"aaa")
    (d / "b.bin").write_bytes(b"bbb")
    agg1, files1 = M.sha256_directory(str(d))
    agg2, files2 = M.sha256_directory(str(d))
    assert agg1 == agg2
    assert set(files1) == {"a.bin", "b.bin"}
    assert files1 == files2


def test_sha256_directory_empty_for_missing_path(tmp_path):
    agg, files = M.sha256_directory(str(tmp_path / "does_not_exist"))
    assert agg == ""
    assert files == {}


def test_find_verifier_binary_raises_with_no_candidate_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.find_verifier_binary(str(tmp_path))
