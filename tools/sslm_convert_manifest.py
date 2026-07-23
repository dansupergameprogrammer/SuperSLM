"""sslm_convert_manifest.py — invoke the §13 item 7 independent C++ verifier
and merge its manifest with the calibration-side fields only Python can
supply (S-HARDEN-3, F13).

`tools/sslm_verify.cpp` (built as the `sslm_verify` CMake target) opens the
just-written `.sslm` through `SslmModel::Load` -- the same entry point every
runtime consumer uses -- and writes a JSON manifest built from what the
loader actually accepted: never from this module's own arrays. This module's
job is the glue: locate the compiled binary, run it, and merge its output
with what only the Python side ever had (the calibration input's source
hashes, and the converter's own git commit). `convert_model.py` calls
`verify_and_merge` once, after `sslm_format.write_artifact` returns.

Kept import-independent from convert_model.py's cross-tree spike import (see
sslm_convert_validate.py's module docstring for why), so it is testable on a
bare checkout against a fake verifier stand-in.
"""

import hashlib
import json
import os
import subprocess


class VerifierFailure(RuntimeError):
    """Raised when the independent verifier rejects the artifact, its own
    geometry cross-check fails, it exits nonzero for any other reason, or its
    reported process exit code is nonzero after a clean parse."""


def find_verifier_binary(repo_root, explicit_path=None):
    """Locates the compiled sslm_verify binary. `explicit_path`, if given, is
    trusted without a filesystem check (a caller-supplied override, e.g. from
    a CI job that just built it to a nonstandard path, is not this module's
    to second-guess); otherwise every CMake single-config and multi-config
    default output location is tried."""
    if explicit_path:
        return explicit_path
    candidates = [
        os.path.join(repo_root, "build", "Release", "sslm_verify.exe"),
        os.path.join(repo_root, "build", "Release", "sslm_verify"),
        os.path.join(repo_root, "build", "sslm_verify.exe"),
        os.path.join(repo_root, "build", "sslm_verify"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "sslm_verify binary not found; build it (cmake --build build --target sslm_verify) "
        "or pass an explicit path. Looked in: " + ", ".join(candidates)
    )


def run_verifier(verifier_cmd, artifact_path, manifest_path):
    """Runs `verifier_cmd + [artifact_path, manifest_path]` and parses the
    manifest it wrote. `verifier_cmd` is a list (a single binary path, or an
    interpreter + script for a test stand-in) so tests can substitute a fake
    verifier without needing the real compiled binary."""
    proc = subprocess.run(list(verifier_cmd) + [artifact_path, manifest_path],
                          capture_output=True, text=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return proc.returncode, manifest, proc.stdout, proc.stderr


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(path):
    """A deterministic aggregate hash of every file under `path` (sorted
    relative paths, POSIX-separated so the hash is platform-independent),
    plus the per-file hash table -- the manifest's `source_hashes` field.
    Empty (`"", {}`) if `path` does not exist or contains no files, so a
    caller can still merge a manifest for a synthetic in-memory model that
    has no real calibration directory (the pinned CI fixture)."""
    if not os.path.isdir(path):
        return "", {}
    entries = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            entries.append((rel, sha256_file(full)))
    entries.sort()
    agg = hashlib.sha256()
    for rel, h in entries:
        agg.update(rel.encode("utf-8"))
        agg.update(h.encode("utf-8"))
    return agg.hexdigest(), dict(entries)


def git_commit(repo_root):
    """The converter's own commit, or None if this is not a git checkout
    (never raises -- a manifest with `converter_commit: null` is honest about
    an unavailable fact; a manifest write that fails because git is absent
    from the build machine is not this function's call to make)."""
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def build_combined_manifest(verifier_manifest, *, source_dir_hash, source_files, converter_commit,
                            reference_commit=None):
    combined = dict(verifier_manifest)
    combined["source_hashes"] = {"aggregate": source_dir_hash, "files": source_files}
    combined["converter_commit"] = converter_commit
    # quantization-error percentiles and fold approximation error require the
    # calibration pipeline's FLOAT reference, which convert_model.py does not
    # currently retain from artifact_cache.load_artifact (only the already-
    # quantized model is returned) -- reserved, not fabricated. See this
    # slot's handoff for the design question this raises.
    combined["quantization_error_percentiles"] = None
    combined["fold_approximation_error"] = None
    combined["reference_commit"] = reference_commit
    return combined


def verify_and_merge(repo_root, artifact_out_path, calibration_dir, *, verifier_cmd=None,
                     manifest_out_path=None, reference_commit=None):
    """The end-to-end §13 item 7 step. Raises VerifierFailure if: the
    verifier's own status is REJECTED (container-level or schema-value gate);
    its config_geometry check reports ok == false (§17.3 cell 4); or its
    process exit code is nonzero for any other reason. On success, writes and
    returns the combined manifest (verifier fields + calibration-side
    fields) to `manifest_out_path` (default: `<artifact_out_path>.manifest.json`).
    """
    manifest_out_path = manifest_out_path or (artifact_out_path + ".manifest.json")
    cmd = verifier_cmd if verifier_cmd is not None else [find_verifier_binary(repo_root)]
    returncode, verifier_manifest, stdout, stderr = run_verifier(cmd, artifact_out_path, manifest_out_path)

    if verifier_manifest.get("status") == "REJECTED":
        raise VerifierFailure(
            f"independent verifier rejected the artifact: {verifier_manifest.get('reject_status')} -- "
            f"{verifier_manifest.get('diagnostic')}")

    geometry = verifier_manifest.get("config_geometry")
    if geometry is not None and not geometry.get("ok", False):
        raise VerifierFailure(
            f"independent geometry cross-check failed: {geometry.get('status')} -- "
            f"{geometry.get('diagnostic')}")

    if returncode != 0:
        raise VerifierFailure(f"independent verifier exited {returncode}: {stderr or stdout}")

    agg_hash, file_hashes = sha256_directory(calibration_dir)
    combined = build_combined_manifest(verifier_manifest, source_dir_hash=agg_hash, source_files=file_hashes,
                                       converter_commit=git_commit(repo_root), reference_commit=reference_commit)
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    return combined
