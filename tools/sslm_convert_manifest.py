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

import numpy as np


class VerifierFailure(RuntimeError):
    """Raised when the independent verifier rejects the artifact, its own
    geometry cross-check fails, it exits nonzero for any other reason, or its
    reported process exit code is nonzero after a clean parse. T-408 also
    raises this for an unresolvable float reference (an unreopenable
    checkpoint) or an empty quantization_error_percentiles population -- both
    §13 item 7's reject-over-degrade discipline applied to the one corner
    Dan's 2026-07-22 ruling turns from "acceptable to omit" into "acceptable
    only to fail loudly on" (design record
    Claude/Vitruvius/SuperSLM_T408_ProofManifestMetrics_Design-2026-07-23.md)."""


# T-408 §3.2's pinned percentile set (p50, p90, p95, p99, p99.9, max) and its pinned
# interpolation rule (linear between the two nearest ranks -- numpy's own
# method="linear" default, equivalently NIST's "linear interpolation between closest
# ranks"). Named explicitly rather than left to a library default: nearest-rank
# interpolation reports a different value at the same tie boundary, so the rule is part
# of the metric's contract (design §3.2), proven by
# tools/test_sslm_convert_manifest.py's own tie-boundary fixture.
_QUANTIZATION_ERROR_PERCENTILE_KEYS = ("p50", "p90", "p95", "p99", "p99.9")
_QUANTIZATION_ERROR_PERCENTILE_RANKS = (50.0, 90.0, 95.0, 99.0, 99.9)


def _channel_scale_row(scales, num_rows):
    """§6.1's W8 weight-scale convention as `quantization_error_percentiles` needs it:
    `weight_scales[name]` is either one scale for the whole tensor (a 1-element
    sequence -- the embedding and norm-gain convention) or one scale per output channel
    (length == the tensor's axis-0 size -- every §6.1 projection, `_PROJECTION_OUTPUT_AXIS
    = 0`). Returns the per-row array either way, so the caller can always reshape and
    broadcast the same way. Raises VerifierFailure (not IndexError/ValueError) on a
    length that matches neither shape -- a calibration or shape defect, not a rounding
    question, and this module's own reject-over-degrade discipline (§11) applies here
    exactly as it does to every other manifest-emission fact."""
    values = np.asarray(scales, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full(num_rows, values[0], dtype=np.float64)
    if values.size != num_rows:
        raise VerifierFailure(
            f"quantization_error_percentiles: weight_scales has {values.size} entries "
            f"for {num_rows} output channels -- a calibration or shape defect")
    return values


def _tensor_error_in_scale_units(name, model):
    """One tensor's per-element dequantization error, in units of its own channel scale
    (§3.2): `error_in_lsb = |float_weight - codes * channel_scale| / channel_scale`.
    Kept import-independent of the cross-tree spike's `dequantize_weight_per_channel`
    (this module's own module-docstring constraint) -- the same per-channel-scale
    broadcast, restated locally rather than imported."""
    codes = np.asarray(model.weights[name], dtype=np.float64)
    if codes.size == 0:
        return np.asarray([], dtype=np.float64)
    float_weight = np.asarray(model.float_weight(name), dtype=np.float64)
    if float_weight.shape != codes.shape:
        raise VerifierFailure(
            f"quantization_error_percentiles: {name!r} float reference shape "
            f"{float_weight.shape} != quantized codes shape {codes.shape}")
    scale_per_row = _channel_scale_row(model.weight_scales[name], codes.shape[0])
    broadcast_shape = [1] * codes.ndim
    broadcast_shape[0] = scale_per_row.shape[0]
    divisor = scale_per_row.reshape(broadcast_shape)
    dequantized = codes * divisor
    return (np.abs(float_weight - dequantized) / divisor).ravel()


def compute_quantization_error_percentiles(model):
    """T-408 §3.2: per-element weight-dequantization error, pooled across every
    quantized weight tensor in `model.weights`, reported as the pinned percentile set
    under the pinned linear-interpolation rule.

    Raises VerifierFailure, never a fabricated report, when:
    - `model.float_weight(name)` raises for any tensor (an unresolvable checkpoint,
      T-408 §4) -- re-raised naming the underlying reason, not left as the bare
      RuntimeError `artifact_cache`'s stub raises three calls down.
    - the pooled error population is empty (zero quantized weight tensors, or every
      tensor zero-element) -- a percentile of nothing is not a fact about the artifact,
      unlike `fold_approximation_error`'s zero-non-identity-folds case, which reports a
      genuine `0.0` (§3.2's own contrast with §3.1).
    """
    pooled = []
    for name in sorted(model.weights):
        try:
            pooled.append(_tensor_error_in_scale_units(name, model))
        except VerifierFailure:
            raise
        except RuntimeError as exc:
            raise VerifierFailure(
                f"quantization_error_percentiles: float reference unavailable for "
                f"{name!r}: {exc}") from exc
    population = np.concatenate(pooled) if pooled else np.asarray([], dtype=np.float64)
    if population.size == 0:
        raise VerifierFailure(
            "quantization_error_percentiles: pooled error population is empty (zero "
            "quantized weight tensors, or every tensor zero-element) -- never reported "
            "as a fabricated all-zero/nan percentile set")
    result = {
        key: float(np.percentile(population, rank, method="linear"))
        for key, rank in zip(_QUANTIZATION_ERROR_PERCENTILE_KEYS, _QUANTIZATION_ERROR_PERCENTILE_RANKS)
    }
    result["max"] = float(population.max())
    return result


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
                            reference_commit=None, model=None, fold_approximation_error=None):
    """`model` (T-408 §5): the `QuantizedModel` (or an object exposing the same
    `.weights`/`.weight_scales`/`.float_weight` contract) `quantization_error_percentiles`
    is computed from, here rather than precomputed by the caller -- this is the one place
    positioned to catch `model.float_weight`'s raise (an unresolvable checkpoint) and the
    empty-population case and translate both into VerifierFailure before they propagate
    as an unrelated RuntimeError. `None` when the caller has no model to offer (every
    pre-T-408 call site, and every rejection-path test that never reaches this function
    at all) -- both new fields stay `None`, unchanged from before this design.

    `fold_approximation_error` (T-408 §5), by contrast, IS precomputed by the caller
    (`convert_model.build_sections` returns it alongside `sections`) and threaded through
    as a plain float -- recomputing it here would mean re-running
    `_fold_ops_tensor`/`_ctx_fold_tensor` a second time over data the caller already
    holds.
    """
    combined = dict(verifier_manifest)
    combined["source_hashes"] = {"aggregate": source_dir_hash, "files": source_files}
    combined["converter_commit"] = converter_commit
    combined["fold_approximation_error"] = fold_approximation_error
    combined["quantization_error_percentiles"] = (
        compute_quantization_error_percentiles(model) if model is not None else None)
    combined["reference_commit"] = reference_commit
    return combined


def verify_and_merge(repo_root, artifact_out_path, calibration_dir, *, verifier_cmd=None,
                     manifest_out_path=None, reference_commit=None, model=None,
                     fold_approximation_error=None):
    """The end-to-end §13 item 7 step. Raises VerifierFailure if: the
    verifier's own status is REJECTED (container-level or schema-value gate);
    its config_geometry check reports ok == false (§17.3 cell 4); its
    process exit code is nonzero for any other reason; the T-408 float
    reference behind `quantization_error_percentiles` is unresolvable; or
    that metric's pooled error population is empty. On success, writes and
    returns the combined manifest (verifier fields + calibration-side
    fields) to `manifest_out_path` (default: `<artifact_out_path>.manifest.json`).

    `model`/`fold_approximation_error` (T-408 §5): passed straight through to
    `build_combined_manifest`; see its own docstring for the asymmetry between the two
    new fields' computation.
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
                                       converter_commit=git_commit(repo_root), reference_commit=reference_commit,
                                       model=model, fold_approximation_error=fold_approximation_error)
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    return combined
