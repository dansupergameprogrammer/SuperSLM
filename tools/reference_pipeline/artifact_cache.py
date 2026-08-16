"""Calibrated-artifact serialization: round-trip bit-identity + the staleness law.

Pins in Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §17.3:
- Directory structure: artifacts/ containing arrays.npz + manifest.json
- Manifest carries format_version (int) and source_fingerprint (64-hex)
- Rejection on version or fingerprint mismatch (never silent load)
- ARTIFACT_FORMAT_VERSION pinned as 1

Round-trip contract: every field of QuantizedModel serializes to bit-identical on load.
Forward and forward_dynamic produce identical outputs with loaded model.
"""

import dataclasses
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

ARTIFACT_FORMAT_VERSION = 1


def _serialize_config(config) -> dict:
    """Convert ModelConfig dataclass to JSON-serializable dict."""
    return dataclasses.asdict(config)


def _deserialize_config(cfg_dict: dict, config_class):
    """Reconstruct ModelConfig from dict."""
    return config_class(**cfg_dict)


def _serialize_scales(scales) -> dict:
    """Convert StaticScales to JSON-serializable dict.

    StaticScales has:
    - requant: tuple of RequantSite objects
    - rescale: tuple of (name, mantissa, shift) tuples
    - nonlinear: tuple of (name, scale_float) tuples
    """
    return {
        "requant": [
            {
                "name": site.name,
                "input_scale": float(site.input_scale),
                "output_scale": float(site.output_scale),
                "weight_scales": list(site.weight_scales),
                "multipliers": list(site.multipliers),
                "shifts": list(site.shifts),
            }
            for site in scales.requant
        ],
        "rescale": [
            {"name": name, "mantissa": int(m), "shift": int(s)}
            for name, m, s in scales.rescale
        ],
        "nonlinear": [
            {"name": name, "scale": float(scale)}
            for name, scale in scales.nonlinear
        ],
    }


def _deserialize_scales(scales_dict: dict, scales_class, requant_site_class):
    """Reconstruct StaticScales from dict."""
    requant_sites = tuple(
        requant_site_class(
            name=site["name"],
            input_scale=site["input_scale"],
            output_scale=site["output_scale"],
            weight_scales=tuple(site["weight_scales"]),
            multipliers=tuple(site["multipliers"]),
            shifts=tuple(site["shifts"]),
        )
        for site in scales_dict["requant"]
    )
    rescale_tuples = tuple(
        (item["name"], item["mantissa"], item["shift"])
        for item in scales_dict["rescale"]
    )
    nonlinear_tuples = tuple(
        (item["name"], item["scale"])
        for item in scales_dict["nonlinear"]
    )
    return scales_class(
        requant=requant_sites,
        rescale=rescale_tuples,
        nonlinear=nonlinear_tuples,
    )


def _serialize_calibration(calibration) -> dict:
    """Convert CalibrationRecord dataclass to JSON-serializable dict."""
    return dataclasses.asdict(calibration)


def _deserialize_calibration(calib_dict: dict, calib_class):
    """Reconstruct CalibrationRecord from dict."""
    return calib_class(**calib_dict)


def _serialize_dicts(data: dict) -> dict:
    """Convert numpy arrays in dicts to lists for JSON serialization."""
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (int, float, str, bool, type(None))):
            result[key] = value
        elif isinstance(value, (tuple, list)):
            # Handle nested tuples/lists with numpy arrays
            result[key] = [
                v.tolist() if isinstance(v, np.ndarray) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def _deserialize_dicts(data: dict) -> dict:
    """Convert lists back to numpy arrays in dicts."""
    result = {}
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            # Try to convert back to numpy array
            try:
                result[key] = np.array(value)
            except (ValueError, TypeError):
                result[key] = value
        else:
            result[key] = value
    return result


def _compute_fingerprint(model) -> str:
    """Compute deterministic SHA-256 fingerprint of model's serializable content.

    Hashes all fields except float_source and tokenize_prompt.
    Same model → same fingerprint (deterministic)
    Different config → different fingerprint (decisive)
    """
    hasher = hashlib.sha256()

    # Hash config
    config_dict = _serialize_config(model.config)
    hasher.update(json.dumps(config_dict, sort_keys=True).encode("utf-8"))

    # Hash weights (sorted by key for determinism)
    for key in sorted(model.weights.keys()):
        arr = model.weights[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(arr.astype(np.int8).tobytes())

    # Hash weight_scales (sorted)
    for key in sorted(model.weight_scales.keys()):
        val = model.weight_scales[key]
        hasher.update(key.encode("utf-8"))
        if isinstance(val, tuple):
            hasher.update(json.dumps(val).encode("utf-8"))
        elif isinstance(val, np.ndarray):
            hasher.update(val.tobytes())
        else:
            hasher.update(str(val).encode("utf-8"))

    # Hash residual_scales (sorted)
    for key in sorted(model.residual_scales.keys()):
        val = model.residual_scales[key]
        hasher.update(key.encode("utf-8"))
        if isinstance(val, np.ndarray):
            hasher.update(val.tobytes())
        else:
            hasher.update(str(val).encode("utf-8"))

    # Hash rope_tables
    for table in model.rope_tables:
        hasher.update(table.tobytes())

    # Hash biases (sorted) — in the artifact format's CANONICAL dtype (int32, the dtype
    # save_artifact writes), so the fingerprint is representation-independent: a model
    # holding int64 bias codes in memory and its int32 reload hash identically (B-2).
    for key in sorted(model.biases.keys()):
        arr = model.biases[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(arr.astype(np.int32).tobytes())

    # Hash scales
    scales_dict = _serialize_scales(model.scales)
    hasher.update(json.dumps(scales_dict, sort_keys=True).encode("utf-8"))

    # Hash calibration
    calib_dict = _serialize_calibration(model.calibration)
    hasher.update(json.dumps(calib_dict, sort_keys=True).encode("utf-8"))

    # Hash composition_constants (sorted)
    for key in sorted(model.composition_constants.keys()):
        val = model.composition_constants[key]
        hasher.update(key.encode("utf-8"))
        if isinstance(val, tuple) and len(val) == 2:
            # (mantissa, shift) tuple
            hasher.update(str(val).encode("utf-8"))
        else:
            hasher.update(str(val).encode("utf-8"))

    # Hash kv_landing_scales (sorted)
    for key in sorted(model.kv_landing_scales.keys()):
        val = model.kv_landing_scales[key]
        hasher.update(key.encode("utf-8"))
        if isinstance(val, np.ndarray):
            hasher.update(val.tobytes())
        else:
            hasher.update(str(val).encode("utf-8"))

    # Hash kv_landing_reciprocals (sorted)
    for key in sorted(model.kv_landing_reciprocals.keys()):
        val = model.kv_landing_reciprocals[key]
        hasher.update(key.encode("utf-8"))
        if isinstance(val, np.ndarray):
            hasher.update(val.tobytes())
        else:
            hasher.update(str(val).encode("utf-8"))

    # Hash kv_calibration_provenance (sorted) -- T-1822 §31.4.1's calibration-policy
    # marker. Included so two builds differing only in calibration policy (e.g. Arm D
    # vs Arm E) fingerprint differently, matching the field's own stated purpose
    # ("distinguishable... from each other", pipeline.py's own QuantizedModel docstring).
    for key in sorted(model.kv_calibration_provenance.keys()):
        val = model.kv_calibration_provenance[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(json.dumps(val, sort_keys=True).encode("utf-8"))

    return hasher.hexdigest()


def save_artifact(model, artifact_path, checkpoint_path=None):
    """Save a QuantizedModel to a directory, with manifest and arrays.npz.

    Creates artifact_path/ containing:
    - manifest.json: format_version and source_fingerprint
    - arrays.npz: numpy arrays (weights, biases, rope_tables, etc.)
    - metadata.json: config, scales, calibration, composition_constants, kv_landing_*,
      checkpoint_path

    The model round-trips to bit-identical: forward() and forward_dynamic()
    produce identical outputs, and scales/dynamic_biases match exactly.

    `checkpoint_path` (T-408 §4): the on-disk directory `pipeline.load_model` built this
    model from, when known -- the caller already holds it in scope at every real call
    site (`load_model(CHECKPOINT)` immediately preceding `save_artifact(model, ...)`).
    Persisted verbatim into `metadata.json` so `load_artifact` can later reopen a lazy
    `float_source` over the same checkpoint without re-running calibration. `None` for a
    `fixture_model()`-built model, which never had one -- absence is the truthful value,
    not a placeholder.
    """
    artifact_path = Path(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Compute fingerprint
    fingerprint = _compute_fingerprint(model)

    # Save manifest
    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "source_fingerprint": fingerprint,
    }
    manifest_path = artifact_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Prepare numpy arrays for npz
    arrays_to_save = {}

    # Save weights
    for key, arr in model.weights.items():
        arrays_to_save[f"weight_{key}"] = arr.astype(np.int8)

    # Save weight_scales (as tuple lists for JSON-like storage)
    weight_scales_dict = {}
    for key, val in model.weight_scales.items():
        if isinstance(val, tuple):
            weight_scales_dict[key] = list(val)
        elif isinstance(val, np.ndarray):
            weight_scales_dict[key] = val.tolist()
        else:
            weight_scales_dict[key] = val

    # Save biases
    for key, arr in model.biases.items():
        arrays_to_save[f"bias_{key}"] = arr.astype(np.int32)

    # Save rope_tables
    for i, table in enumerate(model.rope_tables):
        arrays_to_save[f"rope_table_{i}"] = table.astype(np.int64)

    # Save residual_scales
    for key, val in model.residual_scales.items():
        if isinstance(val, np.ndarray):
            arrays_to_save[f"residual_scale_{key}"] = val
        else:
            # Store as metadata instead
            pass

    # Save arrays.npz
    arrays_path = artifact_path / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays_to_save)

    # Prepare composition_constants for JSON (convert tuples to lists)
    composition_constants_json = {}
    for key, val in model.composition_constants.items():
        if isinstance(val, tuple):
            composition_constants_json[key] = list(val)
        else:
            composition_constants_json[key] = val

    # Prepare kv_landing_scales for JSON
    kv_landing_scales_json = {}
    for key, val in model.kv_landing_scales.items():
        if isinstance(val, np.ndarray):
            kv_landing_scales_json[key] = val.tolist()
        elif isinstance(val, tuple):
            kv_landing_scales_json[key] = list(val)
        else:
            kv_landing_scales_json[key] = val

    # Prepare kv_landing_reciprocals for JSON
    kv_landing_reciprocals_json = {}
    for key, val in model.kv_landing_reciprocals.items():
        if isinstance(val, np.ndarray):
            kv_landing_reciprocals_json[key] = val.tolist()
        elif isinstance(val, tuple):
            kv_landing_reciprocals_json[key] = list(val)
        else:
            kv_landing_reciprocals_json[key] = val

    # Prepare kv_calibration_provenance for JSON -- plain str-keyed nested dicts
    # (T-1822 §31.4.1's own marker shape), already JSON-native, no tuple/array
    # conversion needed.
    kv_calibration_provenance_json = dict(model.kv_calibration_provenance)

    # Prepare metadata
    metadata = {
        "config": _serialize_config(model.config),
        "scales": _serialize_scales(model.scales),
        "weight_scales": weight_scales_dict,
        "calibration": _serialize_calibration(model.calibration),
        "composition_constants": composition_constants_json,
        "kv_landing_scales": kv_landing_scales_json,
        "kv_landing_reciprocals": kv_landing_reciprocals_json,
        "kv_calibration_provenance": kv_calibration_provenance_json,
        "residual_scales": {
            key: (val.tolist() if isinstance(val, np.ndarray) else val)
            for key, val in model.residual_scales.items()
        },
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }

    # Add dynamic_biases if present
    if hasattr(model, 'dynamic_biases') and model.dynamic_biases is not None:
        dynamic_biases_json = {}
        dynamic_biases_to_save = {}
        for key, val in model.dynamic_biases.items():
            if isinstance(val, np.ndarray):
                dynamic_biases_to_save[f"dynamic_bias_{key}"] = val.astype(np.int32)
            elif isinstance(val, tuple):
                # Store tuples in metadata as lists
                dynamic_biases_json[key] = list(val)
            else:
                # Scalar or other type
                dynamic_biases_json[key] = val
        # Merge into arrays
        arrays_to_save.update(dynamic_biases_to_save)
        # Re-save npz with dynamic_biases included
        np.savez_compressed(arrays_path, **arrays_to_save)
        metadata["dynamic_biases"] = dynamic_biases_json
        metadata["has_dynamic_biases"] = True
    else:
        metadata["has_dynamic_biases"] = False

    # Save metadata
    metadata_path = artifact_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def _unavailable_float_source(reason):
    """A `float_weight` that raises `RuntimeError` naming `reason` -- installed when no
    live checkpoint reader can be reconstructed. Distinct reasons for the two forks §7
    names (never recorded; recorded but unresolvable) so the message a caller catches and
    re-raises as `VerifierFailure` (SuperSLM's manifest-merge boundary) names what
    actually happened, not a generic string."""
    def float_source_stub(name):
        raise RuntimeError(
            f"float_source is not available for loaded artifact; "
            f"cannot access {name!r} from the original float checkpoint ({reason})"
        )
    return float_source_stub


def _reopen_checkpoint_float_source(checkpoint_path, config):
    """Reconstruct the SAME lazy `_CheckpointFloatSource` `pipeline.load_model` built the
    first time, without repeating calibration, prompt tokenization, or scale derivation --
    only the lazy float-tensor reader is rebuilt (T-408 §4). Imports `pipeline` lazily,
    matching this module's existing `load_artifact` convention (avoids importing the
    calibration/forward machinery at module load for every artifact_cache caller).
    """
    from reference_pipeline import pipeline

    checkpoint = Path(checkpoint_path)
    names = pipeline._upstream_names(config)
    tensors = pipeline._open_checkpoint_tensors(checkpoint)
    return pipeline._CheckpointFloatSource(tensors, names, config)


def _reopen_or_stub_float_source(checkpoint_path, config):
    """T-408 §4's fork: a present, resolvable `checkpoint_path` reopens a live reader;
    absence (never recorded) or an unresolvable path (moved/deleted since save) installs
    the raising stub instead -- never a silent `None`, never a crash on `load_artifact`
    itself. `float_weight` is not called eagerly here, so a filesystem race between this
    check and the tensor read is only ever surfaced as the same RuntimeError a resolvable
    path's own missing-tensor read would raise -- there is no separate pre-check to race.
    """
    if checkpoint_path is None:
        return _unavailable_float_source("no checkpoint_path recorded for this artifact")
    try:
        return _reopen_checkpoint_float_source(checkpoint_path, config)
    except (OSError, ValueError) as exc:
        return _unavailable_float_source(
            f"checkpoint_path {checkpoint_path!r} could not be reopened "
            f"(moved, deleted, or no longer a valid checkpoint): {exc}"
        )


def load_artifact(artifact_path, expect_fingerprint=None):
    """Load a QuantizedModel from an artifact directory.

    Raises ValueError if:
    - format_version does not match ARTIFACT_FORMAT_VERSION
    - expect_fingerprint is provided and does not match the loaded fingerprint

    Args:
        artifact_path: Path to artifact directory
        expect_fingerprint: Optional expected fingerprint (64-hex string)

    Returns:
        QuantizedModel with all fields reconstructed from disk
    """
    artifact_path = Path(artifact_path)

    # Load manifest
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Check format version
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Artifact format version mismatch: expected {ARTIFACT_FORMAT_VERSION}, "
            f"got {manifest.get('format_version')}. This artifact may be stale or incompatible."
        )

    # Check fingerprint if provided
    saved_fingerprint = manifest.get("source_fingerprint")
    if expect_fingerprint is not None and saved_fingerprint != expect_fingerprint:
        raise ValueError(
            f"Artifact fingerprint mismatch: expected {expect_fingerprint}, "
            f"got {saved_fingerprint}. The model source or configuration has changed."
        )

    # Load metadata
    metadata_path = artifact_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    # Load arrays
    arrays_path = artifact_path / "arrays.npz"
    arrays = np.load(arrays_path, allow_pickle=False)

    # Import required classes from pipeline
    from reference_pipeline import pipeline

    # Reconstruct config
    config = _deserialize_config(metadata["config"], pipeline.ModelConfig)

    # Reconstruct scales
    scales = _deserialize_scales(
        metadata["scales"],
        pipeline.StaticScales,
        pipeline.RequantSite,
    )

    # Reconstruct weights
    weights = {}
    for key in arrays.files:
        if key.startswith("weight_"):
            name = key[7:]  # Remove "weight_" prefix
            weights[name] = arrays[key]

    # Reconstruct weight_scales from metadata
    weight_scales = {}
    weight_scales_meta = metadata.get("weight_scales", {})
    for key, val in weight_scales_meta.items():
        if isinstance(val, list):
            weight_scales[key] = tuple(val)
        else:
            weight_scales[key] = val

    # Reconstruct biases
    biases = {}
    for key in arrays.files:
        if key.startswith("bias_"):
            name = key[5:]  # Remove "bias_" prefix
            biases[name] = arrays[key]

    # Reconstruct rope_tables
    rope_tables = []
    for key in sorted(arrays.files):
        if key.startswith("rope_table_"):
            rope_tables.append(arrays[key])
    rope_tables = tuple(rope_tables)

    # Reconstruct residual_scales
    residual_scales = {}
    for key, val in metadata["residual_scales"].items():
        if isinstance(val, list):
            residual_scales[key] = np.array(val)
        else:
            residual_scales[key] = val
    # Also check arrays.npz for residual_scale entries
    for key in arrays.files:
        if key.startswith("residual_scale_"):
            name = key[15:]  # Remove "residual_scale_" prefix
            residual_scales[name] = arrays[key]

    # Reconstruct calibration
    calibration = _deserialize_calibration(
        metadata["calibration"],
        pipeline.CalibrationRecord,
    )

    # Reconstruct composition_constants (tuples of (mantissa, shift))
    composition_constants = {}
    composition_constants_meta = metadata.get("composition_constants", {})
    for key, val in composition_constants_meta.items():
        if isinstance(val, list):
            composition_constants[key] = tuple(val)
        else:
            composition_constants[key] = val

    # Reconstruct kv_landing_scales (these are tuples)
    kv_landing_scales = {}
    kv_landing_scales_meta = metadata.get("kv_landing_scales", {})
    for key, val in kv_landing_scales_meta.items():
        if isinstance(val, list):
            kv_landing_scales[key] = tuple(val)
        else:
            kv_landing_scales[key] = val

    # Reconstruct kv_landing_reciprocals (these are tuples of ints)
    kv_landing_reciprocals = {}
    kv_landing_reciprocals_meta = metadata.get("kv_landing_reciprocals", {})
    for key, val in kv_landing_reciprocals_meta.items():
        if isinstance(val, list):
            kv_landing_reciprocals[key] = tuple(val)
        else:
            kv_landing_reciprocals[key] = val

    # Reconstruct kv_calibration_provenance -- plain nested dict, no tuple/array
    # conversion needed (T-1822 §31.4.1's own marker shape). `.get(..., {})` rather than
    # `[...]`: an artifact saved before this field existed carries no key at all, and
    # that is the truthful "no per-head calibration policy recorded" state, not an error.
    kv_calibration_provenance = metadata.get("kv_calibration_provenance", {})

    # T-408 §4: reopen a live, lazy float_source over the persisted checkpoint path when
    # one is present and still resolvable on disk; fall back to a raising stub otherwise.
    # `metadata.get("checkpoint_path")` -- never `metadata["checkpoint_path"]` -- so a
    # `metadata.json` with the key entirely absent (an old-schema cache directory
    # predating this change) and one with the key explicitly `null` (a fixture-built
    # model that never had a checkpoint) take the IDENTICAL branch below: one code path,
    # no silent second behavior for provenance nobody can distinguish anyway.
    checkpoint_path = metadata.get("checkpoint_path")
    float_weight = _reopen_or_stub_float_source(checkpoint_path, config)

    def tokenize_prompt_stub(record):
        raise RuntimeError(
            "tokenize_prompt is not available for loaded artifact"
        )

    # Reconstruct dynamic_biases if present
    dynamic_biases = None
    if metadata.get("has_dynamic_biases"):
        dynamic_biases = {}
        # First add numpy arrays from npz
        for key in arrays.files:
            if key.startswith("dynamic_bias_"):
                name = key[13:]  # Remove "dynamic_bias_" prefix
                dynamic_biases[name] = arrays[key]
        # Then add tuples from metadata
        dynamic_biases_meta = metadata.get("dynamic_biases", {})
        for key, val in dynamic_biases_meta.items():
            if isinstance(val, list):
                # Convert lists to tuples, and nested lists to tuples
                def convert_to_tuple(v):
                    if isinstance(v, list):
                        return tuple(convert_to_tuple(item) for item in v)
                    return v
                dynamic_biases[key] = convert_to_tuple(val)
            else:
                dynamic_biases[key] = val

    # Reconstruct QuantizedModel
    model = pipeline.QuantizedModel(
        config=config,
        scales=scales,
        weights=weights,
        weight_scales=weight_scales,
        residual_scales=residual_scales,
        rope_tables=rope_tables,
        biases=biases,
        float_source=float_weight,
        tokenize_prompt=tokenize_prompt_stub,
        calibration=calibration,
        gemm_weights={},
        composition_constants=composition_constants,
        kv_landing_scales=kv_landing_scales,
        kv_landing_reciprocals=kv_landing_reciprocals,
        dynamic_biases=dynamic_biases,
        kv_calibration_provenance=kv_calibration_provenance,
    )

    return model
