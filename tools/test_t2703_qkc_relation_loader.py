"""T-2703 F2 — QKC1 source-to-derived relation loader cells.

The source artifact is emitted by the production converter, then each cell
changes one otherwise legal persisted integer image.  `sslm_verify` is the
production loader: no test-side implementation of its relation pass stands in
for the refusal.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).with_name("reference_pipeline")))

import convert_model as converter  # noqa: E402
import sslm_format as artifact_format  # noqa: E402
import sslm_model_writer as writer  # noqa: E402
import pipeline  # noqa: E402


_ROOT = Path(__file__).resolve().parents[1]


def _verifier() -> Path | None:
    explicit = os.environ.get("SSLM_VERIFY")
    candidates = ([Path(explicit)] if explicit else []) + [
        _ROOT / "out" / "cmake-cpu-only-default" / "Release" / "sslm_verify.exe",
        _ROOT / "out" / "cmake-cpu-only-default" / "Release" / "sslm_verify",
        _ROOT / "build" / "Release" / "sslm_verify.exe",
        _ROOT / "build" / "Release" / "sslm_verify",
        _ROOT / "build" / "sslm_verify.exe",
        _ROOT / "build" / "sslm_verify",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


_VERIFIER = _verifier()
pytestmark = pytest.mark.skipif(_VERIFIER is None, reason="sslm_verify not built")


def _load_legal_qk_model():
    """A production-width, converter-emitted QKC1 artifact source."""
    cfg = pipeline.ModelConfig(
        hidden_size=128, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, head_dim=128, intermediate_size=256,
        vocab_size=32, rope_theta=10000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=True, context_cap=16)
    base = pipeline.fixture_model(cfg)
    weights = dict(base.weights)
    weight_scales = dict(base.weight_scales)
    composition_constants = dict(base.composition_constants)
    weights["layer0.q_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
    weights["layer0.k_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
    # `r_t` for the first 1.25 remains legal after +1.  The rest deliberately
    # exercise normal binary64, adjacent binary64, and small (but legal)
    # source scales through the exact relation pass; 1.75 is the head maximum.
    source_population = (
        1.25, 1.5, 1.75, math.nextafter(1.0, 2.0),
        math.nextafter(1.0, 0.0), 0.1, 2.0 ** -10, 2.0 ** -20,
    )
    channels = [source_population[index % len(source_population)]
                for index in range(cfg.head_dim)]
    weight_scales["layer0.q_norm.gain"] = channels
    weight_scales["layer0.k_norm.gain"] = channels
    composition_constants["layer0.q_norm"] = (1 << 30, -30)
    composition_constants["layer0.k_norm"] = (1 << 30, -30)
    return replace(base, weights=weights, weight_scales=weight_scales,
                   composition_constants=composition_constants)


def _parse_kvc1(payload: bytes) -> dict[str, tuple[int, int]]:
    magic, version, count, words, names_size, reserved = struct.unpack_from("<4sIIIII", payload)
    assert (magic, version, words, reserved) == (b"KVC1", 1, 2, 0)
    descriptor_offset = 24
    values_offset = descriptor_offset + 8 * count
    names_offset = values_offset + 8 * words * count
    assert names_offset + names_size == len(payload)
    names = [struct.unpack_from("<II", payload, descriptor_offset + 8 * index)
             for index in range(count)]
    return {
        payload[names_offset + offset:names_offset + offset + length].decode("utf-8"):
        struct.unpack_from("<qq", payload, values_offset + 16 * index)
        for index, (offset, length) in enumerate(names)
    }


def _artifact_sections(mutation: str | None):
    model = _load_legal_qk_model()
    qkc = converter.build_qk_channel_table(model)
    if mutation == "ratio":
        qkc["k_channel_ratio"][0] -= 1
    elif mutation == "landing":
        qkc["k_channel_r_t"][0] += 1

    sections, _ = converter.build_sections(model)
    replaced = []
    for section in sections:
        if section.type == artifact_format.SectionType.QK_CHANNEL_TABLE and mutation in {"ratio", "landing"}:
            replaced.append(artifact_format.Section(
                section.type, writer.write_tensor_manifest(writer.QKC1, np.int64, qkc)))
        elif section.type == artifact_format.SectionType.COMPOSITION_CONSTANTS and mutation == "head":
            constants = _parse_kvc1(section.data)
            m, e = constants["layer0.softmax_khead0"]
            constants["layer0.softmax_khead0"] = (m + 1, e)
            replaced.append(artifact_format.Section(section.type, writer.write_kvc1(2, constants)))
        else:
            replaced.append(section)
    return model, replaced


def _verify(tmp_path: Path, mutation: str | None):
    model, sections = _artifact_sections(mutation)
    artifact = tmp_path / f"qkc-{mutation or 'coherent'}.sslm"
    artifact_format.write_artifact(
        artifact, sections, flags=converter.artifact_flags_for_model(model))
    manifest = artifact.with_suffix(".manifest.json")
    result = subprocess.run([str(_VERIFIER), str(artifact), str(manifest)],
                            capture_output=True, text=True, check=False)
    return result


def test_coherent_qkc1_artifact_loads_before_relation_refusal_cells(tmp_path):
    result = _verify(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize(("mutation", "field"), [
    ("ratio", "k_channel_ratio"),
    ("landing", "k_channel_r_t"),
    ("head", 'CompositionConstants entry "layer0.softmax_khead0"'),
])
def test_qkc1_legal_domain_relation_mutations_refuse_before_marshal(tmp_path, mutation, field):
    result = _verify(tmp_path, mutation)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "REJECTED: QkChannelScaleRelationMismatch" in output
    assert field in output
