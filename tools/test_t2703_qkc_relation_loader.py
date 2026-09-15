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
from fractions import Fraction
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


def _load_legal_qk_model(channel_scales=None, *, head_dim=128):
    """A production-width, converter-emitted QKC1 artifact source."""
    cfg = pipeline.ModelConfig(
        hidden_size=head_dim, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, head_dim=head_dim, intermediate_size=2 * head_dim,
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
    channels = (list(channel_scales) if channel_scales is not None else
                [source_population[index % len(source_population)]
                 for index in range(cfg.head_dim)])
    weight_scales["layer0.q_norm.gain"] = channels
    weight_scales["layer0.k_norm.gain"] = channels
    composition_constants["layer0.q_norm"] = (1 << 30, -30)
    composition_constants["layer0.k_norm"] = (1 << 30, -30)
    return replace(base, weights=weights, weight_scales=weight_scales,
                   composition_constants=composition_constants)


def _parse_kvc1(payload: bytes) -> dict[str, tuple[int, ...]]:
    magic, version, count, words, names_size, reserved = struct.unpack_from("<4sIIIII", payload)
    assert (magic, version, reserved) == (b"KVC1", 1, 0)
    descriptor_offset = 24
    values_offset = descriptor_offset + 8 * count
    names_offset = values_offset + 8 * words * count
    assert names_offset + names_size == len(payload)
    names = [struct.unpack_from("<II", payload, descriptor_offset + 8 * index)
             for index in range(count)]
    return {
        payload[names_offset + offset:names_offset + offset + length].decode("utf-8"):
        struct.unpack_from("<" + "q" * words, payload, values_offset + 8 * words * index)
        for index, (offset, length) in enumerate(names)
    }


def _artifact_sections(mutation: str | None, *, head_dim=128):
    model = _load_legal_qk_model(
        [2.0 ** -30] * head_dim if mutation == "magnitude" else None,
        head_dim=head_dim)
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
        elif section.type == artifact_format.SectionType.COMPOSITION_CONSTANTS and mutation in {"head", "magnitude"}:
            constants = _parse_kvc1(section.data)
            if mutation == "head":
                m, e = constants["layer0.softmax_khead0"]
                constants["layer0.softmax_khead0"] = (m + 1, e)
            else:
                # F3's load-legal witness: exponent domains currently do not
                # permit the pinned [e=-60, e=7] boundary pair. The retained
                # canonical mantissa therefore loads, but the real
                # per-channel landing exceeds int64 and must refuse in CPU/GPU.
                m, _e = constants["layer0.k_norm"]
                constants["layer0.k_norm"] = (m, 7)
            replaced.append(artifact_format.Section(section.type, writer.write_kvc1(2, constants)))
        elif section.type == artifact_format.SectionType.KV_LANDING_RECIPROCALS and mutation == "legacy_klr":
            constants = _parse_kvc1(section.data)
            constants["layer0.k_normed_head0"] = ((1 << 30), -30, (1 << 31) + 1)
            replaced.append(artifact_format.Section(section.type, writer.write_kvc1(3, constants)))
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


def _verify_unsupported_width(tmp_path: Path, monkeypatch, head_dim: int, *,
                              loader_sqrt128_relation=False):
    """Construct one integrity-valid hostile bit-2 artifact for the loader.

    The test bypasses only the converter's new width preflight so it can exercise
    the compiled loader's independent trust boundary.  It does not bypass the
    writer's QKC1 construction or mutate the loader under test.
    """
    monkeypatch.setattr(converter.V, "check_fused_k_head_dim", lambda _model: None)
    # The reference fixture correctly declines to construct an odd pairwise
    # RoPE shape. Serialize a legal QKC1 source first, then make CFG1 hostile;
    # ParseConfig must reject parity before any QKC1 table interpretation.
    source_head_dim = 128 if head_dim % 2 else head_dim
    model, sections = _artifact_sections(None, head_dim=source_head_dim)
    if source_head_dim != head_dim:
        replaced = []
        for section in sections:
            if section.type != artifact_format.SectionType.CONFIG:
                replaced.append(section)
                continue
            config = bytearray(section.data)
            struct.pack_into("<I", config, 24, head_dim)
            replaced.append(artifact_format.Section(section.type, bytes(config)))
        sections = replaced
    if loader_sqrt128_relation:
        replaced = []
        for section in sections:
            if section.type != artifact_format.SectionType.COMPOSITION_CONSTANTS:
                replaced.append(section)
                continue
            constants = _parse_kvc1(section.data)
            head_max = max(Fraction(value) for value in model.weight_scales["layer0.k_norm.gain"])
            constants["layer0.softmax_khead0"] = converter._canonical_scale(
                head_max / Fraction(math.sqrt(128)))
            replaced.append(artifact_format.Section(section.type, writer.write_kvc1(2, constants)))
        sections = replaced
    artifact = tmp_path / f"qkc-head{head_dim}-hostile.sslm"
    artifact_format.write_artifact(artifact, sections, flags=converter.artifact_flags_for_model(model))
    manifest = artifact.with_suffix(".manifest.json")
    return subprocess.run([str(_VERIFIER), str(artifact), str(manifest)],
                          capture_output=True, text=True, check=False)


def test_coherent_qkc1_artifact_loads_before_relation_refusal_cells(tmp_path):
    result = _verify(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_fused_qk_width_wall_accepts_the_pinned_qwen3_family_geometry(tmp_path):
    """The fused-QK production fixture remains the shipped 128-wide Qwen3 geometry."""
    model = _load_legal_qk_model()
    converter.V.check_fused_k_head_dim(model)
    result = _verify(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fused_qk_odd_head_dim_still_fires_the_generic_parity_refusal(tmp_path, monkeypatch):
    result = _verify_unsupported_width(tmp_path, monkeypatch, 127)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "REJECTED: BadConfigHeadDimParity" in output


@pytest.mark.parametrize("head_dim", [126, 64])
def test_fused_qk_even_non128_width_refuses_in_converter_and_loader(tmp_path, monkeypatch, head_dim):
    model = _load_legal_qk_model(head_dim=head_dim)
    with pytest.raises(converter.V.ConverterValidationError,
                       match="UnsupportedFusedKHeadDim"):
        converter.build_sections(model)
    result = _verify_unsupported_width(tmp_path, monkeypatch, head_dim)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "REJECTED: UnsupportedFusedKHeadDim" in output


def test_fused_qk_converter_width_wall_vitality(monkeypatch):
    """Disabling the converter check admits the same hostile 64-wide input."""
    model = _load_legal_qk_model(head_dim=64)
    with pytest.raises(converter.V.ConverterValidationError,
                       match="UnsupportedFusedKHeadDim"):
        converter.build_sections(model)
    monkeypatch.setattr(converter.V, "check_fused_k_head_dim", lambda _model: None)
    sections, _fold_error = converter.build_sections(model)
    assert any(section.type == artifact_format.SectionType.QK_CHANNEL_TABLE for section in sections)


def test_fused_qk_width_wall_rejects_a_head64_artifact_with_sqrt128_softmax(tmp_path, monkeypatch):
    """A relation-consistent forged carry cannot bypass the width support wall."""
    result = _verify_unsupported_width(
        tmp_path, monkeypatch, 64, loader_sqrt128_relation=True)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "REJECTED: UnsupportedFusedKHeadDim" in output


def test_qkc1_magnitude_witness_is_load_legal(tmp_path):
    result = _verify(tmp_path, "magnitude")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_fused_qk_converter_retires_wsc_gain_rows_and_rejects_static_metadata():
    model = _load_legal_qk_model()
    sections, _ = converter.build_sections(model)
    weight_scales = next(section for section in sections
                         if section.type == artifact_format.SectionType.WEIGHT_SCALES)
    reciprocal = next(section for section in sections
                      if section.type == artifact_format.SectionType.KV_LANDING_RECIPROCALS)
    assert b"layer0.q_norm.gain" not in weight_scales.data
    assert b"layer0.k_norm.gain" not in weight_scales.data
    assert b"layer0.k_normed_head0" not in reciprocal.data

    retired = replace(
        model.scales,
        nonlinear=model.scales.nonlinear + (("layer0.softmax.input", 1.0),))
    with pytest.raises(converter.V.ConverterValidationError,
                       match="LegacyFusedKMetadataPresent"):
        converter.build_sections(replace(model, scales=retired))


def _mixed_qk_layers_model():
    cfg = pipeline.ModelConfig(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=1,
        num_key_value_heads=1, head_dim=128, intermediate_size=256,
        vocab_size=32, rope_theta=10000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=True, context_cap=16)
    base = pipeline.fixture_model(cfg)
    weights = dict(base.weights)
    weight_scales = dict(base.weight_scales)
    constants = dict(base.composition_constants)
    for role in ("q_norm", "k_norm"):
        weights.pop(f"layer1.{role}.gain", None)
        weight_scales.pop(f"layer1.{role}.gain", None)
        constants.pop(f"layer1.{role}", None)
    weights["layer0.q_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
    weights["layer0.k_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
    weight_scales["layer0.q_norm.gain"] = [1.0] * cfg.head_dim
    weight_scales["layer0.k_norm.gain"] = [1.0] * cfg.head_dim
    constants["layer0.q_norm"] = (1 << 30, -30)
    constants["layer0.k_norm"] = (1 << 30, -30)
    return replace(base, weights=weights, weight_scales=weight_scales,
                   composition_constants=constants)


def _mixed_model_with_legacy_key(layer, family):
    model = _mixed_qk_layers_model()
    if family == "k_norm.requant":
        site = replace(model.scales.requant[0], name=f"layer{layer}.{family}")
        return replace(model, scales=replace(model.scales, requant=model.scales.requant + (site,)))
    name = f"layer{layer}.{family}"
    return replace(model, scales=replace(model.scales, nonlinear=model.scales.nonlinear + ((name, 1.0),)))


def _validate(model):
    converter.V.validate_model(
        model, fold_ops_tensor=converter._fold_ops_tensor, ctx_fold_tensor=converter._ctx_fold_tensor,
        unicode_major=converter.V.PINNED_UNICODE_VERSION[0],
        unicode_minor=converter.V.PINNED_UNICODE_VERSION[1],
        unicode_patch=converter.V.PINNED_UNICODE_VERSION[2])


@pytest.mark.parametrize(("layer", "family", "allowed"), [
    (0, "k_norm.requant", False), (1, "k_norm.requant", False),
    (0, "k_normed_head0.scale", False), (1, "k_normed_head0.scale", False),
    (0, "softmax.input", False), (1, "softmax.input", True),
])
def test_mixed_qk_layers_retirement_matrix_through_validation_and_sections(layer, family, allowed):
    """R2a: only no-QK softmax.input survives in a model carrying fused QK."""
    model = _mixed_model_with_legacy_key(layer, family)
    key = f"layer{layer}.{family}"
    if allowed:
        before = tuple(model.scales.nonlinear)
        _validate(model)
        converter.build_sections(model)
        assert tuple(model.scales.nonlinear) == before
        assert (key, 1.0) in model.scales.nonlinear
        return
    message = ('LegacyFusedKMetadataPresent: fused-QK conversion input contains retired '
               f'StaticScales key "{key}"')
    with pytest.raises(converter.V.ConverterValidationError, match=message):
        _validate(model)
    with pytest.raises(converter.V.ConverterValidationError, match=message):
        converter.build_sections(model)


def test_fused_qk_loader_refuses_a_restored_legacy_klr_key_before_marshal(tmp_path):
    result = _verify(tmp_path, "legacy_klr")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "REJECTED: LegacyFusedKKeyPresent" in output
    assert 'fused-QK artifact contains retired KvLandingReciprocals key "layer0.k_normed_head0"' in output


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
