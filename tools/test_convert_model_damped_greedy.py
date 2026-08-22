"""Production-converter coverage for T-2199's opt-in DGC1 artifact surface."""

import os
import struct
import sys


sys.path.insert(0, os.path.dirname(__file__))

import convert_model as C  # noqa: E402
import sslm_format as F  # noqa: E402
from reference_pipeline import pipeline as P  # noqa: E402


def _model():
    cfg = P.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
        context_cap=16,
    )
    return P.fixture_model(cfg)


def test_default_conversion_remains_unflagged_and_has_no_dgc1_section():
    sections, _fold_error = C.build_sections(_model())
    assert all(s.type != F.SectionType.DAMPED_GREEDY_CONSTANTS for s in sections)
    data, _fingerprint = F.build_artifact(sections)
    flags, = struct.unpack_from("<I", data, 16)
    assert flags == 0


def test_damped_conversion_emits_ruled_source_scale_and_known_flag():
    sections, _fold_error = C.build_sections(_model(), enable_damped_greedy=True)
    dgc = [s for s in sections if s.type == F.SectionType.DAMPED_GREEDY_CONSTANTS]
    assert len(dgc) == 1
    assert struct.unpack("<qi", dgc[0].data) == (
        C.DAMPED_GREEDY_SCALE_M, C.DAMPED_GREEDY_SCALE_E)
    assert (C.DAMPED_GREEDY_SCALE_M, C.DAMPED_GREEDY_SCALE_E) == (2883584, -36)

    data, _fingerprint = F.build_artifact(
        sections, flags=F.DAMPED_GREEDY_CONSTANTS_FLAG)
    flags, = struct.unpack_from("<I", data, 16)
    assert flags == F.DAMPED_GREEDY_CONSTANTS_FLAG
    section_count, = struct.unpack_from("<I", data, 12)
    section_types = [struct.unpack_from("<I", data, 64 + i * 40)[0]
                     for i in range(section_count)]
    assert section_types.count(F.SectionType.DAMPED_GREEDY_CONSTANTS) == 1


def test_damped_and_option_g_flags_compose_without_collision(tmp_path):
    sections, _fold_error = C.build_sections(_model(), enable_damped_greedy=True)
    expected = F.OPTION_G_FUSED_K_LANDING_FLAG | F.DAMPED_GREEDY_CONSTANTS_FLAG
    data, _fingerprint = F.build_artifact(sections, flags=expected)
    flags, = struct.unpack_from("<I", data, 16)
    assert flags == expected

    path = tmp_path / "composed-flags.sslm"
    path.write_bytes(data)
    assert F.read_header(path)["flags"] == expected
