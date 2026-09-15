"""T-2700: the direct-K landing consumes the post-RoPE Q30 unit.

The fixture exercises the production scalar/vector landing.  The Qwen3 cell
uses a compiled capture produced from real calibration tokens: its observation
unit and saturation count are independent C++ evidence, rather than a second
Python implementation of the landing.
"""

from __future__ import annotations

import argparse
import dataclasses
from fractions import Fraction
import json
import math
import struct
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "reference_pipeline"))

import t2700_fused_k_calibration as calibration  # noqa: E402
import convert_model as converter  # noqa: E402
from reference_pipeline import artifact_cache, intmath, pipeline  # noqa: E402


def _fixture_qk_model():
    cfg = pipeline.ModelConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, head_dim=128, intermediate_size=512, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16)
    base = pipeline.fixture_model(cfg)
    weights = dict(base.weights)
    scales = dict(base.weight_scales)
    constants = dict(base.composition_constants)
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        weights[f"{prefix}.q_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
        weights[f"{prefix}.k_norm.gain"] = np.ones(cfg.head_dim, dtype=np.int8)
        scales[f"{prefix}.q_norm.gain"] = [1.0] * cfg.head_dim
        scales[f"{prefix}.k_norm.gain"] = [1.0] * cfg.head_dim
        constants[f"{prefix}.q_norm"] = (1 << 30, -30)
        constants[f"{prefix}.k_norm"] = (1 << 30, -30)
    model = dataclasses.replace(base, weights=weights, weight_scales=scales,
                                composition_constants=constants)
    peaks = {f"layer{layer}": np.ones((cfg.num_key_value_heads, cfg.head_dim), dtype=np.float64)
             for layer in range(cfg.num_hidden_layers)}
    return pipeline.with_provisional_qk_channel_table(model, peaks)


def _direct_fixture_observation(model):
    """A float-dimensional post-RoPE reference over the exact direct-K inputs."""
    cfg = model.config
    seed = np.asarray([
        [[17, -11, 31, -7, 13, -19, 23, -29], [-23, 29, -31, 11, -17, 7, 19, -13]],
        [[-37, 41, -43, 47, -53, 59, -61, 67], [71, -73, 79, -83, 89, -97, 101, -103]],
        [[107, -109, 113, -127, 97, -89, 83, -79], [-71, 67, -61, 59, -53, 47, -43, 41]],
    ], dtype=np.int64)
    k = np.tile(seed, (1, 1, cfg.head_dim // seed.shape[-1]))
    cos, sin = model.rope_tables
    gain = model.weights["layer0.k_norm.gain"].astype(np.int64)
    wide = (pipeline._vec_rmsnorm(k.reshape(-1, cfg.head_dim), cfg.head_dim) * gain).reshape(k.shape)
    rotated = pipeline._vec_rope(wide, cos, sin, len(k), 0)
    # This is deliberately independent of the landing helper.  The RoPE
    # primitive returns the Q30-rounded quotient, so its code retains the
    # pre-RoPE wide-code unit.  Supplying a scale with e-30 here would decode
    # the same integer as a value 2**-30 too small.
    m, e = pipeline._qk_wide_source_scale(model, "layer0")
    reference = np.ldexp(rotated.astype(np.float64) * float(m), e)
    # Build a legal per-channel authority from the independent post-RoPE
    # values before asking the production landing to encode them.
    peaks = np.maximum(np.max(np.abs(reference), axis=0) * 1.02, 1.0)
    model = pipeline.with_provisional_qk_channel_table(model, {"layer0": peaks, "layer1": peaks})
    landed = pipeline._qk_direct_k_vector(model, "layer0", k, cos, sin, len(k), 0)
    channel_scale = peaks / 127.0
    decoded = landed.astype(np.float64) * channel_scale.reshape(1, *channel_scale.shape)
    # The pre-fix call had exactly the same inputs but omitted RoPE's Q30 unit.
    old_m, old_e = pipeline._qk_wide_source_scale(model, "layer0")
    old = np.empty_like(landed)
    table = model.qk_channel_table["layer0"]
    for token in range(len(k)):
        for head in range(cfg.num_key_value_heads):
            for channel in range(cfg.head_dim):
                old[token, head, channel] = intmath.residual_reconcile(
                    int(rotated[token, head, channel]), old_m, int(table["r_t"][head, channel]),
                    old_e - pipeline.rope.ROPE_FRAC_BITS, int(table["e_t"][head, channel]))
    return reference, decoded, landed, np.clip(old, -127, 127), channel_scale


def test_fixture_landing_decodes_to_post_rope_float_quantity_and_rejects_wrong_unit():
    model = _fixture_qk_model()
    reference, decoded, landed, old, channel_scale = _direct_fixture_observation(model)
    assert float(np.max(np.abs(decoded - reference) / channel_scale.reshape(1, *channel_scale.shape))) <= 1.0
    assert float(np.mean(np.abs(landed) == 127)) == 0.0
    assert float(np.mean(old == 0)) == 1.0
    old_decoded = old.astype(np.float64) * channel_scale.reshape(1, *channel_scale.shape)
    assert float(np.max(np.abs(old_decoded - reference) / channel_scale.reshape(1, *channel_scale.shape))) > 1.0


def test_scalar_vector_full_fixture_still_agree_after_rotated_unit_fix():
    model = _fixture_qk_model()
    tokens = [1, 2, 3, 4, 5]
    assert np.array_equal(pipeline.forward(model, tokens), pipeline.forward_scalar_reference(model, tokens))


def test_converter_derives_qk_softmax_head_scale_from_qkc1_head_maximum():
    model = _fixture_qk_model()
    peaks = {"layer0": np.array([[2.0] * model.config.head_dim,
                                  [6.0] * model.config.head_dim]),
             "layer1": np.array([[3.0] * model.config.head_dim,
                                  [9.0] * model.config.head_dim])}
    model = pipeline.with_provisional_qk_channel_table(model, peaks)
    sections, _ = converter.build_sections(model)
    payload = next(section.data for section in sections
                   if section.type == converter.F.SectionType.COMPOSITION_CONSTANTS)
    _magic, _version, count, words, names_size, _reserved = struct.unpack_from("<4sIIIII", payload)
    names = [struct.unpack_from("<II", payload, 24 + 8 * index) for index in range(count)]
    values_offset = 24 + 8 * count
    name_blob = payload[values_offset + 8 * words * count:values_offset + 8 * words * count + names_size]
    constants = {
        name_blob[offset:offset + length].decode("utf-8"): struct.unpack_from(
            "<qq", payload, values_offset + 16 * index)
        for index, (offset, length) in enumerate(names)}
    for layer, expected_peaks in enumerate((peaks["layer0"], peaks["layer1"])):
        for head in range(2):
            b = Fraction(float(expected_peaks[head, 0]) / 127.0)
            assert constants[f"layer{layer}.softmax_khead{head}"] == converter._canonical_scale(
                b / Fraction(math.sqrt(model.config.head_dim)))


def _float64_post_rope_from_source(model, prefix, k_codes, positions):
    """Independent float64 QK K model, written here rather than using pipeline helpers.

    ``k_codes`` are the production integer K-projection output under test.  This
    side reconstructs their physical unit from the source K-projection scale,
    dequantizes the source K-norm gain, performs real RMSNorm, and applies the
    mathematical RoPE rotation.  It deliberately does not call ``pipeline``,
    ``intmath``, or ``rope`` for any float calculation.
    """
    cfg = model.config
    k_projection_scale = next(site.output_scale for site in model.scales.requant
                              if site.name == f"{prefix}.k_proj.requant")
    gain = (model.weights[f"{prefix}.k_norm.gain"].astype(np.float64) *
            np.asarray(model.weight_scales[f"{prefix}.k_norm.gain"], dtype=np.float64))
    values = k_codes.astype(np.float64) * np.float64(k_projection_scale)
    normalized = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) +
                                  np.float64(cfg.rms_norm_eps))
    wide = normalized * gain.reshape(1, 1, cfg.head_dim)
    channels = np.arange(0, cfg.head_dim, 2, dtype=np.float64)
    inverse_frequency = np.float64(cfg.rope_theta) ** (-channels / np.float64(cfg.head_dim))
    angles = np.asarray(positions, dtype=np.float64).reshape(-1, 1) * inverse_frequency.reshape(1, -1)
    cos = np.cos(angles).reshape(-1, 1, cfg.head_dim // 2)
    sin = np.sin(angles).reshape(-1, 1, cfg.head_dim // 2)
    rotated = np.empty_like(wide)
    rotated[:, :, 0::2] = wide[:, :, 0::2] * cos - wide[:, :, 1::2] * sin
    rotated[:, :, 1::2] = wide[:, :, 0::2] * sin + wide[:, :, 1::2] * cos
    return rotated


def _qk_codes_for_real_prompt(model, prefix, tokens):
    """Production integer-side input for a named QK layer and real prompt IDs."""
    cfg = model.config
    reader = pipeline._ScaleReader(model.scales)
    hidden = model.weights["embed"][tokens].astype(np.int64)
    normed = pipeline._clamp_int8(pipeline._rescale(
        pipeline._vec_rmsnorm(hidden, cfg.hidden_size) *
        model.weights[f"{prefix}.attn_norm.gain"].astype(np.int64),
        reader, f"{prefix}.attn_norm.requant"))
    return pipeline._clamp_int8(pipeline._requant(
        pipeline._vec_project(model, f"{prefix}.k_proj", normed), reader,
        f"{prefix}.k_proj")).reshape(len(tokens), cfg.num_key_value_heads, cfg.head_dim)


def test_python_qwen3_direct_k_decodes_to_independent_float64_post_rope_quantity():
    """Real Qwen3 prompt IDs at layer 0 and a later QK-norm layer.

    The landing contract is ``clamp(round(float / quantum), -127, 127)``.
    Its independent float64 population is split at the pre-clamp rounding
    boundary (127.5 quanta), not at code 127: an unclipped 126.7-quantum
    value correctly rounds to 127.  On the preserved provisional cache the
    unclipped maximum is 0.501 quanta and the p99 is 0.495; 0.51 leaves a
    float-rounding margin without fitting a saturation error.
    """
    cache = Path("D:/_t2698/qwen3-embedding-0.6b-provisional-cache")
    assert cache.is_dir(), f"missing real Qwen3 provisional cache: {cache}"
    model = artifact_cache.load_artifact(cache)
    cfg = model.config
    tokens = [9707, 151643]
    unclipped_bound = 0.51
    clipped_fraction_bounds = {"layer0": 0.011, f"layer{cfg.num_hidden_layers - 1}": 0.034}
    for prefix in ("layer0", f"layer{cfg.num_hidden_layers - 1}"):
        k_codes = _qk_codes_for_real_prompt(model, prefix, tokens)
        landed = pipeline._qk_direct_k_vector(model, prefix, k_codes, *model.rope_tables, len(tokens), 0)
        reference = _float64_post_rope_from_source(model, prefix, k_codes, range(len(tokens)))
        quantum = model.qk_channel_peaks[prefix] / 127.0
        decoded = landed.astype(np.float64) * quantum.reshape(1, *quantum.shape)
        quantum_view = quantum.reshape(1, *quantum.shape)
        error_quanta = np.abs(decoded - reference) / quantum_view
        float_quanta = np.abs(reference) / quantum_view
        clipped = float_quanta >= 127.5
        unclipped = ~clipped
        assert float(np.max(error_quanta[unclipped])) <= unclipped_bound
        assert float(np.quantile(error_quanta[unclipped], 0.99)) <= 0.50
        assert float(np.mean(clipped)) <= clipped_fraction_bounds[prefix]
        assert np.all(np.abs(landed[clipped]) == 127)
        assert np.all(np.sign(landed[clipped]) == np.sign(reference[clipped]))
        for position in range(len(tokens)):
            for head in range(cfg.num_key_value_heads):
                mask = unclipped[position, head]
                cosine = np.dot(decoded[position, head][mask], reference[position, head][mask])
                cosine /= np.linalg.norm(decoded[position, head][mask]) * np.linalg.norm(reference[position, head][mask])
                assert float(cosine) >= 0.9999
        for exponent_delta in (-1, 1):
            wrong = landed.astype(np.float64) * (quantum * (2.0 ** exponent_delta)).reshape(1, *quantum.shape)
            wrong_error_quanta = np.abs(wrong - reference) / quantum.reshape(1, *quantum.shape)
            assert float(np.max(wrong_error_quanta[unclipped])) >= 57.0


def _capture_report(path, model):
    cfg = model.config
    m, e = pipeline._qk_post_rope_wide_scale(model, "layer0")
    lines = ["T2700_FUSED_K_CAPTURE_V1", "summary\tlanding_saturation_count\t0",
             "layer\thead\tchannel\traw_abs_peak\twide_scale_m\twide_scale_e\treal_peak"]
    for layer in range(cfg.num_hidden_layers):
        for head in range(cfg.num_key_value_heads):
            for channel in range(cfg.head_dim):
                lines.append(f"{layer}\t{head}\t{channel}\t1\t{m}\t{e}\t{math.ldexp(float(m), e).hex()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_merge_cli_persists_explicit_checkpoint_on_fixture_capture(tmp_path):
    model = _fixture_qk_model()
    source = tmp_path / "source-cache"
    artifact_cache.save_artifact(model, source)
    report = tmp_path / "fixture-capture.tsv"
    _capture_report(report, model)
    out_cache = tmp_path / "merged-cache"
    out_sslm = tmp_path / "merged.sslm"
    verifier = Path("D:/SuperSLM/.worktrees/t2693-fused-k-slice2/out/cmake-cpu-only-default/Release/sslm_verify.exe")
    assert verifier.is_file(), f"missing compiled verifier: {verifier}"
    run = subprocess.run([
        sys.executable, str(TOOLS / "t2700_fused_k_calibration.py"), "merge",
        "--cache", str(source), "--checkpoint", "fixture-checkpoint", "--capture-report", str(report),
        "--out-cache", str(out_cache), "--out-sslm", str(out_sslm), "--verifier", str(verifier), "--skip-verify",
    ], text=True, capture_output=True, check=True)
    assert json.loads(run.stdout.splitlines()[-1])["landing_saturation_count"] == 0
    assert json.loads((out_cache / "metadata.json").read_text(encoding="utf-8"))["checkpoint_path"] == "fixture-checkpoint"
    assert out_sslm.is_file()


def _fixture_capture_runner(path):
    path.write_text(
        """import math
import os
from pathlib import Path
import sys

report = Path(sys.argv[3])
is_c = report.stem == 'pass-c'
clipped = int(os.environ.get('T2700_FIXTURE_PASS_C_CLIPPED', '1')) if is_c else 0
m, e = 2130706432, -54
lines = ['T2700_FUSED_K_CAPTURE_V1', 'summary\\tcallback_count\\t1000000',
         f'summary\\tlanding_saturation_count\\t{clipped}',
         'layer\\thead\\tchannel\\traw_abs_peak\\twide_scale_m\\twide_scale_e\\treal_peak\\tlanding_saturation_count']
for layer in range(2):
    for head in range(2):
        for channel in range(128):
            count = clipped if (layer, head, channel) == (0, 0, 0) else 0
            lines.append(f'{layer}\\t{head}\\t{channel}\\t1\\t{m}\\t{e}\\t{math.ldexp(float(m), e).hex()}\\t{count}')
report.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
""", encoding="utf-8")


def _fixture_flow(tmp_path, monkeypatch, *, clipped, name):
    monkeypatch.setenv("T2700_FIXTURE_PASS_C_CLIPPED", str(clipped))
    model = _fixture_qk_model()
    source = tmp_path / f"{name}-source-cache"
    artifact_cache.save_artifact(model, source)
    inputs = tmp_path / f"{name}-inputs"
    inputs.mkdir()
    (inputs / "prefix-453.txt").write_text("1\n", encoding="utf-8")
    for index in range(600):
        (inputs / f"suffix-{index:03d}.txt").write_text("1\n", encoding="utf-8")
    capture = tmp_path / f"{name}-capture.py"
    _fixture_capture_runner(capture)
    verifier = Path("D:/SuperSLM/.worktrees/t2693-fused-k-slice2/out/cmake-cpu-only-default/Release/sslm_verify.exe")
    args = argparse.Namespace(checkpoint="fixture-checkpoint", out=str(tmp_path / f"{name}.sslm"),
                              work=str(tmp_path / f"{name}-work"), capture=str(capture),
                              verifier=str(verifier), skip_verify=True)
    work = Path(args.work)
    work.mkdir()
    return calibration._flow_from_cache(args, source, inputs, work), args


def test_fixture_flow_runs_a_b_c_records_one_allowed_clip_and_is_deterministic(tmp_path, monkeypatch):
    first, first_args = _fixture_flow(tmp_path, monkeypatch, clipped=1, name="first")
    second, second_args = _fixture_flow(tmp_path, monkeypatch, clipped=1, name="second")
    assert first["pass_c_landing_saturation_count"] == 1
    assert first["pass_c_overshooting_channels"] == [{"layer": 0, "head": 0, "channel": 0, "count": 1}]
    assert Path(first["peak_tables"]).is_file()
    assert Path(first_args.out).read_bytes() == Path(second_args.out).read_bytes()
    assert first["artifact_sha256"] == second["artifact_sha256"]


def test_fixture_flow_refuses_pass_c_above_one_clip_per_million(tmp_path, monkeypatch):
    with pytest.raises(calibration.ChannelScaleDidNotConverge, match="ChannelScaleDidNotConverge: pass C clipped 2/1000000"):
        _fixture_flow(tmp_path, monkeypatch, clipped=2, name="refused")


def test_compiled_capture_accepts_a_suffix_manifest_before_opening_the_artifact(tmp_path):
    capture = Path("D:/SuperSLM/.worktrees/t2693-fused-k-slice2/out/cmake-cpu-only-default/Release/t2700_fused_k_capture.exe")
    assert capture.is_file(), f"missing compiled capture driver: {capture}"
    prefix = tmp_path / "prefix.txt"
    prefix.write_text(" ".join(["1"] * 453) + "\n", encoding="utf-8")
    suffix = tmp_path / "suffix.txt"
    suffix.write_text("1\n", encoding="utf-8")
    suffixes = tmp_path / "suffixes.txt"
    suffixes.write_text(str(suffix) + "\n", encoding="utf-8")
    result = subprocess.run([str(capture), str(tmp_path / "missing.sslm"), str(prefix), str(tmp_path / "report.tsv"),
                             "--suffix-list", str(suffixes)], text=True, capture_output=True)
    assert result.returncode == 2
    assert "artifact read failed" in result.stderr
