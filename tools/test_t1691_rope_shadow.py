"""T-1691 RoPE site-comparison session (site kind 4/4, port order position
4/4, D-SLM746/D-SLM749): red-first proof that `rope_apply_site_reference`
and `kv_landing_shadow_codes` -- both already ported and both already
primitive-validated and guard-vitality-proved at site kind 2/4
(`test_t1691_attention_shadow.py`'s own tier 1/tier 2, D-SLM750/751) -- agree
with the REAL compiled engine's OWN isolated RoPE/K-V-landing I/O, now that
D-SLM749's production change makes that I/O directly observable
(`RopeApplySite`'s own trace-hook parameter; `SslmEmitKvLandingTrace` wired
at the K/V landing write). This is the tier that could not exist before this
pass: D-SLM745 named RoPE's own isolated contribution and the K/V pre-
rotation landed value as unobservable by any tool-only construction, so
`attn_ctx`'s own comparison (site kind 2/4, D-SLM751) was the only real-
engine evidence touching either mechanism, and only indirectly (through the
composed attn_ctx output, not either mechanism's own I/O).

No new primitives, no new tier 1/tier 2: this file adds only the tier this
pass makes possible.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_context_trailer_fixture as FIX  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402


def _saturation_trailer_end(dump_path: str, base) -> int:
    """Independently transcribed (this module's own "no shared apparatus"
    construction, matching `test_t1691_attention_shadow.py`'s own
    identically-named helper) rather than imported: the K/V-context trailer
    is appended after the 29-row body AND T-1689's own saturation-delta
    trailer."""
    import struct
    with open(dump_path, "rb") as f:
        data = f.read()
    rows, hidden_size, _fp, _cm = struct.unpack_from("<QQQQ", data, 0)
    body_end = 32 + rows * (16 + hidden_size)
    (num_deltas,) = struct.unpack_from("<Q", data, body_end)
    return body_end + 8 + num_deltas * 8

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAYER_TRACE_DRIVER = os.path.join(_REPO_ROOT, "out", "sslm_layer_trace.exe")

_driver_skip = pytest.mark.skipif(
    not os.path.isfile(_LAYER_TRACE_DRIVER),
    reason="out/sslm_layer_trace.exe not built -- run tools\\build_layer_trace.bat")


@_driver_skip
def test_rope_apply_site_reference_matches_the_real_engine_q_and_k_at_the_fixture(tmp_path):
    """`rope_apply_site_reference(row, position, cos_table, sin_table)` against
    the real `RopeApplySite`'s own captured pre-rotation input (x_int) and
    post-rotation output (codes), for both the Q call (`rope_apply.q.h0`) and
    the K call (`rope_apply.k.h0`) -- this fixture's own num_attention_heads=1
    so there is exactly one of each. Two checks per call: feeding the shadow
    the REAL captured input reproduces the REAL captured output exactly
    (never a value this module's own prior comparison produced -- this
    campaign's own established per-site-independent methodology)."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    dump_path = str(tmp_path / "rope.dump")
    site_dump_path = str(tmp_path / "rope.sitedump")
    proc = subprocess.run(
        [_LAYER_TRACE_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path,
         "--site-dump", site_dump_path],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"driver failed: {proc.stdout}\n{proc.stderr}"

    base = load_int8_layer_dump(dump_path)
    records, _kv_records = S.load_site_dump_full(site_dump_path)
    by_site = {r.site: r for r in records}
    artifact = read_artifact(str(model_path))
    trailer = S.load_kv_context_trailer(dump_path, after_offset=_saturation_trailer_end(dump_path, base))

    layer_index = 4  # the only design-band row (row 5) this 6-layer fixture reaches
    position = trailer.context_length  # the current (last-prompt-token) position, real-derived

    for suffix in ("q", "k"):
        rec = by_site[f"layer{layer_index}.rope_apply.{suffix}.h0"]
        real_input = rec.x_int.astype(np.int8)  # RopeApplySite's own pre-rotation row, widened to int64
        shadow_out = S.rope_apply_site_reference(real_input, position, artifact.rope_cos, artifact.rope_sin)
        assert np.array_equal(shadow_out, rec.codes), (
            f"rope_apply.{suffix}.h0 mismatch at the fixture: "
            f"shadow={shadow_out.tolist()} real={rec.codes.tolist()}")


@_driver_skip
def test_kv_landing_shadow_codes_matches_the_real_engine_k_and_v_at_the_fixture(tmp_path):
    """`kv_landing_shadow_codes` against the real K/V-landing write's own
    captured pre-rotation codes (`kv_landing.k.h0`/`kv_landing.v.h0`) -- this
    fixture's own num_key_value_heads=1 so there is exactly one of each,
    feeding the shadow the REAL captured `attn_norm` codes/scale (the K/V
    landing GEMM's own real input, identical to q_proj's own GEMM input)."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    dump_path = str(tmp_path / "kvland.dump")
    site_dump_path = str(tmp_path / "kvland.sitedump")
    proc = subprocess.run(
        [_LAYER_TRACE_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path,
         "--site-dump", site_dump_path],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"driver failed: {proc.stdout}\n{proc.stderr}"

    records, kv_records = S.load_site_dump_full(site_dump_path)
    by_site = {r.site: r for r in records}
    kv_by_site = {r.site: r for r in kv_records}
    artifact = read_artifact(str(model_path))

    layer_index = 4
    layer = artifact.layers[layer_index]
    attn_norm_rec = by_site[f"layer{layer_index}.attn_norm"]

    k_shadow = S.kv_landing_shadow_codes(
        attn_norm_rec.codes, attn_norm_rec.m_out, attn_norm_rec.e_out, layer.k_weight,
        layer.k_fold[:, 0], layer.k_fold[:, 1], layer.k_fold[:, 2], layer.k_bias,
        layer.kv_landing_r_t_k, layer.kv_landing_e_t_k, 1, artifact.config["head_dim"])
    v_shadow = S.kv_landing_shadow_codes(
        attn_norm_rec.codes, attn_norm_rec.m_out, attn_norm_rec.e_out, layer.v_weight,
        layer.v_fold[:, 0], layer.v_fold[:, 1], layer.v_fold[:, 2], layer.v_bias,
        layer.kv_landing_r_t_v, layer.kv_landing_e_t_v, 1, artifact.config["head_dim"])

    k_real = kv_by_site[f"layer{layer_index}.kv_landing.k.h0"]
    v_real = kv_by_site[f"layer{layer_index}.kv_landing.v.h0"]
    assert np.array_equal(k_shadow[0], k_real.codes), (
        f"kv_landing.k.h0 mismatch at the fixture: shadow={k_shadow[0].tolist()} real={k_real.codes.tolist()}")
    assert np.array_equal(v_shadow[0], v_real.codes), (
        f"kv_landing.v.h0 mismatch at the fixture: shadow={v_shadow[0].tolist()} real={v_real.codes.tolist()}")
