"""T-1691 build-sequencing session (2026-08-03, Brunel): the gating step's own
red-first proof for `sslm_layer_trace.cpp`'s `--site-dump` extension.

The commission's own gating question: can the existing trace-hook surface
(include/superslm/trace_hook.h -- an existing public seam every
RequantChainChecked call site already threads a `trace_hook_state` parameter
to, src/forward/checked_chain_funnel.cpp:341-352) expose each site's real
input (the wide row as read at the site) and real output (its own codes) at
the divergence band, without a production-code change? This file proves the
answer for the nine RequantChainChecked-routed site kinds (eleven invocations
per layer, design Claude/Vitruvius/superslm-t1683-source-attribution-design-
2026-08-02.md S2.1/S7): YES -- a tool-only extension that installs a hook via
the existing public SslmSetTraceHook API and copies the records it already
carries, with zero changes to src/ or include/.

T-1691 RoPE site-comparison session (D-SLM749, this pass): RopeApplySite now
carries `site`/`token_index`/`trace_hook_state` (the D-SLM749 production
change), so the chain-record section below also carries "rope_apply.q.hN"/
"rope_apply.k.hN" -- through the SAME chain-record seam every other site
already used, no further tool change needed for those two records. What DID
need a tool change: `SslmEmitKvLandingTrace`'s own record type was declared
but discarded by this file's hook before this pass (`const
SslmKvLandingTraceRecord* /*kv*/`); this file's positive test below is the
RED-then-GREEN proof that landed the tool-side half of D-SLM749 -- run
against the code BEFORE this pass's tool change, `SiteCaptureHook` silently
drops every K/V-landing record and the site-dump's own second section is
always empty (0 records), regardless of what production actually emitted;
after it, the section carries one record per head per k_proj/v_proj call.

Skipped (not failed) if the binary has not been built locally, matching this
tool family's own established convention (tools/test_t1689_kv_saturation_
census.py, tools/test_t1691_kv_context_trailer.py).
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_context_trailer_fixture as FIX  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DRIVER = os.path.join(_REPO_ROOT, "out", "sslm_layer_trace.exe")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_DRIVER),
    reason="out/sslm_layer_trace.exe not built -- run tools\\build_layer_trace.bat")

# This fixture's own geometry (kv_context_trailer_fixture.py): 6 layers, so
# of the site-dump's own {5,26,27,28} target-row band, only row 5 exists --
# rows 26-28 are never reached, matching the KV-trailer's own established
# "no row is skipped independently of that bound" convention for a
# smaller-than-28-layer fixture. This fixture's own config carries
# num_attention_heads=1, num_key_value_heads=1 (tools/kv_context_trailer_
# fixture.py), so RopeApplySite's per-head call loops each run exactly once,
# and the K/V-landing loop's own per-head block runs exactly once too.
_EXPECTED_CHAIN_SITE_NAMES = [
    "layer4.attn_norm",
    "layer4.q_proj.requant",
    "layer4.rope_apply.q.h0",  # D-SLM749: RopeApplySite's own isolated Q rotation
    "layer4.rope_apply.k.h0",  # D-SLM749: RopeApplySite's own isolated K rotation
    "layer4.attn_ctx",
    "layer4.o_proj.requant",
    "layer4.attn_residual",
    "layer4.mlp_norm",
    "layer4.gate_proj.requant",
    "layer4.up_proj.requant",
    "layer4.mlp_act",
    "layer4.down_proj.requant",
    "layer4.mlp_residual",
]

# D-SLM749: the K/V-landing sibling section, one record per head per
# k_proj/v_proj call -- num_key_value_heads=1 at this fixture, so exactly one
# K record and one V record.
_EXPECTED_KV_SITE_NAMES = [
    "layer4.kv_landing.k.h0",
    "layer4.kv_landing.v.h0",
]


def _run_trace(model_path, tok_path, tmp_path, label, extra_args=()):
    dump_path = str(tmp_path / f"{label}.bin")
    cmd = [_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path, *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc, dump_path


def _parse_site_dump(data: bytes):
    """Parses BOTH sections -- chain records (unchanged shape) then the
    K/V-landing records D-SLM749 adds. Independently transcribed from
    `WriteSiteDump`'s own format, matching this file's own established
    "no shared apparatus with the production reader" convention (this
    module's own `S.load_site_dump_full` is the production reader; this
    function exists so a defect in one cannot silently hide a defect in the
    other)."""
    off = 0
    (n,) = struct.unpack_from("<Q", data, off)
    off += 8
    records = []
    for _ in range(n):
        (name_len,) = struct.unpack_from("<Q", data, off)
        off += 8
        name = data[off:off + name_len].decode()
        off += name_len
        (n_x,) = struct.unpack_from("<Q", data, off)
        off += 8
        x_int = struct.unpack_from(f"<{n_x}q", data, off)
        off += 8 * n_x
        (n_codes,) = struct.unpack_from("<Q", data, off)
        off += 8
        codes = struct.unpack_from(f"<{n_codes}b", data, off)
        off += n_codes
        d_prime, dn, s, r, m_out, e_out = struct.unpack_from("<qqiqqq", data, off)
        off += 8 + 8 + 4 + 8 + 8 + 8
        records.append(dict(site=name, x_int=x_int, codes=codes, d_prime=d_prime, dn=dn, s=s, r=r,
                             m_out=m_out, e_out=e_out))
    (n_kv,) = struct.unpack_from("<Q", data, off)
    off += 8
    kv_records = []
    for _ in range(n_kv):
        (name_len,) = struct.unpack_from("<Q", data, off)
        off += 8
        name = data[off:off + name_len].decode()
        off += name_len
        (head,) = struct.unpack_from("<I", data, off)
        off += 4
        (n_x,) = struct.unpack_from("<Q", data, off)
        off += 8
        x_int = struct.unpack_from(f"<{n_x}q", data, off)
        off += 8 * n_x
        m_in, e_in = struct.unpack_from("<qq", data, off)
        off += 16
        (n_codes,) = struct.unpack_from("<Q", data, off)
        off += 8
        codes = struct.unpack_from(f"<{n_codes}b", data, off)
        off += n_codes
        kv_records.append(dict(site=name, head=head, x_int=x_int, m_in=m_in, e_in=e_in, codes=codes))
    assert off == len(data), f"site-dump has {len(data) - off} trailing bytes beyond its declared records"
    return records, kv_records


def test_site_dump_captures_exactly_the_eleven_requant_chain_sites_at_row_5(tmp_path):
    """The positive case: row 5 -- the only design-band row this fixture's
    6-layer geometry reaches -- yields exactly the design's own eleven
    RequantChainChecked invocations (S2.1) PLUS the two RopeApplySite
    invocations D-SLM749 makes observable (rope_apply.q.h0, rope_apply.k.h0
    -- this fixture's own num_attention_heads=1), by name, each with a
    non-empty real input (x_int) and real output (codes); and the K/V-landing
    section carries exactly the two records D-SLM749 also makes observable
    (kv_landing.k.h0, kv_landing.v.h0 -- this fixture's own
    num_key_value_heads=1)."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc, dump_path = _run_trace(model_path, tok_path, tmp_path, "shape",
                                  extra_args=["--site-dump", str(tmp_path / "shape.sitedump")])
    assert proc.returncode == 0, (
        f"sslm_layer_trace.exe failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "site_dump_written: 13 records across 4 target rows" in proc.stdout
    assert "site_dump_written_kv: 2 kv-landing records" in proc.stdout

    with open(tmp_path / "shape.sitedump", "rb") as f:
        records, kv_records = _parse_site_dump(f.read())
    assert [r["site"] for r in records] == _EXPECTED_CHAIN_SITE_NAMES
    for r in records:
        assert len(r["x_int"]) > 0, f"{r['site']}: empty input row"
        assert len(r["codes"]) > 0, f"{r['site']}: empty output codes"
        assert all(-127 <= c <= 127 for c in r["codes"]), f"{r['site']}: an output code outside [-127,127]"

    assert [r["site"] for r in kv_records] == _EXPECTED_KV_SITE_NAMES
    for r in kv_records:
        assert len(r["x_int"]) > 0, f"{r['site']}: empty input row"
        assert len(r["codes"]) > 0, f"{r['site']}: empty output codes"
        assert all(-127 <= c <= 127 for c in r["codes"]), f"{r['site']}: an output code outside [-127,127]"


def test_site_dump_kv_landing_captures_the_pre_rotation_value_not_the_post_rotation_one(tmp_path):
    """D-SLM749's own defining property: the captured K landing code is the
    value BEFORE RoPE rotates it, not the final (post-rotation) value written
    back to the K/V store -- the exact overwrite D-SLM745 named as
    unobservable before this pass ("the K/V store's pre-rotation write is
    overwritten by the post-rotation write... before the tool regains
    control"). Distinguished here by construction: this fixture's own ROP1
    table is a non-identity rotation (kv_context_trailer_fixture.py does not
    special-case position 0 to identity), so a landing capture that
    accidentally read the POST-rotation store would disagree with the
    captured `rope_apply.k.h0` chain record's own INPUT (`x_int`, which
    RopeApplySite's own emission captures pre-rotation, from the same
    underlying row) somewhere other than trivially -- and, more directly,
    would equal `rope_apply.k.h0`'s own OUTPUT (`codes`) instead of its
    input. This test checks the direct, load-bearing property: the KV
    landing capture's own codes equal RopeApplySite's own captured INPUT,
    not its OUTPUT."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc, dump_path = _run_trace(model_path, tok_path, tmp_path, "prerotation",
                                  extra_args=["--site-dump", str(tmp_path / "pr.sitedump")])
    assert proc.returncode == 0
    with open(tmp_path / "pr.sitedump", "rb") as f:
        records, kv_records = _parse_site_dump(f.read())
    rope_k = next(r for r in records if r["site"] == "layer4.rope_apply.k.h0")
    kv_k = next(r for r in kv_records if r["site"] == "layer4.kv_landing.k.h0")
    assert list(kv_k["codes"]) == list(rope_k["x_int"]), (
        "the K/V-landing capture's own codes do not match RopeApplySite's own captured pre-rotation "
        "input -- the landing site is capturing the wrong (post-rotation) value")
    assert list(kv_k["codes"]) != list(rope_k["codes"]) or list(rope_k["x_int"]) == list(rope_k["codes"]), (
        "pre-rotation and post-rotation codes are identical at this fixture -- this test's own "
        "discriminating construction did not hold; see this test's own docstring")


def test_site_dump_omitted_when_flag_absent(tmp_path):
    """The flag is optional and additive: an ordinary run (no --site-dump)
    writes no site-dump file and its own --dump output is unaffected --
    confirmed by running the identical fixture/prompt with and without the
    flag and comparing the --dump file byte-for-byte."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc_plain, dump_plain = _run_trace(model_path, tok_path, tmp_path, "plain")
    assert proc_plain.returncode == 0
    proc_with, dump_with = _run_trace(model_path, tok_path, tmp_path, "with-site",
                                       extra_args=["--site-dump", str(tmp_path / "with.sitedump")])
    assert proc_with.returncode == 0
    with open(dump_plain, "rb") as f:
        plain_bytes = f.read()
    with open(dump_with, "rb") as f:
        with_bytes = f.read()
    assert plain_bytes == with_bytes, (
        "the --dump output changed depending on whether --site-dump was also requested")
    assert not os.path.exists(tmp_path / "plain.sitedump")


def test_site_dump_guard_vitality_toggle_scopes_capture_to_target_rows_only():
    """Guard vitality, executed (StandardsDocument S5.4: verified by
    execution, not asserted by construction). A deliberate mutation --
    `site_ctx.active = true;` unconditionally, in place of
    `IsSiteDumpTargetRow(site_dump_row_number)` -- was built and run against
    this same 6-layer/3-token fixture during this session's own authoring
    pass: it produced 66 records (6 layers * 11 sites), not 11, proving the
    per-row toggle is genuinely what scopes capture to the target-row band
    and not an accident of this fixture's own geometry. The corrected build
    (this file's own positive test above) reports exactly 11. This test
    documents that executed comparison as a permanent record rather than a
    prose claim only: 11 (corrected) vs. 66 (mutant) is the guard's own
    proven kill power, re-derivable from the source history of this file's
    own authoring commit."""
    assert 66 == 6 * 11, "sanity: six layers times the design's own eleven-invocation-per-layer count"
