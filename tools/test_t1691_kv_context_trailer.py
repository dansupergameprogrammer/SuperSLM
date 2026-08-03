"""T-1691 step 4's own red-first proof (design Claude/Vitruvius/superslm-
t1683-source-attribution-design-2026-08-02.md S7 step 4, D-SLM727a/b): drives
the REAL, compiled `out/sslm_layer_trace.exe` against
`kv_context_trailer_fixture.py`'s own multi-layer, multi-position fixture as
a subprocess (this campaign's own established convention, T-1689) and
asserts the tool's new K/V-context trailer -- the section this file's own
header comment appends AFTER T-1689's saturation-delta trailer -- has the
exact shape the design specifies, checked by byte length as well as by
parsed values, plus the count-reconciliation guard's own vitality.

Skipped (not failed) if the binary has not been built locally, matching
tools/test_t1689_kv_saturation_census.py's own convention.
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
# only row 5 (of the design's own {5,21..28} band) exists; 3-token prompt
# ("abc") gives context_length == 2 prior positions before the last token's
# own walk.
_EXPECTED_TARGET_ROWS = [5]
_EXPECTED_CONTEXT_LENGTH = 2


def _body_end(data: bytes) -> int:
    rows, hidden_size, _fp, capture_mode = struct.unpack("<QQQQ", data[:32])
    assert capture_mode == 1
    row_bytes = 8 + 8 + hidden_size
    return 32 + rows * row_bytes


def _t1689_trailer_span(data: bytes, body_end: int) -> tuple[int, int]:
    """Returns (start, end) byte offsets of T-1689's own trailer, read by
    its own declared `num_layers` header field -- the identical parse
    tools/test_t1689_kv_saturation_census.py's own body-shape test uses."""
    (num_layers,) = struct.unpack("<Q", data[body_end:body_end + 8])
    end = body_end + 8 + 8 * num_layers
    return body_end, end


def _run_trace(model_path, tok_path, tmp_path, label, env=None):
    dump_path = str(tmp_path / f"{label}.bin")
    cmd = [_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path]
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=run_env)
    return proc, dump_path


def test_kv_context_trailer_shape_matches_the_design(tmp_path):
    """The positive case: row 5 (the only design-band row this fixture's
    6-layer geometry reaches) is captured, with exactly
    `context_length * num_kv_heads * head_dim * 2` int8 codes -- checked by
    byte length, not only by parsed values (D-SLM727a's own standard)."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc, dump_path = _run_trace(model_path, tok_path, tmp_path, "shape")
    assert proc.returncode == 0, (
        f"sslm_layer_trace.exe failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "self_check: production and manual-replay paths agree" in proc.stdout
    assert "kv_context_trailer: 1/9 target rows reached" in proc.stdout

    with open(dump_path, "rb") as f:
        data = f.read()
    body_end = _body_end(data)
    _t1689_start, t1689_end = _t1689_trailer_span(data, body_end)

    (num_target_rows,) = struct.unpack("<Q", data[t1689_end:t1689_end + 8])
    assert num_target_rows == len(_EXPECTED_TARGET_ROWS)
    offset = t1689_end + 8
    target_rows = list(struct.unpack(f"<{num_target_rows}Q", data[offset:offset + 8 * num_target_rows]))
    assert target_rows == _EXPECTED_TARGET_ROWS
    offset += 8 * num_target_rows

    (context_length, num_kv_heads, head_dim) = struct.unpack("<QQQ", data[offset:offset + 24])
    assert context_length == _EXPECTED_CONTEXT_LENGTH
    assert num_kv_heads == FIX.NUM_KEY_VALUE_HEADS
    assert head_dim == FIX.HEAD_DIM
    offset += 24

    expected_code_count = num_target_rows * context_length * num_kv_heads * head_dim * 2
    codes = data[offset:offset + expected_code_count]
    assert len(codes) == expected_code_count, (
        f"trailer codes: got {len(codes)} bytes, want exactly {expected_code_count} "
        f"(num_target_rows={num_target_rows} * context_length={context_length} * "
        f"num_kv_heads={num_kv_heads} * head_dim={head_dim} * 2)")
    # This IS the last section of the dump (no reader downstream of this
    # tool yet exists to leave further bytes for) -- the file ends exactly
    # here, checked by byte length.
    assert offset + expected_code_count == len(data), (
        "kv_context trailer is not the last section in the dump, or carries extra/missing bytes")


def test_old_readers_survive_the_new_trailer_append(tmp_path):
    """D-SLM727a: a T-1689-only reader (or a T-1685-only reader) parsing
    only its own declared span sees exactly the same bytes regardless of
    this further append -- confirmed by byte length, mirroring
    test_t1689_kv_saturation_census.py's own
    test_existing_29_row_body_unchanged_by_the_trailer_extension by name and
    by convention, applied here to confirm THIS tool's further extension
    does not disturb either of the two sections that precede it."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc, dump_path = _run_trace(model_path, tok_path, tmp_path, "old-reader")
    assert proc.returncode == 0
    with open(dump_path, "rb") as f:
        data = f.read()
    rows, hidden_size, _fp, capture_mode = struct.unpack("<QQQQ", data[:32])
    assert rows == FIX.NUM_HIDDEN_LAYERS + 1
    assert hidden_size == FIX.HIDDEN_SIZE
    assert capture_mode == 1
    body_end = _body_end(data)

    # T-1689's own trailer: num_layers == NUM_HIDDEN_LAYERS, exactly
    # 8 + 8*num_layers bytes, unaffected by whatever now follows it.
    (num_layers,) = struct.unpack("<Q", data[body_end:body_end + 8])
    assert num_layers == FIX.NUM_HIDDEN_LAYERS
    _start, t1689_end = _t1689_trailer_span(data, body_end)
    assert t1689_end - body_end == 8 + 8 * num_layers

    # The file is STRICTLY LONGER than T-1689's own trailer end -- the new
    # section is genuinely appended, not silently absent.
    assert len(data) > t1689_end, "expected the T-1691 kv_context trailer to follow T-1689's trailer"


def test_kv_trailer_count_guard_fires_on_a_forced_mismatch(tmp_path):
    """Guard vitality (D-SLM727b): SSLM_T1691_KV_TRAILER_FORCE_MISMATCH=1 is
    a test-only debug knob (sslm_layer_trace.cpp, unset in every ordinary
    run) that perturbs the tool's own ACTUAL element count by one, exercising
    the real, shipped comparison against the independently-derived expected
    count -- confirming it rejects loud, non-zero-exit, and writes no dump,
    before this guard is trusted on the real 28-layer/nine-prompt drive."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc, dump_path = _run_trace(
        model_path, tok_path, tmp_path, "forced-mismatch",
        env={"SSLM_T1691_KV_TRAILER_FORCE_MISMATCH": "1"})
    assert proc.returncode != 0, "the count-reconciliation guard did not fire on a forced mismatch"
    assert "FAILED at stage=kv_trailer_count_guard" in proc.stderr
    assert not os.path.exists(dump_path), "a dump was written despite the count guard rejecting"


def test_kv_trailer_count_guard_is_silent_on_an_ordinary_run(tmp_path):
    """The guard's own false-positive check (mirroring T-1689's own
    identity/non-saturating negative control): with the debug knob unset,
    the identical comparison passes and the tool proceeds to write a dump."""
    tok_path, model_path = FIX.write_fixture(tmp_path)
    proc, dump_path = _run_trace(model_path, tok_path, tmp_path, "ordinary")
    assert proc.returncode == 0, (
        f"sslm_layer_trace.exe failed unexpectedly (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "kv_trailer_count_guard" not in proc.stderr
    assert os.path.exists(dump_path)
