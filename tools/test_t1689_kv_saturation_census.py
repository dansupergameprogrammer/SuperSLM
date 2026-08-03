"""T-1689's own red-first proof (design Claude/Vitruvius/superslm-t1683-
source-attribution-design-2026-08-02.md S5): drives the REAL, compiled
`out/sslm_layer_trace.exe` -- built via `tools/build_layer_trace.bat` --
against `kv_saturation_fixture.py`'s hand-derived fixture as a subprocess,
never a `tests/test_main.cpp` construction (this campaign has already paid
two decision-log entries, D-SLM705/D-SLM707, correcting exactly that
ambiguity once). Parses the dump's own new trailing section (this tool's
own S5 extension to the T-1685 dump format) and asserts the per-layer
saturation delta matches the fixture's own hand-computed value exactly,
including the zero (non-saturating) case -- the guard must not report a
false positive on ordinary, in-domain data.

Skipped (not failed) if the binary has not been built locally, matching
this repo's own established convention (test_pinned_calibration_gate.py,
test_sslm_convert_loader_join.py).
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_fixture as FIX  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DRIVER = os.path.join(_REPO_ROOT, "out", "sslm_layer_trace.exe")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_DRIVER),
    reason="out/sslm_layer_trace.exe not built -- run tools\\build_layer_trace.bat")


def _read_saturation_trailer(dump_path):
    """Reads T-1689's own append-only trailer (this file's own extension to
    the T-1685 dump format, sslm_layer_trace.cpp's header comment): skips
    the existing 29-row-shaped body (here: embedding + NUM_HIDDEN_LAYERS
    rows), then reads uint64 num_layers followed by num_layers uint64
    deltas."""
    with open(dump_path, "rb") as f:
        rows, hidden_size, _fingerprint, capture_mode = struct.unpack("<QQQQ", f.read(32))
        assert capture_mode == 1, f"capture_mode == {capture_mode}, want 1"
        row_bytes = 8 + 8 + hidden_size  # int64 m, int64 e, hidden_size int8 codes
        f.seek(rows * row_bytes, os.SEEK_CUR)
        (num_layers,) = struct.unpack("<Q", f.read(8))
        deltas = list(struct.unpack(f"<{num_layers}Q", f.read(8 * num_layers)))
    return deltas


def _run_trace(model_path, tok_path, tmp_path, label):
    dump_path = str(tmp_path / f"{label}.bin")
    cmd = [_DRIVER, str(model_path), str(tok_path), FIX.PROMPT, "--dump", dump_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"[{label}] sslm_layer_trace.exe failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return dump_path, proc.stdout


def test_saturating_fixture_reports_the_hand_computed_delta(tmp_path):
    """The positive case: a K/V weight row chosen (module docstring,
    kv_saturation_fixture.py) so the landing accumulate at output channel 0
    is 16129 -- outside [-127,127] -- for both K and V. Expected delta:
    [2] (this is a single-hidden-layer fixture)."""
    tok_path, sat_path, _nosat_path = FIX.write_fixture(tmp_path)
    dump_path, stdout = _run_trace(sat_path, tok_path, tmp_path, "saturating")
    assert "self_check: production and manual-replay paths agree" in stdout
    deltas = _read_saturation_trailer(dump_path)
    assert deltas == FIX.EXPECTED_DELTAS_SATURATING, (
        f"saturating fixture: got deltas={deltas}, want {FIX.EXPECTED_DELTAS_SATURATING} "
        f"(hand-derived: K dim0 + V dim0 both saturate, dim1 both in-domain)")


def test_identity_fixture_reports_zero_saturation(tmp_path):
    """The negative control (design S5: "a fixture with no saturating value
    reports all-zero deltas -- the guard must not report false positives on
    ordinary, non-saturating data"). Identity K/V weight leaves the landing
    accumulate at exactly {127,-127} -- IN [-127,127], no saturation."""
    tok_path, _sat_path, nosat_path = FIX.write_fixture(tmp_path)
    dump_path, stdout = _run_trace(nosat_path, tok_path, tmp_path, "identity")
    assert "self_check: production and manual-replay paths agree" in stdout
    deltas = _read_saturation_trailer(dump_path)
    assert deltas == FIX.EXPECTED_DELTAS_IDENTITY, (
        f"identity fixture: got deltas={deltas}, want {FIX.EXPECTED_DELTAS_IDENTITY} "
        f"(hand-derived: landing accumulate {{127,-127}} is fully in-domain)")


def test_existing_29_row_body_unchanged_by_the_trailer_extension(tmp_path):
    """Append-only convention (design S5, coverage-model dimension 9): the
    trailer is new bytes AFTER the existing dump body, and the existing
    body's own shape (rows, hidden_size, prompt_fingerprint, capture_mode,
    then per-row m/e/codes) is unchanged -- an old, T-1689-unaware reader
    parsing only the first `32 + rows*row_bytes` bytes must see exactly the
    same body T-1685 already shipped."""
    tok_path, sat_path, _nosat_path = FIX.write_fixture(tmp_path)
    dump_path, _stdout = _run_trace(sat_path, tok_path, tmp_path, "body-shape")
    with open(dump_path, "rb") as f:
        data = f.read()
    rows, hidden_size, _fp, capture_mode = struct.unpack("<QQQQ", data[:32])
    assert rows == FIX.NUM_HIDDEN_LAYERS + 1  # embedding row + one layer
    assert hidden_size == FIX.HIDDEN_SIZE
    assert capture_mode == 1
    row_bytes = 8 + 8 + hidden_size
    body_end = 32 + rows * row_bytes
    trailer = data[body_end:]
    (num_layers,) = struct.unpack("<Q", trailer[:8])
    assert num_layers == FIX.NUM_HIDDEN_LAYERS
    # T-1691 (design S7 step 4, D-SLM727a) appends its OWN trailer after this
    # one, by the identical append-only convention this trailer itself used
    # against T-1685's body -- so "no more bytes after this trailer" is no
    # longer the invariant; "this trailer's own span is exactly this many
    # bytes, checked by byte length, not only by parsed values" still is.
    # tools/test_t1691_kv_context_trailer.py owns the equivalent assertion
    # for the section that now follows.
    own_trailer_span = data[body_end:body_end + 8 + 8 * num_layers]
    assert len(own_trailer_span) == 8 + 8 * num_layers, (
        "T-1689's own trailer carries exactly num_layers uint64 deltas, checked by byte length")
