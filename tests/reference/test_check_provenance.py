"""Curie's red suite for check_provenance.py's guard vitality (S-HARDEN-5, F3,
design S3.1/S8 step 1d/1e).

PROVENANCE.md's recorded SHA-256 values are, without a machine check, a value
a human reads when re-vendoring later -- nothing computes them against the
committed files and asserts equality. This proves the population-validation
requirement StandardsDocument SS4 sets for any new check: shown able to FAIL
on a fault it exists to catch, not only shown to pass twice on unchanged
input. Every test below tampers a byte, confirms `main()` exits non-zero
naming the mismatched file, and restores -- never leaving a corrupted vendor
file behind even if an assertion fails mid-test.
"""

import contextlib
import io

import pytest

import check_provenance as cp


@pytest.fixture
def restore_file(tmp_path):
    """Snapshot and restore a vendored file's bytes around a test that
    tampers it, so a failing assertion never leaves the working tree with a
    corrupted vendored reference."""
    touched = []

    def _track(path):
        touched.append((path, open(path, "rb").read()))
        return path

    yield _track
    for path, original_bytes in touched:
        with open(path, "wb") as f:
            f.write(original_bytes)


def _run_main():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        code = cp.main()
    return code, buf.getvalue()


def test_clean_tree_passes():
    code, _ = _run_main()
    assert code == 0


@pytest.mark.parametrize("rel_path", list(cp._CHECKED_FILES))
def test_tampering_any_vendored_file_is_caught_and_named(rel_path, restore_file):
    import os

    abs_path = os.path.join(cp._THIS_DIR, rel_path)
    restore_file(abs_path)

    with open(abs_path, "r+b") as f:
        data = f.read()
        f.seek(0)
        f.write(bytes([data[0] ^ 0xFF]) + data[1:])

    code, output = _run_main()
    assert code == 1
    assert rel_path in output
    assert "mismatch" in output.lower()


def test_restoring_the_tampered_file_makes_it_pass_again(restore_file):
    import os

    rel_path = cp._CHECKED_FILES[0]
    abs_path = os.path.join(cp._THIS_DIR, rel_path)
    restore_file(abs_path)

    with open(abs_path, "r+b") as f:
        data = f.read()
        f.seek(0)
        f.write(bytes([data[0] ^ 0xFF]) + data[1:])
    bad_code, _ = _run_main()
    assert bad_code == 1

    with open(abs_path, "wb") as f:
        f.write(data)
    good_code, _ = _run_main()
    assert good_code == 0


def test_missing_provenance_entry_is_reported(tmp_path, monkeypatch):
    """A file with no recorded row in PROVENANCE.md is a distinct failure
    mode from a hash mismatch -- both must be caught, not just the common
    case."""
    monkeypatch.setattr(cp, "_CHECKED_FILES", cp._CHECKED_FILES + ("superslm_spike/does_not_exist.py",))
    code, output = _run_main()
    assert code == 1
    assert "does_not_exist.py" in output
