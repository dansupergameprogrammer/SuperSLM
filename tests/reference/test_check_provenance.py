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


# --- B7 (T-2123/T-2137, design SS3.5 ruling item 1, SS6): cross-repo reach ------------------
#
# tools/reference_pipeline/{intmath,rope}.py (the live, product copy) and
# tests/reference/superslm_spike/{intmath,rope}.py (this directory's frozen golden-fixture
# copy) are SUPPOSED to be able to diverge -- the frozen copy stays frozen until someone
# deliberately re-vendors it. What must not happen is a divergence nobody decided and nobody
# can see. These cells prove the new cross-repo check passes on byte-identity OR an explicit,
# dated PROVENANCE.md entry recording an intentional decoupling, and fails on neither.


def test_cross_repo_reach_clean_tree_passes():
    code, _ = _run_main()
    assert code == 0


def test_undocumented_cross_repo_divergence_is_caught_and_named(tmp_path, monkeypatch):
    """A live tools/reference_pipeline/ file edited to diverge from the frozen copy, with no
    PROVENANCE.md entry recording the divergence as intentional, must fail loudly."""
    import os

    live_path = os.path.join(cp._THIS_DIR, "..", "..", "tools", "reference_pipeline", "intmath.py")
    live_path = os.path.normpath(live_path)
    original = open(live_path, "rb").read()
    try:
        with open(live_path, "r+b") as f:
            data = f.read()
            f.seek(0)
            f.write(data + b"\n# undocumented drift\n")
        code, output = _run_main()
        assert code == 1
        assert "intmath.py" in output
    finally:
        with open(live_path, "wb") as f:
            f.write(original)


def test_documented_decoupling_entry_makes_a_real_divergence_pass():
    """rope.py's live copy legitimately diverges from the frozen copy by exactly its
    import-line rename (T-2123/T-2137 B0: `from superslm_spike.intmath import` ->
    `from reference_pipeline.intmath import`). PROVENANCE.md records this as an explicit,
    dated, commit-pinned, intentional decoupling, and the check passes rather than failing
    on it."""
    code, output = _run_main()
    assert code == 0, output


# --- Poirot casebook f83afe0-t2137-vendoring-review.md, S1: the decoupling entry must be
# pinned to the divergence a commit actually produced, not a standing exemption for the file --
# a SECOND, unapproved divergence beyond the recorded one must still fail. This is the
# must-reject construction the prior round's vacuous twin never built (it asserted `main() == 0`
# twice, identically, and never tampered rope.py at all).


def test_a_second_undocumented_divergence_beyond_the_pinned_rope_entry_is_caught(restore_file):
    """rope.py already carries one approved, hash-pinned divergence (the B0 import rename).
    A SECOND edit on top of that -- one nobody recorded -- must still fail: the pin names the
    ONE divergence a commit produced, not blanket permission for the file to drift further."""
    import os

    live_path = os.path.normpath(os.path.join(
        cp._THIS_DIR, "..", "..", "tools", "reference_pipeline", "rope.py"))
    restore_file(live_path)

    with open(live_path, "r+b") as f:
        data = f.read()
        f.seek(0)
        f.write(data + b"\n# a second, unrecorded divergence\n")

    code, output = _run_main()
    assert code == 1, (
        f"a second, undocumented divergence on top of rope.py's already-pinned decoupling "
        f"entry was NOT caught -- the guard is exempting the whole file, not the one recorded "
        f"divergence (output: {output})"
    )
    assert "rope.py" in output


def test_reverting_rope_py_partway_back_still_fails_the_pin(restore_file):
    """The pin cuts both ways: an edit that neither matches the frozen copy nor the pinned
    live hash -- e.g. a partial, botched revert of the B0 import rename that lands on some
    THIRD text -- changes rope.py's hash away from the recorded pin and must also fail until
    a new, dated entry records the new state. (An edit that lands exactly back on the frozen
    copy's own bytes is not this case -- it takes the byte-identity path, correctly, because
    there is no divergence left to record.)"""
    import os

    live_path = os.path.normpath(os.path.join(
        cp._THIS_DIR, "..", "..", "tools", "reference_pipeline", "rope.py"))
    restore_file(live_path)

    with open(live_path, "r+b") as f:
        data = f.read()
        # A botched revert: restores the old import spelling but leaves a stray trailing
        # space this repo's own tooling would never introduce -- neither the frozen copy's
        # bytes nor the pinned live hash.
        botched = data.replace(
            b"from reference_pipeline.intmath import rounding_divide_by_pot",
            b"from superslm_spike.intmath import rounding_divide_by_pot ",
        )
        assert botched != data, "fixture assumption broken: the import line was not found"
        f.seek(0)
        f.truncate()
        f.write(botched)

    code, output = _run_main()
    assert code == 1, (
        f"a botched, partial revert of rope.py's B0 import rename -- matching neither the "
        f"frozen copy's bytes nor the pinned live hash -- was not caught (output: {output})"
    )
