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


# --- T-1529 (§11 S3.1c item 2a): the group-commit-identity property is
# machine-checked, not only a per-file hash. A hand-edit recording two files
# of the same closure at two individually-correct, disagreeing commits must
# fail loudly and name the group and the disagreeing files/commits -- silently
# passing here would defeat the joint-pin property dynamic_engine.py's own
# bit-equality-to-pipeline.forward_dynamic claim depends on. ---


def test_clean_tree_groups_are_all_internally_consistent():
    """Every group PROVENANCE.md currently records agrees on Source commit --
    proven directly against the parsed table, not only via main()'s exit code,
    so a future change to the grouping pass's own return shape is caught here
    even if it happened to leave main()'s aggregate exit code green."""
    with open(cp.PROVENANCE_PATH, "r", encoding="utf-8") as f:
        recorded = cp._parse_provenance(f.read())
    mismatches = cp._check_group_commit_identity(recorded, cp._CHECKED_FILES)
    assert mismatches == []


def test_two_criterion2_closure_files_pinned_at_disagreeing_commits_fails_naming_the_group():
    """The fault this pass exists to catch: two files that share the
    `criterion2-closure` group but record different Source commits."""
    recorded = {
        "superslm_spike/dynamic_engine.py": cp._Row("a" * 64, "1" * 40, "criterion2-closure"),
        "superslm_spike/pipeline.py": cp._Row("b" * 64, "2" * 40, "criterion2-closure"),
        "superslm_spike/silu_lut.py": cp._Row("c" * 64, "1" * 40, "criterion2-closure"),
        "superslm_spike/constrain.py": cp._Row("d" * 64, "1" * 40, "criterion2-closure"),
    }
    checked = tuple(recorded.keys())
    mismatches = cp._check_group_commit_identity(recorded, checked)
    assert len(mismatches) == 1
    assert "criterion2-closure" in mismatches[0]
    assert "superslm_spike/dynamic_engine.py" in mismatches[0]
    assert "superslm_spike/pipeline.py" in mismatches[0]


def test_group_disagreement_in_the_real_provenance_file_is_caught_by_main(restore_file):
    """The wiring cell: hand-editing PROVENANCE.md's own Source-commit column
    for one row of a real group must fail `main()` end to end, naming the
    disagreeing group -- not merely the isolated `_check_group_commit_identity`
    unit above."""
    restore_file(cp.PROVENANCE_PATH)
    with open(cp.PROVENANCE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # dynamic_engine.py's own recorded Source commit, real and correctly
    # hashed -- moved to a different (also real, also correctly-formed)
    # 40-hex-digit commit than the other three criterion2-closure rows still
    # carry, so the per-file hash pass stays green and only the group pass
    # can catch this.
    real_commit = "ca67e90ead90373fc55680a67e2b41e0d7c9abca"
    disagreeing_commit = "111111111111111111111111111111111111111a"
    assert text.count(f"`{real_commit}`") >= 4, "fixture assumption: at least 4 rows share the real commit"
    edited = text.replace(
        f"`superslm_spike/dynamic_engine.py` | `593e9b41dff762d783a620dff28a612e15a086fc85735405c1f41ecb1774a059` | `{real_commit}`",
        f"`superslm_spike/dynamic_engine.py` | `593e9b41dff762d783a620dff28a612e15a086fc85735405c1f41ecb1774a059` | `{disagreeing_commit}`",
        1,
    )
    assert edited != text, "fixture assumption: the targeted row text was found and replaced exactly once"
    with open(cp.PROVENANCE_PATH, "w", encoding="utf-8") as f:
        f.write(edited)

    code, output = _run_main()
    assert code == 1
    assert "criterion2-closure" in output


def test_reverting_the_group_disagreement_passes_again(restore_file):
    """The symmetric close: restoring the shared commit on the edited row
    returns `main()` to green, proving the check is a live comparison against
    current disk content, not a one-shot latch."""
    restore_file(cp.PROVENANCE_PATH)
    with open(cp.PROVENANCE_PATH, "r", encoding="utf-8") as f:
        original = f.read()
    real_commit = "ca67e90ead90373fc55680a67e2b41e0d7c9abca"
    edited = original.replace(
        f"`superslm_spike/dynamic_engine.py` | `593e9b41dff762d783a620dff28a612e15a086fc85735405c1f41ecb1774a059` | `{real_commit}`",
        "`superslm_spike/dynamic_engine.py` | `593e9b41dff762d783a620dff28a612e15a086fc85735405c1f41ecb1774a059` | `111111111111111111111111111111111111111a`",
        1,
    )
    assert edited != original
    with open(cp.PROVENANCE_PATH, "w", encoding="utf-8") as f:
        f.write(edited)
    bad_code, _ = _run_main()
    assert bad_code == 1

    with open(cp.PROVENANCE_PATH, "w", encoding="utf-8") as f:
        f.write(original)
    good_code, _ = _run_main()
    assert good_code == 0
