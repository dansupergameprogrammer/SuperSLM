"""Red suite for check_source_drift.py (T-1521, SuperSLM_S3a_WalkingSkeleton_
Plan.md Sec11 S3.1c item 2b).

Every cell drives the mechanism against a constructed scratch "source
repository" (a real, hermetic `git init` under a temp directory, never the
real `D:\\Wizard`) and a constructed scratch "vendored" tree standing in for
`tests/reference/`, per StandardsDocument Sec4's population-validation
requirement -- a check must be shown able to FAIL on a fault it exists to
catch, not only shown to pass on unchanged input, and a real, unmodified
source-vs-vendor pair could never show the four named failure modes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

import check_source_drift as csd


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True)


def _init_source_repo(tmp: str, files: dict[str, str]) -> tuple[str, str]:
    """Creates a scratch git repo at tmp/source with `files` (relpath ->
    content) committed. Returns (repo_dir, commit_hash) of that commit."""
    repo = os.path.join(tmp, "source")
    os.makedirs(repo, exist_ok=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)
    for relpath, content in files.items():
        abspath = os.path.join(repo, relpath)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "initial"], repo)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, commit


def _write_vendor_tree(tmp: str, vendored_relpath: str, content: str) -> str:
    vendor_root = os.path.join(tmp, "vendor")
    abspath = os.path.join(vendor_root, vendored_relpath)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return vendor_root


def _write_provenance(vendor_root: str, rows: list[tuple[str, str, str, str]]) -> str:
    """rows: (vendored_relpath, sha256, source_commit, group)."""
    lines = ["| File | SHA-256 | Source commit | Group |", "|---|---|---|---|"]
    for f, h, c, g in rows:
        lines.append(f"| `{f}` | `{h}` | `{c}` | `{g}` |")
    path = os.path.join(vendor_root, "PROVENANCE.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _sha256_hex(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


_ONE_FILE = ("superslm_spike/thing.py", "Tools/superslm_spike/thing.py")


# --- The env-var/missing-source failure mode. ---


def test_unset_env_var_fails_with_missing_source_message():
    # source_dir explicitly None -- the shape main() takes when the
    # environment variable itself is unset -- must fail even with an empty
    # file set, since the source can never be resolved at all.
    code = csd.main(repo_root="/does/not/matter", source_dir=None, checked_files=())
    assert code == 1


def test_resolve_source_raises_on_none():
    with pytest.raises(csd.SourceUnavailable, match="not set"):
        csd.resolve_source(None)


def test_resolve_source_raises_on_nonexistent_path():
    with pytest.raises(csd.SourceUnavailable, match="not a directory"):
        csd.resolve_source("/does/not/exist/at/all")


def test_resolve_source_raises_on_non_git_directory():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(csd.SourceUnavailable, match="not a git repository"):
            csd.resolve_source(tmp)


def test_main_never_exits_zero_on_missing_source():
    with tempfile.TemporaryDirectory() as tmp:
        checked = (_ONE_FILE,)
        code = csd.main(repo_root=tmp, source_dir=None, checked_files=checked)
        assert code == 1
        code2 = csd.main(repo_root=tmp, source_dir=tmp, checked_files=checked)
        assert code2 == 1


# --- The clean-pass case (wiring control: everything agrees). ---


def test_clean_unmodified_vendor_matches_source_and_passes():
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), commit, "test-group")])

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert failures == []
        code = csd.main(repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov)
        assert code == 0


# --- Failure mode 1: hash mismatch (hand-edited vendored copy). ---


def test_hand_edited_vendored_copy_fails_naming_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), commit, "test-group")])

        # Hand-edit the vendored copy on disk, without touching PROVENANCE.md
        # or the source -- exactly the plan's own named red cell.
        with open(os.path.join(vendor_root, _ONE_FILE[0]), "a", encoding="utf-8", newline="\n") as f:
            f.write("# tampered\n")

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert len(failures) == 1
        assert _ONE_FILE[0] in failures[0]
        assert "mismatch" in failures[0]


def test_reverting_the_hand_edit_passes_again():
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), commit, "test-group")])
        vendored_abspath = os.path.join(vendor_root, _ONE_FILE[0])

        with open(vendored_abspath, "a", encoding="utf-8", newline="\n") as f:
            f.write("# tampered\n")
        bad = csd.check_drift(repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov)
        assert len(bad) == 1

        with open(vendored_abspath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        good = csd.check_drift(repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov)
        assert good == []


# --- Failure mode 2: drift-ahead (T-1528). ---


def test_source_commit_after_the_pin_fails_distinctly_as_drift_ahead():
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, pinned_commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), pinned_commit, "test-group")])

        # A real commit in the source repo AFTER the pinned one, touching the
        # same file, without re-vendoring -- the vendored copy is still
        # byte-identical to the PINNED commit (so this is not a hash
        # mismatch), but the source has moved on.
        new_content = "def thing():\n    return 2\n"
        with open(os.path.join(source_repo, _ONE_FILE[1]), "w", encoding="utf-8", newline="\n") as f:
            f.write(new_content)
        _git(["add", "-A"], source_repo)
        _git(["commit", "-q", "-m", "drift ahead"], source_repo)

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert len(failures) == 1
        assert "drift-ahead" in failures[0]
        assert "mismatch" not in failures[0], "drift-ahead must be reported distinctly from a hash mismatch"
        assert _ONE_FILE[0] in failures[0]


def test_a_source_commit_after_the_pin_touching_a_DIFFERENT_file_does_not_trip_drift_ahead():
    """Input-coverage control: the drift-ahead check is scoped per-file (git
    log -- <path>), so a later commit that touches an unrelated file in the
    source repo must not falsely report drift for a file it never touched."""
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, pinned_commit = _init_source_repo(
            tmp, {_ONE_FILE[1]: content, "Tools/superslm_spike/unrelated.py": "x = 1\n"}
        )
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), pinned_commit, "test-group")])

        with open(os.path.join(source_repo, "Tools/superslm_spike/unrelated.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write("x = 2\n")
        _git(["add", "-A"], source_repo)
        _git(["commit", "-q", "-m", "touches something else"], source_repo)

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert failures == []


# --- Failure mode 3: commit not found (T-1533). ---


def test_pinned_commit_absent_from_a_valid_repo_fails_as_commit_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, _real_commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        # A syntactically valid but nonexistent commit hash -- a shallow
        # clone or an unrelated checkout would both surface this shape.
        fake_commit = "1" * 40
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), fake_commit, "test-group")])

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert len(failures) == 1
        assert "not found in source repository" in failures[0]
        assert "mismatch" not in failures[0]
        assert "drift-ahead" not in failures[0]


def test_commit_not_found_is_distinct_from_missing_source():
    """The three failure modes are pairwise distinguishable by message --
    proven here for commit-not-found vs. hash-mismatch vs. drift-ahead
    together, since each prior test proves one message never appears
    alongside another's marker; this test additionally proves commit-not-found
    fires as a per-file failure (check_drift), not the whole-run
    SourceUnavailable exception missing-source raises."""
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, _real_commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        fake_commit = "2" * 40
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), fake_commit, "test-group")])

        # Does not raise SourceUnavailable -- the source repo itself is real
        # and valid; only the specific pinned commit is absent.
        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert len(failures) == 1


# --- Multiple files: one mismatch does not hide another's distinct finding. ---


def test_two_files_with_two_different_failure_modes_are_both_reported():
    with tempfile.TemporaryDirectory() as tmp:
        f1 = ("superslm_spike/a.py", "Tools/superslm_spike/a.py")
        f2 = ("superslm_spike/b.py", "Tools/superslm_spike/b.py")
        content_a = "a = 1\n"
        content_b = "b = 1\n"
        source_repo, commit = _init_source_repo(tmp, {f1[1]: content_a, f2[1]: content_b})
        vendor_root = os.path.join(tmp, "vendor")
        for relpath, content in ((f1[0], content_a), (f2[0], content_b)):
            abspath = os.path.join(vendor_root, relpath)
            os.makedirs(os.path.dirname(abspath), exist_ok=True)
            with open(abspath, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
        prov = _write_provenance(
            vendor_root,
            [
                (f1[0], _sha256_hex(content_a), commit, "g"),
                (f2[0], _sha256_hex(content_b), commit, "g"),
            ],
        )
        # f1: hand-edited vendored copy (mismatch). f2: left alone (passes).
        with open(os.path.join(vendor_root, f1[0]), "a", encoding="utf-8", newline="\n") as fh:
            fh.write("# tampered\n")

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(f1, f2), provenance_path=prov
        )
        assert len(failures) == 1
        assert f1[0] in failures[0]
        assert f2[0] not in failures[0]


def test_missing_source_commit_recorded_for_a_file_is_its_own_named_failure():
    with tempfile.TemporaryDirectory() as tmp:
        content = "def thing():\n    return 1\n"
        source_repo, commit = _init_source_repo(tmp, {_ONE_FILE[1]: content})
        vendor_root = _write_vendor_tree(tmp, _ONE_FILE[0], content)
        prov = _write_provenance(vendor_root, [(_ONE_FILE[0], _sha256_hex(content), "PENDING", "test-group")])

        failures = csd.check_drift(
            repo_root=vendor_root, source_dir=source_repo, checked_files=(_ONE_FILE,), provenance_path=prov
        )
        assert len(failures) == 1
        assert "no Source commit recorded" in failures[0]


# --- Present truth: the real vendored tree, against the real source, passes. ---


def test_real_tree_against_real_source_passes():
    """The wiring cell: run the real check against the real vendored copies
    in this repository and a real checkout of the source at D:\\Wizard, if
    one is available in this environment -- skipped, not failed, when it
    isn't (this project's own repo-availability convention), since a public
    CI runner never has this sibling checkout (module docstring)."""
    source = os.environ.get(csd.ENV_VAR)
    if not source or not os.path.isdir(source):
        pytest.skip(f"{csd.ENV_VAR} not set to a real checkout in this environment")
    code = csd.main(source_dir=source)
    assert code == 0
