"""Curie's red suite for the F8 UCD guard and golden instrument (S-HARDEN-5).

Two independent guard-vitality cells (S-HARDEN-5 design S7 dimension 11), each
proven able to fail before being trusted (StandardsDocument SS4's
population-validation requirement -- a check shown able to fail on a fault it
was designed for is not the same as a check merely shown to pass twice):

1. `Unicode.build()`'s `PINNED_UNICODE_VERSION` guard actually rejects a wrong
   version, and does so BEFORE any of the O(0x110000) range/ccc/decomp/compose
   loops run -- not merely reachable in principle.
2. `unicode_tables_golden.format_report()`'s SHA-256 digest actually changes
   when the underlying `Unicode` object's tables change -- a digest script
   that hashed a stale or cached blob would pass a determinism-only check
   forever while discriminating nothing.

Both tests avoid calling `Unicode.self_test()` (~90s, the exhaustive
property/NFC/whitespace/permutation sweep over all 0x110000 codepoints) --
that sufficiency proof is exercised once per CI run by the `generators` job's
own `python tools/unicode_tables_golden.py` step, not duplicated here.
"""

import pytest

import unicode_tables as ut
from unicode_tables_golden import format_report


def test_version_guard_rejects_mismatch():
    with pytest.raises(ut.UnicodeVersionMismatch):
        ut.Unicode.build(running_version="99.0.0")


def test_version_guard_accepts_the_pinned_version():
    # running_version is injectable independent of the actual interpreter;
    # passing the pin's own value must not raise, proving the guard's
    # comparison itself (not merely "always raises").
    pin = ut.PINNED_UNICODE_VERSION
    u = ut.Unicode.build(running_version=f"{pin[0]}.{pin[1]}.{pin[2]}")
    assert u.version == __import__("unicodedata").unidata_version


def test_version_guard_fires_before_any_table_building_loop_runs(monkeypatch):
    """Fault-inject `_ranges` (the table-building primitive every one of
    letter/number/space derives from) to raise if called at all, then confirm
    the version guard still raises ITS OWN exception first -- proving the
    guard is checked before bytes are produced, not merely that it eventually
    raises somewhere in the method body."""

    def boom(pred):
        raise AssertionError("table-building loop ran before the version guard fired")

    monkeypatch.setattr(ut, "_ranges", boom)
    with pytest.raises(ut.UnicodeVersionMismatch):
        ut.Unicode.build(running_version="99.0.0")


def test_golden_digest_changes_when_the_underlying_tables_change():
    """The digest's own discriminating power: mutate an already-built,
    already-`self_test`-proven `Unicode` object directly (bypassing
    `Unicode.build()`/`self_test()` entirely) and confirm `format_report`'s
    digest and the affected per-table entry count both move. This isolates
    "does the report reflect current object state" from "does Unicode.build()
    reproduce unicodedata" (self_test's separate, already-existing job) --
    a digest script with a bug that hashes a stale or partial reference would
    pass a determinism-only check forever while discriminating nothing
    (Mendeleev Gap, S-HARDEN-5 design S7 dimension 6/11)."""
    u = ut.Unicode.build()
    baseline = format_report(u)

    # Fault: add one ccc entry that Unicode.build() never produced.
    assert 0x41 not in u.ccc, "test fixture assumption: 'A' carries no combining class"
    u.ccc[0x41] = 230
    mutated = format_report(u)

    assert mutated != baseline
    baseline_fields = dict(line.split("=", 1) for line in baseline.splitlines())
    mutated_fields = dict(line.split("=", 1) for line in mutated.splitlines())
    assert mutated_fields["blob_sha256"] != baseline_fields["blob_sha256"]
    assert mutated_fields["ccc_entries"] != baseline_fields["ccc_entries"]

    # Restore and confirm the report reverts exactly.
    del u.ccc[0x41]
    assert format_report(u) == baseline


def test_golden_digest_is_deterministic_across_two_independent_builds():
    """Determinism alone does not prove discriminating power (the mutation
    test above does that) but it is still a required, separate property: two
    independent builds of the same pinned interpreter state must serialize to
    the identical digest."""
    u1 = ut.Unicode.build()
    u2 = ut.Unicode.build()
    assert format_report(u1) == format_report(u2)
