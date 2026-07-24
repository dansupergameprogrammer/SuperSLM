"""Curie's red suite for check_no_pow_operator.py (S-HARDEN-5 M1 fold; design
Sec5.2(c), Sec5.3, Sec5.4; T-409).

Mirrors tests/reference/test_check_provenance.py's convention: the
population-validation requirement StandardsDocument Sec4 sets for any new
check -- shown able to FAIL on a fault it exists to catch, not only shown to
pass on unchanged input.

Two of these tests are RED against the current, unfixed tree (5c431d2 plus
this fold's test-only additions, before M1's own code changes land):

- test_current_tree_still_fails: gen_intmath_fixtures.py still contains all
  three `**` sites (:428, :436, :484) this fold's own design names -- this
  test PASSES today (it is asserting the still-broken state), and is the
  fixture the vitality cells below need: a checker that always reports OK
  cannot be told apart from a checker that is broken.
- test_temporarily_reintroducing_pow_is_caught_and_named: tampers a scratch
  copy of an already-clean generator (gen_matmul_fixtures.py, which has zero
  `**` uses today) to reintroduce each of the three eliminated forms in turn,
  confirms check_no_pow_operator.py's underlying AST walk (find_pow_uses)
  reports it, and restores nothing on disk (it never writes gen_matmul_
  fixtures.py itself -- it walks a temp copy).

Once M1's fix lands (gen_intmath_fixtures.py's three sites rewritten to
eliminate `**` entirely), test_current_tree_still_fails is the one obligated
to flip: it must be updated to assert a clean tree, or deleted in favor of
the now-green check_no_pow_operator.py exit code the `generators` CI job
enforces directly. That is this cell's own documented obligation, named here
so it is not silently stale once the build seat's fix lands.
"""

import contextlib
import io
import os
import tempfile

import check_no_pow_operator as cnp


def _run_main():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        code = cnp.main()
    return code, buf.getvalue()


def test_current_tree_still_fails():
    """RED (expected): the three `**` sites named by design Sec5.1 have not
    been rewritten yet. This test passes -- it documents the still-broken
    state -- and must be revisited the moment M1's fix lands."""
    gen_path = os.path.join(cnp._THIS_DIR, "gen_intmath_fixtures.py")
    hits = cnp.find_pow_uses(gen_path)
    assert hits == [428, 436, 484], (
        f"expected the three known-open ** sites at lines [428, 436, 484], got {hits} -- "
        "if this list is empty or different, M1's fix has landed and this test (plus "
        "check_no_pow_operator.py's CI wiring) needs to move from documenting the open "
        "defect to enforcing its absence"
    )


def test_clean_generator_passes():
    gen_path = os.path.join(cnp._THIS_DIR, "gen_matmul_fixtures.py")
    assert cnp.find_pow_uses(gen_path) == []


def _write_temp_module(source: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    return path


def test_temporarily_reintroducing_float_square_is_caught():
    """The original M1 defect shape: `scale ** 2` where `scale` is a float."""
    path = _write_temp_module("scale = 0.1\nq_c = scale ** 2\n")
    try:
        hits = cnp.find_pow_uses(path)
        assert hits == [2]
    finally:
        os.remove(path)


def test_temporarily_reintroducing_integer_literal_power_is_caught():
    """The `10 ** 12`-shaped exact-integer case -- an AST walk over
    ast.BinOp(op=ast.Pow) does not distinguish float from int operands, so
    this legitimate-looking exact-integer use is caught identically to the
    float-squaring defect. That is the whole point of a total ban: the fix is
    eliminating every `**` site, not classifying which ones are "safe"."""
    path = _write_temp_module("x = 10 ** 12 + 7\n")
    try:
        hits = cnp.find_pow_uses(path)
        assert hits == [1]
    finally:
        os.remove(path)


def test_temporarily_reintroducing_variable_exponent_power_is_caught():
    """The `4 ** m`-shaped exact-integer case, m a loop variable."""
    path = _write_temp_module("for m in range(32):\n    y = 4 ** m\n")
    try:
        hits = cnp.find_pow_uses(path)
        assert hits == [2]
    finally:
        os.remove(path)


def test_missing_generator_file_is_reported(monkeypatch):
    """A checked file absent from disk is a distinct failure mode from a
    file that parses clean but contains `**` -- both must be caught, not
    just the common case (mirrors tests/reference/test_check_provenance.py's
    test_missing_provenance_entry_is_reported)."""
    monkeypatch.setattr(cnp, "_CHECKED_FILES", ("does_not_exist_generator.py",))
    code, output = _run_main()
    assert code == 1
    assert "does_not_exist_generator.py" in output
    assert "not found" in output.lower()


def test_main_reports_todays_three_open_sites(monkeypatch):
    """main() end-to-end against the real, current gen_intmath_fixtures.py:
    RED today (all three sites still open), naming each file:line."""
    monkeypatch.setattr(cnp, "_CHECKED_FILES", ("gen_intmath_fixtures.py",))
    code, output = _run_main()
    assert code == 1
    for line in (428, 436, 484):
        assert f"gen_intmath_fixtures.py:{line}" in output


def test_main_passes_on_a_checked_file_set_with_zero_pow_uses(monkeypatch):
    monkeypatch.setattr(cnp, "_CHECKED_FILES", ("gen_matmul_fixtures.py",))
    code, _ = _run_main()
    assert code == 0


def test_nested_pow_inside_an_expression_is_still_found():
    """A `**` use is not only ever a bare top-level statement -- confirm the
    walk finds one nested inside a function call argument, the actual shape
    of the M1 defect (`math.floor(_POLY_C / (_POLY_A * scale ** 2))`)."""
    path = _write_temp_module("import math\nq = math.floor(1 / (2 * (0.1 ** 2)))\n")
    try:
        hits = cnp.find_pow_uses(path)
        assert hits == [2]
    finally:
        os.remove(path)
