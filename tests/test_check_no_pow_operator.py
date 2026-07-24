"""Curie's red suite for check_no_pow_operator.py (S-HARDEN-5 M1 fold; design
Sec5.2(c), Sec5.3, Sec5.4; T-409).

Mirrors tests/reference/test_check_provenance.py's convention: the
population-validation requirement StandardsDocument Sec4 sets for any new
check -- shown able to FAIL on a fault it exists to catch, not only shown to
pass on unchanged input.

M1's fix has landed: gen_intmath_fixtures.py's three `**` sites (:428, :436,
:484) are rewritten (the exact-integer literal, the bit-shift, and
`scale * scale`) and the generator's tree is clean. Per this file's own
documented obligation (Curie, S-HARDEN-5 M1 fold build log), the two tests
that asserted the prior, still-broken state
(test_current_tree_still_fails, test_main_reports_todays_three_open_sites)
are updated here, by the build seat, to assert the now-clean tree instead --
the transition the test author's own docstring anticipated and commissioned
("must be revisited the moment M1's fix lands").

test_temporarily_reintroducing_pow_is_caught_and_named-style cells tamper a
scratch copy of an already-clean generator (gen_matmul_fixtures.py, which
has zero `**` uses) to reintroduce each of the three eliminated forms in
turn, confirm check_no_pow_operator.py's underlying AST walk (find_pow_uses)
reports it, and restore nothing on disk (never writing gen_matmul_
fixtures.py itself -- it walks a temp copy).
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


def test_current_tree_is_clean():
    """M1's fix has landed (design Sec5.4 step 2): the three `**` sites named
    by design Sec5.1 (:428, :436, :484) are rewritten to eliminate `**`
    entirely. Zero hits over the real, current gen_intmath_fixtures.py."""
    gen_path = os.path.join(cnp._THIS_DIR, "gen_intmath_fixtures.py")
    hits = cnp.find_pow_uses(gen_path)
    assert hits == [], (
        f"expected zero ** uses in gen_intmath_fixtures.py after M1's fix, got {hits} -- "
        "a nonempty result here means the ** operator was reintroduced"
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


def test_main_passes_on_the_real_gen_intmath_fixtures(monkeypatch):
    """main() end-to-end against the real, current gen_intmath_fixtures.py:
    green now that M1's fix has landed (all three sites eliminated)."""
    monkeypatch.setattr(cnp, "_CHECKED_FILES", ("gen_intmath_fixtures.py",))
    code, _ = _run_main()
    assert code == 0


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
