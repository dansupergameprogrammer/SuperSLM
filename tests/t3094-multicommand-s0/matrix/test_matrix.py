"""T-3103 design-matrix and partition-script cells.

The real matrix is never partitioned here: the conductor runs the seeded partition. A partition's
per-split composition depends only on signature quotas, never on the seed, so the matrix's own
validity is checked by a seedless canonical deal (each signature's units in listed order) passed to the
partition's check function. Partition mechanics are tested on a small synthetic matrix only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import partition_matrix as pm  # noqa: E402

MATRIX_PATH = HERE / "design-matrix.json"
MATRIX = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def canonical_deal(matrix):
    assignment, by_sig = {}, {}
    for unit in matrix["units"]:
        by_sig.setdefault(unit["signature"], []).append(unit["unit"])
    for sig, units in by_sig.items():
        labels = [s for s in pm.ALL_SPLITS for _ in range(matrix["signatures"][sig]["quota"].get(s, 0))]
        assert len(labels) == len(units)
        assignment.update(zip(sorted(units), labels))
    return assignment


ROWS = pm.rows_with_split(MATRIX, canonical_deal(MATRIX))


def test_builder_is_deterministic(tmp_path):
    before = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(HERE / "build_design_matrix.py")], check=True, capture_output=True)
    assert hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest() == before


def test_matrix_holds_no_text_fields():
    allowed = {"row", "stratum", "construction", "cmp", "watch", "action", "aspect", "role", "negated",
               "entity_type", "placement", "possessive", "atoms", "route", "arm", "reserved", "member"}
    assert set(MATRIX["columns"]) == allowed
    for unit in MATRIX["units"]:
        for row in unit["rows"]:
            for value in row:
                assert isinstance(value, (bool, int)) or len(value) <= 24


def test_matrix_meets_every_plan_check():
    errors, balance = pm.check(MATRIX, ROWS)
    assert errors == [], errors[:10]
    for scope in MATRIX["balance"]["decisive_scopes"]:
        assert balance[scope]["max_v"] <= 0.10


def test_matrix_meets_checks_without_detection_arm():
    rows = [r for r in ROWS if not r["arm"]]
    errors, _ = pm.check(MATRIX, rows)
    # Only the arm's own coarse quotas can differ when its rows are dropped.
    assert all(e.startswith("coarse quota") and e.split("(")[1].rstrip().endswith("True)") or "True)" in e
               for e in errors), errors[:10]


def test_split_totals():
    per = Counter(r["split"] for r in ROWS)
    core_g = sum(1 for r in ROWS if r["split"] == "G" and not r["arm"])
    assert core_g == 560 and per["G"] == 1020
    assert per["T"] == 7000 and per["V"] >= 600 and per["dev"] >= 200


def test_units_with_a_designed_split_are_not_labelled_in_the_matrix():
    assert all(set(u) == {"unit", "signature", "rows"} for u in MATRIX["units"])
    assert "split" not in MATRIX["columns"]


# ------------------------------------------------------------ partition mechanics on a toy matrix

def toy_matrix():
    cols = MATRIX["columns"]
    units, sigs, coarse = [], {}, Counter()
    row_no = 0

    def add(sig, quota, stratum, construction, n_units):
        nonlocal row_no
        sigs[sig] = {"kind": "single", "group": "-", "quota": quota}
        for i in range(n_units):
            row_no += 1
            vals = {c: "-" for c in cols}
            vals.update(row=f"r{row_no:06d}", stratum=stratum, construction=construction, cmp="none",
                        watch="auto", atoms=0, arm=False, reserved=False, member=0)
            units.append({"unit": f"u{row_no:05d}", "signature": sig, "rows": [[vals[c] for c in cols]]})
    add("a", {"H": 3, "T": 5}, "GEN-parallel", "-", 8)
    add("b", {"G": 2, "V": 1}, "GEN-pronoun", "-", 3)
    return {"columns": cols, "units": units, "signatures": sigs, "coarse_quotas": [],
            "balance": MATRIX["balance"]}


def test_deal_hits_quotas_and_is_seed_deterministic():
    m = toy_matrix()
    a1, a2, a3 = pm.deal(m, 7), pm.deal(m, 7), pm.deal(m, 8)
    assert a1 == a2 and a1 != a3
    for a in (a1, a3):
        c = Counter(a.values())
        assert c == Counter({"T": 5, "H": 3, "G": 2, "V": 1})


def test_deal_refuses_quota_mismatch():
    m = toy_matrix()
    m["signatures"]["a"]["quota"] = {"H": 4, "T": 5}
    with pytest.raises(ValueError):
        pm.deal(m, 1)


def test_stream_is_platform_fixed():
    s = pm.Stream(12345)
    assert [s.below(1000) for _ in range(5)] == [pm.Stream(12345).below(1000)] + \
        [x for x in (lambda st: [st.below(1000) for _ in range(5)])(pm.Stream(12345))][1:]
    assert pm.derived_seed(5, 0) == 5 and pm.derived_seed(5, 1) != 5


def test_check_catches_damage():
    rows = [dict(r) for r in ROWS]
    # move one gate F1 row to H: gate composition and coarse quotas must fail
    victim = next(r for r in rows if r["split"] == "G" and r["stratum"] == "F1")
    victim["split"] = "H"
    errors, _ = pm.check(MATRIX, rows)
    assert any("G F1" in e for e in errors) and any("coarse quota" in e for e in errors)
    rows = [dict(r) for r in ROWS]
    next(r for r in rows if r["split"] == "T")["reserved"] = True
    assert any(e.startswith("T:") for e in pm.check(MATRIX, rows)[0])


def test_balance_check_flags_a_planted_association():
    """Vitality: in H, every durative plain F-cell family with watch reuse moves to entity type zone,
    which plants an entity x watch association; the decisive balance check must flag it."""
    rows = [dict(r) for r in ROWS]
    for r in rows:
        if r["split"] == "H" and r["stratum"] in ("F1", "F2", "F5", "F6", "F7", "F8", "F9", "F10")                 and r["aspect"] == "durative" and r["watch"] == "reuse":
            r["entity_type"] = "zone"
    errors, _ = pm.check(MATRIX, rows)
    assert any("balance H durative/plain entity_typexwatch" in e for e in errors), errors[:5]


def test_balance_reports_unresolved_small_tables():
    _, balance = pm.check(MATRIX, ROWS)
    assert balance["eval"]["unresolved"], "small numeric strata must be listed as unresolved"
    assert all("resolving_power=" in u for u in balance["eval"]["unresolved"])
    assert balance["H"]["resolved_tables"] > 0


def test_cramers_v_reference_values():
    assert pm.cramers_v([("a", "x"), ("b", "y")] * 50) == pytest.approx(1.0)
    assert pm.cramers_v([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")] * 25) == pytest.approx(0.0)
    assert pm.cramers_v_corrected([("a", "x"), ("b", "y")] * 50) == pytest.approx(1.0, abs=0.02)
    assert pm.cramers_v_corrected([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")] * 25) == 0.0


def test_cli_refuses_existing_output_dir(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    toy = tmp_path / "toy.json"
    toy.write_text(json.dumps(toy_matrix()), encoding="utf-8")
    assert pm.main(["--matrix", str(toy), "--seed", "1", "--out", str(out)]) == 2


def test_cli_fails_cleanly_on_a_matrix_that_misses_the_plan(tmp_path, capsys):
    toy = tmp_path / "toy.json"
    toy.write_text(json.dumps(toy_matrix()), encoding="utf-8")
    out = tmp_path / "out"
    assert pm.main(["--matrix", str(toy), "--seed", "3", "--out", str(out), "--max-attempts", "2"]) == 1
    text = capsys.readouterr().out
    assert "PARTITION FAILED" in text and "FAILED: " in text and not out.exists()
