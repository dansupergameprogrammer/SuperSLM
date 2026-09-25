"""Self-test for tools/ci/check_gpu_catch_totality.py, authored by the test author (TE-437), not the
gate's builder.

Runs the commissioning constructions of commission_gpu_catch_totality.py against the live src/gpu:
every `catch (...)` clause removed as a real edit would remove it (the gate must name exactly that
clause's `try` when the try is outermost, and read OK when it is nested), a typed-only try added to
an existing file, as a function-try-block and in a new file in a new subdirectory, and the two
refusal shapes (a raw string literal, no try at all). Expectations are derived independently of the
gate's lexer (see the commissioning module's docstring). Each construction runs a byte-identical
copy of the gate with no arguments, as the CI step does.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("te437_commission", HERE / "commission_gpu_catch_totality.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

TIP_GPU = cm.REPO / "src" / "gpu"
REJECT = cm.reject_legs(TIP_GPU, None)
ACCEPT = cm.accept_legs(TIP_GPU)


def test_the_live_tree_has_catch_all_sites_to_remove():
    # The population the per-site cells range over: every catch (...) at the tip, 18 closing an
    # outermost try and 1 (Device::Init's innermost) closing a nested one, read at 91779ad. A tree
    # where the walk finds none would make every per-site cell vacuous.
    sites = [l for l in REJECT if "catch (...)" in l["name"]]
    nested = [l for l in ACCEPT if "catch (...)" in l["name"]]
    assert len(sites) >= 18 and len(nested) >= 1


@pytest.mark.parametrize("idx", range(len(REJECT)), ids=[l["name"] for l in REJECT])
def test_must_reject(idx, tmp_path):
    leg = REJECT[idx]
    code, out = cm.execute(leg, tmp_path, cm.DEFAULT_GATE, TIP_GPU, None, idx)
    assert cm.judge_reject(leg, code, out, None), f"exit {code}\n{out}"


@pytest.mark.parametrize("idx", range(len(ACCEPT)), ids=[l["name"] for l in ACCEPT])
def test_must_accept(idx, tmp_path):
    code, out = cm.execute(ACCEPT[idx], tmp_path, cm.DEFAULT_GATE, TIP_GPU, None, idx)
    assert code == 0 and "OK:" in out, f"exit {code}\n{out}"
