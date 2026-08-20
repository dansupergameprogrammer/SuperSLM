"""T-2202 (Poirot M1/O2): the pooled-gate notice's real print sites, exercised through the real
CLI subprocess chain -- not a monkeypatched `build_runtime_additive_sections`, not a source-text
occurrence count.

`test_sslm_convert_adapter_b3_diagnostic.py`'s own `test_quarantine_notice_is_printed_at_every_
pooled_gate_output_site` is a static wiring check (it counts source references to the notice
constant), by its own docstring's admission the ONLY cover the `--verbose` print site inside
`build_runtime_additive_sections` had -- that print fires before `main()`'s own accept/reject
branches are even reached, so `test_main_accept_path_prints_pooled_quarantine_notice` and
`test_main_reject_path_prints_pooled_quarantine_notice` (both of which monkeypatch `build_
runtime_additive_sections` itself away) never exercise it. And O2 found the accept branch's own
print block had never run against a real conversion at all -- only against a hand-built
three-key dict.

This file closes both gaps with the SAME construction: a real, on-disk base checkpoint (`tools/
_calibrate_checkpoint_fixture.py`) calibrated and converted via the real `calibrate_checkpoint.py`
and `convert_model.py` CLIs, then a real, on-disk BF16 PEFT LoRA adapter (`tools/_t2194_bf16_
lora_fixture.py`, already used by `test_sslm_convert_adapter_bf16.py`) converted via the real
`sslm_convert_adapter.py` CLI -- every step a genuine subprocess (`sys.executable <script>.py
...`), no mocks, no monkeypatching. Two seeds against the identical fixture shapes reach the two
branches that matter:

- `seed=0` (this fixture module's own default): the pooled gate ACCEPTS (`accepted=True`,
  `margin_exceeded=False`), so `main()` takes `ArtifactOutcome.RUNTIME_ADDITIVE` and prints the
  notice from its own accept branch (closing O2) in addition to the `--verbose` line inside
  `build_runtime_additive_sections` (closing M1's accept-side gap).
- `seed=5`: the pooled gate REJECTS on `composed_mean` (`margin_exceeded=True`,
  `domain_trip=False` -- confirmed below, so this is the B3 margin branch and not the unrelated
  domain-rejection branch), so `main()` takes `RejectionBranch.RUNTIME_VS_BAKED_MARGIN_EXCEEDED`
  and prints the notice from its own reject branch (closing M1's reject-side gap, already
  covered for the mocked case by `test_main_reject_path_prints_pooled_quarantine_notice`, now
  also covered for a real conversion).

Both seeds were found by a direct sweep of this fixture's own `seed=0..29` against the real CLI
chain (recorded in the T-2202 build log) -- not reasoned from the arithmetic, since the B3
statistic's behavior on a 2-draw-per-partition rank-2 fixture is not something worth predicting
by construction (`StandardsDocument.md` §5.4: exactness is verified at source or by execution).
"""

import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(_TOOLS_DIR))
from _calibrate_checkpoint_fixture import build_fixture_checkpoint  # noqa: E402
from _t2194_bf16_lora_fixture import build_bf16_lora_fixture  # noqa: E402

import sslm_convert_adapter as A  # noqa: E402

# The notice's own text, quoted once here so a future rewrite of the constant does not silently
# desync this file's assertions from what the CLI actually prints -- every assertion below reads
# a substring straight from the live constant, never a hand-copied literal.
_NOTICE = A._B3_POOLED_GATE_QUARANTINE_NOTICE


def _run(cmd):
    return subprocess.run(cmd, cwd=str(_TOOLS_DIR), capture_output=True, text=True)


@pytest.fixture(scope="module")
def real_base_artifact(tmp_path_factory):
    """A real, on-disk `.sslm` base artifact, built once via the real `calibrate_checkpoint.py`
    and `convert_model.py` CLIs (both real subprocesses) against the tiny fixture checkpoint --
    shared read-only across every test in this module, since building it does not depend on the
    adapter under test."""
    tmp = tmp_path_factory.mktemp("b3_notice_e2e_base")
    checkpoint_dir = build_fixture_checkpoint(tmp / "checkpoint")
    calib_dir = tmp / "calib"
    r = _run([sys.executable, "calibrate_checkpoint.py",
             "--checkpoint", str(checkpoint_dir), "--out", str(calib_dir)])
    assert r.returncode == 0, f"calibrate_checkpoint.py failed:\n{r.stdout}\n{r.stderr}"

    base_sslm = tmp / "base.sslm"
    r = _run([sys.executable, "convert_model.py",
             "--artifact", str(calib_dir), "--out", str(base_sslm), "--skip-verify"])
    assert r.returncode == 0, f"convert_model.py failed:\n{r.stdout}\n{r.stderr}"

    return checkpoint_dir, base_sslm


def _convert_adapter(tmp_path, real_base_artifact, seed):
    checkpoint_dir, base_sslm = real_base_artifact
    adapter_dir = build_bf16_lora_fixture(tmp_path / "adapter",
                                          base_model_name_or_path=str(checkpoint_dir), seed=seed)
    out_sslm = tmp_path / "adapter_out.sslm"
    r = _run([sys.executable, "sslm_convert_adapter.py",
             "--adapter", str(adapter_dir), "--base", str(base_sslm),
             "--out", str(out_sslm), "--skip-verify"])
    return r, out_sslm


def test_real_conversion_accepts_and_prints_the_final_notice_at_both_its_sites(
    tmp_path, real_base_artifact,
):
    """seed=0: a real conversion through the real CLI subprocess chain that ACCEPTS -- closing
    O2 (the accept branch's print block had never run against a real conversion) and M1's
    accept-side gap (the `--verbose` print site inside `build_runtime_additive_sections`, whose
    only prior cover was a source-text reference count)."""
    r, out_sslm = _convert_adapter(tmp_path, real_base_artifact, seed=0)

    assert r.returncode == 0, f"expected the ACCEPT branch (rc=0):\n{r.stdout}\n{r.stderr}"
    assert out_sslm.is_file(), "the accept branch must have written the runtime-additive artifact"
    assert "domain_trip=False" not in r.stderr  # no rejection branch fired at all on stderr
    assert "REJECTED" not in r.stderr

    # The verbose print site inside `build_runtime_additive_sections` fires before `main()`'s own
    # dispatch and always prints to stdout -- present regardless of the branch `main()` later
    # takes. `main()`'s own accept branch prints the notice a second time, immediately after
    # `wrote <path>`. Both are real prints from a real process, not `capsys` on a mocked call.
    assert r.stdout.count(_NOTICE) == 2, (
        f"expected the notice twice on stdout (the --verbose line, then main()'s accept branch); "
        f"got {r.stdout.count(_NOTICE)}. Full stdout:\n{r.stdout}"
    )
    assert "pooled B3 gate: accepted=True" in r.stdout


def test_real_conversion_rejects_on_composed_mean_and_prints_the_final_notice_at_both_its_sites(
    tmp_path, real_base_artifact,
):
    """seed=5: a real conversion through the real CLI subprocess chain that REJECTS via the B3
    pooled margin (never a domain trip) -- closing M1's reject-side gap for a real conversion
    (the mocked case was already covered)."""
    r, out_sslm = _convert_adapter(tmp_path, real_base_artifact, seed=5)

    assert r.returncode == 1, f"expected the REJECT branch (rc=1):\n{r.stdout}\n{r.stderr}"
    assert not out_sslm.exists(), "a rejected conversion must not leave an artifact on disk"
    assert "domain_trip=False margin_exceeded=True" in r.stderr, (
        f"expected the B3 margin branch specifically, not a domain rejection. Full stderr:\n"
        f"{r.stderr}"
    )
    assert "composed_mean_accepts=False" in r.stderr

    # The --verbose print site (inside build_runtime_additive_sections, before dispatch) prints
    # to stdout exactly as it does on the accept path; main()'s own reject branch prints the
    # notice a second time, to STDERR, beside the per-conjunct accepts.
    assert _NOTICE in r.stdout, f"expected the notice on stdout (--verbose site). stdout:\n{r.stdout}"
    assert _NOTICE in r.stderr, f"expected the notice on stderr (main() reject branch). stderr:\n{r.stderr}"
