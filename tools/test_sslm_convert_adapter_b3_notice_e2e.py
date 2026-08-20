"""T-2213 (D-SLM3783): the B3 pooled-report notice's real print sites, exercised through the real
CLI subprocess chain -- not a monkeypatched `build_runtime_additive_sections`, not a source-text
occurrence count.

`test_sslm_convert_adapter_b3_diagnostic.py`'s own `test_status_notice_is_printed_at_every_
pooled_report_output_site` is a static wiring check (it counts source references to the notice
constant); this file exercises the notice's real print sites end to end, against a real, on-disk
base checkpoint (`tools/_calibrate_checkpoint_fixture.py`) calibrated and converted via the real
`calibrate_checkpoint.py` and `convert_model.py` CLIs, then a real, on-disk BF16 PEFT LoRA
adapter (`tools/_t2194_bf16_lora_fixture.py`, already used by `test_sslm_convert_adapter_bf16.py`)
converted via the real `sslm_convert_adapter.py` CLI -- every step a genuine subprocess
(`sys.executable <script>.py ...`), no mocks, no monkeypatching.

T-2213 retired the B3 pooled ACCEPT/REJECT gate (`Claude/Loki/t2205/t2207/t2209/t2210/t2211`,
`Claude/Vitruvius/t2204`'s own fold round 4, Wizard repo; Dan's ruling D-SLM3783): nothing B3
computes can refuse to write an artifact any more, so every seed this fixture can produce now
reaches `main()`'s ACCEPT branch on B3 grounds (a domain trip remains the only structural
refusal, and this fixture's own fixed small geometry never trips it). `seed=0` (this fixture
module's own default) is used below; the retired suite's own `seed=7` (previously pinned as a
REJECT-branch cell, `Claude/Poirot/4299d84-t2206-b3-round2-confirmation.md` M2) is confirmed
below to now reach the ACCEPT branch too, closing the loop on the retirement rather than leaving
a stale reject-path cell in this file.
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
_NOTICE = A._B3_POOLED_GATE_STATUS_NOTICE


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


def _convert_adapter(tmp_path, real_base_artifact, seed, reference_delta_norm=None):
    checkpoint_dir, base_sslm = real_base_artifact
    adapter_dir = build_bf16_lora_fixture(tmp_path / "adapter",
                                          base_model_name_or_path=str(checkpoint_dir), seed=seed)
    out_sslm = tmp_path / "adapter_out.sslm"
    cmd = [sys.executable, "sslm_convert_adapter.py",
          "--adapter", str(adapter_dir), "--base", str(base_sslm),
          "--out", str(out_sslm), "--skip-verify"]
    if reference_delta_norm is not None:
        cmd += ["--reference-delta-norm", str(reference_delta_norm)]
    r = _run(cmd)
    return r, out_sslm


@pytest.mark.parametrize("seed", [0, 7])
def test_real_conversion_accepts_and_prints_the_final_notice_at_both_its_sites(
    tmp_path, real_base_artifact, seed,
):
    """A real conversion through the real CLI subprocess chain, with no `--reference-delta-norm`
    configured: T-2213 retired the B3 pooled ACCEPT/REJECT gate, so B3 can no longer refuse to
    write an artifact -- both `seed=0` (this fixture's own default) and `seed=7` (the retired
    suite's own pinned REJECT-branch seed, `Claude/Poirot/4299d84-t2206-b3-round2-confirmation.md`
    M2) now reach the ACCEPT branch. This confirms the retirement closed the reject path rather
    than leaving a stale cell asserting a rejection that can no longer happen."""
    r, out_sslm = _convert_adapter(tmp_path, real_base_artifact, seed=seed)

    assert r.returncode == 0, f"expected the ACCEPT branch (rc=0):\n{r.stdout}\n{r.stderr}"
    assert out_sslm.is_file(), "the accept branch must have written the runtime-additive artifact"
    assert "REJECTED" not in r.stderr

    # The verbose print site inside `build_runtime_additive_sections` fires before `main()`'s own
    # dispatch and always prints to stdout -- present regardless of the branch `main()` later
    # takes. `main()`'s own accept branch prints the notice a second time, immediately after
    # `wrote <path>`. Both are real prints from a real process, not `capsys` on a mocked call.
    assert r.stdout.count(_NOTICE) == 2, (
        f"expected the notice twice on stdout (the --verbose line, then main()'s accept branch); "
        f"got {r.stdout.count(_NOTICE)}. Full stdout:\n{r.stdout}"
    )
    assert "accepted=" not in r.stdout, "the retired pooled verdict must not reappear"
    assert "pooled B3 report: n_pairs=" in r.stdout
    assert "delta_norm=" in r.stdout
    assert "MAGNITUDE WARNING" not in r.stdout, "no --reference-delta-norm was passed; none must warn"

    # This fixture's own pooled run flags zero pairs (n_pairs=1, 0 flagged) -- the empty case
    # must print an explicit statement, not silence (T-2201/T-2206's own remedy, unaffected by
    # the T-2213 retirement).
    assert "0 pair(s) flagged for review" in r.stdout, (
        f"expected the empty-flag case to print explicitly, not silently. Full stdout:\n{r.stdout}"
    )
    assert "not evidence this adapter is sound" in r.stdout


def test_real_conversion_with_a_far_reference_still_accepts_and_prints_the_magnitude_warning(
    tmp_path, real_base_artifact,
):
    """T-2213: `--reference-delta-norm` wired through the real CLI, end to end, against a real
    conversion. A reference far below this fixture's own real `delta_norm` (this tiny CI fixture's
    own scale is unrelated to the reference figure -- the point of this cell is to prove the flag
    reaches `run_b3_pooled_report` and the warning prints, not to reproduce a specific ratio)
    trips the wide-tolerance sanity band. The artifact is still written (rc=0): the warning is
    UNRESOLVED, never a REJECT."""
    r, out_sslm = _convert_adapter(tmp_path, real_base_artifact, seed=0, reference_delta_norm=1e-9)

    assert r.returncode == 0, f"a magnitude warning must never refuse the artifact:\n{r.stdout}\n{r.stderr}"
    assert out_sslm.is_file()
    assert "MAGNITUDE WARNING" in r.stdout, f"expected a magnitude warning. stdout:\n{r.stdout}"
    assert "unresolved" in r.stdout.lower()
    assert "never a REJECT" in r.stdout
