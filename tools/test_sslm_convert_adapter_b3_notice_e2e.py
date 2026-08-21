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

import numpy as np
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


# --- Checkpoint round-trip -- fix for D-SLM3787 finding C1 --------------------------------------
# `_b3_collect_pair_raw_draws` used to return `delta_norm_sq` as a plain Python `float`, where
# every other value in its dict is an `np.ndarray`. `build_runtime_additive_sections`'s checkpoint
# writer serializes the whole dict with `{k: v.tolist() for k, v in raw.items()}` -- a bare
# `float` has no `.tolist()`, so any call with `checkpoint_path` set raised `AttributeError` on
# the first non-tripping pair. Neither this suite nor the one before this fix round ever set
# `checkpoint_path` at all -- this is the cell that closes that gap, calling
# `build_runtime_additive_sections` directly (not through the CLI, which does not expose
# `checkpoint_path`) against the real fixture checkpoint and a real bf16 adapter.

def test_checkpoint_path_round_trips_a_real_conversion_and_resumes_with_the_same_pooled_delta_norm(
    tmp_path, real_base_artifact,
):
    checkpoint_dir, base_sslm = real_base_artifact
    adapter_dir = build_bf16_lora_fixture(tmp_path / "adapter",
                                          base_model_name_or_path=str(checkpoint_dir), seed=0)
    ckpt_path = tmp_path / "b3_raw_checkpoint.jsonl"

    # First pass: no checkpoint file exists yet, so this call must WRITE it -- the exact path
    # C1's `AttributeError` fired on, before this fix round.
    _sections1, verdict1, _rt1 = A.build_runtime_additive_sections(
        adapter_dir, base_sslm, verbose=False, checkpoint_path=ckpt_path)
    assert ckpt_path.is_file(), "checkpoint_path must be written on a run that has no checkpoint yet"
    assert verdict1["pooled"] is not None
    delta_norm_first_pass = verdict1["pooled"]["delta_norm"]
    assert delta_norm_first_pass > 0.0

    # Second pass: the checkpoint file from the first pass exists, so every pair's raw draws are
    # read back from disk (`saved_draws`) rather than recomputed -- the resume path C1 also broke
    # (any resumed pair's `delta_norm_sq` came back through the SAME serialize/deserialize round
    # trip). The pooled `delta_norm` must match the first pass exactly: same adapter, same base.
    _sections2, verdict2, _rt2 = A.build_runtime_additive_sections(
        adapter_dir, base_sslm, verbose=False, checkpoint_path=ckpt_path)
    assert verdict2["pooled"]["delta_norm"] == pytest.approx(delta_norm_first_pass, rel=1e-12), (
        "a resumed run's pooled delta_norm must round-trip exactly through the checkpoint file"
    )


def test_resuming_a_pre_t2213_checkpoint_recomputes_delta_norm_sq_instead_of_reading_zero(
    tmp_path, real_base_artifact,
):
    """D-SLM3787 finding S1: `raw.get("delta_norm_sq", 0.0)` silently turned a pre-T-2213
    checkpoint file's missing key into `delta_norm=0` for every resumed pair. A checkpoint file in
    the OLD format (no `delta_norm_sq` key at all, mirroring every `_t2065_raw_checkpoint.jsonl`
    written before this ticket) is written here by hand; resuming from it must report the SAME
    pooled `delta_norm` a fresh, non-resumed run of the identical adapter reports -- never 0."""
    import json

    checkpoint_dir, base_sslm = real_base_artifact
    adapter_dir = build_bf16_lora_fixture(tmp_path / "adapter",
                                          base_model_name_or_path=str(checkpoint_dir), seed=0)

    # The ground truth: a fresh run with no checkpoint at all.
    _sections_fresh, verdict_fresh, _rt = A.build_runtime_additive_sections(
        adapter_dir, base_sslm, verbose=False, checkpoint_path=None)
    expected_delta_norm = verdict_fresh["pooled"]["delta_norm"]
    assert expected_delta_norm > 0.0

    # A hand-written OLD-FORMAT checkpoint: real draws for the fixture's one pair
    # ("layer0.q_proj"), `delta_norm_sq` deliberately omitted from the "draws" dict -- exactly the
    # shape a checkpoint written before T-2213 would have, recomputed here from the pair's own
    # real A/B rather than reusing `verdict_fresh`'s own already-pooled figures.
    old_ckpt = tmp_path / "old_format_checkpoint.jsonl"
    a_f, b_scaled = A.read_peft_lora_pair(adapter_dir, 0, "q_proj", A.load_adapter_meta(adapter_dir))
    w_f = A.read_base_projection_weight(A._load_spike()._open_checkpoint_tensors(checkpoint_dir), 0, "q_proj")
    pipeline = A._load_spike()
    Wc, w_scales = pipeline.quantize_weight_per_channel(w_f, output_axis=0)
    w = np.asarray(w_scales, dtype=np.float64)
    S = float(np.max(w))
    Ac, alpha_scales = pipeline.quantize_weight_per_channel(a_f, output_axis=0)
    alpha = np.asarray(alpha_scales, dtype=np.float64)
    Bc, beta_scales = pipeline.quantize_weight_per_channel(b_scaled, output_axis=0)
    beta = np.asarray(beta_scales, dtype=np.float64)
    g = np.random.default_rng(0xC0FFEE)
    xf = g.standard_normal(a_f.shape[1])
    xmax = float(np.max(np.abs(xf)))
    X = xmax / 127.0 if xmax > 0.0 else 1.0
    xc = np.clip(np.round(xf / X), -127, 127).astype(np.int64)
    u_acc = Ac.astype(np.int64) @ xc
    T_value = A.compute_t(list(alpha), list(np.abs(u_acc)))
    raw = A._b3_collect_pair_raw_draws(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_value)
    old_format_draws = {k: v.tolist() for k, v in raw.items() if k != "delta_norm_sq"}
    with open(old_ckpt, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": "layer0.q_proj", "draws": old_format_draws}) + "\n")

    _sections_resumed, verdict_resumed, _rt3 = A.build_runtime_additive_sections(
        adapter_dir, base_sslm, verbose=False, checkpoint_path=old_ckpt)
    assert verdict_resumed["pooled"]["delta_norm"] == pytest.approx(expected_delta_norm, rel=1e-9), (
        "resuming a pre-T-2213 checkpoint (missing delta_norm_sq) must recompute the real value, "
        "never silently report 0"
    )
    assert verdict_resumed["pooled"]["delta_norm"] != 0.0
