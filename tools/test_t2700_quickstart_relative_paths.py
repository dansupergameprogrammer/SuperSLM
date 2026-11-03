"""Engine 1.19: the quickstart's QK-norm flow resolves relative --capture/--verifier paths
against the caller's working directory (red suite).

The command under test is docs/quickstart.md step 1, as typed:

    python tools/t2700_fused_k_calibration.py flow --checkpoint ... --out ... --work ...
        --capture <compiled t2700_fused_k_capture> --verifier <compiled sslm_verify>

Every cell runs the real CLI entry point (`main()` -> argparse -> `flow` -> A/B/C) in a
caller-chosen working directory. Two things are stood in, neither on the path-handling
surface: `prepare` (the 600-record corpus tokenization, which needs the real Qwen tokenizer's
453-token shared prefix) writes the same file layout from a fixture, and `pipeline.load_model`
returns a model loaded once from the same on-disk fixture checkpoint (it is the 24 s
calibration, not path handling; the flow still persists and reopens the real checkpoint path).

The capture and verifier are small executables that log the absolute path they were
launched from and do their job (a well-formed capture report; a verifier manifest that
accepts). A decoy of the same name sits first on PATH and logs itself and exits non-zero, so
a relative path resolved anywhere but the caller's working directory is visible as WHICH
binary ran, not only as a crash.

Red today (Linux, measured 2026-11-03): `--capture ./<name>` run from the binary's own
directory. `_run_capture` rebuilds the argument as `str(Path(capture))`, and pathlib drops
the leading `./`, so the explicit cwd-relative path becomes a bare name and the OS looks it up
on PATH: the decoy runs (or, with no decoy, FileNotFoundError). `--verifier` is passed through
as the raw string and resolves correctly in every form. The `dir/name` and `./dir/name` forms
are green on Linux today for both flags; they are kept as the pins the contract asks for, and
they are the forms a Windows caller types. Whether they are red on Windows was not executed
from here (no Windows host) -- see the test-design record.

Test-design record: /mnt/project-files/plans/superslm-array-converter/records/
curie-qwen3-red-suite.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "reference_pipeline"))

import t2700_fused_k_calibration as calibration  # noqa: E402
import _calibrate_checkpoint_fixture as fixture_mod  # noqa: E402
from reference_pipeline import pipeline  # noqa: E402

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE = 8, 1, 2, 1, 128, 16
CAPTURE = "t2700_fused_k_capture"
VERIFIER = "sslm_verify"
_WINDOWS = os.name == "nt"


# ==============================================================================
# Fixture checkpoint (QK-norm, head_dim 128: the fused-QK converter refuses any other)
# ==============================================================================


def _write_qk_checkpoint(checkpoint_dir: Path) -> Path:
    rng = np.random.default_rng(0)

    def small(*shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    tensors = {"model.embed_tokens.weight": small(fixture_mod.VOCAB_SIZE, HIDDEN),
               "model.norm.weight": np.ones(HIDDEN, dtype=np.float32)}
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = small(HEADS * HEAD_DIM, HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = small(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = small(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = small(HIDDEN, HEADS * HEAD_DIM)
        tensors[f"{p}.self_attn.q_norm.weight"] = (0.5 + rng.random(HEAD_DIM)).astype(np.float32)
        tensors[f"{p}.self_attn.k_norm.weight"] = (0.5 + rng.random(HEAD_DIM)).astype(np.float32)
        tensors[f"{p}.mlp.gate_proj.weight"] = small(INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = small(INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = small(HIDDEN, INTERMEDIATE)
    checkpoint_dir.mkdir(parents=True)
    fixture_mod._write_safetensors_shard(checkpoint_dir, tensors)
    config = {"hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
              "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
              "intermediate_size": INTERMEDIATE, "vocab_size": fixture_mod.VOCAB_SIZE,
              "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
              "max_position_embeddings": fixture_mod.MAX_POSITION_EMBEDDINGS}
    (checkpoint_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    fixture_mod._write_tokenizer(checkpoint_dir)
    return checkpoint_dir


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """One checkpoint on disk and the model `load_model` builds from it, built once."""
    checkpoint = _write_qk_checkpoint(tmp_path_factory.mktemp("qk") / "checkpoint")
    return checkpoint, pipeline.load_model(checkpoint)


# ==============================================================================
# Stand-in executables: the real pair (logs, then works) and the PATH decoy (logs, fails)
# ==============================================================================


_CAPTURE_BODY = """
import math, sys
from pathlib import Path
m, e = 2130706432, -54
lines = ['T2700_FUSED_K_CAPTURE_V1', 'summary\\tcallback_count\\t1000000',
         'summary\\tlanding_saturation_count\\t0',
         'layer\\thead\\tchannel\\traw_abs_peak\\twide_scale_m\\twide_scale_e\\treal_peak\\tlanding_saturation_count']
for layer in range({layers}):
    for head in range({kv_heads}):
        for channel in range({head_dim}):
            lines.append(f'{{layer}}\\t{{head}}\\t{{channel}}\\t1\\t{{m}}\\t{{e}}\\t{{math.ldexp(float(m), e).hex()}}\\t0')
Path(sys.argv[3]).write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
"""

_VERIFIER_BODY = """
import json, sys
with open(sys.argv[2], 'w', encoding='utf-8') as handle:
    json.dump({'status': 'OK', 'config_geometry': {'ok': True}}, handle)
"""


def _log_prefix(role):
    return f"""import os, sys
from pathlib import Path
with open(os.environ['T2700_REL_LOG'], 'a', encoding='utf-8') as log:
    log.write({role!r} + '\\t' + str(Path(sys.argv[0]).resolve()) + '\\n')
"""


def _executable(directory: Path, name: str, role: str, body: str) -> Path:
    """An executable `directory/name` (POSIX: a shebang script; Windows: `name.cmd` wrapping a
    sibling `.py`, since CreateProcess appends only `.exe`). Returns the path a caller types."""
    directory.mkdir(parents=True, exist_ok=True)
    source = _log_prefix(role) + body
    if _WINDOWS:
        script = directory / f"{name}-impl.py"
        script.write_text(source, encoding="utf-8")
        wrapper = directory / f"{name}.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return wrapper
    path = directory / name
    # argv[0] for a shebang script is the script path the OS resolved, which is what the log
    # needs: the binary that actually ran.
    path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _install(tmp_path: Path, monkeypatch, model):
    """Real capture/verifier under `tmp_path/caller/bin`, decoys first on PATH."""
    cfg = model.config
    capture_body = _CAPTURE_BODY.format(layers=cfg.num_hidden_layers,
                                        kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim)
    bin_dir = tmp_path / "caller" / "bin"
    real = {"capture": _executable(bin_dir, CAPTURE, "capture", capture_body),
            "verifier": _executable(bin_dir, VERIFIER, "verifier", _VERIFIER_BODY)}
    decoy_dir = tmp_path / "decoy-on-path"
    fail = "\nsys.exit(97)\n"
    _executable(decoy_dir, CAPTURE, "DECOY-capture", fail)
    _executable(decoy_dir, VERIFIER, "DECOY-verifier", fail)
    log = tmp_path / "launches.tsv"
    log.write_text("", encoding="utf-8")
    monkeypatch.setenv("T2700_REL_LOG", str(log))
    monkeypatch.setenv("PATH", str(decoy_dir) + os.pathsep + os.environ.get("PATH", ""))
    return real, log


def _stand_in_prepare_and_load(monkeypatch, checkpoint, model):
    def prepare(args):
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prefix-453.txt").write_text("1\n", encoding="utf-8")
        for index in range(600):
            (out / f"suffix-{index:03d}.txt").write_text("1\n", encoding="utf-8")

    def load_model(path, *args, **kwargs):
        assert Path(path).resolve() == checkpoint.resolve(), path
        return model

    monkeypatch.setattr(calibration, "prepare", prepare)
    monkeypatch.setattr(calibration.pipeline, "load_model", load_model)


def _run_flow(monkeypatch, cwd: Path, argv, log: Path):
    """Run the CLI; if it raises, report a PATH-decoy launch first -- that is the reason a
    mis-resolved path fails, and the raw CalledProcessError hides it."""
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "argv", ["t2700_fused_k_calibration.py", "flow", *argv])
    try:
        calibration.main()
    except Exception as error:
        decoys = [entry for entry in _launches(log) if entry[0].startswith("DECOY")]
        assert not decoys, (
            f"a relative path resolved on PATH, not in the caller's directory "
            f"({cwd}): {decoys}; the flow then raised {type(error).__name__}")
        raise


def _launches(log: Path):
    return [tuple(line.split("\t")) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _assert_flow_used(log: Path, real, out: Path):
    launches = _launches(log)
    decoys = [entry for entry in launches if entry[0].startswith("DECOY")]
    assert not decoys, f"a relative path resolved on PATH, not in the caller's directory: {decoys}"
    for role, expected_count in (("capture", 3), ("verifier", 3)):
        ran = [Path(path) for logged_role, path in launches if logged_role == role]
        assert len(ran) == expected_count, (role, launches)
        expected = real[role].resolve()
        if _WINDOWS:
            # The log records the script that ran: the `.cmd` wrapper's sibling `-impl.py` (see _executable).
            expected = expected.with_name(f"{expected.stem}-impl.py")
        assert all(path == expected for path in ran), (role, ran)
    assert out.is_file(), "the flow did not publish its final artifact"
    assert json.loads((out.parent / "work" / "flow-report.json").read_text(encoding="utf-8"))


# ==============================================================================
# Cells
# ==============================================================================


_FORMS = {
    # (caller's working directory relative to tmp_path/caller, path as typed)
    "dir/name": ("", lambda name: f"bin/{name}"),
    "./dir/name": ("", lambda name: f"./bin/{name}"),
    "./name": ("bin", lambda name: f"./{name}"),
}


@pytest.mark.parametrize("flag", ["capture", "verifier"])
@pytest.mark.parametrize("form", sorted(_FORMS))
def test_flow_resolves_a_relative_executable_path_against_the_callers_directory(
        flag, form, tmp_path, monkeypatch, loaded):
    """FEAT. One flag relative (in the named form), the other absolute: the flow completes,
    and every launch of the relatively-named tool is the binary at <caller cwd>/<path> --
    three captures (A/B/C) and three verifier runs (provisional, after-A, final) -- and the
    same-named decoy on PATH never runs. Red today: `--capture ./name` (the decoy runs)."""
    checkpoint, model = loaded
    real, log = _install(tmp_path, monkeypatch, model)
    _stand_in_prepare_and_load(monkeypatch, checkpoint, model)
    cwd_suffix, typed = _FORMS[form]
    cwd = tmp_path / "caller" / cwd_suffix
    paths = {role: str(path) for role, path in real.items()}
    paths[flag] = typed(real[flag].name)
    out = tmp_path / "result" / "model.sslm"
    _run_flow(monkeypatch, cwd, [
        "--checkpoint", str(checkpoint), "--out", str(out), "--work", str(out.parent / "work"),
        "--capture", paths["capture"], "--verifier", paths["verifier"]], log)
    _assert_flow_used(log, real, out)


def test_flow_as_the_quickstart_types_it_with_every_path_relative(tmp_path, monkeypatch, loaded):
    """FEAT. docs/quickstart.md step 1 with every path relative to the caller's directory
    (`--checkpoint`, `--out`, `--work`, `--capture`, `--verifier`), the binaries under a
    `bin/` directory the caller names relatively, as the quickstart names `build/...`. Green on Linux
    today; pinned because it is the documented invocation."""
    checkpoint, model = loaded
    real, log = _install(tmp_path, monkeypatch, model)
    _stand_in_prepare_and_load(monkeypatch, checkpoint, model)
    caller = tmp_path / "caller"
    relative_checkpoint = os.path.relpath(checkpoint, caller)
    _run_flow(monkeypatch, caller, [
        "--checkpoint", relative_checkpoint, "--out", "result/model.sslm", "--work", "result/work",
        "--capture", f"bin/{real['capture'].name}", "--verifier", f"bin/{real['verifier'].name}"],
        log)
    _assert_flow_used(log, real, caller / "result" / "model.sslm")


# ==============================================================================
# Bare name (no directory part). Added by the builder on the release owner's instruction
# (2026-11-03): a bare name is the file of that name in the caller's working directory when
# one exists there, and is otherwise left to the OS's PATH lookup.
# ==============================================================================


@pytest.mark.parametrize("flag", ["capture", "verifier"])
def test_flow_runs_a_bare_name_from_the_callers_directory_when_the_file_is_there(
        flag, tmp_path, monkeypatch, loaded):
    """A bare `--<flag> <name>` typed in the directory that holds `<name>`: every launch is
    that file, and the same-named decoy first on PATH never runs."""
    checkpoint, model = loaded
    real, log = _install(tmp_path, monkeypatch, model)
    _stand_in_prepare_and_load(monkeypatch, checkpoint, model)
    paths = {role: str(path) for role, path in real.items()}
    paths[flag] = real[flag].name
    out = tmp_path / "result" / "model.sslm"
    _run_flow(monkeypatch, real[flag].parent, [
        "--checkpoint", str(checkpoint), "--out", str(out), "--work", str(out.parent / "work"),
        "--capture", paths["capture"], "--verifier", paths["verifier"]], log)
    _assert_flow_used(log, real, out)


@pytest.mark.parametrize("flag", ["capture", "verifier"])
def test_flow_leaves_a_bare_name_to_the_path_lookup_when_the_callers_directory_lacks_it(
        flag, tmp_path, monkeypatch, loaded):
    """A bare `--<flag> <name>` typed in a directory with no file of that name: the OS's PATH
    lookup decides. The working binary is put first on PATH (ahead of the decoy), so the flow
    completes only if the bare name reached the PATH lookup unchanged."""
    checkpoint, model = loaded
    real, log = _install(tmp_path, monkeypatch, model)
    _stand_in_prepare_and_load(monkeypatch, checkpoint, model)
    monkeypatch.setenv("PATH", str(real[flag].parent) + os.pathsep + os.environ["PATH"])
    paths = {role: str(path) for role, path in real.items()}
    paths[flag] = real[flag].name
    caller = tmp_path / "caller" / "elsewhere"
    caller.mkdir(parents=True)
    out = tmp_path / "result" / "model.sslm"
    _run_flow(monkeypatch, caller, [
        "--checkpoint", str(checkpoint), "--out", str(out), "--work", str(out.parent / "work"),
        "--capture", paths["capture"], "--verifier", paths["verifier"]], log)
    _assert_flow_used(log, real, out)
