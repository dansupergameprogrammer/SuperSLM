"""T-2851 row-10 token oracle and row-5 malformed-run matrix.

The reference and candidate executables are built from cell_real_decode.cpp
against independent v1.6.0 and candidate libraries. This runner records exact
token lists. It takes no timing measurements.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "identity"
ARTIFACTS = Path(r"D:\hf_cache\superslm_artifacts")
R05 = ARTIFACTS / "example" / "qwen2.5-0.5b-instruct-cap4096-aex-tok.sslm"
R15 = ARTIFACTS / "qwen2.5-1.5b-instruct-tok.sslm"
RD = Path(r"D:\_t2907\artifacts\qwen2.5-1.5b-instruct-dgc1-g5schema.sslm")
RU = HERE / "out" / "ru" / "ru.sslm"
SCHEMA = "potion_shop_order"


def execute(exe: Path, backend: str, model: Path, schema: str, count: int,
            tasks: int, device: bool, prompt="-", malformed="", damped=False,
            overflow=False, shader_dir: Path | None = None):
    args = [str(exe), backend, str(model), schema, str(count), str(tasks), str(int(device)), prompt]
    if malformed:
        args.append(malformed)
    env = dict(os.environ)
    env.pop("T2956_DAMPED", None)
    if damped:
        env["T2956_DAMPED"] = "1"
    env.pop("T2956_OVERFLOW", None)
    if overflow:
        env["T2956_OVERFLOW"] = "1"
    env.pop("T2956_SHADER_DIR", None)
    if shader_dir:
        env["T2956_SHADER_DIR"] = str(shader_dir)
    result = subprocess.run(args, cwd=exe.parent, text=True, capture_output=True,
                            env=env, timeout=240)
    if result.returncode:
        raise RuntimeError(f"{exe.name} {backend} {model.name} tasks={tasks} "
                           f"malformed={malformed}: exit {result.returncode}\n"
                           f"{result.stdout}{result.stderr}")
    lines = [x for x in result.stdout.splitlines() if x.startswith("TOKENS")]
    if len(lines) != 1:
        raise RuntimeError(f"{exe.name} emitted {len(lines)} token lines: {result.stdout}")
    tokens = [int(x) for x in lines[0].split()[1:]]
    if not tokens:
        raise RuntimeError(f"{exe.name} generated zero tokens")
    if not malformed and not overflow:
        census = [x for x in result.stdout.splitlines() if x.startswith("THREADS")]
        if len(census) != 1 or "same=1" not in census[0] or "changed_at_hook=0" not in census[0]:
            raise RuntimeError(f"thread census failed: {result.stdout}")
        if tasks > 1 and not device and "hook_calls=0 " in census[0]:
            raise RuntimeError(f"hook installed but not invoked: {result.stdout}")
    return tokens, result.stdout


def run(base: Path, candidate: Path, malformed: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("R05", R05, "-", 64, "-", False),
        ("R15", R15, "-", 64, "-", False),
        ("R05_schema", R05, SCHEMA, 64, "-", False),
        ("RD_damped", RD, "-", 40, "-", True),
        ("RD_damped_schema", RD, "shopkeeper_intent_extraction", 40, "-", True),
        ("RU", RU, "-", 16, "0,1,2", False),
    ]
    records = []
    for name, path, schema, count, prompt, damped in scenarios:
        if not path.is_file():
            raise FileNotFoundError(path)
        backends = ["cpu"] if damped else ["cpu", "gpu"]
        for backend in backends:
            baseline, base_log = execute(base, backend, path, schema, count, 0, False,
                                         prompt, damped=damped)
            (OUT / f"{name}_{backend}_base.txt").write_text(base_log)
            for tasks in (0, 2, 4, 8, 16):
                got, log = execute(candidate, backend, path, schema, count, tasks,
                                   False, prompt, damped=damped)
                (OUT / f"{name}_{backend}_T{tasks}.txt").write_text(log)
                if got != baseline:
                    raise AssertionError(f"{name}/{backend}/T{tasks}: {got} != {baseline}")
                records.append((name, backend, tasks, 0, len(got)))
            if backend == "gpu" and name in ("R05", "R15", "R05_schema", "RU"):
                got, log = execute(candidate, backend, path, schema, count, 0, True, prompt)
                (OUT / f"{name}_gpu_device.txt").write_text(log)
                if got != baseline:
                    raise AssertionError(f"{name}/device: {got} != {baseline}")
                records.append((name, backend, 0, 1, len(got)))
            if name == "R05_schema" and len(baseline) != 29:
                raise AssertionError(f"schema generated {len(baseline)} tokens, expected 29")
    if malformed:
        cases = ("omit0", "omitlast", "dupinplace", "dupextra", "concurrentdup", "outofrange")
        for backend in ("cpu", "gpu"):
            baseline, _ = execute(base, backend, R05, "-", 1, 0, False)
            for tasks in (3, 4):
                for case in cases:
                    got, log = execute(candidate, backend, R05, "-", 1, tasks, False,
                                       malformed=case)
                    (OUT / f"malformed_{backend}_{tasks}_{case}.txt").write_text(log)
                    if got != baseline:
                        raise AssertionError(f"malformed {backend}/{tasks}/{case}: {got} != {baseline}")
                    records.append((case, backend, tasks, 0, len(got)))
    baseline, _ = execute(base, "gpu", R05, "-", 1, 0, False)
    got, log = execute(candidate, "gpu", R05, "-", 1, 0, True, overflow=True)
    (OUT / "device_overflow_retry.txt").write_text(log)
    if got != baseline:
        raise AssertionError(f"overflow retry: {got} != {baseline}")
    records.append(("overflow_retry", "gpu", 0, 1, len(got)))
    source_shaders = candidate.parent / "shaders"
    if not (source_shaders / "logits_site.cso").is_file():
        raise FileNotFoundError(source_shaders / "logits_site.cso")
    override = OUT / "override_shaders"
    override.mkdir(exist_ok=True)
    for shader in source_shaders.glob("*.cso"):
        shutil.copy2(shader, override / shader.name)
    isolated = OUT / "override_bin"
    isolated.mkdir(exist_ok=True)
    isolated_exe = isolated / candidate.name
    shutil.copy2(candidate, isolated_exe)
    got, log = execute(isolated_exe, "gpu", R05, "-", 1, 0, True,
                       shader_dir=override.resolve())
    (OUT / "device_shader_override.txt").write_text(log)
    if got != baseline:
        raise AssertionError(f"shader override: {got} != {baseline}")
    records.append(("shader_override", "gpu", 0, 1, len(got)))
    (OUT / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"PASS identity cases={len(records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--malformed", action="store_true")
    args = parser.parse_args()
    run(args.base.resolve(), args.candidate.resolve(), args.malformed)
