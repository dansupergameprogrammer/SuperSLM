"""T-1872 portable probe -- orchestrator.

Runs each of the four questions as its OWN subprocess, in order, each under a
hard wall-clock timeout. This is deliberate: a hang inside the HIP runtime or a
CUDA/HIP kernel call is not something Python's own signal handling can reliably
interrupt from inside the same process (a stuck driver call blocks the whole
interpreter, including any timer meant to cancel it). Running each stage as a
child process means the orchestrator -- which never touches the GPU itself --
stays responsive no matter what the child does, and `subprocess.run(timeout=...)`
kills a child that outlives its budget rather than leaving Dan staring at a
frozen window.

One entry point for Dan: run_probe.bat calls this file with the bundle's own
portable Python. Nothing here reads or writes anything outside the bundle
directory itself.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = PROBE_DIR.parent
PYTHON_EXE = BUNDLE_ROOT / "python" / "python.exe"
RESULTS_DIR = BUNDLE_ROOT / "results"

# (stage module, human question, timeout seconds)
STAGES = [
    ("stage_import", "Q1: does PyTorch import and see the device as gfx1100?", 60),
    ("stage_bf16", "Q2: does bf16 work, natively or at all?", 60),
    ("stage_forward", "Q3: one real forward over one real prompt -- output sane?", 600),
    ("stage_throughput", "Q4: measured throughput (s/doc), matching the design's cost table", 900),
]


def run_stage(module: str, timeout_s: int) -> tuple[bool, dict, str]:
    """Runs one stage as a subprocess. Returns (passed, result_dict, full_stdout+stderr)."""
    env = os.environ.copy()
    # Help the ROCm "wheel variant" provider pick the right architecture/runtime
    # if auto-detection is not conclusive. Process-local only -- never written to
    # the machine's persistent environment, and has no effect at all if the
    # provider does not look at it.
    env.setdefault("AMD_VARIANT_PROVIDER_FORCE_GFX_ARCH", "gfx1100")
    env.setdefault("AMD_VARIANT_PROVIDER_FORCE_ROCM_VERSION", "7.2.1")
    # NOTE: the embeddable Python's python312._pth file puts the interpreter in
    # isolated path mode, which ignores PYTHONPATH and does not add a script's
    # own directory to sys.path the way a normal install does. That is why
    # "import common" works from a stage script at all: the assembly script
    # (assemble_bundle.ps1) adds a "..\probe" line to python312._pth itself,
    # not an environment variable set here. Confirmed by execution: setting
    # PYTHONPATH here alone does not make the import succeed under this
    # interpreter's isolated mode.

    cmd = [str(PYTHON_EXE), f"{module}.py"]
    print(f"\n{'='*70}\nRunning {module} (timeout {timeout_s}s)\n{'='*70}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROBE_DIR), env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        elapsed = time.time() - t0
        combined = proc.stdout + proc.stderr
        print(combined)
        result = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("RESULT_JSON:"):
                try:
                    result = json.loads(line[len("RESULT_JSON:"):])
                except json.JSONDecodeError:
                    result = None
                break
        if result is None:
            result = {
                "stage": module, "pass": False,
                "detail": {"reason": "stage process exited without printing a "
                                      "parseable RESULT_JSON line",
                           "exit_code": proc.returncode},
            }
        result["elapsed_seconds"] = elapsed
        result["exit_code"] = proc.returncode
        passed = bool(result.get("pass")) and proc.returncode == 0
        return passed, result, combined
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        combined = (e.stdout or "") + (e.stderr or "")
        print(combined)
        print(f"\n*** {module} TIMED OUT after {timeout_s}s -- process killed ***")
        result = {
            "stage": module, "pass": False,
            "detail": {"reason": f"stage exceeded its {timeout_s}s timeout and was killed; "
                                  "this counts as a FAIL, not a hang -- see README"},
            "elapsed_seconds": elapsed, "exit_code": None,
        }
        return False, result, combined


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "full_log.txt"
    results_json_path = RESULTS_DIR / "results.json"
    results_txt_path = BUNDLE_ROOT / "RESULTS.txt"

    all_results = []
    all_logs = []
    overall_start = time.time()

    print("T-1872 portable feasibility probe")
    print(f"Bundle root: {BUNDLE_ROOT}")
    print(f"Python: {PYTHON_EXE}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for module, question, timeout_s in STAGES:
        passed, result, combined_log = run_stage(module, timeout_s)
        result["question"] = question
        all_results.append(result)
        all_logs.append(f"\n\n===== {module} =====\n{combined_log}")
        status = "PASS" if passed else "FAIL"
        print(f"\n>>> {module}: {status}\n")
        # A hard failure at stage 1 or 2 means every later stage would fail for
        # the same reason (no device) -- stop instead of burning the remaining
        # timeouts to reach the same answer twice.
        if not passed and module == "stage_import":
            print("stage_import failed -- no device is usable; skipping remaining "
                  "stages rather than waiting out their timeouts for a foregone result.")
            for skipped_module, skipped_question, _ in STAGES[len(all_results):]:
                all_results.append({
                    "stage": skipped_module, "question": skipped_question,
                    "pass": False, "detail": {"reason": "skipped -- stage_import failed"},
                    "elapsed_seconds": 0, "exit_code": None,
                })
            break

    total_elapsed = time.time() - overall_start

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_logs))

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "total_elapsed_seconds": total_elapsed}, f, indent=2)

    lines = []
    lines.append("T-1872 PORTABLE PROBE -- RESULTS")
    lines.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total wall time: {total_elapsed:.1f}s")
    lines.append("")
    for r in all_results:
        status = "PASS" if r.get("pass") else "FAIL"
        lines.append(f"[{status}] {r['stage']} -- {r.get('question', '')}")
        detail = r.get("detail", {})
        if not r.get("pass"):
            reason = detail.get("reason", "(no reason recorded)")
            lines.append(f"       reason: {reason}")
            if "exception" in detail:
                lines.append(f"       exception: {detail['exception']}")
        else:
            if r["stage"] == "stage_import":
                lines.append(f"       device: {detail.get('device_name')} "
                              f"(gcnArchName={detail.get('gcn_arch_name')}, "
                              f"gfx1100_confirmed={detail.get('gfx1100_confirmed')})")
            elif r["stage"] == "stage_bf16":
                lines.append(f"       native bf16 reported: "
                             f"{detail.get('torch_reports_native_bf16')}; "
                             f"matmul ran and finite: {detail.get('result_finite')}")
            elif r["stage"] == "stage_forward":
                lines.append(f"       prompt: {detail.get('prompt_text')!r}")
                lines.append(f"       generated: {detail.get('generated_text')!r}")
                lines.append(f"       model load: {detail.get('model_load_seconds', 0):.1f}s, "
                             f"generation: {detail.get('generation_seconds', 0):.2f}s")
            elif r["stage"] == "stage_throughput":
                for dtype_name, d in detail.get("by_dtype", {}).items():
                    if "mean_s_per_doc" in d:
                        lines.append(f"       {dtype_name}: {d['mean_s_per_doc']:.4f} s/doc "
                                     f"(median {d['median_s_per_doc']:.4f}, "
                                     f"stdev {d['stdev_s_per_doc']:.4f}, "
                                     f"n={d['n_timed']})")
                    else:
                        lines.append(f"       {dtype_name}: FAILED -- {d.get('error')}")
                lines.append("       compare directly against T-1870's own cited lower-bound "
                             "rate: 3.30 s/doc bf16, 3.02 s/doc fp32 (RTX 2080 Super, T-1777)")
        lines.append("")

    n_pass = sum(1 for r in all_results if r.get("pass"))
    lines.append(f"SUMMARY: {n_pass}/{len(all_results)} stages passed.")
    lines.append("")
    lines.append(f"Full stage-by-stage console output: results/full_log.txt")
    lines.append(f"Machine-readable results: results/results.json")

    text = "\n".join(lines)
    print("\n" + text)
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
