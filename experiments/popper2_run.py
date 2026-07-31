# T-1421 independent-population driver.
#
# Walks the seeded shuffle of experiments/popper2_population.json, applies one
# mutation at a time to the shipped decode-path source, and measures three
# instruments against it:
#
#   I1  the T-1409 construction's discrimination matrix (solver harness
#       `mutate` mode -- the packet's own Experiment C protocol)
#   I2  the pinned golden decode row, as an oracle: does the DecodeLoopFixture
#       replica's baseline decode output change (harness `null` mode)
#   I3  the committed suite (build.bat's test binary): total failures, and
#       specifically the committed golden-pin CHECK lines
#
# Plus detectability: does either fixture's BASELINE decode output move at all.
#
# Every mutant is reverted after measurement. Results append to
# experiments/popper2_results.jsonl so the run is resumable.

import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out2")
RESULTS = os.path.join(ROOT, "experiments", "popper2_results.jsonl")

CL_FLAGS = ["/nologo", "/std:c++20", "/O2", "/W4", "/fp:precise", "/EHsc",
            "/Iinclude", "/Itests"]

SRCS = [
    "src/artifact.cpp", "src/sha256.cpp", "src/tokenizer.cpp", "src/model.cpp",
    "src/intmath.cpp", "src/silu_lut.cpp", "src/matmul.cpp",
    "src/proof_manifest.cpp", "src/trace_hook.cpp",
    "src/forward/checked_chain_funnel.cpp", "src/forward/forward_sites.cpp",
    "src/decode_digest.cpp",
]
HARNESS_MAIN = "experiments/decode_discrimination.cpp"
SUITE_MAIN = "tests/test_main.cpp"

HARNESS_DEFS = ["/DSOLVER_TRACE_WIRED"]
SUITE_DEFS = ["/DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION"]

# Committed CHECK lines in tests/test_main.cpp that compare a decode output
# against a value pinned from the DecodeLoopFixture geometry. Declared from a
# source sweep before any mutant was run.
GOLDEN_PIN_LINES = {14352, 14363, 14676, 14685, 14698, 14702}

VSDEVCMD = r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"


def msvc_env():
    bat = os.path.join(OUT, "_env.bat")
    os.makedirs(OUT, exist_ok=True)
    with open(bat, "w", encoding="ascii") as fh:
        fh.write(f'@echo off\r\ncall "{VSDEVCMD}" -arch=x64 -no_logo >nul\r\nset\r\n')
    p = subprocess.run([bat], capture_output=True, text=True, shell=False)
    env = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    if "INCLUDE" not in env:
        raise RuntimeError("VsDevCmd did not populate INCLUDE")
    return env


ENV = None
CL = "cl"


def find_cl(env):
    path = env.get("Path") or env.get("PATH") or ""
    for d in path.split(";"):
        cand = os.path.join(d, "cl.exe")
        if d and os.path.exists(cand):
            return cand
    raise RuntimeError("cl.exe not found on the VsDevCmd Path")


def obj_dir(tag):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    return d


def compile_tu(src, tag, defs, timeout=180):
    d = obj_dir(tag)
    obj = os.path.join(d, os.path.basename(src).replace(".cpp", ".obj"))
    cmd = [CL] + CL_FLAGS + defs + ["/c", src.replace("/", "\\"), "/Fo:" + obj]
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, obj, (p.stdout + p.stderr)


def link(tag, exe, timeout=180):
    d = obj_dir(tag)
    objs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".obj")]
    cmd = [CL, "/nologo"] + objs + ["/Fe:" + os.path.join(OUT, exe)]
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


def build_all(tag, main_src, defs):
    for src in SRCS + [main_src]:
        rc, _, log = compile_tu(src, tag, defs)
        if rc != 0:
            return rc, log
    return 0, ""


def run(exe, args, timeout=120):
    try:
        p = subprocess.run([os.path.join(OUT, exe)] + args, cwd=ROOT, env=ENV,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return -999, "<TIMEOUT>"
    except OSError as e:
        return -998, f"<OSERROR {e}>"


def suite_signals(stdout):
    fails = re.findall(r"^FAIL ([^:]*test_main\.cpp):(\d+):", stdout, re.M)
    lines = sorted({int(n) for _, n in fails})
    m = re.search(r"superslm tests: (\d+) checks, (\d+) failures", stdout)
    checks = int(m.group(1)) if m else -1
    nfail = int(m.group(2)) if m else -1
    return {"fail_lines": lines, "checks": checks, "failures": nfail,
            "golden_pin_hit": bool(set(lines) & GOLDEN_PIN_LINES)}


def measure(tag_suffix=""):
    """Build both binaries from the current tree and take all observations."""
    obs = {}
    rc, log = build_all("h", HARNESS_MAIN, HARNESS_DEFS)
    if rc != 0:
        obs["harness_build"] = "FAIL"
        obs["harness_build_log"] = log[-2000:]
        return obs
    rc, log = link("h", "popper2_harness.exe")
    if rc != 0:
        obs["harness_build"] = "LINKFAIL"
        obs["harness_build_log"] = log[-2000:]
        return obs
    obs["harness_build"] = "OK"
    for mode in ("null", "candidate", "mutate", "qk"):
        rc, out = run("popper2_harness.exe", [mode])
        obs["rc_" + mode] = rc
        obs[mode] = out

    rc, log = build_all("t", SUITE_MAIN, SUITE_DEFS)
    if rc != 0:
        obs["suite_build"] = "FAIL"
        obs["suite_build_log"] = log[-2000:]
        return obs
    rc, log = link("t", "popper2_tests.exe")
    if rc != 0:
        obs["suite_build"] = "LINKFAIL"
        obs["suite_build_log"] = log[-2000:]
        return obs
    obs["suite_build"] = "OK"
    rc, out = run("popper2_tests.exe", [], timeout=300)
    obs["suite_rc"] = rc
    obs["suite"] = suite_signals(out)
    if rc == -999 or rc == -998:
        obs["suite"]["failures"] = -1
    return obs


def apply_mutation(site):
    path = os.path.join(ROOT, site["file"].replace("/", os.sep))
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    pos = site["pos"]
    old = site["old"]
    if text[pos:pos + len(old)] != old:
        return None, None
    mutated = text[:pos] + site["new"] + text[pos + len(old):]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(mutated)
    return path, text


def restore(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main():
    global ENV, CL
    ENV = msvc_env()
    CL = find_cl(ENV)
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(ROOT, "experiments", "popper2_population.json"),
              encoding="utf-8") as fh:
        pop = json.load(fh)

    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    # Clean reference.
    ref_path = os.path.join(ROOT, "experiments", "popper2_reference.json")
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as fh:
            ref = json.load(fh)
    else:
        t0 = time.time()
        ref = measure()
        with open(ref_path, "w", encoding="utf-8") as fh:
            json.dump(ref, fh)
        print(f"reference measured in {time.time()-t0:.1f}s: "
              f"suite={ref['suite']}", flush=True)

    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["idx"])

    n_new = 0
    for idx in pop["order"]:
        if n_new >= budget:
            break
        if idx in done:
            continue
        site = pop["sites"][idx]
        path, orig = apply_mutation(site)
        if path is None:
            rec = {"idx": idx, "site": site, "status": "PATCH_MISMATCH"}
            with open(RESULTS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            continue
        t0 = time.time()
        try:
            obs = measure()
        finally:
            restore(path, orig)
        rec = {"idx": idx, "site": site, "secs": round(time.time() - t0, 1)}

        if obs.get("harness_build") != "OK":
            rec["status"] = "BUILD_FAIL"
            rec["log"] = obs.get("harness_build_log", "")[-400:]
        elif obs.get("suite_build") != "OK":
            rec["status"] = "SUITE_BUILD_FAIL"
            rec["log"] = obs.get("suite_build_log", "")[-400:]
        else:
            rec["status"] = "MEASURED"
            rec["null_changed"] = obs["null"] != ref["null"]
            rec["candidate_changed"] = obs["candidate"] != ref["candidate"]
            rec["mutate_changed"] = obs["mutate"] != ref["mutate"]
            rec["qk_changed"] = obs["qk"] != ref["qk"]
            rec["mutate_summary"] = (re.search(r"SUMMARY.*", obs["mutate"]).group(0)
                                     if "SUMMARY" in obs["mutate"] else "<none>")
            rec["mutate_summary_changed"] = rec["mutate_summary"] != (
                re.search(r"SUMMARY.*", ref["mutate"]).group(0))
            rec["rc"] = {k: obs["rc_" + k] for k in ("null", "candidate", "mutate", "qk")}
            rec["suite"] = obs["suite"]
            rec["suite_rc"] = obs["suite_rc"]
            rec["null_out"] = obs["null"]
            rec["candidate_out"] = obs["candidate"]
            rec["mutate_out"] = obs["mutate"]
        with open(RESULTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_new += 1
        print(f"[{n_new}/{budget}] idx={idx} {site['kind']} "
              f"{site['file']}:{site['line']} {site['old']!r}->{site['new']!r} "
              f"{rec['status']} {rec.get('secs')}s", flush=True)


if __name__ == "__main__":
    main()
