#!/usr/bin/env python3
"""GPU paged-KV plan (rev 16), step G3: the D3D12-free core's arms and mutants.

Builds tests/gpu-g3-core's cells against src/gpu's D3D12-free sources with g++ (13.3 is the plan's
ThreadSanitizer route, §4.4 *The submission ledger*), runs each arm one at a time, and grades:

  green arms   plain (-O2), ThreadSanitizer (-O1 -g -fsanitize=thread, halt_on_error=0) and
               AddressSanitizer+UBSan: every cell green, no sanitizer report, inside its timeout;
  mutants      each a textual patch of one production file in a scratch copy of the tree (the
               tree itself is never written), built in the flavour its cell is graded under and run
               on that cell; red means a failed check, a sanitizer report, a crash, or no exit
               within the timeout ("hung"). Mutant arms run with TSAN_OPTIONS=halt_on_error=1.
               A mutant that is not red is a failed cell, not a flake.

Plus the serialised REGRACE case's second reading (the unlocked registry at 100,000 decodes,
halt_on_error=0, its race count recorded, not graded; plan §4.4 *The registry cell*).

Timeouts (plan §4.4 *Hangs and timeouts*): three times the arm's slowest measured run, never under
60 s. Usage:
  python3 tests/gpu-g3-core/run_g3_core.py [--work DIR] [--only NAME ...] [--list]
Exit status 0 only when every green arm is green and every mutant is red.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_FILES = ["submission_ledger.h", "gpu_inflight_token.h", "gpu_inflight_registry.h",
             "gpu_inflight_registry.cpp", "gpu_call_timing_snapshot.h"]
TEST_DIR = "tests/gpu-g3-core"
TEST_SOURCES = ["g3_main.cpp", "g3_cells_ledger.cpp", "g3_cells_token.cpp", "g3_cells_registry.cpp",
                "g3_cells_tail.cpp"]
CXX = os.environ.get("CXX_G3", "g++")
FLAVOURS = {
    "plain": ["-O2"],
    "tsan": ["-O1", "-g", "-fsanitize=thread"],
    "asan": ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"],
}
ARM_FACTOR, ARM_FLOOR_S = 3, 60
# Slowest measured run per arm, seconds, at the cells' own lengths, on this container's four cores
# oversubscribed with six busy loops (nice 19) beside the other sessions' load (load average 12):
# plain 14.8, ThreadSanitizer 24.6, AddressSanitizer 22.0, and the unlocked serialised REGRACE at
# 100,000 decodes 190.6 (17 races). An arm missing here gets 100 s, so 300 s.
ARM_MEASURED_S = {
    ("plain", "all"): 15.0,
    ("tsan", "all"): 25.0,
    ("asan", "all"): 25.0,
    ("tsan", "REGRACE/serial@100000"): 191.0,
}


def arm_limit(flavour: str, key: str) -> int:
    return max(ARM_FLOOR_S, int(ARM_FACTOR * ARM_MEASURED_S.get((flavour, key), 100) + 0.999))


class Mutant:
    def __init__(self, name, what, file, old, new, flavour, cells, params=(), more=(), must_print=()):
        # `more`: further (old, new) replacements in the same file, for a defect that spans two sites.
        # `must_print`: lines the red run must print (the cells the plan says the mutant turns red).
        self.name, self.what, self.file = name, what, file
        self.patches = [(old, new)] + list(more)
        self.must_print = list(must_print)
        self.flavour, self.cells, self.params = flavour, list(cells), list(params)


# Each mutant: the production defect it plants, the file and the exact text it replaces (which must
# land exactly once), the flavour and the cells that grade it.
MUTANTS = [
    Mutant("stranded-drain-frees", "review-g23 F1: a token's drain frees its payload even when the drain strands",
           "gpu_inflight_token.h",
           "\t\tif (p->submission_.DrainNoThrow()) {\n\t\t\tdelete p;\n\t\t} else {\n\t\t\tdev->KeepAliveStranded(p);\n\t\t}\n",
           "\t\t(void)dev;\n\t\tp->submission_.DrainNoThrow();\n\t\tdelete p;\n",
           "plain", ["G3/token/stranded-keeps-payload"]),
    Mutant("stranded-wait-frees", "review-g23 F1: the throwing wait leaves the payload on the token, to be freed",
           "gpu_inflight_token.h",
           "\t\tif (!payload_->submission_.DrainNoThrow()) {\n\t\t\tGpuInFlightPayload* const p = payload_;\n\t\t\tpayload_ = nullptr;\n\t\t\tdev->KeepAliveStranded(p);\n",
           "\t\tif (!payload_->submission_.DrainNoThrow()) {\n\t\t\t(void)dev;\n",
           "plain", ["G3/token/stranded-keeps-payload"]),
    Mutant("ledger-plain-store", "3.6h's first named mutant: observed_ stored plainly, not by a CAS maximum",
           "submission_ledger.h",
           "\t\twhile (v > cur && !observed_.compare_exchange_weak(cur, v)) {\n\t\t}\n",
           "\t\t(void)cur;\n\t\tobserved_.store(v);\n",
           "plain", ["3.6h/orders"]),
    Mutant("ledger-load-compare-store", "3.6h's second named mutant: a load, a compare and a separate store "
           "(the hook kept between the load and the store)",
           "submission_ledger.h",
           "\t\twhile (v > cur && !observed_.compare_exchange_weak(cur, v)) {\n\t\t}\n",
           "\t\tif (v > cur) observed_.store(v);\n",
           "plain", ["3.6h/yield-hook"]),
    Mutant("registry-nolock", "the registry cell's mutant: the list mutexes removed",
           "gpu_inflight_registry.cpp",
           "using ListMutex = std::mutex;\n",
           "struct ListMutex {\n\tvoid lock() {}\n\tvoid unlock() {}\n};\n",
           "tsan", ["REGRACE/pollers"]),
    Mutant("registry-plain-list", "a maker that exits: a plain thread_local list (records do not own it)",
           "gpu_inflight_registry.cpp",
           "\t\trec->list_ = L;\n",
           "\t\trec->list_ = std::shared_ptr<RegList>(L.get(), [](RegList*) {});\n",
           "asan", ["REGISTRY/churn"]),
    Mutant("registry-depth0-nocall", "rev 13's rule: a depth-0 move or lowering registers under no call",
           "gpu_inflight_registry.cpp",
           "\t\tif (t_depth != 0) return false;\n",
           "\t\t(void)holder;\n\t\tif (true) return false;\n",
           "plain", ["GRID/depth0"], must_print=["grid: event=move-construct record=fresh thread=host reports=0 next_guarded_call=SSLM_BUSY", "grid: event=move-assign record=fresh thread=host reports=0 next_guarded_call=SSLM_BUSY", "grid: event=accessor-lowering record=fresh thread=host reports=0 next_guarded_call=SSLM_BUSY"]),
    Mutant("trampoline-installs-nothing", "rev 14: the task runs with no binding installed",
           "gpu_inflight_registry.cpp",
           "\tstatic void Install(const RegistryBinding& b) {\n\t\tif (b.depth_ > 0) {",
           "\tstatic void Install(const RegistryBinding& b) {\n\t\treturn;\n\t\tif (b.depth_ > 0) {",
           "plain", ["GRID/depth0"], must_print=["grid: event=submit record=fresh thread=task reports=0 next_guarded_call=SSLM_BUSY", "grid: event=move-construct record=prior thread=task reports=0 next_guarded_call=SSLM_BUSY", "grid: event=move-assign record=prior thread=task reports=0 next_guarded_call=SSLM_BUSY", "grid: event=accessor-lowering record=prior thread=task reports=0 next_guarded_call=SSLM_BUSY"]),
    Mutant("trampoline-list-call-only", "rev 14.2's B1: the trampoline installs the list and the call, not the depth",
           "gpu_inflight_registry.cpp",
           "\t\t\tt_call = b.call_;\n\t\t\tt_depth = b.depth_;\n",
           "\t\t\tt_call = b.call_;\n",
           "plain", ["GRID/depth0"], must_print=["grid: event=submit record=fresh thread=task reports=0 next_guarded_call=SSLM_BUSY", "grid: event=move-construct record=prior thread=task reports=0 next_guarded_call=SSLM_BUSY", "grid: event=move-assign record=prior thread=task reports=0 next_guarded_call=SSLM_BUSY", "grid: event=accessor-lowering record=prior thread=task reports=0 next_guarded_call=SSLM_BUSY"]),
    Mutant("trampoline-no-empty", "rev 14.2's W1: with no scope open the trampoline installs nothing",
           "gpu_inflight_registry.cpp",
           "\t\t} else {\n\t\t\tt_bound_list.reset();\n\t\t\tt_call = 0;\n\t\t\tt_depth = 0;\n\t\t}\n",
           "\t\t}\n",
           "tsan", ["GRID/empty-binding"]),
    Mutant("walk-drains-under-lock", "the free-under-lock mutant: the walk drains while holding its list lock",
           "gpu_inflight_registry.cpp",
           "\t\t\t\t\tholders[n] = h;\n\t\t\t\t\t++n;\n",
           "\t\t\t\t\th->DrainFromWalk();\n\t\t\t\t\tg_reports.fetch_add(1);\n\t\t\t\t\tbreak;\n",
           "plain", ["G3/submission/file-scope"]),
    Mutant("walk-every-frame", "the check at every C ABI return, not the outermost only",
           "gpu_inflight_registry.cpp",
           "\tif (--t_depth == 0) {\n\t\tRegistryAccess::Walk();",
           "\t--t_depth;\n\t{\n\t\tRegistryAccess::Walk();",
           "plain", ["G3/scope/outermost-only"]),
    Mutant("walk-process-wide", "a process-wide check: one list for every thread, walked whole",
           "gpu_inflight_registry.cpp",
           "\tif (!t_own_list) t_own_list = std::make_shared<RegList>();\n",
           "\tstatic std::shared_ptr<RegList> g_one = std::make_shared<RegList>();\n\tif (!t_own_list) t_own_list = g_one;\n",
           "plain", ["G3/scope/per-thread"],
           more=[("\t\t\t\t\tif (r->call_ != t_call) continue;\n", "")]),
    Mutant("boundary-guard-removed", "the boundary guard check never refuses",
           "gpu_inflight_registry.cpp",
           "bool BoundaryGuardCheckRefuses() noexcept { return t_depth > 0 && !t_guard_ran; }",
           "bool BoundaryGuardCheckRefuses() noexcept { return false; }",
           "plain", ["G3/ledger/boundary-guard"]),
    Mutant("check-after-reset", "the check runs after the reset",
           "submission_ledger.h",
           "\t\tPoolSubmitAssert(ledger_);\n\t\tResetAllocatorAndList();\n",
           "\t\tResetAllocatorAndList();\n\t\tPoolSubmitAssert(ledger_);\n",
           "plain", ["G3/ledger/check-before-reset"]),
    Mutant("check-unobserved-dropped", "the check fires on a stranded ledger only",
           "submission_ledger.h",
           "\tif (ledger.Unobserved()) {\n\t\tthrow PoolSubmitContractError(",
           "\tif (false) {\n\t\tthrow PoolSubmitContractError(",
           "plain", ["G3/ledger/check-before-reset"]),
    Mutant("wait-always-arms", "Wait(v) arms the event even when the fence is at v",
           "submission_ledger.h",
           "\t\tif (CompletedFenceValue() >= v) {\n\t\t\tledger_.RaiseObserved(v);\n\t\t\treturn true;\n\t\t}\n\t\tif (!WaitFenceEvent(v)) {",
           "\t\tif (!WaitFenceEvent(v)) {",
           "plain", ["G3/ledger/wait-fast-path"]),
    Mutant("retry-same-value", "the retry signals the failed value again",
           "submission_ledger.h",
           "\tconst uint64_t retry_value = ++fence_val_;\n",
           "\tconst uint64_t retry_value = first_value;\n",
           "plain", ["G3/ledger/submit-recovered"]),
    Mutant("no-unsignaled", "unsignaled_ never set: a failed retry is not stranded",
           "submission_ledger.h",
           "\tvoid MarkUnsignaled() noexcept { unsignaled_.store(true); }",
           "\tvoid MarkUnsignaled() noexcept {}",
           "plain", ["G3/ledger/submit-retry-strands"]),
    Mutant("lost-cleared", "a later observation clears lost_",
           "submission_ledger.h",
           "\t\tuint64_t cur = observed_.load();\n",
           "\t\tlost_.store(false);\n\t\tuint64_t cur = observed_.load();\n",
           "plain", ["3.6h/stranded-terminal"]),
    Mutant("submission-dtor-no-drain", "a GpuSubmission destroyed while holding its record frees it unobserved",
           "submission_ledger.h",
           "\t~GpuSubmission() {\n\t\tif (rec_) DrainNoThrow();\n\t}",
           "\t~GpuSubmission() {\n\t\tif (rec_) {\n\t\t\tinflight::Unlink(rec_);\n\t\t\tdelete rec_;\n\t\t}\n\t}",
           "plain", ["G3/submission/early-return"]),
    Mutant("token-dtor-no-drain", "a token destroyed while holding work frees it unobserved",
           "gpu_inflight_token.h",
           # re-anchored at review-g23 F1 (the drain now frees only a confirmed payload); the same defect
           "\t\tSubmissionDevice* const dev = p->submission_.device();\n\t\tif (p->submission_.DrainNoThrow()) {\n"
           "\t\t\tdelete p;\n\t\t} else {\n\t\t\tdev->KeepAliveStranded(p);\n\t\t}\n",
           "\t\tif (p->submission_.rec_) {\n\t\t\tinflight::Unlink(p->submission_.rec_);\n"
           "\t\t\tdelete p->submission_.rec_;\n\t\t\tp->submission_.rec_ = nullptr;\n\t\t}\n\t\tdelete p;\n",
           "plain", ["G3/token/drain-on-destruction"]),
    Mutant("token-move-assign-no-drain", "a move-assignment over a live token overwrites it",
           "gpu_inflight_token.h",
           "\t\tif (this != &other) {\n\t\t\tif (payload_) DrainAndFreeNoThrow();\n",
           "\t\tif (this != &other) {\n",
           "plain", ["G3/token/move-assign-drains"]),
    Mutant("token-move-no-reregister", "a move-construction does not re-register the record it carries",
           "gpu_inflight_token.h",
           "\t\tother.payload_ = nullptr;\n\t\tAdopt();\n\t}\n\t// Drains the record",
           "\t\tother.payload_ = nullptr;\n\t}\n\t// Drains the record",
           "asan", ["G3/token/reregister"]),
    Mutant("token-move-drains", "a move observes (drains) the work it carries",
           "gpu_inflight_token.h",
           "\t\tother.payload_ = nullptr;\n\t\tAdopt();\n\t}\n\t// Drains the record",
           "\t\tother.payload_ = nullptr;\n\t\tAdopt();\n\t\tif (payload_) payload_->submission_.Wait();\n\t}\n\t// Drains the record",
           "plain", ["G3/token/move"]),
    Mutant("not-ready-poll-empties", "the not-ready test waits and empties the token",
           "gpu_inflight_token.h",
           "\tbool ReadyNoWait() const noexcept { return payload_ == nullptr || payload_->submission_.ReadyNoWait(); }",
           "\tbool ReadyNoWait() const noexcept {\n\t\tif (payload_) payload_->submission_.DrainNoThrow();\n\t\treturn true;\n\t}",
           "plain", ["G3/token/not-ready-poll"]),
    Mutant("lowering-no-reregister", "the accessor's lowering leaves the record where it was",
           "gpu_inflight_token.h",
           "\tvoid ReregisterOnLowering() noexcept { Adopt(); }",
           "\tvoid ReregisterOnLowering() noexcept {}",
           "plain", ["G3/token/reregister"]),
    Mutant("snapshot-unlocked", "decision 80's mutant: the finish tail's shared timing state written without its lock",
           "gpu_call_timing_snapshot.h",
           "\tvoid Swap(std::shared_ptr<const Block>& fresh) {\n\t\tstd::lock_guard<std::mutex> lock(mu_);\n",
           "\tvoid Swap(std::shared_ptr<const Block>& fresh) {\n",
           "tsan", ["TAIL/poller-first", "TAIL/poller-pp"]),
]


def run(cmd, timeout, env=None):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr, time.time() - t0, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return None, out, err, time.time() - t0, True


def make_tree(work: Path, name: str, mutant: Mutant | None) -> Path:
    tree = work / name
    if tree.exists():
        shutil.rmtree(tree)
    (tree / "src/gpu").mkdir(parents=True)
    (tree / TEST_DIR).mkdir(parents=True)
    for f in SRC_FILES:
        shutil.copy(ROOT / "src/gpu" / f, tree / "src/gpu" / f)
    for f in TEST_SOURCES + ["g3_harness.h", "g3_fakes.h"]:
        shutil.copy(ROOT / TEST_DIR / f, tree / TEST_DIR / f)
    if mutant is not None:
        p = tree / "src/gpu" / mutant.file
        text = p.read_text()
        for old, new in mutant.patches:
            n = text.count(old)
            if n != 1:
                raise RuntimeError(f"mutant {mutant.name}: anchor lands {n} times in {mutant.file}, not once")
            text = text.replace(old, new)
        p.write_text(text)
    return tree


def build(tree: Path, flavour: str) -> tuple[Path | None, str]:
    exe = tree / f"g3_{flavour}"
    cmd = ([CXX, "-std=c++20", "-Wall", "-Wextra", "-DSUPERSLM_LEDGER_YIELD_HOOK"] + FLAVOURS[flavour]
           + [f"-I{tree / 'src/gpu'}", f"-I{tree / TEST_DIR}"]
           + [str(tree / TEST_DIR / s) for s in TEST_SOURCES] + [str(tree / "src/gpu/gpu_inflight_registry.cpp")]
           + ["-o", str(exe), "-pthread"])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return (exe if r.returncode == 0 else None), r.stderr


def sanitizer_reports(err: str) -> int:
    return (err.count("WARNING: ThreadSanitizer") + err.count("ERROR: AddressSanitizer")
            + len(re.findall(r"runtime error:", err)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", help="scratch directory (default: a fresh temporary one)")
    ap.add_argument("--only", nargs="*", help="arms or mutants to run, by name")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for f in FLAVOURS:
            print(f"green-{f}")
        for m in MUTANTS:
            print(f"{m.name:28s} {m.flavour:5s} {' '.join(m.cells):34s} {m.what}")
        print("regrace-serial-unlocked (second reading)")
        return 0
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="g3core_"))
    work.mkdir(parents=True, exist_ok=True)
    want = set(args.only or [])
    sel = lambda n: not want or n in want
    ok = True
    print(f"g3 core arms: {CXX} ({subprocess.run([CXX, '--version'], capture_output=True, text=True).stdout.splitlines()[0]}); work {work}")

    for flavour in FLAVOURS:
        name = f"green-{flavour}"
        if not sel(name):
            continue
        tree = make_tree(work, name, None)
        exe, err = build(tree, flavour)
        if exe is None:
            print(f"{name}: FAIL (does not compile)\n{err[-2000:]}")
            ok = False
            continue
        limit = arm_limit(flavour, "all")
        rc, out, err, secs, hung = run([str(exe)], limit, dict(os.environ, TSAN_OPTIONS="halt_on_error=0"))
        reports = sanitizer_reports(err)
        last = (out.strip().splitlines() or ["(no output)"])[-1]
        good = rc == 0 and reports == 0 and not hung
        ok = ok and good
        print(f"{name}: {'green' if good else 'FAIL'}; {last}; sanitizer reports {reports}; {secs:.1f} s"
              f"{' (timed out at %d s)' % limit if hung else ''}")
        if not good:
            print("\n".join(l for l in err.splitlines() if l.startswith("FAIL"))[:4000])

    for m in MUTANTS:
        if not sel(m.name):
            continue
        tree = make_tree(work, "mutant-" + m.name, m)
        exe, err = build(tree, m.flavour)
        if exe is None:
            print(f"mutant {m.name}: FAIL (does not compile; a mutant must build to be graded)\n{err[-1500:]}")
            ok = False
            continue
        limit = arm_limit(m.flavour, "all")
        cmd = [str(exe)] + ["=" + c for c in m.cells] + list(m.params)
        rc, out, err, secs, hung = run(cmd, limit, dict(os.environ, TSAN_OPTIONS="halt_on_error=1"))
        reports = sanitizer_reports(err)
        fails = [l for l in err.splitlines() if l.startswith("FAIL")]
        if hung:
            reading = f"hung (no exit within {limit} s)"
        elif reports:
            reading = f"{reports} sanitizer report(s), exit {rc}"
        elif rc != 0 and fails:
            reading = f"{len(fails)} failed check(s): {fails[0].split(': ', 1)[-1][:100]}"
        elif rc != 0:
            reading = f"exit {rc} (crash or abort)"
        else:
            reading = "every check passed"
        red = hung or reports > 0 or rc != 0
        missing = [l for l in m.must_print if l not in out]
        if missing and not hung:
            red = False
            reading += f"; but {len(missing)} expected red line(s) absent, first: {missing[0]!r}"
        ok = ok and red
        print(f"mutant {m.name} [{m.flavour}, {' '.join(m.cells)}]: {'RED' if red else 'NOT RED (failed cell)'}; "
              f"{reading}; {secs:.1f} s")

    if sel("regrace-serial-unlocked"):
        m = next(x for x in MUTANTS if x.name == "registry-nolock")
        tree = make_tree(work, "second-reading-serial", m)
        exe, err = build(tree, "tsan")
        if exe is None:
            print("REGRACE/serial unlocked, second reading: does not compile")
            ok = False
        else:
            limit = arm_limit("tsan", "REGRACE/serial@100000")
            rc, out, err, secs, hung = run([str(exe), "=REGRACE/serial"], limit,
                                           dict(os.environ, TSAN_OPTIONS="halt_on_error=0"))
            races = err.count("WARNING: ThreadSanitizer")
            state = f"hung after {races} reports" if hung else f"{races} races, exit {rc}"
            print(f"REGRACE/serial unlocked, 100000 decodes, second reading (not graded): {state}; {secs:.1f} s")

    print(f"g3 core: {'every green arm green and every mutant red' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
