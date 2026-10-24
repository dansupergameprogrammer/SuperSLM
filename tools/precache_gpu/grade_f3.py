#!/usr/bin/env python3
"""Precached-prefixes plan, F3: the 10.4 GPU speed bar's commissioning and grading, and 4.4's summary.

pcg_cell_10_4_timing never decides; it prints readings labelled with the role it ran as (§8, "GPU speed bar: the
same harness and the same commissioning as 10.3"). This tool turns them into decisions, in the order §8 requires:

  commission --accept LOG [LOG ...] --reject LOG [LOG ...]
      The must-accept (the built restore) must read bar=met, resolved, at every (adapter, model, L) it ran; the
      must-reject (the same cell built on the engine with apply_gpu_mutant.py's restore-prefills mutant) must read
      bar=missed at every (adapter, model, L) the must-accept ran. The two runners must differ (the mutant is a
      different build), and every reading of one role must come from one runner per adapter. Prints the verdict and,
      when it is COMMISSIONABLE, a PC-GPU-TIMING-1 receipt template for the debunker to review and sign. This tool
      signs nothing: a receipt counts only once the debunker's line is filled in.

  grade --receipt FILE --log LOG [LOG ...]
      Refuses an unsigned receipt. Grades --role=grade readings only, and only those taken by the runner the
      receipt commissioned on that adapter. Prints each required reading (1.5B on both GPUs and 0.5B on the RTX
      2080 SUPER, at 128 and 1,000 tokens) as met, missed, or absent; the bar is met only when all six are met.

  summary [--box DIR] [--kit DIR]
      4.4's roll-up over every step each machine owes (BOX_STEPS, KIT_STEPS): its verdict (the .done marker: PASS,
      FAIL ..., or NOT RUN when it never finished), every suite line in its log, and 4.6's findings apart.

Exit status: 0 for COMMISSIONABLE / MET / every suite PASS, 1 otherwise, 2 for a usage or input error.
"""
import argparse
import glob
import os
import re
import sys

RUN = re.compile(r"^10\.4 run (.*)$")
READING = re.compile(r"^10\.4 reading (.*)$")
SUITE = re.compile(r"^(\S+): checks=(\d+) failures=(\d+).*-> (PASS|FAIL)\s*$")
FIELD = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")
REQUIRED = [("10de_1e81", "1p5"), ("1002_744c", "1p5"), ("10de_1e81", "0p5")]
REQUIRED_L = [128, 1000]
ADAPTER_NAMES = {"10de_1e81": "RTX 2080 SUPER", "1002_744c": "RX 7900 XTX"}


def fields(text):
    return {k: v for k, v in FIELD.findall(text)}


def read_log(path):
    """Every reading in a log, as dicts; the run lines are checked against them."""
    readings, runs = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                m = READING.match(line)
                if m:
                    r = fields(m.group(1))
                    r["_log"] = path
                    readings.append(r)
                    continue
                m = RUN.match(line)
                if m:
                    runs.append(fields(m.group(1)))
    except OSError as e:
        raise SystemExit(f"grade_f3: cannot read {path}: {e}") from e
    return runs, readings


def key(r):
    return (r.get("adapter", "?"), r.get("model", "?"), int(r.get("L", "0")))


def collect(paths, role):
    """The readings of one role, with the problems that make them unusable."""
    problems, out = [], []
    for p in paths:
        runs, readings = read_log(p)
        if not runs:
            problems.append(f"{p}: no '10.4 run' line (not a pcg_cell_10_4_timing log)")
        for r in readings:
            if r.get("role") != role:
                problems.append(f"{p}: a role={r.get('role')} reading in the {role} logs")
                continue
            if len(r.get("runner", "")) != 64:
                problems.append(f"{p}: a reading with no runner SHA-256")
                continue
            if r.get("quiet") != "1":
                # The quiet guard (§8): a busy GPU slows the prefill arm far more than the restore arm, so a
                # reading taken with an offender present can read met falsely. Not usable in either direction.
                problems.append(f"{p}: {key(r)} was taken on a busy machine (quiet={r.get('quiet', 'absent')}); rerun it")
                continue
            if any(key(o) == key(r) for o in out):
                problems.append(f"{p}: a second {role} reading of {key(r)}; pass each reading once")
                continue
            out.append(r)
    runners = {}
    for r in out:
        runners.setdefault(r["adapter"], set()).add(r["runner"])
    for adapter, rs in sorted(runners.items()):
        if len(rs) != 1:
            problems.append(f"{role}: {len(rs)} different runners on {adapter}; one build per role per adapter")
    return out, problems


def commission(args):
    acc, problems = collect(args.accept, "accept")
    rej, p2 = collect(args.reject, "reject")
    problems += p2
    if not acc:
        problems.append("no must-accept readings")
    acc_by = {key(r): r for r in acc}
    rej_by = {key(r): r for r in rej}
    lines = []
    # Every reading the bar grades needs its own must-accept and must-reject (code review N4): an adapter commissioned
    # at one model is not commissioned at the other.
    for adapter, model in REQUIRED:
        for L in REQUIRED_L:
            if (adapter, model, L) not in acc_by:
                problems.append(f"must-accept {(adapter, model, L)}: no reading")
                lines.append(f"  must-accept {adapter} {model} L={L}: absent -> REFUSED")
    for r in sorted(acc, key=key):
        k = key(r)
        ok_a = r.get("bar") == "met" and r.get("resolved") == "1"
        lines.append(f"  must-accept {k[0]} {k[1]} L={k[2]}: bar={r.get('bar')} ratio95={r.get('ratio95')} "
                     f"resolved={r.get('resolved')} -> {'ok' if ok_a else 'REFUSED'}")
        if not ok_a:
            problems.append(f"must-accept {k} did not read met (resolved)")
        j = rej_by.get(k)
        if j is None:
            problems.append(f"must-reject {k}: no reading")
            lines.append(f"  must-reject {k[0]} {k[1]} L={k[2]}: absent -> REFUSED")
            continue
        ok_r = j.get("bar") == "missed"
        lines.append(f"  must-reject {k[0]} {k[1]} L={k[2]}: bar={j.get('bar')} ratio95={j.get('ratio95')} "
                     f"-> {'ok' if ok_r else 'REFUSED'}")
        if not ok_r:
            problems.append(f"must-reject {k} read met: the instrument cannot see a restore that prefills")
        if j.get("runner") == r.get("runner"):
            problems.append(f"{k}: the must-reject ran the must-accept's own build (same runner SHA-256)")
    verdict = "COMMISSIONABLE" if not problems else "NOT COMMISSIONABLE"
    print("10.4 GPU speed bar: instrument commissioning (plan §8)")
    print("\n".join(lines))
    for p in problems:
        print(f"  problem: {p}")
    print(f"verdict: {verdict}")
    if problems:
        return 1
    adapters = sorted({r["adapter"] for r in acc})
    print("\n----- PC-GPU-TIMING-1 receipt template (review, then fill the two signature lines) -----")
    print("receipt: PC-GPU-TIMING-1")
    print("instrument: tests/precache-gpu/pcg_cell_10_4_timing.cpp (plain configuration)")
    print("must-reject construction: tests/precache-gpu/apply_gpu_mutant.py restore-prefills")
    print("# Plan §8: commissioned by a seat independent of the builder. The builder wrote the instrument, the mutant and")
    print("# this tool, so the debunker reviews the construction as well as the readings: the mutant's diff (apply it to a")
    print("# copy of src/gpu/gpu_1p0.cpp and read it) and the instrument's quiet guard, and records that below.")
    print("construction_reviewed: ")
    for a in adapters:
        ra = next(r for r in acc if r["adapter"] == a)
        rr = next((r for r in rej if r["adapter"] == a), None)
        print(f"adapter: {a} ({ADAPTER_NAMES.get(a, 'unknown')})")
        print(f"  accept_runner: {ra['runner']}")
        print(f"  reject_runner: {rr['runner'] if rr else 'absent'}")
        for r in sorted((r for r in acc if r["adapter"] == a), key=key):
            j = rej_by[key(r)]
            print(f"  {r['model']} L={r['L']}: accept {r['ratio95']} met; reject {j['ratio95']} missed")
    for p in args.accept:
        print(f"accept_log: {os.path.basename(p)}")
    for p in args.reject:
        print(f"reject_log: {os.path.basename(p)}")
    print("verdict: COMMISSIONED")
    print("signed_by: ")
    print("signed_at_utc: ")
    return 0


def read_receipt(path):
    try:
        text = open(path, "r", encoding="utf-8").read()
    except OSError as e:
        raise SystemExit(f"grade_f3: cannot read receipt {path}: {e}") from e
    if not re.search(r"^receipt: PC-GPU-TIMING-1\s*$", text, re.M):
        return None, "not a PC-GPU-TIMING-1 receipt"
    if not re.search(r"^verdict: COMMISSIONED\s*$", text, re.M):
        return None, "the receipt's verdict is not COMMISSIONED"
    signer = re.search(r"^signed_by:[ \t]*(\S.*)$", text, re.M)
    when = re.search(r"^signed_at_utc:[ \t]*(\S.*)$", text, re.M)
    if not signer or not when:
        return None, "the receipt is unsigned (signed_by and signed_at_utc must both be filled in)"
    if not re.search(r"^construction_reviewed:[ \t]*(\S.*)$", text, re.M):
        return None, "the receipt does not record the debunker's review of the construction (construction_reviewed)"
    runners, adapter = {}, None
    for line in text.splitlines():
        m = re.match(r"^adapter: (\w+)", line)
        if m:
            adapter = m.group(1)
        m = re.match(r"^\s+accept_runner: ([0-9a-f]{64})\s*$", line)
        if m and adapter:
            runners[adapter] = m.group(1)
    if not runners:
        return None, "the receipt names no commissioned runner"
    return runners, None


def grade(args):
    runners, err = read_receipt(args.receipt)
    if err:
        print(f"10.4 GPU speed bar: REFUSED TO GRADE ({err}); readings before commissioning are recorded, not acted on")
        return 1
    readings, problems = collect(args.log, "grade")
    usable = {}
    for r in readings:
        want = runners.get(r["adapter"])
        if want is None:
            problems.append(f"{r['_log']}: {r['adapter']} has no commissioning on the receipt")
            continue
        if r["runner"] != want:
            problems.append(f"{r['_log']}: {r['adapter']} reading by runner {r['runner'][:12]}..., "
                            f"not the commissioned {want[:12]}...")
            continue
        usable[key(r)] = r
    met_all = True
    print("10.4 GPU speed bar: graded readings (bar: ratio95 lower end >= 100, resolved)")
    for adapter, model in REQUIRED:
        for L in REQUIRED_L:
            r = usable.get((adapter, model, L))
            if r is None:
                met_all = False
                print(f"  {adapter} ({ADAPTER_NAMES[adapter]}) {model} L={L}: absent")
                continue
            ok = r.get("bar") == "met" and r.get("resolved") == "1"
            met_all = met_all and ok
            print(f"  {adapter} ({ADAPTER_NAMES[adapter]}) {model} L={L}: ratio95={r['ratio95']} "
                  f"prefill_ms={r['prefill_ms']} restore_ms={r['restore_ms']} -> {'met' if ok else 'MISSED'}")
    for p in problems:
        print(f"  problem: {p}")
    print(f"verdict: {'MET' if met_all and not problems else 'NOT MET (R1b: fold, then re-gate before F4)'}")
    return 0 if met_all and not problems else 1


# The steps each machine owes (code review B1: a step that never started leaves no file, so the list cannot come from
# the directory). The box's `kit` piece only copies files and is not a certification step.
BOX_STEPS = ["build", "build-mut", "cpu95", "f2set", "real15", "real05", "r10x", "t-acc-15", "t-rej-15", "t-acc-05",
             "t-rej-05", "t-grade-15", "t-grade-05", "mutants", "cross"]
KIT_STEPS = ["f2set", "real15", "real05", "r10x", "t-acc-15", "t-rej-15", "t-grade-15"]
FINDING = re.compile(r"4\.6-FINDING: ([^;]*)")


def summary(args):
    """4.4's roll-up. A step's verdict is its .done marker's first word: PASS, FAIL, or NOT RUN when there is no
    marker (a log without one is a crashed step). A mutant step's PASS means the mutant was killed, so its suites'
    FAIL lines are expected and shown only as detail. 4.6 is graded apart (plan §9 F3: "4.6 records a finding either
    way"): a 4.6 difference is printed as a finding and does not fail 4.4."""
    pairs = [(d, BOX_STEPS, "box") for d in (args.box or [])] + [(d, KIT_STEPS, "kit") for d in (args.kit or [])]
    if not pairs:
        print("summary: pass --box DIR and/or --kit DIR")
        return 2
    worst, findings = 0, []
    for d, steps, kind in pairs:
        if not os.path.isdir(d):
            print(f"{d}: not a directory")
            worst = 2
            continue
        print(f"== {kind} {d}")
        for step in steps:
            done = os.path.join(d, step + ".done")
            lp = os.path.join(d, step + ".log")
            mark = open(done, encoding="utf-8", errors="replace").read().strip() if os.path.exists(done) else ""
            if mark:
                verdict = mark.split(" ", 1)[0]
                print(f"  {step}: {mark}")
            else:
                verdict = "NOT RUN"
                print(f"  {step}: NOT RUN{' (a log without a marker: it crashed; rerun it)' if os.path.exists(lp) else ''}")
            if verdict != "PASS":
                worst = max(worst, 1)
            findings += [f"{kind} {step}: {f.strip()}" for f in FINDING.findall(mark)]
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m = SUITE.match(line.strip())
                        if m:
                            print(f"    {m.group(1)}: checks={m.group(2)} failures={m.group(3)} -> {m.group(4)}")
    print("4.6 (a finding either way, not a 4.4 gate):")
    for f in findings or ["no difference recorded"]:
        print(f"  {f}")
    if not (args.box and args.kit):
        # 4.4 is "on both certified GPUs": one machine's table is not a verdict.
        print("4.4: INCOMPLETE (pass both --box and --kit for a 4.4 verdict)")
        return max(worst, 1)
    print(f"4.4: {'PASS' if worst == 0 else 'NOT PASSED'}")
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commission")
    c.add_argument("--accept", nargs="+", required=True)
    c.add_argument("--reject", nargs="+", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--receipt", required=True)
    g.add_argument("--log", nargs="+", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--box", nargs="+", help="the box's logs directory")
    s.add_argument("--kit", nargs="+", help="the kit's results\\<computer> directory")
    a = ap.parse_args(argv)
    return {"commission": commission, "grade": grade, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
