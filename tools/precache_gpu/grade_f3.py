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

  summary DIR [DIR ...]
      4.4's roll-up from the box's logs directory and the kit's results directory: each step's verdict (its .done
      marker: PASS, FAIL ..., or NOT DONE), and every suite line ("<suite>: checks=N ... -> PASS|FAIL") in its log.

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
    rej_by = {key(r): r for r in rej}
    lines = []
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


NOT_STEPS = {"rest", "kit"}  # the box's rest log and the kit's own run log carry no step


def summary(args):
    """Each step's verdict is its .done marker's first word (the pieces mark PASS or FAIL ...; a mutant step's PASS
    means the mutant was killed, so its suites' FAIL lines are expected and shown only as detail)."""
    worst = 0
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"{d}: not a directory")
            worst = 2
            continue
        print(f"== {d}")
        steps = sorted({os.path.basename(p).rsplit(".", 1)[0] for p in glob.glob(os.path.join(d, "*.done"))}
                       | {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, "*.log"))})
        for step in steps:
            if step in NOT_STEPS:
                continue
            done = os.path.join(d, step + ".done")
            mark = open(done, encoding="utf-8", errors="replace").read().strip() if os.path.exists(done) else ""
            verdict = mark.split(" ", 1)[0] if mark else "NOT DONE"
            print(f"  {step}: {mark if mark else verdict}")
            if verdict != "PASS":
                worst = max(worst, 1)
            lp = os.path.join(d, step + ".log")
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m = SUITE.match(line.strip())
                        if m:
                            print(f"    {m.group(1)}: checks={m.group(2)} failures={m.group(3)} -> {m.group(4)}")
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
    s.add_argument("dirs", nargs="+")
    a = ap.parse_args(argv)
    return {"commission": commission, "grade": grade, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
