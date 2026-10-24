#!/usr/bin/env python3
"""tools/precache_gpu/grade_f3.py on constructed logs: commissioning accepts only a met must-accept beside a missed
must-reject from a different build, at every reading the bar grades, on a quiet machine, each reading once; grading
refuses an unsigned receipt, one without the construction review, a busy reading and an uncommissioned runner; the
summary lists every step each machine owes (NOT RUN for a missing one) and grades 4.6 apart from 4.4."""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "precache_gpu"))
import grade_f3  # noqa: E402

A, R, X = "a" * 64, "b" * 64, "c" * 64


def log(role, runner, adapter, model, rows, quiet=1):
    out = [f"10.4 run role={role} runner={runner} adapter={adapter} model={model} model_sha256={'d' * 64} pairs=8 "
           f"bar=100 tick_ms=0.000100"]
    for L, lo, hi, bar in rows:
        out.append(f"10.4 reading role={role} runner={runner} adapter={adapter} model={model} L={L} pairs=8 "
                   f"bytes_including_digest=1000 prefill_ms=900.000 restore_ms=2.000 ratio95=[{lo},{hi}] resolved=1 "
                   f"quiet={quiet} bar={bar}")
    out.append("pcg_cell_10_4_timing: checks=4 failures=0 -> PASS")
    return "\n".join(out) + "\n"


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = grade_f3.main(argv)
    return rc, buf.getvalue()


def write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def main():
    fails = 0

    def expect(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    with tempfile.TemporaryDirectory() as d:
        met = [(128, 410.2, 480.9, "met"), (1000, 900.0, 1010.0, "met")]
        missed = [(128, 0.9, 1.1, "missed"), (1000, 0.9, 1.0, "missed")]

        def both(role, runner, rows_box, rows_x=None, **kw):
            return (log(role, runner, "10de_1e81", "1p5", rows_box, **kw) + log(role, runner, "10de_1e81", "0p5", rows_box, **kw)
                    + log(role, runner, "1002_744c", "1p5", rows_x if rows_x is not None else rows_box, **kw))

        acc = write(d, "acc.log", both("accept", A, met))
        rej = write(d, "rej.log", both("reject", R, missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej])
        expect(rc == 0 and "verdict: COMMISSIONABLE" in out and "receipt: PC-GPU-TIMING-1" in out,
               "must-accept met + must-reject missed at all six readings, different builds: commissionable")
        tmpl = out.split("-----\n", 1)[1]

        rej_met = write(d, "rejmet.log", both("reject", R, missed, rows_x=met))
        rc, out = run(["commission", "--accept", acc, "--reject", rej_met])
        expect(rc == 1 and "NOT COMMISSIONABLE" in out, "a must-reject that reads met refuses commissioning")

        rej_same = write(d, "rejsame.log", both("reject", A, missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej_same])
        expect(rc == 1 and "same runner" in out, "a must-reject run by the must-accept's build refuses commissioning")

        acc_miss = write(d, "accmiss.log", both("accept", A, met, rows_x=missed))
        rc, out = run(["commission", "--accept", acc_miss, "--reject", rej])
        expect(rc == 1, "a must-accept that misses refuses commissioning")

        rej_short = write(d, "rejshort.log", log("reject", R, "10de_1e81", "1p5", missed) + log("reject", R, "1002_744c", "1p5", missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej_short])
        expect(rc == 1 and "no reading" in out, "a must-reject missing a graded reading refuses commissioning")

        acc_no05 = write(d, "accno05.log", log("accept", A, "10de_1e81", "1p5", met) + log("accept", A, "1002_744c", "1p5", met))
        rc, out = run(["commission", "--accept", acc_no05, "--reject", rej])
        expect(rc == 1 and "must-accept ('10de_1e81', '0p5', 128): no reading" in out,
               "N4: an adapter commissioned at 1.5B only is not commissioned at 0.5B")

        acc_busy = write(d, "accbusy.log", both("accept", A, met, quiet=0))
        rc, out = run(["commission", "--accept", acc_busy, "--reject", rej])
        expect(rc == 1 and "busy machine" in out, "B3: a must-accept taken on a busy machine is refused")

        acc_dup = write(d, "accdup.log", log("accept", A, "10de_1e81", "1p5", missed))
        rc, out = run(["commission", "--accept", acc_dup, acc, "--reject", rej])
        expect(rc == 1 and "a second accept reading" in out, "N3: a duplicate reading is refused, not overridden")

        g_ok = write(d, "g.log", both("grade", A, met))
        unsigned = write(d, "unsigned.txt", tmpl)
        rc, out = run(["grade", "--receipt", unsigned, "--log", g_ok])
        expect(rc == 1 and "unsigned" in out, "an unsigned receipt is refused before any reading is graded")

        signed_only = tmpl.replace("signed_by: ", "signed_by: debunker").replace("signed_at_utc: ", "signed_at_utc: 2026-10-25T00:00:00Z")
        nocon = write(d, "nocon.txt", signed_only)
        rc, out = run(["grade", "--receipt", nocon, "--log", g_ok])
        expect(rc == 1 and "construction_reviewed" in out, "N8: a receipt without the construction review is refused")

        signed = write(d, "signed.txt", signed_only.replace("construction_reviewed: ", "construction_reviewed: mutant diff read"))
        rc, out = run(["grade", "--receipt", signed, "--log", g_ok])
        expect(rc == 0 and "verdict: MET" in out, "six met readings by the commissioned runners: MET")

        g_part = write(d, "gpart.log", log("grade", A, "10de_1e81", "1p5", met))
        rc, out = run(["grade", "--receipt", signed, "--log", g_part])
        expect(rc == 1 and "absent" in out, "a required reading absent: not met")

        g_busy = write(d, "gbusy.log", both("grade", A, met, quiet=0))
        rc, out = run(["grade", "--receipt", signed, "--log", g_busy])
        expect(rc == 1 and "busy machine" in out, "B3: a graded reading taken on a busy machine is not graded")

        g_dup = write(d, "gdup.log", log("grade", A, "10de_1e81", "1p5", [(128, 60.0, 90.0, "missed"), (1000, 900.0, 1010.0, "met")]))
        rc, out = run(["grade", "--receipt", signed, "--log", g_dup, g_ok])
        expect(rc == 1 and "a second grade reading" in out, "N3: a later met cannot override an earlier missed")

        g_other = write(d, "gother.log", both("grade", R, met))
        rc, out = run(["grade", "--receipt", signed, "--log", g_other])
        expect(rc == 1 and "not the commissioned" in out, "a reading by an uncommissioned build is not graded")

        g_miss = write(d, "gmiss.log", log("grade", A, "10de_1e81", "1p5", [(128, 60.0, 90.0, "missed"), (1000, 900.0, 1010.0, "met")])
                       + log("grade", A, "10de_1e81", "0p5", met) + log("grade", A, "1002_744c", "1p5", met))
        rc, out = run(["grade", "--receipt", signed, "--log", g_miss])
        expect(rc == 1 and "MISSED" in out and "R1b" in out, "one missed reading: not met, R1b named")

        box = os.path.join(d, "box")
        kit = os.path.join(d, "kit")
        os.mkdir(box)
        os.mkdir(kit)
        for st in grade_f3.BOX_STEPS:
            write(box, st + ".log", "pcg_cell_pool: checks=10 failures=0 skips=0 -> PASS\n")
            write(box, st + ".done", "PASS 2026-10-24T10:00:00")
        write(box, "mutants.log", "pcg_cell_6_5: checks=10 failures=1 skips=0 -> FAIL\n")
        write(box, "real15.done", "PASS 4.6-FINDING: pcg_cell_4_6 1p5 differs 2026-10-24T10:00:00")
        for st in grade_f3.KIT_STEPS:
            write(kit, st + ".done", "PASS 2026-10-24T10:00:00")
        rc, out = run(["summary", "--box", box, "--kit", kit])
        expect(rc == 0 and "-> FAIL" in out and "4.4: PASS" in out and "box real15: pcg_cell_4_6 1p5 differs" in out,
               "summary: every step done; a killed mutant's FAIL suite and a 4.6 finding do not fail 4.4")
        os.remove(os.path.join(kit, "t-grade-15.done"))
        os.remove(os.path.join(box, "cross.done"))
        os.remove(os.path.join(box, "cross.log"))
        rc, out = run(["summary", "--box", box, "--kit", kit])
        expect(rc == 1 and "cross: NOT RUN" in out and "t-grade-15: NOT RUN" in out,
               "B1: a step that never started (no files at all) is listed NOT RUN and fails 4.4")
        write(box, "cross.done", "FAIL pcg_cell_4_6 (b, c) 1p5 exit 124 2026-10-24T10:00:00")
        write(kit, "t-grade-15.done", "PASS 2026-10-24T10:00:00")
        rc, out = run(["summary", "--box", box, "--kit", kit])
        expect(rc == 1 and "cross: FAIL" in out, "summary: a failed step fails 4.4")

    print(f"test_grade_f3: {'PASS' if fails == 0 else 'FAIL'} ({fails} failure(s))")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
