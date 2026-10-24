#!/usr/bin/env python3
"""tools/precache_gpu/grade_f3.py on constructed logs: commissioning accepts only a met must-accept beside a missed
must-reject from a different build, grading refuses an unsigned receipt and a reading by an uncommissioned runner,
and the summary reports a missing .done marker and a failing suite."""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "precache_gpu"))
import grade_f3  # noqa: E402

A, R, X = "a" * 64, "b" * 64, "c" * 64


def log(role, runner, adapter, model, rows):
    out = [f"10.4 run role={role} runner={runner} adapter={adapter} model={model} model_sha256={'d' * 64} pairs=8 "
           f"bar=100 tick_ms=0.000100"]
    for L, lo, hi, bar in rows:
        out.append(f"10.4 reading role={role} runner={runner} adapter={adapter} model={model} L={L} pairs=8 "
                   f"bytes_including_digest=1000 prefill_ms=900.000 restore_ms=2.000 ratio95=[{lo},{hi}] resolved=1 "
                   f"bar={bar}")
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
        acc = write(d, "acc.log", log("accept", A, "10de_1e81", "1p5", met) + log("accept", A, "10de_1e81", "0p5", met))
        rej = write(d, "rej.log", log("reject", R, "10de_1e81", "1p5", missed) + log("reject", R, "10de_1e81", "0p5", missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej])
        expect(rc == 0 and "verdict: COMMISSIONABLE" in out and "receipt: PC-GPU-TIMING-1" in out,
               "must-accept met + must-reject missed, different builds: commissionable")

        rej_met = write(d, "rejmet.log", log("reject", R, "10de_1e81", "1p5", met) + log("reject", R, "10de_1e81", "0p5", missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej_met])
        expect(rc == 1 and "NOT COMMISSIONABLE" in out, "a must-reject that reads met refuses commissioning")

        rej_same = write(d, "rejsame.log", log("reject", A, "10de_1e81", "1p5", missed) + log("reject", A, "10de_1e81", "0p5", missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej_same])
        expect(rc == 1 and "same runner" in out, "a must-reject run by the must-accept's build refuses commissioning")

        acc_miss = write(d, "accmiss.log", log("accept", A, "10de_1e81", "1p5", missed))
        rc, out = run(["commission", "--accept", acc_miss, "--reject", rej])
        expect(rc == 1, "a must-accept that misses refuses commissioning")

        rej_short = write(d, "rejshort.log", log("reject", R, "10de_1e81", "1p5", missed))
        rc, out = run(["commission", "--accept", acc, "--reject", rej_short])
        expect(rc == 1 and "no reading" in out, "a must-reject missing an accepted length refuses commissioning")

        # The XTX's commissioning, then a receipt over both adapters.
        accx = write(d, "accx.log", log("accept", X, "1002_744c", "1p5", met))
        rejx = write(d, "rejx.log", log("reject", R, "1002_744c", "1p5", missed))
        rc, out = run(["commission", "--accept", acc, accx, "--reject", rej, rejx])
        tmpl = out.split("-----\n", 1)[1]
        unsigned = write(d, "unsigned.txt", tmpl)
        g_ok = write(d, "g.log", log("grade", A, "10de_1e81", "1p5", met) + log("grade", A, "10de_1e81", "0p5", met)
                     + log("grade", X, "1002_744c", "1p5", met))
        rc, out = run(["grade", "--receipt", unsigned, "--log", g_ok])
        expect(rc == 1 and "unsigned" in out, "an unsigned receipt is refused before any reading is graded")

        signed = write(d, "signed.txt", tmpl.replace("signed_by: ", "signed_by: debunker")
                       .replace("signed_at_utc: ", "signed_at_utc: 2026-10-25T00:00:00Z"))
        rc, out = run(["grade", "--receipt", signed, "--log", g_ok])
        expect(rc == 0 and "verdict: MET" in out, "six met readings by the commissioned runners: MET")

        g_part = write(d, "gpart.log", log("grade", A, "10de_1e81", "1p5", met))
        rc, out = run(["grade", "--receipt", signed, "--log", g_part])
        expect(rc == 1 and "absent" in out, "a required reading absent: not met")

        g_other = write(d, "gother.log", log("grade", R, "10de_1e81", "1p5", met) + log("grade", A, "10de_1e81", "0p5", met)
                        + log("grade", X, "1002_744c", "1p5", met))
        rc, out = run(["grade", "--receipt", signed, "--log", g_other])
        expect(rc == 1 and "not the commissioned" in out, "a reading by an uncommissioned build is not graded")

        g_miss = write(d, "gmiss.log", log("grade", A, "10de_1e81", "1p5", [(128, 60.0, 90.0, "missed"), (1000, 900.0, 1010.0, "met")])
                       + log("grade", A, "10de_1e81", "0p5", met) + log("grade", X, "1002_744c", "1p5", met))
        rc, out = run(["grade", "--receipt", signed, "--log", g_miss])
        expect(rc == 1 and "MISSED" in out and "R1b" in out, "one missed reading: not met, R1b named")

        res = os.path.join(d, "results")
        os.mkdir(res)
        write(res, "a.log", "pcg_cell_pool: checks=10 failures=0 -> PASS\n")
        write(res, "a.done", "PASS 2026-10-24T10:00:00")
        write(res, "m.log", "pcg_cell_6_5: checks=10 failures=1 skips=0 -> FAIL\n")
        write(res, "m.done", "PASS 2026-10-24T10:00:00")
        write(res, "rest.log", "rest: 300 s\n")
        rc, out = run(["summary", res])
        expect(rc == 0 and "-> FAIL" in out, "summary: a killed mutant's FAIL suite under a PASS step is a pass")
        write(res, "b.log", "pcg_cell_6_5: checks=10 failures=1 skips=0 -> FAIL\n")
        write(res, "b.done", "FAIL pcg_cell_6_5 exit 1 2026-10-24T10:00:00")
        write(res, "c.log", "partial\n")
        rc, out = run(["summary", res])
        expect(rc == 1 and "c: NOT DONE" in out and "b: FAIL" in out and "rest" not in out,
               "summary: a failed step and a step with no marker")

    print(f"test_grade_f3: {'PASS' if fails == 0 else 'FAIL'} ({fails} failure(s))")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
