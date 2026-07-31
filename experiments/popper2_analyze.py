# T-1421 analysis: detection rates with resolving power.

import json
import math
import re
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "popper2_results.jsonl")


# Committed test line RANGES in tests/test_main.cpp. MSVC reports __LINE__ at
# the last line of a multi-line macro invocation, so a range is the reliable
# unit. Derived by parsing the file's function starts, before analysis.
PIN_STRICT = range(14302, 14384)   # TestDecodeLoopFixtureRealCompositionMatchesItsOwnDerivedLogits
PIN_FAMILY = range(14302, 14731)   # + every committed RunGreedyDecodeLoop cell


def verdict_only(mutate_stdout):
    """Per-element MOVES/SILENT/REFUSES verdicts with the delta field stripped."""
    out = []
    for ln in mutate_stdout.splitlines():
        out.append(re.sub(r" delta=-?\d+", "", ln))
    return chr(10).join(out)


def in_range(lines, rng):
    return any(l in rng for l in lines)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n, label):
    lo, hi = wilson(k, n)
    r = k / n if n else float("nan")
    return (f"{label}: {k}/{n} = {r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
            f"(width {hi-lo:.3f})")


def main():
    recs = []
    with open(RESULTS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))

    status = Counter(r["status"] for r in recs)
    print("attempted:", len(recs))
    for k, v in sorted(status.items()):
        print(f"  {k}: {v}")

    with open(os.path.join(ROOT, "experiments", "popper2_reference.json"),
              encoding="utf-8") as fh:
        ref = json.load(fh)
    ref_verdicts = verdict_only(ref["mutate"])

    m = [r for r in recs if r["status"] == "MEASURED"]
    print(f"\nviable mutants (compiled and ran): {len(m)}")
    print("by operator class:", dict(Counter(r["site"]["kind"] for r in m)))
    print("by file:", dict(Counter(r["site"]["file"] for r in m)))

    # Instruments.
    def I_matrix(r):
        # The T-1409 construction's discrimination property: the Experiment C
        # matrix signature (per-element verdicts, not just the counts).
        return r["mutate_changed"]

    def I_matrix_verdicts(r):
        # Per-element MOVES/SILENT/REFUSES pattern, delta field ignored.
        return verdict_only(r["mutate_out"]) != ref_verdicts

    def I_matrix_counts(r):
        return r["mutate_summary_changed"]

    def I_pin_geom(r):
        # The pinned golden decode row as an oracle: the DecodeLoopFixture
        # geometry's decode output through RunGreedyDecodeLoop.
        return r["null_changed"]

    def I_pin_committed(r):
        return in_range(r["suite"]["fail_lines"], PIN_STRICT)

    def I_pin_family(r):
        return in_range(r["suite"]["fail_lines"], PIN_FAMILY)

    def I_suite(r):
        return r["suite"]["failures"] != 0 or r["suite_rc"] != 0

    # Denominators.
    decode_detectable = [r for r in m if I_pin_geom(r) or r["candidate_changed"]
                         or I_matrix(r)]
    suite_detectable = [r for r in m if I_suite(r)]
    any_detectable = [r for r in m if I_suite(r) or I_pin_geom(r)
                      or r["candidate_changed"] or I_matrix(r)]

    print(f"\ndecode-cell detectable (some decode observation moves): "
          f"{len(decode_detectable)}/{len(m)}")
    print(f"committed-suite detectable: {len(suite_detectable)}/{len(m)}")
    print(f"detectable by anything measured here: {len(any_detectable)}/{len(m)}")

    print("\n--- detection rates over the decode-cell-detectable population ---")
    D = decode_detectable
    n = len(D)
    print(fmt(sum(1 for r in D if I_matrix(r)), n,
              "I1 discrimination matrix (full per-element signature)"))
    print(fmt(sum(1 for r in D if I_matrix_verdicts(r)), n,
              "I1v discrimination matrix (per-element verdicts, delta ignored)"))
    print(fmt(sum(1 for r in D if I_matrix_counts(r)), n,
              "I1c discrimination matrix (MOVES/SILENT/REFUSES counts only)"))
    print(fmt(sum(1 for r in D if I_pin_geom(r)), n,
              "I2 golden pin geometry (decode output at DecodeLoopFixture)"))
    print(fmt(sum(1 for r in D if I_pin_committed(r)), n,
              "I2c golden pin as committed (suite CHECK lines)"))
    print(fmt(sum(1 for r in D if I_pin_family(r)), n,
              "I2f committed DecodeLoopFixture decode cells (whole family)"))
    print(fmt(sum(1 for r in D if I_suite(r)), n,
              "I3 whole committed suite"))

    print("\n--- detection rates over the suite-detectable population ---")
    S = suite_detectable
    n = len(S)
    print(fmt(sum(1 for r in S if I_matrix(r)), n, "I1 discrimination matrix"))
    print(fmt(sum(1 for r in S if I_pin_geom(r)), n, "I2 golden pin geometry"))
    print(fmt(sum(1 for r in S if I_pin_committed(r)), n, "I2c golden pin committed"))
    print(fmt(sum(1 for r in S if I_pin_family(r)), n, "I2f decode-cell family committed"))

    print("\n--- contingency: matrix vs golden pin, over decode-detectable ---")
    tab = Counter((I_matrix(r), I_pin_geom(r)) for r in decode_detectable)
    print("  matrix=Y pin=Y :", tab[(True, True)])
    print("  matrix=Y pin=N :", tab[(True, False)])
    print("  matrix=N pin=Y :", tab[(False, True)])
    print("  matrix=N pin=N :", tab[(False, False)])

    print("\n--- the blind spot: decode-detectable, matrix says nothing ---")
    blind = [r for r in decode_detectable if not I_matrix(r)]
    print(fmt(len(blind), len(decode_detectable),
              "matrix MISS rate over decode-detectable"))
    for r in blind:
        print(f"  idx={r['idx']} {r['site']['kind']} {r['site']['file']}:"
              f"{r['site']['line']} {r['site']['old']!r}->{r['site']['new']!r}"
              f" | pin_geom={I_pin_geom(r)} pin_committed={I_pin_committed(r)}"
              f" pin_family={I_pin_family(r)}"
              f" suite_fail={r['suite']['failures']}")

    print("\n--- what the matrix catches that the pin does not ---")
    for r in decode_detectable:
        if I_matrix(r) and not I_pin_geom(r):
            print(f"  idx={r['idx']} {r['site']['kind']} {r['site']['file']}:"
                  f"{r['site']['line']} {r['site']['old']!r}->"
                  f"{r['site']['new']!r} | {r['mutate_summary']}"
                  f" | suite_fail={r['suite']['failures']}")

    print("\n--- suite-detectable but decode-invisible at this cell ---")
    q = [r for r in m if I_suite(r) and not (I_pin_geom(r)
         or r["candidate_changed"] or I_matrix(r))]
    print(fmt(len(q), len(m), "suite-only (whole decode cell blind)"))
    print("  by file:", dict(Counter(r["site"]["file"] for r in q)))

    print("\n--- fully equivalent on every instrument measured ---")
    eq = [r for r in m if not I_suite(r) and not I_pin_geom(r)
          and not r["candidate_changed"] and not I_matrix(r)]
    print(fmt(len(eq), len(m), "no instrument moved"))
    print("  by file:", dict(Counter(r["site"]["file"] for r in eq)))

    print("\n--- reference: the prior pass's 9-defect sample ---")
    print(fmt(4, 9, "prior hand-picked miss rate 4/9"))


if __name__ == "__main__":
    main()
