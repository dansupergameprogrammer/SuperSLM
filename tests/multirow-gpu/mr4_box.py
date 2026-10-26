"""Multi-row GPU plan (rev 1.3.2), MR4's box cells 10.2 and 10.5 (Canonical), graded from MB4's measurements. [box]

Run on the box's output (MB4.1, MB4.2); the cloud runs only `--self-test`. Exact integer arithmetic, plan §5.1 (C18):
every product in Python ints, divisions as exact floor and ceiling, never floats.

  10.2  python3 mr4_box.py check102 TABLE.csv
        The cost table the 10.2 bench files (rulings item 12: the builder's bench writes it). One row per
        (model, M, logits): `model,M,logits,runs,warmup,median_us,p95_us,max_us,clock[,wall_*]`, M = 1..8, logits 0 (C(M),
        the recorder's GPU time: clock `gpu`) and 1 (the wall stand-in for the logits stage: clock `wall`, information
        only), for the 0.5B and the 1.5B. Ruling 18b: r(M) comes only from the logits = 0 rows, and g(M) is never derived
        from the logits = 1 rows (g stays at its estimate until MR5's GPU bracket). Red if a row is missing, if runs != 30
        or warmup != 1, if median <= p95 <= max fails, if a logits = 0 row is not clock `gpu`, if a logits = 1 row claims
        clock `gpu` (flagged, not used: no GPU logits timing exists before MR5), or if MR0's rule on the port fails:
        C(4) / (4 x C(1)) <= 0.6 at the median (MB2's gate, re-checked here). Prints r(M) per mille from the GPU maxima
        (MR-D15), the values the shipped header takes for this card (MR7 takes the per-M maximum across certified cards).
  10.5  python3 mr4_box.py canon105 TABLE.csv [--margin=150] [--spans=3,5,6,5,2]   RETIRED at MRX (prints RETIRED)
        Canonical over B2's spans, red if not faster: the M-row frames are >= 1.17's one-row Canonical frames at the default
        budget (one row per frame: sum of spans). C18: frames are decided from the configuration alone, so they follow from
        the measured table: per span, chunks of up to M_max rows (the chooser of §5.1: the M in 1..8 maximizing M / D(M),
        ties to fewer rows), each chunk of k rows run on the smallest variant H in {2, 4, 8} >= k (k = 1 runs one-row) and
        taking D(H) ticks. B_eff = 6,200 us (the knob at 0: one row's cost); C(M) at its maximum; m per mille. Run at the
        default margin (150) and at 0, on both models.
  --self-test   checks the arithmetic against §4.7's and §5.2's worked figures (COST's [A] inputs).
"""
import csv
import sys

B_EFF_US = 6200
LAYERS = {"0.5B": 24, "1.5B": 28}


def ceil_div(a, b):
    return -((-a) // b)


def ell(L, b_eff, c_m, m):
    # plan §5.1: l(M) = floor(L x B_eff x 1000 / (C(M) x (1000 + m)))
    return (L * b_eff * 1000) // (c_m * (1000 + m))


def chooser(L, b_eff, cost, m):
    """M_max and D per variant: cost = {M: C(M) us}. M = 1 is priced at C(1) without the margin (one row is always admitted)."""
    d = {1: 1}
    for h in (2, 4, 8):
        l = ell(L, b_eff, cost[h], m)
        d[h] = None if l == 0 else ceil_div(L, l)
    best, best_m = None, 1
    for M in range(1, 9):
        h = 1 if M == 1 else next(x for x in (2, 4, 8) if x >= M)
        if d[h] is None:
            continue
        # compare M / D(M) exactly: M1 * D2 vs M2 * D1; ties to fewer rows
        if best is None or M * best[1] > best[0] * d[h]:
            best, best_m = (M, d[h]), M
    return best_m, d


def frames(spans, L, b_eff, cost, m):
    m_max, d = chooser(L, b_eff, cost, m)
    total = 0
    for s in spans:
        left = s
        while left > 0:
            k = min(left, m_max)
            h = 1 if k == 1 else next(x for x in (2, 4, 8) if x >= k)
            total += d[h]
            left -= k
    return total, m_max, d


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {k: int(r[k]) for k in ("runs", "warmup", "median_us", "p95_us", "max_us")}
            row["clock"] = (r.get("clock") or "").strip()
            rows[(r["model"], int(r["M"]), int(r["logits"]))] = row
    return rows


def clock_problems(rows):
    """Ruling 18b: logits = 0 rows are the GPU clock (they price r); logits = 1 rows are the wall stand-in (info only)."""
    bad = []
    for (model, M, logits), r in sorted(rows.items()):
        if logits == 0 and r["clock"] != "gpu":
            bad.append("%s M=%d logits=0: clock %r, want 'gpu' (r(M) is priced from the recorder's GPU time only)" % (model, M, r["clock"]))
        if logits == 1 and r["clock"] == "gpu":
            bad.append("%s M=%d logits=1: clock 'gpu' before MR5 (flagged, not used: g is never derived from these rows)" % (model, M))
        elif logits == 1 and r["clock"] != "wall":
            bad.append("%s M=%d logits=1: clock %r, want 'wall'" % (model, M, r["clock"]))
    return bad


def check102(path):
    rows = load(path)
    bad = clock_problems(rows)
    for model in LAYERS:
        for logits in (0, 1):
            for M in range(1, 9):
                r = rows.get((model, M, logits))
                if r is None:
                    bad.append("missing %s M=%d logits=%d" % (model, M, logits))
                    continue
                if r["runs"] != 30 or r["warmup"] != 1:
                    bad.append("%s M=%d logits=%d: %d runs, %d warm-up (want 30 and 1)" % (model, M, logits, r["runs"], r["warmup"]))
                if not r["median_us"] <= r["p95_us"] <= r["max_us"]:
                    bad.append("%s M=%d logits=%d: median %d, p95 %d, max %d out of order" % (model, M, logits, r["median_us"], r["p95_us"], r["max_us"]))
        c1, c4 = rows.get((model, 1, 0)), rows.get((model, 4, 0))
        if c1 and c4 and c4["median_us"] * 1000 > 600 * 4 * c1["median_us"]:
            bad.append("%s: port gate C(4) / (4 C(1)) = %d / %d > 0.6" % (model, c4["median_us"], 4 * c1["median_us"]))
        if c1:
            r = [ceil_div(rows[(model, M, 0)]["max_us"] * 1000, rows[(model, 1, 0)]["max_us"]) for M in range(1, 9) if (model, M, 0) in rows]
            print("10.2 %s: r(M) per mille %s (logits = 0, clock gpu); g(M) not derived (ruling 18b: the logits = 1 rows are"
                  " the wall stand-in, information only)" % (model, r))
    for b in bad:
        print("RED 10.2: " + b)
    print("%s 10.2" % ("RED" if bad else "PASS"))
    return 1 if bad else 0


def canon105(path, margin, spans):
    # MRX (E-11, ruling 20), marked, not deleted: Canonical's frames retire, or re-point to Exact's with MR6.
    rows = load(path)
    bad = [b for b in clock_problems(rows) if "logits=0" in b]  # C(M) is priced from the GPU clock only
    one_row = sum(spans)
    for model, L in LAYERS.items():
        if (model, 1, 0) not in rows:
            bad.append("no table for %s" % model)
            continue
        cost = {M: rows[(model, M, 0)]["max_us"] for M in range(1, 9) if (model, M, 0) in rows}
        for m in sorted({margin, 0}):
            f, m_max, d = frames(spans, L, B_EFF_US, cost, m)
            print("10.5 Canonical %s m=%d: M_max %d, D %s, %d frames against 1.17's one-row %d" % (model, m, m_max, d, f, one_row))
            if f >= one_row:
                bad.append("%s m=%d: %d M-row frames >= %d one-row frames" % (model, m, f, one_row))
    for b in bad:
        print("RED 10.5: " + b)
    print("%s 10.5 Canonical" % ("RED" if bad else "PASS"))
    return 1 if bad else 0


def self_test():
    # §4.7's inputs [A]: B_eff 6,200 us, L 24, C = {1: 6,200; 2: 7,900; 4: 11,500; 8: 18,600}
    cost = {1: 6200, 2: 7900, 4: 11500, 8: 18600}
    want = {(2, 0): (18, 2), (4, 0): (12, 2), (8, 0): (8, 3), (2, 150): (16, 2), (4, 150): (11, 3), (8, 150): (6, 4)}
    bad = []
    for (M, m), (l, D) in want.items():
        got = ell(24, 6200, cost[M], m)
        if got != l or ceil_div(24, got) != D:
            bad.append("l(%d) at m=%d: %d / D %d, want %d / %d" % (M, m, got, ceil_div(24, got), l, D))
    # §5.2: at m = 150 M_max = 8 with D = 4; at m = 0 D(8) = 3; the knife edge 24 x 6.2 / 18.6 = 8 exactly
    if chooser(24, 6200, cost, 150)[0] != 8 or chooser(24, 6200, cost, 150)[1][8] != 4 or chooser(24, 6200, cost, 0)[1][8] != 3:
        bad.append("chooser: %s / %s" % (chooser(24, 6200, cost, 150), chooser(24, 6200, cost, 0)))
    if ell(24, 6200, 18600, 0) != 8:
        bad.append("the knife edge l(8) at m = 0 is %d, want 8" % ell(24, 6200, 18600, 0))
    # B2's spans at m = 150: 3 -> H4 (D 3), 5, 6, 5 -> H8 (D 4 each), 2 -> H2 (D 2): 17 frames against 21
    f = frames([3, 5, 6, 5, 2], 24, 6200, cost, 150)[0]
    if f != 17:
        bad.append("B2's spans at m = 150: %d frames, want 17" % f)
    # Ruling 18b: the clock rules on a synthetic table.
    def table(clock0, clock1):
        return {("0.5B", M, lg): {"runs": 30, "warmup": 1, "median_us": 1, "p95_us": 1, "max_us": 1, "clock": clock0 if lg == 0 else clock1}
                for M in range(1, 9) for lg in (0, 1)}
    if clock_problems(table("gpu", "wall")):
        bad.append("clock rules: a well-formed table is flagged")
    if len(clock_problems(table("gpu", "gpu"))) != 8:
        bad.append("clock rules: logits = 1 rows with clock 'gpu' are not flagged")
    if len(clock_problems(table("wall", "wall"))) != 8:
        bad.append("clock rules: logits = 0 rows on the wall clock are not red")
    for b in bad:
        print("RED self-test: " + b)
    print("%s mr4_box self-test" % ("RED" if bad else "PASS"))
    return 1 if bad else 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) >= 3 and argv[1] == "check102":
        return check102(argv[2])
    if len(argv) >= 3 and argv[1] == "canon105":
        margin, spans = 150, [3, 5, 6, 5, 2]
        for a in argv[3:]:
            if a.startswith("--margin="):
                margin = int(a.split("=", 1)[1])
            elif a.startswith("--spans="):
                spans = [int(x) for x in a.split("=", 1)[1].split(",")]
        # MRX (E-11, ruling 22): RETIRED. There are no Canonical frames to price; re-points to Exact's with MR6.
        print("RETIRED 10.5 Canonical (MRX, red-suite-mrx.md): re-points to Exact's frames with MR6")
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
