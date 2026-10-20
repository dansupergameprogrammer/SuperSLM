#!/usr/bin/env python3
"""Seeded mechanical partition of the T-3103 abstract design matrix (plan rev 15 s5.3, s8 R14-W1).

Run by the conductor, before any row is written. Reads only design-matrix.json: abstract slots, with
no row text. For every signature (identical abstract units), it shuffles the units with a seeded
SHA-256 counter generator and deals them to the splits in the signature's quota. Each split's
counts per stratum x construction x cmp x watch therefore equal the matrix's quotas: the partition is
stratified by those keys and finer. It then checks both parts:
  - every coarse cell quota, per split;
  - the plan's own floors (s0.1, s5.2, s5.3), hard-coded here from the plan text, not read from the
    matrix;
  - pairs and families never spanning a split;
  - the within-stratum balance tolerance: bias-corrected Cramer's V (one observation per paraphrase
    family for family-level attributes, per row for row-level ones), failing any table above the
    tolerance; each table reports its resolving power, and passes on tables that cannot resolve the
    tolerance are listed as unresolved; the uncorrected row-level V is reported beside it.
On any failure it reseeds with a recorded derived seed, up to --max-attempts.

Usage:
  py -3 partition_matrix.py --matrix <design-matrix.json> --seed <integer> --out <new directory>

Writes <out>/eval-part.json (splits G, H, dev, RE), <out>/tv-part.json (T, V, RT) and
<out>/partition-summary.json. Prints the seed, the attempt, every output hash and the per-part and
per-split counts. Exit 0 = PARTITION OK; 1 = every attempt failed a check (reasons printed); 2 = usage
or input error. It refuses to write into an existing output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_SPLITS = ("G", "H", "dev", "RE")
TV_SPLITS = ("T", "V", "RT")
ALL_SPLITS = EVAL_SPLITS + TV_SPLITS


# ------------------------------------------------------------------ deterministic generator

class Stream:
    """SHA-256 counter stream; identical output on every platform and Python version."""

    def __init__(self, seed: int):
        self.seed, self.counter, self.buf = seed, 0, b""

    def _bytes(self, n):
        while len(self.buf) < n:
            self.buf += hashlib.sha256(f"T-3103-partition:{self.seed}:{self.counter}".encode()).digest()
            self.counter += 1
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def below(self, n: int) -> int:
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = int.from_bytes(self._bytes(8), "little")
            if x < limit:
                return x % n

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]


def derived_seed(seed: int, attempt: int) -> int:
    if attempt == 0:
        return seed
    return int.from_bytes(hashlib.sha256(f"T-3103-reseed:{seed}:{attempt}".encode()).digest()[:8], "little")


# ------------------------------------------------------------------ partition

def deal(matrix: dict, seed: int) -> dict:
    stream = Stream(seed)
    by_sig = defaultdict(list)
    for unit in matrix["units"]:
        by_sig[unit["signature"]].append(unit["unit"])
    assignment = {}
    for sig in sorted(by_sig):
        quota = matrix["signatures"][sig]["quota"]
        units = sorted(by_sig[sig])
        labels = [s for s in ALL_SPLITS for _ in range(quota.get(s, 0))]
        if len(labels) != len(units):
            raise ValueError(f"signature {sig}: {len(units)} units, quota {len(labels)}")
        stream.shuffle(units)
        assignment.update(zip(units, labels))
    return assignment


def rows_with_split(matrix: dict, assignment: dict) -> list[dict]:
    cols = matrix["columns"]
    out = []
    for unit in matrix["units"]:
        for values in unit["rows"]:
            row = dict(zip(cols, values))
            row["unit"] = unit["unit"]
            row["split"] = assignment[unit["unit"]]
            out.append(row)
    return out


# ------------------------------------------------------------------ checks

F_CELLS = [f"F{i}" for i in range(1, 13)]
F_CONSTRUCTIONS = {
    "F1": ["when", "once", "as_soon_as", "the_moment", "if"],
    "F2": ["when", "once", "as_soon_as", "the_moment", "if", "after"],
    "F3": ["drops_below", "at_least", "or_less", "reaches", "above"],
    "F4": ["fewer_than", "at_least", "none_of", "all", "more_than"],
    "F5": ["unless", "if_not", "neg_action_while"],
    "F6": ["dont_until", "hold_fire_till", "not_before", "never_until", "dont_before"],
    "F7": ["until", "keep_till"], "F8": ["while", "as_long_as"], "F9": ["and", "and_also"],
    "F10": ["or", "either"],
}
GATE_F = {"F1": 40, "F2": 40, "F3": 50, "F4": 50, "F5": 90, "F6": 120, "F7": 90, "F8": 40, "F11": 40}
GEN = ["conjunction", "punctuation_free", "correction", "shared_argument", "pronoun", "nested_condition",
       "parallel", "cancellation", "zero_action", "above_cap"]
UC = ["correction", "shared_argument", "pronoun", "nested_condition", "parallel"]
COVERED_T = (["GEN-" + g for g in GEN if g not in ("cancellation", "zero_action")] + ["UC-" + u for u in UC]
             + ["CAP-at_cap", "CAP-cap_plus_one", "PC2-simple", "PC2-dependency", "PC2-ordering"]
             + F_CELLS + [f"B{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 12)])
NONCMP = ("below", "at_least", "at_most", "reaches", "above")


def _table(pairs):
    table = Counter(pairs)
    xs = sorted({a for a, _ in pairs})
    ys = sorted({b for _, b in pairs})
    n = len(pairs)
    rx = Counter(a for a, _ in pairs)
    ry = Counter(b for _, b in pairs)
    chi2 = sum((table[(x, y)] - rx[x] * ry[y] / n) ** 2 / (rx[x] * ry[y] / n) for x in xs for y in ys)
    return chi2, n, len(xs), len(ys)


def cramers_v(pairs):
    """Uncorrected Cramer's V (reported)."""
    if len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return 0.0
    chi2, n, r, k = _table(pairs)
    return math.sqrt(chi2 / (n * (min(r, k) - 1)))


def cramers_v_corrected(pairs):
    """Bias-corrected Cramer's V (Bergsma 2013), the decisive balance statistic."""
    if len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return 0.0
    chi2, n, r, k = _table(pairs)
    if n < 2:
        return 0.0
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return math.sqrt(phi2 / denom) if denom > 0 else 0.0


def balance_report(rows, spec):
    """Within aspect x measure strata of one scope's F-cell rows: for each nonsemantic attribute and
    label, the bias-corrected V over the attribute's observation unit (one per paraphrase family for
    family-level attributes, one per row for row-level attributes), with the uncorrected row-level V
    reported beside it."""
    findings = []
    strata = defaultdict(list)
    for r in rows:
        if r["stratum"] in F_CELLS and not r["arm"]:
            measure = {"F3": "measured", "F4": "count"}.get(r["stratum"], "plain")
            strata[(r["aspect"], measure)].append(r)
    for key, rs in sorted(strata.items()):
        if len(rs) < spec["min_rows"]:
            continue
        families = {}
        for r in rs:
            families.setdefault((r["unit"], r["stratum"], r["construction"]), r)
        for attr in spec["nonsemantic"]:
            exempt = set(spec["exempt"].get(attr, []))
            obs = families.values() if spec["observation"][attr] == "family" else rs
            for label in spec["labels"]:
                pairs = [(r[attr], r[label]) for r in obs if r["stratum"] not in exempt]
                row_pairs = [(r[attr], r[label]) for r in rs if r["stratum"] not in exempt]
                if not pairs:
                    continue
                r_levels = len({a for a, _ in pairs})
                k_levels = len({b for _, b in pairs})
                if r_levels < 2 or k_levels < 2:
                    continue
                # Resolving power: the corrected V a chi-square at its 95th percentile under independence
                # would produce (normal approximation). A pass on a table whose resolving power exceeds
                # the tolerance is "no association beyond chance detected", not "V within tolerance".
                df = (r_levels - 1) * (k_levels - 1)
                n_obs = len(pairs)
                chi95 = df + 1.645 * math.sqrt(2 * df)
                floor = math.sqrt(max(0.0, chi95 / n_obs - df / (n_obs - 1)) / (min(r_levels, k_levels) - 1))
                findings.append({"stratum": "/".join(key), "attr": attr, "label": label, "n": len(pairs),
                                 "v": round(cramers_v_corrected(pairs), 4),
                                 "v_rows_uncorrected": round(cramers_v(row_pairs), 4),
                                 "resolving_power": round(floor, 4),
                                 "resolved": floor <= spec["tolerance"]})
    return findings


def check(matrix: dict, rows: list[dict]) -> tuple[list[str], dict]:
    errors = []
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    core = {s: [r for r in rs if not r["arm"]] for s, rs in by_split.items()}

    # 1. coarse quotas per split, exactly as the matrix states them
    got = defaultdict(Counter)
    for r in rows:
        got[(r["stratum"], r["construction"], r["cmp"], r["watch"], r["arm"])][r["split"]] += 1
    for key, quota in matrix["coarse_quotas"]:
        k = tuple(key)
        for s in ALL_SPLITS:
            if got[k][s] != quota.get(s, 0):
                errors.append(f"coarse quota {k} {s}: {got[k][s]} != {quota.get(s, 0)}")
    # 2. units never span a split; pairs and families are units
    unit_split = {}
    for r in rows:
        if unit_split.setdefault(r["unit"], r["split"]) != r["split"]:
            errors.append(f"unit {r['unit']} spans splits")
    # 3. reserved vocabulary only in G, H, RE; never dev, T, V, RT
    for s in ("dev", "T", "V", "RT"):
        n = sum(r["reserved"] for r in by_split[s])
        if n:
            errors.append(f"{s}: {n} reserved-vocabulary rows")

    def count(s, pred):
        return sum(1 for r in core[s] if pred(r))

    def atoms(s, pred):
        return sum(r["atoms"] for r in core[s] if pred(r))

    # 4. held-out floors (s5.2)
    H = "H"
    for cell in F_CELLS:
        n = count(H, lambda r: r["stratum"] == cell)
        if n < (120 if cell == "F2" else 100):
            errors.append(f"H {cell}: {n} rows")
        if cell in ("F1", "F2", "F3", "F4", "F5", "F7", "F8"):
            p = count(H, lambda r: r["stratum"] == cell and r["possessive"] == "yes")
            if p < 20:
                errors.append(f"H {cell}: {p} possessive rows < 20")
        place = Counter(r["placement"] for r in core[H] if r["stratum"] == cell)
        for pl, floor in (("fronted", 20), ("postposed", 20), ("anaphoric", 20), ("mid_subject_verb", 8),
                          ("mid_verb_object", 8), ("mid_vocative", 8)):
            if place[pl] < floor:
                errors.append(f"H {cell}: placement {pl} {place[pl]} < {floor}")
        total_atoms = atoms(H, lambda r: r["stratum"] == cell)
        for w in ("reuse", "new"):
            a = atoms(H, lambda r: r["stratum"] == cell and r["watch"] == w)
            if a * 10 < total_atoms:
                errors.append(f"H {cell}: watch {w} {a}/{total_atoms} atoms < 10%")
    for cell, cons in F_CONSTRUCTIONS.items():
        for c in cons:
            n = count(H, lambda r: r["stratum"] == cell and r["construction"] == c)
            if n < 20:
                errors.append(f"H {cell}/{c}: {n} rows < 20")
    for cmp in NONCMP:
        a = atoms(H, lambda r: r["stratum"] in ("F3", "F4") and r["cmp"] == cmp)
        if a < 40:
            errors.append(f"H F3+F4 cmp {cmp}: {a} atoms < 40")
    for prefix, cells, floor in (("", [f"B{i}" for i in range(1, 6)], 100), ("", [f"D{i}" for i in range(1, 12)], 40),
                                 ("", [f"S{i}" for i in range(1, 8)], 20), ("GEN-", GEN, 100), ("UC-", UC, 40)):
        for c in cells:
            n = count(H, lambda r: r["stratum"] == prefix + c)
            if n < floor:
                errors.append(f"H {prefix}{c}: {n} < {floor}")
    for cond in ("simple", "dependency", "ordering"):
        for w in ("reuse", "new"):
            n = count(H, lambda r: r["stratum"] == "PC2-" + cond and r["watch"] == w)
            if n != 40:
                errors.append(f"H PC2-{cond} {w}: {n} != 40")
    if count(H, lambda r: r["stratum"] == "CAP-at_cap") != 40 or \
            count(H, lambda r: r["stratum"] == "CAP-cap_plus_one") != 40:
        errors.append("H cap boundary is not 40 + 40")
    if count(H, lambda r: r["construction"] == "inserted_wait") < 10 or \
            count(H, lambda r: r["construction"] == "two_waits") < 5:
        errors.append("H cap-plus-one inserted-wait cases short")
    if count(H, lambda r: r["stratum"] == "N-ambiguous_executable") != 40:
        errors.append("H ambiguous executable != 40")
    cond_atoms = atoms(H, lambda r: True)
    res_atoms = atoms(H, lambda r: r["reserved"])
    if res_atoms * 10 < cond_atoms * 3:
        errors.append(f"H reserved atoms {res_atoms}/{cond_atoms} < 30%")
    # 5. gate (s0.1): exact composition, with and without the detection arm
    G = "G"
    for cell, n in GATE_F.items():
        got_n = count(G, lambda r: r["stratum"] == cell)
        if got_n != n:
            errors.append(f"G {cell}: {got_n} != {n}")
    other = [r for r in core[G] if r["stratum"] not in GATE_F]
    if other:
        errors.append(f"G: {len(other)} non-slice rows outside the detection arm")
    if len(core[G]) != 560:
        errors.append(f"G core rows {len(core[G])} != 560")
    if any(r["atoms"] != 1 for r in core[G]):
        errors.append("G: a gate row is not single-atom")
    if any(r["role"] not in ("start", "start_any") for r in core[G] if r["stratum"] in ("F3", "F4")):
        errors.append("G: F3/F4 row not start-side")
    for cmp in NONCMP:
        a = count(G, lambda r: r["stratum"] in ("F3", "F4") and r["cmp"] == cmp)
        if a != 20:
            errors.append(f"G F3+F4 cmp {cmp}: {a} != 20")
    if count(G, lambda r: r["role"] in ("start", "start_any", "during")) != 470:
        errors.append("G start-side or during atoms != 470")
    if count(G, lambda r: r["stratum"] in ("F5", "F6")) != 210 or count(G, lambda r: r["stratum"] in ("F6", "F7")) != 210:
        errors.append("G F5+F6 or F6+F7 != 210")
    for w in ("reuse", "new"):
        if count(G, lambda r: r["watch"] == w) < 60:
            errors.append(f"G watch {w} < 60")
    if count(G, lambda r: r["reserved"]) < 210:
        errors.append(f"G reserved atoms {count(G, lambda r: r['reserved'])} < 210")
    arm = [r for r in by_split[G] if r["arm"]]
    if arm:
        if len(arm) != 460 or sum(r["stratum"] == "A0-plain" for r in arm) != 230:
            errors.append(f"G detection arm: {len(arm)} rows")
        fam = Counter(r["stratum"] for r in arm if r["stratum"] != "A0-plain")
        if set(fam) != {"D1", "D3", "D5", "D6", "D7", "D8", "D9", "D10", "D11"} or min(fam.values()) < 25:
            errors.append(f"G detection arm decoys {dict(fam)}")
    if any(r["arm"] for s in ALL_SPLITS if s != G for r in by_split[s]):
        errors.append("detection-arm rows outside G")
    # 6. development
    if len(by_split["dev"]) < 200:
        errors.append(f"dev {len(by_split['dev'])} < 200")
    # 7. training splits (s5.3)
    for s in COVERED_T:
        h = count(H, lambda r: r["stratum"] == s)
        t = count("T", lambda r: r["stratum"] == s)
        if t < 2 * h:
            errors.append(f"T {s}: {t} < 2 x {h}")
    for cell, cons in F_CONSTRUCTIONS.items():
        for c in cons:
            t = count("T", lambda r: r["stratum"] == cell and r["construction"] == c)
            if t < 40:
                errors.append(f"T {cell}/{c}: {t} < 40")
    if len(by_split["T"]) < 6960:
        errors.append(f"T {len(by_split['T'])} < 6960")
    excluded = [r for r in by_split["T"] + by_split["V"]
                if r["stratum"] in ("GEN-cancellation", "GEN-zero_action", "N-ambiguous_executable")
                or r["stratum"].startswith("S")]
    if excluded:
        errors.append(f"T/V carry {len(excluded)} no-action, ambiguous or stress rows")
    if len(by_split["V"]) < 600:
        errors.append(f"V {len(by_split['V'])} < 600")
    single_atom_v = sum(1 for r in by_split["V"] if r["atoms"] == 1 and (
        r["stratum"] in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F11", "F12") or r["stratum"].startswith("PC2")))
    if single_atom_v < 300:
        errors.append(f"V single-atom rows {single_atom_v} < 300")
    # 8. balance
    spec = matrix["balance"]
    balance = {}
    scopes = {"eval": [r for s in EVAL_SPLITS for r in core.get(s, [])],
              "tv": [r for s in TV_SPLITS for r in core.get(s, [])]}
    scopes.update({s: core.get(s, []) for s in ALL_SPLITS})
    for name, rs in scopes.items():
        findings = balance_report(rs, spec)
        resolved = [f for f in findings if f["resolved"]]
        worst = max(findings, key=lambda f: f["v"], default=None)
        balance[name] = {"tables": len(findings), "resolved_tables": len(resolved),
                         "max_v": worst["v"] if worst else 0.0, "worst": worst,
                         "max_v_rows_uncorrected": max((f["v_rows_uncorrected"] for f in findings), default=0.0),
                         "unresolved": [f"{f['stratum']} {f['attr']}x{f['label']} n={f['n']} resolving_power={f['resolving_power']}"
                                        for f in findings if not f["resolved"]]}
        if name in spec["decisive_scopes"]:
            for f in findings:
                if f["v"] > spec["tolerance"]:
                    errors.append(f"balance {name} {f['stratum']} {f['attr']}x{f['label']}: V {f['v']} > "
                                  f"{spec['tolerance']} (n {f['n']})")
    return errors, balance


# ------------------------------------------------------------------ main

def canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-attempts", type=int, default=20)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"USAGE ERROR: output directory {out} exists; choose a new one")
        return 2
    try:
        raw = Path(args.matrix).read_bytes()
        matrix = json.loads(raw)
    except (OSError, ValueError) as error:
        print(f"INPUT ERROR: {error}")
        return 2
    matrix_sha = hashlib.sha256(raw).hexdigest()
    print(f"matrix {args.matrix}")
    print(f"matrix_sha256 {matrix_sha}")
    print(f"seed {args.seed}")
    failures = []
    for attempt in range(args.max_attempts):
        used = derived_seed(args.seed, attempt)
        assignment = deal(matrix, used)
        rows = rows_with_split(matrix, assignment)
        errors, balance = check(matrix, rows)
        if not errors:
            break
        failures.append({"attempt": attempt, "seed": used, "errors": errors[:20], "error_count": len(errors)})
        print(f"attempt {attempt} seed {used}: {len(errors)} check failures; reseeding")
    else:
        for f in failures[-1]["errors"]:
            print("FAILED: " + f)
        print(f"PARTITION FAILED after {args.max_attempts} attempts")
        return 1
    cols = matrix["columns"]
    out_cols = ["split", "unit"] + cols

    def part(splits):
        sel = sorted((r for r in rows if r["split"] in splits), key=lambda r: r["row"])
        return {"part_of": "T-3103 design matrix", "matrix_sha256": matrix_sha, "columns": out_cols,
                "rows": [[r[c] for c in out_cols] for r in sel]}

    eval_part, tv_part = part(EVAL_SPLITS), part(TV_SPLITS)
    assign_all = sorted((u, s) for u, s in assignment.items())
    arm_units = {r["unit"] for r in rows if r["arm"] and r["stratum"] in ("A0-plain",)} | \
        {r["unit"] for r in rows if r["arm"]}
    # Without the arm: arm-only units are dropped; mixed pair units keep their non-arm rows' split.
    assign_no_arm = sorted((r["row"], r["split"]) for r in rows if not r["arm"])
    hashes = {"assignment_sha256": hashlib.sha256(canonical(assign_all)).hexdigest(),
              "assignment_without_detection_arm_sha256": hashlib.sha256(canonical(assign_no_arm)).hexdigest()}
    out.mkdir(parents=True)
    files = {"eval-part.json": canonical(eval_part), "tv-part.json": canonical(tv_part)}
    for name, data in files.items():
        (out / name).write_bytes(data)
        hashes[name.replace(".json", "") + "_sha256"] = hashlib.sha256(data).hexdigest()
    per_split = Counter(r["split"] for r in rows)
    per_split_no_arm = Counter(r["split"] for r in rows if not r["arm"])
    per_part = {"eval": sum(per_split[s] for s in EVAL_SPLITS), "tv": sum(per_split[s] for s in TV_SPLITS)}
    summary = {"script": "tests/t3094-multicommand-s0/matrix/partition_matrix.py", "matrix_sha256": matrix_sha,
               "seed": args.seed, "attempt": attempt, "seed_used": used, "failed_attempts": failures,
               "hashes": hashes, "rows_per_part": per_part,
               "rows_per_split": {s: per_split[s] for s in ALL_SPLITS},
               "rows_per_split_without_detection_arm": {s: per_split_no_arm[s] for s in ALL_SPLITS},
               "detection_arm_rows": sum(1 for r in rows if r["arm"]), "detection_arm_units": len(arm_units),
               "balance": balance, "checks": "all passed"}
    summary_bytes = canonical(summary)
    (out / "partition-summary.json").write_bytes(summary_bytes)
    print(f"attempt {attempt} seed_used {used}")
    for k, v in hashes.items():
        print(f"{k} {v}")
    print(f"partition-summary_sha256 {hashlib.sha256(summary_bytes).hexdigest()}")
    print("rows_per_part " + json.dumps(per_part))
    print("rows_per_split " + json.dumps({s: per_split[s] for s in ALL_SPLITS}))
    print("rows_per_split_without_detection_arm " + json.dumps({s: per_split_no_arm[s] for s in ALL_SPLITS}))
    print("max_balance_v " + json.dumps({k: v["max_v"] for k, v in balance.items()}))
    print("balance_tables_resolved " + json.dumps({k: [v["resolved_tables"], v["tables"]] for k, v in balance.items()}))
    print("PARTITION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
