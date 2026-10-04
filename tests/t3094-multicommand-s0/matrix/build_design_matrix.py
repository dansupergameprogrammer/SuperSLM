#!/usr/bin/env python3
"""T-3103 S0 abstract design matrix builder (plan rev 15 at d628055561: s5.2, s5.3, s8 R14-W1).

Writes design-matrix.json: abstract row slots only, with no utterance text, entity words or state
words. One row is one utterance slot, described by stratum x construction x action x role x polarity x
cmp x watch x placement x entity type x possessive (R14-W1). Rows are grouped into assignment
units: a paraphrase family of 4-7 slots sharing one meaning, a published pair of families (minimal pairs,
PC-2 arms), or a single slot. A unit never spans a split. Every unit carries a signature (the
abstract attributes of all its rows). The matrix gives, for each signature, how many units each split
(G gate, H held-out, dev, RE eval probe reserve, T, V, RT T/V probe reserve) receives. The conductor's
partition_matrix.py deals units with identical signatures to splits with a recorded seed.

Deterministic: no randomness, no clock. Re-running it reproduces design-matrix.json byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "design-matrix.json"

SPLITS = ("G", "H", "dev", "RE", "T", "V", "RT")
PART = {"G": "eval", "H": "eval", "dev": "eval", "RE": "eval", "T": "tv", "V": "tv", "RT": "tv"}
FAMILY_SPLITS = ("G", "H", "dev", "T", "V")
F_SPLITS = ("H", "G", "dev", "RE", "T", "V", "RT")   # balancing order within each part
ACTIONS = ("fire", "move", "scan", "hold", "wait")
DURATIVE = ("move", "scan", "hold", "wait")
ENTITY_TYPES = ("structure", "unit", "signal", "zone", "resource")
PLACEMENTS = (("fronted", 23), ("postposed", 23), ("anaphoric", 23), ("mid_subject_verb", 10),
              ("mid_verb_object", 10), ("mid_vocative", 11))
POSSESSIVE = (("no", 3), ("yes", 1))
COLUMNS = ("row", "stratum", "construction", "cmp", "watch", "action", "aspect", "role", "negated",
           "entity_type", "placement", "possessive", "atoms", "route", "arm", "reserved", "member")

# F cells: constructions with (name, cmp) and per-construction held-out (H) and gate (G) rows.
# H counts follow s5.2 except F3, which is 120 (finding MX-1). G counts follow s0.1.
F = {
    "F1": dict(cons=[("when", "none"), ("once", "none"), ("as_soon_as", "none"), ("the_moment", "none"),
                     ("if", "none")], H=[20] * 5, G=[8] * 5, measure="plain", rule="start"),
    "F2": dict(cons=[("when", "none"), ("once", "none"), ("as_soon_as", "none"), ("the_moment", "none"),
                     ("if", "none"), ("after", "none")], H=[20] * 6, G=[8, 8, 8, 8, 4, 4], measure="plain",
               rule="start"),
    "F3": dict(cons=[("drops_below", "below"), ("at_least", "at_least"), ("or_less", "at_most"),
                     ("reaches", "reaches"), ("above", "above")], H=[20, 20, 20, 40, 20], G=[8, 7, 8, 20, 7],
               measure="measured", rule="start"),
    "F4": dict(cons=[("fewer_than", "below"), ("at_least", "at_least"), ("none_of", "at_most"),
                     ("all", "at_least"), ("more_than", "above")], H=[20] * 5, G=[12, 7, 12, 6, 13],
               measure="count", rule="start"),
    "F5": dict(cons=[("unless", "none"), ("if_not", "none"), ("neg_action_while", "none")], H=[34, 33, 33],
               G=[30, 30, 30], measure="plain", rule="negated"),
    "F6": dict(cons=[("dont_until", "none"), ("hold_fire_till", "none"), ("not_before", "none"),
                     ("never_until", "none"), ("dont_before", "none")], H=[20] * 5, G=[24] * 5, measure="plain",
               rule="start"),
    "F7": dict(cons=[("until", "none"), ("keep_till", "none")], H=[50, 50], G=[45, 45], measure="plain",
               rule="stop"),
    "F8": dict(cons=[("while", "none"), ("as_long_as", "none")], H=[50, 50], G=[20, 20], measure="plain",
               rule="during"),
    "F9": dict(cons=[("and", "none"), ("and_also", "none")], H=[50, 50], G=[0, 0], measure="plain", rule="start"),
    "F10": dict(cons=[("or", "none"), ("either", "none")], H=[50, 50], G=[0, 0], measure="plain",
                rule="start_any"),
    "F11": dict(cons=[("-", "none")], H=[100], G=[40], measure="plain", rule="start"),
    "F12": dict(cons=[("-", "none")], H=[100], G=[0], measure="plain", rule="start"),
}
F_ORDER = ["F6", "F1", "F2", "F3", "F4", "F5", "F7", "F8", "F9", "F10", "F11", "F12"]
SINGLE_ATOM_F = {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F11", "F12"}
B = {"B1": 1, "B2": 1, "B3": "2-3", "B4": 2, "B5": 1}
D = [f"D{i}" for i in range(1, 12)]
S = {"S1": 3, "S2": 2, "S3": 1, "S4": 1, "S5": 2, "S6": 1, "S7": 1}
GEN = ["conjunction", "punctuation_free", "correction", "shared_argument", "pronoun", "nested_condition",
       "parallel", "cancellation", "zero_action", "above_cap"]
UC = ["correction", "shared_argument", "pronoun", "nested_condition", "parallel"]
PC2 = ["simple", "dependency", "ordering"]
CAP = {"at_cap": [("four_intents", 40, 1)],
       "cap_plus_one": [("five_requested", 25, 1), ("inserted_wait", 10, 1), ("two_waits", 5, 2)]}
ARM_DECOYS = {"D1": 26, "D3": 25, "D5": 25, "D6": 25, "D7": 26, "D8": 26, "D9": 26, "D10": 26, "D11": 25}
ARM_PLAIN = 230
GATE_WATCH_FAMILIES = 17          # per non-auto value: >= 17 x 4 = 68 >= 60 gate atoms (s0.1)
RESERVED_FRACTION = {"G": 0.40, "H": 0.35, "RE": 0.35}
V_FACTOR_SINGLE, V_FACTOR_OTHER = 0.15, 0.09
RESERVE_FACTOR = 0.10


def ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def family_sizes(n: int) -> list[int]:
    if n == 0:
        return []
    if n < 4:
        raise SystemExit(f"family quota {n} below 4")
    k = n // 4
    sizes = [4] * k
    for i in range(n - 4 * k):
        sizes[i % k] += 1
    return sizes


def smooth(weights, n):
    """Largest-remainder sequence of length n with the given integer weights, evenly interleaved."""
    total = sum(w for _, w in weights)
    out, given = [], Counter()
    for i in range(n):
        best = max(weights, key=lambda kv: ((i + 1) * kv[1] / total - given[kv[0]], -weights.index(kv)))
        out.append(best[0])
        given[best[0]] += 1
    return out


def evenly(count, n, phase):
    """`count` distinct indices of range(n), evenly spaced with the given phase in [0, 1)."""
    if count <= 0:
        return []
    picks = []
    for j in range(count):
        idx = int((j + phase) * n / count) % n
        while idx in picks:
            idx = (idx + 1) % n
        picks.append(idx)
    return picks


class Family:
    __slots__ = ("cell", "cons", "cmp", "size", "watch", "action", "entity", "atoms", "arm", "partner",
                 "fixed_action", "domain", "split", "placements", "possessives")

    def __init__(self, split, cell, cons, cmp, size, arm=False):
        self.split, self.cell, self.cons, self.cmp, self.size, self.arm = split, cell, cons, cmp, size, arm
        self.watch = "-"
        self.action = self.entity = None
        self.atoms = 1
        self.partner = None
        self.fixed_action = None
        self.domain = None
        self.placements = self.possessives = None


def f_counts(cell, split):
    spec = F[cell]
    if split == "H":
        return list(spec["H"])
    if split == "G":
        return list(spec["G"])
    if split == "T":
        return [2 * h for h in spec["H"]]
    if split == "V":
        f = V_FACTOR_SINGLE if cell in SINGLE_ATOM_F else V_FACTOR_OTHER
        return [max(4, ceil(f * 2 * h)) for h in spec["H"]]
    if split == "dev":
        n_fam = max(2, ceil(sum(spec["H"]) / 40))
        counts = [0] * len(spec["cons"])
        for i in range(n_fam):
            counts[i % len(counts)] += 4
        return counts
    if split == "RE":
        return [ceil(RESERVE_FACTOR * sum(t)) for t in zip(*(f_counts(cell, s) for s in ("G", "H", "dev")))]
    if split == "RT":
        return [ceil(RESERVE_FACTOR * sum(t)) for t in zip(*(f_counts(cell, s) for s in ("T", "V")))]
    raise ValueError(split)


def interleave(lists):
    out, i = [], 0
    while any(i < len(x) for x in lists):
        for x in lists:
            if i < len(x):
                out.append(x[i])
        i += 1
    return out


def assign_watch(fams, split, n_reuse=None, n_new=None):
    nf = len(fams)
    if nf == 0:
        return
    if n_reuse is None:
        n_reuse = n_new = ceil(0.15 * nf)
    reuse = evenly(n_reuse, nf, 0.25)
    new = [i for i in evenly(n_new, nf, 0.75) if i not in reuse]
    while len(new) < n_new:
        new.append(next(i for i in range(nf) if i not in reuse and i not in new))
    for i, fam in enumerate(fams):
        fam.watch = "reuse" if i in reuse else ("new" if i in new else "auto")


def greedy(items, domain_of, prefill_of, keyfuncs):
    counts = [Counter() for _ in keyfuncs]
    for item in items:
        value = prefill_of(item)
        if value is not None:
            for c, kf in zip(counts, keyfuncs):
                c[kf(item, value)] += item.size
    for item in items:
        if prefill_of(item) is not None:
            continue
        domain = domain_of(item)
        best = min(domain, key=lambda v: tuple(c[kf(item, v)] for c, kf in zip(counts, keyfuncs))
                   + (domain.index(v),))
        yield item, best
        for c, kf in zip(counts, keyfuncs):
            c[kf(item, best)] += item.size


def action_domain(fam):
    if fam.cell == "F6" and fam.cons == "hold_fire_till":
        return ("fire",)
    if fam.cell in ("F7", "F8"):
        return DURATIVE
    return ACTIONS


def entity_domain(fam):
    if fam.cell == "F11":
        return ("task",)
    if fam.cell == "F12":
        return ("unit",)
    return ENTITY_TYPES


SHARE_W = {"auto": 0.70, "reuse": 0.15, "new": 0.15}
SHARE_P = {name: w / 100 for name, w in PLACEMENTS}
SHARE_O = {"no": 0.75, "yes": 0.25}


def aspect_of(action):
    return "punctual" if action == "fire" else "durative"


COUNTS = Counter()          # shared across cells: the balance check pools cells within a stratum
TICK = [0]                  # rotates tie-break order so ties never favour one value systematically


def label_of(cell, cons, cmp, action):
    rule = F[cell]["rule"]
    asp = aspect_of(action)
    if rule == "negated":
        return ("start" if asp == "punctual" else "during", "true", cmp)
    return (rule, "false", cmp)


def rotated(seq):
    k = TICK[0] % len(seq)
    return list(seq[k:]) + list(seq[:k])


def balance_cell(part, cell, order, fams, watch_quota):
    """Joint balanced design for one F cell across one part's splits.

    A unit takes (watch, action, entity type) together. The cost is the share-normalized count of every
    table the balance check reads (action and entity type against the role/polarity/cmp label tuple and
    against watch, within aspect), at four scopes: split x cell, part x cell, split x measure stratum and
    part x measure stratum. Watch counts per split are exact quotas. Each row then takes placement
    and possessive the same way; a family's rows take distinct placements where it can."""
    counts = COUNTS
    meas = F[cell]["measure"]
    for split in order:
        items = fams[(split, cell)]
        remaining = dict(watch_quota[(split, cell)])
        for f in items:
            TICK[0] += 1
            if f.cell == "F5" and f.partner is not None and f.partner.cell == "F6":
                f.fixed_action, f.entity = f.partner.action, f.partner.entity
            if f.cell == "F7" and f.partner is not None and f.partner.cell == "F6":
                f.entity = f.partner.entity
            adom_full, edom_full = action_domain(f), entity_domain(f)
            adom = (f.fixed_action,) if f.fixed_action else rotated(adom_full)
            edom = (f.entity,) if f.entity else rotated(edom_full)
            sa, se = 1 / len(adom_full), 1 / len(edom_full)
            scopes = ((split, cell), (part, cell), (split, meas), (part, meas))
            best, best_key = None, None
            for wi, w in enumerate(rotated(("auto", "reuse", "new"))):
                if remaining[w] <= 0:
                    continue
                sw = SHARE_W[w]
                for ai, a in enumerate(adom):
                    asp, lab = aspect_of(a), label_of(cell, f.cons, f.cmp, a)
                    for ei, e in enumerate(edom):
                        score = 0.0
                        for sc in scopes:
                            score += (counts[(sc, "w", asp, w)] / sw + counts[(sc, "a", asp, a)] / sa
                                      + counts[(sc, "e", asp, e)] / se
                                      + counts[(sc, "la", lab, a)] / sa + counts[(sc, "le", asp, lab, e)] / se
                                      + counts[(sc, "wa", w, a)] / (sw * sa)
                                      + counts[(sc, "we", asp, w, e)] / (sw * se))
                        key = (round(score, 9), wi, ai, ei)
                        if best_key is None or key < best_key:
                            best, best_key = (w, a, e), key
            w, a, e = best
            f.watch, f.action, f.entity = w, a, e
            remaining[w] -= 1
            asp, lab = aspect_of(a), label_of(cell, f.cons, f.cmp, a)
            for sc in scopes:
                for k in ((sc, "w", asp, w), (sc, "a", asp, a), (sc, "e", asp, e), (sc, "la", lab, a),
                          (sc, "le", asp, lab, e), (sc, "wa", w, a), (sc, "we", asp, w, e)):
                    counts[k] += f.size
            f.placements, f.possessives = [], []
            for _ in range(f.size):
                TICK[0] += 1
                for attr, shares, out in (("p", SHARE_P, f.placements), ("o", SHARE_O, f.possessives)):
                    best_v, best_s = None, None
                    for vi, v in enumerate(rotated(list(shares))):
                        sh = shares[v]
                        s_ = 1000 if attr == "p" and v in out and len(out) < len(shares) else 0
                        for sc in scopes:
                            s_ += (counts[(sc, attr, asp, v)] / sh + counts[(sc, "l" + attr, asp, lab, v)] / sh
                                   + counts[(sc, "w" + attr, asp, w, v)] / (sh * SHARE_W[w]))
                        if best_s is None or (round(s_, 9), vi) < best_s:
                            best_v, best_s = v, (round(s_, 9), vi)
                    out.append(best_v)
                    for sc in scopes:
                        for k in ((sc, attr, asp, best_v), (sc, "l" + attr, asp, lab, best_v),
                                  (sc, "w" + attr, asp, w, best_v)):
                            counts[k] += 1


def build():
    fams = defaultdict(list)        # (split, cell) -> families
    units = []                      # list of (kind, group, [family-or-row-spec ...], split)

    # ---- F families
    for split in F_SPLITS:
        for cell in F:
            per_cons = []
            for (cons, cmp), n in zip(F[cell]["cons"], f_counts(cell, split)):
                sizes = [1] * n if split in ("RE", "RT") else family_sizes(n)
                per_cons.append([Family(split, cell, cons, cmp, s) for s in sizes])
            fams[(split, cell)] = interleave(per_cons)
            if cell in ("F9", "F10"):
                for i, fam in enumerate(fams[(split, cell)]):
                    fam.atoms = 2 if i % 2 == 0 else 3
    # ---- B, D, S families
    for split in FAMILY_SPLITS:
        for cell, atoms in B.items():
            n = {"H": 100, "T": 200, "V": ceil(V_FACTOR_OTHER * 200), "dev": 12, "G": 0}[split]
            fams[(split, cell)] = [Family(split, cell, "-", "none", s) for s in family_sizes(n)]
            for i, fam in enumerate(fams[(split, cell)]):
                fam.atoms = (2 if i % 2 == 0 else 3) if atoms == "2-3" else atoms
        for cell in D:
            n = {"H": 40, "T": 80, "V": ceil(V_FACTOR_OTHER * 80), "dev": 4, "G": 0}[split]
            fams[(split, cell)] = [Family(split, cell, "-", "-", s) for s in family_sizes(n)]
            for fam in fams[(split, cell)]:
                fam.atoms = 0
        if split == "H":
            for cell, atoms in S.items():
                fams[(split, cell)] = [Family(split, cell, "-", "none", s) for s in family_sizes(20)]
                for fam in fams[(split, cell)]:
                    fam.atoms = atoms
    for cell, n in ARM_DECOYS.items():
        fams[("G", "arm_" + cell)] = [Family("G", cell, "-", "-", s, arm=True) for s in family_sizes(n)]
        for fam in fams[("G", "arm_" + cell)]:
            fam.atoms = 0

    # ---- watch (F and B families)
    gate_cells = [c for c in F if fams[("G", c)]]
    nf_gate = {c: len(fams[("G", c)]) for c in gate_cells}
    total_gate = sum(nf_gate.values())
    share = {c: GATE_WATCH_FAMILIES * nf_gate[c] / total_gate for c in gate_cells}
    alloc = {c: int(share[c]) for c in gate_cells}
    for c in sorted(gate_cells, key=lambda c: (-(share[c] - alloc[c]), F_ORDER.index(c)))[
             :GATE_WATCH_FAMILIES - sum(alloc.values())]:
        alloc[c] += 1
    watch_quota = {}
    for split in F_SPLITS:
        for cell in F:
            nf = len(fams[(split, cell)])
            if split == "G":
                k = alloc.get(cell, 0)
            else:
                k = ceil(0.15 * nf) if nf else 0
            watch_quota[(split, cell)] = {"reuse": k, "new": k, "auto": nf - 2 * k}
    for split in FAMILY_SPLITS:
        for cell in B:
            assign_watch(fams[(split, cell)], split)

    # ---- pairs (families) within each split
    pairs = []   # (type, first family, second family)
    for split in FAMILY_SPLITS:
        f6 = fams[(split, "F6")]
        unless = [f for f in fams[(split, "F5")] if f.cons == "unless"]
        until = [f for f in fams[(split, "F7")] if f.cons == "until"]
        for kind, members, partners in (("F6dont_until-F5unless", [f for f in f6 if f.cons == "dont_until"], unless),
                                        ("F6hold_fire_till-F7until", [f for f in f6 if f.cons == "hold_fire_till"],
                                         until)):
            if len(partners) < len(members):
                raise SystemExit(f"{split} {kind}: {len(members)} members, {len(partners)} partners")
            for m, p in zip(members, partners):
                m.partner, p.partner = p, m
                if kind.endswith("F7until"):
                    p.fixed_action = "hold"
                pairs.append((kind, m, p))
        if split == "H":
            for m, p in zip(fams[("H", "S6")], [f for f in f6 if f.cons == "dont_before"]):
                m.partner, p.partner = p, m
                pairs.append(("S6-F6dont_before", m, p))
        dsplits = [(split, f"D{k}") for k in (3, 5, 6)]
        if split == "G":
            dsplits = [("G", f"arm_D{k}") for k in (3, 5, 6)]
        pref = {"D3": ("F1", "F2"), "D5": ("F2", "F1"), "D6": ("F1", "F2")}
        for key in dsplits:
            name = key[1].replace("arm_", "")
            for dfam in fams[key]:
                for pc in pref[name]:
                    cand = next((f for f in fams[(split, pc)] if f.partner is None), None)
                    if cand is not None:
                        break
                else:
                    raise SystemExit(f"{split} {name}: no free F1/F2 partner family")
                dfam.partner, cand.partner = cand, dfam
                pairs.append((f"{name}-{cand.cell}", dfam, cand))

    # ---- F cells: joint balanced design per part (watch, action, entity per unit; placement and
    # possessive per row), counted at split level and at part level
    for part, order in (("eval", ("H", "G", "dev", "RE")), ("tv", ("T", "V", "RT"))):
        for cell in F_ORDER:
            balance_cell(part, cell, order, fams, watch_quota)
    for split in FAMILY_SPLITS:
        for cell in list(B) + D + list(S):
            for key in ((split, cell), ("G", "arm_" + cell)) if split == "G" else ((split, cell),):
                items = fams.get(key, [])
                for f in items:
                    if f.partner is not None:
                        f.entity = f.partner.entity
                        if cell == "S6":
                            f.action = f.partner.action
                for f, e in greedy(items, lambda f: ENTITY_TYPES, lambda f: f.entity,
                                   [lambda f, v: v, lambda f, v: (f.watch, v)]):
                    f.entity = e

    # ---- expand families into rows
    rows_of = {}
    for (split, cell), items in fams.items():
        k = 0
        for f in items:
            rows = []
            for m in range(f.size):
                if cell in F:
                    aspect = "punctual" if f.action == "fire" else "durative"
                    rule = F[cell]["rule"]
                    if rule == "negated":
                        role, neg = ("start" if aspect == "punctual" else "during"), "true"
                    else:
                        role, neg = rule, "false"
                    rows.append(dict(stratum=cell, construction=f.cons, cmp=f.cmp, watch=f.watch,
                                     action=f.action, aspect=aspect, role=role, negated=neg,
                                     entity_type=f.entity, placement=f.placements[m], possessive=f.possessives[m],
                                     atoms=f.atoms, route="Executable", arm=False, reserved=False, member=m))
                else:
                    is_b = cell in B
                    rows.append(dict(stratum=f.cell, construction="-", cmp=f.cmp if is_b else ("none" if cell in S else "-"),
                                     watch=f.watch if is_b else "-", action=f.action or "-",
                                     aspect="-", role="-", negated="-", entity_type=f.entity, placement="-",
                                     possessive="-", atoms=f.atoms,
                                     route="NoAction" if f.cell == "D2" else "Executable",
                                     arm=f.arm, reserved=False, member=m))
                k += 1
            rows_of[id(f)] = rows

    # ---- units: pairs, then remaining families, then singles
    in_pair = set()
    for kind, a, b in pairs:
        units.append(dict(kind="pair", group=kind, split=a.split, rows=rows_of[id(a)] + rows_of[id(b)]))
        in_pair.update((id(a), id(b)))
    for (split, cell), items in fams.items():
        for f in items:
            if id(f) not in in_pair:
                units.append(dict(kind="single" if f.size == 1 else "family", group="-", split=split,
                                  rows=rows_of[id(f)]))

    def single(split, stratum, construction, atoms, route="Executable", arm=False, **attrs):
        row = dict(stratum=stratum, construction=construction, cmp="-", watch="-", action="-", aspect="-",
                   role="-", negated="-", entity_type="-", placement="-", possessive="-", atoms=atoms,
                   route=route, arm=arm, reserved=False, member=0)
        row.update(attrs)
        units.append(dict(kind="single", group="-", split=split, rows=[row]))

    # PC-2 pairs of single rows (reuse arm, new arm)
    pc2_pairs = {"H": 40, "dev": 4, "T": 80, "V": ceil(V_FACTOR_SINGLE * 80), "RE": ceil(RESERVE_FACTOR * 44),
                 "RT": ceil(RESERVE_FACTOR * 92)}
    for split, n in pc2_pairs.items():
        for cond in PC2:
            for i in range(n):
                ent = ENTITY_TYPES[i % 5]
                rows = [dict(stratum="PC2-" + cond, construction="-", cmp="none", watch=w, action="fire",
                             aspect="punctual", role="start", negated="false", entity_type=ent, placement="-",
                             possessive="-", atoms=1, route="Executable", arm=False, reserved=False, member=m)
                        for m, w in enumerate(("reuse", "new"))]
                units.append(dict(kind="pair", group="PC2-arms", split=split, rows=rows))
    # cap boundary
    for stratum, cons_list in CAP.items():
        for cons, h, atoms in cons_list:
            counts = {"H": h, "T": 2 * h, "V": ceil(V_FACTOR_OTHER * 2 * h),
                      "dev": {"four_intents": 4, "five_requested": 2, "inserted_wait": 1, "two_waits": 1}[cons]}
            counts["RE"] = ceil(RESERVE_FACTOR * (counts["H"] + counts["dev"]))
            counts["RT"] = ceil(RESERVE_FACTOR * (counts["T"] + counts["V"]))
            for split, n in counts.items():
                for _ in range(n):
                    single(split, "CAP-" + stratum, cons, atoms)
    # general, reserved compositional, ambiguous executable
    for cls in GEN:
        noaction = cls in ("cancellation", "zero_action")
        route = {"cancellation": "Cancelled", "zero_action": "NoAction"}.get(cls, "Executable")
        counts = {"H": 100, "dev": 10}
        if not noaction:
            counts.update(T=200, V=ceil(V_FACTOR_OTHER * 200))
        for split, n in counts.items():
            for _ in range(n):
                single(split, "GEN-" + cls, "-", 1 if cls == "nested_condition" else 0, route=route)
    for cls in UC:
        for split, n in {"H": 40, "dev": 4, "T": 80, "V": ceil(V_FACTOR_OTHER * 80)}.items():
            for _ in range(n):
                single(split, "UC-" + cls, "-", 1 if cls == "nested_condition" else 0)
    for split, n in {"H": 40, "dev": 4}.items():
        for _ in range(n):
            single(split, "N-ambiguous_executable", "-", 0)
    # probe reserve pools: F reserve slots are size-1 units built with the F design; B and D here
    for cell, atoms in B.items():
        for split, n in (("RE", ceil(RESERVE_FACTOR * 112)), ("RT", ceil(RESERVE_FACTOR * 218))):
            watches = smooth((("auto", 70), ("reuse", 15), ("new", 15)), n)
            for i in range(n):
                single(split, cell, "-", (2 if i % 2 == 0 else 3) if atoms == "2-3" else atoms, cmp="none",
                       watch=watches[i], entity_type=ENTITY_TYPES[i % 5])
    for cell in D:
        for split, n in (("RE", ceil(RESERVE_FACTOR * 44)), ("RT", ceil(RESERVE_FACTOR * 88))):
            for i in range(n):
                single(split, cell, "-", 0, route="NoAction" if cell == "D2" else "Executable",
                       entity_type=ENTITY_TYPES[i % 5])
    # detection arm plain commands (optional, D-SLM8098)
    for i in range(ARM_PLAIN):
        single("G", "A0-plain", "-", 0, arm=True, action=ACTIONS[i % 5])

    # ---- reserved-vocabulary assignment (eval only: G, H, RE)
    for split, frac in RESERVED_FRACTION.items():
        cond = [u for u in units if u["split"] == split and any(r["atoms"] > 0 for r in u["rows"])]
        cond.sort(key=lambda u: (u["rows"][0]["stratum"], u["rows"][0]["construction"]))
        n = ceil(frac * len(cond))
        for idx in evenly(n, len(cond), 0.5):
            for r in cond[idx]["rows"]:
                r["reserved"] = True
    return units


def signature(unit):
    rows = [{k: r[k] for k in COLUMNS if k != "row"} for r in unit["rows"]]
    blob = json.dumps({"kind": unit["kind"], "group": unit["group"], "rows": rows}, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def coarse_key(r):
    return (r["stratum"], r["construction"], r["cmp"], r["watch"], r["arm"])


def main():
    units = build()
    sigs = {}
    for u in units:
        u["sig"] = signature(u)
        entry = sigs.setdefault(u["sig"], {"kind": u["kind"], "group": u["group"], "quota": Counter()})
        entry["quota"][u["split"]] += 1
    # Units are listed by signature, then in build order; their designed split is not written.
    units.sort(key=lambda u: u["sig"])
    row_no = 0
    out_units = []
    coarse = defaultdict(Counter)
    for i, u in enumerate(units, 1):
        rows = []
        for r in u["rows"]:
            row_no += 1
            rows.append([f"r{row_no:06d}"] + [r[c] for c in COLUMNS[1:]])
            coarse[coarse_key(r)][u["split"]] += 1
        out_units.append({"unit": f"u{i:05d}", "signature": u["sig"], "rows": rows})
    matrix = {
        "matrix": "T-3103 S0 abstract design matrix",
        "matrix_version": "MX-1",
        "plan": {"path": "Claude/Plans/superslm-multicommand-array/plan.md", "rev": 15, "commit": "d628055561"},
        "builder": "tests/t3094-multicommand-s0/matrix/build_design_matrix.py",
        "content": "abstract slots only; no utterance text, entity words, state words or gold strings",
        "splits": {s: {"part": PART[s]} for s in SPLITS},
        "parts": {"eval": [s for s in SPLITS if PART[s] == "eval"], "tv": [s for s in SPLITS if PART[s] == "tv"]},
        "columns": list(COLUMNS),
        "balance": {"statistic": "bias-corrected Cramer's V (Bergsma 2013); uncorrected row-level V reported",
                    "tolerance": 0.10,
                    "observation": {"action": "family", "entity_type": "family", "placement": "row",
                                    "possessive": "row"},
                    "family": "rows sharing (unit, stratum, construction): one paraphrase family, one meaning",
                    "within": "aspect (fire punctual; move, scan, hold, wait durative) x measure (F3 measured, "
                              "F4 count, other F cells plain)",
                    "rows": "F-cell rows (stratum F1-F12), detection-arm rows excluded",
                    "nonsemantic": ["action", "placement", "entity_type", "possessive"],
                    "labels": ["role", "negated", "cmp", "watch"],
                    "exempt": {"entity_type": ["F11", "F12"]}, "min_rows": 100,
                    "min_rows_applies_to": "rows in the stratum within the scope",
                    "decision": "fail when the bias-corrected V exceeds the tolerance on any table in a decisive "
                                "scope",
                    "resolution": "each table reports the corrected V a chi-square at its 95th percentile under "
                                  "independence produces; a pass where that exceeds the tolerance means no "
                                  "association beyond chance was detected, and is listed as unresolved",
                    "decisive_scopes": ["eval", "tv", "G", "H", "T"]},
        "optional_strata": {"detection_arm": {"decision": "D-SLM8098 (pending with the principal)",
                                              "marker": "arm == true", "rows": ARM_PLAIN + sum(ARM_DECOYS.values()),
                                              "split": "G",
                                              "removal": "drop every row with arm == true; no other row moves"}},
        "reserved_vocabulary": {"marker": "reserved == true",
                                "rule": "a reserved row's entity and state words come only from the sealed "
                                        "reserved list; set on whole units (a family or pair shares its words)",
                                "fractions_of_condition_units": RESERVED_FRACTION,
                                "never": ["dev", "T", "V", "RT"]},
        "probe_reserve_pool": {"splits": {"RE": "eval writers", "RT": "T/V writers"},
                               "rule": "written before the probe first runs; never inspected; drawn only for the "
                                       "s5.2 rule (b) re-test (CA14-2)",
                               "strata": "F, B, D, PC-2, cap boundary"},
        "signatures": {k: {"kind": v["kind"], "group": v["group"],
                           "quota": {s: v["quota"][s] for s in SPLITS if v["quota"][s]}}
                       for k, v in sorted(sigs.items())},
        "coarse_quotas": [[list(k[:4]) + [k[4]], {s: v[s] for s in SPLITS if v[s]}]
                          for k, v in sorted(coarse.items(), key=lambda kv: tuple(map(str, kv[0])))],
        "units": out_units,
    }
    text = json.dumps(matrix, separators=(",", ":"), sort_keys=False) + "\n"
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"rows {row_no} units {len(out_units)} signatures {len(sigs)}")
    print("sha256 " + hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    sys.exit(main())
