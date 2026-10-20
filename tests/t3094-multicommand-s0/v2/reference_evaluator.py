"""T-3103 S0 reference ConditionEvaluator (plan rev 15 s4.5), written from the plan text without S3b
product code. A deterministic discrete-time evaluator.

Batch: ordered commands {"action", "aspect": "punctual"|"durative", "trigger": ResolvedCondition groups,
  "after": [ordinals], "during_policy": "suspend"|"end"} (inserted waits are durative "wait" commands).
Timeline (integer steps; step -1 is the pre-request snapshot, step 0 is publication t0):
  {"t": int, "values": {"instance|predicate": bool | int | [num, scale]}, "events": ["instance|predicate"],
   "complete": [ordinals], "cancel": [ordinals], "save_load": bool}
Count entries name a group: values["group|predicate"] is a list of member booleans.
Output: a ledger of [t, ordinal, decision], with decisions start, suspend, resume, end and
ConditionUnreachable. An action attempt is a start.

Rules (plan s4.5):
- A command becomes current at t_c, when its last `after` ordinal completes (t0 when `after` is empty).
- start_all, start_any and during_all entries are evaluated from t_c; stop_any entries from t_s.
- A state is level. An event latches from its first occurrence strictly after the entry's reference time.
- Numeric and count entries compare exactly by cross-multiplication. reaches fixes its side at the
  reference time and latches.
- negated inverts a level predicate or comparison. A wait binding holds once its wait command completes.
- A command starts when current, with every start_all entry, at least one start_any entry (if any) and every
  during_all entry holding.
- A durative command ends when any stop_any entry holds. A during_all entry ceasing to hold suspends or
  ends the command, by during_policy. A punctual command starts at most once and completes when it starts.
- A cancelled inserted wait makes its gated command unreachable (ConditionUnreachable, zero attempts).
- Latches survive a save and load.

`mutants` switches one deliberate defect on (s4.5's mutant list), for the vitality cells.
"""
from __future__ import annotations

import copy
import json

MUTANTS = ("start_any_as_and", "start_all_as_or", "state_as_edge", "event_as_level", "ignore_during", "ignore_stop",
           "ignore_negated", "reaches_as_below", "rational_rounded", "ignore_wait_completion", "latch_dropped_on_load",
           "events_from_t0", "reaches_side_from_pre_request", "scaled_as_float")


def _key(e):
    return e["instance"] + "|" + e["predicate"]


def _cmp(cmp, v_num, v_den, n, d, m):
    if "rational_rounded" in m:
        n, d = round(n / d), 1
    if "scaled_as_float" in m:
        v = v_num / v_den
        thr = n / d
        return {"below": v < thr, "above": v > thr, "at_most": v <= thr, "at_least": v >= thr}[cmp]
    lhs, rhs = v_num * d, n * v_den
    return {"below": lhs < rhs, "above": lhs > rhs, "at_most": lhs <= rhs, "at_least": lhs >= rhs}[cmp]


def _value(snapshot, e):
    v = snapshot["values"].get(_key(e))
    if e["kind"] == "count":
        return (sum(bool(x) for x in v), 1)
    if isinstance(v, list):
        return (v[0], v[1])
    return (v, 1)


class Evaluator:
    def __init__(self, batch, mutants=()):
        self.batch = batch
        self.m = set(mutants)
        n = len(batch)
        self.state = {"t_c": [None] * n, "t_s": [None] * n, "done": [False] * n, "ended": [False] * n,
                      "suspended": [False] * n, "unreachable": [False] * n, "latch": {}, "side": {}}
        self.ledger = []

    def _ref_time(self, i, group):
        return self.state["t_s"][i] if group == "stop_any" else self.state["t_c"][i]

    def _holds(self, i, group, j, e, snap, t, history):
        st = self.state
        ref = self._ref_time(i, group)
        lk = f"{i}|{group}|{j}"
        mode = e["binding"]["mode"]
        if mode == "wait" and "ignore_wait_completion" not in self.m:
            return st["done"][e["binding"]["ref"]]
        kind = e["kind"]
        neg = e["negated"] and "ignore_negated" not in self.m
        if kind == "event" and "event_as_level" not in self.m:
            return bool(st["latch"].get(lk))     # latched by the per-step pass in _latch_events
        if kind == "event":                      # mutant: event treated as level
            return _key(e) in snap["events"]
        if kind == "state":
            level = bool(snap["values"].get(_key(e)))
            if "state_as_edge" in self.m:
                prev = history.get(t - 1, {"values": {}})["values"].get(_key(e))
                if st["latch"].get(lk):
                    return True
                if level and not prev and t > ref:
                    st["latch"][lk] = True
                return bool(st["latch"].get(lk))
            return level != neg
        num, den = _value(snap, e)
        n, d = e["value"]["num"], e["value"]["den"]
        if e["cmp"] == "reaches" and "reaches_as_below" not in self.m:
            if st["latch"].get(lk):
                return True
            if lk not in st["side"]:
                side_t = -1 if "reaches_side_from_pre_request" in self.m else ref
                sv = _value(history[side_t], e) if side_t in history else (num, den)
                st["side"][lk] = "eq" if sv[0] * d == n * sv[1] else ("below" if sv[0] * d < n * sv[1] else "above")
            side = st["side"][lk]
            crossed = side == "eq" or (side == "below" and num * d >= n * den) or (side == "above" and num * d <= n * den)
            if crossed:
                st["latch"][lk] = True
            return crossed
        cmp = "below" if e["cmp"] == "reaches" else e["cmp"]
        return _cmp(cmp, num, den, n, d, self.m) != neg

    def _latch_events(self, snap, t):
        """Latch event entries. An entry latches an occurrence strictly after its own reference time:
        t_c for start_all, start_any and during_all; t_s for stop_any. The events_from_t0 mutant latches
        from t0 for every entry of a live command, whatever its reference time."""
        st = self.state
        for i, c in enumerate(self.batch):
            if st["ended"][i] or st["unreachable"][i]:
                continue
            for g in ("start_all", "start_any", "stop_any", "during_all"):
                ref = self._ref_time(i, g)
                for j, e in enumerate(c["trigger"][g]):
                    if e["kind"] != "event" or _key(e) not in snap["events"]:
                        continue
                    if "events_from_t0" in self.m or (ref is not None and t > ref):
                        st["latch"][f"{i}|{g}|{j}"] = True

    def step(self, snap, history):
        t = snap["t"]
        st = self.state
        if snap.get("save_load"):
            saved = json.loads(json.dumps(st))
            if "latch_dropped_on_load" in self.m:
                saved["latch"] = {}
            self.state = st = saved
        for o in snap.get("complete", []):
            if not st["ended"][o] and st["t_s"][o] is not None:
                st["ended"][o] = st["done"][o] = True
                self.ledger.append([t, o, "end"])
        for o in snap.get("cancel", []):
            st["ended"][o] = True
            st["done"][o] = False
            self.ledger.append([t, o, "cancelled"])
            for i, c in enumerate(self.batch):
                for g in ("start_all", "start_any", "stop_any", "during_all"):
                    for e in c["trigger"][g]:
                        if e["binding"]["mode"] == "wait" and e["binding"]["ref"] == o and not st["unreachable"][i] \
                                and st["t_s"][i] is None:
                            st["unreachable"][i] = st["ended"][i] = True
                            self.ledger.append([t, i, "ConditionUnreachable"])
        for i, c in enumerate(self.batch):
            if not st["ended"][i] and st["t_c"][i] is None and all(st["done"][a] for a in c["after"]):
                st["t_c"][i] = t
        self._latch_events(snap, t)
        for i, c in enumerate(self.batch):
            if st["ended"][i] or st["unreachable"][i]:
                continue
            if st["t_c"][i] is None and all(st["done"][a] for a in c["after"]):
                st["t_c"][i] = t
            if st["t_c"][i] is None:
                continue
            tr = c["trigger"]
            if st["t_s"][i] is None:
                sa = [self._holds(i, "start_all", j, e, snap, t, history) for j, e in enumerate(tr["start_all"])]
                sy = [self._holds(i, "start_any", j, e, snap, t, history) for j, e in enumerate(tr["start_any"])]
                du = [self._holds(i, "during_all", j, e, snap, t, history) for j, e in enumerate(tr["during_all"])]
                ok_all = all(sa) if "start_all_as_or" not in self.m else (any(sa) if sa else True)
                ok_any = (any(sy) if "start_any_as_and" not in self.m else all(sy)) if sy else True
                ok_during = all(du) or "ignore_during" in self.m
                if ok_all and ok_any and ok_during:
                    st["t_s"][i] = t
                    self.ledger.append([t, i, "start"])
                    if c["aspect"] == "punctual":
                        st["ended"][i] = st["done"][i] = True
                        continue
                else:
                    continue
            # started durative command: stop, during
            stop = [self._holds(i, "stop_any", j, e, snap, t, history) for j, e in enumerate(tr["stop_any"])]
            if any(stop) and "ignore_stop" not in self.m:
                st["ended"][i] = st["done"][i] = True
                self.ledger.append([t, i, "end"])
                continue
            if tr["during_all"] and "ignore_during" not in self.m:
                du = [self._holds(i, "during_all", j, e, snap, t, history) for j, e in enumerate(tr["during_all"])]
                if not all(du) and not st["suspended"][i]:
                    if c.get("during_policy", "suspend") == "end":
                        st["ended"][i] = st["done"][i] = True
                        self.ledger.append([t, i, "end"])
                    else:
                        st["suspended"][i] = True
                        self.ledger.append([t, i, "suspend"])
                elif all(du) and st["suspended"][i]:
                    st["suspended"][i] = False
                    self.ledger.append([t, i, "resume"])

    def run(self, timeline):
        history = {s["t"]: s for s in timeline}
        for snap in sorted(timeline, key=lambda s: s["t"]):
            if snap["t"] >= 0:
                self.step(snap, history)
        return self.ledger


def evaluate(fixture, mutants=()):
    return Evaluator(copy.deepcopy(fixture["batch"]), mutants).run(fixture["timeline"])
