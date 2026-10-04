"""T-3103 S0 timeline fixtures TL-1 to TL-20 (plan rev 15 s4.5), plus extension fixtures TL-13c and TL-21
(finding MX-7). Every expected ledger is derived by hand from the s4.5 rules, not by running an evaluator.
The S3c product evaluator replays the same fixtures (`py -3 timeline_fixtures.py > timeline-fixtures.json`).

Notation: step -1 is the pre-request snapshot and step 0 is publication t0. A ledger entry is
[t, ordinal, decision].
"""
from __future__ import annotations

import json


def E(inst, pred, kind, cmp="none", value=None, neg=False, mode="passive", ref=None):
    return {"instance": inst, "predicate": pred, "kind": kind, "cmp": cmp,
            "value": None if value is None else {"num": value[0], "den": value[1]}, "negated": neg,
            "binding": {"mode": mode, "ref": ref}}


def C(action, start_all=(), start_any=(), stop_any=(), during_all=(), after=(), policy="suspend"):
    return {"action": action, "aspect": "punctual" if action == "fire" else "durative",
            "trigger": {"start_all": list(start_all), "start_any": list(start_any), "stop_any": list(stop_any),
                        "during_all": list(during_all)}, "after": list(after), "during_policy": policy}


def T(steps, values=None, events=None, complete=None, cancel=None, save_load=()):
    """steps: iterable of ints; values: {key: {t: v}} filled forward; events: {key: [t...]}."""
    values, events, complete, cancel = values or {}, events or {}, complete or {}, cancel or {}
    out = []
    for t in steps:
        snap = {"t": t, "values": {}, "events": [k for k, ts in events.items() if t in ts],
                "complete": [o for o, ts in complete.items() if t in ts],
                "cancel": [o for o, ts in cancel.items() if t in ts], "save_load": t in save_load}
        for k, series in values.items():
            known = [s for s in series if s <= t]
            if known:
                snap["values"][k] = series[max(known)]
        out.append(snap)
    return out


R = range(-1, 8)
FIXTURES = {
    "TL-1": dict(batch=[C("fire", start_all=[E("beacon", "lit", "state")])],
                 timeline=T(R, {"beacon|lit": {-1: False, 3: True}}), expected=[[3, 0, "start"]]),
    "TL-2": dict(batch=[C("fire", start_all=[E("beacon", "lit", "state")])],
                 timeline=T(R, {"beacon|lit": {-1: True}}), expected=[[0, 0, "start"]]),
    "TL-3": dict(batch=[C("fire", start_all=[E("gate", "opened", "event")])],
                 timeline=T(R, events={"gate|opened": [-1, 5]}), expected=[[5, 0, "start"]]),
    "TL-4": dict(batch=[C("fire", start_all=[E("gate", "open", "state", neg=True)])],
                 timeline=T(R, {"gate|open": {-1: True, 2: False}}), expected=[[2, 0, "start"]]),
    "TL-5a": dict(batch=[C("fire", start_any=[E("green flare", "fires", "event"), E("red flare", "fires", "event")])],
                  timeline=T(R, events={"green flare|fires": [2]}), expected=[[2, 0, "start"]]),
    "TL-5b": dict(batch=[C("fire", start_any=[E("green flare", "fires", "event"), E("red flare", "fires", "event")])],
                  timeline=T(R, events={"red flare|fires": [3]}), expected=[[3, 0, "start"]]),
    "TL-6": dict(batch=[C("fire", start_all=[E("a", "on", "state"), E("b", "on", "state")])],
                 timeline=T(R, {"a|on": {-1: False, 1: True}, "b|on": {-1: False}}), expected=[]),
    "TL-7": dict(batch=[C("hold", stop_any=[E("beacon", "lit", "state")])],
                 timeline=T(R, {"beacon|lit": {-1: False, 4: True}}), expected=[[0, 0, "start"], [4, 0, "end"]]),
    "TL-8a": dict(batch=[C("scan", during_all=[E("gate", "open", "state")], policy="suspend")],
                  timeline=T(R, {"gate|open": {-1: True, 2: False, 4: True}}),
                  expected=[[0, 0, "start"], [2, 0, "suspend"], [4, 0, "resume"]]),
    "TL-8b": dict(batch=[C("scan", during_all=[E("gate", "open", "state")], policy="end")],
                  timeline=T(R, {"gate|open": {-1: True, 2: False, 4: True}}),
                  expected=[[0, 0, "start"], [2, 0, "end"]]),
    "TL-9": dict(batch=[C("fire", start_all=[E("gauge", "level", "numeric", "reaches", (50, 1))])],
                 timeline=T(R, {"gauge|level": {-1: 30, 1: 45, 2: 55}}), expected=[[2, 0, "start"]]),
    "TL-10": dict(batch=[C("fire", start_all=[E("gauge", "level", "numeric", "reaches", (50, 1))])],
                  timeline=T(R, {"gauge|level": {-1: 70, 1: 60, 2: 45}}), expected=[[2, 0, "start"]]),
    "TL-11": dict(batch=[C("fire", start_all=[E("turrets", "standing", "count", "at_most", (0, 1))])],
                  timeline=T(R, {"turrets|standing": {-1: [True, True], 1: [True, False], 2: [False, False]}}),
                  expected=[[2, 0, "start"]]),
    "TL-12": dict(batch=[C("hold", stop_any=[E("charlie", "health", "numeric", "below", (75, 2))])],
                  timeline=T(R, {"charlie|health": {-1: 40, 1: 38, 2: 37}}), expected=[[0, 0, "start"], [2, 0, "end"]]),
    "TL-13a": dict(batch=[C("wait", stop_any=[E("red flare", "fires", "event")]),
                          C("wait", stop_any=[E("green flare", "fires", "event")]),
                          C("fire", start_all=[E("green flare", "fires", "event", mode="wait", ref=1),
                                               E("red flare", "fires", "event", mode="wait", ref=0)])],
                   timeline=T(R, events={"green flare|fires": [2], "red flare|fires": [4]}),
                   expected=[[0, 0, "start"], [0, 1, "start"], [2, 1, "end"], [4, 0, "end"], [4, 2, "start"]]),
    "TL-13b": dict(batch=[C("wait", stop_any=[E("red flare", "fires", "event")]),
                          C("wait", stop_any=[E("green flare", "fires", "event")]),
                          C("fire", start_all=[E("green flare", "fires", "event", mode="wait", ref=1),
                                               E("red flare", "fires", "event", mode="wait", ref=0)])],
                   timeline=T(R, events={"green flare|fires": [2], "red flare|fires": [5]}, cancel={0: [3]}),
                   expected=[[0, 0, "start"], [0, 1, "start"], [2, 1, "end"], [3, 0, "cancelled"],
                             [3, 2, "ConditionUnreachable"]]),
    # extension (MX-7): wait completion versus the raw predicate. The beacon is lit at t2 (the wait completes)
    # and unlit from t3; the second start atom holds from t5. Correct: start at t5. Ignoring the wait binding
    # (evaluating the raw level predicate) gives no start.
    "TL-13c": dict(batch=[C("wait", stop_any=[E("beacon", "lit", "state")]),
                          C("fire", start_all=[E("beacon", "lit", "state", mode="wait", ref=0),
                                               E("gate", "open", "state")])],
                   timeline=T(R, {"beacon|lit": {-1: False, 2: True, 3: False}, "gate|open": {-1: False, 5: True}}),
                   expected=[[0, 0, "start"], [2, 0, "end"], [5, 1, "start"]]),
    "TL-14": dict(batch=[C("move"), C("fire", start_all=[E("beacon", "lit", "state")], after=[0])],
                  timeline=T(R, {"beacon|lit": {-1: False, 1: True}}, complete={0: [3]}),
                  expected=[[0, 0, "start"], [3, 0, "end"], [3, 1, "start"]]),
    "TL-15": dict(batch=[C("fire", start_all=[E("beacon", "lit", "state"), E("gate", "opened", "event")])],
                  timeline=T(R, {"beacon|lit": {-1: False, 5: True}}, events={"gate|opened": [2]}, save_load=(3,)),
                  expected=[[5, 0, "start"]]),
    "TL-16": dict(batch=[C("move"), C("fire", start_all=[E("alpha", "arrives", "event")], after=[0])],
                  timeline=T(R, events={"alpha|arrives": [2, 5]}, complete={0: [3]}),
                  expected=[[0, 0, "start"], [3, 0, "end"], [5, 1, "start"]]),
    "TL-17": dict(batch=[C("move"), C("hold", stop_any=[E("alpha", "arrives", "event")], after=[0])],
                  timeline=T(R, events={"alpha|arrives": [2, 6]}, complete={0: [3]}),
                  expected=[[0, 0, "start"], [3, 0, "end"], [3, 1, "start"], [6, 1, "end"]]),
    "TL-18": dict(batch=[C("move"), C("wait", stop_any=[E("flare", "fires", "event")], after=[0]),
                         C("fire", start_all=[E("flare", "fires", "event", mode="wait", ref=1)], after=[0])],
                  timeline=T(R, events={"flare|fires": [2, 5]}, complete={0: [3]}),
                  expected=[[0, 0, "start"], [3, 0, "end"], [3, 1, "start"], [5, 1, "end"], [5, 2, "start"]]),
    "TL-19": dict(batch=[C("fire", start_all=[E("gauge", "level", "numeric", "reaches", (50, 1))])],
                  timeline=T(R, {"gauge|level": {-1: 30, 0: 70, 1: 60, 2: 50}}), expected=[[2, 0, "start"]]),
    "TL-20": dict(batch=[C("hold", stop_any=[E("charlie", "health", "numeric", "below", (75, 2))])],
                  timeline=T(R, {"charlie|health": {-1: [375, 10], 1: [374, 10]}}),
                  expected=[[0, 0, "start"], [1, 0, "end"]]),
    # extension (MX-7): a fixed-point value beyond binary64 precision. 100000000000000001/10^17 is above 1
    # exactly, and equals 1.0 as a float.
    "TL-21": dict(batch=[C("fire", start_all=[E("gauge", "ratio", "numeric", "above", (1, 1))])],
                  timeline=T(R, {"gauge|ratio": {-1: [100000000000000000, 10 ** 17], 2: [100000000000000001, 10 ** 17]}}),
                  expected=[[2, 0, "start"]]),
}

# s4.5 mutant list -> the fixture(s) that must fail under it
MUTANT_FIXTURES = {
    "start_any_as_and": ["TL-5a", "TL-5b"], "start_all_as_or": ["TL-6"], "state_as_edge": ["TL-2"],
    "event_as_level": ["TL-15"], "ignore_during": ["TL-8a", "TL-8b"], "ignore_stop": ["TL-7"],
    "ignore_negated": ["TL-4"], "reaches_as_below": ["TL-9"], "rational_rounded": ["TL-20"],
    "ignore_wait_completion": ["TL-13c"], "latch_dropped_on_load": ["TL-15"],
    "events_from_t0": ["TL-16", "TL-17"], "reaches_side_from_pre_request": ["TL-19"],
    "scaled_as_float": ["TL-21"],
}

if __name__ == "__main__":
    print(json.dumps({"fixtures": FIXTURES, "mutant_fixtures": MUTANT_FIXTURES}, indent=1))
