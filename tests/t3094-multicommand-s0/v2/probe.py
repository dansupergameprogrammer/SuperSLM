"""T-3103 masked-feature predictability probe (plan rev 15 s5.2 item 3; D-SLM8102, D-SLM8118, D-SLM8137,
D-SLM8155). The decisive shortcut check. It blocks a freeze, so it is registered QUARANTINED and
commissioned by a seat other than its builder before it reads any real T, V or eval row.

One observation per gold atom. Labels: role, negated, cmp, watch and the bound-intent index.
Structural determinants, the baseline: the action's aspect class and the state's measure status
(measured, count or plain).
Features, none containing a masked word:
- text: every word unigram, every word bigram and every punctuation token of the utterance outside the
  masked list (probe-masked-words.json). Every n-gram that contains a masked word is masked too, including
  across punctuation.
- structured: action, entity noun, state word, length bucket, placement frame, paraphrase family, writer
  and writer engine.

Rule (a), incremental predictability. Under family-grouped cross-validation, take the balanced accuracy
of the structural determinants plus a feature, minus that of the structural determinants alone. It is
evaluated for each feature singly (a lookup within stratum) and for a regularized classifier over all
features. A label flags when the increment exceeds the margin (0.03), after the single-feature
multiplicity correction set at commissioning: `single_feature_min_support`, and
`single_feature_margin_extra` added to the margin.
Rule (b), per-value purity. Within each structural stratum, a feature value with at least 20 rows whose
share of one label value is at least 0.90, and exceeds that label value's stratum share by at least 0.20,
flags. A flagged value blocks only if it flags again on at least 20 fresh reserve rows bearing it; with
fewer reserve rows the block stands until blind writing supplies them (CA14-2).

Rows JSONL: {"row", "utterance", "family", "writer", "engine", "placement",
             "atoms": [{"intent_index", "action", "entity", "state", "measure", "role", "negated", "cmp",
                        "watch"}, ...]}
Usage:
  py -3 probe.py run --rows <jsonl> [--reserve <jsonl>] [--config <json>]
  py -3 probe.py calibrate --rows <must-accept jsonl>   (chance flag rate under each candidate correction)
Exit 0: no flag; 1: blocking flags (rule, label, feature and stratum printed, never row text); 4: input error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
MASKED = json.loads((HERE / "probe-masked-words.json").read_text(encoding="utf-8"))
LABELS = ("role", "negated", "cmp", "watch", "intent_index")
DEFAULT_CONFIG = {"margin": 0.03, "folds": 5, "purity_min_rows": 20, "purity_share": 0.90, "purity_excess": 0.20,
                  "single_feature_min_support": 20, "single_feature_margin_extra": 0.0, "C": 1.0}
_TOKEN = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def _masked_spans():
    phrases = set()
    for k in ("connectives", "negators", "comparison"):
        phrases |= {tuple(p.lower().split()) for p in MASKED[k]}
    return sorted(phrases, key=len, reverse=True)


PHRASES = _masked_spans()


def text_features(utterance: str):
    toks = [t.lower() for t in _TOKEN.findall(utterance)]
    masked = [False] * len(toks)
    words = [(i, t) for i, t in enumerate(toks) if re.match(r"[\w']", t)]
    wseq = [t for _, t in words]
    for phrase in PHRASES:
        n = len(phrase)
        for j in range(len(wseq) - n + 1):
            if tuple(wseq[j:j + n]) == phrase:
                for k in range(j, j + n):
                    masked[words[k][0]] = True
    feats = set()
    for i, t in enumerate(toks):
        if not masked[i]:
            feats.add("u:" + t)
        if i + 1 < len(toks) and not masked[i] and not masked[i + 1]:
            feats.add("b:" + t + " " + toks[i + 1])     # bigrams run across punctuation tokens too
    return feats


def observations(rows):
    obs = []
    for r in rows:
        tf = text_features(r["utterance"])
        n_words = len(r["utterance"].split())
        for a in r["atoms"]:
            aspect = "punctual" if a["action"] == "fire" else "durative"
            stratum = (aspect, a.get("measure", "plain"))
            structured = {"s:action=" + a["action"], "s:entity=" + a["entity"].lower(), "s:state=" + a["state"].lower(),
                          "s:len=" + str(min(n_words // 5, 8)), "s:placement=" + str(r.get("placement")),
                          "s:family=" + str(r.get("family")), "s:writer=" + str(r.get("writer")),
                          "s:engine=" + str(r.get("engine"))}
            labels = {"role": a["role"], "negated": str(a["negated"]), "cmp": a["cmp"], "watch": a["watch"],
                      "intent_index": str(a.get("intent_index", 0))}
            obs.append({"row": r["row"], "group": str(r.get("family") or r["row"]), "stratum": stratum,
                        "features": tf | structured, "labels": labels})
    return obs


def _cv_lookup(obs, label, feature_fn, folds):
    """Out-of-sample balanced accuracy of a majority lookup keyed on (stratum, feature value)."""
    groups = [o["group"] for o in obs]
    y = [o["labels"][label] for o in obs]
    n_splits = min(folds, len(set(groups)))
    if n_splits < 2:
        return None
    pred = [None] * len(obs)
    for train, test in GroupKFold(n_splits=n_splits).split(obs, y, groups):
        table = defaultdict(Counter)
        base = defaultdict(Counter)
        for i in train:
            table[(obs[i]["stratum"], feature_fn(obs[i]))][y[i]] += 1
            base[obs[i]["stratum"]][y[i]] += 1
        for i in test:
            c = table.get((obs[i]["stratum"], feature_fn(obs[i]))) or base.get(obs[i]["stratum"])
            pred[i] = c.most_common(1)[0][0] if c else y[i]
    return balanced_accuracy_score(y, pred)


def _bearing(obs, feature):
    return sorted({o["row"] for o in obs if feature in o["features"]})


def rule_a(obs, cfg):
    flags = []
    feature_counts = Counter(f for o in obs for f in o["features"])
    singles = [f for f, n in feature_counts.items() if n >= cfg["single_feature_min_support"]]
    for label in LABELS:
        if len({o["labels"][label] for o in obs}) < 2:
            continue
        base = _cv_lookup(obs, label, lambda o: None, cfg["folds"])
        if base is None:
            continue
        threshold = cfg["margin"] + cfg["single_feature_margin_extra"]
        for f in singles:
            acc = _cv_lookup(obs, label, lambda o, f=f: f in o["features"], cfg["folds"])
            if acc is not None and acc - base > threshold:
                flags.append({"rule": "a-single", "label": label, "feature": f, "increment": round(acc - base, 4),
                              "row_ids": _bearing(obs, f)})
        # pooled regularized classifier over all features plus the structural determinants
        dicts = [{**{f: 1 for f in o["features"]}, "stratum=" + "/".join(o["stratum"]): 1} for o in obs]
        base_dicts = [{"stratum=" + "/".join(o["stratum"]): 1} for o in obs]
        y = np.array([o["labels"][label] for o in obs])
        groups = [o["group"] for o in obs]
        n_splits = min(cfg["folds"], len(set(groups)))
        accs = []
        for feats in (base_dicts, dicts):
            pred = np.empty(len(obs), dtype=object)
            for train, test in GroupKFold(n_splits=n_splits).split(feats, y, groups):
                vec = DictVectorizer()
                xtr = vec.fit_transform([feats[i] for i in train])
                if len(set(y[train])) < 2:
                    pred[test] = y[train][0]
                    continue
                clf = LogisticRegression(C=cfg["C"], max_iter=2000)
                clf.fit(xtr, y[train])
                pred[test] = clf.predict(vec.transform([feats[i] for i in test]))
            accs.append(balanced_accuracy_score(y, list(pred)))
        if accs[1] - accs[0] > cfg["margin"]:
            flags.append({"rule": "a-pooled", "label": label, "feature": "*",
                          "increment": round(accs[1] - accs[0], 4)})
    return flags


def rule_b(obs, cfg):
    flags = []
    by_stratum = defaultdict(list)
    for o in obs:
        by_stratum[o["stratum"]].append(o)
    for stratum, group in by_stratum.items():
        for label in LABELS:
            share = Counter(o["labels"][label] for o in group)
            n = len(group)
            values = defaultdict(Counter)
            for o in group:
                for f in o["features"]:
                    values[f][o["labels"][label]] += 1
            for f, c in values.items():
                total = sum(c.values())
                if total < cfg["purity_min_rows"]:
                    continue
                v, k = c.most_common(1)[0]
                if k / total >= cfg["purity_share"] and k / total - share[v] / n >= cfg["purity_excess"]:
                    flags.append({"rule": "b", "label": label, "feature": f, "value": v,
                                  "stratum": "/".join(stratum), "rows": total, "purity": round(k / total, 3),
                                  "row_ids": sorted({o["row"] for o in group if f in o["features"]})})
    return flags


def run(rows, reserve, cfg):
    obs = observations(rows)
    a = rule_a(obs, cfg)
    b = rule_b(obs, cfg)
    blocking = list(a)
    if b:
        robs = observations(reserve) if reserve else []
        for f in b:
            bearing = [o for o in robs if f["feature"] in o["features"]
                       and "/".join(o["stratum"]) == f["stratum"]]
            if len(bearing) < cfg["purity_min_rows"]:
                f["retest"] = f"blocked: {len(bearing)} reserve rows bear the value (< {cfg['purity_min_rows']})"
                blocking.append(f)
                continue
            c = Counter(o["labels"][f["label"]] for o in bearing)
            purity = c[f["value"]] / len(bearing)
            if purity >= cfg["purity_share"]:
                f["retest"] = f"flagged again on {len(bearing)} reserve rows (purity {purity:.3f})"
                blocking.append(f)
            else:
                f["retest"] = f"cleared on {len(bearing)} reserve rows (purity {purity:.3f})"
    return {"rule_a": a, "rule_b": b, "blocking": blocking}


def _load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "calibrate"))
    ap.add_argument("--rows", required=True)
    ap.add_argument("--reserve")
    ap.add_argument("--config")
    ap.add_argument("--show-features", action="store_true",
                    help="eval-side use inside the sealed store only; default output is rule, label, stratum and row IDs")
    args = ap.parse_args(argv)
    try:
        cfg = dict(DEFAULT_CONFIG)
        if args.config:
            cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
        rows = _load(args.rows)
        reserve = _load(args.reserve) if args.reserve else []
        if args.cmd == "calibrate":
            obs = observations(rows)
            out = []
            for extra in (0.0, 0.01, 0.02, 0.03, 0.05):
                for support in (10, 20, 40):
                    c = {**cfg, "single_feature_margin_extra": extra, "single_feature_min_support": support}
                    out.append({"margin_extra": extra, "min_support": support,
                                "chance_flags": len([f for f in rule_a(obs, c) if f["rule"] == "a-single"])})
            print(json.dumps({"must_accept_chance_flags": out}, sort_keys=True))
            return 0
        result = run(rows, reserve, cfg)
    except Exception as error:  # noqa: BLE001
        print(f"CONSTRUCTION ERROR: {type(error).__name__}: {error}")
        return 4
    for f in result["blocking"]:
        shown = dict(f) if args.show_features else {k: v for k, v in f.items() if k not in ("feature", "value")}
        print("FLAG " + json.dumps(shown, sort_keys=True))
    print(f"rule_a flags {len(result['rule_a'])} rule_b flags {len(result['rule_b'])} blocking {len(result['blocking'])}")
    return 1 if result["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main())
