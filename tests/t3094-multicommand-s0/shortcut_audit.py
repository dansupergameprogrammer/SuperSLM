"""Predeclared raw-text shortcut challenge for the 320 wait-boundary rows.

The labels are inferred from scheduler events, never from source IDs or arms.
Features are lower-case word n-grams of length one through four. The fixed
group split keeps both sides of a matched pair together to avoid pair leakage.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(3094, 3102))
TRAIN_GROUP_FRACTION = 0.7
MAX_PURITY = 0.8
MIN_SUPPORT = {"all": 8, "pc2": 8, "cap_boundary": 4}
MAX_CLASSIFIER_ACCURACY = 0.60
MIN_MEAN_CLASSIFIER_ACCURACY = 0.35
MAX_MEAN_CLASSIFIER_ACCURACY = 0.65
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def ngrams(text: str) -> set[tuple[str, ...]]:
    words = TOKEN.findall(text.casefold())
    return {tuple(words[i:i+n]) for n in range(1, 5) for i in range(len(words)-n+1)}


def label(row: dict) -> int:
    events = row["game_action_trace"]["events"]
    watch = any(e.get("event") == "task_queued" and e.get("task") == "signal_watch" for e in events)
    callback = any(e.get("event") == "completion_callback_registered" for e in events)
    listener = any(e.get("event") == "signal_listener_registered" for e in events)
    if watch and callback and not listener:
        return 1
    if listener and not watch and not callback:
        return 0
    raise ValueError("boundary trace does not resolve one wait interpretation")


def selected(rows: list[dict], family: str) -> list[dict]:
    classes = {"all": {"pc2", "at_cap", "cap_plus_one"},
               "pc2": {"pc2"}, "cap_boundary": {"at_cap", "cap_plus_one"}}
    return [r for r in rows if r["source_class"] in classes[family]]


def purity(rows: list[dict], minimum: int) -> dict:
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        y = label(row)
        for gram in ngrams(row["utterance"]):
            counts[gram][y] += 1
    eligible = [(gram, count) for gram, count in counts.items() if sum(count) >= minimum]
    offenders = sorted(
        ({"ngram": " ".join(gram), "condition": count[0], "action": count[1],
          "support": sum(count), "purity": round(max(count) / sum(count), 6)}
         for gram, count in eligible if max(count) / sum(count) > MAX_PURITY),
        key=lambda x: (-x["support"], -x["purity"], x["ngram"]),
    )
    return {"minimum_support": minimum, "maximum_allowed_purity": MAX_PURITY,
            "eligible_ngrams": len(eligible), "offender_count": len(offenders),
            "maximum_observed_purity": round(max((max(c)/sum(c) for _, c in eligible), default=0), 6),
            "top_offenders": offenders[:20]}


def classifier(rows: list[dict], seed: int) -> dict:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    # The same authored phrasing is adapted to the three PC-2 contexts and
    # the cap boundary. Hold all four appearances out together; otherwise a
    # learner can memorize the phrasing in one context and score its sibling.
    groups = defaultdict(list)
    for row in rows:
        if row["source_class"] == "pc2":
            group = row["pair"].rsplit("-", 1)[-1]
        else:
            group = row["id"].rsplit("-", 1)[-1]
        groups[group].append(row)
    assert len(groups) == 40
    assert all(len(members) == len(rows)//40 and
               sum(label(r) for r in members) == len(members)//2 for members in groups.values())
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * TRAIN_GROUP_FRACTION)
    train = [r for key in keys[:cut] for r in groups[key]]
    test = [r for key in keys[cut:] for r in groups[key]]
    vectorizer = CountVectorizer(ngram_range=(1, 4), binary=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+(?:'\w+)?\b")
    xtrain = vectorizer.fit_transform(r["utterance"] for r in train)
    xtest = vectorizer.transform(r["utterance"] for r in test)
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=seed)
    model.fit(xtrain, [label(r) for r in train])
    correct = sum(int(got == label(row)) for got, row in zip(model.predict(xtest), test))
    return {"seed": seed, "train_groups": cut, "test_groups": len(keys)-cut,
            "train_rows": len(train), "test_rows": len(test), "correct": correct,
            "accuracy": round(correct/len(test), 6),
            "features": len(vectorizer.vocabulary_)}


def audit(rows: list[dict]) -> dict:
    import sklearn
    report = {}
    for family in ("all", "pc2", "cap_boundary"):
        subset = selected(rows, family)
        readings = [classifier(subset, seed) for seed in SEEDS]
        mean = sum(r["accuracy"] for r in readings) / len(readings)
        report[family] = {"rows": len(subset), "purity": purity(subset, MIN_SUPPORT[family]),
                          "classifier": {"seeds": list(SEEDS), "splits": readings,
                                         "mean_accuracy": round(mean, 6),
                                         "minimum_accuracy": min(r["accuracy"] for r in readings),
                                         "maximum_accuracy": max(r["accuracy"] for r in readings),
                                         "allowed_mean_accuracy": [MIN_MEAN_CLASSIFIER_ACCURACY,
                                                                   MAX_MEAN_CLASSIFIER_ACCURACY],
                                         "allowed_maximum_accuracy": MAX_CLASSIFIER_ACCURACY,
                                         "sklearn_version": sklearn.__version__}}
        report[family]["passed"] = (report[family]["purity"]["offender_count"] == 0 and
                                    MIN_MEAN_CLASSIFIER_ACCURACY <= mean <= MAX_MEAN_CLASSIFIER_ACCURACY and
                                    report[family]["classifier"]["maximum_accuracy"] <= MAX_CLASSIFIER_ACCURACY)
    report["passed"] = all(report[k]["passed"] for k in ("all", "pc2", "cap_boundary"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus.jsonl")
    args = parser.parse_args()
    raw = args.corpus.read_bytes()
    result = audit([json.loads(line) for line in raw.decode("utf-8").splitlines()])
    result["corpus_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
