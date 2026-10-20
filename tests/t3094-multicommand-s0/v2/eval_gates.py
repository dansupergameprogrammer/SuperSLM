"""T-3103 eval-side gates (plan rev 15 s5.3, s8 CA14-3, R14-N1). Run by an eval-side reader. Every
output an adapter-side seat may see is a T/V row ID (or file path) plus a rule name; no eval content and
no reserved word is ever printed.

Subcommands:
  reserved  --reserved <sealed list> (--rows <jsonl> | --file <path>)...
            Refuses any row or file whose text or gold strings use a reserved lemma (any paradigm member,
            folded inflection or compound component). Covers T, V, paraphrase-generator inputs and outputs,
            development rows, the frozen prompts and the train-to-serve template.
  collide   --eval <jsonl> --tv <jsonl> [--ledger <json>]
            T/V refusals against every development, held-out and gate row, by rule:
            string-exact, string-normalized, body-run (8 words after masking actor and place names) and
            meaning (equal canonical gold tuple). The ledger counts refusal rounds per T/V gold tuple; a tuple
            refused in three rounds is reported as DROP (R14-N1).
  examples  --eval <jsonl> --examples <jsonl>
            CA14-3: eval rows colliding at string, body or gold-tuple level with any example in the frozen
            prompts, the template or the annotation guide. The output is eval row IDs to rewrite (eval-side only).

Row JSONL fields: {"row", "utterance", "document": IntentDoc v2 gold or null, "names": [actor/place strings]}.
Exit 0: no refusal; 1: refusals printed; 4: construction error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402
import morph  # noqa: E402

BODY_RUN = 8
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str, names=()) -> str:
    s = unicodedata.normalize("NFC", text).casefold()
    for n in sorted({unicodedata.normalize("NFC", x).casefold() for x in names if x}, key=len, reverse=True):
        s = re.sub(r"(?<!\w)" + re.escape(n) + r"(?!\w)", " NAME ", s)
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def runs(text: str, n: int = BODY_RUN):
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1))}


def names_of(row):
    names = list(row.get("names", []))
    doc = row.get("document") or {}
    for it in doc.get("intents", []):
        names += [it.get("subject", ""), it.get("target", "")]
    return names


def gold_tuple(doc):
    if not doc:
        return None
    out = []
    for it in doc["intents"]:
        atoms = sorted((canon.canon(a["entity"], "entity"), canon.canon(a["state"], "state"), a["cmp"],
                        canon.canon(a["value"], "value"), a["role"], a["negated"], a["watch"])
                       for a in it["conditions"])
        out.append((it["action"], canon.canon(it["subject"], "entity"), canon.canon(it["target"], "entity"),
                    tuple(atoms)))
    return tuple(out)


def tuple_hash(doc):
    t = gold_tuple(doc)
    return None if t is None else hashlib.sha256(json.dumps(t).encode()).hexdigest()


def gold_strings(doc):
    out = []
    for it in (doc or {}).get("intents", []):
        out += [it["subject"], it["target"]]
        for a in it["conditions"]:
            out += [a["entity"], a["state"], a["value"]]
    return out


def _load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------ reserved vocabulary

def reserved_refusals(index, rows=(), files=()):
    refused = []
    for row in rows:
        texts = [row.get("utterance", "")] + gold_strings(row.get("document"))
        if any(index.match(t) for t in texts if t):
            refused.append((row["row"], "reserved-vocabulary"))
    for path in files:
        if index.match(Path(path).read_text(encoding="utf-8")):
            refused.append(("file:" + str(path), "reserved-vocabulary"))
    return refused


# ------------------------------------------------------------------ collision gate

class EvalIndex:
    """Eval-side lookup. Actor and place names are masked with one shared name set (the union of both sides'
    names), so a body reused under different names, or under names only one side's gold lists, still meets."""

    def __init__(self, eval_rows, names=()):
        self.names = set(names)
        for r in eval_rows:
            self.names |= set(names_of(r))
        self.exact, self.norm, self.runs, self.tuples = set(), set(), set(), set()
        for r in eval_rows:
            u = r.get("utterance", "")
            self.exact.add(u)
            n = normalize(u, self.names)
            self.norm |= {normalize(u), n}
            self.runs |= runs(n)
            t = gold_tuple(r.get("document"))
            if t is not None:
                self.tuples.add(t)

    def rules_hit(self, row):
        u = row.get("utterance", "")
        hits = []
        if u in self.exact:
            hits.append("string-exact")
        n = normalize(u, self.names | set(names_of(row)))
        if normalize(u) in self.norm or n in self.norm:
            hits.append("string-normalized")
        if runs(n) & self.runs:
            hits.append("body-run")
        t = gold_tuple(row.get("document"))
        if t is not None and t in self.tuples:
            hits.append("meaning")
        return hits


def collide(eval_rows, tv_rows, ledger=None):
    index = EvalIndex(eval_rows, {n for r in tv_rows for n in names_of(r)})
    ledger = ledger if ledger is not None else {"rounds": 0, "refusals": {}}
    ledger["rounds"] += 1
    out = []
    for row in tv_rows:
        hits = index.rules_hit(row)
        if not hits:
            continue
        key = tuple_hash(row.get("document")) or hashlib.sha256(row["row"].encode()).hexdigest()
        ledger["refusals"][key] = ledger["refusals"].get(key, 0) + 1
        verdict = "DROP" if ledger["refusals"][key] >= 3 else "REFUSED"
        out.append((row["row"], ",".join(hits), verdict))
    return out, ledger


def example_collisions(eval_rows, example_rows):
    index = EvalIndex(example_rows, {n for r in eval_rows for n in names_of(r)})
    return [(r["row"], ",".join(h)) for r in eval_rows for h in [index.rules_hit(r)] if h]


# ------------------------------------------------------------------ command line

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("reserved")
    a.add_argument("--reserved", required=True)
    a.add_argument("--rows", action="append", default=[])
    a.add_argument("--file", action="append", default=[])
    b = sub.add_parser("collide")
    b.add_argument("--eval", required=True)
    b.add_argument("--tv", required=True)
    b.add_argument("--ledger")
    c = sub.add_parser("examples")
    c.add_argument("--eval", required=True)
    c.add_argument("--examples", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "reserved":
            sealed = json.loads(Path(args.reserved).read_text(encoding="utf-8"))
            index = morph.ReservedIndex(sealed["entity_lemmas"] + sealed["state_lemmas"])
            rows = [r for path in args.rows for r in _load(path)]
            out = reserved_refusals(index, rows, args.file)
            for rid, rule in out:
                print(f"REFUSED {rid} {rule}")
            print(f"checked rows {len(rows)} files {len(args.file)} refused {len(out)}")
        elif args.cmd == "collide":
            ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8")) \
                if args.ledger and Path(args.ledger).exists() else None
            out, ledger = collide(_load(args.eval), _load(args.tv), ledger)
            for rid, rules, verdict in out:
                print(f"{verdict} {rid} {rules}")
            if args.ledger:
                Path(args.ledger).write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
            print(f"round {ledger['rounds']} refused {len(out)}")
        else:
            out = example_collisions(_load(args.eval), _load(args.examples))
            for rid, rules in out:
                print(f"REWRITE {rid} {rules}")
            print(f"eval rows colliding with frozen examples {len(out)}")
    except Exception as error:  # noqa: BLE001
        print(f"CONSTRUCTION ERROR: {type(error).__name__}: {error}")
        return 4
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())
