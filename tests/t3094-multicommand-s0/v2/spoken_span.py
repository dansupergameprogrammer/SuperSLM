"""T-3103 spoken-span gold check (plan rev 15 s5.2 items 7 and 8, R12-N2). Blocks a freeze on violation.

Every acceptable entity, state and value string, after canonicalizer C, must be a contiguous token span of
the utterance. The utterance is normalized the same way: NFC, lowercase, possessive clitic removed,
punctuation stripped, number words zero to twenty and percent mapped. Three exemptions apply:
- an atom field marked resolved (a pronoun or possessive resolved by s1.2 rows 14-15), whose members may
  be any span of the utterance, the addressee's name or `player`;
- the value sentinel '-', which is the only acceptable value when cmp is none (item 8), and is never a
  spoken value;
- the detection arm's fixed placeholder atom (`no condition`).

Gold rows JSONL: {"row", "utterance", "document", "spans", "resolved": [[{"entity": bool, ...}]] optional,
                  "addressee": optional}
Usage: py -3 spoken_span.py --gold <jsonl>    prints only row IDs and rule names; exit 1 on any violation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon  # noqa: E402

_CLITIC = re.compile(r"(?<=\w)['’]s\b")
_PUNCT = re.compile(r"[^\w\s%]", re.UNICODE)


def norm_tokens(text: str):
    s = unicodedata.normalize("NFC", text).lower()
    s = _CLITIC.sub("", s)
    s = re.sub(r"\bper cent\b", "%", s)
    s = re.sub(r"\bpercent\b", "%", s)
    s = _PUNCT.sub(" ", s)
    return [canon.NUMBERS.get(w, w) for w in s.split()]


def is_span(member: str, field: str, utt_tokens):
    toks = canon.canon(member, field).replace("%", " % ").split()
    if not toks:
        return False
    n = len(toks)
    return any(utt_tokens[i:i + n] == toks for i in range(len(utt_tokens) - n + 1))


def check_row(row):
    violations = []
    utt = norm_tokens(row["utterance"])
    doc = row.get("document") or {"intents": []}
    for i, it in enumerate(doc["intents"]):
        for j, atom in enumerate(it["conditions"]):
            spans = row["spans"][i][j]
            resolved = (row.get("resolved") or [[{}] * len(x["conditions"]) for x in doc["intents"]])[i][j]
            if atom["entity"] == "no condition" and atom["state"] == "no condition":
                continue
            if atom["cmp"] == "none":
                if atom["value"] != "-" or spans.get("value") not in ([], ["-"], None):
                    violations.append((row["row"], f"intent{i}.atom{j}.value", "sentinel"))
            elif "-" in spans.get("value", []):
                violations.append((row["row"], f"intent{i}.atom{j}.value", "sentinel-as-spoken-value"))
            for field in ("entity", "state", "value"):
                if field == "value" and atom["cmp"] == "none":
                    continue
                for member in spans.get(field, []):
                    if resolved.get(field) and (member in ("player", row.get("addressee")) or
                                                is_span(member, field, utt)):
                        continue
                    if not is_span(member, field, utt):
                        violations.append((row["row"], f"intent{i}.atom{j}.{field}", "not-a-spoken-span"))
                        break
    return violations


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    args = ap.parse_args(argv)
    try:
        rows = [json.loads(l) for l in Path(args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
        out = [v for r in rows for v in check_row(r)]
    except Exception as error:  # noqa: BLE001
        print(f"CONSTRUCTION ERROR: {type(error).__name__}: {error}")
        return 4
    for rid, where, rule in out:
        print(f"VIOLATION {rid} {where} {rule}")
    print(f"rows {len(rows)} violations {len(out)}")
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())
