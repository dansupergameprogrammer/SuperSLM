"""S2 fixtures for the JSON-array runtime harness (tests/array-s2/array_s2_main.cpp).

Builds the token-id inputs the harness reads, using the pinned Qwen2.5-0.5B-Instruct tokenizer
(`tokenizer.json` of the checkpoint the artifact was converted from), and post-processes the
harness's outputs. Everything here is independent of the engine: token ids come from the
`tokenizers` library, the expected rejection token of each bad document is derived from the
byte offset at which the document leaves IntentDoc v2 (known by construction), and generated
documents are validated with `jsonschema`.

    python fixtures.py build  --ckpt <checkpoint dir> --out <fixture file>
    python fixtures.py saves  --ckpt <checkpoint dir> --gen <harness gen output> --out <save plan>
    python fixtures.py check  --ckpt <checkpoint dir> --gen <harness gen output> [--json <report>]

The fixture file is line-oriented text, one record per line, `kind name count id id ...`, so the
C++ harness needs no JSON parser.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "t3094-multicommand-s0" / "v2"
SCHEMA_PATH = V2 / "intentdoc-v2.schema.json"
PROMPT_PATH = V2 / "prompt-intentdoc-v2.txt"
EXAMPLES_PATH = V2 / "frozen-examples.jsonl"

# Ten complete utterances from frozen-examples.jsonl (the remaining rows are fragments or guide
# prose). ex041, ex042 and ex053 request two actions each.
PROMPT_ROWS = ["ex005", "ex006", "ex033", "ex034", "ex035", "ex041", "ex042", "ex051", "ex052", "ex053"]

COND_STOP = '{"role":"stop","entity":"beacon","state":"lit","cmp":"none","value":"-","negated":false,"watch":"auto"}'
COND_START = '{"role":"start","entity":"gate","state":"open","cmp":"none","value":"-","negated":true,"watch":"auto"}'
COND_HEALTH = '{"role":"start","entity":"your health","state":"health","cmp":"below","value":"half","negated":false,"watch":"new"}'


def intent(action: str, target: str, conds: list[str], dep: bool) -> str:
    return ('{"action":"%s","subject":"you","target":"%s","conditions":[%s],"depends_on_previous":%s}'
            % (action, target, ",".join(conds), "true" if dep else "false"))


def doc(intents: list[str]) -> str:
    return '{"intents":[' + ",".join(intents) + "]}"


# Teacher-forced walks, each a valid IntentDoc v2 document in the compiler's compact form.
WALKS = {
    "one_intent_0_conditions": doc([intent("fire", "the bridge", [], False)]),
    "one_intent_1_condition": doc([intent("hold", "the bridge", [COND_STOP], False)]),
    "three_intents_0_2_1_conditions": doc([
        intent("scan", "north ridge", [], False),
        intent("fire", "the tower", [COND_START, COND_HEALTH], True),
        intent("move", "the gate", [COND_STOP], False),
    ]),
}

_W1 = intent("fire", "the bridge", [], False)
# Rejections: (document, byte offset of the first byte no IntentDoc v2 document has there).
REJECTS = {
    # `,` after an item admits only `{`; the `]` after it is the first bad byte.
    "trailing_comma": ('{"intents":[' + _W1 + ",]}", len('{"intents":[' + _W1 + ",")),
    # minItems 1: `[` admits only `{`.
    "empty_intents": ('{"intents":[]}', len('{"intents":[')),
    # "target" is required after "subject"; the `"` of the next key is shared, its `c` is not `t`.
    "missing_target": ('{"intents":[{"action":"fire","subject":"you","conditions":[],"depends_on_previous":false}]}',
                       len('{"intents":[{"action":"fire","subject":"you","')),
}


def load_tokenizer(ckpt: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ckpt / "tokenizer.json"))


def chat_prompt(system: str, user: str) -> str:
    return ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user
            + "<|im_end|>\n<|im_start|>assistant\n")


def utterances() -> dict[str, str]:
    rows = {}
    for line in EXAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        rows[d["row"]] = d["utterance"]
    return {r: rows[r] for r in PROMPT_ROWS}


FREE_TEXT_KEYS = ("subject", "target", "entity", "state", "value")
_OPENINGS = re.compile(r'"(?:%s)":(")' % "|".join(FREE_TEXT_KEYS))


def schema_encode(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and byte spans of `text`, encoded so the compiled table can admit them: the
    compiler opens a free-text string only through the bare `"` token (the T-2910 open-boundary
    rule in tools/sslm_convert_schema.py), so the text is encoded in segments split around each
    free-text value's opening quote, and each segment is encoded with the pinned tokenizer."""
    cuts = [0]
    for m in _OPENINGS.finditer(text):
        cuts += [m.start(1), m.end(1)]
    cuts.append(len(text))
    ids, spans = [], []
    for a, b in zip(cuts, cuts[1:]):
        if a == b:
            continue
        enc = tok.encode(text[a:b], add_special_tokens=False)
        ids += enc.ids
        spans += [(a + s, a + e) for s, e in enc.offsets]
    assert tok.decode(ids) == text
    return ids, spans


def token_containing(spans: list[tuple[int, int]], offset: int) -> int:
    for i, (b, e) in enumerate(spans):
        if b <= offset < e:
            return i
    raise ValueError(f"no token covers offset {offset}")


def build(args) -> int:
    tok = load_tokenizer(args.ckpt)
    system = PROMPT_PATH.read_text(encoding="utf-8").strip()
    lines = []
    for row, utt in utterances().items():
        ids = tok.encode(chat_prompt(system, utt), add_special_tokens=False).ids
        lines.append(f"prompt {row} {len(ids)} " + " ".join(map(str, ids)))
    for name, text in WALKS.items():
        json.loads(text)
        ids, _ = schema_encode(tok, text)
        lines.append(f"walk {name} {len(ids)} " + " ".join(map(str, ids)))
    for name, (text, bad) in REJECTS.items():
        assert text.isascii(), name
        ids, spans = schema_encode(tok, text)
        expect = token_containing(spans, bad)
        lines.append(f"reject {name}@{expect} {len(ids)} " + " ".join(map(str, ids)))
    args.out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(lines)} records to {args.out}")
    return 0


def read_gen(path: Path) -> dict[str, dict]:
    """Harness gen output: `gen <backend> <row> <end> <count> id ...`."""
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        f = line.split()
        if not f or f[0] != "gen":
            continue
        backend, row, end, n = f[1], f[2], f[3], int(f[4])
        out.setdefault(row, {})[backend] = {"end": end, "ids": [int(x) for x in f[5:5 + n]]}
    return out


def structural_saves(tok, ids: list[int]) -> list[tuple[int, str]]:
    """Token indices i such that a save after ids[i] sits at a JSON array position: the token's
    bytes contain an array `[` or `]`, or a comma whose innermost container is an array. The kind
    names which, and in which array (`intents` or `conditions`)."""
    saves = []
    stack: list[str] = []  # "o" for an object, else the array's key
    in_str = esc = False
    last_key = None
    key_buf = None
    for i, tid in enumerate(ids):
        kinds = []
        for ch in tok.decode([tid]):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                    if key_buf is not None:
                        last_key, key_buf = key_buf, None
                elif key_buf is not None:
                    key_buf += ch
                continue
            if ch == '"':
                in_str = True
                key_buf = "" if (stack and stack[-1] == "o") else None
            elif ch == "{":
                stack.append("o")
            elif ch == "}":
                stack.pop()
            elif ch == "[":
                stack.append(last_key or "?")
                kinds.append(f"open_{stack[-1]}")
            elif ch == "]":
                kinds.append(f"close_{stack.pop()}")
            elif ch == "," and stack and stack[-1] != "o":
                kinds.append(f"comma_{stack[-1]}")
        if kinds:
            saves.append((i, "+".join(kinds)))
    return saves


def saves(args) -> int:
    tok = load_tokenizer(args.ckpt)
    gen = read_gen(args.gen)
    lines = []
    for row, by_backend in gen.items():
        ref = by_backend.get("cpu")
        if not ref:
            continue
        for i, kind in structural_saves(tok, ref["ids"]):
            lines.append(f"save {row} {i} {kind}")
    args.out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(lines)} save points to {args.out}")
    return 0


def check(args) -> int:
    import jsonschema
    tok = load_tokenizer(args.ckpt)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    gen = read_gen(args.gen)
    report = {}
    failures = 0
    for row, by_backend in gen.items():
        entry = {}
        for backend, g in by_backend.items():
            text = tok.decode(g["ids"])
            res = {"end": g["end"], "tokens": len(g["ids"]), "text": text}
            try:
                errs = [e.message for e in validator.iter_errors(json.loads(text))]
                res["valid"] = not errs
                if errs:
                    res["errors"] = errs[:3]
            except json.JSONDecodeError as e:
                res["valid"] = False
                res["errors"] = [f"json: {e}"]
            failures += not res["valid"]
            entry[backend] = res
        backends = sorted(by_backend)
        base = by_backend[backends[0]]["ids"]
        for b in backends[1:]:
            other = by_backend[b]["ids"]
            same = base == other
            if not same:
                failures += 1
                k = next((j for j in range(min(len(base), len(other))) if base[j] != other[j]),
                         min(len(base), len(other)))
                entry[f"mismatch_{backends[0]}_vs_{b}"] = {
                    "position": k, "a": base[k] if k < len(base) else None,
                    "b": other[k] if k < len(other) else None}
            entry[f"identical_{backends[0]}_vs_{b}"] = same
        report[row] = entry
        valid = all(entry[b]["valid"] for b in backends)
        print(f"{row}: {'valid' if valid else 'INVALID'} "
              + " ".join(f"{b}={entry[b]['tokens']}t/{entry[b]['end']}" for b in backends)
              + "".join(f" {k}={v}" for k, v in entry.items() if k.startswith(("identical", "mismatch"))))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"failures: {failures}")
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "saves", "check"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", type=Path, required=True)
        if name != "build":
            s.add_argument("--gen", type=Path, required=True)
        if name != "check":
            s.add_argument("--out", type=Path, required=True)
        else:
            s.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    return {"build": build, "saves": saves, "check": check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
