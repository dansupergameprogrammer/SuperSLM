"""S2 fixtures for the JSON-array runtime harness (tests/array-s2/array_s2_main.cpp).

Builds the token-id inputs the harness reads, using the pinned Qwen2.5-0.5B-Instruct tokenizer
(`tokenizer.json` of the checkpoint the artifact was converted from), and post-processes the
harness's outputs. Everything here is independent of the engine: token ids come from the
`tokenizers` library, the expected rejection token of each bad document is derived from the
byte offset at which the document leaves IntentDoc v2 (known by construction), and generated
documents are validated with `jsonschema`.

    python fixtures.py build    --ckpt <checkpoint dir> --out <fixture file>
    python fixtures.py precheck --ckpt <checkpoint dir> --fixtures <fixture file> [--out <report>]
    python fixtures.py saves    --ckpt <checkpoint dir> --gen <harness gen output> --out <save plan>
    python fixtures.py check    --ckpt <checkpoint dir> --gen <harness gen output> [--json <report>]

The fixture file is line-oriented text, one record per line, `kind name count id id ...`, so the
C++ harness needs no JSON parser. Record kinds: `prompt`, `walk`, `reject` (name `case@index`),
`tfdoc` (a teacher-forced document) and `tfsave` (name `doc:kind`, one id: the token index the
save follows).

`precheck` compiles IntentDoc v2 with the repository's converter over the pinned tokenizer and
walks every walk, reject and tfdoc record through the compiled table in Python, so a fixture the
table cannot admit is caught before the box runs it. It needs no engine and no model.
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
# Inside `conditions`: `,` after an atom admits only `{`; the `]` after it is the first bad byte.
_COND_TRAILING = ('{"intents":[{"action":"hold","subject":"you","target":"the bridge","conditions":['
                  + COND_STOP + ",")
REJECTS["trailing_comma_conditions"] = (_COND_TRAILING + '],"depends_on_previous":false}]}', len(_COND_TRAILING))

# Teacher-forced save points (step 4e). Each walk document is re-tokenized with a token boundary
# exactly after every array byte the brief lists -- after `[` of `intents`, after each `,` between
# intents, after `[` and after each `,` inside `conditions`, and after each `]` -- plus, in the
# three-intent document, one boundary inside an intent's free-text value (intent 2's target) and
# one inside a condition's free-text value (intent 3's entity). Each entry: (needle, offset of the
# cut inside the needle, kind); the needle occurs exactly once in its document.
TF_STRING_CUTS = {
    "three_intents_0_2_1_conditions": [
        ('"target":"the tower"', len('"target":"the'), "string_intent"),
        ('"entity":"beacon"', len('"entity":"be'), "string_condition"),
    ],
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


def schema_encode(tok, text: str, extra_cuts=()) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and byte spans of `text`, encoded so the compiled table can admit them: the
    compiler opens a free-text string only through the bare `"` token (the T-2910 open-boundary
    rule in tools/sslm_convert_schema.py), so the text is encoded in segments split around each
    free-text value's opening quote, and each segment is encoded with the pinned tokenizer.
    `extra_cuts` adds byte offsets that must also fall on a token boundary."""
    cuts = [0]
    for m in _OPENINGS.finditer(text):
        cuts += [m.start(1), m.end(1)]
    cuts += list(extra_cuts)
    cuts.append(len(text))
    cuts = sorted(set(cuts))
    ids, spans = [], []
    for a, b in zip(cuts, cuts[1:]):
        if a == b:
            continue
        enc = tok.encode(text[a:b], add_special_tokens=False)
        ids += enc.ids
        spans += [(a + s, a + e) for s, e in enc.offsets]
    assert tok.decode(ids) == text
    return ids, spans


def array_events(text: str) -> list[tuple[int, str]]:
    """(byte offset, class) of every array byte in a JSON text: `open_<array>` for `[`,
    `close_<array>` for `]`, and `comma_<array>` for a comma whose innermost container is that
    array, where <array> is the array's key (`intents` or `conditions`)."""
    events = []
    stack: list[str] = []  # "o" for an object, else the array's key
    in_str = esc = False
    last_key = None
    key_buf = None
    for off, ch in enumerate(text):
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
            events.append((off, f"open_{stack[-1]}"))
        elif ch == "]":
            events.append((off, f"close_{stack.pop()}"))
        elif ch == "," and stack and stack[-1] != "o":
            events.append((off, f"comma_{stack[-1]}"))
    return events


def tf_points(name: str, text: str) -> list[tuple[int, str]]:
    """(cut offset, kind) of every teacher-forced save point of a document: right after each
    array byte, and at the free-text cuts TF_STRING_CUTS names."""
    points = [(off + 1, kind) for off, kind in array_events(text)]
    for needle, inner, kind in TF_STRING_CUTS.get(name, []):
        assert text.count(needle) == 1, (name, needle)
        points.append((text.index(needle) + inner, kind))
    return sorted(points)


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
    tf_kinds = []
    for name, text in WALKS.items():
        points = tf_points(name, text)
        ids, spans = schema_encode(tok, text, [off for off, _ in points])
        ends = {e: i for i, (_, e) in enumerate(spans)}
        lines.append(f"tfdoc {name} {len(ids)} " + " ".join(map(str, ids)))
        # The same tokens as an ordinary walk, so step (b) also admits them token by token.
        lines.append(f"walk tf_{name} {len(ids)} " + " ".join(map(str, ids)))
        for off, kind in points:
            assert off in ends, f"{name}: no token boundary at byte {off} ({kind})"
            lines.append(f"tfsave {name}:{kind} 1 {ends[off]}")
            tf_kinds.append(kind)
    args.out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(lines)} records to {args.out}")
    print(f"teacher-forced save points: {len(tf_kinds)}, every one exactly at its byte")
    for kind in sorted(set(tf_kinds)):
        print(f"  {kind}: {tf_kinds.count(kind)}")
    return 0


def read_records(path: Path) -> list[tuple[str, str, list[int]]]:
    out = []
    for line in path.read_text(encoding="ascii").splitlines():
        f = line.split()
        if len(f) >= 3:
            out.append((f[0], f[1], [int(x) for x in f[3:3 + int(f[2])]]))
    return out


def precheck(args) -> int:
    """Walks every fixture through IntentDoc v2 compiled by the repository's converter (the same
    compile `sslm_attach_schema.py` runs), over the pinned tokenizer."""
    import time
    sys.path[:0] = [str(HERE.parents[1] / "tools"), str(HERE.parents[1] / "tools" / "reference_pipeline")]
    import sslm_convert_schema as SC
    from convert_tokenizer import TokenizerTables
    vocab_size = json.loads((args.ckpt / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    tables = TokenizerTables(str(args.ckpt))
    vocab = [tables.id_to_bytes[i] if i < len(tables.id_to_bytes) else b"" for i in range(vocab_size)]
    special = frozenset(token_id for _, token_id in tables.specials)
    t0 = time.perf_counter()
    mp = SC.compile_schema_to_mask_pages(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")), vocab,
                                         special_ids=special, name=b"intents")
    out = [f"compiled IntentDoc v2 in {time.perf_counter() - t0:.1f} s: {mp.resource_counts['dfa_states']} states, "
           f"{mp.resource_counts['token_transitions']} transitions"]
    failures = checks = 0
    tfdocs = {}
    for kind, name, ids in read_records(args.fixtures):
        if kind == "tfdoc":
            tfdocs[name] = ids
        if kind not in ("walk", "reject"):
            continue
        state, admitted, trace = mp.start, 0, []
        for t in ids:
            row = mp.transitions.get(state, {})
            if t not in row:
                break
            state = row[t]
            admitted += 1
            trace.append(1 if mp.is_accepting(state) else 0)
        if kind == "walk":
            want_trace = [0] * (len(ids) - 1) + [1]
            ok = admitted == len(ids) and trace == want_trace
        else:
            expect = int(name.split("@")[1])
            ok = admitted == expect and not any(trace)
        checks += 1
        failures += not ok
        out.append(f"{'PASS' if ok else 'FAIL'} {kind} {name} tokens={len(ids)} admitted={admitted} "
                   f"accepting_before_end={sum(trace[:-1]) if kind == 'walk' else sum(trace)}")
    for kind, name, ids in read_records(args.fixtures):
        if kind != "tfsave":
            continue
        doc, what = name.split(":", 1)
        ok = doc in tfdocs and 0 <= ids[0] < len(tfdocs[doc]) - 1
        checks += 1
        failures += not ok
        out.append(f"{'PASS' if ok else 'FAIL'} tfsave {name} after={ids[0]}")
    if checks == 0:
        failures += 1
        out.append("FAIL no records checked")
    out.append(f"checks={checks} failures={failures}")
    text = "\n".join(out) + "\n"
    print(text, end="")
    if args.out:
        args.out.write_text(text, encoding="ascii")
    return 1 if failures else 0


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
    """Token indices i whose bytes hold an array byte, with a label for each such byte:
    `x:<class>` when it is the token's last byte (a save after ids[i] sits exactly after it),
    `b:<class>` when later bytes of the same token follow it (the save sits past it, bundled),
    and `c:<class>` for an `[` whose `]` is in the same token (crossed: the save is never at the
    after-`[` state). Labels are joined with `+`."""
    pieces = [tok.decode([t]) for t in ids]
    text = "".join(pieces)
    assert text == tok.decode(ids), "a token is not a whole UTF-8 piece; spans would be wrong"
    ends, pos = [], 0
    for piece in pieces:
        pos += len(piece)
        ends.append(pos)
    by_token: dict[int, list[tuple[int, str]]] = {}
    i = 0
    for off, cls in array_events(text):
        while ends[i] <= off:
            i += 1
        by_token.setdefault(i, []).append((off, cls))
    saves = []
    for i, events in sorted(by_token.items()):
        end = ends[i]
        crossed = set()
        opens = []
        for k, (off, cls) in enumerate(events):
            if cls.startswith("open_"):
                opens.append(k)
            elif cls.startswith("close_") and opens:
                crossed.add(opens.pop())
        labels = []
        for k, (off, cls) in enumerate(events):
            tag = "c" if k in crossed else ("x" if off == end - 1 else "b")
            labels.append(f"{tag}:{cls}")
        saves.append((i, "+".join(labels)))
    return saves


def save_counts(kinds: list[str]) -> dict[str, dict[str, int]]:
    """Per array-byte class: how many saves sit exactly after it, how many are bundled past it,
    and how many tokens crossed it (an `[` closed in the same token)."""
    out: dict[str, dict[str, int]] = {}
    for kind in kinds:
        for label in kind.split("+"):
            tag, cls = label.split(":", 1)
            if tag not in "xbc":
                continue
            row = out.setdefault(cls, {"exact": 0, "bundled": 0, "crossed": 0})
            row[{"x": "exact", "b": "bundled", "c": "crossed"}[tag]] += 1
    return out


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
    if not lines:
        print("no save points: the gen file holds no cpu stream")
        return 1
    args.out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(lines)} save points to {args.out}")
    print_counts(save_counts([line.split()[3] for line in lines]))
    return 0


def print_counts(counts: dict[str, dict[str, int]]) -> None:
    for cls in sorted(counts):
        c = counts[cls]
        print(f"  after {cls}: exact {c['exact']}, bundled {c['bundled']}, crossed {c['crossed']}")


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
    for name in ("build", "precheck", "saves", "check"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", type=Path, required=True)
        if name in ("saves", "check"):
            s.add_argument("--gen", type=Path, required=True)
        if name == "precheck":
            s.add_argument("--fixtures", type=Path, required=True)
            s.add_argument("--out", type=Path)
        elif name == "check":
            s.add_argument("--json", type=Path)
        else:
            s.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    return {"build": build, "precheck": precheck, "saves": saves, "check": check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
