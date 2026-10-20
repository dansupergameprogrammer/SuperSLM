"""Schema fast-decode plan (rev 1.2.1) §11.0: fixture F-SFD and the reference rule's tables.

Builds a runnable `.sslm` artifact with the real Qwen vocabulary (151,936 rows) through the
reference pipeline on `main`, by the path `gen_decode_threading_fixture.py` takes
(`pipeline.fixture_model`, `convert_model.build_sections`, `sslm_format.build_artifact`), and
appends the TOK1 and UNI1 sections of `tests/fixtures/qwen2.5-1.5b.tok.sslm` and an SCM1 section
compiled against that TOK1 by `tools/sslm_convert_schema.py`. Generated fresh by every CI leg that
runs `superslm_tests` (or a forced suite), never committed. `superslm_tests` reads the directory named
by SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR, and a missing file fails every cell that needs it.

  fsfd.sslm      F-SFD: hidden 64, 2 layers, 2 query heads and 1 KV head of 32, intermediate 128,
                 tied embeddings, context cap 256, QK-norm stripped (no QKC1 table, so the
                 artifact is host-independent), the damped-greedy constants (DGC1), and four
                 schemas: POTION, PROMPT_RESULT, LONGKEY and UTF8ENUM.
  fsfd_ref.txt   The independent reference (§5.9 (d), §11.0): for every state of every schema, the
                 compiler's character-DFA verdict (forced or not), the forced text F (hex), the
                 reference rule's chain (next_forced = Encode(F)[0] if legal and a byte prefix of F,
                 else the longest legal byte-prefix token; iterated), and a token path that reaches
                 the state from the start (for SCHEMA_CONTENT prefill). Plus the golden digest of
                 (next_forced, |F|) per schema (cell 6.6).

"Reference" is computed here, outside the engine, from the compiler's character DFA and this
file's Python Encode (convert_tokenizer.TokenizerTables.ref_encode, run over the TOK1 parsed back
from the fixture). It works on bytes: F is decoded with errors="replace" (U+FFFD for an invalid
sequence, the engine's maximal-subpart rule) before Encode, never strictly.

Self-checks, each a hard failure: the Python Encode reproduces the committed golden corpus
(tests/fixtures/qwen_tok_golden.gld) id for id; Encode(A9 "pee") == [5691, 61498] (plan G12); the
five-condition token-DFA forced test equals the character DFA on every state (§5.1, cell V); every
reference chain's bytes equal F; POTION's start chain is the 3 tokens {" intent ":" (cell 8.6 (i));
LONGKEY's start text encodes to at least 16 tokens (§11.0).

Usage: python tools/gen_schema_fast_decode_fixture.py OUT_DIR
"""

import collections
import dataclasses
import hashlib
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference_pipeline"))

import convert_model as C  # noqa: E402
import convert_tokenizer as CT  # noqa: E402
import pipeline as P  # noqa: E402
import sslm_convert_schema as SC  # noqa: E402
import sslm_format as F  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402
from unicode_tables import Unicode  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_FIXTURE = os.path.join(REPO, "tests", "fixtures", "qwen2.5-1.5b.tok.sslm")
TOK_GOLDEN = os.path.join(REPO, "tests", "fixtures", "qwen_tok_golden.gld")

VOCAB = 151936
CONTEXT_CAP = 256

POTION = {"type": "object", "properties": {
    "intent": {"enum": ["buy", "sell", "ask_price", "haggle", "leave"]},
    "item": {"enum": ["health_potion", "mana_potion", "rope", "torch", "lantern", "none"]},
    "quantity": {"enum": ["one", "two", "three", "several", "none"]},
    "polite": {"type": "boolean"}}}
PROMPT_RESULT = {"type": "object", "properties": {"Prompt_Result": {"type": "string"}}}
# One key whose forced text encodes to >= 16 tokens (rare letter pairs split into many BPE pieces),
# for spans of 9+ tokens (cells 4.2, 6.2).
LONGKEY_NAME = "xqzjvkwpqxjzvkqwxpzjqvkx_zqvxjkwqpzxv"
LONGKEY = {"type": "object", "properties": {LONGKEY_NAME: {"type": "boolean"}}}
UTF8ENUM = {"type": "object", "properties": {"w": {"enum": ["épée", "aaa"]}}}

SCHEMAS = (("potion_shop_order", POTION), ("prompt_result", PROMPT_RESULT),
           ("longkey", LONGKEY), ("utf8enum", UTF8ENUM))


def config():
    return P.ModelConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, intermediate_size=128, vocab_size=VOCAB, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=CONTEXT_CAP)


def strip_qk_norm(model):
    def keep(k):
        return not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))
    return dataclasses.replace(
        model, weights={k: v for k, v in model.weights.items() if keep(k)},
        weight_scales={k: v for k, v in model.weight_scales.items() if keep(k)})


# ---- TOK1, parsed back (tests/sslm_tokenizer_fixtures.h ParseTokenizerBlob's layout) -------------

def parse_tok1(blob):
    if blob[:4] != b"TOK1":
        raise ValueError("TOK1 magic")
    _ver, vc, mc, sc, _res = struct.unpack_from("<IIIII", blob, 4)
    p = 24
    byte_to_id = list(struct.unpack_from("<256I", blob, p)); p += 1024
    offs = struct.unpack_from("<%dI" % (vc + 1), blob, p); p += 4 * (vc + 1)
    bl, = struct.unpack_from("<I", blob, p); p += 4
    vb = blob[p:p + bl]; p += bl
    id_to_bytes = [bytes(vb[offs[i]:offs[i + 1]]) for i in range(vc)]
    merges = [struct.unpack_from("<III", blob, p + 12 * i) for i in range(mc)]; p += 12 * mc
    special_ids = list(struct.unpack_from("<%dI" % sc, blob, p)); p += 4 * sc
    soffs = struct.unpack_from("<%dI" % (sc + 1), blob, p); p += 4 * (sc + 1)
    sbl, = struct.unpack_from("<I", blob, p); p += 4
    sb = blob[p:p + sbl]
    specials = [(sb[soffs[i]:soffs[i + 1]].decode("utf-8"), special_ids[i]) for i in range(sc)]
    return byte_to_id, id_to_bytes, merges, specials


def tables_from_tok1(blob):
    """A TokenizerTables built from the parsed TOK1 (no checkpoint): its ref_encode is the reference
    Encode."""
    byte_to_id, id_to_bytes, merges, specials = parse_tok1(blob)
    t = object.__new__(CT.TokenizerTables)
    t.u = Unicode.build()
    t.byte_to_id = byte_to_id
    t.merge_triples = merges
    t._rank = {(a, b): i for i, (a, b, _) in enumerate(merges)}
    t._merged = {(a, b): m for a, b, m in merges}
    t.specials = specials
    t.special_ids = {c: i for c, i in specials}
    t.id_to_bytes = id_to_bytes
    return t, id_to_bytes, frozenset(i for _, i in specials)


def encode_bytes(t, data):
    return t.ref_encode(data.decode("utf-8", errors="replace"))


def check_golden(t):
    with open(TOK_GOLDEN, "rb") as f:
        g = f.read()
    if g[:4] != b"GLD1":
        raise SystemExit("golden magic")
    n, = struct.unpack_from("<I", g, 4)
    p = 40
    for r in range(n):
        tl, = struct.unpack_from("<I", g, p); p += 4
        text = g[p:p + tl].decode("utf-8"); p += tl
        ic, = struct.unpack_from("<I", g, p); p += 4
        ids = list(struct.unpack_from("<%di" % ic, g, p)); p += 4 * ic
        got = t.ref_encode(text)
        if got != ids:
            raise SystemExit(f"reference Encode differs from the golden corpus at record {r}: {text!r}")
    got = encode_bytes(t, b"\xa9pee")
    if got != [5691, 61498]:
        raise SystemExit(f"Encode(A9 'pee') == {got}, plan G12 says [5691, 61498]")
    return n


# ---- the reference rule ---------------------------------------------------------------------------

def schema_reference(name, schema, t, vocab, special):
    mp = SC.compile_schema_to_mask_pages(schema, vocab, special_ids=special)
    states = sorted(set(mp.transitions) | {mp.start} |
                    {n for r in mp.transitions.values() for n in r.values()})
    n_states = max(states) + 1
    if states != list(range(n_states)):
        raise SystemExit(f"[{name}] state ids are not dense")
    byte_tok = {}
    for i, v in enumerate(vocab):
        if len(v) == 1 and i not in special and v[0] not in byte_tok:
            byte_tok[v[0]] = i

    # A token path from the start to every state (BFS, ascending ids), and its bytes.
    path = {mp.start: []}
    q = collections.deque([mp.start])
    while q:
        s = q.popleft()
        for tok, nxt in sorted(mp.transitions.get(s, {}).items()):
            if nxt not in path:
                path[nxt] = path[s] + [tok]
                q.append(nxt)
    if len(path) != n_states:
        raise SystemExit(f"[{name}] unreachable states")

    # The five-condition test (§5.1), cycles included.
    def step_byte(s):
        row = mp.transitions.get(s, {})
        if not row or s in mp.accepting:
            return None
        if any(tok in special or not vocab[tok] for tok in row):
            return None
        fb = {vocab[tok][0] for tok in row}
        if len(fb) != 1:
            return None
        b = fb.pop()
        bt = byte_tok.get(b)
        if bt is None or bt not in row:
            return None
        return b

    forced = {}
    ftext = {}
    for s in range(n_states):
        out = bytearray()
        w = s
        seen = set()
        cyc = False
        while True:
            b = step_byte(w)
            if b is None:
                break
            if w in seen:
                cyc = True
                break
            seen.add(w)
            out.append(b)
            w = mp.transitions[w][byte_tok[b]]
        forced[s] = bool(out) and not cyc
        ftext[s] = bytes(out) if forced[s] else b""

    # The compiler's character DFA, walked over each state's path bytes.
    cdfa, cstart, cacc, _content = SC._char_dfa(SC._groups(schema, "$"))
    mism = 0
    for s in range(n_states):
        c = cstart
        for tok in path[s]:
            for b in vocab[tok]:
                c = cdfa[c][b]
        run = bytearray()
        while c not in cacc and len(cdfa[c]) == 1:
            (b, nxt), = cdfa[c].items()
            run.append(b)
            c = nxt
        if bytes(run) != ftext[s]:
            mism += 1
    if mism:
        raise SystemExit(f"[{name}] five-condition test differs from the character DFA on {mism} states")

    # next_forced per forced state, then the chain.
    nxt_tok = {}
    fallback = 0
    for s in range(n_states):
        if not forced[s]:
            continue
        Fs = ftext[s]
        row = mp.transitions[s]
        ids = encode_bytes(t, Fs)
        first = ids[0] if ids else None
        if first is not None and first in row and Fs.startswith(vocab[first]):
            nxt_tok[s] = first
        else:
            cands = [tok for tok in row if vocab[tok] and Fs.startswith(vocab[tok])]
            best = max(len(vocab[tok]) for tok in cands)
            top = [tok for tok in cands if len(vocab[tok]) == best]
            if len(top) != 1:
                raise SystemExit(f"[{name}] state {s}: {len(top)} longest legal prefixes")
            nxt_tok[s] = top[0]
            fallback += 1
    chains = {}
    for s in range(n_states):
        chain = []
        w = s
        while w in nxt_tok:
            chain.append(nxt_tok[w])
            w = mp.transitions[w][nxt_tok[w]]
            if len(chain) > n_states:
                raise SystemExit(f"[{name}] chain from {s} does not end")
        if b"".join(vocab[x] for x in chain) != ftext[s]:
            raise SystemExit(f"[{name}] chain bytes from {s} differ from F")
        chains[s] = chain

    h = hashlib.sha256()
    for s in range(n_states):
        h.update(struct.pack("<iI", nxt_tok.get(s, -1), len(ftext[s])))
    info = {"mp": mp, "n": n_states, "forced": forced, "F": ftext, "chains": chains, "path": path,
            "fallback": fallback, "digest": h.hexdigest()}
    print(f"[{name}] states={n_states} forced={sum(forced.values())} fallbacks={fallback} "
          f"max_chain={max((len(c) for c in chains.values()), default=0)} "
          f"max_row={max(len(r) for r in mp.transitions.values())} digest={info['digest']}")
    return info


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out_dir = argv[1]
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()

    tok_blob = F.read_section_bytes(TOK_FIXTURE, F.SectionType.TOKENIZER)
    uni_blob = F.read_section_bytes(TOK_FIXTURE, F.SectionType.UNICODE_TABLES)
    if tok_blob is None or uni_blob is None:
        raise SystemExit(f"{TOK_FIXTURE}: no TOK1/UNI1")
    t, id_to_bytes, special = tables_from_tok1(tok_blob)
    records = check_golden(t)
    print(f"reference Encode reproduces {records} golden records; {len(id_to_bytes)} TOK1 entries, "
          f"{len(special)} specials")
    vocab = id_to_bytes + [b""] * (VOCAB - len(id_to_bytes))

    infos = []
    for name, schema in SCHEMAS:
        infos.append((name, schema_reference(name, schema, t, vocab, special)))
    potion = infos[0][1]
    start_chain = [vocab[x] for x in potion["chains"][potion["mp"].start]]
    if start_chain != [b'{"', b"intent", b'":"']:
        raise SystemExit(f"POTION start chain is {start_chain}, cell 8.6 (i) expects 3 tokens")
    lk = infos[2][1]
    lk_enc = encode_bytes(t, lk["F"][lk["mp"].start])
    if len(lk_enc) < 16 or len(lk["chains"][lk["mp"].start]) < 9:
        raise SystemExit(f"LONGKEY start text encodes to {len(lk_enc)} tokens; §11.0 needs >= 16")

    model = strip_qk_norm(P.fixture_model(config()))
    sections, _ = C.build_sections(model, enable_damped_greedy=True)
    flags = C.artifact_flags_for_model(model) | F.DAMPED_GREEDY_CONSTANTS_FLAG
    sections.append(F.Section(F.SectionType.TOKENIZER, tok_blob))
    sections.append(F.Section(F.SectionType.UNICODE_TABLES, uni_blob))
    sections.append(F.Section(F.SectionType.SCHEMA_MASKS,
                              _serialize_scm1([(n, i["mp"]) for n, i in infos], VOCAB)))
    data, fingerprint = F.build_artifact(sections, flags=flags)
    path = os.path.join(out_dir, "fsfd.sslm")
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {path}: {len(data)} bytes, fingerprint={fingerprint}")

    lines = ["SFDREF 1"]
    for name, i in infos:
        mp = i["mp"]
        lines.append(f"schema {name} {i['n']} {mp.start} {i['digest']}")
        for s in range(i["n"]):
            fhex = i["F"][s].hex() or "-"
            ch = i["chains"][s]
            pa = i["path"][s]
            lines.append(" ".join(
                ["state", str(s), "1" if s in mp.accepting else "0", "1" if i["forced"][s] else "0",
                 str(len(i["F"][s])), fhex, str(len(ch))] + [str(x) for x in ch] +
                [str(len(pa))] + [str(x) for x in pa]))
    lines.append("end")
    ref = os.path.join(out_dir, "fsfd_ref.txt")
    with open(ref, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {ref}: {len(lines)} lines; {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
