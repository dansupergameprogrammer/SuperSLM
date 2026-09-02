"""Convert a Qwen/GPT-lineage byte-level BPE tokenizer into a `.sslm` artifact, and
emit the golden reference pack. Build-time tooling (Python ships nothing).

Two jobs:
  1. Extract the tokenizer tables (byte->id base map, id->raw-bytes vocab, rank-ordered
     merges as (id_a,id_b,id_merged) triples, special tokens, chat template) plus the
     pinned Unicode tables (NFC + \\p{L}/\\p{N} property classes) and write them into
     the artifact's TOKENIZER / UNICODE_TABLES / CHAT_TEMPLATE sections.
  2. A reference `encode()` that reads exactly those tables and reproduces the tokenizer
     with no third-party dependency and no float — the algorithm the C++ TokenizerView
     mirrors. `--verify` checks it byte-for-byte against the upstream HF tokenizer over a
     corpus; that parity is the whole gate (SuperSLM_Plan.md §10).

The pre-tokenization pattern is the fixed Qwen/GPT byte-level-BPE regex; its only
Unicode-dependent inputs are \\p{L} / \\p{N} membership, which come from the pinned
tables so the runtime never calls a platform regex or Unicode library (D-SLM13).
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import sslm_format as F
import sslm_model_writer as W
from unicode_tables import Unicode

# The pattern this converter and the runtime implement (documented, pinned for v1).
PRETOK_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


# --- GPT-2 byte<->unicode mapping (deterministic, standard) ---------------------
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}


class UnsupportedTokenizerShape(ValueError):
    """Raised, with a named diagnostic, when `tokenizer.json` carries a `model.merges`
    element or a `post_processor` shape this converter does not recognize -- never a
    silent guess (N3 discipline:
    `Claude/Vitruvius/t2408-superslm-ask5-qwen3-arch-design-2026-08-29.md` §6 Track C
    step 1, applied to Track E in §6 Track E steps 1-2)."""


def _parse_merge_element(m, index):
    """Each element of `model["merges"]` is either the incumbent schema -- a single
    space-separated string, e.g. "a b" -- or the schema this checkpoint's own
    `tokenizers` library version emits: a 2-element [a, b] list (no join/split
    needed). Closes `TOK-04` / D-SLM5573 (t2408 §2.9, §6 Track E step 1): any other
    element shape is an explicit rejection, never a silent guess."""
    if isinstance(m, str):
        return m.split(" ")
    if isinstance(m, (list, tuple)) and len(m) == 2:
        return [m[0], m[1]]
    raise UnsupportedTokenizerShape(
        f"model.merges[{index}]: unrecognized merge element {m!r} "
        f"(expected a space-separated string or a 2-element list/tuple)"
    )


def derive_model_name(ckpt_dir):
    """Best-effort model label for the CONFIG blob, from the checkpoint path alone --
    it is the only identifying signal available. `tokenizer_class`/`model_type` in
    tokenizer_config.json/config.json are generic across the whole Qwen2 family and do
    not distinguish 1.5B from 3B, or base from instruct.

    Two checkpoint-path shapes occur in this project: the HF hub cache's content-hash
    snapshot dirs (`models--<org>--<repo>/snapshots/<rev>`, where the leaf name is a
    hash and the repo name -- the useful part -- sits two levels up), and flat exported
    directories whose own leaf name is already descriptive (e.g. a merged-LoRA output).
    Detect the first shape by its `models--` cache-key convention; fall back to the
    checkpoint directory's own leaf name for everything else.

    T-2529 (converter-validate, linux-x64 job): a checkpoint path is not always authored
    on the OS that later runs this converter -- a Windows-produced HF hub cache copied
    onto a Linux build box, or a path recorded verbatim in a manifest and read back on a
    different machine, both carry Windows-style backslash separators regardless of where
    the string is finally parsed. `pathlib.Path` splits on the HOST's own separator
    convention (`PurePosixPath` never treats `\\` as a separator), so a backslash path
    handed to this function on Linux was read as one opaque leaf component and returned
    whole rather than reaching either branch above -- reproduced by direct execution
    (`tools/test_convert_tokenizer.py`'s own fixtures, all Windows-style paths, run under
    Linux). Parses the string directly on both separators instead of delegating to
    `pathlib`, so the same path string derives the same model name regardless of which OS
    is running this function.
    """
    parts = [seg for seg in re.split(r"[\\/]+", str(ckpt_dir)) if seg not in ("", ".")]
    leaf = parts[-1] if parts else str(ckpt_dir)
    parent = parts[-2] if len(parts) >= 2 else None
    grandparent = parts[-3] if parent == "snapshots" and len(parts) >= 3 else None
    if grandparent and grandparent.startswith("models--") and grandparent.count("--") >= 2:
        repo = grandparent.split("--", 2)[2]
        return repo.lower()
    return leaf.lower()


# --- The tokenizer tables extracted from an HF checkpoint -----------------------
class TokenizerTables:
    def __init__(self, ckpt_dir):
        self.ckpt = ckpt_dir
        self.model_name = derive_model_name(ckpt_dir)
        tj = json.loads((Path(ckpt_dir) / "tokenizer.json").read_text(encoding="utf-8"))
        cfg = json.loads((Path(ckpt_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
        model = tj["model"]
        assert model["type"] == "BPE", model["type"]
        assert tj.get("normalizer", {}).get("type") == "NFC", tj.get("normalizer")

        self.vocab = model["vocab"]                       # byte-level-string -> id
        self.id_to_tok = {v: k for k, v in self.vocab.items()}
        self.merges = [_parse_merge_element(m, i)          # [ [a,b], ... ] rank order
                       for i, m in enumerate(model["merges"])]
        self.added = tj.get("added_tokens", [])
        self.chat_template = cfg.get("chat_template")
        # All NFC + \p{L}/\p{N}/\s classification runs through the table-driven Unicode
        # (proven == unicodedata by unicode_tables.self_test) — never unicodedata in the
        # tokenizer path, so the runtime and this reference share one source of truth.
        self.u = Unicode.build()
        self.unicode_version = self.u.version

        # base byte -> id
        self.byte_to_id = [self.vocab[BYTE_ENCODER[b]] for b in range(256)]
        # rank-ordered merge triples (id_a, id_b, id_merged)
        self.merge_triples = []
        for a, b in self.merges:
            self.merge_triples.append((self.vocab[a], self.vocab[b], self.vocab[a + b]))
        # merge rank lookup for the reference encoder
        self._rank = {(a, b): i for i, (a, b, _) in enumerate(self.merge_triples)}
        self._merged = {(a, b): m for a, b, m in self.merge_triples}
        # special tokens, longest-content first so overlapping specials match greedily.
        # T-2243 review finding 11 (D-SLM4113): sorted by ENCODED BYTE length, not Python
        # character count -- the reader (src/tokenizer.cpp) validates the stored order against
        # byte length, so a non-ASCII special (where character count and UTF-8 byte length
        # diverge) used to be written in an order the reader could reject as "not stored
        # longest-content-first" even though every special was in fact written longest-first by
        # the writer's own intended invariant. Latent on every artifact shipped so far (every
        # special observed in the field is ASCII, where the two measures agree).
        self.specials = sorted(((x["content"], x["id"]) for x in self.added),
                               key=lambda kv: len(kv[0].encode("utf-8")), reverse=True)
        self.special_ids = {c: i for c, i in self.specials}
        # id -> raw bytes (for decode): the byte-level chars mapped back to bytes
        max_id = max(max(self.vocab.values()), max((x["id"] for x in self.added), default=0))
        self.id_to_bytes = [b""] * (max_id + 1)
        for tok, i in self.vocab.items():
            self.id_to_bytes[i] = bytes(BYTE_DECODER[c] for c in tok)
        for x in self.added:
            self.id_to_bytes[x["id"]] = x["content"].encode("utf-8")

    # --- pre-tokenization: the fixed pattern, hand-scanned deterministically -----
    def _pretokenize(self, text):
        # table-driven classification (identical to what the C++ TokenizerView uses)
        is_letter = lambda cp: self.u.is_letter(cp)
        is_number = lambda cp: self.u.is_number(cp)
        is_space = lambda ch: self.u.is_space(ord(ch))
        pieces = []
        i, n = 0, len(text)
        contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        while i < n:
            ch = text[i]
            # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
            matched = None
            if ch == "'":
                low = text[i:i + 3].lower()
                for c in contractions:
                    if low.startswith(c):
                        matched = text[i:i + len(c)]
                        break
            if matched is not None:
                pieces.append(matched); i += len(matched); continue
            # 2. [^\r\n\p{L}\p{N}]? \p{L}+
            j = i
            if ch not in "\r\n" and not is_letter(ord(ch)) and not is_number(ord(ch)):
                if j + 1 < n and is_letter(ord(text[j + 1])):
                    j += 1
            if j < n and is_letter(ord(text[j])):
                k = j
                while k < n and is_letter(ord(text[k])):
                    k += 1
                pieces.append(text[i:k]); i = k; continue
            # 3. \p{N}  (a single number char)
            if is_number(ord(ch)):
                pieces.append(ch); i += 1; continue
            # 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
            j = i
            if ch == " ":
                j = i + 1
            if j < n and not is_space(text[j]) and not is_letter(ord(text[j])) and not is_number(ord(text[j])):
                k = j
                while k < n and not is_space(text[k]) and not is_letter(ord(text[k])) and not is_number(ord(text[k])):
                    k += 1
                while k < n and text[k] in "\r\n":
                    k += 1
                pieces.append(text[i:k]); i = k; continue
            # 5. \s*[\r\n]+   6. \s+(?!\S)   7. \s+
            if is_space(ch):
                k = i
                while k < n and is_space(text[k]):
                    k += 1
                # a run of whitespace ending in CR/LF collapses to rule 5's span; a run
                # not at end-of-text gives back its last char to the following token
                # (the (?!\S) / greedy-\s+ split HF makes). Reproduce that split:
                has_nl = any(text[t] in "\r\n" for t in range(i, k))
                if has_nl:
                    last_nl = max(t for t in range(i, k) if text[t] in "\r\n")
                    pieces.append(text[i:last_nl + 1]); i = last_nl + 1; continue
                if k < n:  # \s+(?!\S) fails (a non-space follows) -> \s+ but leave the last space
                    if k - i > 1:
                        pieces.append(text[i:k - 1]); i = k - 1; continue
                    else:
                        pieces.append(text[i:k]); i = k; continue
                pieces.append(text[i:k]); i = k; continue
            # fallback: consume one char (should not happen for valid input)
            pieces.append(ch); i += 1
        return pieces

    def _bpe(self, piece_bytes):
        ids = [self.byte_to_id[b] for b in piece_bytes]
        if len(ids) < 2:
            return ids
        while True:
            best_rank, best_pos = None, None
            for p in range(len(ids) - 1):
                r = self._rank.get((ids[p], ids[p + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, p
            if best_pos is None:
                break
            merged = self._merged[(ids[best_pos], ids[best_pos + 1])]
            ids[best_pos:best_pos + 2] = [merged]
        return ids

    def ref_encode(self, text):
        """Reference encode: the exact algorithm the C++ TokenizerView reproduces."""
        out = []
        # special tokens split the RAW text first (added_tokens are normalized=false)
        spans = self._split_specials(text)
        for is_special, s in spans:
            if is_special:
                out.append(self.special_ids[s])
                continue
            norm = self.u.nfc(s)
            for piece in self._pretokenize(norm):
                out.extend(self._bpe(piece.encode("utf-8")))
        return out

    def _split_specials(self, text):
        spans, i, n = [], 0, len(text)
        while i < n:
            hit = None
            for content, _id in self.specials:
                if content and text.startswith(content, i):
                    hit = content
                    break
            if hit is not None:
                spans.append((True, hit)); i += len(hit)
            else:
                j = i
                while j < n:
                    if any(text.startswith(c, j) for c, _ in self.specials if c):
                        break
                    j += 1
                spans.append((False, text[i:j])); i = j
        return spans

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")

    # --- serialization: the .sslm TOKENIZER blob (parsed by the C++ TokenizerView) ---
    def serialize_tokenizer(self):
        b = bytearray()
        b += b"TOK1"
        vocab_count = len(self.id_to_bytes)
        b += struct.pack("<IIIII", 1, vocab_count, len(self.merge_triples), len(self.specials), 0)
        for i in range(256):
            b += struct.pack("<I", self.byte_to_id[i])
        # id -> raw bytes: offset table then blob
        offs, blob = [0], bytearray()
        for by in self.id_to_bytes:
            blob += by; offs.append(len(blob))
        for o in offs:
            b += struct.pack("<I", o)
        b += struct.pack("<I", len(blob)); b += blob
        # merges (rank order)
        for a, bb, m in self.merge_triples:
            b += struct.pack("<III", a, bb, m)
        # specials (longest-content-first, for greedy encode matching)
        s_offs, s_blob = [0], bytearray()
        for content, _id in self.specials:
            s_blob += content.encode("utf-8"); s_offs.append(len(s_blob))
        for _content, _id in self.specials:
            b += struct.pack("<I", _id)
        for o in s_offs:
            b += struct.pack("<I", o)
        b += struct.pack("<I", len(s_blob)); b += s_blob
        return bytes(b)

    def emit_artifact(self, out_path):
        config = {"model": self.model_name, "tokenizer": "byte-bpe",
                  "unicode_version": self.u.version, "pretok": "qwen-gpt-v1"}
        sections = [
            F.Section(F.SectionType.CONFIG, json.dumps(config, sort_keys=True).encode("utf-8")),
            F.Section(F.SectionType.TOKENIZER, self.serialize_tokenizer()),
            F.Section(F.SectionType.UNICODE_TABLES, self.u.serialize()),
            F.Section(F.SectionType.CHAT_TEMPLATE,
                      json.dumps({"chat_template": self.chat_template}, sort_keys=True).encode("utf-8")),
            # S-HARDEN-1 (F1): this converter is coupled to the same missing-required-
            # section defect as convert_model.py — the artifact format has ONE
            # required-section schema per format_version (docs/sslm_format.md), and
            # SigmoidLut is required from v2 regardless of which converter emitted
            # the bytes. Without this, the C++ loader's version-indexed schema check
            # rejects every tokenizer-only artifact this converter produces.
            F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()),
        ]
        return F.write_artifact(out_path, sections)

    def emit_golden(self, corpus_path, out_path):
        """Golden reference pack: the upstream HF ids for a corpus + a hash. The C++
        tokenizer must reproduce every record's ids and the same hash (§10 gate)."""
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(self.ckpt)
        lines = read_corpus_records(corpus_path)
        records, h = [], hashlib.sha256()
        for text in lines:
            ids = hf.encode(text, add_special_tokens=False)
            records.append({"text": text, "ids": ids})
            h.update(text.encode("utf-8")); h.update(b"\x00")
            for i in ids:
                h.update(struct.pack("<I", i))
        golden = {"unicode_version": self.u.version, "count": len(records),
                  "ids_hash": h.hexdigest(), "records": records}
        Path(out_path).write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
        # Binary golden for the C++ gate (no JSON parser in the std-lib-only tests):
        #   'GLD1', u32 record_count, u8[32] ids_hash,
        #   per record: u32 text_len, text utf-8, u32 id_count, i32[id_count] ids
        bin_path = Path(out_path).with_suffix(".gld")
        b = bytearray(b"GLD1")
        b += struct.pack("<I", len(records))
        b += bytes.fromhex(golden["ids_hash"])
        for r in records:
            t = r["text"].encode("utf-8")
            b += struct.pack("<I", len(t)) + t
            b += struct.pack("<I", len(r["ids"]))
            for i in r["ids"]:
                b += struct.pack("<i", i)
        bin_path.write_bytes(bytes(b))
        return golden["ids_hash"], len(records)


# --- Unicode table generation (pinned version) ----------------------------------
def gen_property_ranges(pred):
    """Contiguous [lo,hi] codepoint ranges where pred(cp) holds, over U+0000..U+10FFFF."""
    ranges, start = [], None
    for cp in range(0x110000):
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            ranges.append((start, cp - 1)); start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    return ranges


def read_corpus_records(corpus_path):
    """Reads a golden/verify corpus: ONE JSON STRING LITERAL PER PHYSICAL LINE
    (JSON Lines), e.g. `"Hello, world!"` or `"line one\\nline two"`. This is the
    corpus format from S-HARDEN-2 onward (Claude/Plans/SuperSLM_Plan.md
    S-HARDEN-2 slot): the prior plain-text-one-record-per-line format
    (str.splitlines(), discarding empty lines) could not represent an embedded
    \\n/\\r\\n WITHIN one record, or a genuinely empty-string record (a blank line
    was indistinguishable from "no record here" and was silently dropped) --
    both are named coverage classes the JSON-string encoding represents exactly,
    using the same escaping every JSON tool already implements correctly.
    A physical line that is empty (a blank line IN THE FILE, not a record whose
    decoded value is empty) is still skipped -- that is source-file formatting,
    not a corpus record; an intentional empty-string record is written as `""`.
    """
    records = []
    for lineno, raw_line in enumerate(Path(corpus_path).read_text(encoding="utf-8").split("\n"), start=1):
        line = raw_line.rstrip("\r")
        if not line:
            continue
        try:
            text = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{corpus_path}:{lineno}: not a valid JSON string literal ({e}): {line!r}") from e
        if not isinstance(text, str):
            raise ValueError(f"{corpus_path}:{lineno}: JSON value is not a string: {line!r}")
        records.append(text)
    return records


def verify(ckpt_dir, corpus_path, limit=None):
    from transformers import AutoTokenizer
    tables = TokenizerTables(ckpt_dir)
    hf = AutoTokenizer.from_pretrained(ckpt_dir)
    lines = read_corpus_records(corpus_path)
    if limit:
        lines = lines[:limit]
    mism = 0
    for ln, text in enumerate(lines):
        got = tables.ref_encode(text)
        want = hf.encode(text, add_special_tokens=False)
        if got != want:
            mism += 1
            if mism <= 8:
                print(f"MISMATCH line {ln}: {text!r}\n  ref : {got}\n  hf  : {want}")
    print(f"\n{len(lines)} lines, {mism} mismatches, Unicode {tables.unicode_version}")
    return mism


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir with tokenizer.json")
    ap.add_argument("--verify", help="corpus file to check ref_encode vs HF")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--emit", help="output .sslm path (tokenizer + unicode + chat sections)")
    ap.add_argument("--golden", nargs=2, metavar=("CORPUS", "OUT_JSON"),
                    help="emit the golden reference pack for a corpus")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.verify:
        sys.exit(1 if verify(args.ckpt, args.verify, args.limit) else 0)
    if args.emit or args.golden:
        tables = TokenizerTables(args.ckpt)
        if args.emit:
            fp = tables.emit_artifact(args.emit)
            print(f"wrote {args.emit}  fingerprint {fp}")
        if args.golden:
            ids_hash, n = tables.emit_golden(args.golden[0], args.golden[1])
            print(f"wrote {args.golden[1]}  {n} records  ids_hash {ids_hash}")
        sys.exit(0)
    print("use --verify <corpus>, --emit <out.sslm>, or --golden <corpus> <out.json>")
