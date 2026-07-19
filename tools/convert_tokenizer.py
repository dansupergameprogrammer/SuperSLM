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
import struct
import sys
import unicodedata
from pathlib import Path

import sslm_format as F

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


# --- Unicode classification, from the pinned unicodedata version ----------------
def is_letter(cp):
    return unicodedata.category(chr(cp))[0] == "L"


def is_number(cp):
    return unicodedata.category(chr(cp))[0] == "N"


def is_space(ch):
    # Python's \s over Unicode: whitespace per the Unicode property, matching the
    # `regex`/onig \s the reference tokenizer uses.
    return ch.isspace() or ch in "\x1c\x1d\x1e\x1f\x85"


# --- The tokenizer tables extracted from an HF checkpoint -----------------------
class TokenizerTables:
    def __init__(self, ckpt_dir):
        tj = json.loads((Path(ckpt_dir) / "tokenizer.json").read_text(encoding="utf-8"))
        cfg = json.loads((Path(ckpt_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
        model = tj["model"]
        assert model["type"] == "BPE", model["type"]
        assert tj.get("normalizer", {}).get("type") == "NFC", tj.get("normalizer")

        self.vocab = model["vocab"]                       # byte-level-string -> id
        self.id_to_tok = {v: k for k, v in self.vocab.items()}
        self.merges = [m.split(" ") for m in model["merges"]]  # [ [a,b], ... ] rank order
        self.added = tj.get("added_tokens", [])
        self.chat_template = cfg.get("chat_template")
        self.unicode_version = unicodedata.unidata_version

        # base byte -> id
        self.byte_to_id = [self.vocab[BYTE_ENCODER[b]] for b in range(256)]
        # rank-ordered merge triples (id_a, id_b, id_merged)
        self.merge_triples = []
        for a, b in self.merges:
            self.merge_triples.append((self.vocab[a], self.vocab[b], self.vocab[a + b]))
        # merge rank lookup for the reference encoder
        self._rank = {(a, b): i for i, (a, b, _) in enumerate(self.merge_triples)}
        self._merged = {(a, b): m for a, b, m in self.merge_triples}
        # special tokens, longest-content first so overlapping specials match greedily
        self.specials = sorted(((x["content"], x["id"]) for x in self.added),
                               key=lambda kv: len(kv[0]), reverse=True)
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
            norm = unicodedata.normalize("NFC", s)
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


def verify(ckpt_dir, corpus_path, limit=None):
    from transformers import AutoTokenizer
    tables = TokenizerTables(ckpt_dir)
    hf = AutoTokenizer.from_pretrained(ckpt_dir)
    lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    mism = 0
    for ln, text in enumerate(lines):
        if not text:
            continue
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
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.verify:
        sys.exit(1 if verify(args.ckpt, args.verify, args.limit) else 0)
    print("use --verify <corpus> for now; artifact emission lands next")
