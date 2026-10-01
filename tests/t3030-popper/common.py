"""Shared inputs for the T-3030 commissioning scripts: real vocab, local schemas, hashing."""
import hashlib
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import sslm_convert_schema as SC  # noqa: E402
from convert_tokenizer import TokenizerTables  # noqa: E402

# The strike's colliding pair (plan S25, recomputed here: both Fnv1a64 == 0x25da8c1836a8d66d).
COLLIDING_A = "P1z16@?^_s(Q"
COLLIDING_B = "abcdefgh"
RENAMED_B = "abcdefgi"
THREE_NAMES = ["t3030_alpha", "t3030_beta", "t3030_gamma"]

SCHEMA_A = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
SCHEMA_B = {"type": "object", "properties": {"kind": {"enum": ["a", "b", "c"]},
                                             "flag": {"type": "boolean"}}}
SCHEMA_C = {"type": "object", "properties": {"mode": {"enum": ["fast", "slow"]}}}


def fnv1a64(b: bytes) -> int:
    h = 14695981039346656037
    for x in b:
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def real_vocab(ckpt_dir: str, vocab_size: int):
    t = TokenizerTables(ckpt_dir)
    pieces = [t.id_to_bytes[i] if i < len(t.id_to_bytes) else b"" for i in range(vocab_size)]
    specials = frozenset(token_id for _, token_id in t.specials)
    return pieces, specials


def compile_all(schemas, vocab, specials):
    return [SC.compile_schema_to_mask_pages(s, vocab, special_ids=specials) for s in schemas]
