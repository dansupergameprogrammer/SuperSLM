"""Independent must-reject for the SCM1 writer's byte-equality check.

Emulate a serializer that emits the two names in sorted order while retaining
the input order for descriptors and data blocks. Both names have equal UTF-8
length, so the result remains a structurally valid SCM1 section but binds each
compiled schema to the other name. Exit 1 only for the intended byte mismatch;
an invalid construction or setup error exits 0 and cannot commission the writer.
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POPPER = HERE.parent / "t3030-popper"
TOOLS = HERE.parents[1] / "tools"
sys.path.insert(0, str(POPPER))
sys.path.insert(0, str(TOOLS))

import common  # noqa: E402
import scm1_writer  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402


def check(ckpt: str, vocab_size: int) -> bool:
    vocab, specials = common.real_vocab(ckpt, vocab_size)
    pages = common.compile_all([common.SCHEMA_A, common.SCHEMA_B], vocab, specials)
    # Reverse lexical order makes a name-only sort change the emitted bytes.
    names = ("t3043_bravo", "t3043_alpha")
    encoded = [name.encode("utf-8") for name in names]
    if len(encoded[0]) != len(encoded[1]):
        raise ValueError("names must have equal UTF-8 lengths")
    hashes = [common.fnv1a64(name) for name in encoded]
    if 0 in hashes or len(set(hashes)) != 2:
        raise ValueError("names must have distinct nonzero hashes")

    named = list(zip(names, pages))
    reference = _serialize_scm1(named, vocab_size)
    healthy = scm1_writer.write_scm1(named, vocab_size)
    if healthy != reference:
        raise ValueError("healthy writer differs from production serializer")

    magic, version, count, wire_vocab, blob_len, reserved = struct.unpack_from("<4sIIIII", healthy)
    if (magic, version, count, wire_vocab, blob_len, reserved) != (
        b"SCM1", 1, 2, vocab_size, sum(map(len, encoded)), 0
    ):
        raise ValueError("unexpected SCM1 header")
    blob_start = 24 + count * 56
    blob_end = blob_start + blob_len
    if healthy[blob_start:blob_end] != b"".join(encoded):
        raise ValueError("unexpected SCM1 name blob")

    mutant = (
        healthy[:blob_start] + encoded[1] + encoded[0] + healthy[blob_end:]
    )
    if mutant[:blob_start] != reference[:blob_start] or mutant[blob_end:] != reference[blob_end:]:
        raise ValueError("mutation changed bytes outside the name blob")
    if mutant[blob_start:blob_end] != b"".join(sorted(encoded)):
        raise ValueError("name blob did not take the sorted-name variant")

    equal = mutant == reference
    print(f"names={names[0]},{names[1]} noncolliding=True")
    print(f"bytes={len(mutant)} sha256_variant={hashlib.sha256(mutant).hexdigest()}")
    print(f"sha256_serialize_scm1={hashlib.sha256(reference).hexdigest()}")
    print(f"byte_equal={equal}")
    print("BYTE_EQUALITY_REFUSED" if not equal else "BYTE_EQUALITY_ACCEPTED")
    return not equal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab-size", type=int, default=151936)
    args = ap.parse_args()
    try:
        return 1 if check(args.ckpt, args.vocab_size) else 0
    except Exception as exc:
        print(f"INVALID_CONSTRUCTION: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
