"""Must-accept half of scm1_writer's commissioning: byte equality with the production serializer.

On two non-colliding lists (the renamed pair; the three-schema list of cell 6.7) the writer's
bytes must equal `_serialize_scm1`'s byte for byte. Exit 0 only when both lists agree.

The must-reject half (plan Sec8: one descriptor field offset by one byte must break the
equality) is NOT authored here: the conductor routes it to the test author, who is independent
of this writer (T-3030, 2026-10-01). The writer's commissioning is pending on that construction.

Usage: python commission_scm1_writer.py --ckpt <hf-ckpt-dir> [--vocab-size N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import scm1_writer  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402  (comparison only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab-size", type=int, default=151936)
    a = ap.parse_args()

    vocab, specials = common.real_vocab(a.ckpt, a.vocab_size)
    mps = common.compile_all([common.SCHEMA_A, common.SCHEMA_B, common.SCHEMA_C], vocab, specials)
    lists = {
        "renamed-pair": [(common.COLLIDING_A, mps[0]), (common.RENAMED_B, mps[1])],
        "three-schema": list(zip(common.THREE_NAMES, mps)),
    }
    ok = True
    for label, named in lists.items():
        hashes = [common.fnv1a64(n.encode()) for n, _ in named]
        assert len(set(hashes)) == len(hashes) and 0 not in hashes, "list must be non-colliding"
        ours = scm1_writer.write_scm1(named, a.vocab_size)
        theirs = _serialize_scm1(named, a.vocab_size)
        equal = ours == theirs
        ok &= equal
        print(f"must-accept {label}: bytes={len(ours)} equal={equal} "
              f"sha256_writer={common.sha256_hex(ours)} sha256_serialize_scm1={common.sha256_hex(theirs)}")
    print("WRITER must-accept", "PASSED" if ok else "FAILED", "(must-reject pending: test author)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
