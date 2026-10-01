"""Add three distinct, noncolliding SCM1 names to the pinned pkv_def synthetic model.

The model's other sections and flags are copied unchanged. All three schemas use
the pinned synthetic vocabulary and the same simple DFA; index 2 is reachable
through the engine's real schema lookup path for precache plan cell 6.7.
"""

import argparse
import hashlib

import sslm_convert_schema as SC
import sslm_format as F
from t2132_build_g5_fixture import _serialize_scm1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    args = ap.parse_args()
    source = F.read_header(args.base)
    # A two-state synthetic DFA: token 0 advances the walk from state 0;
    # state 1 admits all 256 tokens and remains at 1 for a full 64-token stream.
    # The parser validates the mask/transition equivalence on the real data path.
    vocab = [bytes([i]) for i in range(256)]
    masks = SC.MaskPages(vocab, {0: {0: 1}, 1: {i: 1 for i in range(256)}},
                         0, frozenset({1}))
    names = ("schema_zero", "schema_one", "schema_two")
    scm1 = _serialize_scm1([(name, masks) for name in names], 256)
    sections = []
    for entry in source["sections"]:
        data = (scm1 if entry["type"] == F.SectionType.SCHEMA_MASKS else
                source["data"][entry["offset"]:entry["offset"] + entry["byte_size"]])
        sections.append(F.Section(entry["type"], data, dtype=entry["dtype"],
                                  alignment=entry["alignment"]))
    F.write_artifact(args.out, sections, flags=source["flags"])
    with open(args.out, "rb") as stream:
        print(f"{args.out} sha256={hashlib.file_digest(stream, 'sha256').hexdigest()}")


if __name__ == "__main__":
    main()
