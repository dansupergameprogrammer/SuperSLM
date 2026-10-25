"""S2 mutant check: writes two scratch copies of tools/sslm_attach_schema.py, each patched to
corrupt the `intents` schema's SCM1 bytes right after `serialize_scm1`, so the verifier, the
walks and the generation of tests/array-s2 can be run against a bad artifact.

- accept_shift: moves the last accepting-state id by one, in the direction that keeps the
  accepting array in range and strictly ascending (structurally valid, semantically wrong).
- offset_swap: swaps the first two unequal adjacent CSR state offsets.

    python make_mutants.py <repo tools dir> <scratch dir>

Run each copy with the repo's tools directory on PYTHONPATH (the copy's own directory comes first,
so its patched module is the one imported).
"""
import shutil
import sys
from pathlib import Path

MUTATE = r'''

def _desc(scm1):
    import struct
    count, = struct.unpack_from("<I", scm1, 8)
    for k in range(count):
        base = 24 + 56 * k
        name_off, name_len, states, acc, trans = struct.unpack_from("<5I", scm1, base)
        name_start = 24 + 56 * count + name_off
        if scm1[name_start:name_start + name_len] == b"intents":
            mp, so, ao, to = struct.unpack_from("<4Q", scm1, base + 24)
            return states, acc, trans, mp, so, ao, to
    raise SystemExit("intents not found")


def _mutate(scm1):
    import struct
    b = bytearray(scm1)
    states, acc, trans, mp, so, ao, to = _desc(scm1)
    if MODE == "accept_shift":
        ids = list(struct.unpack_from(f"<{acc}I", b, ao))
        k = len(ids) - 1
        old = ids[k]
        new = old - 1 if old + 1 >= states or (k + 1 < len(ids) and ids[k + 1] == old + 1) else old + 1
        if (k > 0 and ids[k - 1] >= new) or new in ids:
            raise SystemExit("no structurally clean shift")
        struct.pack_into("<I", b, ao + 4 * k, new)
        print(f"MUTANT accept_shift: accepting[{k}] {old} -> {new} (state_count {states}, accepting {ids})")
    elif MODE == "offset_swap":
        offs = list(struct.unpack_from(f"<{states + 1}I", b, so))
        i = next(i for i in range(1, states) if offs[i] != offs[i + 1])
        struct.pack_into("<I", b, so + 4 * i, offs[i + 1])
        struct.pack_into("<I", b, so + 4 * (i + 1), offs[i])
        print(f"MUTANT offset_swap: state_offsets[{i}]={offs[i]} <-> state_offsets[{i + 1}]={offs[i + 1]}")
    return bytes(b)
'''


def main() -> int:
    tools, out = Path(sys.argv[1]), Path(sys.argv[2])
    for mode in ("accept_shift", "offset_swap"):
        d = out / mode
        d.mkdir(parents=True, exist_ok=True)
        for f in ("sslm_format.py", "sslm_convert_schema.py"):
            shutil.copy(tools / f, d / f)
        src = (tools / "sslm_attach_schema.py").read_text(encoding="utf-8")
        anchor = "    scm1 = SC.serialize_scm1(compiled, vocab_size)\n"
        assert src.count(anchor) == 1
        src = src.replace(anchor, anchor + "    scm1 = _mutate(scm1)\n")
        src = src.replace("\ndef main(", f"\nMODE = {mode!r}\n" + MUTATE + "\n\ndef main(", 1)
        (d / "sslm_attach_schema.py").write_text(src, encoding="utf-8")
        print("wrote", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
