"""The two S2 SCM1 mutations, applied to one named schema of a serialized SchemaMasks section.

- accept_shift: moves the last accepting-state id by one, in the direction that keeps the
  accepting array in range and strictly ascending (structurally valid, semantically wrong).
- offset_swap: swaps the first two unequal adjacent CSR state offsets (non-monotonic
  `state_offsets`, which the runtime loaders and the verifier must refuse).

Used by make_mutants.py (patched attach copies for the box) and verify_schema_masks_check.py.
"""
from __future__ import annotations

import struct

MODES = ("accept_shift", "offset_swap")


def describe(scm1: bytes, name: bytes) -> tuple[int, int, int, int, int, int, int]:
    """(state_count, accepting_count, transition_count, mask_pages_off, state_offsets_off,
    accepting_off, transitions_off) of schema `name`."""
    count, = struct.unpack_from("<I", scm1, 8)
    for k in range(count):
        base = 24 + 56 * k
        name_off, name_len, states, acc, trans = struct.unpack_from("<5I", scm1, base)
        name_start = 24 + 56 * count + name_off
        if scm1[name_start:name_start + name_len] == name:
            mp, so, ao, to = struct.unpack_from("<4Q", scm1, base + 24)
            return states, acc, trans, mp, so, ao, to
    raise SystemExit(f"schema {name!r} not found in the SCM1 section")


def mutate(scm1: bytes, mode: str, name: bytes = b"intents") -> tuple[bytes, str]:
    """The mutated section and a one-line description of the mutation."""
    b = bytearray(scm1)
    states, acc, _trans, _mp, so, ao, _to = describe(scm1, name)
    if mode == "accept_shift":
        ids = list(struct.unpack_from(f"<{acc}I", b, ao))
        k = len(ids) - 1
        old = ids[k]
        new = old - 1 if old + 1 >= states or (k + 1 < len(ids) and ids[k + 1] == old + 1) else old + 1
        if (k > 0 and ids[k - 1] >= new) or new in ids:
            raise SystemExit("no structurally clean shift")
        struct.pack_into("<I", b, ao + 4 * k, new)
        return bytes(b), f"accept_shift: accepting[{k}] {old} -> {new} (state_count {states}, accepting {ids})"
    if mode == "offset_swap":
        offs = list(struct.unpack_from(f"<{states + 1}I", b, so))
        i = next(i for i in range(1, states) if offs[i] != offs[i + 1])
        struct.pack_into("<I", b, so + 4 * i, offs[i + 1])
        struct.pack_into("<I", b, so + 4 * (i + 1), offs[i])
        return bytes(b), f"offset_swap: state_offsets[{i}]={offs[i]} <-> state_offsets[{i + 1}]={offs[i + 1]}"
    raise ValueError(f"unknown mutant {mode!r}")


def _format():
    import sys
    from pathlib import Path
    tools = Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import sslm_format as SF
    return SF


def write_mutant_artifact(src, mode: str, out, name: bytes = b"intents") -> str:
    """Writes `src` with schema `name`'s SCM1 bytes mutated by `mode` (every other section kept),
    returning the mutation's description. `mode == "clean"` rewrites the section unchanged."""
    from dataclasses import replace
    SF = _format()
    header = SF.read_header(str(src))
    sections = []
    what = "clean: SCM1 unchanged"
    for s in header["sections"]:
        data = bytes(header["data"][s["offset"]:s["offset"] + s["byte_size"]])
        sec = SF.Section(type=s["type"], data=data, dtype=s["dtype"], elem_count=s["elem_count"],
                         alignment=s["alignment"])
        if s["type"] == SF.SectionType.SCHEMA_MASKS and mode != "clean":
            mutated, what = mutate(data, mode, name)
            sec = replace(sec, data=mutated)
        sections.append(sec)
    SF.write_artifact(str(out), sections, header["flags"])
    return what


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4 or sys.argv[2] not in MODES + ("clean",):
        sys.exit("usage: scm1_mutants.py <in.sslm> <accept_shift|offset_swap|clean> <out.sslm>")
    print("MUTANT " + write_mutant_artifact(sys.argv[1], sys.argv[2], sys.argv[3]))
