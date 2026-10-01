"""Test-only SCM1 section writer (precached-prefixes plan PC-12, D-SLM7985 / D-SLM7990).

Packs the SchemaMasks wire format as the engine reads it at 5d5ca58
(include/superslm/schema_masks.h, SchemaMasksTable::Parse): a 24-byte header, 56-byte
descriptors, the name blob, then per schema four contiguous blocks (mask pages, CSR row starts,
sorted accepting set, per-row ascending transitions). Performs no name-hash check, which is why
it exists. Never imports the production serializer; it is commissioned by byte equality with it
on non-colliding name lists (commission_scm1_writer.py).
"""
import struct

HEADER_BYTES = 24
DESCRIPTOR_BYTES = 56


def _dense_state_count(mp) -> int:
    states = set(mp.transitions.keys()) | {mp.start}
    for row in mp.transitions.values():
        states.update(row.values())
    count = (max(states) + 1) if states else 1
    if sorted(states) != list(range(count)):
        raise ValueError("MaskPages state ids are not dense [0, N)")
    return count


def _pack_descriptor(name_off, name_len, state_count, accepting_count, transition_count,
                     mask_pages_off, state_offsets_off, accepting_off, transitions_off) -> bytes:
    return struct.pack("<IIIIIIQQQQ", name_off, name_len, state_count, accepting_count,
                       transition_count, 0, mask_pages_off, state_offsets_off, accepting_off,
                       transitions_off)


def write_scm1(named_schemas, vocab_size: int) -> bytes:
    """SCM1 bytes for [(name, MaskPages), ...] over vocab_size. No hash check of any kind."""
    named = list(named_schemas)
    encoded = [name.encode("utf-8") for name, _ in named]
    name_blob = b"".join(encoded)
    manifest_end = HEADER_BYTES + len(named) * DESCRIPTOR_BYTES + len(name_blob)
    page_bytes = (vocab_size + 7) // 8

    descriptors = []
    blocks = bytearray()
    cursor = manifest_end
    name_off = 0
    for (name, mp), name_bytes in zip(named, encoded):
        state_count = _dense_state_count(mp)

        mask_pages_off = cursor
        for s in range(state_count):
            page = mp.page(s)
            if len(page) != page_bytes:
                raise ValueError(f"schema {name!r}: page {s} is {len(page)} bytes, want {page_bytes}")
            blocks += page
        cursor += state_count * page_bytes

        row_starts = [0]
        transitions = bytearray()
        for s in range(state_count):
            row = mp.transitions.get(s, {})
            for tok in sorted(row):
                transitions += struct.pack("<II", tok, row[tok])
            row_starts.append(row_starts[-1] + len(row))
        state_offsets_off = cursor
        for v in row_starts:
            blocks += struct.pack("<I", v)
        cursor += 4 * len(row_starts)

        accepting_off = cursor
        accepting = sorted(mp.accepting)
        for v in accepting:
            blocks += struct.pack("<I", v)
        cursor += 4 * len(accepting)

        transitions_off = cursor
        blocks += transitions
        cursor += len(transitions)

        descriptors.append(_pack_descriptor(
            name_off, len(name_bytes), state_count, len(accepting), row_starts[-1],
            mask_pages_off, state_offsets_off, accepting_off, transitions_off))
        name_off += len(name_bytes)

    header = struct.pack("<4sIIIII", b"SCM1", 1, len(named), vocab_size, len(name_blob), 0)
    return header + b"".join(descriptors) + name_blob + bytes(blocks)
