"""t2132_build_g5_fixture.py -- G5-2 (T-2132, Brunel): builds a real G5 fixture .sslm carrying
a SchemaMasks (SCM1) section, from an already-converted real artifact plus its own HF checkpoint
(for the real tokenizer vocab the schema compiler needs).

Not a new converter: reads every existing section out of a real, already-shipped `.sslm`
(`sslm_format.read_header`) UNCHANGED, compiles one or more real schemas against the real
tokenizer vocabulary (`tools/sslm_convert_schema.py`, `tools/convert_tokenizer.py`'s own
`TokenizerTables.id_to_bytes`), serializes them into one `SCM1` section per the design's own
byte format (Claude/Vitruvius/t2119-g5-constrained-decoding-design-2026-08-16.md Sec13, Wizard
repo), and writes ONE new `.sslm` carrying the original sections plus the new one -- the same
`sslm_format.write_artifact` every other converter tool already calls.

Usage: python t2132_build_g5_fixture.py --base <existing.sslm> --ckpt <hf-ckpt-dir> --out <out.sslm>
"""
import argparse
import struct
import sys

import sslm_format as F
import sslm_convert_schema as SC
from convert_tokenizer import TokenizerTables

# The reference task's own schema (Claude/Docs/Shopkeeper_IntentExtraction_Schema.md Sec3,
# Wizard repo) -- reproduced field-for-field (the JSON block is the decode-time-guaranteed half;
# the cross-field invariants in that doc's prose are deliberately NOT compiled, D-SLM45).
def _enum(values):
    return {"enum": list(values)}


SHOPKEEPER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": _enum(["book", "reschedule", "cancel", "price_query", "availability_query",
                          "design_query", "complaint", "out_of_domain"]),
        "slots": {
            "type": "object",
            "properties": {
                "artist": _enum(["unspecified", "rin", "mox", "vale", "any"]),
                "day": _enum(["unspecified", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]),
                "time_block": _enum(["unspecified", "morning", "afternoon", "evening"]),
                "size": _enum(["unspecified", "small", "medium", "large", "sleeve"]),
                "placement": _enum(["unspecified", "arm", "leg", "back", "chest", "hand", "neck"]),
                "style": _enum(["unspecified", "blackwork", "traditional", "realism", "script",
                                 "geometric"]),
                "budget_band": _enum(["unspecified", "under_200", "200_500", "500_1000",
                                        "over_1000"]),
                "deposit_ok": _enum(["unspecified", "yes", "no"]),
            },
        },
        "polarity": _enum(["affirm", "negate"]),
        "negated_slot": _enum(["none", "artist", "day", "time_block", "size", "placement",
                                 "style", "budget_band"]),
        "correction": {"type": "boolean"},
    },
}

# A trivial one-field schema -- the suite's own default secondary-schema name
# (fixture_common.h: g_secondary_schema_name == "g5_minimal_one_field").
MINIMAL_ONE_FIELD_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
}


def _real_vocab(ckpt_dir: str, vocab_size: int) -> list[str]:
    """The real tokenizer's own per-token text, index == token id, padded/truncated to
    `vocab_size` (CFG1's own declared width -- may exceed the tokenizer's own base+added vocab
    count, the padded-vocabulary case this codebase already names, SSLM_TOKEN_ID_UNMAPPED). A
    byte that fails to decode as UTF-8 on its own (a multi-byte BPE piece split across a
    non-boundary) decodes with 'replace' -- imprecise, but the compiler only needs vocab pieces
    to walk a DFA over ASCII structural/literal characters (every one of which IS a real,
    correctly-decoding single-byte token in any byte-level BPE vocabulary by construction), so
    imprecision on genuinely multi-byte pieces costs nothing this schema's own literal text ever
    touches."""
    t = TokenizerTables(ckpt_dir)
    pieces = []
    for i in range(vocab_size):
        raw = t.id_to_bytes[i] if i < len(t.id_to_bytes) else b""
        pieces.append(raw.decode("utf-8", errors="replace"))
    return pieces


def _read_config_vocab_size(base_path: str) -> int:
    data = F.read_section_bytes(base_path, F.SectionType.CONFIG)
    if data is None:
        raise SystemExit(f"{base_path}: no CONFIG section")
    # CFG1: magic(4) + version(4) + ... vocab_size at payload offset 32 (docs/sslm_format.md).
    (vocab_size,) = struct.unpack_from("<I", data, 32)
    return vocab_size


def _serialize_scm1(named_schemas: list[tuple[str, "SC.MaskPages"]], vocab_size: int) -> bytes:
    """Byte format per design Sec13.2 -- 24-byte fixed header, schema_count x 56-byte
    descriptors, name blob, then four per-schema data blocks (mask pages, CSR state-offset
    array, accepting-state array, flattened transition array), reproducing MaskPages' own
    construction verbatim as the wire format (no repacking)."""
    schema_count = len(named_schemas)
    name_blob = b"".join(name.encode("utf-8") for name, _ in named_schemas)
    name_offsets = []
    off = 0
    for name, _ in named_schemas:
        name_offsets.append((off, len(name.encode("utf-8"))))
        off += len(name.encode("utf-8"))

    manifest_end = 24 + schema_count * 56 + len(name_blob)
    cursor = manifest_end
    descriptors = []
    blocks = bytearray()
    for (name, mp), (name_off, name_len) in zip(named_schemas, name_offsets):
        states = sorted(set(mp.transitions.keys()) | {mp.start} |
                         set(t for row in mp.transitions.values() for t in row.values()))
        # Every state id [0, state_count) must be present and contiguous, by MaskPages' own
        # BFS-order construction (compile_schema_to_mask_pages assigns ids 0..N-1 densely) --
        # asserted here rather than trusted, since this tool is the one place that could drift.
        state_count = max(states) + 1 if states else 1
        assert states == list(range(state_count)), "MaskPages state ids are not dense [0,N)"

        mask_page_bytes = (vocab_size + 7) // 8
        mask_pages_off = cursor
        for s in range(state_count):
            blocks += mp.page(s)
        cursor += state_count * mask_page_bytes

        state_offsets_off = cursor
        row_starts = [0]
        flat_transitions = bytearray()
        for s in range(state_count):
            row = mp.transitions.get(s, {})
            for tok in sorted(row):
                flat_transitions += struct.pack("<II", tok, row[tok])
            row_starts.append(row_starts[-1] + len(row))
        for v in row_starts:
            blocks += struct.pack("<I", v)
        cursor += len(row_starts) * 4

        accepting_off = cursor
        accepting_sorted = sorted(mp.accepting)
        for v in accepting_sorted:
            blocks += struct.pack("<I", v)
        cursor += len(accepting_sorted) * 4

        transitions_off = cursor
        blocks += flat_transitions
        cursor += len(flat_transitions)

        descriptors.append(struct.pack(
            "<IIIIIIQQQQ",
            name_off, name_len, state_count, len(accepting_sorted), row_starts[-1], 0,
            mask_pages_off, state_offsets_off, accepting_off, transitions_off))

    header = struct.pack("<4sIIIII", b"SCM1", 1, schema_count, vocab_size, len(name_blob), 0)
    return header + b"".join(descriptors) + name_blob + bytes(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing real .sslm to carry every section from")
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir (tokenizer.json) matching --base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    header = F.read_header(args.base)
    vocab_size = _read_config_vocab_size(args.base)
    vocab = _real_vocab(args.ckpt, vocab_size)

    print(f"compiling shopkeeper_intent_extraction against {vocab_size}-token real vocab...")
    mp_shopkeeper = SC.compile_schema_to_mask_pages(SHOPKEEPER_SCHEMA, vocab)
    n_states = len(set(mp_shopkeeper.transitions.keys()) |
                    set(t for row in mp_shopkeeper.transitions.values() for t in row.values()) |
                    {0})
    print(f"  compiled: {n_states} states")

    print("compiling g5_minimal_one_field...")
    mp_minimal = SC.compile_schema_to_mask_pages(MINIMAL_ONE_FIELD_SCHEMA, vocab)
    n_states2 = len(set(mp_minimal.transitions.keys()) |
                     set(t for row in mp_minimal.transitions.values() for t in row.values()) |
                     {0})
    print(f"  compiled: {n_states2} states")

    scm1 = _serialize_scm1(
        [("shopkeeper_intent_extraction", mp_shopkeeper), ("g5_minimal_one_field", mp_minimal)],
        vocab_size)

    sections = []
    for s in header["sections"]:
        data = header["data"][s["offset"]:s["offset"] + s["byte_size"]]
        sections.append(F.Section(type=s["type"], data=bytes(data), dtype=s["dtype"],
                                    elem_count=s["elem_count"], alignment=s["alignment"]))
    sections.append(F.Section(type=F.SectionType.SCHEMA_MASKS, data=scm1))

    # Preserve every recognized base feature bit. In particular, a schema-bearing derivative
    # of a 1.2 DGC-enabled model must remain DGC-enabled; clearing the bit leaves a DGC1 section
    # that the loader correctly rejects as inconsistent.
    fp = F.write_artifact(args.out, sections, flags=header["flags"])
    print(f"wrote {args.out}")
    print(f"fingerprint {fp}")
    print(f"sections {len(sections)}, SCM1 bytes {len(scm1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
