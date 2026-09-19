"""Builds fixture G5-an for cell_schema_guard_status.cpp (the GPU-path plan's Sec3.4 row 5, cell Q5-1:
the schema twin's device-guard refusal stays SSLM_DEVICE_LOST, D-SLM7311).

G5-an is the G5 suite's schema-bearing argv artifact, t2132_g5_fixture_1p5b.sslm (the --g5fixture of
this suite and of tests/t2130-g5-red-suite), with its CompositionConstants entry "layer0.attn_norm" set
to mantissa -2^28 (-268,435,456), its exponent unchanged, and the artifact's integrity SHA-256
recomputed over the whole file with the 32 hash bytes at offset 32 zeroed (src/artifact.cpp's integrity
check), exactly as make_gan_fixture.py builds G-an.

WHY THIS ARTIFACT AND THIS MANTISSA. Q5-1 needs a DFA-legal chain [pass, trip] from a schema's start
state. The plan's first candidate, the C39 synthetic (synthetic_te62_c39.sslm, 2ea564f7...4930), carries
one schema, g5_minimal_one_field, with 2 states and 1 transition: its longest legal chain is one token,
so no [pass, trip] chain exists on it at any mantissa. On this artifact, shopkeeper_intent_extraction
admits two ids at its start state (90 and 4913). At -2^31+1, -2^30 and -2^29 both trip layer 0; at -2^27
and above nothing that a chain uses trips; at -2^28, 4913 passes all 28 layers and the continuation 72
refuses at layer 0 with CarriedScaleMantissaOutOfDomain (Claude/Curie/t2814-probe/, q5_sweep).
cell_schema_guard_status.cpp re-derives the chain with its own CPU oracle before any GPU call.

Both input and output hashes are pinned: the script refuses any other source and refuses to leave an
output whose hash differs from the one the cell pins. The output is about 1.6 GB.

Usage: python make_g5an_fixture.py <t2132_g5_fixture_1p5b.sslm> <out_g5an.sslm>
"""
import hashlib
import struct
import sys

G5_SHA256 = "078df885060d5dea23a88983bb68014843d142cb6ad55c7f70ef9ff9a932a019"
# Reproduced independently by Claude/Curie/t2814-probe/q5_sweep.cpp, which patches the same entry in
# memory and re-seals through the engine's own superslm::Sha256Hash.
G5AN_SHA256_PINNED = "05b5d5c58ea14dbb15a3adb7f24668edcd4d7ab83344bc4381c5f7347b5a128b"
ENTRY = "layer0.attn_norm"
G5_LAYER0_ATTN_NORM = (1482694395, -58)
G5AN_M = -(1 << 28)
SECTION_COMPOSITION_CONSTANTS = 7


def sha256_of(b):
    h = hashlib.sha256()
    view = memoryview(b)
    for i in range(0, len(view), 1 << 24):
        h.update(view[i:i + (1 << 24)])
    return h.hexdigest()


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: make_g5an_fixture.py <t2132_g5_fixture_1p5b.sslm> <out_g5an.sslm>")
    src, dst = argv[1], argv[2]
    with open(src, "rb") as f:
        b = bytearray(f.read())
    got = sha256_of(b)
    if got != G5_SHA256:
        sys.exit("source is not the G5 fixture: sha256 %s, want %s" % (got, G5_SHA256))
    if b[:4] != b"SSLM":
        sys.exit("source is not an SSLM artifact")

    nsec = struct.unpack_from("<I", b, 12)[0]
    hit = None
    for i in range(nsec):
        t, _dt, off, size, _elems, _al, _rsv = struct.unpack_from("<IIQQQII", b, 64 + 40 * i)
        if t == SECTION_COMPOSITION_CONSTANTS:
            hit = (off, size)
    if hit is None:
        sys.exit("no CompositionConstants section")
    off, _size = hit
    magic, _ver, n, vw, _nbl, _rsv = struct.unpack_from("<4sIIIII", b, off)
    if magic != b"KVC1" or vw != 2:
        sys.exit("unexpected CompositionConstants layout")
    desc = off + 24
    vals = desc + 8 * n
    names = vals + 8 * n * vw
    patched = 0
    for i in range(n):
        no, nl = struct.unpack_from("<II", b, desc + 8 * i)
        name = bytes(b[names + no: names + no + nl]).decode()
        if name != ENTRY:
            continue
        m, e = struct.unpack_from("<qq", b, vals + 16 * i)
        if (m, e) != G5_LAYER0_ATTN_NORM:
            sys.exit("%s before patch is (%d, %d), want %r" % (ENTRY, m, e, G5_LAYER0_ATTN_NORM))
        struct.pack_into("<q", b, vals + 16 * i, G5AN_M)
        print("patched %s (%d, %d) -> (%d, %d)" % (ENTRY, m, e, G5AN_M, e))
        patched += 1
    if patched != 1:
        sys.exit("expected exactly one %s entry, patched %d" % (ENTRY, patched))

    b[32:64] = bytes(32)
    b[32:64] = hashlib.sha256(b).digest()
    out = sha256_of(b)
    if out != G5AN_SHA256_PINNED:
        sys.exit("G5-an sha256 %s, want %s" % (out, G5AN_SHA256_PINNED))
    with open(dst, "wb") as f:
        f.write(b)
    print("G5-an written: %s sha256 %s bytes %d" % (dst, out, len(b)))


if __name__ == "__main__":
    main(sys.argv)
