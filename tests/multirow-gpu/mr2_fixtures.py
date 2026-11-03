"""MR2 red suite: the refusal fixtures for cells 6.6 (guard at row j) and MX4/MX5/MX20/MX21.

Plan §7 6.6 wants F-sat-then-refuse: a row that saturates at layer 0's kv_proj (kv_sat and kv_landing rise) and
then refuses at the LAST layer's attn_residual (pkv_def has 2 layers, so layer 1), and F-qk, the same on pkv_qk
with the clamp at qk_norm. On the pkv fixtures every row already saturates at kv_proj (and qk_norm on pkv_qk):
mr2_fixture_probe.exe shows non-zero kv_sat/kv_landing deltas (and k_channel on pkv_qk) for every token. So the
fixture only has to add the refusal: the CompositionConstants entry "layer<L-1>.attn_residual" gets an extreme
mantissa, which makes the attn_residual site's guard refuse for some rows and not others (whether a row trips
depends on its residual), as make_gan_fixture.py does for layer 0's attn_norm in the T-2791 suite. The artifact's
integrity SHA-256 is recomputed over the whole file with its 32 hash bytes zeroed (src/artifact.cpp).

The cells do not trust this file for which rows trip: each 6.6 case finds its trap token at run time on the one-row
path (the first stream token that refuses at the last layer after the clean prefix) and pins nothing else.

Usage: python3 mr2_fixtures.py FIXTURE_DIR [--mantissa=M]
  writes FIXTURE_DIR/pkv_def_refuse.sslm and pkv_qk_refuse.sslm from pkv_def.sslm and pkv_qk.sslm.
"""
import hashlib
import struct
import sys

SECTION_COMPOSITION_CONSTANTS = 7
# Chosen with mr2_fixture_probe.exe (record: red-suite-mr2.md): the value that gives both tripping and passing rows
# at layer 1's attn_residual on pkv_def and on pkv_qk.
DEFAULT_MANTISSA = -(1 << 30)


def patch(src, dst, entry, mantissa):
    with open(src, "rb") as f:
        b = bytearray(f.read())
    if b[:4] != b"SSLM":
        sys.exit("%s is not an SSLM artifact" % src)
    nsec = struct.unpack_from("<I", b, 12)[0]
    hit = None
    for i in range(nsec):
        t, _dt, off, size, _e, _a, _r = struct.unpack_from("<IIQQQII", b, 64 + 40 * i)
        if t == SECTION_COMPOSITION_CONSTANTS:
            hit = off
    if hit is None:
        sys.exit("%s: no CompositionConstants section" % src)
    magic, _ver, n, vw, _nbl, _rsv = struct.unpack_from("<4sIIIII", b, hit)
    if magic != b"KVC1" or vw != 2:
        sys.exit("%s: unexpected CompositionConstants layout" % src)
    desc = hit + 24
    vals = desc + 8 * n
    names = vals + 8 * n * vw
    patched = 0
    for i in range(n):
        no, nl = struct.unpack_from("<II", b, desc + 8 * i)
        if bytes(b[names + no:names + no + nl]).decode() == entry:
            struct.pack_into("<q", b, vals + 16 * i, mantissa)
            patched += 1
    if patched != 1:
        sys.exit("%s: expected one %s entry, found %d" % (src, entry, patched))
    b[32:64] = bytes(32)
    b[32:64] = hashlib.sha256(b).digest()
    with open(dst, "wb") as f:
        f.write(b)
    return hashlib.sha256(b).hexdigest()


def layers(path):
    # The last layer index from the constants' names (layer<k>.attn_residual), without a config parser.
    with open(path, "rb") as f:
        data = f.read()
    k = 0
    while (b"layer%d.attn_residual" % (k + 1)) in data:
        k += 1
    return k


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    d = argv[1].rstrip("/")
    m = DEFAULT_MANTISSA
    for a in argv[2:]:
        if a.startswith("--mantissa="):
            m = int(a.split("=", 1)[1], 0)
    for name in ("pkv_def", "pkv_qk"):
        src = "%s/%s.sslm" % (d, name)
        last = layers(src)
        h = patch(src, "%s/%s_refuse.sslm" % (d, name), "layer%d.attn_residual" % last, m)
        print("%s_refuse.sslm: layer%d.attn_residual mantissa %d, sha256 %s" % (name, last, m, h))


if __name__ == "__main__":
    main(sys.argv)
