"""Builds fixture A2-fn for cell_final_norm_guard.cpp (the GPU-path plan's Sec3.4 row 5).

A2-fn is the U1 pair model (A2, the synthetic fused-K fixture of Sec3.4 row 4) with its
CompositionConstants entry "final_norm" set to mantissa -2,147,483,647 (the loader's floor,
|m| <= 2^31-1), its exponent unchanged, and the artifact's integrity SHA-256 recomputed over the
whole file with the 32 hash bytes at offset 32 zeroed (src/artifact.cpp's integrity check).

The loader admits it, the GPU map reads the constant for presence only, and the prefill never
applies final_norm, so a prefill succeeds; RmsNormSite's site-constant fold then refuses with
CarriedScaleMantissaOutOfDomain, which is the read's step-5 guard refusal.

The patch logic is the adversary's TE-276 probe generator (make_a2fn.py), with its paths made
arguments. Both input and output hashes are pinned: the script refuses any other source and
refuses to leave an output whose hash differs from the one the cell pins.

Usage: python make_a2fn_fixture.py <u1_pair_model.sslm> <out_a2fn.sslm>
"""
import hashlib
import struct
import sys

A2_SHA256 = "a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70"
A2FN_SHA256 = "73a1ec9e1846be16af7f8dc13511207d4129a9e825cc5271f783f6bc1ff4f940"
A2_FINAL_NORM = (1090717716, -60)
A2FN_FINAL_NORM_M = -2147483647
SECTION_COMPOSITION_CONSTANTS = 7


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: make_a2fn_fixture.py <u1_pair_model.sslm> <out_a2fn.sslm>")
    src, dst = argv[1], argv[2]
    with open(src, "rb") as f:
        b = bytearray(f.read())
    got = hashlib.sha256(b).hexdigest()
    if got != A2_SHA256:
        sys.exit("source is not the U1 pair model: sha256 %s, want %s" % (got, A2_SHA256))
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
        if name != "final_norm":
            continue
        m, e = struct.unpack_from("<qq", b, vals + 16 * i)
        if (m, e) != A2_FINAL_NORM:
            sys.exit("final_norm before patch is (%d, %d), want %s" % (m, e, A2_FINAL_NORM))
        struct.pack_into("<q", b, vals + 16 * i, A2FN_FINAL_NORM_M)
        patched += 1
    if patched != 1:
        sys.exit("expected exactly one final_norm entry, patched %d" % patched)

    b[32:64] = bytes(32)
    b[32:64] = hashlib.sha256(b).digest()
    out = hashlib.sha256(b).hexdigest()
    if out != A2FN_SHA256:
        sys.exit("A2-fn sha256 %s, want %s" % (out, A2FN_SHA256))
    with open(dst, "wb") as f:
        f.write(b)
    print("A2-fn written: %s sha256 %s bytes %d" % (dst, out, len(b)))


if __name__ == "__main__":
    main(sys.argv)
