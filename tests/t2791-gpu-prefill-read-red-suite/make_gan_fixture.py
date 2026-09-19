"""Builds fixture G-an for cell_prompt_guard_status.cpp and cell_prompt_guard_removal.cpp (the
GPU-path plan's Sec3.4 row 5, the prompt twin's device-guard refusal, Sec3.6).

G-an is the U1 pair model (A2, the synthetic fused-K fixture of Sec3.4 row 4) with its
CompositionConstants entry "layer0.attn_norm" set to mantissa -2^30 (-1,073,741,824), its exponent
unchanged, and the artifact's integrity SHA-256 recomputed over the whole file with the 32 hash bytes
at offset 32 zeroed (src/artifact.cpp's integrity check), exactly as make_a2fn_fixture.py re-seals A2-fn.

The loader admits any |m| <= 2^31-1 for this entry (only fused-K q_norm/k_norm are held
canonical-positive), and the CPU and GPU forwards read the constant from the same packed field, so
whether a token trips layer 0's attn_norm site-constant fold depends on the token's embedding row.

WHY -2^30 AND NOT THE LOADER FLOOR. Plan Sec3.4 row 5 starts at m = -2,147,483,647 and says to adjust m
within the admitted range if no pass/trip split exists. At the floor every one of the 384 ids trips,
so there is no split; at -2^30, 120 ids pass and 264 trip, on the CPU and the GPU alike (the
adversary's whole-vocabulary sweep, Claude/Loki/te280-gpu-plan-restrike-2026-09-18.md Sec3). The cell
re-derives that split with its own CPU oracle and pins it.

The patch logic is the adversary's TE-280 probe generator (make_gan.py), with the entry and the
mantissa fixed. Both input and output hashes are pinned: the script refuses any other source and
refuses to leave an output whose hash differs from the one the cells pin.

Usage: python make_gan_fixture.py <u1_pair_model.sslm> <out_gan.sslm>
"""
import hashlib
import struct
import sys

A2_SHA256 = "a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70"
GAN_SHA256 = "cf48079cd3b50eb053c8f8cd57d4feb0f102d8ec9622aaf86300a15daf96e76b"
ENTRY = "layer0.attn_norm"
A2_LAYER0_ATTN_NORM_M = 1090717716
GAN_M = -(1 << 30)
SECTION_COMPOSITION_CONSTANTS = 7


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: make_gan_fixture.py <u1_pair_model.sslm> <out_gan.sslm>")
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
        if name != ENTRY:
            continue
        m, e = struct.unpack_from("<qq", b, vals + 16 * i)
        if m != A2_LAYER0_ATTN_NORM_M:
            sys.exit("%s before patch is (%d, %d), want mantissa %d" % (ENTRY, m, e, A2_LAYER0_ATTN_NORM_M))
        struct.pack_into("<q", b, vals + 16 * i, GAN_M)
        print("patched %s (%d, %d) -> (%d, %d)" % (ENTRY, m, e, GAN_M, e))
        patched += 1
    if patched != 1:
        sys.exit("expected exactly one %s entry, patched %d" % (ENTRY, patched))

    b[32:64] = bytes(32)
    b[32:64] = hashlib.sha256(b).digest()
    out = hashlib.sha256(b).hexdigest()
    if out != GAN_SHA256:
        sys.exit("G-an sha256 %s, want %s" % (out, GAN_SHA256))
    with open(dst, "wb") as f:
        f.write(b)
    print("G-an written: %s sha256 %s bytes %d" % (dst, out, len(b)))


if __name__ == "__main__":
    main(sys.argv)
