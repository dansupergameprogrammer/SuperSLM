"""T-2380 (Curie) -- fixture construction and verification helpers for the
archive-based ship gate's own red suite (test_archive_gate.py,
test_archive_composition.py), design Sec4.1/Sec7 dim 11 as amended by fold
round 42 (Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md).

WHAT THIS MODULE IS NOT, mirroring fp_scan_common.py's own opening
disclaimer: it is not the deciding instrument and it implements none of the
instrument's own logic. `raw_iterate_archive_members`, below, is NOT
`check_fp_free_scan.iterate_archive_members` (design Sec4.1, D-SLM5035) --
it is a fixture-verification-only reader, used solely to (a) extract the
real archive's own real object members so they can be re-archived with a
floating-point carrier inserted at a chosen position, and (b) confirm a
constructed fixture genuinely has the byte-level shape its own population
claims (member counts, kinds, an odd-sized member for the padding check)
before any test asserts against the PRODUCTION reader. Every test that
grades whether the ship gate itself is correct calls
`check_fp_free_scan.iterate_archive_members` / `.enumerate_archive_objects`
/ `.scan_object` directly -- never this module's own reader. Building the
production reader's own logic here would make this module a maker-authored
stand-in for the thing under test, exactly the failure fp_scan_common.py's
docstring already names for the byte-decode helpers it provides.

CONSTRUCTION, adopted rather than re-derived. The forty-seventh and
forty-eighth dimension-11 populations (D-SLM5053, D-SLM5068) are adopted
unmodified from the adversary's own already-executed constructions:

  - the floating-point carrier: `fp_scan_fixtures/fp_carrier.cpp`, adopted
    from `Claude/Loki/t2376-probe/fp_carrier.cpp` (T-2376) -- the function
    body (three lines of real C++, the construction that matters) is
    byte-identical; a 15-line adoption header naming the source and this
    ticket's own copy-in convention differs (T-2382 M2, verified this
    session by diffing both files);
  - the per-position archive family (`fp_at_k.lib`, k in 1..17) and the
    single-append archive (`poisoned.lib`, k=18): the same construction
    method `Claude/Loki/t2378-probe/gen_fixture_cmds.py` uses (extract every
    real OBJECT member, re-archive with `lib.exe`, carrier inserted at
    argument position N+1-k because `lib.exe` writes members in the REVERSE
    of its own argument order -- confirmed by direct execution this
    session, matching T-2378's own documented finding).

Per this ticket's own contract ("adopt that construction rather than
authoring your own ... copy what you need into the suite's own fixture
space; do not write into the probe directory"), nothing here reads from or
writes to `Claude/Loki/t2378-probe/` or `Claude/Loki/t2376-probe/` at test
time -- the one file worth copying (`fp_carrier.cpp`, three lines of real
C++) is copied into this suite's own fixture directory, and every archive
byte this module produces is built fresh, in a temp directory, from a real
`lib.exe` invocation against a real build's own real objects.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_scan_common as fc  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES_DIR = os.path.join(_HERE, "fp_scan_fixtures")
FP_CARRIER_CPP = os.path.join(_FIXTURES_DIR, "fp_carrier.cpp")
FP_CARRIER_SYMBOL = "SslmProbeFpCarrier"

MAGIC = b"!<arch>\n"


class MalformedArchiveError(Exception):
    """Fixture-verification-only. NOT check_fp_free_scan.MalformedArchiveError
    -- this module never imports the production module's exception classes,
    so a test asserting `pytest.raises(scan.MalformedArchiveError)` against
    the PRODUCTION reader can never be satisfied by this helper accidentally
    raising the same class for an unrelated, fixture-construction-side
    reason."""


def raw_iterate_archive_members(path):
    """Fixture-verification-only archive reader, implementing the printed
    member-iterator contract (design Sec4.1) AS AMENDED BY FOLD ROUND 41's
    own even-byte padding correction (D-SLM5054) -- used only to build and
    verify fixtures, never to grade the production reader. Yields
    (offset, kind, disp, data) tuples, kind one of "SYMTAB"/"LONGNAMES"/
    "OBJECT".
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 8 or blob[:8] != MAGIC:
        raise MalformedArchiveError("magic absent or file shorter than 8 bytes: %r" % path)
    off, n, longnames = 8, len(blob), None
    while off < n:
        if off + 60 > n:
            raise MalformedArchiveError("header truncated at byte offset %d" % off)
        hdr = blob[off:off + 60]
        name, size_f, end = hdr[0:16], hdr[48:58], hdr[58:60]
        if end != b"\x60\x0a":
            raise MalformedArchiveError("header at %d: end bytes %r" % (off, end))
        st = size_f.rstrip(b" ")
        if not st or not all(0x30 <= c <= 0x39 for c in st):
            raise MalformedArchiveError("header at %d: size field %r not base-10 ASCII" % (off, size_f))
        size = int(st)
        if off + 60 + size > n:
            raise MalformedArchiveError("header at %d: size %d overruns file" % (off, size))
        pay = off + 60
        nm = name.rstrip(b" ")
        if nm in (b"/", b"/SYM64/"):
            kind, disp = "SYMTAB", nm.decode()
        elif nm == b"//":
            kind, disp = "LONGNAMES", "//"
            longnames = blob[pay:pay + size]
        elif b".obj" in name or b".o" in name:
            kind, disp = "OBJECT", nm.rstrip(b"/").decode(errors="replace")
        elif nm.startswith(b"/") and nm[1:].rstrip(b"/").isdigit():
            o = int(nm[1:].rstrip(b"/"))
            if longnames is None or o >= len(longnames):
                raise MalformedArchiveError("header at %d: bad longnames offset" % off)
            e = longnames.index(b"\x00", o) if b"\x00" in longnames[o:] else len(longnames)
            kind, disp = "OBJECT", longnames[o:e].split(b"/")[0].decode(errors="replace")
        else:
            raise MalformedArchiveError(
                "header at %d: member name %r matches none of the three kinds" % (off, name))
        yield off, kind, disp, blob[pay:pay + size]
        off = pay + size + (size % 2)


def raw_member_counts(path):
    """(total, {kind: count}) over the whole archive -- fixture verification
    only, per this module's own docstring."""
    counts = {}
    total = 0
    for _off, kind, _disp, _data in raw_iterate_archive_members(path):
        counts[kind] = counts.get(kind, 0) + 1
        total += 1
    return total, counts


def raw_object_payloads(path):
    """[(basename, bytes)], in archive order -- every OBJECT-kind member's
    own payload, fixture verification only. The basename strips any
    directory prefix lib.exe stored (e.g. "superslm.dir\\Release\\x.obj"),
    since only a unique, filesystem-legal name is needed to re-extract each
    object to its own file for re-archiving."""
    out = []
    for _off, kind, disp, data in raw_iterate_archive_members(path):
        if kind != "OBJECT":
            continue
        base = os.path.basename(disp.replace("\\", "/"))
        out.append((base, data))
    return out


def has_odd_sized_member(path):
    """True if any member of the real archive at `path` has an odd `size`
    field -- confirms, on demand, that a given real archive genuinely
    exercises ar's own even-byte padding rule (D-SLM5054) rather than
    happening to contain only even-sized members. Fixture verification
    only."""
    with open(path, "rb") as fh:
        blob = fh.read()
    off, n = 8, len(blob)
    while off < n:
        hdr = blob[off:off + 60]
        size = int(hdr[48:58].rstrip(b" "))
        if size % 2 == 1:
            return True
        off = off + 60 + size + (size % 2)
    return False


def build_minimal_padded_archive(out_path):
    """A minimal, REAL, well-formed two-member ar archive (magic + one
    SYMTAB member of odd size 3 + one OBJECT member of size 2), hand-built
    byte-for-byte to the ar format's own documented shape -- not a byte
    patch of a real archive, and not a construction any real production
    toolchain emits, but a genuine, valid `!<arch>` file per the format's
    own public specification, used ONLY as a minimal, isolated pin of the
    even-byte padding rule (D-SLM5054) independent of whether any given
    real toolchain's own archive happens to contain an odd-sized member at
    all. The real-archive-at-real-size cell (test_archive_gate.py's own
    must-accept, real_build_dir-derived) is the population's product claim;
    this is its mechanism check, per Curie's own degenerate-construction
    discipline."""
    def header(name, size):
        name_f = name.encode("ascii").ljust(16, b" ")
        mtime_f = b"0".ljust(12, b" ")
        uid_f = b"0".ljust(6, b" ")
        gid_f = b"0".ljust(6, b" ")
        mode_f = b"100644".ljust(8, b" ")
        size_f = str(size).encode("ascii").ljust(10, b" ")
        end_f = b"\x60\x0a"
        h = name_f + mtime_f + uid_f + gid_f + mode_f + size_f + end_f
        assert len(h) == 60, len(h)
        return h

    sym_payload = b"\x00\x00\x00"          # 3 bytes: odd size -> one pad byte
    obj_payload = b"\xde\xad"              # 2 bytes: even size -> no pad byte
    blob = bytearray(MAGIC)
    blob += header("/", len(sym_payload))
    blob += sym_payload
    blob += b"\n"                          # the one pad byte D-SLM5054 requires
    blob += header("dummy.obj", len(obj_payload))
    blob += obj_payload
    with open(out_path, "wb") as fh:
        fh.write(bytes(blob))
    return out_path


# ---------------------------------------------------------------------------
# lib.exe invocation -- MSVC's own archiver, resolved via the identical
# VsDevCmd.bat this suite's other real-toolchain helpers already use
# (fp_scan_common.py's find_vsdevcmd/_run_via_env_script), pinned per this
# ticket's own environment note (D-SLM5011): an unpinned invocation can
# silently resolve to either installed VS 2022 edition.
# ---------------------------------------------------------------------------

class ToolUnavailable(fc.ToolUnavailable):
    pass


def run_lib_exe(args, cwd):
    """Runs `lib.exe <args>` inside a VsDevCmd -arch=x64 environment, cwd
    fixed so every relative object path in `args` resolves against it.
    Raises ToolUnavailable if no VS install is found; RuntimeError if
    lib.exe itself fails."""
    vsdevcmd = fc.find_vsdevcmd()
    if vsdevcmd is None:
        raise ToolUnavailable(
            "no VsDevCmd.bat found at either well-known VS2022 install "
            "location -- cannot invoke lib.exe in this environment")
    r = fc._run_via_env_script(vsdevcmd, "-arch=x64 -no_logo", ["lib", "/nologo"] + list(args), cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("lib.exe failed ({}): {}\n{}".format(r.returncode, r.stdout, r.stderr))
    return r


def compile_fp_carrier(out_obj):
    """Compiles fp_scan_fixtures/fp_carrier.cpp (adopted from T-2376's own
    probe, function body byte-identical, see this module's docstring) with
    MSVC cl.exe. Raises ToolUnavailable if no VS install is found."""
    return fc.compile_cl(FP_CARRIER_CPP, out_obj)


def build_archive_with_extra_member_at(base_archive, extra_member_obj, position, out_path, work_dir):
    """A genuine `lib.exe`-produced archive holding every real OBJECT member
    of `base_archive`, in its own original order, PLUS `extra_member_obj`
    inserted as the archive's own OBJECT-member position `position`
    (1-based; `position == n_objects + 1` appends it last, matching
    poisoned.lib's own shape). Adopts T-2378's own gen_fixture_cmds.py
    method: `lib.exe` writes members in the REVERSE of its own argument
    order, so placing the extra object at argument index
    `(n_objects + 1) - position` (0-based, from the end) yields OBJECT
    position `position` in the resulting archive -- confirmed by direct
    execution this session against the real archive (18 constructions, one
    per k in 1..18, every one's own OBJECT position checked with
    raw_iterate_archive_members below).

    Returns out_path. Raises ToolUnavailable if no VS install is found.
    """
    objs = raw_object_payloads(base_archive)
    n = len(objs)
    if not (1 <= position <= n + 1):
        raise ValueError("position %d out of range for %d real objects" % (position, n))
    os.makedirs(work_dir, exist_ok=True)
    obj_paths = []
    for i, (base, data) in enumerate(objs):
        p = os.path.join(work_dir, "_src_%02d_%s" % (i, base))
        with open(p, "wb") as fh:
            fh.write(data)
        obj_paths.append(p)
    # Desired OBJECT-member order (position is 1-based):
    desired = obj_paths[:position - 1] + [extra_member_obj] + obj_paths[position - 1:]
    # lib.exe writes members in the REVERSE of the given argument order.
    args = list(reversed(desired))
    run_lib_exe(["/OUT:" + out_path] + args, cwd=work_dir)
    return out_path


def build_poisoned_archive(base_archive, extra_member_obj, out_path, work_dir):
    """The forty-seventh population's own must-reject (D-SLM5053): the real
    archive plus `extra_member_obj` appended last -- the k = n+1 case of
    `build_archive_with_extra_member_at`, kept as its own named function
    because the forty-seventh population's own construction is adopted
    independently of the forty-eighth's per-position family (design Sec7
    dim 11: "k = 18 is poisoned.lib, the forty-seventh population's own
    must-reject")."""
    objs = raw_object_payloads(base_archive)
    return build_archive_with_extra_member_at(
        base_archive, extra_member_obj, len(objs) + 1, out_path, work_dir)


# ---------------------------------------------------------------------------
# Malformed-archive constructions (thirty-seventh/-eighth/-ninth/forty-fourth
# populations' own must-rejects) -- real files, byte-patched from a real
# archive, adopted from T-2376's/T-2378's own build_fixtures() method.
# ---------------------------------------------------------------------------

def _member_offsets(blob):
    offs, off, n = [], 8, len(blob)
    while off < n:
        offs.append(off)
        size = int(bytes(blob[off + 48:off + 58]).rstrip(b" "))
        off = off + 60 + size + (size % 2)
    return offs


def build_malformed_fixtures(base_archive, out_dir, macho_obj_position=2):
    """Every reader-stage must-reject construction this ticket's populations
    37/38/39/44 need, built once from `base_archive`'s own real bytes.
    `macho_obj_position` selects which real OBJECT member (1-based, among
    the archive's own object members) has its payload overwritten with a
    Mach-O magic header -- the forty-fourth population's own text notes
    this pin is "chosen by whoever writes the fixture," so the position is
    a parameter here rather than a hardcoded constant.

    Returns a dict of label -> path.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(base_archive, "rb") as fh:
        blob = bytearray(fh.read())

    def w(name, data):
        p = os.path.join(out_dir, name)
        with open(p, "wb") as fh:
            fh.write(bytes(data))
        return p

    out = {}

    b = bytearray(blob)
    b[0:8] = b"XXXXXXXX"
    out["bad_magic"] = w("bad_magic.lib", b)

    out["truncated"] = w("truncated.lib", blob[:len(blob) // 2 + 13])

    offs = _member_offsets(blob)
    # offs[0..1] = the two SYMTAB members (COFF) or [0] = the one SYMTAB
    # member (ELF); the LONGNAMES member and the object members follow.
    # Locate the LONGNAMES member's own index by kind rather than assuming
    # a fixed prefix count, so this helper works against either toolchain's
    # own archive shape.
    kinds = []
    off = 8
    for o in offs:
        hdr = blob[o:o + 60]
        name = hdr[0:16]
        nm = name.rstrip(b" ")
        if nm in (b"/", b"/SYM64/"):
            kinds.append("SYMTAB")
        elif nm == b"//":
            kinds.append("LONGNAMES")
        else:
            kinds.append("OBJECT")
    obj_indices = [i for i, k in enumerate(kinds) if k == "OBJECT"]
    target_idx = obj_indices[macho_obj_position - 1]
    h = offs[target_idx]

    b = bytearray(blob)
    b[h + 48:h + 58] = b"999999999 "
    out["size_overrun"] = w("size_overrun.lib", b)

    b = bytearray(blob)
    b[h + 48:h + 58] = b"12x4      "
    out["size_nondigit"] = w("size_nondigit.lib", b)

    b = bytearray(blob)
    b[h:h + 16] = b"zzzzzzzzzzzzzzzz"
    out["bad_name"] = w("bad_name.lib", b)

    # zero_objects: keep only the non-OBJECT (index/longnames) members.
    keep = bytearray(MAGIC)
    for i, o in enumerate(offs):
        if kinds[i] == "OBJECT":
            continue
        size = int(bytes(blob[o + 48:o + 58]).rstrip(b" "))
        tot = 60 + size + (size % 2)
        keep += blob[o:o + tot]
    out["zero_objects"] = w("zero_objects.lib", keep)

    b = bytearray(blob)
    size = int(bytes(blob[h + 48:h + 58]).rstrip(b" "))
    macho_magic = bytearray(b"\xcf\xfa\xed\xfe")  # Mach-O 64-bit magic (0xFEEDFACF, little-endian on disk)
    b[h + 60:h + 60 + size] = macho_magic + bytearray(size - len(macho_magic))
    out["macho_member"] = w("macho_member.lib", b)
    out["macho_member_position"] = macho_obj_position

    return out


class TempDir(fc.TempDir):
    """Reuses fp_scan_common.py's own TempDir convention (fresh scratch dir
    per test, removed afterward), named locally so this module's own
    imports read self-containedly."""
    pass
