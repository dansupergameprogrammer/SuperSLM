"""check_fp_free_scan.py -- design Sec4.1's deciding instrument (T-2338, Brunel),
built to the RATIFIED production contract (fold round 33,
`Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md` Sec4.1 / Sec7
dimension 11; D-SLM4826/D-SLM4827/D-SLM4830/D-SLM4834; test contract
`tests/t2296-fp-free-open-red-suite/test_check_fp_free_scan.py`).

WHAT THIS IS. An instruction-level, byte-accounting, default-deny scan over
compiled machine code. It decodes an object's own bytes (never a disassembler's
rendered text), classifies every instruction and every call/tail-jmp edge, and
emits a per-symbol ACCEPT/REJECT verdict -- or REFUSES the whole object,
emitting no verdict for any symbol in it, whenever any byte of any code section
cannot be fully accounted for.

THE FIRST LAW (design Sec4.1, fold round 9/11, restated as the instrument's own
governing rule): every stage accounts for what it discarded, and a stage that
cannot classify its input emits a rejection rather than a silence. Concretely:
this reader counts, per code section, the bytes it could not attribute to a
decoded instruction, a symbol-grounded data range, or a recognised padding
run, and REFUSES -- emits no verdict for any symbol in the OBJECT -- whenever
that count is nonzero anywhere in the object. A REFUSE is not a defect in the
scan; it is the scan doing its job.

THE COVERAGE RELATION (fold rounds 13/14). Every byte of a code section is
covered by exactly one of three things, or it is unaccounted:
  1. a CODE EXTENT -- the byte range a (possibly aliased) code symbol's own
     instructions occupy, chain-decoded exactly from the symbol's own start to
     its own end (declared size when the object states one, else the next
     WITNESSED boundary -- any function-typed symbol regardless of storage
     class, or any external/object-typed non-function symbol; COFF
     STATIC/LABEL bookkeeping symbols and any ELF symbol that is neither
     function- nor object-typed witness nothing). A gap, an overlap, or a
     SPILL (an instruction whose own span would read past the extent's own
     end -- the classic movabs-swallow shape) stops the chain at the point of
     failure; an empty extent, or any partial chain, makes that symbol's own
     attribution unverifiable and REFUSES THE WHOLE OBJECT, not only that
     symbol -- a symbol table shown to place one code symbol inconsistently is
     not trusted to have bounded its neighbours correctly either.
  2. a DATA EXTENT -- grounded in the object's own relocation/unwind metadata.
     THIS READER IMPLEMENTS NO SUCH GROUNDING (named residual, this ticket's
     own build record) -- an object-typed/external non-function witnessing
     symbol still marks a BOUNDARY (so it correctly clamps a neighbouring
     code extent) but is never credited as covering its own bytes; those
     bytes fall through to test 3 and REFUSE when they are not pure padding,
     the conservative, fail-closed disposition the design's own text names as
     correct rather than a defect.
  3. a RECOGNISED PADDING RUN -- every byte in an otherwise-uncovered span
     (an inter-extent gap, or a partial extent's own unclaimed tail) matches
     one of the ISA's own no-op byte patterns (x86: 0x90/0xCC repeated;
     AArch64: the fixed 4-byte NOP encoding repeated). Decodability plays no
     role in this test.

CHECKS (A)/(B)/(C), run only once a section's accounting is clean: (A) a
register-file classification, default-deny over any instruction touching a
vector/FP/mask/tile register unless the mnemonic is on a closed movement-only
allowlist (or is packed-integer/self-zeroing-xor by a structural naming rule);
(B) a pinned GPR/control-flow mnemonic allowlist, default-deny over everything
that does not touch a vector/FP register at all; (C) a default-deny call/
tail-jmp target allowlist -- an edge to a symbol inside this object's own
scanned table needs no separate vetting (the target is scanned in its own
right); an edge to a symbol outside it is accepted only if individually
vetted (EXTERN_ALLOW); an edge this reader cannot statically resolve (an
indirect call/jmp, or a direct edge with no relocation and no matching local
symbol) is rejected.

NAMED RESIDUALS (this build; see the build record's own Sec10-style list for
the full disposition and what would close each): no Mach-O reader (Sec4.1
itself discloses this design has never had one); the AArch64 GPR/control-flow
allowlist below is derived from this ticket's own probe objects, not a
re-derivation against the real 17-TU corpus rebuilt for AArch64 (unchanged
disposition from design Sec7 dim 11's own tenth/eleventh populations); check
(C)'s relocation-to-symbol resolution is COFF- and ELF-RELA-complete for the
object shapes this repo's own toolchains produce, but does not implement every
relocation TYPE a linker recognises -- an edge whose relocation type this
reader does not recognise is treated as unresolved (rejected), never silently
accepted.
"""
from __future__ import annotations

import os
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import capstone

# ---------------------------------------------------------------------------
# ScanResult / public dataclass surface (design Sec4.1's ratified contract).
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    object_format: str
    refuse: bool
    unclassified_bytes: int
    verdicts: dict = field(default_factory=dict)
    # T-2367 (Brunel), design Sec4.1/Sec5.5 fold round 39 (D-SLM4985/D-SLM4996):
    # the ship gate decides on checks (A)/(B) alone; check (C) keeps running and
    # keeps reporting through `verdicts` above (the combined ab_accept-and-
    # c_accept verdict, unchanged, still the full diagnostic), but nothing
    # gating may read it. `verdicts` alone cannot express that distinction --
    # every caller of `scan_object` before this field existed read one ANDed
    # string with no way to tell which check produced a REJECT (T-2364 Finding
    # C6-C9, T-2365 Finding 3, both independently). `ab_verdicts` is the
    # checks-(A)/(B)-only verdict, per symbol, computed identically to
    # `verdicts` except that a check-(C) failure never turns an ACCEPT into a
    # REJECT here. The production gate (`scan_build_output.py`) reads THIS
    # field to decide pass/fail; `verdicts` remains the non-gating diagnostic
    # surface check (C)'s own attribution is read from.
    ab_verdicts: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Object-format detection -- read from the object's own header, never a
# caller-supplied label (design Sec4.1's correction to T-2326's own proposal,
# D-SLM4826/D-SLM4831).
# ---------------------------------------------------------------------------

_ELF_MAGIC = b"\x7fELF"
_COFF_MACHINES = {0x8664, 0xAA64, 0x14C, 0x1C0, 0x1C4}  # x64, arm64, i386, arm, armnt


def _read_object_format(data: bytes) -> str:
    if data[:4] == _ELF_MAGIC:
        return "elf"
    if len(data) >= 2:
        machine, = struct.unpack_from("<H", data, 0)
        if machine in _COFF_MACHINES:
            return "coff"
    raise ValueError(
        "unrecognized object format: not ELF, not a known COFF machine type "
        "(Mach-O and any other format are a named residual -- this reader "
        "does not implement a Mach-O header reader)"
    )


# ---------------------------------------------------------------------------
# A parsed symbol, uniform across ELF/COFF.
# ---------------------------------------------------------------------------


@dataclass
class _Sym:
    name: str
    value: int          # byte offset within its section
    size: int            # 0 == undeclared
    is_function: bool
    is_witness: bool     # witnesses a boundary (function, or external/object non-function)
    raw_index: int       # position in the object's own raw symbol table (relocation-addressable)


@dataclass
class _CodeSection:
    index: int
    name: str
    data: bytes          # this section's own raw bytes (never concatenated with another)
    symbols: list         # list[_Sym], value relative to this section


@dataclass
class _Reloc:
    offset: int           # byte offset within the section the relocation applies to
    sym_raw_index: int


# ---------------------------------------------------------------------------
# ELF64 reader.
# ---------------------------------------------------------------------------

_SHF_EXECINSTR = 0x4
_SHT_SYMTAB = 2
_SHT_RELA = 4
_SHT_REL = 9
_STT_MASK = 0xF
_STT_OBJECT = 1
_STT_FUNC = 2


def _parse_elf(data: bytes):
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, = struct.unpack_from("<H", data, 0x3A)
    e_shnum, = struct.unpack_from("<H", data, 0x3C)
    e_shstrndx, = struct.unpack_from("<H", data, 0x3E)

    def shdr(i):
        off = e_shoff + i * e_shentsize
        (name_off, sh_type, flags, addr, offset, size, link, info, align,
         entsize) = struct.unpack_from("<IIQQQQIIQQ", data, off)
        return dict(name_off=name_off, sh_type=sh_type, flags=flags,
                    offset=offset, size=size, link=link, info=info,
                    entsize=entsize)

    sections = [shdr(i) for i in range(e_shnum)]
    shstrtab = sections[e_shstrndx]
    shstr_data = data[shstrtab["offset"]:shstrtab["offset"] + shstrtab["size"]]

    def sec_name(name_off):
        end = shstr_data.index(b"\x00", name_off)
        return shstr_data[name_off:end].decode("ascii", "replace")

    for s in sections:
        s["name"] = sec_name(s["name_off"])

    symtab_idx = next((i for i, s in enumerate(sections) if s["sh_type"] == _SHT_SYMTAB), None)
    raw_syms = []  # (name, value, size, shndx, is_function, is_object)
    if symtab_idx is not None:
        symtab = sections[symtab_idx]
        strtab = sections[symtab["link"]]
        strtab_data = data[strtab["offset"]:strtab["offset"] + strtab["size"]]
        entsize = symtab["entsize"] or 24
        n = symtab["size"] // entsize
        for i in range(n):
            off = symtab["offset"] + i * entsize
            st_name, st_info, st_other, st_shndx, st_value, st_size = \
                struct.unpack_from("<IBBHQQ", data, off)
            end = strtab_data.index(b"\x00", st_name)
            name = strtab_data[st_name:end].decode("ascii", "replace")
            st_type = st_info & _STT_MASK
            raw_syms.append(dict(name=name, value=st_value, size=st_size,
                                 shndx=st_shndx, is_function=(st_type == _STT_FUNC),
                                 is_object=(st_type == _STT_OBJECT), raw_index=i))

    # Relocations: SHT_RELA/SHT_REL sections, .info names the section they apply to.
    relocs_by_target_section: dict[int, list] = {}
    for i, s in enumerate(sections):
        if s["sh_type"] not in (_SHT_RELA, _SHT_REL):
            continue
        target = s["info"]
        entries = []
        if s["sh_type"] == _SHT_RELA:
            entsize = s["entsize"] or 24
            n = s["size"] // entsize
            for j in range(n):
                off = s["offset"] + j * entsize
                r_offset, r_info, _r_addend = struct.unpack_from("<QQq", data, off)
                sym_idx = r_info >> 32
                entries.append(_Reloc(offset=r_offset, sym_raw_index=sym_idx))
        else:
            entsize = s["entsize"] or 16
            n = s["size"] // entsize
            for j in range(n):
                off = s["offset"] + j * entsize
                r_offset, r_info = struct.unpack_from("<QQ", data, off)
                sym_idx = r_info >> 32
                entries.append(_Reloc(offset=r_offset, sym_raw_index=sym_idx))
        relocs_by_target_section.setdefault(target, []).extend(entries)

    code_sections = []
    for i, s in enumerate(sections):
        if not (s["flags"] & _SHF_EXECINSTR):
            continue
        sec_data = data[s["offset"]:s["offset"] + s["size"]]
        syms = []
        for rs in raw_syms:
            if rs["shndx"] != i:
                continue
            witness = rs["is_function"] or rs["is_object"]
            syms.append(_Sym(name=rs["name"], value=rs["value"], size=rs["size"],
                             is_function=rs["is_function"], is_witness=witness,
                             raw_index=rs["raw_index"]))
        code_sections.append(_CodeSection(index=i, name=s["name"], data=sec_data,
                                          symbols=syms))

    # Global raw-symbol lookup (by raw index), for relocation resolution.
    sym_by_raw = {rs["raw_index"]: rs for rs in raw_syms}
    return code_sections, sym_by_raw, relocs_by_target_section


# ---------------------------------------------------------------------------
# COFF reader.
# ---------------------------------------------------------------------------

_IMAGE_SCN_CNT_CODE = 0x20
_IMAGE_SCN_MEM_EXECUTE = 0x20000000
_IMAGE_SYM_CLASS_EXTERNAL = 2


def _coff_sym_name(data: bytes, strtab: bytes, raw8: bytes) -> str:
    if raw8[:4] == b"\x00\x00\x00\x00":
        off, = struct.unpack_from("<I", raw8, 4)
        end = strtab.index(b"\x00", off)
        return strtab[off:end].decode("ascii", "replace")
    return raw8.rstrip(b"\x00").decode("ascii", "replace")


def _parse_coff(data: bytes):
    _machine, nsec = struct.unpack_from("<HH", data, 0)
    symptr, nsym = struct.unpack_from("<II", data, 8)
    strtab_off = symptr + nsym * 18
    strtab = data[strtab_off:]

    sections_raw = []
    for i in range(nsec):
        off = 20 + i * 40
        raw_name = data[off:off + 8]
        _va_size, _va, size_raw, ptr_raw = struct.unpack_from("<IIII", data, off + 8)
        ptr_reloc, _ptr_line, nreloc, _nline = struct.unpack_from("<IIHH", data, off + 24)
        characteristics, = struct.unpack_from("<I", data, off + 36)
        sections_raw.append(dict(name=raw_name, size=size_raw, ptr=ptr_raw,
                                 ptr_reloc=ptr_reloc, nreloc=nreloc,
                                 characteristics=characteristics))

    raw_syms = []  # dict per raw slot (aux slots included as None placeholders)
    i = 0
    while i < nsym:
        off = symptr + i * 18
        name8 = data[off:off + 8]
        value, sec_num, sym_type, storage, naux = struct.unpack_from("<IhHBB", data, off + 8)
        name = _coff_sym_name(data, strtab, name8)
        is_function = ((sym_type & 0xFF) >> 4) == 2
        is_external = storage == _IMAGE_SYM_CLASS_EXTERNAL
        witness = is_function or is_external
        raw_syms.append(dict(name=name, value=value, sec_num=sec_num,
                             is_function=is_function, is_witness=witness,
                             raw_index=i, size=0))
        for k in range(1, naux + 1):
            raw_syms.append(None)  # aux slot: occupies a raw index, carries nothing
        i += 1 + naux

    sym_by_raw = {idx: rs for idx, rs in enumerate(raw_syms) if rs is not None}

    relocs_by_section: dict[int, list] = {}
    for i, s in enumerate(sections_raw):
        entries = []
        for j in range(s["nreloc"]):
            off = s["ptr_reloc"] + j * 10
            va, sym_idx, _rtype = struct.unpack_from("<IIH", data, off)
            entries.append(_Reloc(offset=va, sym_raw_index=sym_idx))
        relocs_by_section[i] = entries

    code_sections = []
    for i, s in enumerate(sections_raw):
        if not (s["characteristics"] & (_IMAGE_SCN_CNT_CODE | _IMAGE_SCN_MEM_EXECUTE)):
            continue
        sec_data = data[s["ptr"]:s["ptr"] + s["size"]]
        syms = []
        for rs in raw_syms:
            if rs is None:
                continue
            if rs["sec_num"] != i + 1:  # COFF section numbers are 1-based
                continue
            syms.append(_Sym(name=rs["name"], value=rs["value"], size=0,
                             is_function=rs["is_function"], is_witness=rs["is_witness"],
                             raw_index=rs["raw_index"]))
        code_sections.append(_CodeSection(index=i, name=s["name"].rstrip(b"\x00").decode("latin1"),
                                          data=sec_data, symbols=syms))

    return code_sections, sym_by_raw, relocs_by_section


def _function_symbol_names_from_bytes(data: bytes) -> set:
    """The byte-based core of `_read_function_symbol_names`, below --
    factored out (T-2381, Brunel, design Sec4.1's archive member-iterator
    contract) so the archive-based corpus driver
    (`scan_build_output.py`'s own per-archive-member path) can build the
    identical `corpus_symbols` index from an in-memory member payload,
    without a temporary file, the same "by byte range, with no extraction
    to a temporary file" discipline the design states for `scan_object`'s
    own `data` parameter. `_read_function_symbol_names` is unchanged in
    signature and behavior; it now delegates here rather than duplicating
    the parse."""
    object_format = _read_object_format(data)
    if object_format == "elf":
        code_sections, _sym_by_raw, _relocs = _parse_elf(data)
    else:
        code_sections, _sym_by_raw, _relocs = _parse_coff(data)
    names = set()
    for section in code_sections:
        for sym in section.symbols:
            if sym.is_function:
                names.add(sym.name)
    return names


def _read_function_symbol_names(obj_path: str) -> set:
    """The corpus_symbols index's own per-object contribution (design
    Sec4.1 fold round 34 gap (b)): every function-typed symbol name defined
    in this one compiled object, read directly from its own symbol table --
    no decode, no accounting, no per-instruction check, the same data
    `_check_c_for_symbol`'s own relocation resolution already reads.

    T-2371 (Brunel), D-SLM5018 M2: moved here from
    `run_fp_free_scan_real_corpus.py` (T-2338/T-2343), which retired as the
    ship gate's own mechanism at fold round 39 (D-SLM4981). This module's
    scanning entry points (`scan_object`, `ci_gate`, `ci_gate_corpus`) are
    the ship gate's own production surface; `scan_build_output.py` -- the
    gate's current driver -- called this function through the retired
    module, making a file the design calls "no longer load-bearing" a hard
    import dependency of the gate. It is defined here instead, so the
    retired driver can be deleted without breaking the gate, and the
    retired driver now calls this copy rather than defining its own."""
    with open(obj_path, "rb") as f:
        data = f.read()
    return _function_symbol_names_from_bytes(data)


# ---------------------------------------------------------------------------
# Archive member iterator (T-2381, Brunel, design Sec4.1, D-SLM5035, as
# amended by fold rounds 41/42 -- the even-byte padding correction,
# D-SLM5054, and the toolchain-specific member-count correction, D-SLM5055).
# The corpus becomes the archive the build actually ships (superslm.lib /
# libsuperslm.a), read member-by-member here, replacing the object-directory
# glob as the gate's own membership mechanism (`scan_build_output.py`,
# below).
# ---------------------------------------------------------------------------

_ARCHIVE_MAGIC = b"!<arch>\n"
_ARCHIVE_HEADER_SIZE = 60
_ARCHIVE_END_MARKER = b"\x60\x0a"


class MalformedArchiveError(Exception):
    """Design Sec4.1, D-SLM5035: reserved for HEADER-LEVEL corruption inside
    an archive whose own magic and file structure otherwise parse -- a
    member's end marker, size field, or name field failing its own
    well-formedness check. A distinct failure from the archive's own magic
    being absent or the file being too short to be an archive at all (a
    plain `ValueError`, below): "the container's own structure could not be
    trusted" and "no container was found" are the same KIND of failure to
    the gate (both an infrastructure failure, exit 2, never a REFUSE), but
    they are not the same class of Python exception, per the design's own
    text naming this class specifically for header-level corruption."""


class ArchiveHasNoObjectsError(Exception):
    """Design Sec4.1, D-SLM5035: raised by `enumerate_archive_objects` when
    a well-formed archive (magic present, every header well-formed) carries
    no OBJECT-kind member at all -- an archive containing only its own
    index/longnames members, or nothing after the magic. This is a distinct
    failure from `MalformedArchiveError`: the container is not corrupt, it
    simply has nothing to scan, "identical treatment to today's
    zero-objects-found disposition, never a pass on an archive that opens
    cleanly but ships nothing to scan.\""""


@dataclass
class ArchiveMember:
    """One member of an `!<arch>` archive, as `iterate_archive_members`
    yields it. `name` is the member's own resolved name (the index name for
    SYMTAB, the literal "//" for LONGNAMES, or the object's own filename --
    resolved through the longnames table for a GNU-style "/<offset>"
    reference -- for OBJECT). `payload` is this member's own bytes (exactly
    `size` bytes; the even-byte pad byte, when present, is neither part of
    nor counted in this range)."""
    name: str
    kind: str          # "SYMTAB" | "LONGNAMES" | "OBJECT"
    offset: int        # byte offset of this member's own 60-byte header
    payload: bytes


def iterate_archive_members(path: str):
    """Design Sec4.1, D-SLM5035, as amended by D-SLM5054 (even-byte member
    padding). Opens `path` and reads its first 8 bytes: they MUST equal the
    literal magic `b"!<arch>\\n"`, the common `ar` container both GNU/Unix
    `ar` and Microsoft's `lib.exe` share. A file shorter than 8 bytes, a
    mismatched magic, or a file that cannot be opened at all raises
    `ValueError` -- an INFRASTRUCTURE FAILURE (exit 2, never a pass),
    identical in kind to today's "build directory not found," but the
    design's own text names no specific exception class for a magic
    mismatch (unlike `MalformedArchiveError`, reserved for header-level
    corruption below), so any caller here catches `Exception` broadly for
    this branch.

    After the magic, members are read sequentially, each preceded by a
    fixed 60-byte header: name[16], mtime[12], uid[6], gid[6], mode[8],
    size[10], end[2]. Three checks, each individually load-bearing: `end`
    MUST equal the literal two bytes 0x60 0x0A; `size` MUST parse as a
    base-10 ASCII integer (space-padded, no other character permitted); the
    member's declared size MUST NOT exceed the bytes actually remaining in
    the file. Any header failing any one of these three RAISES
    `MalformedArchiveError` naming the byte offset of the failing header.

    Each member's own 16-byte `name` field resolves to exactly one of three
    kinds, with no fourth branch: SYMTAB ("/" or "/SYM64/"), LONGNAMES
    ("//"), or OBJECT (a literal ".obj"/".o" in the name field directly, or
    a GNU-style "/<offset>" reference into the already-read longnames
    member). A member matching none of the three RAISES
    `MalformedArchiveError` -- there is no "assume object" default.

    D-SLM5054: `ar`'s own even-byte member padding. When a member's own
    `size` is odd, exactly one pad byte follows the payload before the next
    60-byte header begins -- this reader advances past it when computing
    the next header's offset; the pad byte is never yielded as part of any
    `ArchiveMember`'s payload and is not itself subject to the three
    header-level checks above.
    """
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError as exc:
        raise ValueError(
            "cannot open {!r} as an archive: {}".format(path, exc)) from exc
    if len(blob) < 8 or blob[:8] != _ARCHIVE_MAGIC:
        raise ValueError(
            "{!r} is not a recognized archive: magic {!r} absent or file "
            "shorter than 8 bytes".format(path, _ARCHIVE_MAGIC))

    off = 8
    n = len(blob)
    longnames = None
    while off < n:
        if off + _ARCHIVE_HEADER_SIZE > n:
            raise MalformedArchiveError(
                "header truncated at byte offset {}".format(off))
        hdr = blob[off:off + _ARCHIVE_HEADER_SIZE]
        name_field = hdr[0:16]
        end = hdr[58:60]
        if end != _ARCHIVE_END_MARKER:
            raise MalformedArchiveError(
                "header at byte offset {}: end marker {!r} != {!r}".format(
                    off, end, _ARCHIVE_END_MARKER))
        size_field = hdr[48:58]
        stripped = size_field.rstrip(b" ")
        if not stripped or not all(0x30 <= c <= 0x39 for c in stripped):
            raise MalformedArchiveError(
                "header at byte offset {}: size field {!r} is not base-10 "
                "ASCII".format(off, size_field))
        size = int(stripped)
        payload_start = off + _ARCHIVE_HEADER_SIZE
        if payload_start + size > n:
            raise MalformedArchiveError(
                "header at byte offset {}: declared size {} overruns the "
                "file (only {} byte(s) remain)".format(
                    off, size, n - payload_start))
        payload = blob[payload_start:payload_start + size]

        nm = name_field.rstrip(b" ")
        if nm in (b"/", b"/SYM64/"):
            kind, resolved = "SYMTAB", nm.decode("ascii")
        elif nm == b"//":
            kind, resolved = "LONGNAMES", "//"
            longnames = payload
        elif b".obj" in name_field or b".o" in name_field:
            kind = "OBJECT"
            resolved = nm.rstrip(b"/").decode("utf-8", errors="replace")
        elif nm.startswith(b"/") and nm[1:].rstrip(b"/").isdigit():
            longname_offset = int(nm[1:].rstrip(b"/"))
            if longnames is None or longname_offset >= len(longnames):
                raise MalformedArchiveError(
                    "header at byte offset {}: /<offset> name references a "
                    "longnames table that is absent or too short".format(off))
            terminator = longnames.find(b"\x00", longname_offset)
            end_idx = terminator if terminator != -1 else len(longnames)
            kind = "OBJECT"
            resolved = longnames[longname_offset:end_idx].split(b"/")[0].decode(
                "utf-8", errors="replace")
        else:
            raise MalformedArchiveError(
                "header at byte offset {}: member name {!r} matches none of "
                "the three kinds (symbol table, longnames, object)".format(
                    off, name_field))

        yield ArchiveMember(name=resolved, kind=kind, offset=off, payload=payload)
        off = payload_start + size + (size % 2)


def enumerate_archive_objects(archive_path: str) -> list:
    """Design Sec4.1, D-SLM5035: every OBJECT-kind member
    `iterate_archive_members` yields, in archive order, each carrying the
    byte range of its own payload within the archive file. Zero object
    members -- an archive containing only index/longnames members, or
    nothing at all -- raises `ArchiveHasNoObjectsError`, an infrastructure
    failure (exit 2), never a pass."""
    objects = [m for m in iterate_archive_members(archive_path) if m.kind == "OBJECT"]
    if not objects:
        raise ArchiveHasNoObjectsError(
            "archive {!r} contains no OBJECT-kind members -- nothing to "
            "scan is an infrastructure failure, never a pass".format(archive_path))
    return objects


# ---------------------------------------------------------------------------
# ISA decoders.
# ---------------------------------------------------------------------------


def _decoder(isa: str):
    if isa == "aarch64":
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    elif isa == "x86-64":
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    else:
        raise ValueError("unsupported isa: {!r} (expected 'x86-64' or 'aarch64')".format(isa))
    md.detail = True
    return md


def _decode_one(md, data: bytes, offset: int):
    """Decode exactly one instruction at `offset` within `data`. Returns the
    capstone instruction, or None on decode failure."""
    for insn in md.disasm(data[offset:offset + 16], offset):
        return insn
    return None


_X86_PAD_BYTES = (0x90, 0xCC)


def _is_multi_byte_nop_run(md, data: bytes, start: int, end: int) -> bool:
    """T-2343 (Brunel), diagnosing D-SLM4849 (Popper's undiagnosed
    clang/ELF/x86-64 REFUSE-on-everything finding). clang and GCC pad
    function alignment with multi-byte NOP encodings (Intel's own documented
    1-9-byte NOP table, `0F 1F ...` forms, optionally preceded by one or
    more `0x66`/`0x2E` legacy-prefix bytes for longer runs) -- bytes that are
    neither bare `0x90` nor `0xCC`, so `_is_padding_run`'s own bytewise test
    (unchanged, still the first and cheaper test) never matched them and the
    whole object REFUSEd on every real clang/ELF object, FP-carrying or not.

    Decodability is deliberately excluded from `_is_padding_run` itself
    (design Sec4.1's own fold-13 law: a byte range's accounting must not
    depend on whether an arbitrary decode of it happens to succeed, which is
    exactly the movabs-swallow escape that law closes). This function does
    NOT reopen that escape: it runs only over an INTER-EXTENT GAP no code or
    data extent has claimed (never a symbol's own declared/inferred range,
    where a false decode could misattribute bytes to a verdict), and it
    requires every decoded instruction in the gap to be a genuine NOP
    (capstone's own mnemonic for every documented multi-byte NOP encoding is
    literally "nop", the same string a bare 0x90 decodes to) with ZERO
    residue -- a materially narrower target than "decodes as something,"
    which is what made the movabs-swallow shape exploitable in the first
    place. Requires the WHOLE gap to decode as nothing but NOPs; any
    non-NOP instruction or decode failure anywhere in the span means this is
    not a recognised pattern, and the gap remains unaccounted."""
    offset = start
    while offset < end:
        insn = _decode_one(md, data, offset)
        if insn is None or insn.mnemonic.lower() != "nop":
            return False
        if offset + insn.size > end:
            return False  # would spill past the gap's own end
        offset += insn.size
    return offset == end


def _is_padding_run(data: bytes, start: int, end: int, isa: str, md=None) -> bool:
    if end <= start:
        return True
    span = data[start:end]
    if isa == "aarch64":
        if len(span) % 4 != 0:
            return False
        nop_word = bytes([0x1F, 0x20, 0x03, 0xD5])
        return all(span[i:i + 4] == nop_word for i in range(0, len(span), 4))
    if all(b in _X86_PAD_BYTES for b in span):
        return True
    if md is not None:
        return _is_multi_byte_nop_run(md, data, start, end)
    return False


# ---------------------------------------------------------------------------
# Extent computation: group aliased (same-start) symbols, bound each group's
# own end by its own declared size (if any) clamped to the next WITNESSED
# boundary, per design Sec4.1 fold rounds 14/15.
# ---------------------------------------------------------------------------


@dataclass
class _Extent:
    start: int
    end: int
    names: list


def _compute_extents(section: _CodeSection) -> list:
    witness_starts = sorted({s.value for s in section.symbols if s.is_witness})
    by_start: dict[int, list] = {}
    for s in section.symbols:
        if not s.is_function:
            continue
        by_start.setdefault(s.value, []).append(s)

    sec_size = len(section.data)
    extents = []
    for start in sorted(by_start):
        group = by_start[start]
        # T-2343 (Brunel), Poirot's M6: design Sec4.1's fold-14 rule requires
        # one extent per alias group, computed once, but does not name which
        # reduction to take when declared sizes within a group disagree
        # (they never do on any real toolchain observed so far). `max` is
        # chosen deliberately: an ALIAS's own declared sizes are independent
        # facts the compiler asserted about the SAME bytes, and the extent
        # is clamped to `next_witness`/`sec_size` immediately below
        # regardless of which reduction is used, so neither choice can grow
        # past a real neighbouring boundary -- `min` would silently shrink
        # the extent below what one alias's own declared size states without
        # a boundary forcing it to, which is the direction more likely to
        # produce a spurious spill on a real, correctly-sized alias.
        declared = max((s.size for s in group), default=0)
        later = [w for w in witness_starts if w > start]
        next_witness = later[0] if later else sec_size
        if declared:
            end = min(start + declared, next_witness, sec_size)
        else:
            end = min(next_witness, sec_size)
        extents.append(_Extent(start=start, end=end, names=[s.name for s in group]))
    return extents


def _chain_decode(md, data: bytes, start: int, end: int):
    """Attempt a full, exact chain-decode of [start, end). Returns
    (success, instructions, stopped_at) -- stopped_at == end iff success."""
    offset = start
    insns = []
    while offset < end:
        insn = _decode_one(md, data, offset)
        if insn is None:
            break
        if offset + insn.size > end:
            break  # spill: this instruction would read past the extent's own end
        insns.append(insn)
        offset += insn.size
    return (offset == end and end > start), insns, offset


def _account_section(section: _CodeSection, isa: str, md):
    """Returns (ok, unclassified_bytes, verdict_instructions) where
    verdict_instructions maps EVERY alias name in a successful extent to its
    own decoded instruction list. ok is False iff this section REFUSES."""
    extents = _compute_extents(section)
    sec_size = len(section.data)
    ok = True
    unclassified = 0
    empty_extent_charged = False
    per_symbol_insns: dict = {}

    cursor = 0
    all_extents_sorted = sorted(extents, key=lambda e: e.start)
    boundaries = []
    for ext in all_extents_sorted:
        if ext.start > cursor:
            boundaries.append(("gap", cursor, ext.start))
        boundaries.append(("extent", ext.start, ext.end, ext))
        cursor = max(cursor, ext.end)
    if cursor < sec_size:
        boundaries.append(("gap", cursor, sec_size))

    for b in boundaries:
        if b[0] == "gap":
            _, g_start, g_end = b
            if g_end > g_start and not _is_padding_run(section.data, g_start, g_end, isa, md=md):
                ok = False
                unclassified += (g_end - g_start)
            continue
        _, e_start, e_end, ext = b
        if e_end <= e_start:
            ok = False
            # T-2343 (Brunel), Poirot's O5: an empty extent has no byte range
            # of its own to attribute a residue to, but reporting
            # unclassified_bytes == 0 on a REFUSE reads identically to "this
            # object had nothing wrong at the byte level," which is false --
            # the whole section's own attribution is what became
            # unverifiable (fold-14's own rule: "a symbol table shown to
            # place one code symbol inconsistently is not trusted to have
            # bounded its neighbours correctly either"). Charge the section's
            # own total size, so a caller reading unclassified_bytes alone
            # sees a nonzero figure whenever any code symbol in it failed
            # this test, whichever shape the failure took.
            #
            # T-2348 (Brunel), Poirot's M2 (8a28460-t2344 confirmation): the
            # O5 remedy above charged sec_size ONCE PER EMPTY EXTENT -- three
            # empty extents on one 64-byte section read unclassified_bytes
            # 192, three times the section's own real size. The section's
            # own attribution became unverifiable once; charge its size once,
            # however many empty extents that single failure decomposes
            # into.
            if not empty_extent_charged:
                unclassified += sec_size
                empty_extent_charged = True
            continue
        success, insns, stopped_at = _chain_decode(md, section.data, e_start, e_end)
        if not success:
            ok = False
            unclassified += (e_end - stopped_at)
            continue
        for name in ext.names:
            per_symbol_insns[name] = insns

    return ok, unclassified, per_symbol_insns


# ---------------------------------------------------------------------------
# Check (A): register-file classification, x86-64.
# ---------------------------------------------------------------------------

_X86_VEC_REG_RE = re.compile(r"\b(?:xmm|ymm|zmm|mm|tmm)\d+\b|\bk[0-7]\b|\bst\(?[0-7]?\)?\b")

_X86_VEC_MOVE_ALLOW = {
    # T-2343 (Brunel), Poirot's Critical C1 (78535ed-t2339 review): `movsd`
    # (scalar-double load/store, MSVC's ordinary 8-byte copy of a
    # pair<int,int>/double-sized value) was absent while its AVX form
    # `vmovsd` and its SSE sibling `movss` were both present -- 62 real
    # symbols across 6 real translation units, including two members of
    # Sec3.1's own must-accept commissioning population
    # (`FixedIntMap::_Emplace_back`/`_Uninitialized_fill_n`), REJECTed on
    # nothing but a data move. Design Sec4.1 names the category directly:
    # "movss/movaps/movdqa/movd/movq and their AVX v-prefixed equivalents" --
    # `movsd` is `movss`'s own scalar-double sibling, not a distinct case.
    # `movsldup`/`movshdup` (lane-DUPLICATION, not a pure copy) are
    # deliberately NOT added here -- they remain rejected as genuine
    # rearrangement, pinned as a named boundary control
    # (test_movsldup_movshdup_boundary_control).
    "movsd",
    "movss", "movaps", "movapd", "movups", "movupd", "movdqa", "movdqu",
    "movd", "movq", "movhlps", "movlhps", "movhps", "movlps", "movhpd", "movlpd",
    "vmovss", "vmovsd", "vmovaps", "vmovapd", "vmovups", "vmovupd",
    "vmovdqa", "vmovdqu", "vmovdqa32", "vmovdqa64", "vmovdqu8", "vmovdqu16",
    "vmovdqu32", "vmovdqu64", "vmovd", "vmovq", "vmovhps", "vmovlps",
    "vmovhpd", "vmovlpd",
    "unpcklps", "unpckhps", "unpcklpd", "unpckhpd",
    "vunpcklps", "vunpckhps", "vunpcklpd", "vunpckhpd",
    "shufps", "shufpd", "vshufps", "vshufpd",
    "pshufb", "pshufd", "pshuflw", "pshufhw",
    "vpshufb", "vpshufd", "vpshuflw", "vpshufhw",
    "punpcklbw", "punpckhbw", "punpcklwd", "punpckhwd",
    "punpckldq", "punpckhdq", "punpcklqdq", "punpckhqdq",
    "vpunpcklbw", "vpunpckhbw", "vpunpcklwd", "vpunpckhwd",
    "vpunpckldq", "vpunpckhdq", "vpunpcklqdq", "vpunpckhqdq",
    "palignr", "vpalignr",
    "pxor", "pand", "por", "pandn",
    "vpxor", "vpand", "vpor", "vpandn",
    "vzeroupper", "vzeroall", "vbroadcastss", "vbroadcastsd",
    "vpermilps", "vpermilpd", "vpermps", "vpermpd",
    "vpblendmps", "vpblendmpd", "vblendps", "vblendpd", "vpblendw",
    "psrldq", "pslldq",
    "movmskps", "movmskpd", "pmovmskb",
    "vmovmskps", "vmovmskpd", "vpmovmskb",
    # T-2381 (Brunel), design Sec4.1 fold round 40 (D-SLM5032/D-SLM5037), the
    # ELF/GCC leg's own third measured false reject: vextracti128/
    # vextracti64x4 (matmul.o, real GCC-built corpus) are pure lane-movement
    # -- a 128/256/512-bit lane select by immediate, no rounding, no
    # exception, no dependence on the operands' numeric value. The design's
    # own ruling widens this to the whole family, both suffixes ('i'/'f'
    # naming which register-file typing convention the toolchain's
    # instruction selector favored, never whether arithmetic occurred -- the
    # identical boundary D-SLM4987 already drew for orps/andps/xorps), every
    # AVX-512 width (128/32x4/64x2/64x4), and the vinsert-mnemonic
    # counterparts (the inverse lane-insert operation, the same movement
    # class). The genuinely arithmetic float family (addps/mulps/divps/
    # sqrtps/cvt*/comis*/fma* and their forms) is a distinct set of
    # mnemonics, unaffected by this addition and still rejected by this
    # function's own default.
    "vextracti128", "vextracti32x4", "vextracti64x2", "vextracti64x4",
    "vextractf128", "vextractf32x4", "vextractf64x2", "vextractf64x4",
    "vinserti128", "vinserti32x4", "vinserti64x2", "vinserti64x4",
    "vinsertf128", "vinsertf32x4", "vinsertf64x2", "vinsertf64x4",
}

# 3DNow int->float (genuinely arithmetic) plus the AVX-512 BF16 dot-product/
# convert family (T-2343, Brunel; design Sec4.1 fold round 34 gap (e),
# D-SLM4860) -- `vdpbf16ps` is the design's own printed name; `vpdpbf16ps` is
# the SAME instruction under the rendering that also matches the vp-prefix
# structural accept (conductor-measured: the two spellings classified
# oppositely before this fix), and it is the ONLY one of the four entries
# below that reaches the p/vp branch at all under the shipped decoder's own
# mnemonic vocabulary -- `vdpbf16ps`, `vcvtne2ps2bf16`, and `vcvtneps2bf16`
# start with neither `p` nor `vp`, so they already fall off the movement
# allowlist and REJECT via this function's own default at the bottom,
# whether or not they are named here (8a28460-t2344's own O2: removing any
# of the three from this set changes `_x86_check_a`'s answer for none of
# them, executed). All four are still named explicitly rather than pruned
# to the one load-bearing entry: this is a closed, individually-vetted
# family (`vcvtne2ps2bf16`/`vcvtneps2bf16` are format conversion, not
# dot-product, the family's other two members), and keeping every member
# listed documents the whole vetting decision in one place against a future
# capstone mnemonic-naming change, rather than only the entry today's
# decoder happens to need. None is emitted by any toolchain in the matrix
# without explicit AVX-512-BF16 intrinsics; none is vetted onto
# VEC_MOVE_ALLOW, since no legitimate use of BFloat16 arithmetic exists
# anywhere in SUPERSLM_CORE_SOURCES. `vpdpbusd`/`vpdpwssd` (VNNI, genuine
# packed-integer dot-product) are deliberately NOT excluded -- they stay
# accepted under the structural rule, per the named control in
# test_bf16_x86_rendering_pair.
_X86_P_PREFIX_EXCLUDE = {
    "pi2fd", "pi2fw",
    "vdpbf16ps", "vpdpbf16ps", "vcvtne2ps2bf16", "vcvtneps2bf16",
}

# D-SLM5155/D-SLM5156 (Dan, 2026-08-29): the fail-closed question the deny
# list above left open is ruled closed. This is the fail-closed replacement
# design Sec4.1 specifies: an explicit, checked-in allow-list, built by
# enumerating capstone's real x86-64 p/vp-prefixed vocabulary (the same
# 1523-mnemonic census this module's own test suite pins) and individually
# vetting each entry against ISA-reference IEEE-754 semantics (admitted iff
# the mnemonic's documented semantics perform no rounding, no exception, and
# produce no float-typed numeric result). Built the same way GPR_ALLOW's own
# fold-8 freeze was (D-SLM4374): a closed set computed once against a named,
# re-executable reference vocabulary, requiring an explicit, reviewed diff
# for any future addition -- an unknown future p/vp mnemonic REJECTs by
# default instead of being admitted by the old rule's own silence.
#
# This is the real vocabulary's p/vp-prefixed, non-`pf`-prefixed membership
# (excluding `_X86_VEC_MOVE_ALLOW`'s own named entries, which ACCEPT via
# that list and never reach this one) with the six documented FP leaks
# `_X86_P_PREFIX_EXCLUDE` names above removed -- every member here was
# already individually vetted as the deny list's own complement; none is a
# genuine IEEE-754 operation. `_X86_P_PREFIX_EXCLUDE`'s six entries are
# retained above as documentation of what was individually considered and
# excluded; they are no longer load-bearing as a gate -- this allow-list is
# now the sole authority for the p/vp branch below.
_X86_P_VP_STRUCTURAL_ALLOW = frozenset({
    "pabsb", "pabsd", "pabsw", "packssdw", "packsswb", "packusdw", "packuswb", "paddb",
    "paddd", "paddq", "paddsb", "paddsw", "paddusb", "paddusw", "paddw", "pause",
    "pavgb", "pavgusb", "pavgw", "pblendvb", "pblendw", "pclmulqdq", "pcmpeqb", "pcmpeqd",
    "pcmpeqq", "pcmpeqw", "pcmpestri", "pcmpestrm", "pcmpgtb", "pcmpgtd", "pcmpgtq", "pcmpgtw",
    "pcmpistri", "pcmpistrm", "pconfig", "pdep", "pext", "pextrb", "pextrd", "pextrq",
    "pextrw", "phaddd", "phaddsw", "phaddw", "phminposuw", "phsubd", "phsubsw", "phsubw",
    "pinsrb", "pinsrd", "pinsrq", "pinsrw", "pmaddubsw", "pmaddwd", "pmaxsb", "pmaxsd",
    "pmaxsw", "pmaxub", "pmaxud", "pmaxuw", "pminsb", "pminsd", "pminsw", "pminub",
    "pminud", "pminuw", "pmovsxbd", "pmovsxbq", "pmovsxbw", "pmovsxdq", "pmovsxwd", "pmovsxwq",
    "pmovzxbd", "pmovzxbq", "pmovzxbw", "pmovzxdq", "pmovzxwd", "pmovzxwq", "pmuldq", "pmulhrsw",
    "pmulhrw", "pmulhuw", "pmulhw", "pmulld", "pmullw", "pmuludq", "pop", "popal",
    "popaw", "popcnt", "popf", "popfd", "popfq", "prefetch", "prefetchnta", "prefetcht0",
    "prefetcht1", "prefetcht2", "prefetchw", "prefetchwt1", "psadbw", "pshufw", "psignb", "psignd",
    "psignw", "pslld", "psllq", "psllw", "psrad", "psraw", "psrld", "psrlq",
    "psrlw", "psubb", "psubd", "psubq", "psubsb", "psubsw", "psubusb", "psubusw",
    "psubw", "pswapd", "ptest", "ptwrite", "push", "pushal", "pushaw", "pushf",
    "pushfd", "pushfq", "vp4dpwssd", "vp4dpwssds", "vpabsb", "vpabsd", "vpabsq", "vpabsw",
    "vpackssdw", "vpacksswb", "vpackusdw", "vpackuswb", "vpaddb", "vpaddd", "vpaddq", "vpaddsb",
    "vpaddsw", "vpaddusb", "vpaddusw", "vpaddw", "vpandd", "vpandnd", "vpandnq", "vpandq",
    "vpavgb", "vpavgw", "vpblendd", "vpblendmb", "vpblendmd", "vpblendmq", "vpblendmw", "vpblendvb",
    "vpbroadcastb", "vpbroadcastd", "vpbroadcastmb2q", "vpbroadcastmw2d", "vpbroadcastq", "vpbroadcastw", "vpclmulqdq", "vpcmov",
    "vpcmp", "vpcmpb", "vpcmpd", "vpcmpeqb", "vpcmpeqd", "vpcmpeqq", "vpcmpeqw", "vpcmpestri",
    "vpcmpestrm", "vpcmpgtb", "vpcmpgtd", "vpcmpgtq", "vpcmpgtw", "vpcmpistri", "vpcmpistrm", "vpcmpq",
    "vpcmpub", "vpcmpud", "vpcmpuq", "vpcmpuw", "vpcmpw", "vpcom", "vpcomb", "vpcomd",
    "vpcompressb", "vpcompressd", "vpcompressq", "vpcompressw", "vpcomq", "vpcomub", "vpcomud", "vpcomuq",
    "vpcomuw", "vpcomw", "vpconflictd", "vpconflictq", "vpdpbusd", "vpdpbusds", "vpdpwssd", "vpdpwssds",
    "vperm2f128", "vperm2i128", "vpermb", "vpermd", "vpermi2b", "vpermi2d", "vpermi2pd", "vpermi2ps",
    "vpermi2q", "vpermi2w", "vpermil2pd", "vpermil2ps", "vpermq", "vpermt2b", "vpermt2d", "vpermt2pd",
    "vpermt2ps", "vpermt2q", "vpermt2w", "vpermw", "vpexpandb", "vpexpandd", "vpexpandq", "vpexpandw",
    "vpextrb", "vpextrd", "vpextrq", "vpextrw", "vpgatherdd", "vpgatherdq", "vpgatherqd", "vpgatherqq",
    "vphaddbd", "vphaddbq", "vphaddbw", "vphaddd", "vphadddq", "vphaddsw", "vphaddubd", "vphaddubq",
    "vphaddubw", "vphaddudq", "vphadduwd", "vphadduwq", "vphaddw", "vphaddwd", "vphaddwq", "vphminposuw",
    "vphsubbw", "vphsubd", "vphsubdq", "vphsubsw", "vphsubw", "vphsubwd", "vpinsrb", "vpinsrd",
    "vpinsrq", "vpinsrw", "vplzcntd", "vplzcntq", "vpmacsdd", "vpmacsdqh", "vpmacsdql", "vpmacssdd",
    "vpmacssdqh", "vpmacssdql", "vpmacsswd", "vpmacssww", "vpmacswd", "vpmacsww", "vpmadcsswd", "vpmadcswd",
    "vpmadd52huq", "vpmadd52luq", "vpmaddubsw", "vpmaddwd", "vpmaskmovd", "vpmaskmovq", "vpmaxsb", "vpmaxsd",
    "vpmaxsq", "vpmaxsw", "vpmaxub", "vpmaxud", "vpmaxuq", "vpmaxuw", "vpminsb", "vpminsd",
    "vpminsq", "vpminsw", "vpminub", "vpminud", "vpminuq", "vpminuw", "vpmovb2m", "vpmovd2m",
    "vpmovdb", "vpmovdw", "vpmovm2b", "vpmovm2d", "vpmovm2q", "vpmovm2w", "vpmovq2m", "vpmovqb",
    "vpmovqd", "vpmovqw", "vpmovsdb", "vpmovsdw", "vpmovsqb", "vpmovsqd", "vpmovsqw", "vpmovswb",
    "vpmovsxbd", "vpmovsxbq", "vpmovsxbw", "vpmovsxdq", "vpmovsxwd", "vpmovsxwq", "vpmovusdb", "vpmovusdw",
    "vpmovusqb", "vpmovusqd", "vpmovusqw", "vpmovuswb", "vpmovw2m", "vpmovwb", "vpmovzxbd", "vpmovzxbq",
    "vpmovzxbw", "vpmovzxdq", "vpmovzxwd", "vpmovzxwq", "vpmuldq", "vpmulhrsw", "vpmulhuw", "vpmulhw",
    "vpmulld", "vpmullq", "vpmullw", "vpmultishiftqb", "vpmuludq", "vpopcntb", "vpopcntd", "vpopcntq",
    "vpopcntw", "vpord", "vporq", "vpperm", "vprold", "vprolq", "vprolvd", "vprolvq",
    "vprord", "vprorq", "vprorvd", "vprorvq", "vprotb", "vprotd", "vprotq", "vprotw",
    "vpsadbw", "vpscatterdd", "vpscatterdq", "vpscatterqd", "vpscatterqq", "vpshab", "vpshad", "vpshaq",
    "vpshaw", "vpshlb", "vpshld", "vpshldd", "vpshldq", "vpshldvd", "vpshldvq", "vpshldvw",
    "vpshldw", "vpshlq", "vpshlw", "vpshrdd", "vpshrdq", "vpshrdvd", "vpshrdvq", "vpshrdvw",
    "vpshrdw", "vpshufbitqmb", "vpsignb", "vpsignd", "vpsignw", "vpslld", "vpslldq", "vpsllq",
    "vpsllvd", "vpsllvq", "vpsllvw", "vpsllw", "vpsrad", "vpsraq", "vpsravd", "vpsravq",
    "vpsravw", "vpsraw", "vpsrld", "vpsrldq", "vpsrlq", "vpsrlvd", "vpsrlvq", "vpsrlvw",
    "vpsrlw", "vpsubb", "vpsubd", "vpsubq", "vpsubsb", "vpsubsw", "vpsubusb", "vpsubusw",
    "vpsubw", "vpternlogd", "vpternlogq", "vptest", "vptestmb", "vptestmd", "vptestmq", "vptestmw",
    "vptestnmb", "vptestnmd", "vptestnmq", "vptestnmw", "vpxord", "vpxorq",
})

# T-2367 (Brunel), design Sec4.1 fold round 39 (D-SLM4987). AND, OR, ANDN, and
# XOR are Boolean functions of their operand bits under every x86 encoding --
# packed-integer (`p`-prefix, already accepted unconditionally above) or
# packed-single/-double (`ps`/`pd`-suffix) -- with no rounding, no exception,
# and no vendor- or microarchitecture-dependent behavior possible for a pure
# bitwise operation. The design's own pre-fold-39 text already named the
# distinct-operand case "a genuine bitwise operation" while still rejecting
# it (self-zeroing-only carve-out, retired below): the correct classification
# paired with the wrong verdict. All eight base mnemonics and their VEX forms
# perform no floating-point arithmetic and are accepted unconditionally, on
# any operand list -- joining pand/por/pandn/pxor on the same footing. The
# genuinely arithmetic float instructions (addps/mulps/divps/sqrtps/cvt*/
# comis*/fma* and their pd/VEX forms) are distinct mnemonics, unaffected by
# this rule and still rejected by this function's own default.
_X86_BITWISE_FP_FAMILY = {
    "orps", "orpd", "andps", "andpd", "andnps", "andnpd", "xorps", "xorpd",
    "vorps", "vorpd", "vandps", "vandpd", "vandnps", "vandnpd", "vxorps", "vxorpd",
}


def _x86_touches_vector_register(op_str: str) -> bool:
    return bool(_X86_VEC_REG_RE.search(op_str or ""))


def _x86_check_a(mnemonic: str, op_str: str) -> Optional[str]:
    """Non-None (a reason string) == ACCEPT under check (A); the returned
    string names the admitting branch. None == REJECT. Only called for
    instructions that touch a vector/FP/mask/tile register at all.

    D-SLM5157 (R4, T-2404): the contract was `bool`; every acceptance
    reported the literal True regardless of which rule admitted it, so a
    census over check (A)'s own acceptances could not distinguish an
    accept attributed to a named allow-list from one that reached no
    named rule at all. Every admitting branch below now returns its own
    name; callers that only need ACCEPT/REJECT keep working unchanged
    (`not check_a(...)` is True for both `None` and the pre-existing
    `False`, and every non-empty reason string is truthy)."""
    m = mnemonic.lower()
    if m in _X86_VEC_MOVE_ALLOW:
        return "vec_move_allow"
    if m in _X86_P_VP_STRUCTURAL_ALLOW:
        # D-SLM5156: fail-closed allow-list membership, not the deny-list-
        # guarded "starts with p/vp and isn't excluded" rule it replaces --
        # an mnemonic absent from this list REJECTs by default, whether or
        # not it happens to start with p/vp.
        return "p_vp_structural_allow"
    if m in _X86_BITWISE_FP_FAMILY:
        # T-2367 (Brunel), D-SLM4987: accepted unconditionally, on any
        # operand list. Supersedes the pre-fold-39 self-zeroing-only
        # carve-out for xorps/xorpd/vxorps/vxorpd (every operand identical),
        # which is retired as an intermediate, historically-cautious
        # approximation of the rule stated here -- not left standing
        # alongside it. Population sixteen's own must-reject construction
        # for a differing-operand vxorps is reconciled to must-accept by the
        # same fold (D-SLM5002).
        return "bitwise_fp_family"
    return None


# ---------------------------------------------------------------------------
# Check (A): register-file classification, AArch64.
# ---------------------------------------------------------------------------

_AARCH64_VEC_REG_RE = re.compile(r"\b[bhsdq]\d+\b|\bv\d+(?:\.\w+)?\b")
_AARCH64_NAMED_CONVERSIONS = {"scvtf", "ucvtf", "fcvtzs", "fcvtzu", "fjcvtzs"}
# BFloat16 dot-product/multiply-accumulate/convert family (T-2343, Brunel;
# design Sec4.1 fold round 34 gap (e), D-SLM4860; Popper's D-SLM4850,
# commissioned dead through the real production route -- three real
# arm_neon.h intrinsic bodies, clang --target=aarch64-linux-gnu
# -march=armv8.6-a+bf16, ACCEPTed). None carries the leading `f` the
# structural rule keys on (it carries `bf`, indistinguishable from a
# `b`-prefixed integer mnemonic like `bic` by that test alone) and none is
# one of the three named domain-crossing conversions, so the rule as
# printed accepted it structurally; every one is genuine BFloat16
# floating-point arithmetic and none is vetted onto VEC_MOVE_ALLOW, since no
# legitimate use of BFloat16 arithmetic exists anywhere in
# SUPERSLM_CORE_SOURCES.
_AARCH64_BF16_EXCLUDE = {"bfdot", "bfmmla", "bfmlalb", "bfmlalt", "bfcvt", "bfcvtn", "bfcvtn2"}
_AARCH64_VEC_MOVE_ALLOW = {
    "fmov", "mov", "movi",
    "ldr", "str", "ldur", "stur", "ldp", "stp",
    "ldrb", "strb", "ldrh", "strh",
    "ldursb", "ldursh", "ldursw", "ldrsb", "ldrsh", "ldrsw",
}


def _aarch64_touches_vector_register(op_str: str) -> bool:
    return bool(_AARCH64_VEC_REG_RE.search(op_str or ""))


def _aarch64_check_a(mnemonic: str, op_str: str) -> bool:
    """True == ACCEPT under check (A). T-2343 (Brunel), Popper's D-SLM4850:
    the prior form of this function ended `return m in _AARCH64_VEC_MOVE_ALLOW
    or True`, which is the constant True -- default-ALLOW, the inverse of the
    default-deny property this whole design rests on. A real BFloat16 ELF
    object (bfdot/bfmmla) ACCEPTed through the production route. Rewritten to
    actually consult the membership test, with the BF16 family excluded by
    name from the non-f-prefix structural accept, mirroring x86's own
    pi2fd/pi2fw/BF16 exclusions."""
    m = mnemonic.lower()
    if m in _AARCH64_NAMED_CONVERSIONS:
        return False
    if m.startswith("f"):
        return m in _AARCH64_VEC_MOVE_ALLOW
    if m in _AARCH64_BF16_EXCLUDE:
        return False
    # T-2343 (Brunel), Poirot's M1: this is a STRUCTURAL accept, not a
    # membership test -- a mnemonic reaching this line touches a
    # vector/FP register family, is not `f`-prefixed, is not one of the
    # three named conversions, and is not the BF16 family, which makes it
    # NEON packed-integer arithmetic/logic/compare/shift/permute BY THE
    # ISA'S OWN NAMING CONVENTION (design Sec4.1's own AArch64 text: "a
    # mnemonic touching one of these register families with no f prefix and
    # not one of the three named conversions is NEON packed-integer... and
    # is accepted structurally, with no mnemonic added to any list"). The
    # prior form of this line, `m in _AARCH64_VEC_MOVE_ALLOW or True`, was
    # the constant True wearing a membership test that was never consulted
    # on this branch -- write the structural accept as what it is.
    return True


# ---------------------------------------------------------------------------
# Check (B): GPR/control-flow mnemonic allowlist, pinned per ISA.
# ---------------------------------------------------------------------------

# T-2343 (Brunel), Poirot's S6: this set (plus _X86_JCC below) is 185
# entries against design Sec4.1's own stated "pinned, 159-entry" production
# GPR_ALLOW -- the design prints that count with only a partial
# enumeration, which Poirot's own review states plainly: "an implementer
# cannot reproduce it exactly from the text... the finding is the absent
# reconciliation, not any particular entry." Audited this session rather
# than silently re-affirmed: no entry here is a floating-point mnemonic, and
# the two names that collide with SSE mnemonics by spelling (`movsd`,
# `cmpsd` -- both real x86 string-operation names AND real SSE2 scalar-
# double mnemonics) are safe by construction regardless of this set's own
# membership, because `scan_object`'s own dispatch routes an instruction to
# check (A), never check (B), whenever it touches a vector register at all
# (`touches_vec`, above) -- these two names reach check (B) only on their
# genuine string-operation reading. This module's own actual count is 185,
# stated here as the citable fact the design's own text does not yet state;
# reconciling the design's own printed 159 against this module's 185 (a
# line-by-line diff with a vetting note per addition) is spec-side work
# this build does not perform -- filed as a residual, not silently closed
# by this comment.
_X86_GPR_ALLOW = {
    "mov", "movzx", "movsx", "movsxd", "movabs", "lea",
    "add", "adc", "sub", "sbb", "inc", "dec", "neg", "not",
    "imul", "mul", "idiv", "div",
    "and", "or", "xor", "shl", "shr", "sal", "sar", "rol", "ror", "rcl", "rcr",
    "bt", "bts", "btr", "btc", "bsf", "bsr", "popcnt", "lzcnt", "tzcnt",
    "andn", "bzhi", "pdep", "pext", "shrx", "shlx", "sarx",
    # T-2381 (Brunel), design Sec4.1 fold round 40 (D-SLM5032/D-SLM5037), the
    # ELF/GCC leg's own second measured false reject: shrd/shld
    # (intmath.o/forward_sites.o, real GCC-built corpus) are ordinary
    # integer double-precision shifts across a register pair -- "double"
    # names the operand width, not a float type -- the identical semantic
    # class as shl/shr/sar/rol/ror, already on this allow-list.
    "shrd", "shld",
    "cmp", "test",
    "jmp",
    "call", "ret", "retn", "retf",
    "push", "pop", "pushf", "pushfq", "popf", "popfq", "enter", "leave",
    "movsb", "movsw", "movsd", "movsq", "stosb", "stosw", "stosd", "stosq",
    "lodsb", "lodsw", "lodsd", "lodsq", "scasb", "scasw", "scasd", "scasq",
    "cmpsb", "cmpsw", "cmpsd", "cmpsq",
    "xadd", "xchg", "cmpxchg", "cmpxchg8b", "cmpxchg16b",
    "nop", "int3", "int", "ud2", "hlt", "pause",
    "mfence", "lfence", "sfence",
    "cpuid", "rdtsc", "rdtscp", "xgetbv", "endbr64",
    "cdq", "cqo", "cdqe", "cwde", "cbw", "cwd",
    "syscall", "sysenter",
    "rep", "repe", "repz", "repne", "repnz", "lock",
    "vzeroupper", "vzeroall",
    "seta", "setae", "setb", "setbe", "sete", "setne", "setg", "setge",
    "setl", "setle", "sets", "setns", "seto", "setno", "setp", "setnp",
    "cmova", "cmovae", "cmovb", "cmovbe", "cmove", "cmovne", "cmovg", "cmovge",
    "cmovl", "cmovle", "cmovs", "cmovns", "cmovo", "cmovno", "cmovp", "cmovnp",
}
# jcc family, generated rather than hand-enumerated (a mnemonic-shape closed
# under the ISA's own condition-code vocabulary, unlike an open enumeration):
_X86_JCC = {
    "ja", "jae", "jb", "jbe", "jc", "jcxz", "jecxz", "jrcxz", "je", "jg",
    "jge", "jl", "jle", "jna", "jnae", "jnb", "jnbe", "jnc", "jne", "jng",
    "jnge", "jnl", "jnle", "jno", "jnp", "jns", "jnz", "jo", "jp", "jpe",
    "jpo", "js", "jz", "loop", "loope", "loopne", "loopnz", "loopz",
}
_X86_GPR_ALLOW |= _X86_JCC


def _x86_strip_prefix(mnemonic: str) -> str:
    m = mnemonic.lower()
    # T-2381 (Brunel), design Sec4.1 fold round 40 (D-SLM5032/D-SLM5037), the
    # ELF/GCC leg's own first measured false reject: "notrack " is GCC's own
    # CET indirect-branch-tracking prefix (-fcf-protection's default on
    # Ubuntu), ahead of an ordinary indirect jmp/call already on
    # _X86_GPR_ALLOW -- stripped here on the same footing as the other six
    # named prefixes below, never a general "strip everything before the
    # first space" rule (design Sec7 dim 11's forty-first population, D-
    # SLM5058/D-SLM5070: that looser shape is the specification's own
    # excluded control, which must still REJECT a fabricated two-token
    # mnemonic this narrow, literal-tuple strip does not touch).
    for pfx in ("rep ", "repe ", "repz ", "repne ", "repnz ", "lock ", "notrack "):
        if m.startswith(pfx):
            return m[len(pfx):]
    return m


def _x86_check_b(mnemonic: str) -> bool:
    return _x86_strip_prefix(mnemonic) in _X86_GPR_ALLOW


_AARCH64_GPR_ALLOW = {
    "add", "adds", "sub", "subs", "adc", "adcs", "sbc", "sbcs",
    "and", "ands", "orr", "orn", "eor", "eon", "bic", "bics", "mvn",
    "asr", "lsl", "lsr", "ror",
    "adrp", "adr",
    "cmp", "cmn", "tst",
    "b", "bl", "blr", "br", "ret",
    "cbz", "cbnz", "tbz", "tbnz",
    "csel", "cset", "csetm", "csinc", "csinv", "csneg", "cinc", "cinv", "cneg",
    "ccmp", "ccmn",
    "clz", "cls", "rbit", "rev", "rev16", "rev32", "rev64",
    "extr", "bfi", "bfxil", "sbfx", "ubfx", "sbfiz", "ubfiz",
    "sxtb", "sxth", "sxtw", "uxtb", "uxth", "uxtw",
    "ldr", "str", "ldur", "stur", "ldp", "stp",
    "ldrb", "strb", "ldrh", "strh",
    "ldrsb", "ldrsh", "ldrsw", "ldursb", "ldursh", "ldursw",
    "ldaxr", "stlxr", "ldxr", "stxr", "ldar", "stlr",
    "mov", "movz", "movn", "movk",
    "mul", "madd", "msub", "mneg", "sdiv", "udiv",
    "umull", "smull", "umulh", "smulh",
    "dmb", "dsb", "isb", "svc", "hlt", "brk", "nop", "yield",
    "wfe", "wfi", "sev", "sevl",
    "prfh", "prfm", "prfw",
}


def _aarch64_check_b(mnemonic: str) -> bool:
    m = mnemonic.lower()
    if "." in m:
        base, _, _cond = m.partition(".")
        if base in ("b", "csel", "cset"):
            return True
    return m in _AARCH64_GPR_ALLOW


# ---------------------------------------------------------------------------
# Check (C): call/tail-jmp target default-deny.
# ---------------------------------------------------------------------------

_X86_EXTERN_ALLOW = {
    # The eight remaining unvetted CRT/EH import targets the real 17-TU corpus
    # reaches through a relocation at a call/jmp. Measured (T-2350 counterfactual,
    # reproduced by the conductor 2026-08-27): vetting exactly these takes the real
    # corpus from 630 REJECT to 8. None performs floating-point arithmetic on the
    # caller's behalf; each is a control-transfer target, not a computation.
    "__imp__invalid_parameter_noinfo_noreturn", "__imp_abort",
    "?_Throw_C_error@std@@YAXH@Z", "??_M@YAXPEAX_K1P6AX0@Z@Z",
    "?_Xout_of_range@std@@YAXPEBD@Z", "__imp___stdio_common_vsprintf",
    "__imp_strncmp", "??_L@YAXPEAX_K1P6AX0@Z2@Z",
    "memmove", "memcpy", "memset", "memcmp", "memchr",
    "malloc", "free", "calloc", "realloc",
    "??2@YAPEAX_K@Z", "??_U@YAPEAX_K@Z",
    "??2@YAPEAX_KAEBUnothrow_t@std@@@Z", "??_U@YAPEAX_KAEBUnothrow_t@std@@@Z",
    "??3@YAXPEAX_K@Z", "??3@YAXPEAX@Z", "??_V@YAXPEAX@Z", "??_V@YAXPEAX_K@Z",
    "??3@YAXPEAXAEBUnothrow_t@std@@@Z", "??_V@YAXPEAXAEBUnothrow_t@std@@@Z",
    "?_Xlength_error@std@@YAXPEBD@Z",
    "?_Throw_bad_array_new_length@std@@YAXXZ",
    "__security_check_cookie",
    "_Mtx_lock", "_Mtx_unlock", "_Mtx_init", "_Mtx_destroy",
    "_Mtx_lock_in_situ", "_Mtx_unlock_in_situ", "_Mtx_init_in_situ", "_Mtx_destroy_in_situ",
    "_Throw_C_error", "_CxxThrowException",
    "_wassert", "abort", "terminate", "?terminate@@YAXXZ",
    "__std_exception_copy", "__std_exception_destroy",
    "fgetc", "fputc", "ungetc", "strncmp", "atexit",
    "_Init_thread_header", "_Init_thread_footer", "_Init_thread_epoch",
    "__stdio_common_vsprintf",
    "_invalid_parameter_noinfo_noreturn",
}

_ELF_EXTERN_ALLOW = {
    "memmove", "memcpy", "memset", "memcmp", "memchr",
    "malloc", "free", "calloc", "realloc",
    "__cxa_throw", "__cxa_allocate_exception", "__cxa_free_exception",
    "__cxa_begin_catch", "__cxa_end_catch", "_Unwind_Resume",
    "abort", "terminate",
}


def _extern_allow_for(object_format: str) -> set:
    return _COFF_EXTERN_ALLOW_SET if object_format == "coff" else _ELF_EXTERN_ALLOW


_COFF_EXTERN_ALLOW_SET = _X86_EXTERN_ALLOW


def _is_x86_call_or_jmp(mnemonic: str) -> str | None:
    m = mnemonic.lower()
    if m == "call":
        return "call"
    if m == "jmp":
        return "jmp"
    return None


def _is_aarch64_call_or_jmp(mnemonic: str) -> str | None:
    m = mnemonic.lower()
    if m in ("bl", "blr"):
        return "call"
    if m == "b":
        return "jmp"
    if m in ("br",):
        return "jmp"
    return None


def _operand_is_direct_immediate(insn, isa: str) -> bool:
    if not insn.operands:
        return False
    op = insn.operands[0]
    if isa == "aarch64":
        return op.type == capstone.arm64_const.ARM64_OP_IMM
    return op.type == capstone.x86_const.X86_OP_IMM


def _check_c_for_symbol(md, section: _CodeSection, ext_start: int, ext_end: int,
                        insns, isa: str, object_format: str,
                        reloc_by_offset: Mapping[int, int],
                        sym_by_raw: Mapping[int, dict],
                        local_starts: set,
                        corpus_symbols: frozenset | None = None) -> bool:
    """True == every call/tail-jmp edge in this extent's own instructions
    passes check (C). `local_starts` is the set of byte offsets, WITHIN THIS
    SECTION ONLY (T-2343, Brunel, Poirot's O3 -- a pooled cross-section set
    compared against a section-relative operand address was a latent
    correctness bug, 0 live impact measured because MSVC relocates every
    real cross-function edge, but wrong regardless), where a real symbol
    begins -- used to recognise an intra-object direct edge that carries no
    relocation because the assembler could resolve it directly (same-section
    target). `corpus_symbols`, when supplied (design Sec4.1 fold round 34
    gap (b), D-SLM4857), is the full set of function-typed symbol names
    defined anywhere in the closed corpus; a relocated edge to a name
    undefined in THIS object but present in `corpus_symbols` is accepted as
    a genuine first-party in-corpus reference rather than forced through
    EXTERN_ALLOW."""
    extern_allow = _extern_allow_for(object_format)
    is_call_or_jmp = _is_x86_call_or_jmp if isa == "x86-64" else _is_aarch64_call_or_jmp

    for insn in insns:
        kind = is_call_or_jmp(insn.mnemonic)
        if kind is None:
            continue

        # A relocation on this instruction's own bytes means the target was
        # NOT resolved at assembly time -- it is necessarily a real edge
        # (in-object, cross-section, or external), and the raw immediate
        # capstone decodes is a meaningless pre-link placeholder (typically
        # zero), so the relocation is checked FIRST and unconditionally,
        # before any "does the immediate look intra-function" shortcut --
        # a placeholder displacement of zero makes an unresolved tail-jmp's
        # computed target equal to "right after this instruction," which
        # would otherwise satisfy an intra-extent test by construction on
        # almost every such edge (found by direct execution against a real
        # tail-jmp-to-libm construction, this session).
        reloc_off = None
        for cand in range(insn.address, insn.address + insn.size):
            if cand in reloc_by_offset:
                reloc_off = cand
                break

        if reloc_off is not None:
            # T-2348 (Brunel), design Sec4.1 fold round 35 (D-SLM4886),
            # closing Poirot's Critical C1: resolve the relocation, THEN
            # classify -- a relocation's presence says the TARGET SYMBOL is
            # known; the fold-34 remedy (T-2343, Poirot's S2) applied the
            # addressing-mode test before ever reading that target, which
            # rejected every relocated memory-operand edge uniformly,
            # including MSVC's own `call qword ptr [__imp_<name>]`
            # dllimport rendering -- whose relocation names the exact
            # callee, a static fact, not a runtime-determined value. The
            # addressing-mode test is reserved for the one case it is
            # actually about: an in-object relocation target that is NOT
            # itself a callable entity (a data symbol whose runtime
            # CONTENTS, not its address, determine which function actually
            # runs -- the original S2 construction, `jmp QWORD PTR [gp]`,
            # `gp` a QWORD in .data).
            raw_idx = reloc_by_offset[reloc_off]
            target_sym = sym_by_raw.get(raw_idx)
            if target_sym is None:
                return False
            is_external = (object_format == "coff" and target_sym.get("sec_num", -1) == 0) or \
                           (object_format == "elf" and target_sym.get("shndx", -1) == 0)
            if is_external:
                # Vetted by the symbol that relocation names, exactly as
                # every other external/in-corpus edge -- regardless of this
                # instruction's own addressing-mode encoding, since the
                # relocation already resolved WHICH callee this is.
                if corpus_symbols is not None and target_sym["name"] in corpus_symbols:
                    continue  # in-corpus target (a sibling TU): the design's own membership rule already covers it
                if target_sym["name"] not in extern_allow:
                    return False
                continue
            # In-object target. If it is itself a function symbol -- the
            # callable entity -- it receives its own independent verdict
            # wherever its own extent is scanned in this same object, so
            # accepting the edge here does not excuse it; the addressing-
            # mode encoding is irrelevant to that case. Only when the
            # relocation names something that is NOT itself callable (a
            # data symbol holding a pointer) does the addressing-mode test
            # apply, and only then does a non-immediate encoding REJECT as
            # genuinely unvettable.
            if target_sym.get("is_function"):
                continue
            if not _operand_is_direct_immediate(insn, isa):
                return False
            continue

        # No relocation: the target, if a direct immediate, was resolved by
        # the assembler itself -- only possible for an intra-object address.
        if not _operand_is_direct_immediate(insn, isa):
            return False  # indirect call/jmp: cannot statically vet
        target = insn.operands[0].imm
        if kind == "jmp" and ext_start <= target < ext_end:
            continue  # ordinary intra-function control flow, not an edge
        if target in local_starts:
            continue  # in-object edge the assembler resolved directly
        return False  # a direct target with no relocation and no known symbol

    return True


# ---------------------------------------------------------------------------
# scan_object -- the production entry point.
# ---------------------------------------------------------------------------


def scan_object(path: str, isa: str,
                corpus_symbols: frozenset | None = None,
                data: bytes | None = None) -> ScanResult:
    """`corpus_symbols` (design Sec4.1 fold round 34 gap (b), D-SLM4857): the
    full set of function-typed symbol NAMES defined anywhere in the closed
    corpus (every object `enumerate_scan_targets()` names). When supplied, a
    check-(C) edge whose target is undefined in THIS object but whose name
    IS a member of `corpus_symbols` is accepted as a genuine first-party
    in-corpus reference rather than forced through `EXTERN_ALLOW` -- the
    target is itself a member of the scanned corpus and receives its own
    independent verdict wherever its own translation unit is scanned, so
    accepting the edge here does not excuse the target from its own bytes
    being checked. `None` (the default) preserves prior behavior exactly,
    for a call outside the full-corpus driver (an isolated must-accept/
    must-reject construction has no corpus to index).

    `data` (T-2381, Brunel, design Sec4.1's archive member-iterator contract,
    D-SLM5056/D-SLM5071): when not None, parsing reads `data` directly -- the
    identical `_read_object_format`/`_parse_coff`/`_parse_elf` code path a
    file-based call already exercises -- and `path` is used only for error
    messages and the returned `ScanResult`'s own identification, never
    opened or read. When `data` is None (unchanged from every pre-fold-41
    caller), `path` is opened and read as before. This is additive over the
    fold-34 signature: `corpus_symbols` keeps its fold-34 position (third),
    unaffected by `data`'s own addition after it -- a caller binding a third
    positional argument keeps its pre-fold-41 behavior (D-SLM5071 exists
    precisely because a naive "additive" patch deleted this parameter)."""
    try:
        raw = data
        if raw is None:
            with open(path, "rb") as f:
                raw = f.read()
        object_format = _read_object_format(raw)
    except ValueError:
        # T-2343 (Brunel), 78535ed-t2339's own M2: design Sec4.1's own
        # contract states "a leg whose declared isa has no decoder mode
        # REFUSES via clause (0) rather than guessing at the bytes" -- an
        # unrecognised object format is exactly that shape, and previously
        # raised instead of returning a ScanResult, so it never reached
        # ci_gate at all (an uncaught exception inside a caller's own
        # try/loop silently skips the leg rather than failing the job). The
        # header itself could not be read here, so "unknown" is the real
        # fact about the format, not a fallback value.
        return ScanResult(object_format="unknown", refuse=True,
                          unclassified_bytes=0, verdicts={})

    try:
        md = _decoder(isa)
    except (ValueError, capstone.CsError):
        # T-2348 (Brunel), 8a28460-t2344's own M5: the fold-34 remedy above
        # covered `_read_object_format` and `_decoder` with one shared
        # `except`, so an unsupported ISA on an otherwise perfectly readable
        # object also reported object_format="unknown" -- a second false
        # fact riding along with the true REFUSE. The header already parsed
        # cleanly by this point; carry the real, confirmed format into the
        # refusal instead of overwriting it.
        return ScanResult(object_format=object_format, refuse=True,
                          unclassified_bytes=0, verdicts={})

    if object_format == "elf":
        code_sections, sym_by_raw, relocs_by_section = _parse_elf(raw)
    else:
        code_sections, sym_by_raw, relocs_by_section = _parse_coff(raw)

    refuse = False
    unclassified_total = 0
    per_symbol_insns_by_section = []  # list[(section, per_symbol_insns, extents)]

    for section in code_sections:
        ok, unclassified, per_symbol_insns = _account_section(section, isa, md)
        unclassified_total += unclassified
        if not ok:
            refuse = True
        else:
            extents = _compute_extents(section)
            per_symbol_insns_by_section.append((section, per_symbol_insns, extents))

    if refuse:
        return ScanResult(object_format=object_format, refuse=True,
                          unclassified_bytes=unclassified_total, verdicts={})

    # local_starts_by_section: every real symbol start offset, keyed by ITS
    # OWN section (T-2343, Brunel, Poirot's O3) -- used by check (C) to
    # recognise an intra-object direct edge the assembler resolved without a
    # relocation. A single pooled set across every section was compared
    # against `insn.operands[0].imm`, which `_chain_decode` produces as an
    # offset relative to the CURRENT section only (each section is decoded
    # independently, never concatenated) -- an offset from one section could
    # coincidentally match a symbol start in a different section, a latent
    # cross-section confusion (0 live impact measured: MSVC relocates every
    # real cross-function edge, so this path was never exercised on the real
    # corpus) that a per-section keying closes structurally.
    local_starts_by_section: dict = {}
    for section, _per_symbol_insns, extents in per_symbol_insns_by_section:
        local_starts_by_section[section.index] = {ext.start for ext in extents}

    touches_vec = _x86_touches_vector_register if isa == "x86-64" else _aarch64_touches_vector_register
    check_a = _x86_check_a if isa == "x86-64" else _aarch64_check_a
    check_b = _x86_check_b if isa == "x86-64" else _aarch64_check_b

    verdicts: dict = {}
    ab_verdicts: dict = {}
    for section, per_symbol_insns, extents in per_symbol_insns_by_section:
        reloc_by_offset = {r.offset: r.sym_raw_index for r in relocs_by_section.get(section.index, [])}
        local_starts = local_starts_by_section[section.index]
        for ext in extents:
            insns = per_symbol_insns.get(ext.names[0] if ext.names else None, [])
            ab_accept = True
            for insn in insns:
                op_str = insn.op_str or ""
                if touches_vec(op_str):
                    if not check_a(insn.mnemonic, op_str):
                        ab_accept = False
                        break
                else:
                    if not check_b(insn.mnemonic):
                        ab_accept = False
                        break
            c_accept = True
            if ab_accept:
                c_accept = _check_c_for_symbol(
                    md, section, ext.start, ext.end, insns, isa, object_format,
                    reloc_by_offset, sym_by_raw, local_starts,
                    corpus_symbols=corpus_symbols,
                )
            verdict = "ACCEPT" if (ab_accept and c_accept) else "REJECT"
            ab_verdict = "ACCEPT" if ab_accept else "REJECT"
            for name in ext.names:
                verdicts[name] = verdict
                ab_verdicts[name] = ab_verdict

    return ScanResult(object_format=object_format, refuse=False,
                      unclassified_bytes=0, verdicts=verdicts,
                      ab_verdicts=ab_verdicts)


# ---------------------------------------------------------------------------
# ci_gate -- the three-independent-check enforcement contract.
# ---------------------------------------------------------------------------


def ci_gate(result: ScanResult, expected_symbols: Sequence[str]) -> bool:
    if result.refuse:
        # Guarantee (i) is a structural property this result is expected to
        # hold (verdicts == {}), never itself a reason to pass -- a REFUSE
        # leg fails the job unconditionally.
        return False
    for name in expected_symbols:
        if name not in result.verdicts:
            return False  # guarantee (iii): the absent-report leg
    return True


def ci_gate_corpus(results: Mapping[str, ScanResult],
                   expected_symbols: Mapping[str, Sequence[str]],
                   manifest_path: str = None,
                   build_dir: str = None) -> bool:
    """The production CI gate (design Sec4.1 fold round 34 gap (a),
    D-SLM4856; Sec5.5's own three-way ship-gate disjunction): an aggregate
    over the WHOLE corpus. Returns True iff, for every object path in
    `expected_symbols`: `ci_gate(results[path], expected_symbols[path])` is
    True, AND every value in `results[path].verdicts` is "ACCEPT". A single
    REJECT anywhere, on any object, makes the whole call return False --
    additive to `ci_gate`'s own three guarantees, never a replacement:
    `ci_gate` still decides whether a leg is well-formed (refuses cleanly,
    or reports every expected symbol); this function's own fourth condition
    decides whether what every leg reported is ACCEPTABLE.

    Before this function existed, nothing in this design or in the tree
    converted a REJECT verdict into a failed job (Popper's D-SLM4851,
    commissioned DEAD: `ci_gate` returned True on a must-reject construction
    at every magnitude tested, because neither it nor anything upstream of
    it ever read a verdict's VALUE).

    Corrected at fold round 35 (D-SLM4888, closing Popper's own t2345
    Sec5.7 KILL): the contract above is vacuously satisfiable. "For every
    object path in `expected_symbols`" is true of every object in an empty
    map, so `ci_gate_corpus({}, {})` returned True, as did a map that
    listed every object except the one carrying a REJECT (reachable
    end-to-end from the shipped driver through an ordinary edit to a real
    CMakeLists.txt -- `derive_core_sources`'s own zero-source failure mode,
    corrected above, used to produce exactly this map). `expected_symbols`
    is not entitled to be read as authoritative on its own: it must be
    non-empty, and it must name EXACTLY the same object paths `results`
    does -- a caller can no longer avoid a REJECT by supplying a map
    smaller than the corpus it actually scanned, nor pass two empty maps
    and read the vacuous True as a clean corpus.

    When `manifest_path` and/or `build_dir` is supplied (the production
    driver's own real corpus run always supplies its own `build_dir`),
    this additionally re-derives the corpus's OWN current membership via
    `enumerate_scan_targets()` against that same manifest/build_dir and
    requires `expected_symbols.keys()` to be exactly that independently
    re-derived set too -- closing the gap that comparing `results` against
    `expected_symbols` alone cannot: both could agree by omission if an
    object silently never made it into either map. `None` (the default)
    skips this second, stronger check for an isolated must-accept/
    must-reject construction with no real manifest or build directory to
    re-derive against, mirroring `scan_object`'s own `corpus_symbols=None`
    convention for the identical reason."""
    if not expected_symbols or set(expected_symbols.keys()) != set(results.keys()):
        return False
    if manifest_path is not None or build_dir is not None:
        derived_paths = {obj for _src, obj in
                         enumerate_scan_targets(manifest_path=manifest_path, build_dir=build_dir)}
        if not derived_paths or set(expected_symbols.keys()) != derived_paths:
            return False
    for path, expected in expected_symbols.items():
        result = results.get(path)
        if result is None or not ci_gate(result, expected):
            return False
        if any(v != "ACCEPT" for v in result.verdicts.values()):
            return False
    return True


# ---------------------------------------------------------------------------
# enumerate_scan_targets / derive_core_sources -- production membership,
# derived from SUPERSLM_CORE_SOURCES, never a second hand-maintained list.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_DEFAULT_MANIFEST = os.path.join(_REPO_ROOT, "CMakeLists.txt")

_CORE_SOURCES_RE = re.compile(r"set\(\s*SUPERSLM_CORE_SOURCES(.*?)\)", re.DOTALL)


class CoreSourcesDerivationError(ValueError):
    """Raised by derive_core_sources() when the manifest names
    SUPERSLM_CORE_SOURCES but the parse cannot positively confirm a
    complete, correct capture (design Sec4.1 fold round 35, D-SLM4887) --
    mirroring enumerate_scan_targets()'s own DuplicateStemError
    convention."""


def derive_core_sources(manifest_path: str = None) -> list:
    """Corrected at fold round 35 (D-SLM4887, closing a gap the
    re-commissioning found by execution): must not return an empty (or
    truncated) list as though it were a true reading of the manifest.
    Executed against three ordinary, unmodified CMake idioms -- `set(X)`
    immediately followed by `list(APPEND X ...)`; a `set(...)` block
    containing a comment holding a stray `)`; an earlier, unrelated mention
    of the variable's own name inside a comment -- the pre-fix regex
    returned zero (or a silently truncated) source list, with no error,
    from a manifest that in fact declares real sources. Zero is not a
    silent, valid reading of a manifest that names the variable at all: a
    parser that cannot tell "the variable is genuinely absent" from "the
    variable is present and my own regex could not capture it" is required
    to raise CoreSourcesDerivationError rather than guess."""
    path = manifest_path or _DEFAULT_MANIFEST
    with open(path) as f:
        text = f.read()
    m = _CORE_SOURCES_RE.search(text)
    if not m:
        raise ValueError("no set(SUPERSLM_CORE_SOURCES ...) block found in {!r}".format(path))

    # The non-greedy capture above terminates at the FIRST ")" it finds
    # after the variable name -- which may be a ")" embedded in a comment
    # inside the block's own span, never the variable's own real closing
    # parenthesis. Check the LAST line of the whole match (the line the
    # regex's own terminating ")" actually landed on): if a "#" precedes
    # that ")" on the same line, the parse cannot positively confirm which
    # ")" it reached and is required to say so rather than guess.
    match_lines = m.group(0).splitlines()
    last_match_line = match_lines[-1] if match_lines else ""
    hash_pos = last_match_line.find("#")
    close_paren_pos = last_match_line.rfind(")")
    if hash_pos != -1 and close_paren_pos != -1 and hash_pos < close_paren_pos:
        raise CoreSourcesDerivationError(
            "derive_core_sources: cannot positively confirm the parse reached "
            "SUPERSLM_CORE_SOURCES's own balanced closing parenthesis in {!r} "
            "-- a '#' precedes the matched ')' on the same line, which may be "
            "a comment containing a stray ')' rather than the variable's own "
            "terminator".format(path)
        )

    sources = [line.strip() for line in m.group(1).splitlines()
              if line.strip() and not line.strip().startswith("#")]
    if not sources:
        raise CoreSourcesDerivationError(
            "derive_core_sources: {!r} names SUPERSLM_CORE_SOURCES but the "
            "parse captured zero source paths -- no real SuperSLM manifest "
            "has ever declared the variable with genuinely zero sources, and "
            "a parser that cannot tell 'genuinely empty' from 'my own regex "
            "failed to capture' (an earlier unrelated mention matched first, "
            "or a separate list(APPEND ...) call populates the variable "
            "instead) is required to raise rather than guess".format(path)
        )
    return sources


class DuplicateStemError(ValueError):
    """Raised by enumerate_scan_targets() when two sources in the manifest
    share a basename stem in different directories (design Sec4.1 fold
    round 34 gap (d), D-SLM4859)."""


def enumerate_scan_targets(manifest_path: str = None, build_dir: str = None,
                          obj_ext: str = ".obj") -> list:
    """Returns (translation_unit_name, compiled_object_path) pairs, derived
    from SUPERSLM_CORE_SOURCES at scan time. `build_dir` names where each
    TU's own compiled object is expected (a caller-supplied convention; this
    function derives WHICH objects are expected, not how they were built).
    `obj_ext` (T-2348, Brunel, 8a28460-t2344's own O3) names the compiled
    object's own file extension -- `.obj` by default, matching every caller
    in this tree today (the MSVC/COFF leg), but this function is the RULED
    single production membership entry point for every leg in the 29-job
    matrix, including the clang/GCC legs that emit `.o`; a caller on such a
    leg passes `obj_ext=".o"` rather than this function hardcoding the one
    extension it happened to be built against.

    RULED the single production membership entry point (design Sec4.1 fold
    round 34 gap (d), D-SLM4859, closing Poirot's S3): every driver that
    builds or scans SUPERSLM_CORE_SOURCES calls this function rather than
    re-deriving the (translation-unit, object-path) pairing inline -- a
    second, hand-written copy is exactly where `run_fp_free_scan_real_
    corpus.py`'s own pairing drifted from this one (Poirot's S3, this
    fold's own delta manifest item 2).

    REFUSES (raises DuplicateStemError) the moment two sources anywhere in
    the manifest share a stem: the stem-based pairing would otherwise
    silently overwrite one translation unit's own expected object path with
    another's, with no error and no visible sign that one TU's own object is
    never actually scanned -- the identical absence-reads-as-clean shape
    Sec4.1's own membership rule exists to close, one layer up, in the
    pairing rather than the symbol table."""
    sources = derive_core_sources(manifest_path)
    out_dir = build_dir or os.path.join(_REPO_ROOT, "out", "fp_scan")
    seen_stems: dict = {}
    targets = []
    for src in sources:
        stem = os.path.splitext(os.path.basename(src))[0]
        if stem in seen_stems:
            raise DuplicateStemError(
                "enumerate_scan_targets: two sources share the stem {!r} -- {!r} and "
                "{!r} -- which would silently collapse onto the identical object path "
                "and leave one translation unit never scanned".format(
                    stem, seen_stems[stem], src)
            )
        seen_stems[stem] = src
        obj_path = os.path.join(out_dir, stem + obj_ext)
        targets.append((src, obj_path))
    return targets


# ---------------------------------------------------------------------------
# Diagnostic surface -- text-based, over dumpbin-shaped disassembly, never
# the byte-accounting/checks-(A)/(B)/(C) instrument above. Feeds nothing into
# scan_object's own ACCEPT/REJECT/REFUSE verdict (design Sec4.1, fold round 8
# demotion, unchanged).
# ---------------------------------------------------------------------------

_DISASM_SYM_RE = re.compile(r'^(\S.*):$')
_DISASM_INS_RE = re.compile(
    r'^\s+[0-9A-F]{16}:\s+(?:[0-9A-F]{2}\s)+\s*([a-zA-Z][a-zA-Z0-9.]*)(?:\s+(.*))?$')

_DIAG_FP_RE = re.compile(
    r'^(?:addsd|subsd|mulsd|divsd|addss|subss|mulss|divss|cvtsi2sd|cvtsi2ss|'
    r'cvttsd2si|cvttss2si|comisd|comiss|ucomisd|ucomiss|sqrtsd|sqrtss|'
    r'fadd|fsub|fmul|fdiv|fmadd|fmsub|fnmadd|fnmsub|fcmp|fcmpe|fsqrt|'
    r'scvtf|ucvtf|fcvtzs|fcvtzu)$'
)


def _iter_disasm_text(text: str):
    cur = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        m = _DISASM_SYM_RE.match(line)
        if m:
            cur = m.group(1).split(" (")[0]
            continue
        m = _DISASM_INS_RE.match(line)
        if not m or cur is None:
            continue
        yield cur, m.group(1), (m.group(2) or "")


def _tu_disasm_path(disasm_dir: str, tu_name: str) -> str:
    stem = os.path.splitext(os.path.basename(tu_name))[0]
    return os.path.join(disasm_dir, stem + ".disasm.txt")


_LOCAL_LABEL_RE = re.compile(r'^\$[A-Za-z]{1,4}[0-9]+$')


def _looks_like_symbol_target(target: str) -> bool:
    """True for an operand token that names a real symbol (a call/tail-jmp
    edge's own target), false for a bare address, a register, or a compiler-
    generated local label (the `$LN22`-shape design Sec4.1 already names) --
    the only tokens an intra-function unconditional jmp's own operand takes."""
    if not target or _LOCAL_LABEL_RE.match(target):
        return False
    if re.match(r'^(0[xX])?[0-9A-Fa-f]+$', target):
        return False  # a bare hex/decimal address: intra-function branch target
    return True


class MissingDisassemblyError(FileNotFoundError):
    """Raised by build_call_graph/flagged_symbols (T-2343, Brunel, Poirot's
    S4) when a translation unit derive_core_sources() names has no
    corresponding disassembly file in disasm_dir."""


# T-2343 (Brunel), Poirot's S4: the manifest default (derive_core_sources's
# own _DEFAULT_MANIFEST) is an ABSOLUTE path computed from this module's own
# location; the disasm_dir default was the bare relative string "obj",
# resolved against whatever the process's current working directory
# happened to be -- the two defaults did not agree about what "default"
# means. This one is absolute for the identical reason.
_DEFAULT_DISASM_DIR = os.path.join(_REPO_ROOT, "obj")


def _require_disasm_path(disasm_dir: str, tu: str) -> str:
    p = _tu_disasm_path(disasm_dir, tu)
    if not os.path.exists(p):
        # T-2343 (Brunel), Poirot's S4: a missing disassembly file and a
        # genuinely clean (empty) report were the same value, {} -- design
        # Sec4.1's own first law ("a stage that cannot classify its input
        # emits a rejection rather than a silence") applies here exactly as
        # it does to the byte-accounting instrument: this surface decides
        # nothing about ACCEPT/REJECT/REFUSE, but its own commissioning
        # population (population six) grades precisely the distinction
        # between "reported" and "silently missed," and a production
        # function that can return "silently missed" for the WHOLE corpus
        # (an absent disasm_dir) cannot be trusted to make that distinction.
        raise MissingDisassemblyError(
            "no disassembly found for translation unit {!r} at {!r} -- "
            "build_call_graph/flagged_symbols refuse rather than silently "
            "reporting an empty (indistinguishable from clean) result".format(tu, p)
        )
    return p


def build_call_graph(manifest_path: str = None, disasm_dir: str = None) -> Mapping[str, set]:
    """Edges from both `call` AND unconditional tail-`jmp` instructions --
    routed to Brunel by the test author (D-SLM4839) as a finding to examine
    rather than inherit: the reference `build_call_graph` in
    `Claude/Vitruvius/t2265-fold33-probe/` recognises only `call`, and a real
    MSVC /O2 build genuinely turns a call into a tail jmp (confirmed by direct
    execution this ticket's own session: `__builtin_fmaf`'s own compiled body,
    T-2326's own pop07_fpblind.cpp fixture, is a single `jmp` to an external
    libm target under `-msse4.1` with no hardware FMA). Excludes conditional
    jumps (always intra-function) and any jmp whose own operand is a bare
    address or a compiler-generated local label rather than a named symbol --
    the only shapes an ordinary intra-function branch's own operand takes in
    this text format.

    Raises MissingDisassemblyError (T-2343, Poirot's S4) if any derived
    translation unit has no corresponding disassembly file -- an absent
    corpus and a clean corpus are not the same value."""
    disasm_dir = disasm_dir if disasm_dir is not None else _DEFAULT_DISASM_DIR
    graph: dict = {}
    for tu in derive_core_sources(manifest_path):
        p = _require_disasm_path(disasm_dir, tu)
        with open(p) as f:
            text = f.read()
        for cur, mn, ops in _iter_disasm_text(text):
            m = mn.lower()
            if m not in ("call", "jmp"):
                continue
            target = ops.strip().split()[0] if ops.strip() else ""
            if not target:
                continue
            if m == "jmp" and not _looks_like_symbol_target(target):
                continue  # ordinary intra-function branch, not an edge
            graph.setdefault(cur, set()).add(target)
    return graph


def flagged_symbols(manifest_path: str = None, disasm_dir: str = None) -> Mapping[str, list]:
    """Raises MissingDisassemblyError (T-2343, Poirot's S4) under the
    identical condition build_call_graph does -- see that function's own
    docstring."""
    disasm_dir = disasm_dir if disasm_dir is not None else _DEFAULT_DISASM_DIR
    flagged: dict = {}
    for tu in derive_core_sources(manifest_path):
        p = _require_disasm_path(disasm_dir, tu)
        with open(p) as f:
            text = f.read()
        for cur, mn, _ops in _iter_disasm_text(text):
            if _DIAG_FP_RE.match(mn.lower()):
                flagged.setdefault(cur, []).append(mn)
    return flagged


def diagnostic_walk(roots: Sequence[str], call_graph: Mapping[str, set]) -> set:
    seen: set = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for callee in call_graph.get(node, ()):
            if callee not in seen:
                stack.append(callee)
    return seen


def diagnostic_fp_report(manifest_path: str, disasm_dir: str,
                         roots: Sequence[str]) -> Mapping[str, list]:
    graph = build_call_graph(manifest_path, disasm_dir)
    fp = flagged_symbols(manifest_path, disasm_dir)
    reachable = diagnostic_walk(roots, graph)
    return {s: fp[s] for s in reachable if fp.get(s)}
