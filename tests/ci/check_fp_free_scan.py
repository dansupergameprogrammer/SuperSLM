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
from typing import Mapping, Sequence

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
        if not (s["characteristics"] & _IMAGE_SCN_CNT_CODE):
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


def _is_padding_run(data: bytes, start: int, end: int, isa: str) -> bool:
    if end <= start:
        return True
    span = data[start:end]
    if isa == "aarch64":
        if len(span) % 4 != 0:
            return False
        nop_word = bytes([0x1F, 0x20, 0x03, 0xD5])
        return all(span[i:i + 4] == nop_word for i in range(0, len(span), 4))
    return all(b in _X86_PAD_BYTES for b in span)


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
            if g_end > g_start and not _is_padding_run(section.data, g_start, g_end, isa):
                ok = False
                unclassified += (g_end - g_start)
            continue
        _, e_start, e_end, ext = b
        if e_end <= e_start:
            ok = False
            unclassified += 0  # empty extent: unverifiable, no residue bytes to count
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
}

_X86_P_PREFIX_EXCLUDE = {"pi2fd", "pi2fw"}  # 3DNow int->float: genuinely arithmetic


def _x86_touches_vector_register(op_str: str) -> bool:
    return bool(_X86_VEC_REG_RE.search(op_str or ""))


def _x86_check_a(mnemonic: str, op_str: str) -> bool:
    """True == ACCEPT under check (A). Only called for instructions that
    touch a vector/FP/mask/tile register at all."""
    m = mnemonic.lower()
    if m in _X86_VEC_MOVE_ALLOW:
        return True
    if (m.startswith("p") or m.startswith("vp")) and m not in _X86_P_PREFIX_EXCLUDE:
        # 3DNow pf* family and named int->float conversions excluded above;
        # every other p/vp-prefixed mnemonic is packed-integer by construction.
        if not m.startswith("pf"):
            return True
    if m in ("xorps", "xorpd"):
        ops = [o.strip() for o in (op_str or "").split(",")]
        if len(ops) == 2 and ops[0] == ops[1]:
            return True  # self-zeroing idiom
        return False
    return False


# ---------------------------------------------------------------------------
# Check (A): register-file classification, AArch64.
# ---------------------------------------------------------------------------

_AARCH64_VEC_REG_RE = re.compile(r"\b[bhsdq]\d+\b|\bv\d+(?:\.\w+)?\b")
_AARCH64_NAMED_CONVERSIONS = {"scvtf", "ucvtf", "fcvtzs", "fcvtzu", "fjcvtzs"}
_AARCH64_VEC_MOVE_ALLOW = {
    "fmov", "mov", "movi",
    "ldr", "str", "ldur", "stur", "ldp", "stp",
    "ldrb", "strb", "ldrh", "strh",
    "ldursb", "ldursh", "ldursw", "ldrsb", "ldrsh", "ldrsw",
}


def _aarch64_touches_vector_register(op_str: str) -> bool:
    return bool(_AARCH64_VEC_REG_RE.search(op_str or ""))


def _aarch64_check_a(mnemonic: str, op_str: str) -> bool:
    m = mnemonic.lower()
    if m in _AARCH64_NAMED_CONVERSIONS:
        return False
    if m.startswith("f"):
        return m in _AARCH64_VEC_MOVE_ALLOW
    return m in _AARCH64_VEC_MOVE_ALLOW or True  # non-f-prefixed NEON integer op: structural accept


# ---------------------------------------------------------------------------
# Check (B): GPR/control-flow mnemonic allowlist, pinned per ISA.
# ---------------------------------------------------------------------------

_X86_GPR_ALLOW = {
    "mov", "movzx", "movsx", "movsxd", "movabs", "lea",
    "add", "adc", "sub", "sbb", "inc", "dec", "neg", "not",
    "imul", "mul", "idiv", "div",
    "and", "or", "xor", "shl", "shr", "sal", "sar", "rol", "ror", "rcl", "rcr",
    "bt", "bts", "btr", "btc", "bsf", "bsr", "popcnt", "lzcnt", "tzcnt",
    "andn", "bzhi", "pdep", "pext", "shrx", "shlx", "sarx",
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
    for pfx in ("rep ", "repe ", "repz ", "repne ", "repnz ", "lock "):
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
                        local_starts: set) -> bool:
    """True == every call/tail-jmp edge in this extent's own instructions
    passes check (C). `local_starts` is the set of byte offsets (within this
    same object, across every code section) where a real symbol begins --
    used to recognise an intra-object direct edge that carries no relocation
    because the assembler could resolve it directly (same-section target)."""
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
            raw_idx = reloc_by_offset[reloc_off]
            target_sym = sym_by_raw.get(raw_idx)
            if target_sym is None:
                return False
            is_external = (object_format == "coff" and target_sym.get("sec_num", -1) == 0) or \
                           (object_format == "elf" and target_sym.get("shndx", -1) == 0)
            if not is_external:
                continue  # in-corpus target: no separate vetting needed
            if target_sym["name"] not in extern_allow:
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


def scan_object(path: str, isa: str) -> ScanResult:
    with open(path, "rb") as f:
        data = f.read()
    object_format = _read_object_format(data)

    if object_format == "elf":
        code_sections, sym_by_raw, relocs_by_section = _parse_elf(data)
    else:
        code_sections, sym_by_raw, relocs_by_section = _parse_coff(data)

    md = _decoder(isa)

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

    # local_starts: every real symbol start offset, across every code section
    # of this object -- used by check (C) to recognise an intra-object direct
    # edge the assembler resolved without a relocation.
    local_starts = set()
    for section, _per_symbol_insns, extents in per_symbol_insns_by_section:
        for ext in extents:
            local_starts.add(ext.start)

    touches_vec = _x86_touches_vector_register if isa == "x86-64" else _aarch64_touches_vector_register
    check_a = _x86_check_a if isa == "x86-64" else _aarch64_check_a
    check_b = _x86_check_b if isa == "x86-64" else _aarch64_check_b

    verdicts: dict = {}
    for section, per_symbol_insns, extents in per_symbol_insns_by_section:
        reloc_by_offset = {r.offset: r.sym_raw_index for r in relocs_by_section.get(section.index, [])}
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
                )
            verdict = "ACCEPT" if (ab_accept and c_accept) else "REJECT"
            for name in ext.names:
                verdicts[name] = verdict

    return ScanResult(object_format=object_format, refuse=False,
                      unclassified_bytes=0, verdicts=verdicts)


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


# ---------------------------------------------------------------------------
# enumerate_scan_targets / derive_core_sources -- production membership,
# derived from SUPERSLM_CORE_SOURCES, never a second hand-maintained list.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_DEFAULT_MANIFEST = os.path.join(_REPO_ROOT, "CMakeLists.txt")

_CORE_SOURCES_RE = re.compile(r"set\(\s*SUPERSLM_CORE_SOURCES(.*?)\)", re.DOTALL)


def derive_core_sources(manifest_path: str = None) -> list:
    path = manifest_path or _DEFAULT_MANIFEST
    with open(path) as f:
        text = f.read()
    m = _CORE_SOURCES_RE.search(text)
    if not m:
        raise ValueError("no set(SUPERSLM_CORE_SOURCES ...) block found in {!r}".format(path))
    return [line.strip() for line in m.group(1).splitlines()
            if line.strip() and not line.strip().startswith("#")]


def enumerate_scan_targets(manifest_path: str = None, build_dir: str = None) -> list:
    """Returns (translation_unit_name, compiled_object_path) pairs, derived
    from SUPERSLM_CORE_SOURCES at scan time. `build_dir` names where each
    TU's own compiled object is expected (a caller-supplied convention; this
    function derives WHICH objects are expected, not how they were built)."""
    sources = derive_core_sources(manifest_path)
    out_dir = build_dir or os.path.join(_REPO_ROOT, "out", "fp_scan")
    targets = []
    for src in sources:
        stem = os.path.splitext(os.path.basename(src))[0]
        obj_path = os.path.join(out_dir, stem + ".obj")
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


def build_call_graph(manifest_path: str = None, disasm_dir: str = "obj") -> Mapping[str, set]:
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
    this text format."""
    graph: dict = {}
    for tu in derive_core_sources(manifest_path):
        p = _tu_disasm_path(disasm_dir, tu)
        if not os.path.exists(p):
            continue
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


def flagged_symbols(manifest_path: str = None, disasm_dir: str = "obj") -> Mapping[str, list]:
    flagged: dict = {}
    for tu in derive_core_sources(manifest_path):
        p = _tu_disasm_path(disasm_dir, tu)
        if not os.path.exists(p):
            continue
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
