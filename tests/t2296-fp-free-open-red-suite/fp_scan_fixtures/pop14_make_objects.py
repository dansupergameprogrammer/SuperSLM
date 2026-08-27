"""T-2277 -- object constructor for the strike on the T-2265 fold-13 delta.

Emits ELF64 (x86-64 and AArch64) and COFF (x86-64) relocatable objects with an
explicitly controlled symbol table, so that exactly one coordinate varies between
a control object and a fracture object: the presence of a SECOND code symbol at
an existing code symbol's own start address (the shape a C++ complete-object /
base-object constructor alias, an `__attribute__((alias))`, a `.set` directive,
or a MASM label-plus-PROC pair produces), or a DATA-typed symbol covering a byte
range in an executable section.

The section bytes are byte-identical across every variant of a given ISA. Only
the symbol table differs.

Disposable. Nothing here enters product source or the test suite.
"""
import struct

# ---------------------------------------------------------------------------
# Section payloads. Identical across every variant of a given ISA.
# ---------------------------------------------------------------------------
# x86-64: HashSite = add edi,esi / mov eax,edi / ret          (offsets 0..4)
#         3 bytes of 0x90 structural padding                  (offsets 5..7)
#         BodyDivide = divsd xmm0,xmm1 / ret                  (offsets 8..12)
X86_TEXT = bytes([0x01, 0xF7, 0x89, 0xF8, 0xC3,
                  0x90, 0x90, 0x90,
                  0xF2, 0x0F, 0x5E, 0xC1, 0xC3])
X86_HASH = (0, 5)
X86_BODY = (8, 5)

# AArch64: HashSite = add w0,w0,w1 / ret                      (offsets 0..7)
#          BodyDivide = fdiv d0,d0,d1 / ret                   (offsets 8..15)
AARCH64_TEXT = (struct.pack('<I', 0x0B010000) + struct.pack('<I', 0xD65F03C0) +
                struct.pack('<I', 0x1E611800) + struct.pack('<I', 0xD65F03C0))
A64_HASH = (0, 8)
A64_BODY = (8, 8)

STB_GLOBAL = 1
STT_OBJECT = 1
STT_FUNC = 2


def build_elf64(path, machine, text, symbols):
    """symbols: list of (name, value, size, sym_type). Emitted in the given
    order -- symbol-table order is what decides which of two symbols sharing a
    start address the scanner's own stable sort places first."""
    shstr = b'\x00.text\x00.symtab\x00.strtab\x00.shstrtab\x00'
    off_text = shstr.index(b'.text\x00')
    off_symtab = shstr.index(b'.symtab\x00')
    off_strtab = shstr.index(b'.strtab\x00')
    off_shstrtab = shstr.index(b'.shstrtab\x00')

    strtab = b'\x00'
    sym_bytes = b'\x00' * 24  # the mandatory null symbol
    for name, value, size, sym_type in symbols:
        name_off = len(strtab)
        strtab += name.encode('ascii') + b'\x00'
        st_info = (STB_GLOBAL << 4) | sym_type
        sym_bytes += struct.pack('<IBBHQQ', name_off, st_info, 0, 1, value, size)

    ehdr_size = 64
    shentsize = 64
    nsec = 5
    cur = ehdr_size
    text_off = cur
    cur += len(text)
    symtab_off = cur
    cur += len(sym_bytes)
    strtab_off = cur
    cur += len(strtab)
    shstr_off = cur
    cur += len(shstr)
    shoff = cur

    e = bytearray(64)
    e[0:4] = b'\x7fELF'
    e[4] = 2      # ELFCLASS64
    e[5] = 1      # little endian
    e[6] = 1      # EV_CURRENT
    struct.pack_into('<H', e, 0x10, 1)          # ET_REL
    struct.pack_into('<H', e, 0x12, machine)
    struct.pack_into('<I', e, 0x14, 1)
    struct.pack_into('<Q', e, 0x28, shoff)
    struct.pack_into('<H', e, 0x34, ehdr_size)
    struct.pack_into('<H', e, 0x3a, shentsize)
    struct.pack_into('<H', e, 0x3c, nsec)
    struct.pack_into('<H', e, 0x3e, 4)          # shstrndx

    def shdr(name_off, sh_type, flags, offset, size, link, info, align, entsize):
        return struct.pack('<IIQQQQIIQQ', name_off, sh_type, flags, 0, offset,
                           size, link, info, align, entsize)

    shdrs = b''
    shdrs += shdr(0, 0, 0, 0, 0, 0, 0, 0, 0)
    shdrs += shdr(off_text, 1, 0x6, text_off, len(text), 0, 0, 16, 0)       # PROGBITS ALLOC|EXECINSTR
    shdrs += shdr(off_symtab, 2, 0, symtab_off, len(sym_bytes), 3, 1, 8, 24)  # SYMTAB
    shdrs += shdr(off_strtab, 3, 0, strtab_off, len(strtab), 0, 0, 1, 0)
    shdrs += shdr(off_shstrtab, 3, 0, shstr_off, len(shstr), 0, 0, 1, 0)

    with open(path, 'wb') as f:
        f.write(bytes(e) + text + sym_bytes + strtab + shstr + shdrs)


IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000


def build_coff(path, machine, text, symbols):
    """symbols: list of (name, value, is_function). COFF carries no per-symbol
    size, exactly as the fold-13 reader states."""
    strtab_body = b''
    sym_bytes = b''
    for name, value, is_function in symbols:
        raw = name.encode('ascii')
        if len(raw) <= 8:
            name8 = raw + b'\x00' * (8 - len(raw))
        else:
            name8 = struct.pack('<II', 0, 4 + len(strtab_body))
            strtab_body += raw + b'\x00'
        symtype = 0x20 if is_function else 0x00
        sym_bytes += name8 + struct.pack('<IhHBB', value, 1, symtype, 2, 0)
    strtab = struct.pack('<I', 4 + len(strtab_body)) + strtab_body

    hdr_size = 20 + 40
    text_off = hdr_size
    symptr = text_off + len(text)
    nsym = len(symbols)

    hdr = struct.pack('<HHIIIHH', machine, 1, 0, symptr, nsym, 0, 0)
    sec = (b'.text\x00\x00\x00' +
           struct.pack('<IIII', 0, 0, len(text), text_off) +
           struct.pack('<IIHH', 0, 0, 0, 0) +
           struct.pack('<I', IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ))
    assert len(sec) == 40
    with open(path, 'wb') as f:
        f.write(hdr + sec + text + sym_bytes + strtab)


EM_X86_64 = 0x3e
EM_AARCH64 = 0xb7
COFF_X86_64 = 0x8664
COFF_ARM64 = 0xaa64
