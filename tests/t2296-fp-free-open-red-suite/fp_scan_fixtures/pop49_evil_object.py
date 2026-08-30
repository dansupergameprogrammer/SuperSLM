"""T-2403 (Curie), R1 -- population forty-nine's two-cell must-reject pair
(`Claude/Vitruvius/t2265-fold46-delta-manifest.md` Sec2's own R1 entry;
design `t2265-superslm-fp-free-open-design-2026-08-24.md` Sec7 dimension
11's forty-ninth population; D-SLM5149/D-SLM5153/D-SLM5163/D-SLM5169).

Builds two hand-emitted x86-64 COFF objects, each carrying ONE section
flagged `IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_CNT_INITIALIZED_DATA`
(0x20000040) -- MEM_EXECUTE without CNT_CODE -- containing the four raw
bytes of `divsd xmm0, xmm0` (`f2 0f 5e c0`). `_parse_coff`'s own
section-selection criterion at tests/ci/check_fp_free_scan.py:334 keys on
CNT_CODE alone, so a section flagged this way is skipped entirely and never
enters `code_sections` -- the bypass R1 closes.

`build_zero_symbol` reproduces the conductor's own construction
(`Claude/Bach/probes/t2391-exec-section-bypass.py`, D-SLM5149) BYTE FOR
BYTE, unmodified: zero symbols, empty string table. `build_symbol_carrying`
is the design's own "symbol-carrying variant (one COFF function symbol
added, otherwise identical)" -- every byte of the zero-symbol construction
is reproduced unchanged; the only addition is one COFF symbol-table entry
(one function-typed, externally-visible symbol named `EvilExecSectionFn`,
value 0, section 1) and the string table its >8-character name requires.

Two different mechanisms, two must-reject observables (design doc, this
population's own text, corrected by D-SLM5163 and restated at
`t2265-fold45-delta-manifest.md` Sec5's own R1 entry):

  - The zero-symbol object has no function symbol, so `_compute_extents`
    derives zero extents for its one section -- the whole section is one
    uncovered "gap," and `divsd`'s own bytes are neither a recognised
    padding run nor attributable to any symbol. Once R1 widens section
    selection, this section enters `code_sections` and REFUSES (byte-
    accounting cannot account for the divsd bytes): `ScanResult.refuse`
    is True, `ScanResult.verdicts` stays empty. THE CLASSIFIER (check A/B/C)
    IS NEVER REACHED for this construction, at any point before or after
    R1 -- there is no per-symbol verdict for a zero-symbol object to carry.
  - The symbol-carrying object's one function symbol witnesses an extent
    covering the whole section (no declared size, no later witness, so the
    extent is clamped to the section's own end) -- byte-accounting succeeds
    (the extent's own `divsd` decodes cleanly, chain-decoding to the
    extent's exact end), and check (A) classifies the decoded `divsd`:
    not on `_X86_VEC_MOVE_ALLOW`, not p/vp-prefixed, not in
    `_X86_BITWISE_FP_FAMILY` -- default-deny REJECT. `ScanResult.verdicts`
    carries `{"EvilExecSectionFn": "REJECT"}` once R1 lands.

Both must-reject cells hold BEFORE R1 lands too, but for the wrong
(pre-fix) reason: the section is skipped outright, so `code_sections` is
empty for both, `ScanResult.refuse` is False and `ScanResult.verdicts` is
empty -- `scan_object` reports a clean, unremarkable object either way,
exit 0. That is the live vulnerability R1 closes.

Disposable. Nothing here enters product source or the test suite proper.
"""
import struct

CODE = bytes([0xF2, 0x0F, 0x5E, 0xC0])  # divsd xmm0, xmm0

IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_MEM_EXECUTE = 0x20000000
CHARACTERISTICS = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_EXECUTE  # 0x20000040

_HDR_SIZE = 20
_SEC_HDR_SIZE = 40
_PTR_RAW = _HDR_SIZE + _SEC_HDR_SIZE  # section data immediately follows the one section header

# The conductor's own construction's function symbol name is not fixed by
# any prior fold -- chosen here, distinct from `fp_carrier.cpp`'s own
# `SslmProbeFpCarrier` (a different, separately-compiled fixture shared by
# populations forty-seven/forty-eight/fifty) to avoid conflating two
# unrelated constructions under one name.
SYMBOL_NAME = "EvilExecSectionFn"

IMAGE_SYM_CLASS_EXTERNAL = 2
_FUNCTION_TYPE = 0x20  # (sym_type & 0xFF) >> 4 == 2 -- `_parse_coff`'s own is_function test


def build_zero_symbol(out_dir):
    """Byte-for-byte reproduction of `Claude/Bach/probes/
    t2391-exec-section-bypass.py`'s own `evil.obj` -- zero COFF symbols, an
    empty (4-byte length-prefix-only) string table."""
    ptr_symtab = _PTR_RAW + len(CODE)

    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, ptr_symtab, 0, 0, 0)
    coff += struct.pack("<8sIIIIIIHHI", b".evil\0\0\0", 0, 0, len(CODE), _PTR_RAW,
                        0, 0, 0, 0, CHARACTERISTICS)
    coff += CODE
    coff += struct.pack("<I", 4)  # empty string table

    import os
    path = os.path.join(out_dir, "evil_zero_symbol.obj")
    with open(path, "wb") as f:
        f.write(coff)
    return path, coff


def build_symbol_carrying(out_dir):
    """The zero-symbol construction, unmodified byte for byte, plus ONE
    COFF function symbol (`SYMBOL_NAME`, value 0, section 1, function-typed,
    externally visible, zero aux entries) and the string table its
    >8-character name requires."""
    ptr_symtab = _PTR_RAW + len(CODE)

    name_raw = SYMBOL_NAME.encode("ascii")
    strtab_body = name_raw + b"\x00"
    name_off_in_strtab = 4  # the 4-byte length prefix occupies offsets [0,4)
    sym_name8 = struct.pack("<II", 0, name_off_in_strtab)  # first 4 bytes zero -> long-name form
    sym_entry = (sym_name8 +
                struct.pack("<IhHBB", 0, 1, _FUNCTION_TYPE, IMAGE_SYM_CLASS_EXTERNAL, 0))
    assert len(sym_entry) == 18
    strtab = struct.pack("<I", 4 + len(strtab_body)) + strtab_body

    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, ptr_symtab, 1, 0, 0)
    coff += struct.pack("<8sIIIIIIHHI", b".evil\0\0\0", 0, 0, len(CODE), _PTR_RAW,
                        0, 0, 0, 0, CHARACTERISTICS)
    coff += CODE
    coff += sym_entry
    coff += strtab

    import os
    path = os.path.join(out_dir, "evil_symbol_carrying.obj")
    with open(path, "wb") as f:
        f.write(coff)
    return path, coff
