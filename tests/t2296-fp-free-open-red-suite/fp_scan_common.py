"""T-2326 (Curie) -- shared toolchain and raw-decode helpers for
test_check_fp_free_scan.py.

WHAT THIS MODULE IS NOT: it is not the deciding instrument (design Sec4.1) and it
implements none of that instrument's own logic -- no symbol-table membership rule,
no per-ISA register-file/mnemonic allowlist (checks (A)/(B)/(C)), no REFUSE
control-action, no CI-gate contract. Building any of that here would make this
module a maker-authored stand-in for the thing under test, which is Brunel's
build, not Curie's. What lives here is strictly narrower and answers only "does
this fixture, once compiled, genuinely carry the byte-level property its own
population claims" -- a raw capstone decode of a byte string or a compiled
object's own code bytes, and nothing that adjudicates ACCEPT/REJECT/REFUSE.

WHAT THIS MODULE IS: three toolchain invokers (clang, MSVC cl.exe, MSVC ml64.exe,
each via subprocess, each returning a clean (ok, message) pair rather than raising,
so a caller can SKIP -- loudly, with a named reason -- when a toolchain is not
installed in this environment, exactly the `requires_clang`/`_clang_available()`
convention already established in tests/ci/test_check_no_forward_leaf_calls.py)
and a minimal object-file byte reader (`code_section_bytes`) that extracts a named
section's raw bytes from a compiled ELF64 or COFF object without interpreting
them -- capstone does the actual decoding, in the test file itself, one call per
population so each test states its own expected mnemonics next to its own
assertion.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile

CLANG_CANDIDATES = (
    "clang++",
    r"C:\Program Files\LLVM\bin\clang++.exe",
)
CLANG_C_CANDIDATES = (
    "clang",
    r"C:\Program Files\LLVM\bin\clang.exe",
)
VSDEVCMD_CANDIDATES = (
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
)
# The Hostx64/ARM64 cross-compiler's own env script -- distinct from
# VsDevCmd.bat -arch=x64, needed for population ten's own AArch64 leg (matching
# Claude/Loki/t2273-probe/build-arm.bat's own toolchain choice exactly: real
# MSVC cl.exe cross-compiling for AArch64, not clang).
VCVARSARM64_CANDIDATES = (
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsamd64_arm64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsamd64_arm64.bat",
)


def _first_working(candidates):
    for c in candidates:
        if os.path.isabs(c):
            if os.path.exists(c):
                return c
        else:
            found = shutil.which(c)
            if found:
                return found
    return None


def find_clangxx():
    return _first_working(CLANG_CANDIDATES)


def find_clang():
    return _first_working(CLANG_C_CANDIDATES)


def find_vsdevcmd():
    for c in VSDEVCMD_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def find_vcvars_arm64():
    for c in VCVARSARM64_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


class ToolUnavailable(RuntimeError):
    """Raised by a compile helper when its own toolchain is not present. Callers
    catch this and SKIP the cell -- an environment gap, not a mechanism defect,
    per this repo's own `requires_clang` convention."""


def compile_clang_asm(src_path, out_obj, target_triple, extra_args=()):
    """Assemble a GAS-syntax .s file with clang's integrated assembler for the
    given target triple (e.g. 'x86_64-pc-linux-gnu' for ELF, 'x86_64-pc-windows-
    msvc' for COFF, 'aarch64-none-elf' for AArch64/ELF). Raises ToolUnavailable
    if no clang is on PATH or at the well-known LLVM install location."""
    clang = find_clang()
    if clang is None:
        raise ToolUnavailable("no clang.exe found on PATH or at the LLVM install location")
    cmd = [clang, "-target", target_triple, "-c", src_path, "-o", out_obj, *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_obj):
        raise RuntimeError(f"clang assemble failed ({r.returncode}): {r.stdout}\n{r.stderr}")
    return out_obj


def compile_clangxx(src_path, out_obj, target_triple, extra_args=()):
    """Compile a C++ source file with clang++ for the given target triple,
    object-only (-c), no link. Raises ToolUnavailable if clang++ is absent."""
    clangxx = find_clangxx()
    if clangxx is None:
        raise ToolUnavailable("no clang++.exe found on PATH or at the LLVM install location")
    cmd = [clangxx, "-target", target_triple, "-std=c++20", "-O2", "-c",
           src_path, "-o", out_obj, *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_obj):
        raise RuntimeError(f"clang++ compile failed ({r.returncode}): {r.stdout}\n{r.stderr}")
    return out_obj


def _run_via_env_script(env_script, env_args, tool_and_args, cwd):
    """Source an MSVC environment script (VsDevCmd.bat -arch=x64, or
    vcvarsamd64_arm64.bat for the AArch64 cross-compiler), then run a tool in
    the same process -- the same shape this repo's own T-2296/T-2268/T-2273
    probe .bat scripts use (`call <env script> ... && cl ...`).

    Written to a temporary .bat file and invoked by its own ABSOLUTE path,
    never as a quoted compound string handed to `cmd /c` directly: cmd.exe's
    own nested-quote handling of a command line built by Python's list-to-
    command-line conversion (subprocess.list2cmdline) mis-parses a quoted path
    containing spaces nested inside an already-quoted /c argument -- confirmed
    by direct execution, T-2326's own session: the identical `call
    "...VsDevCmd.bat"` line that runs correctly from an interactive shell
    reports "is not recognized" when passed as `["cmd", "/c", inner_string]`
    from subprocess.run. A temp .bat file removes the nested-quoting problem
    entirely: cmd.exe reads its own file, no re-quoting through a second
    layer. A bare (non-absolute) filename passed to `cmd /c` was ALSO found not
    to resolve against `cwd=...` reliably in this environment -- the absolute
    path is what actually works, confirmed the same way.
    """
    fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="t2326_msvcenv_")
    os.close(fd)
    try:
        with open(bat_path, "w") as f:
            f.write("@echo off\r\n")
            # NOT redirected to nul: redirecting VsDevCmd.bat's own stdout/
            # stderr to nul was found, by direct execution this session, to
            # make the environment script fail silently (ml64/cl then absent
            # from PATH even though a non-redirected call succeeds) -- the
            # subprocess call below already captures this script's combined
            # noise, so nothing is lost by leaving it unredirected.
            f.write(f'call "{env_script}" {env_args}\r\n')
            f.write(" ".join(tool_and_args) + "\r\n")
        r = subprocess.run(["cmd", "/c", os.path.abspath(bat_path)],
                            capture_output=True, text=True, cwd=cwd)
        return r
    finally:
        try:
            os.remove(bat_path)
        except OSError:
            pass


def compile_cl(src_path, out_obj, extra_args=()):
    """Compile a C++ source with MSVC cl.exe (/c, object-only), targeting the
    host x64 ISA, via VsDevCmd.bat -arch=x64. Raises ToolUnavailable if no VS
    install is found."""
    vsdevcmd = find_vsdevcmd()
    if vsdevcmd is None:
        raise ToolUnavailable("no VsDevCmd.bat found at either well-known VS2022 install location")
    src_dir = os.path.dirname(os.path.abspath(src_path))
    src_name = os.path.basename(src_path)
    out_name = os.path.basename(out_obj)
    args = ["cl", "/nologo", "/c", "/std:c++20", "/O2", "/EHsc",
            *extra_args, src_name, f"/Fo:{out_name}"]
    r = _run_via_env_script(vsdevcmd, "-arch=x64 -no_logo", args, cwd=src_dir)
    produced = os.path.join(src_dir, out_name)
    if r.returncode != 0 or not os.path.exists(produced):
        raise RuntimeError(f"cl.exe compile failed ({r.returncode}): {r.stdout}\n{r.stderr}")
    if os.path.abspath(produced) != os.path.abspath(out_obj):
        shutil.move(produced, out_obj)
    return out_obj


def compile_cl_arm64(src_path, out_obj, extra_args=()):
    """Compile a C++ source with MSVC cl.exe cross-compiling for AArch64
    (Hostx64/arm64), via vcvarsamd64_arm64.bat -- the real toolchain
    Claude/Loki/t2273-probe/build-arm.bat used for population ten's own
    historical construction. Raises ToolUnavailable if no VS install with an
    AArch64 cross target is found."""
    vcvars = find_vcvars_arm64()
    if vcvars is None:
        raise ToolUnavailable("no vcvarsamd64_arm64.bat found -- MSVC's AArch64 "
                               "cross-compiler is not installed in this environment")
    src_dir = os.path.dirname(os.path.abspath(src_path))
    src_name = os.path.basename(src_path)
    out_name = os.path.basename(out_obj)
    args = ["cl", "/nologo", "/c", "/std:c++20", "/O2", "/EHsc", "/fp:precise",
            *extra_args, src_name, f"/Fo:{out_name}"]
    r = _run_via_env_script(vcvars, "", args, cwd=src_dir)
    produced = os.path.join(src_dir, out_name)
    if r.returncode != 0 or not os.path.exists(produced):
        raise RuntimeError(f"cl.exe (arm64) compile failed ({r.returncode}): {r.stdout}\n{r.stderr}")
    if os.path.abspath(produced) != os.path.abspath(out_obj):
        shutil.move(produced, out_obj)
    return out_obj


def assemble_ml64(src_path, out_obj):
    """Assemble a MASM .asm file with ml64.exe (/c, object-only), via
    VsDevCmd.bat. Raises ToolUnavailable if no VS install is found."""
    vsdevcmd = find_vsdevcmd()
    if vsdevcmd is None:
        raise ToolUnavailable("no VsDevCmd.bat found at either well-known VS2022 install location")
    src_dir = os.path.dirname(os.path.abspath(src_path))
    src_name = os.path.basename(src_path)
    out_name = os.path.basename(out_obj)
    args = ["ml64", "/nologo", "/c", f"/Fo{out_name}", src_name]
    r = _run_via_env_script(vsdevcmd, "-arch=x64 -no_logo", args, cwd=src_dir)
    produced = os.path.join(src_dir, out_name)
    if r.returncode != 0 or not os.path.exists(produced):
        raise RuntimeError(f"ml64.exe assemble failed ({r.returncode}): {r.stdout}\n{r.stderr}")
    if os.path.abspath(produced) != os.path.abspath(out_obj):
        shutil.move(produced, out_obj)
    return out_obj


# ---------------------------------------------------------------------------
# Minimal ELF64 / COFF section-byte extraction -- structural header reads only,
# no instruction decoding (capstone does that, in the test file). This is
# intentionally far narrower than design Sec4.1's own byte-accounting law: it
# does not classify a byte as code/data/padding, it does not attribute a byte
# range to a symbol, and it does not compute UNCLASSIFIED. It answers one
# question only: "here are the raw bytes of the section named X."
# ---------------------------------------------------------------------------

def elf64_code_sections(obj_path, prefix=".text"):
    """Returns a list of raw byte blobs, one per section whose name STARTS
    WITH `prefix`, in section-table order -- never concatenated into one
    blob (see coff_code_sections's own docstring for why: a decoder that
    processes one continuous blob stops at the first section that is not
    genuine machine code, silently never reaching every section after it)."""
    with open(obj_path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        raise ValueError(f"{obj_path} is not an ELF object (magic {data[:4]!r})")
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, = struct.unpack_from("<H", data, 0x3a)
    e_shnum, = struct.unpack_from("<H", data, 0x3c)
    e_shstrndx, = struct.unpack_from("<H", data, 0x3e)

    def shdr(i):
        off = e_shoff + i * e_shentsize
        name_off, sh_type, flags, addr, offset, size, link, info, align, entsize = \
            struct.unpack_from("<IIQQQQIIQQ", data, off)
        return dict(name_off=name_off, sh_type=sh_type, flags=flags, offset=offset, size=size)

    shstrtab = shdr(e_shstrndx)
    shstr_data = data[shstrtab["offset"]:shstrtab["offset"] + shstrtab["size"]]

    def sec_name(name_off):
        end = shstr_data.index(b"\x00", name_off)
        return shstr_data[name_off:end].decode("ascii")

    chunks = []
    for i in range(e_shnum):
        s = shdr(i)
        if sec_name(s["name_off"]).startswith(prefix):
            chunks.append(data[s["offset"]:s["offset"] + s["size"]])
    if not chunks:
        raise KeyError(f"no section starting with {prefix!r} in {obj_path}")
    return chunks


def coff_code_sections(obj_path, prefix=".text"):
    """Returns a list of raw byte blobs, one per section whose (8-byte-
    truncated) name STARTS WITH `prefix`, in section-table order.

    NEVER CONCATENATED into one blob for decoding purposes -- confirmed by
    direct execution, this session, on two separate defects this shape
    produces:

    (1) A real MSVC cl.exe object compiled from ordinary STL-using C++
    carries hundreds of COMDAT code sections (`.text$mn` per template-
    instantiated function, `.text$x` for associated unwind thunks), never a
    single plain `.text` section -- one real object inspected this session
    carried 668 total sections, the large majority `.text$mn`. An exact-
    match reader silently returns "no section named .text" on any such
    object, which is every cl.exe object this suite compiles -- fixed by a
    PREFIX match (`.text$mn`/`.text$x`/plain `.text` all start with `.text`).

    (2) A prefix match alone is not sufficient: concatenating every matched
    section into one continuous byte stream and decoding it start-to-finish
    was found, this session, to silently stop finding real instructions
    partway through a real AArch64 object -- `.text$x` unwind-thunk sections
    interleaved among the `.text$mn` code sections are NOT AArch64
    instructions, and capstone's own `disasm()` generator stops permanently
    at the first byte range it cannot decode, so every `.text$mn` section
    AFTER the first `.text$x` thunk in file order was silently never reached
    (confirmed directly: a real object's own BodyDivide/BodyConvertCompare
    sections, both real and both containing genuine fdiv/scvtf instructions,
    produced ZERO f-prefixed mnemonics anywhere when decoded as one
    concatenated blob). Returning one blob PER SECTION and decoding each
    independently is what this repair requires, and what this function does.
    """
    with open(obj_path, "rb") as f:
        data = f.read()
    machine, nsec = struct.unpack_from("<HH", data, 0)
    if machine not in (0x8664, 0xAA64, 0x14C):
        raise ValueError(f"{obj_path} does not look like a COFF object (machine=0x{machine:x})")
    sec_table_off = 20  # COFF file header is 20 bytes; no optional header in a .obj
    chunks = []
    name_bytes = prefix.encode("ascii")
    for i in range(nsec):
        off = sec_table_off + i * 40
        raw_name = data[off:off + 8].rstrip(b"\x00")
        _pa, _va, size, ptr_raw = struct.unpack_from("<IIII", data, off + 8)
        if raw_name.startswith(name_bytes):
            chunks.append(data[ptr_raw:ptr_raw + size])
    if not chunks:
        raise KeyError(f"no section starting with {prefix!r} in {obj_path}")
    return chunks


def code_sections(obj_path, prefix=".text"):
    """Dispatch to the ELF or COFF reader by magic number. Returns a LIST of
    raw byte blobs, one per matching section -- see coff_code_sections's own
    docstring for why this is never a single concatenated blob."""
    with open(obj_path, "rb") as f:
        magic = f.read(4)
    if magic == b"\x7fELF":
        return elf64_code_sections(obj_path, prefix)
    return coff_code_sections(obj_path, prefix)


def code_section_bytes(obj_path, section_name=".text"):
    """Convenience wrapper: concatenates every matching section into one
    blob. SAFE ONLY for a raw byte/substring search (population thirteen's
    own "is this exact byte sequence present anywhere" ground-truth check) --
    NEVER for sequential instruction decoding (see code_sections's own
    docstring; use that function and decode each returned chunk separately
    for anything that walks instructions)."""
    return b"".join(code_sections(obj_path, section_name))


class TempDir:
    """Thin wrapper so every test uses the same discipline: a fresh scratch
    directory per test, removed afterward, never a fixture compiled into the
    working tree (this repo's own tempfile.mkdtemp() convention, tests/ci/
    test_check_no_forward_leaf_calls.py's own docstring)."""

    def __enter__(self):
        self._d = tempfile.mkdtemp(prefix="t2326_fpscan_")
        return self._d

    def __exit__(self, *exc):
        shutil.rmtree(self._d, ignore_errors=True)
