#!/usr/bin/env python3
"""codeid.py -- the codegen witness (plan rev 9; the conductor's rev-9 call). Cell 10.0 E3 and Q1 B5.

The gate is code identity: the engine the consumer actually linked must be the reference engine, built by the
harness from the candidate commit in a clean `env -i` build with the declared Release flags. The
build-configuration record stays, as a readable diagnosis of WHY two engines differ; it is not the gate.

Code identity of a static library or object ("code_sha256"):
  * per archive member, extracted with `ar p` (so the archive's own headers -- member timestamps, uid/gid, mode,
    which MSVC's archiver and a non-deterministic `ar` write -- never enter the hash; NORMALISATION 1);
  * per member, a canonical listing of the ELF object with every path-bearing and build-time part left out
    (NORMALISATION 2): each SHF_ALLOC section by name, type, flags, size and the sha256 of its bytes (.text*,
    .rodata*, .data*, .bss by size, .init_array, .eh_frame, ...); the relocations against those sections by
    offset, type, target symbol name and addend; the defined and undefined symbols by name, type, binding,
    section and size. Left out: STT_FILE symbols (the source path), .comment (the compiler banner), .note.*,
    .debug* and their relocations, .group bodies (section indices), and the symbol-table order;
  * the library's code_sha256 is sha256 over the sorted (member name, member digest) pairs.
The rule is written for ELF (GCC and Clang on Linux). COFF (MSVC) needs the same listing from `dumpbin`, not
executed here [I].

  digest <lib-or-obj>...            print {"path": {"code_sha256", "members": {name: digest}}} for each file
  text <obj>                        print the sha256 of the object's .text section bytes alone (the minimum form)
  linked <consumer-build-dir> <target>
                                    resolve the engine library <target> actually linked, from the consumer's
                                    build tree (the target's link statement in build.ninja: its LINK_LIBRARIES),
                                    and print {"target", "library", "resolved_from", "code_sha256", "members"}
  build-reference <src> <out> [--toolchain-pin <json>] [--cxx <compiler>] [flag...]
                                    build the reference engine: `env -i PATH=/usr/bin:/bin cmake -G Ninja
                                    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=<abs> -DCMAKE_CXX_FLAGS="<flags>"`
                                    over <src> (a git archive of the commit), target superslm; print its digest
                                    as JSON. Rev 10 (F3): the compiler (--cxx, else $CXX, else c++, resolved on the
                                    caller's PATH) must be a certified pair of the embedder's toolchain pin
                                    (--toolchain-pin, else $CODEID_TOOLCHAIN_PIN), checked before the build;
                                    otherwise "reference refused" and exit 3
  toolchain [--toolchain-pin <json>] [--cxx <compiler>]
                                    print the resolved compiler's CMake identity and whether it is certified
  witness <out.json> <consumer-build-dir> <tool-library>
                                    route E's code witness: "semb" (linked, above) and "route_e_reach" (the library
                                    the harness's own link command for the 10.0 tool names)
  compare <a.json> <b.json>         exit 0 when the code_sha256 values are equal; otherwise print which members
                                    differ and exit 1
"""
import hashlib, json, re, shlex, struct, subprocess, sys, tempfile
from pathlib import Path

SHF_ALLOC = 0x2
SKIP_PREFIX = (".comment", ".note", ".debug", ".rela.debug", ".rel.debug", ".group")
# Rev 10 (the LTO plane): .gnu.lto_* sections are NOT skipped any more. They are not SHF_ALLOC, so they are hashed
# by an explicit rule below (name with GCC's per-object random suffix removed, size, sha256 of the bytes) and
# enter the listing under their own key only when present -- an object without them hashes exactly as in rev 9.
LTO_PREFIX = ".gnu.lto_"

def _elf(data):
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian object")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for i in range(shnum):
        n, t, fl, _a, off, sz, link, info, _al, esz = struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        secs.append(dict(nameoff=n, type=t, flags=fl, off=off, size=sz, link=link, info=info, entsize=esz))
    so = secs[shstrndx]
    def s(tab, o):
        e = data.index(b"\0", tab["off"] + o); return data[tab["off"] + o:e].decode(errors="replace")
    for x in secs:
        x["name"] = s(so, x["nameoff"])
    return secs, s

def object_digest(data):
    secs, s = _elf(data)
    canon = {"sections": [], "relocs": {}, "symbols": []}
    syms = []
    for x in secs:
        if x["type"] == 2:   # SHT_SYMTAB
            strtab = secs[x["link"]]
            for k in range(x["size"] // 24):
                nm, info, _o, shndx, val, sz = struct.unpack_from("<IBBHQQ", data, x["off"] + k * 24)
                typ, bind = info & 0xF, info >> 4
                name = s(strtab, nm) if nm else ""
                if typ == 3:  # STT_SECTION: name it by its section
                    name = "§" + (secs[shndx]["name"] if shndx < len(secs) else str(shndx))
                sec = secs[shndx]["name"] if 0 < shndx < len(secs) else {0: "UND", 0xFFF1: "ABS", 0xFFF2: "COM"}.get(shndx, str(shndx))
                syms.append((name, typ, bind, sec, sz, val))
    for name, typ, bind, sec, sz, val in syms:
        if typ == 4 or (not name and typ == 0):   # STT_FILE (a path) and the null symbol
            continue
        canon["symbols"].append([name, typ, bind, sec, sz, val])
    canon["symbols"].sort()
    lto = []
    for x in secs:
        if x["name"].startswith(SKIP_PREFIX):
            continue
        if x["name"].startswith(LTO_PREFIX):
            lto.append([re.sub(r"\.[0-9a-f]{16}$", "", x["name"]), x["size"],
                        hashlib.sha256(data[x["off"]:x["off"] + x["size"]]).hexdigest()])
            continue
        if x["flags"] & SHF_ALLOC:
            body = b"" if x["type"] == 8 else data[x["off"]:x["off"] + x["size"]]   # SHT_NOBITS
            canon["sections"].append([x["name"], x["type"], x["flags"], x["size"], hashlib.sha256(body).hexdigest()])
        if x["type"] == 4:   # SHT_RELA
            tgt = secs[x["info"]]["name"]
            if tgt.startswith(SKIP_PREFIX) or not (secs[x["info"]]["flags"] & SHF_ALLOC):
                continue
            rel = []
            for k in range(x["size"] // 24):
                off, info, add = struct.unpack_from("<QQq", data, x["off"] + k * 24)
                si, ty = info >> 32, info & 0xFFFFFFFF
                rel.append([off, ty, syms[si][0] if si < len(syms) else str(si), add])
            canon["relocs"][tgt] = rel
    canon["sections"].sort()
    if lto:
        canon["lto"] = sorted(lto)
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()

def members(path):
    p = Path(path); data = p.read_bytes()
    if data[:8] != b"!<arch>\n":
        return [(p.name, data)]
    names = subprocess.run(["ar", "t", str(p)], capture_output=True, text=True, check=True).stdout.split()
    out, seen = [], {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        # `ar p` with a count (N) reads the n-th member of a duplicated name
        cmd = ["ar", "pN", str(seen[n]), str(p), n] if names.count(n) > 1 else ["ar", "p", str(p), n]
        out.append((n, subprocess.run(cmd, capture_output=True, check=True).stdout))
    return out

def digest(path):
    mem = {}
    pairs = []
    for n, d in members(path):
        try:
            h = object_digest(d)
        except ValueError:
            h = "raw:" + hashlib.sha256(d).hexdigest()
        pairs.append([n, h]); mem[n if n not in mem else f"{n}#{len(pairs)}"] = h
    pairs.sort()
    return {"code_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest(), "members": mem}

def text_sha(obj):
    data = Path(obj).read_bytes(); secs, _ = _elf(data)
    t = [x for x in secs if x["name"] == ".text"][0]
    return hashlib.sha256(data[t["off"]:t["off"] + t["size"]]).hexdigest()

def linked(build, target):
    nj = (Path(build) / "build.ninja").read_text()
    m = re.search(rf"^build {re.escape(target)}: [^\n]*\n((?:  [^\n]*\n)*)", nj, re.M)
    if not m:
        return {"target": target, "library": None, "resolved_from": "build.ninja", "error": f"no link statement for {target}"}
    ll = re.search(r"^  LINK_LIBRARIES = (.*)$", m.group(1), re.M)
    libs = [x for x in shlex.split(ll.group(1) if ll else "") if Path(x).name == "libsuperslm.a"]
    if len(libs) != 1:
        return {"target": target, "library": None, "resolved_from": "build.ninja LINK_LIBRARIES", "error": f"{len(libs)} engine libraries on the link line"}
    lib = libs[0] if Path(libs[0]).is_absolute() else str(Path(build) / libs[0])
    return {"target": target, "library": lib, "resolved_from": "build.ninja LINK_LIBRARIES of " + target, **digest(lib)}

class Refused(Exception):
    pass

def _compiler_file(build):
    """The compiler identity CMake resolved for THIS configure: CMakeFiles/<ver>/CMakeCXXCompiler.cmake."""
    f = sorted(Path(build).glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if len(f) != 1:
        raise Refused(f"no single CMakeCXXCompiler.cmake under {build}")
    t = f[0].read_text()
    g = lambda k: (re.search(rf'^set\({k} "([^"]*)"\)', t, re.M) or [None, ""])[1]
    return {"id": g("CMAKE_CXX_COMPILER_ID"), "version": g("CMAKE_CXX_COMPILER_VERSION"),
            "frontend": g("CMAKE_CXX_COMPILER_FRONTEND_VARIANT"), "path": g("CMAKE_CXX_COMPILER")}

def _certified(tc, pin):
    for e in json.loads(Path(pin).read_text())["certified"]:
        if e["id"] == tc["id"] and e["version"] == tc["version"] and e.get("frontend", "") in ("", tc["frontend"]):
            return True
    return False

def probe_toolchain(cxx):
    """Rev 10 (F3): the compiler's identity, asked of CMake itself (the same variables ToolchainPin.cmake reads),
    in the same `env -i` the reference build runs in, before anything is built."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(p LANGUAGES CXX)\n")
        r = subprocess.run(["env", "-i", "PATH=/usr/bin:/bin", "cmake", "-S", d, "-B", d + "/b", "-G", "Ninja",
                            f"-DCMAKE_CXX_COMPILER={cxx}"], capture_output=True, text=True)
        if r.returncode:
            raise Refused(f"the compiler {cxx} does not configure: {r.stderr.strip()[-200:]}")
        return _compiler_file(d + "/b")

def resolve_cxx(cxx):
    import os, shutil
    cxx = cxx or os.environ.get("CXX") or "c++"
    words = shlex.split(cxx)
    if len(words) != 1:
        raise Refused(f"the compiler '{cxx}' carries arguments; the reference compiler is one executable")
    w = shutil.which(words[0])
    if not w:
        raise Refused(f"the compiler '{words[0]}' is not on PATH")
    return w

def build_reference(src, out, flags, pin=None, cxx=None):
    """Rev 10 (F3): the reference engine is built with the consumer's CERTIFIED toolchain. The compiler is the one
    named (--cxx, else the caller's CXX, else c++), resolved on the CALLER's PATH to an absolute path and passed to
    CMake explicitly (rev 9 let `env -i` drop CXX and PATH, so it built with whatever /usr/bin/c++ was). Its
    identity is asked of CMake before the build and must be one of the certified pairs in the embedder's
    toolchain pin (cmake/toolchain-pin.json, exactly as ToolchainPin.cmake compares them); it is read again from
    the reference build's own CMake files after configure. Any miss raises Refused: no reference, no gate."""
    import os
    pin = pin or os.environ.get("CODEID_TOOLCHAIN_PIN")
    if not pin or not Path(pin).is_file():
        raise Refused("no toolchain pin (--toolchain-pin or CODEID_TOOLCHAIN_PIN): the reference cannot be built "
                      "with a certified toolchain")
    exe = resolve_cxx(cxx)
    tc = probe_toolchain(exe)
    if not _certified(tc, pin):
        raise Refused(f"{exe} is {tc['id']} {tc['version']} (frontend={tc['frontend']}), not a certified pair in {pin}")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    env = ["env", "-i", "PATH=/usr/bin:/bin"]
    subprocess.run([*env, "cmake", "-S", src, "-B", str(out), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_CXX_COMPILER={exe}", f"-DCMAKE_CXX_FLAGS={' '.join(flags)}"], check=True, capture_output=True)
    built = _compiler_file(out)
    if (built["id"], built["version"], built["frontend"]) != (tc["id"], tc["version"], tc["frontend"]):
        raise Refused(f"the reference build configured {built}, not the probed {tc}")
    subprocess.run([*env, "cmake", "--build", str(out), "--target", "superslm"], check=True, capture_output=True)
    lib = out / "libsuperslm.a"
    return {"library": str(lib), "resolved_from": "the harness's reference build", "flags": " ".join(flags),
            "toolchain": {**tc, "path": exe, "certified_by": str(pin)}, **digest(lib)}

def compare(a, b):
    if a.get("code_sha256") and a.get("code_sha256") == b.get("code_sha256"):
        return True, "code identical"
    ma, mb = a.get("members") or {}, b.get("members") or {}
    d = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
    return False, f"{len(d)} of {len(set(ma) | set(mb))} objects differ: " + ", ".join(d[:8]) + (" ..." if len(d) > 8 else "")

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "digest":
        print(json.dumps({p: digest(p) for p in sys.argv[2:]}, indent=1))
    elif c == "text":
        print(text_sha(sys.argv[2]))
    elif c == "linked":
        print(json.dumps(linked(sys.argv[2], sys.argv[3]), indent=1))
    elif c == "build-reference":
        a = sys.argv[4:]; kw = {}
        while a and a[0] in ("--toolchain-pin", "--cxx"):
            kw[{"--toolchain-pin": "pin", "--cxx": "cxx"}[a[0]]] = a[1]; a = a[2:]
        try:
            print(json.dumps(build_reference(sys.argv[2], sys.argv[3], a, **kw), indent=1))
        except Refused as e:
            print(f"reference refused: {e}", file=sys.stderr); sys.exit(3)
    elif c == "toolchain":   # [--toolchain-pin P] [--cxx C]: print the resolved compiler, its identity, certified or not
        a = sys.argv[2:]; kw = dict(zip(a[0::2], a[1::2]))
        import os
        exe = resolve_cxx(kw.get("--cxx")); tc = probe_toolchain(exe)
        pin = kw.get("--toolchain-pin") or os.environ.get("CODEID_TOOLCHAIN_PIN")
        print(json.dumps({**tc, "path": exe, "certified": bool(pin) and _certified(tc, pin)}))
    elif c == "witness":   # out.json consumer-build-dir tool-library: route E's code witness (semb and the 10.0 tool)
        out, build, toollib = sys.argv[2:5]
        w = {"semb": linked(build, "semb")}
        try:
            w["route_e_reach"] = {"target": "route_e_reach", "library": toollib,
                                  "resolved_from": "the harness's link command for route_e_reach", **digest(toollib)}
        except (OSError, subprocess.CalledProcessError) as e:
            w["route_e_reach"] = {"target": "route_e_reach", "library": toollib, "error": str(e)}
        Path(out).write_text(json.dumps(w, indent=1))
    elif c == "compare":
        ok, why = compare(*(json.loads(Path(p).read_text()) for p in sys.argv[2:4])); print(why); sys.exit(0 if ok else 1)
