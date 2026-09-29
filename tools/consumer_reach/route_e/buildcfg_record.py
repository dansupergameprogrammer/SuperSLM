#!/usr/bin/env python3
"""buildcfg_record.py -- the build-configuration record witness (plan rev 7; rev 8; cell 10.0 E3, Q1 B5).

Slice 1 (S1-B) adds one constant to src/matmul.cpp: a fixed marker "SSLM-BUILDCFG/2{" followed by the
SUPERSLM_* macros that decide the tier, the force macros, the tiled threshold and (rev 8) the compiler's
optimisation and assertion state (__OPTIMIZE__, __OPTIMIZE_SIZE__, NDEBUG, _DEBUG), as the compiler saw them,
each "D:<value>" when defined and "U" when not, closed by "}". It is a value the compiler produced, in the file that ships (C11 kind (a)). Every channel that
can change the compile -- flags in any spelling, -include/-imacros, response files, include-path environment,
launchers -- changes it, because it is the compile's own output. Known exception (rev 8, executed): a
`#pragma GCC optimize` reached through a shadow header changes code generation but not __OPTIMIZE__ (GCC 13).

  extract <file>...                  print {"files": {path: [records...]}} for each binary or archive.
  reference <src> <std> [defines]    compile <src>/src/matmul.cpp (a git archive of the candidate) in a clean
                                     environment (env -i) with the DECLARED Release flags (RELEASE_FLAGS, CMake's
                                     GNU default for CMAKE_CXX_FLAGS_RELEASE; the pin tool configures Release) and
                                     only the declared defines, and print the record the compiler embeds. This is
                                     the DERIVED reference.
  verdict <json> <expected-record>   exactly one distinct record across all files, present in every file, equal
                                     to the expected one; prints the reason and exits 0/1.
"""
import json, re, subprocess, sys, tempfile
from pathlib import Path

MARK = re.compile(rb"SSLM-BUILDCFG/2\{[^}\x00]*\}")
RELEASE_FLAGS = ["-O3", "-DNDEBUG"]   # DECLARED (rev 8): what -DCMAKE_BUILD_TYPE=Release adds for GNU compilers

def extract(paths):
    out = {}
    for p in paths:
        try:
            data = Path(p).read_bytes()
        except OSError as e:
            out[p] = {"error": str(e)}; continue
        out[p] = sorted({m.group(0).decode() for m in MARK.finditer(data)})
    return {"files": out}

def reference(src, std, defines):
    with tempfile.TemporaryDirectory() as d:
        obj = Path(d) / "matmul.o"
        cmd = ["env", "-i", "PATH=/usr/bin:/bin", "g++", f"-std={std}", *RELEASE_FLAGS, f"-I{src}/include", f"-I{src}/tests",
               *[f"-D{x}" for x in defines], "-c", f"{src}/src/matmul.cpp", "-o", str(obj)]
        subprocess.run(cmd, check=True, capture_output=True)
        recs = extract([str(obj)])["files"][str(obj)]
    if len(recs) != 1:
        sys.exit(f"reference: expected one record in the reference object, found {recs}")
    return recs[0]

def verdict(w, expected):
    files = w.get("files", {})
    if not files:
        return False, "no binaries read"
    bad = [p for p, r in files.items() if not isinstance(r, list) or not r]
    if bad:
        return False, "no build-configuration record in: " + ", ".join(Path(p).name for p in bad)
    distinct = sorted({r for recs in files.values() for r in recs})
    if len(distinct) != 1:
        return False, f"{len(distinct)} distinct records linked: {distinct}"
    if distinct[0] != expected:
        a = dict(x.split("=", 1) for x in distinct[0][16:-1].split(";") if x)
        b = dict(x.split("=", 1) for x in expected[16:-1].split(";") if x)
        return False, "record differs from the reference: " + "; ".join(
            f"{k}={a.get(k)} (reference {b.get(k)})" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k))
    return True, "embedded record equals the reference in " + ", ".join(Path(p).name for p in files)

if __name__ == "__main__":
    if sys.argv[1] == "extract":
        print(json.dumps(extract(sys.argv[2:]), indent=1))
    elif sys.argv[1] == "reference":
        print(json.dumps(reference(sys.argv[2], sys.argv[3], sys.argv[4:])))
    elif sys.argv[1] == "verdict":
        ok, why = verdict(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3])
        print(why); sys.exit(0 if ok else 1)
