#!/usr/bin/env python3
"""engine_macros.py -- what the compiler actually saw on the engine's src/matmul.cpp (plan rev 6, B1).

build <engine build dir>   Re-executes the build's OWN compile command for the production `superslm`
                           target's matmul.cpp object, exactly as `ninja -t commands` expands it from
                           build.ninja (the file the build consumes; it includes any compiler launcher,
                           which compile_commands.json omits), with the object output replaced by
                           `-dM -E` into a scratch file and the depfile flags dropped. Prints the
                           SUPERSLM_* macros as JSON, plus whether ninja considers that object up to date
                           (`ninja -n` has no work for it), i.e. whether the object was produced by this
                           command.
reference <src> <std> <defines...>
                           The reference: the same dump from a clean environment (`env -i`, g++ on PATH)
                           over the candidate SOURCE (a git archive of the tagged commit) with only the
                           DECLARED defines. The -std value is declared (the engine's CMake standard).
"""
import json, os, re, shlex, subprocess, sys, tempfile

def macros(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"#define (SUPERSLM_\w+)(?:\s+(.*))?$", line)
        if m:
            out[m.group(1)] = (m.group(2) or "").strip()
    return out

def build(bdir):
    tgts = subprocess.run(["ninja", "-C", bdir, "-t", "targets", "all"], capture_output=True, text=True).stdout
    obj = next(l.split(":")[0] for l in tgts.splitlines() if l.startswith("CMakeFiles/superslm.dir/") and l.split(":")[0].endswith("/matmul.cpp.o"))
    cmd = [l for l in subprocess.run(["ninja", "-C", bdir, "-t", "commands", obj], capture_output=True, text=True).stdout.splitlines() if "matmul.cpp" in l][-1]
    uptodate = "no work to do" in subprocess.run(["ninja", "-C", bdir, "-n", obj], capture_output=True, text=True).stdout
    toks = shlex.split(cmd)
    dump = tempfile.mktemp(suffix=".macros", dir=bdir)
    out, skip = [], 0
    for i, t in enumerate(toks):
        if skip: skip -= 1; continue
        if t == "-MD": continue
        if t in ("-MT", "-MF"): skip = 1; continue
        if t == "-o": out += ["-o", dump]; skip = 1; continue
        if t == "-c": out += ["-dM", "-E"]; continue
        out.append(t)
    r = subprocess.run(out, cwd=bdir, capture_output=True, text=True)
    text = open(dump).read() if os.path.exists(dump) else ""
    print(json.dumps({"object": obj, "object_up_to_date_with_command": uptodate, "rc": r.returncode,
                      "command": cmd, "macros": macros(text)}, indent=1, sort_keys=True))

def reference(src, std, defines):
    cmd = ["env", "-i", "PATH=/usr/bin:/bin", "g++", f"-std={std}", f"-I{src}/include", f"-I{src}/tests"] + \
          [f"-D{d}" for d in defines] + ["-dM", "-E", f"{src}/src/matmul.cpp"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(json.dumps(macros(r.stdout), sort_keys=True))

if __name__ == "__main__":
    if sys.argv[1] == "build": build(sys.argv[2])
    else: reference(sys.argv[2], sys.argv[3], sys.argv[4:])
