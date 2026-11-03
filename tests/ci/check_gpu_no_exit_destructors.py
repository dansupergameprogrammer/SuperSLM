#!/usr/bin/env python3
"""No D3D12 object is released by a static destructor at process exit (1.18.1).

    python tests/ci/check_gpu_no_exit_destructors.py [--src <tree>]

A static-storage object that holds a D3D12 object (a ComPtr, a CachedPipeline, a harness Device or DeviceHolder, or a
container of any of them) is destroyed among the static destructors at exit. In a DLL host (the SuperSLM-Unreal
editor plugin) that is the DLL's process detach, after the D3D12 runtime has torn down its own state, and the last
device release fails fast there (0xC0000409, FAST_FAIL_FATAL_APP_EXIT). Such objects are held through a reference to
a heap object that is never deleted (`static auto& x = *new T();`), and the process device's holder keeps its
Device in a union its destructor does not destroy.

The lint reads the code tokens of every .h and .cpp file under src/ and include/ (comments and string literals
stripped) and reports:

* a `static` or namespace-scope `inline` declaration, held by value, whose declared type names ComPtr,
  CachedPipeline or Device (a reference, a pointer or a function is not reported);
* a DeviceHolder whose Device member is not inside a union.

tests/ and tools/ are outside its scope. Output: one line per finding, "EXIT-DTOR: <file>:<line> | <text>". Exit 1 if
any line was printed, else 0.
"""
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOLDER_TYPES = re.compile(r"\b(ComPtr|CachedPipeline|Device)\b")  # DeviceHolder: the union rule below
DECL = re.compile(r"^\s*(static|inline)\s+(?!constexpr\b)([^;=(){}]*?)\b([A-Za-z_]\w*)\s*(;|=|\{)")


def cstrip(src):
    """Comments, string and character literals -> blanks; newlines kept so line numbers hold."""
    out = []; i = 0; n = len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i); j = n if j < 0 else j; out.append(" " * (j - i)); i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j < 0 else j + 2; out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n": j += 2 if src[j] == "\\" else 1
            out.append(c + " " * (min(j, n) - i - 1) + (c if j < n else "")); i = j + 1
        else:
            out.append(c); i += 1
    return "".join(out)


def check_text(path, text):
    findings = []
    code = cstrip(text)
    lines = code.split("\n")
    for no, line in enumerate(lines, 1):
        m = DECL.match(line)
        if not m:
            continue
        type_part = m.group(2)
        if "&" in type_part or "*" in type_part:
            continue
        if not HOLDER_TYPES.search(type_part):
            continue
        findings.append((path, no, line.strip()))
    for m in re.finditer(r"\bstruct\s+DeviceHolder\b[^{;]*\{", code):
        depth = 1; i = m.end(); union_depth = None; body_start = i
        while i < len(code) and depth:
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
            i += 1
        body = code[body_start:i - 1]
        # the Device member must sit inside a nested union { ... }
        stripped = re.sub(r"\bunion\s*\{[^{}]*\}\s*;", "", body)
        if re.search(r"^\s*Device\s+\w+\s*;", stripped, re.M):
            no = code.count("\n", 0, m.start()) + 1
            findings.append((path, no, "DeviceHolder holds its Device outside a union (destroyed at exit)"))
    return findings


def scan(root):
    findings = []
    for sub in ("src", "include"):
        base = os.path.join(root, sub)
        for dirpath, _, files in os.walk(base):
            for f in sorted(files):
                if not f.endswith((".h", ".cpp")):
                    continue
                p = os.path.join(dirpath, f)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    findings += check_text(os.path.relpath(p, root).replace(os.sep, "/"), fh.read())
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.normpath(os.path.join(HERE, "..", "..")))
    a = ap.parse_args(argv)
    findings = scan(a.src)
    for path, no, text in findings:
        print(f"EXIT-DTOR: {path}:{no} | {text}")
    print(f"check_gpu_no_exit_destructors: {len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
