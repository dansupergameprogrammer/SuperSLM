"""Cloud checks for the row-widened GPU site shaders (multi-row plan §3.1/§3.2, cells 4.3-4.5, 4.7).

Checks, each selected with --checks (default: all of them, in this order):

  gen          src/gpu/shaders_wide/*_site_wide.hlsl equal a fresh run of gen_wide_shaders.py.
  subst        Each generated shader differs from its one-row source only by the generator's
               substitutions (cell 4.4, "the source diff shows only the substitutions").
  compile      Every widened shader compiles at cs_6_2 -E main -O3 -HV 2018 -WX for H = 2, 4, 8
               (each GEMM site of proj_gemm_wide.hlsl separately), as the build compiles it (4.3).
  determinism  Two regenerations into fresh directories are byte-identical, and two compiles of each
               regenerated variant give byte-identical .cso (4.4).
  groupshared  Each compiled variant's groupshared allocation, read from dxc's disassembly, is at
               most 32,768 bytes; the H = 8 maxima are printed (4.5).
  lint         The wave lint (4.7) over the widened shaders and every header they include:
                 - no Wave*/QuadRead* intrinsic, no WaveGetLaneCount, no [WaveSize];
                 - no 32 or 64 literal outside the allowlist below (data widths only);
                 - every groupshared tree step (a write to a groupshared element that reads another
                   element of the same array) is followed by GroupMemoryBarrierWithGroupSync before
                   that array is touched again or the enclosing loop iterates;
                 - dxc's disassembly of every compiled variant names no wave-operation feature.
  host         The host lane-plan lint (4.7): in the host widened recorder, the widened GEMM's lane
               count comes only from ComputeGpuGemmSiteGroupPlan:
                 - every GpuGemmSiteGroupPlan is initialised by a ComputeGpuGemmSiteGroupPlan call,
                   and none is brace-constructed or left uninitialised;
                 - no `.lanes` is assigned, and every use of the name `lanes` is a member read;
                 - no 32 or 64 literal appears.

Inputs can be redirected so a mutated copy can be checked (the guard-vitality mutants):
  --wide-dir DIR      the widened shader directory (default src/gpu/shaders_wide)
  --include-dir DIR   the one-row shader directory, for #include and for gen/subst
                      (default src/gpu/shaders)
  --host FILE         the host widened recorder (default src/gpu/multirow_wide.inl)
  --dxc PATH          the DirectX Shader Compiler (default $DXC, else /opt/dxc/bin/dxc, else dxc on
                      PATH)
  --out DIR           where compiled .cso land (default a temporary directory)

Exit status 0 when every selected check passes, 1 when any fails, 2 on a usage or tool error.
"""
import argparse
import difflib
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HS = (2, 4, 8)
GEMM_SITES = ("q_proj", "kv_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GROUPSHARED_LIMIT = 32768
ALL_CHECKS = ("gen", "subst", "compile", "determinism", "groupshared", "lint", "host")

# 32 and 64 literals that are data widths, not lane counts. Keyed by header file name and the
# line's text with comments removed and whitespace collapsed; each entry says why it is a width.
LITERAL_ALLOWLIST = {
    ("site_common.hlsli", "for (int i = 0; i < 32; ++i)"):
        "IntSqrtU64Gpu: one iteration per result bit of a 64-bit square root",
    ("site_common.hlsli", "const int64_t kC32_2 = (2 * ((int64_t)32 << 31) + 17) / 34;"):
        "a fixed-point constant (32/17 in Q31), the C++ intmath twin's own literal",
    ("site_common.hlsli", "int64_t delta = ((int64_t)1 << 32) - dn_y;"):
        "2^32, the reciprocal iteration's fixed-point one",
    ("site_common2.hlsli", "if (k >= 64) { U128 r; r.lo = v.hi >> (uint)(k - 64); r.hi = 0ULL; return r; }"):
        "128-bit shift: the 64-bit word width",
    ("site_common2.hlsli", "r.lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));"):
        "128-bit shift: the 64-bit word width",
    ("funnel_guards.hlsli", "uint hi = (uint)(n >> 32);"):
        "the high 32-bit half of a 64-bit value",
    ("funnel_guards.hlsli", "return 32 + (31 - (int)firstbithigh(lo));"):
        "leading-zero count of a 64-bit value: the high half's 32 bits",
    ("wide128.hlsli", 'uint64_t lh = (a & 0xFFFFFFFFULL) * (b >> 32);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'uint64_t hl = (a >> 32) * (b & 0xFFFFFFFFULL);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'uint64_t hh = (a >> 32) * (b >> 32);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'r.lo = (ll & 0xFFFFFFFFULL) | (mid << 32);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'r.hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'if (k >= 64) return v.hi >> (uint)(k - 64);'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'return (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'uint64_t lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'if (k >= 64) { U128 r; r.lo = 0ULL; r.hi = v.lo << (uint)(k - 64); return r; }'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'r.hi = (v.hi << (uint)k) | (v.lo >> (uint)(64 - k));'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'if (k >= 64) { U128 r; r.lo = v.hi >> (uint)(k - 64); r.hi = 0ULL; return r; }'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("wide128.hlsli", 'r.lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));'):
        "128-bit arithmetic: 32-bit halves and 64-bit words",
    ("proj_gemm_wide.hlsl", "return uint2((uint)v, (uint)((uint64_t)v >> 32));"):
        "carry-explicit 64-bit add: the 32-bit halves of a 64-bit value",
    ("proj_gemm_wide.hlsl", "return (int64_t)(((uint64_t)v.y << 32) | (uint64_t)v.x);"):
        "carry-explicit 64-bit add: the 32-bit halves of a 64-bit value",
    ("context_accumulate_site_wide.hlsl", "uint off_ctx_shift = layer_base + Layout.Load<uint>(32 * 4);"):
        "Layout field index 32 (ctx shift table), not a lane count",
}
LITERAL_RE = re.compile(r"(?<![\w.])(32|64)(?:[uUlL]{0,3})(?![\w.])")
WAVE_RE = re.compile(r"\b(Wave[A-Z]\w*|QuadRead\w*|QuadSwap\w*)\b|\[\s*WaveSize\b")
GROUPSHARED_DECL = re.compile(r"\bgroupshared\s+[\w:]+\s+(\w+)\s*(\[|;)")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
BARRIER_RE = re.compile(r"\b(GroupMemoryBarrierWithGroupSync|AllMemoryBarrierWithGroupSync)\s*\(")


def strip_comments(text: str) -> str:
    """Comments (and string literals) blanked to spaces, newlines kept, so offsets are stable."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append('"' + " " * (j - i - 2) + '"' if j - i >= 2 else text[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def find_dxc(arg):
    for cand in (arg, os.environ.get("DXC"), "/opt/dxc/bin/dxc", shutil.which("dxc")):
        if cand and Path(cand).is_file():
            return cand
    return None


def variants(wide_dir: Path):
    """(output name, source path, extra defines) for every widened binary the build makes."""
    sites = sorted(p for p in wide_dir.glob("*_site_wide.hlsl"))
    out = []
    for h in HS:
        for s in sites:
            out.append((f"{s.stem}_h{h}", s, [f"-DSSLM_WIDE_H={h}"]))
        for gi, g in enumerate(GEMM_SITES, start=1):
            out.append((f"{g}_gemm_wide_h{h}", wide_dir / "proj_gemm_wide.hlsl",
                        [f"-DSSLM_WIDE_H={h}", f"-DSSLM_GEMM_SITE={gi}"]))
    return out


def dxc_compile(dxc, src, defs, include_dir, out_cso, disasm=False):
    cmd = [dxc, "-T", "cs_6_2", "-E", "main", "-Fo", str(out_cso), str(src), "-O3", "-HV", "2018", "-WX",
           *defs, "-I", str(include_dir)]
    if disasm:
        cmd += ["-Fc", str(out_cso) + ".asm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


class Report:
    def __init__(self):
        self.failures = []

    def fail(self, check, msg):
        self.failures.append((check, msg))
        print(f"FAIL [{check}] {msg}")

    def ok(self, check, msg):
        print(f"ok   [{check}] {msg}")


# ---- gen / subst ---------------------------------------------------------------------------------

def load_generator(wide_dir: Path, include_dir: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_wide_shaders", wide_dir / "gen_wide_shaders.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SRC = include_dir
    return mod


def check_gen(rep, wide_dir, include_dir):
    try:
        gen = load_generator(wide_dir, include_dir)
        files = gen.generate()
    except SystemExit as e:
        rep.fail("gen", f"generator refused: {e}")
        return
    stale = [f for f, t in files.items()
             if not (wide_dir / f).exists() or (wide_dir / f).read_bytes() != t.encode("utf-8")]
    extra = sorted(p.name for p in wide_dir.glob("*_site_wide.hlsl") if p.name not in files)
    if stale:
        rep.fail("gen", "committed output differs from regeneration: " + ", ".join(stale))
    if extra:
        rep.fail("gen", "widened site shaders the generator does not produce: " + ", ".join(extra))
    if not stale and not extra:
        rep.ok("gen", f"{len(files)} generated shaders equal regeneration")


SUBST_ADDED = (
    re.compile(r"^// GENERATED by gen_wide_shaders\.py from \.\./shaders/\w+\.hlsl -- do not edit by hand\.$"),
    re.compile(r"^// The row-widened variant \(multi-row plan §3\.1\); compile with -D SSLM_WIDE_H=<rows> -I \.\./shaders\.$"),
    re.compile(r"^RWByteAddressBuffer (SeqState|LayerScratch|WorkScratch)Rows\[SSLM_WIDE_H\] : register\(u0, space[123]\);.*$"),
    re.compile(r"^// Row machinery \(generated, gen_wide_shaders\.py\): group row y is sequence row y\.$"),
    re.compile(r"^static uint g_row;$"),
    re.compile(r"^#define (SeqState|LayerScratch|WorkScratch) \1Rows\[g_row\]$"),
    re.compile(r"^#define g_position \(g_position \+ g_row\)$"),
    re.compile(r"^#define g_width \(g_width \+ g_row\)$"),
    re.compile(r"^\s*g_row = \w+\.y;  // dispatch row y is sequence row y$"),
)


def check_subst(rep, wide_dir, include_dir):
    bad = 0
    count = 0
    for wide in sorted(wide_dir.glob("*_site_wide.hlsl")):
        base = include_dir / (wide.name[:-len("_wide.hlsl")] + ".hlsl")
        if not base.exists():
            rep.fail("subst", f"{wide.name}: no one-row source {base.name}")
            bad += 1
            continue
        count += 1
        a = base.read_text(encoding="utf-8").split("\n")
        b = wide.read_text(encoding="utf-8").split("\n")
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            removed, added = a[i1:i2], b[j1:j2]
            for ln in removed:
                if re.search(r"RWByteAddressBuffer\s+(SeqState|LayerScratch|WorkScratch)\s*:\s*register\(u[013]\)", ln):
                    continue
                if re.search(r"void\s+main\s*\(", ln):
                    continue
                rep.fail("subst", f"{wide.name}: one-row line removed or changed: {ln.strip()!r}")
                bad += 1
            for ln in added:
                if any(p.match(ln) for p in SUBST_ADDED):
                    continue
                if re.search(r"void\s+main\s*\(", ln) and "SV_GroupID" in ln:
                    # main's own signature, with SV_GroupID added when absent: the rest must equal
                    # one of the removed lines with that parameter appended.
                    stripped = ln.replace(", uint3 sslm_wide_gid : SV_GroupID", "")
                    if stripped in removed or ln in removed:
                        continue
                if re.match(r"^\s*$", ln):
                    continue
                rep.fail("subst", f"{wide.name}: unexpected line: {ln.strip()!r}")
                bad += 1
    if not bad:
        rep.ok("subst", f"{count} generated shaders differ from their sources only by the substitutions")


# ---- compile / determinism / groupshared --------------------------------------------------------

def check_compile(rep, dxc, wide_dir, include_dir, out_dir, disasm=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, src, defs in variants(wide_dir):
        cso = out_dir / f"{name}.cso"
        code, log = dxc_compile(dxc, src, defs, include_dir, cso, disasm=disasm)
        if code != 0:
            rep.fail("compile", f"{name}: dxc exit {code}: {log[:400]}")
        else:
            built[name] = cso
    if len(built) == len(variants(wide_dir)):
        rep.ok("compile", f"{len(built)} variants compile at cs_6_2 -O3 -HV 2018 -WX (H = 2, 4, 8)")
    return built


def check_determinism(rep, dxc, wide_dir, include_dir):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dirs = []
        for k in (1, 2):
            d = td / f"gen{k}"
            d.mkdir()
            r = subprocess.run([sys.executable, "-I", str(wide_dir / "gen_wide_shaders.py"), "--out", str(d)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                rep.fail("determinism", f"generator run {k} failed: {r.stderr.strip()}")
                return
            shutil.copy(wide_dir / "proj_gemm_wide.hlsl", d / "proj_gemm_wide.hlsl")
            dirs.append(d)
        names = sorted(p.name for p in dirs[0].iterdir())
        if names != sorted(p.name for p in dirs[1].iterdir()):
            rep.fail("determinism", "the two regenerations produced different file sets")
            return
        diff = [n for n in names if not filecmp.cmp(dirs[0] / n, dirs[1] / n, shallow=False)]
        if diff:
            rep.fail("determinism", "regenerated sources differ: " + ", ".join(diff))
        bad = []
        for name, _src, defs in variants(dirs[0]):
            src_name = _src.name
            outs = []
            for k, d in enumerate(dirs, start=1):
                cso = td / f"{name}.{k}.cso"
                code, log = dxc_compile(dxc, d / src_name, defs, include_dir, cso)
                if code != 0:
                    rep.fail("determinism", f"{name} (regeneration {k}): dxc exit {code}: {log[:300]}")
                    return
                outs.append(cso)
            if outs[0].read_bytes() != outs[1].read_bytes():
                bad.append(name)
        if bad:
            rep.fail("determinism", "compiled binaries differ between regenerations: " + ", ".join(bad))
        elif not diff:
            rep.ok("determinism", f"two regenerations: identical sources, identical .cso for "
                                  f"{len(variants(dirs[0]))} variants")


GS_GLOBAL = re.compile(r"addrspace\(3\) global (\[\d+ x [^\]]+\]|i\d+|float|double|half)")
GS_ARRAY = re.compile(r"\[(\d+) x (.+)\]")
SCALAR_BYTES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "half": 2, "float": 4, "double": 8}


def gs_type_bytes(t: str) -> int:
    t = t.strip()
    m = GS_ARRAY.fullmatch(t)
    if m:
        return int(m.group(1)) * gs_type_bytes(m.group(2))
    if t in SCALAR_BYTES:
        return SCALAR_BYTES[t]
    raise ValueError(f"unhandled groupshared type {t!r}")


def groupshared_bytes(asm_text: str) -> int:
    return sum(gs_type_bytes(m.group(1)) for m in GS_GLOBAL.finditer(asm_text))


def check_groupshared(rep, built):
    worst = {}
    bad = 0
    for name, cso in sorted(built.items()):
        asm = Path(str(cso) + ".asm")
        if not asm.exists():
            rep.fail("groupshared", f"{name}: no disassembly")
            bad += 1
            continue
        try:
            nbytes = groupshared_bytes(asm.read_text(encoding="utf-8", errors="replace"))
        except ValueError as e:
            rep.fail("groupshared", f"{name}: {e}")
            bad += 1
            continue
        if nbytes > GROUPSHARED_LIMIT:
            rep.fail("groupshared", f"{name}: {nbytes} bytes of groupshared > {GROUPSHARED_LIMIT}")
            bad += 1
        if name.endswith("_h8"):
            worst[name] = nbytes
    if not bad:
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:3]
        rep.ok("groupshared", f"every variant <= {GROUPSHARED_LIMIT} bytes; H = 8 largest: " +
               ", ".join(f"{n} {b}" for n, b in top))


# ---- the wave lint -------------------------------------------------------------------------------

def lint_sources(wide_dir: Path, include_dir: Path):
    """Every widened source and, transitively, every header it includes (by file name)."""
    seen = {}
    stack = sorted(wide_dir.glob("*_site_wide.hlsl")) + [wide_dir / "proj_gemm_wide.hlsl"]
    while stack:
        p = stack.pop()
        if p.name in seen:
            continue
        text = p.read_text(encoding="utf-8")
        seen[p.name] = (p, text)
        for inc in INCLUDE_RE.findall(text):
            cand = [p.parent / inc, include_dir / inc]
            found = next((c for c in cand if c.exists()), None)
            if found is None:
                raise FileNotFoundError(f"{p.name}: #include \"{inc}\" not found")
            stack.append(found)
    return seen


def _match_paren_back(code: str, close: int) -> int:
    depth = 0
    for k in range(close, -1, -1):
        if code[k] == ")":
            depth += 1
        elif code[k] == "(":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _keyword_before(code: str, pos: int) -> str:
    m = re.search(r"(\w+)\s*$", code[:pos])
    return m.group(1) if m else ""


def _prev_nonspace(code: str, pos: int) -> int:
    k = pos - 1
    while k >= 0 and code[k].isspace():
        k -= 1
    return k


def brace_kinds(code: str):
    """For each '{' offset: (matching '}' offset, True when it opens a loop body)."""
    kinds = {}
    stack = []
    for i, c in enumerate(code):
        if c == "{":
            k = _prev_nonspace(code, i)
            loop = False
            if k >= 0 and code[k] == ")":
                op = _match_paren_back(code, k)
                loop = _keyword_before(code, op) in ("for", "while")
            elif code[max(0, k - 1):k + 1] == "do" and not code[max(0, k - 2):max(0, k - 1)].isalnum():
                loop = True
            stack.append((i, loop))
        elif c == "}" and stack:
            o, loop = stack.pop()
            kinds[o] = (i, loop)
    return kinds


def _statement_start(code: str, end: int) -> int:
    k = end - 1
    depth = 0
    while k >= 0:
        c = code[k]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
        elif depth == 0 and c in ";{}":
            return k + 1
        k -= 1
    return 0


def _unbraced_loop_body(code: str, start: int) -> bool:
    """True when the statement at `start` is a loop's whole body without braces (through any
    number of unbraced `if (...)` heads)."""
    k = _prev_nonspace(code, start)
    while k >= 0 and code[k] == ")":
        op = _match_paren_back(code, k)
        kw = _keyword_before(code, op)
        if kw in ("for", "while"):
            return True
        if kw != "if":
            return False
        k = _prev_nonspace(code, op - len(kw))
    return False


def tree_violations(fname: str, code: str, names):
    """Tree steps on groupshared arrays not followed by a barrier before the array is touched again
    or the enclosing loop iterates."""
    out = []
    kinds = brace_kinds(code)
    for name in names:
        access = re.compile(r"\b" + re.escape(name) + r"\s*\[")
        assign = re.compile(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*(?:[-+*/|&^]|<<|>>)?=(?!=)")
        other_read = re.compile(r"\b" + re.escape(name) + r"\s*\[[^\]]*\+[^\]]*\]")
        for m in assign.finditer(code):
            end = code.find(";", m.end())
            if end < 0:
                continue
            start = _statement_start(code, end)
            # The statement minus its own assignment target: a tree step reads another element.
            stmt = code[start:m.start()] + code[m.end():end]
            if not other_read.search(stmt):
                continue  # a plain store, not a tree step
            line = code.count("\n", 0, m.start()) + 1
            if _unbraced_loop_body(code, start):
                out.append(f"{fname}:{line}: tree step on {name} is an unbraced loop body with no barrier")
                continue
            # Loops enclosing this statement, innermost first.
            enclosing = sorted(((o, c) for o, (c, loop) in kinds.items() if loop and o < start < c),
                               key=lambda oc: -oc[0])
            limit = enclosing[0][1] if enclosing else len(code)
            b = BARRIER_RE.search(code, end, limit)
            a = access.search(code, end, limit)
            if b and (not a or b.start() < a.start()):
                continue
            if a:
                aline = code.count("\n", 0, a.start()) + 1
                out.append(f"{fname}:{line}: tree step on {name} is followed by another access (line {aline}) "
                           f"before any group barrier")
            elif enclosing:
                out.append(f"{fname}:{line}: tree step on {name} reaches the end of its loop body before "
                           f"any group barrier")
    return out


def check_lint(rep, wide_dir, include_dir, built):
    try:
        sources = lint_sources(wide_dir, include_dir)
    except FileNotFoundError as e:
        rep.fail("lint", str(e))
        return
    bad = 0
    gs_names = set()
    stripped = {}
    for fname, (_p, text) in sources.items():
        code = strip_comments(text)
        stripped[fname] = code
        gs_names.update(m.group(1) for m in GROUPSHARED_DECL.finditer(code))
    for fname, code in sorted(stripped.items()):
        for m in WAVE_RE.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            rep.fail("lint", f"{fname}:{line}: wave intrinsic or attribute {m.group(0)!r}")
            bad += 1
        lines = code.split("\n")
        for i, ln in enumerate(lines, start=1):
            if not LITERAL_RE.search(ln):
                continue
            key = (fname, " ".join(ln.split()))
            if key in LITERAL_ALLOWLIST:
                continue
            rep.fail("lint", f"{fname}:{i}: 32/64 literal not on the data-width allowlist: {ln.strip()!r}")
            bad += 1
        for v in tree_violations(fname, code, sorted(gs_names)):
            rep.fail("lint", v)
            bad += 1
    for name, cso in sorted(built.items()):
        asm = Path(str(cso) + ".asm")
        if not asm.exists():
            continue
        text = asm.read_text(encoding="utf-8", errors="replace")
        feats = []
        m = re.search(r"; Note: shader requires additional functionality:\n((?:;.*\n)+?);\s*\n", text)
        if m:
            feats = [ln.strip("; ").strip() for ln in m.group(1).split("\n") if ln.strip("; ").strip()]
        wave = [f for f in feats if "wave" in f.lower()]
        if wave:
            rep.fail("lint", f"{name}: dxc reports wave-operation features: {wave}")
            bad += 1
    if not bad:
        rep.ok("lint", f"{len(sources)} sources ({', '.join(sorted(n for n in sources if n.endswith('.hlsli')))} "
                       f"included); {len(gs_names)} groupshared names; {len(built)} disassemblies without wave "
                       f"features")


def check_host(rep, host: Path):
    if not host.exists():
        rep.fail("host", f"{host}: not found")
        return
    code = strip_comments(host.read_text(encoding="utf-8"))
    bad = 0
    lines = code.split("\n")

    def where(pos):
        return code.count("\n", 0, pos) + 1

    plans = 0
    for m in re.finditer(r"\bGpuGemmSiteGroupPlan\b", code):
        rest = code[m.end():]
        decl = re.match(r"\s*(&|\*)?\s*(\w+)\s*(=|;|\{|\(|,|\))", rest)
        if re.match(r"\s*\{", rest):
            rep.fail("host", f"line {where(m.start())}: GpuGemmSiteGroupPlan brace-constructed")
            bad += 1
            continue
        if not decl:
            continue
        ref, ident, tok = decl.groups()
        if ref:  # a reference or pointer parameter: the plan comes from the caller's call
            continue
        if tok == "=":
            after = rest[decl.end():]
            if not re.match(r"\s*ComputeGpuGemmSiteGroupPlan\s*\(", after):
                rep.fail("host", f"line {where(m.start())}: {ident} is not initialised by "
                                 f"ComputeGpuGemmSiteGroupPlan")
                bad += 1
            else:
                plans += 1
        elif tok in (";", "{"):
            rep.fail("host", f"line {where(m.start())}: {ident} is not initialised by ComputeGpuGemmSiteGroupPlan")
            bad += 1
    for m in re.finditer(r"(\.|->)\s*lanes\s*(?:[-+*/|&^]|<<|>>)?=(?!=)", code):
        rep.fail("host", f"line {where(m.start())}: a plan's lanes is assigned")
        bad += 1
    for m in re.finditer(r"\blanes\b", code):
        before = code[:m.start()].rstrip()
        if before.endswith(".") or before.endswith("->"):
            continue
        rep.fail("host", f"line {where(m.start())}: `lanes` used other than as a plan member: "
                         f"{lines[where(m.start()) - 1].strip()!r}")
        bad += 1
    for i, ln in enumerate(lines, start=1):
        if LITERAL_RE.search(ln):
            rep.fail("host", f"line {i}: 32/64 literal in the host widened recorder: {ln.strip()!r}")
            bad += 1
    if plans == 0 and not bad:
        rep.fail("host", "no ComputeGpuGemmSiteGroupPlan call found: the lint has nothing to check")
        bad += 1
    if not bad:
        rep.ok("host", f"{host.name}: {plans} GEMM plans, all from ComputeGpuGemmSiteGroupPlan; no lane "
                       f"literal; lanes read only as a plan member")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks", default=",".join(ALL_CHECKS))
    ap.add_argument("--wide-dir", default=str(REPO / "src/gpu/shaders_wide"))
    ap.add_argument("--include-dir", default=str(REPO / "src/gpu/shaders"))
    ap.add_argument("--host", default=str(REPO / "src/gpu/multirow_wide.inl"))
    ap.add_argument("--dxc", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[1:])
    checks = [c for c in a.checks.split(",") if c]
    unknown = [c for c in checks if c not in ALL_CHECKS]
    if unknown:
        print(f"unknown check(s): {unknown}", file=sys.stderr)
        return 2
    wide_dir, include_dir = Path(a.wide_dir), Path(a.include_dir)
    needs_dxc = {"compile", "determinism", "groupshared", "lint"} & set(checks)
    dxc = find_dxc(a.dxc)
    if needs_dxc and not dxc:
        print("dxc not found (pass --dxc or set $DXC)", file=sys.stderr)
        return 2
    rep = Report()
    tmp = None
    built = {}
    if "gen" in checks:
        check_gen(rep, wide_dir, include_dir)
    if "subst" in checks:
        check_subst(rep, wide_dir, include_dir)
    if {"compile", "groupshared", "lint"} & set(checks):
        out = Path(a.out) if a.out else Path(tmp := tempfile.mkdtemp(prefix="sslm_wide_"))
        built = check_compile(rep, dxc, wide_dir, include_dir, out)
    if "determinism" in checks:
        check_determinism(rep, dxc, wide_dir, include_dir)
    if "groupshared" in checks:
        check_groupshared(rep, built)
    if "lint" in checks:
        check_lint(rep, wide_dir, include_dir, built)
    if "host" in checks:
        check_host(rep, Path(a.host))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(rep.failures)} failure(s)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
