"""Multi-row GPU plan (rev 1.3.2), slice MR1: the instruments behind cells 4.3, 4.4, 4.5 and 4.7.

Host-only (Linux dxc, python3, mingw-w64 g++); no GPU. test_mr1_cells.py holds the cells.

What the suite assumes about the port (the plan names these; nothing else is assumed):
  - the generator is `gen_wide_shaders.py`, somewhere under src/gpu/ (plan §3.1, E20); the directory that holds
    it is the widened-shader directory, and its output is committed there and never hand-edited (§3.1);
  - the base site shaders are src/gpu/shaders/*.hlsl (§3.1: "run against the base's own site shaders");
  - the widened GEMM is `proj_gemm_wide.hlsl` in that directory (§3.2), one source for the six GEMM sites;
  - the row-variant define is SSLM_WIDE_H and the GEMM-site define SSLM_GEMM_SITE (the spike's names, E17/E20;
    MR1 ports them). A file that mentions SSLM_GEMM_SITE compiles once per site 1..6;
  - variants H in {2, 4, 8} compile at cs_6_2 -O3 -HV 2018 -WX (§3.1, cell 4.3);
  - the host lane plan is `ComputeGpuGemmSiteGroupPlan` (E21, §3.2).
A generated file is recognised by the word GENERATED in its first three lines (the spike generator's header).

Every function here either returns data or raises MR1Red with the reason a cell is red.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_SHADERS = REPO / "src" / "gpu" / "shaders"
GPU_SRC = REPO / "src" / "gpu"
VARIANTS = (2, 4, 8)
GEMM_SITES = (1, 2, 3, 4, 5, 6)
DXC_FLAGS = ("-T", "cs_6_2", "-E", "main", "-O3", "-HV", "2018", "-WX")
GROUPSHARED_LIMIT = 32768  # cell 4.5: bytes at H = 8

# The 19 non-GEMM sites the one-row layer body records and the spike widened (E20). The port keeps them.
SPIKE_SITES = (
    "attn_norm_site", "q_proj_site", "kv_proj_site", "qk_norm_site", "rope_guard_site",
    "rope_commit_site", "attention_score_site", "softmax_site", "context_accumulate_site",
    "ctx_fold_site", "o_proj_site", "attn_residual_site", "mlp_norm_site", "gate_proj_site",
    "up_proj_site", "mlp_act_site", "down_proj_site", "mlp_residual_site", "commit_site",
)
WIDE_GEMM = "proj_gemm_wide.hlsl"


class MR1Red(AssertionError):
    """A cell's red reason."""


def find_dxc() -> str:
    for cand in (os.environ.get("DXC"), shutil.which("dxc"), "/opt/dxc/bin/dxc", "/tmp/claude-0/tools/dxc/bin/dxc"):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    raise MR1Red("no Linux dxc: set DXC, put dxc on PATH, or unpack a DirectXShaderCompiler release to /opt/dxc")


# --------------------------------------------------------------------------------------------- discovery

@dataclass
class WideSet:
    gpu_root: Path          # the src/gpu directory the set lives under (the repo's, or a copy's)
    generator: Path
    wide_dir: Path
    base_dir: Path
    files: list[Path] = field(default_factory=list)       # every widened .hlsl in wide_dir
    generated: list[Path] = field(default_factory=list)   # the subset carrying the GENERATED header

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.gpu_root))


def is_generated(p: Path) -> bool:
    head = p.read_text(encoding="utf-8", errors="replace").split("\n", 3)[:3]
    return any("GENERATED" in ln for ln in head)


def discover(gpu_root: Path | None = None) -> WideSet:
    """MR1_GPU_ROOT overrides the tree (another checkout's src/gpu, e.g. to run the cells against a stand-in)."""
    if gpu_root is None:
        gpu_root = Path(os.environ["MR1_GPU_ROOT"]) if os.environ.get("MR1_GPU_ROOT") else GPU_SRC
    gens = sorted(p for p in gpu_root.rglob("gen_wide_shaders.py") if "__pycache__" not in p.parts)
    if not gens:
        raise MR1Red(f"red-unimplemented: no gen_wide_shaders.py under {gpu_root} (plan §3.1: the generator is "
                     f"ported to main)")
    if len(gens) > 1:
        raise MR1Red(f"more than one gen_wide_shaders.py under {gpu_root}: {[str(g) for g in gens]}")
    gen = gens[0]
    wide_dir = gen.parent
    base_dir = gpu_root / "shaders"
    if wide_dir == base_dir:
        raise MR1Red("the generator sits in src/gpu/shaders: the shipped shader glob (CMakeLists.txt) would compile "
                     "every widened file without SSLM_WIDE_H")
    files = sorted(wide_dir.glob("*.hlsl"))
    ws = WideSet(gpu_root, gen, wide_dir, base_dir, files, [p for p in files if is_generated(p)])
    if not ws.files:
        raise MR1Red(f"red-unimplemented: no widened .hlsl in {wide_dir}")
    return ws


def expected_names_missing(ws: WideSet) -> list[str]:
    have = {p.name for p in ws.files}
    missing = [f"{s}_wide.hlsl" for s in SPIKE_SITES if f"{s}_wide.hlsl" not in have]
    if WIDE_GEMM not in have:
        missing.append(WIDE_GEMM)
    return missing


def variants_of(p: Path) -> list[tuple[str, list[str]]]:
    """(tag, extra dxc args) for every compiled variant of one widened file."""
    text = p.read_text(encoding="utf-8", errors="replace")
    out = []
    for h in VARIANTS:
        if "SSLM_GEMM_SITE" in text:
            for s in GEMM_SITES:
                out.append((f"H{h}_S{s}", ["-D", f"SSLM_WIDE_H={h}", "-D", f"SSLM_GEMM_SITE={s}"]))
        else:
            out.append((f"H{h}", ["-D", f"SSLM_WIDE_H={h}"]))
    return out


# --------------------------------------------------------------------------------------------- dxc

@dataclass
class Compiled:
    src: Path
    tag: str
    ok: bool
    cso: Path | None
    log: str
    disasm: str = ""


def compile_one(ws: WideSet, src: Path, tag: str, extra: list[str], out_dir: Path, disasm: bool = True) -> Compiled:
    dxc = find_dxc()
    out_dir.mkdir(parents=True, exist_ok=True)
    cso = out_dir / f"{src.stem}.{tag}.cso"
    # Relative paths from the gpu root, so two copies of the tree compile with identical arguments.
    args = [dxc, *DXC_FLAGS, *extra, "-I", os.path.relpath(ws.base_dir, ws.gpu_root),
            "-Fo", str(cso), os.path.relpath(src, ws.gpu_root)]
    r = subprocess.run(args, cwd=ws.gpu_root, capture_output=True, text=True)
    c = Compiled(src, tag, r.returncode == 0 and cso.is_file(), cso if cso.is_file() else None, r.stdout + r.stderr)
    if c.ok and disasm:
        d = subprocess.run([dxc, "-dumpbin", str(cso)], capture_output=True, text=True)
        c.disasm = d.stdout
    return c


def compile_all(ws: WideSet, out_dir: Path, disasm: bool = True) -> list[Compiled]:
    res = []
    for f in ws.files:
        for tag, extra in variants_of(f):
            res.append(compile_one(ws, f, tag, extra, out_dir, disasm))
    return res


GS_DECL = re.compile(r"^@\"[^\"]+\"\s*=\s*(?:external\s+)?addrspace\(3\)\s+global\s+(.+?),\s*align", re.M)


def llvm_type_bytes(t: str) -> int:
    t = t.strip()
    m = re.fullmatch(r"\[(\d+)\s+x\s+(.+)\]", t)
    if m:
        return int(m.group(1)) * llvm_type_bytes(m.group(2))
    m = re.fullmatch(r"i(\d+)", t)
    if m:
        return max(1, int(m.group(1)) // 8)
    sizes = {"half": 2, "float": 4, "double": 8}
    if t in sizes:
        return sizes[t]
    m = re.fullmatch(r"<(\d+)\s+x\s+(.+)>", t)
    if m:
        return int(m.group(1)) * llvm_type_bytes(m.group(2))
    m = re.fullmatch(r"\{(.*)\}", t)
    if m:
        return sum(llvm_type_bytes(x) for x in split_top(m.group(1)))
    raise MR1Red(f"groupshared size: unrecognised LLVM type {t!r} (extend llvm_type_bytes)")


def split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[<{":
            depth += 1
        elif ch in "]>}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def groupshared_bytes(disasm: str) -> int:
    """Total addrspace(3) (groupshared) bytes the compiled DXIL declares."""
    return sum(llvm_type_bytes(t) for t in GS_DECL.findall(disasm))


def wave_feature_flags(disasm: str) -> list[str]:
    """The 'shader requires additional functionality' lines that name a wave capability."""
    lines = disasm.split("\n")
    flags, inside = [], False
    for ln in lines:
        if "requires additional functionality" in ln:
            inside = True
            continue
        if inside:
            body = ln.lstrip(";").strip()
            if not body:
                inside = False
                continue
            if re.search(r"wave", body, re.I):
                flags.append(body)
    return flags


# --------------------------------------------------------------------------------------------- generator

def copy_gpu_tree(ws: WideSet, dest: Path) -> WideSet:
    """Copy src/gpu into dest/src/gpu with the generated outputs removed, so the generator must recreate them."""
    target = dest / "src" / "gpu"
    shutil.copytree(ws.gpu_root, target, ignore=shutil.ignore_patterns("__pycache__", "*.cso"))
    copy = WideSet(target, target / ws.generator.relative_to(ws.gpu_root), target / ws.wide_dir.relative_to(ws.gpu_root),
                   target / "shaders")
    for g in ws.generated:
        (copy.wide_dir / g.name).unlink()
    return copy


def regenerate(copy: WideSet) -> WideSet:
    r = subprocess.run(["python3", "-I", str(copy.generator)], cwd=copy.wide_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise MR1Red(f"generator failed (exit {r.returncode}): {r.stdout}{r.stderr}")
    files = sorted(copy.wide_dir.glob("*.hlsl"))
    return WideSet(copy.gpu_root, copy.generator, copy.wide_dir, copy.base_dir, files,
                   [p for p in files if is_generated(p)])


# --------------------------------------------------------------------------------------------- source diff (4.4)

COMMENT_LINE = re.compile(r"^\s*(//.*)?$")
ADDED_OK = [
    re.compile(r"^\s*(//.*)?$"),                                   # blank or comment
    re.compile(r"^\s*#\s*(define|undef|if|ifdef|ifndef|elif|else|endif|include|error|pragma)\b"),
    re.compile(r"^\s*static\s+uint\s+g_row\s*;\s*(//.*)?$"),
    re.compile(r"^\s*RWByteAddressBuffer\s+\w+Rows\s*\[\s*SSLM_WIDE_H\s*\]\s*:\s*register\([^)]*\)\s*;\s*(//.*)?$"),
    re.compile(r"^\s*g_row\s*=\s*\w+\.y\s*;\s*(//.*)?$"),
]
_BASE_DECL = re.compile(r"RWByteAddressBuffer\s+(SeqState|LayerScratch|WorkScratch)\s*:\s*register\(\s*u[013]\s*\)\s*;")
_GEN_DECL = re.compile(r"RWByteAddressBuffer\s+(SeqState|LayerScratch|WorkScratch)Rows\s*\[\s*SSLM_WIDE_H\s*\]\s*:\s*"
                       r"register\([^)]*\)\s*;")
_GEN_GID = re.compile(r",\s*uint3\s+\w+\s*:\s*SV_GroupID\s*(?=\))")


def _norm_base(ln: str) -> str:
    ln = _BASE_DECL.sub(r"<ROWS:\1>", ln)
    if re.search(r"\bvoid\s+main\s*\(", ln):
        ln = _GEN_GID.sub("", ln)
    return re.sub(r"\s+", " ", ln).strip()


def _norm_gen(ln: str) -> str:
    ln = _GEN_DECL.sub(r"<ROWS:\1>", ln)
    if re.search(r"\bvoid\s+main\s*\(", ln):
        ln = _GEN_GID.sub("", ln)
    return re.sub(r"\s+", " ", ln).strip()


def substitution_violations(base_text: str, gen_text: str) -> list[str]:
    """Differences between a generated site and its base shader that are not the generator's substitutions (§3.1;
    the spike generator's four edits): the three per-row buffers become SSLM_WIDE_H-arrays (any register), main
    gains an SV_GroupID parameter, and the only added lines are comments, preprocessor lines, `static uint g_row;`
    and `g_row = <gid>.y;`. Every base code line must survive, in order, up to those substitutions. A base line
    that names a substituted identifier inside an expression is unaffected: the macros carry it. Empty = only the
    substitutions."""
    base = [ln.rstrip() for ln in base_text.split("\n")]
    gen = [ln.rstrip() for ln in gen_text.split("\n")]
    gnorm = [_norm_gen(g) for g in gen]
    out = []
    j = 0
    for i, b in enumerate(base):
        if COMMENT_LINE.match(b):
            continue  # comments are not pinned; a dropped or kept comment changes no code
        nb = _norm_base(b)
        k = j
        while k < len(gen) and gnorm[k] != nb:
            k += 1
        if k == len(gen):
            out.append(f"base code line {i + 1} missing, changed or reordered: {b.strip()[:120]!r}")
            continue
        for g in gen[j:k]:
            if not any(p.match(g) for p in ADDED_OK):
                out.append(f"added line is not a substitution: {g.strip()[:120]!r}")
        j = k + 1
    for g in gen[j:]:
        if not any(p.match(g) for p in ADDED_OK):
            out.append(f"added line is not a substitution: {g.strip()[:120]!r}")
    return out


# --------------------------------------------------------------------------------------------- HLSL lexing

def strip_comments(text: str) -> str:
    """Comments replaced by spaces (newlines kept), string literals kept."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in text[i:j]))
            i = j
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def include_closure(src: Path, search: list[Path]) -> list[Path]:
    seen: list[Path] = []
    stack = [src]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        for inc in INCLUDE.findall(strip_comments(p.read_text(encoding="utf-8", errors="replace"))):
            for d in [p.parent, *search]:
                cand = (d / inc).resolve()
                if cand.is_file():
                    stack.append(cand)
                    break
            else:
                raise MR1Red(f"{src.name}: include {inc!r} not found")
    return seen


# --------------------------------------------------------------------------------------------- wave lint (4.7)

WAVE_CALL = re.compile(r"\b(Wave\w+|QuadRead\w+|QuadAny|QuadAll)\s*\(")
WAVE_ATTR = re.compile(r"\[\s*WaveSize\s*\(")
LANE_LITERAL = re.compile(r"(?<![\w.])(31|32|63|64)(?:[uUlL]{1,3})?(?![\w.])")  # wave sizes 32/64, their masks
# Data-width uses of those literals (cell 4.7's allowlist). Matched against the comment-stripped line.
LANE_LITERAL_ALLOW = [
    (re.compile(r"(<<|>>)\s*\(?\s*(31|32|63|64)\b"), "bit shift by a data width"),
    (re.compile(r"\bLoad\s*<\s*\w+\s*>\s*\(\s*(31|32|63|64)\s*\*\s*4\s*\)"), "layout slot index (slot * 4 bytes)"),
    (re.compile(r"\bRoundingDivideByPot\w*\s*\([^;]*,\s*(31|63)\s*\)"), "power-of-two exponent"),
]


def lane_literal_hits(text: str) -> list[tuple[int, str]]:
    hits = []
    for no, ln in enumerate(strip_comments(text).split("\n"), 1):
        masked = ln
        for rx, _ in LANE_LITERAL_ALLOW:
            masked = rx.sub(lambda m: " " * len(m.group(0)), masked)
        if LANE_LITERAL.search(masked):
            hits.append((no, ln.strip()))
    return hits


BARRIER = re.compile(r"\b(GroupMemoryBarrierWithGroupSync|AllMemoryBarrierWithGroupSync)\s*\(\s*\)")
GS_VAR = re.compile(r"\bgroupshared\s+[\w:<>]+(?:\s+\w+)*?\s+(\w+)\s*(\[[^\]]*\])?\s*;")
ASSIGN_AFTER = re.compile(r"^\s*(\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|=(?!=))")
THREAD_NAMES = {"t", "tid", "gi", "lane", "lane1", "gtid", "dtid", "local", "local_id", "thread"}
TREE_UPDATE = re.compile(r"(>>=\s*1\b|<<=\s*1\b|/=\s*2\b|\*=\s*2\b)")


@dataclass
class Node:
    kind: str                      # "stmt" | "block" | "loop" | "if"
    text: str = ""                 # stmt text, or the loop/if header
    body: list["Node"] = field(default_factory=list)
    orelse: list["Node"] = field(default_factory=list)
    line: int = 0


class _Parser:
    def __init__(self, text: str, line0: int):
        self.t, self.i, self.line0 = text, 0, line0

    def line(self) -> int:
        return self.line0 + self.t.count("\n", 0, self.i)

    def ws(self):
        while self.i < len(self.t):
            if self.t[self.i].isspace():
                self.i += 1
            elif self.t[self.i] == "[" and re.match(r"\[\s*(unroll|loop|branch|flatten|fastopt|allow_uav_condition)\b[^\]]*\]",
                                                     self.t[self.i:]):
                self.i = self.t.index("]", self.i) + 1
            else:
                break

    def balanced(self, open_ch: str, close_ch: str) -> str:
        assert self.t[self.i] == open_ch
        depth, j = 0, self.i
        while j < len(self.t):
            if self.t[j] == open_ch:
                depth += 1
            elif self.t[j] == close_ch:
                depth -= 1
                if depth == 0:
                    s = self.t[self.i:j + 1]
                    self.i = j + 1
                    return s
            j += 1
        raise MR1Red("unbalanced brackets in HLSL")

    def block(self) -> list[Node]:
        nodes = []
        while True:
            self.ws()
            if self.i >= len(self.t) or self.t[self.i] == "}":
                return nodes
            nodes.append(self.stmt())

    def stmt(self) -> Node:
        self.ws()
        ln = self.line()
        rest = self.t[self.i:]
        if rest.startswith("{"):
            self.i += 1
            body = self.block()
            self.i += 1
            return Node("block", body=body, line=ln)
        m = re.match(r"(for|while)\b", rest)
        if m:
            self.i += len(m.group(1))
            self.ws()
            hdr = self.balanced("(", ")")
            body = self.stmt()
            return Node("loop", text=m.group(1) + hdr, body=[body], line=ln)
        if re.match(r"do\b", rest):
            self.i += 2
            body = self.stmt()
            self.ws()
            m2 = re.match(r"while\s*", self.t[self.i:])
            self.i += m2.end()
            hdr = self.balanced("(", ")")
            self.ws()
            if self.i < len(self.t) and self.t[self.i] == ";":
                self.i += 1
            return Node("loop", text="do-while" + hdr, body=[body], line=ln)
        if re.match(r"if\b", rest):
            self.i += 2
            self.ws()
            hdr = self.balanced("(", ")")
            body = self.stmt()
            self.ws()
            orelse = []
            if re.match(r"else\b", self.t[self.i:]):
                self.i += 4
                orelse = [self.stmt()]
            return Node("if", text="if" + hdr, body=[body], orelse=orelse, line=ln)
        # Plain statement up to ';' at paren depth 0.
        depth, j = 0, self.i
        while j < len(self.t):
            c = self.t[j]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif c == ";" and depth == 0:
                break
            elif c == "{" and depth == 0:
                # an initializer list: consume it whole
                save = self.i
                self.i = j
                self.balanced("{", "}")
                j = self.i
                self.i = save
                continue
            j += 1
        s = self.t[self.i:j]
        self.i = j + 1
        return Node("stmt", text=s.strip(), line=ln)


def functions(text: str) -> list[tuple[str, int, list[Node]]]:
    """(name, line, body) for every function body at file scope of comment-stripped text."""
    out = []
    i, depth = 0, 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            if depth == 0:
                head = text[max(0, i - 400):i]
                m = re.search(r"(\w+)\s*\(([^()]|\([^()]*\))*\)\s*(:\s*\w+\s*)?$", head)
                if m and m.group(1) not in ("if", "for", "while", "switch", "register"):
                    p = _Parser(text[i + 1:], text.count("\n", 0, i + 1) + 1)
                    body = p.block()
                    out.append((m.group(1), text.count("\n", 0, i) + 1, body))
                    i = i + 1 + p.i + 1
                    continue
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return out


def _accesses(stmt: str, gs: set[str]) -> list[tuple[str, str, bool]]:
    """(var, index-or-'', is_write) for each groupshared access in one statement."""
    out = []
    for m in re.finditer(r"\b(\w+)\b", stmt):
        v = m.group(1)
        if v not in gs:
            continue
        j = m.end()
        idx = ""
        k = j
        while k < len(stmt) and stmt[k].isspace():
            k += 1
        if k < len(stmt) and stmt[k] == "[":
            depth, e = 0, k
            while e < len(stmt):
                if stmt[e] == "[":
                    depth += 1
                elif stmt[e] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            idx = re.sub(r"\s+", "", stmt[k + 1:e])
            j = e + 1
        before = stmt[:m.start()]
        atomic = re.search(r"\bInterlocked\w+\s*\(\s*$", before) is not None
        write = atomic or ASSIGN_AFTER.match(stmt[j:]) is not None
        out.append((v, idx, write))
        if write and not atomic and not re.match(r"^\s*=(?!=)", stmt[j:]):
            out.append((v, idx, False))  # compound assignment also reads its own slot
    return out


def barrier_violations(text: str) -> list[str]:
    """Groupshared reads that can observe another thread's write with no group barrier between them.

    Model: per function, statements in program order; if-bodies are inlined (both branches), and a barrier inside
    an if whose condition depends on the thread (a thread-index name, a lane name, or a tree loop's stride) does
    not count -- it is divergent, which is no barrier at all; a loop whose update halves or doubles its stride (a tree loop) is walked twice so a
    missing barrier at the end of a step shows on the next step; other loops are walked once. A read of a
    groupshared slot is a violation when, since the last barrier, the variable was written at a different slot
    (or is a scalar written by any thread)."""
    clean = strip_comments(text)
    gs = {m.group(1) for m in GS_VAR.finditer(clean)}
    if not gs:
        return []
    violations: list[str] = []

    def divergent(cond: str, strides: set[str]) -> bool:
        names = set(re.findall(r"\b[A-Za-z_]\w*\b", cond))
        return bool(names & (THREAD_NAMES | strides)) or any(re.search(r"lane|thread|gtid|dtid", n, re.I) for n in names)

    def walk(nodes: list[Node], dirty: dict[str, set[str]], in_if: bool, fn: str, strides: set[str] = frozenset()):
        for nd in nodes:
            if nd.kind == "stmt":
                if BARRIER.search(nd.text):
                    if not in_if:
                        dirty.clear()
                    continue
                acc = _accesses(nd.text, gs)
                for v, idx, w in acc:
                    if not w and v in dirty:
                        others = dirty[v] - {idx} if idx else dirty[v]
                        if others or (not idx and dirty[v]):
                            violations.append(f"{fn}:{nd.line}: reads {v}[{idx}] after a write at "
                                              f"{sorted(dirty[v])} with no barrier between: {nd.text[:100]!r}")
                for v, idx, w in acc:
                    if w:
                        dirty.setdefault(v, set()).add(idx)
            elif nd.kind == "block":
                walk(nd.body, dirty, in_if, fn, strides)
            elif nd.kind == "if":
                div = in_if or divergent(nd.text, strides)
                walk(nd.body, dirty, div, fn, strides)
                walk(nd.orelse, dirty, div, fn, strides)
            elif nd.kind == "loop":
                tree = TREE_UPDATE.search(nd.text) is not None
                st = set(strides)
                if tree:
                    st.update(re.findall(r"(\w+)\s*(?:>>=|<<=|/=|\*=)", nd.text))
                for _ in range(2 if tree else 1):
                    walk(nd.body, dirty, in_if, fn, st)

    for name, _line, body in functions(clean):
        walk(body, {}, False, name)
    # De-duplicate (tree loops are walked twice).
    seen, uniq = set(), []
    for v in violations:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


@dataclass
class LintReport:
    wave_calls: list[str] = field(default_factory=list)
    lane_literals: list[str] = field(default_factory=list)
    barriers: list[str] = field(default_factory=list)
    host: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def red(self) -> list[str]:
        return [*self.wave_calls, *self.lane_literals, *self.barriers, *self.host, *self.flags]


def shipped_closure(base_dir: Path) -> set[Path]:
    """Every file a shipped one-row shader includes (or is): code already shipped and shared with one-row."""
    s: set[Path] = set()
    for f in base_dir.glob("*.hlsl"):
        s.update(include_closure(f.resolve(), [base_dir]))
    return s


def lint_shaders(ws: WideSet) -> LintReport:
    rep = LintReport()
    shipped = shipped_closure(ws.base_dir)
    closure: list[Path] = []
    for f in ws.files:
        for p in include_closure(f.resolve(), [ws.base_dir]):
            if p not in closure:
                closure.append(p)
    for p in closure:
        text = p.read_text(encoding="utf-8", errors="replace")
        clean = strip_comments(text)
        for no, ln in enumerate(clean.split("\n"), 1):
            if WAVE_CALL.search(ln) or WAVE_ATTR.search(ln):
                rep.wave_calls.append(f"{p.name}:{no}: wave intrinsic or attribute: {ln.strip()[:100]!r}")
        # Lane-count literals: the widened-only code (the widened files and any header no shipped shader
        # includes). Shared one-row headers are shipped code; their literals are data widths (E21).
        if p not in shipped:
            for no, ln in lane_literal_hits(text):
                rep.lane_literals.append(f"{p.name}:{no}: lane-count literal: {ln[:100]!r}")
        for v in barrier_violations(text):
            rep.barriers.append(f"{p.name}: {v}")
    return rep


HOST_EXT = (".cpp", ".h", ".inl", ".hpp")


def host_scope(gpu_root: Path) -> list[Path]:
    return sorted(p for p in gpu_root.rglob("*") if p.suffix in HOST_EXT and p.is_file()
                  and "proj_gemm_wide" in p.read_text(encoding="utf-8", errors="replace"))


def _without_plan_function(text: str) -> str:
    """The text with ComputeGpuGemmSiteGroupPlan's own definition body blanked (its literals are the plan)."""
    m = re.search(r"GpuGemmSiteGroupPlan\s+ComputeGpuGemmSiteGroupPlan\s*\([^)]*\)\s*\{", text)
    if not m:
        return text
    i, depth = m.end() - 1, 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[:m.end()] + "".join("\n" if c == "\n" else " " for c in text[m.end():i]) + text[i:]


def lint_host(gpu_root: Path) -> list[str]:
    """The widened GEMM's `lanes` comes only from ComputeGpuGemmSiteGroupPlan (E21, §3.2)."""
    scope = host_scope(gpu_root)
    if not scope:
        return [f"red-unimplemented: no host C++ under {gpu_root} references proj_gemm_wide (the recorder that "
                f"binds the widened GEMM's lanes)"]
    out = []
    for p in scope:
        text = _without_plan_function(strip_comments(p.read_text(encoding="utf-8", errors="replace")))
        if "ComputeGpuGemmSiteGroupPlan(" not in text:
            out.append(f"{p.name}: binds proj_gemm_wide but never calls ComputeGpuGemmSiteGroupPlan")
        for no, ln in enumerate(text.split("\n"), 1):
            # A lanes value set from a literal.
            if re.search(r"\b\w*lanes\w*\s*(=|\{)\s*\(?\s*(static_cast<\w+>\()?\s*\d", ln, re.I):
                out.append(f"{p.name}:{no}: lanes set from a literal: {ln.strip()[:100]!r}")
            # A plan field rewritten after the plan returned it.
            if re.search(r"\.\s*lanes\s*(\+|-|\*|/|<<|>>|\||&)?=(?!=)", ln):
                out.append(f"{p.name}:{no}: a GEMM plan's lanes rewritten: {ln.strip()[:100]!r}")
            # A plan object built by hand instead of by the plan function.
            m = re.search(r"\bGpuGemmSiteGroupPlan\s+(\w+)\s*(;|\{|=(?!\s*ComputeGpuGemmSiteGroupPlan))", ln)
            if m:
                out.append(f"{p.name}:{no}: GpuGemmSiteGroupPlan not taken from ComputeGpuGemmSiteGroupPlan: "
                           f"{ln.strip()[:100]!r}")
    return out


def lint_all(ws: WideSet, compiled: list[Compiled] | None) -> LintReport:
    rep = lint_shaders(ws)
    rep.host = lint_host(ws.gpu_root)
    if compiled is not None:
        for c in compiled:
            for f in wave_feature_flags(c.disasm):
                rep.flags.append(f"{c.src.name} [{c.tag}]: dxc reflection requires {f!r}")
    return rep


def scratch_dir(prefix: str) -> Path:
    base = os.environ.get("MR1_SCRATCH")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


# --------------------------------------------------------------------------------------------- mingw (4.3)

MINGW_FLAGS = ["-std=c++20", "-fsyntax-only", "-Iinclude", "-Isrc", "-Isrc/gpu", "-Itests", "-DNOMINMAX",
               "-DWIN32_LEAN_AND_MEAN", "-include", "d3d12.h", "-include", "d3d12sdklayers.h"]


def host_translation_units(gpu_root: Path) -> list[Path]:
    """The .cpp files that compile the host code binding the widened shaders (a scoped .cpp itself, or every
    .cpp under src/gpu that #includes a scoped header/.inl)."""
    tus: list[Path] = []
    scope = host_scope(gpu_root)
    cpps = sorted(gpu_root.rglob("*.cpp"))
    for p in scope:
        if p.suffix == ".cpp":
            tus.append(p)
            continue
        for c in cpps:
            if re.search(r'#\s*include\s+"([^"]*/)?' + re.escape(p.name) + '"', c.read_text(encoding="utf-8", errors="replace")):
                tus.append(c)
    return sorted(set(tus))


def mingw_typecheck(tu: Path, repo: Path | None = None, extra_defs: list[str] | None = None) -> tuple[bool, str]:
    repo = repo or tu.resolve().parents[2]  # <repo>/src/gpu/<tu>
    cxx = os.environ.get("MR_MINGW_CXX", "x86_64-w64-mingw32-g++-posix")
    if not shutil.which(cxx):
        raise MR1Red(f"mingw-w64 g++ not found ({cxx}): apt-get install g++-mingw-w64-x86-64-posix")
    compat = Path(__file__).resolve().parent / "mingw_d3d12_compat.h"
    defs = [*(extra_defs or []), *os.environ.get("MR1_HOST_DEFS", "").split()]
    args = [cxx, *MINGW_FLAGS, "-include", str(compat), *defs, str(tu.relative_to(repo))]
    r = subprocess.run(args, cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        return False, r.stdout + r.stderr
    # Not vacuous: the code that binds the widened GEMM must survive preprocessing in this configuration (a
    # recorder compiled only under a define nobody sets type-checks nothing).
    pre = subprocess.run([a for a in args if a != "-fsyntax-only"] + ["-E", "-P"], cwd=repo, capture_output=True,
                         text=True)
    if "proj_gemm_wide" not in pre.stdout:
        return False, (f"{tu.name} type-checks, but the code naming proj_gemm_wide is preprocessed out in this "
                       f"configuration (set MR1_HOST_DEFS if the recorder is built under a define)")
    return True, ""


# --------------------------------------------------------------------------------------------- mutants (4.7)

def _tree_loop(text: str) -> re.Match:
    """The first tree loop (stride halving by >>= 1) in the comment-stripped text, with its braced body."""
    clean = strip_comments(text)
    m = re.search(r"for\s*\(\s*uint\s+(\w+)\s*=\s*([^;]+);\s*\1\s*>\s*0u?\s*;\s*\1\s*>>=\s*1\s*\)\s*\{", clean)
    if not m:
        raise MR1Red("mutant: no tree loop `for (uint s = ...; s > 0; s >>= 1) {` in the widened GEMM")
    return m


def _body_end(text: str, open_brace: int) -> int:
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise MR1Red("mutant: unbalanced tree loop body")


def mutate_mx14(text: str) -> str:
    """MX14: the widened tree's reduction step `X[i] += X[i + s]` becomes `X[i] = WaveActiveSum(X[i])`."""
    m = _tree_loop(text)
    end = _body_end(text, m.end() - 1)
    body = text[m.end():end]
    step = re.search(r"(\w+)\[([^\]]+)\]\s*\+=\s*\1\[[^\]]+\]\s*;", body)
    if not step:
        raise MR1Red("mutant MX14: no `X[i] += X[i + s];` step in the tree loop body")
    new_body = body[:step.start()] + f"{step.group(1)}[{step.group(2)}] = WaveActiveSum({step.group(1)}[{step.group(2)}]);" \
        + body[step.end():]
    return text[:m.end()] + new_body + text[end:]


def mutate_mx15(text: str) -> str:
    """MX15: the widened tree's last log2(32) = 5 steps (s = 16, 8, 4, 2, 1) run unrolled with no barrier."""
    m = _tree_loop(text)
    var = m.group(1)
    end = _body_end(text, m.end() - 1)
    body = text[m.end():end]
    tail_body = BARRIER.sub("/* barrier removed (MX15) */ (void)0", body)
    head = text[:m.start()] + text[m.start():m.end()].replace(f"{var} > 0u", f"{var} > 16u").replace(f"{var} > 0", f"{var} > 16u")
    tail = "".join(f"\n        {{ const uint {var} = {k}u; {tail_body} }}" for k in (16, 8, 4, 2, 1))
    return head + body + "}" + tail + text[end + 1:]


def apply_mutant(ws: WideSet, name: str, dest: Path) -> WideSet:
    """A copy of the widened set's tree with mutant `name` applied to the widened GEMM."""
    target = dest / "src" / "gpu"
    shutil.copytree(ws.gpu_root, target, ignore=shutil.ignore_patterns("__pycache__", "*.cso"))
    copy = discover(target)
    gemm = copy.wide_dir / WIDE_GEMM
    text = gemm.read_text(encoding="utf-8")
    mutated = {"MX14": mutate_mx14, "MX15": mutate_mx15}[name](text)
    if mutated == text:
        raise MR1Red(f"mutant {name} changed nothing")
    gemm.write_text(mutated, encoding="utf-8", newline="\n")
    return copy
