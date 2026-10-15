#!/usr/bin/env python3
"""The submit-guard census and the token lint (GPU paged-KV plan, step G3; plan §4.4 *The census*, §8, 11.10).

    python3 tests/ci/check_gpu_pool_submit_guard.py --src <tree> --plan <plan.md> [--json]
    python3 tests/ci/check_gpu_pool_submit_guard.py [--lint-only] [--src <tree>] [--json]   (no --plan: the lint alone)

Two instruments share this script and its record format, and nothing else:

* **The token lint** ("sink call outside the harness"), P1's instrument and a G3 exit criterion. It reads the code
  tokens of every file under src/ and include/ (comments, string and character literals and preprocessor lines
  stripped) and reports every ExecuteCommandLists, CreateCommandAllocator, CreateCommandQueue or D3D12CreateDevice
  token outside src/gpu/d3d12_harness.h, so every device, allocator and queue stays inside the harness, where the
  submission ledger keeps P1. tests/ and tools/ are outside its scope (te425's device fixtures and
  tools/t2116_list_adapters.cpp build their own devices on purpose), and the injected-fault string literal in
  superslm_gpu.cpp that names ExecuteCommandLists is stripped. A creation call whose name is built by token pasting
  or stringizing is outside what it reads. --lint-only runs it alone and needs no plan; CI runs that form.

* **The census**, advisory: a lint that reports its named reasons, on which no promise and no gate rests (plan §4.4,
  rev 11; triage round 3, D1). Its derivation (population, whole-tree graph, sinks by method name, allocator
  classification, the handle's state by token, the in-flight holders) is the plan probe's own algorithm
  (probes/plan_probes.py, section 5A), carried here with the G3 adaptations each marked "G3:" in the code: the
  ledger's entry points BeginRecording() and Submit() are the sinks a recording site reaches, and the harness's own
  D3D12 calls behind them are the ledger's inside; the counting accessor SetGpuSeqState raises and lowers the counts
  with the state; the token (GpuInFlightToken::Make, a GpuInFlightToken* out-parameter, a GpuInFlightToken&&
  consumer) replaces the raw record. Its rows are §4.4's tables, parsed from --plan, so there is no second copy.
  The G3 built-tree reasons it adds: "reset site without `PoolSubmitAssert`" (a raw allocator Reset() outside the
  ledger's ResetAllocatorAndList, or that function called from anywhere but BeginRecording), and T1-T3 on the token.
  Not implemented here, and reported as such in the summary line: "state change before `EntrySubmitGuard`" and
  "pure helper writes outside the caller's locals" (the guard's placement is carried by 3.6d's runs, P2 (b)).

Output: one record per error, "CENSUS: <reason> | <entry> | <where> | <detail>", or with --json one object per line
with the keys reason, entry, where and detail (the census battery runner's convention). Exit 1 if any record was
printed, else 0. Summary lines go to stderr.
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL6 = []        # the probe's collector for table-parse anomalies (s44_table7); reported as records
PLAN = ""
D7 = None
HDIR = "include/superslm"
INTERNAL = {"gpu_port.h"}                      # the internal layer (s4.4, rev 7 narrowing): rows there may be "internal"
CKW = {"if", "for", "while", "switch", "return", "sizeof", "catch", "decltype", "static_cast", "reinterpret_cast",
       "const_cast", "dynamic_cast", "alignof", "noexcept", "new", "delete", "throw", "defined", "assert",
       "static_assert", "alignas", "operator"}
KW7 = CKW | {"int", "char", "void", "bool", "short", "long", "float", "double", "unsigned", "signed", "const", "volatile",
             "auto", "struct", "class", "enum", "union", "typedef", "using", "template", "typename", "extern", "static",
             "inline", "constexpr", "namespace", "friend", "virtual", "explicit", "int8_t", "int16_t", "int32_t",
             "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t"}
def cstrip(src):                               # comments, string and char literals, preprocessor lines -> blanks
    out = []; i = 0; n = len(src)             # (rev 7: newlines kept inside continued directives, so line numbers hold)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i); j = n if j < 0 else j; out.append(" " * (j - i)); i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j < 0 else j + 2; out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c: j += 2 if src[j] == "\\" else 1
            out.append(c + re.sub(r"[^\n]", " ", src[i + 1:j]) + c if j < n else ""); i = j + 1
        elif c == "#" and (i == 0 or src[i - 1] == "\n" or src[:i].split("\n")[-1].strip() == ""):
            j = src.find("\n", i); j = n if j < 0 else j
            while j < n and src[j - 1] == "\\":
                j2 = src.find("\n", j + 1); j = n if j2 < 0 else j2
            out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        else: out.append(c); i += 1
    return "".join(out)
FNSPANS = {}                                   # rev 7.1: each file's function-body spans, for cvars
FNHEADS = {}                                   # rev 8: each function's head (its parameter list), by name
def cmacros(text, path):
    """Rev 8 (coverage audit round 6, F1). Every #define, function-like or object-like, as a node named by the macro,
    whose edges are the names its replacement text uses. (name, path, line, body)."""
    out = []; lines = text.split("\n"); i = 0
    while i < len(lines):
        m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)(\([^)]*\))?(.*)$", lines[i]); j = i
        if m:
            body = m.group(3)
            while lines[j].rstrip().endswith("\\") and j + 1 < len(lines): j += 1; body += " " + lines[j]
            body = re.sub(r"//.*|/\*.*?\*/", " ", body)
            out.append((m.group(1), path, i + 1, body))
        i = j + 1
    return out
VARKW = CKW | {"namespace", "struct", "class", "enum", "union", "const", "constexpr", "noexcept", "override", "final",
               "mutable", "try", "default", "else", "do", "public", "private", "protected", "typename", "template", "using"}
def cvars(text, path):
    """Rev 7.1 (the census battery's I1). Every initializer outside a function body: a namespace-scope or file-scope
    variable, a static data member (in the class or out of it), a constexpr table, a lambda stored in a global. Each
    becomes a node named by its variable, whose edges are the names its initializer uses; a function that names the
    variable inherits them. (name, path, line, initializer)."""
    s = list(cstrip(text))
    for a, b in FNSPANS.get(path, []):                        # blank every function body, keeping its braces
        for i in range(a + 1, b): s[i] = s[i] if s[i] == "\n" else " "
    s = "".join(s); out = []; nl = [m.start() for m in re.finditer("\n", s)]
    import bisect
    depth = []; d = 0                                         # paren depth at each offset, so parameter defaults are skipped
    for ch in s:
        depth.append(d); d += (ch == "(") - (ch == ")")
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]\n]*\]\s*)*(=(?!=)|\{)", s):
        name, i = m.group(1), m.start()
        if name in VARKW or depth[i] > 0 or s[max(0, i - 1)] in ".>": continue
        if i >= 1 and s[i - 1] in "=!<>+-*/%&|^": continue
        st = max(s.rfind(";", 0, i), s.rfind("{", 0, i), s.rfind("}", 0, i)) + 1; pre = s[st:i]
        if re.search(r"\b(namespace|struct|class|union|enum|typename|template|using|operator|return)\b", pre): continue
        if re.search(r"\)\s*(?:const\s*)?(?:noexcept\s*)?:", pre) or pre.count("<") > pre.count(">"): continue
        if m.group(2) == "{" and not pre.strip(): continue       # a bare name before a brace is not a declarator
        j, k = m.end() - (m.group(2) == "{"), 0
        while j < len(s):                                     # the initializer, to the ';' or ',' at its own depth
            c = s[j]
            if c in "({[": k += 1
            elif c in ")}]":
                k -= 1
                if k < 0: break
            elif c in ";," and k == 0: break
            j += 1
        out.append((name, path, bisect.bisect_right(nl, i) + 1, s[m.end() - (m.group(2) == "{"):j]))
    return out
def cfunctions(text, path):                    # (name, path, line, body) for every function definition
    s = cstrip(text); nl = [m.start() for m in re.finditer("\n", s)]
    import bisect
    funcs, stack, start = [], [], 0; FNSPANS[path] = []
    for t in re.finditer(r"[{};]", s):
        p, ch = t.start(), t.group()
        infn = bool(stack) and stack[-1][0] in ("fn", "blk")
        if ch == ";":
            if not infn: start = p + 1
            continue
        if ch == "{":
            if infn: stack.append(("blk",)); continue
            head = s[start:p]
            m = re.search(r"(~?[A-Za-z_]\w*)\s*\((?:[^()]|\([^()]*\))*\)\s*(?:const\s*)?(?:noexcept(?:\([^)]*\))?\s*)?"
                          r"(?:override\s*)?(?:->\s*[\w:<>\s\*&]+)?\s*(?::[^{]*)?$", head, re.S)
            if m and m.group(1) not in CKW and not re.search(r"\b(namespace|struct|class|enum|union)\b[^(]*$", head.split("(")[0]):
                stack.append(("fn", p, m.group(1), bisect.bisect_right(nl, p) + 1, head))
            else: stack.append(("ns",))
            start = p + 1
        else:
            if not stack: continue
            top = stack.pop()
            if top[0] == "fn":
                funcs.append((top[2], path, top[3], s[top[1]:p])); FNSPANS.setdefault(path, []).append((top[1], p))
                FNHEADS.setdefault(top[2], []).append(top[4])
            if not stack or stack[-1][0] not in ("fn", "blk"): start = p + 1
    return funcs
def tree_files(root, d, exts):
    out = []
    for dp, dn, fn in os.walk(os.path.join(root, d)):
        for f in fn:
            if f.rsplit(".", 1)[-1] in exts: out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)
def decls7(text):
    """The declarator parser. Statements at a transparent scope (the file, a namespace, extern "C"); a statement is a
    function declaration when its first identifier followed by '(' at parenthesis depth 0 is not a keyword and only
    qualifiers follow the matching ')'. An inline definition counts as a declaration. Any return type, any linkage."""
    s = cstrip(text); out = []; stack = []; start = 0; pd = 0
    def opaque(): return any(k != "t" for k in stack)
    def consider(stmt):
        st = stmt.strip()
        if not st or re.match(r"(typedef|using|static_assert|friend|template\s*<[^>]*>\s*(class|struct))\b", st): return
        depth = 0; cand = None
        for m in re.finditer(r"[()]|(operator\s*\S+?|~?[A-Za-z_]\w*(?:\s*::\s*~?[A-Za-z_]\w*)*)\s*(?=\()", st):
            t = m.group(0)
            if t == "(": depth += 1; continue
            if t == ")": depth -= 1; continue
            if depth == 0 and m.group(1) and cand is None:
                nm = re.sub(r"\s+", "", m.group(1)).split("::")[-1]
                if nm not in KW7: cand = (nm, m.end())
        if not cand or "=" in st[:cand[1]]: return
        j, d = cand[1], 0
        while j < len(st):
            if st[j] == "(": d += 1
            elif st[j] == ")":
                d -= 1
                if d == 0: break
            j += 1
        if re.fullmatch(r"(\s|const|noexcept(\s*\([^)]*\))?|override|final|&&?|->\s*[\w:<>\s\*&,]+|=\s*(delete|default|0)|[A-Z_][A-Z0-9_]*(\([^)]*\))?)*", st[j + 1:]):
            out.append(cand[0])
    for i, c in enumerate(s):
        if c == "(": pd += 1
        elif c == ")": pd -= 1
        if pd > 0 or c not in "{};": continue
        if c == ";":
            if not opaque(): consider(s[start:i]); start = i + 1
        elif c == "{":
            head = s[start:i]
            if opaque(): stack.append("o")
            elif re.search(r"\bnamespace\b[^;{}()]*$", head) or re.search(r'\bextern\s*"\s*"\s*$', head): stack.append("t"); start = i + 1
            elif re.search(r"\b(struct|class|union|enum)\b", head.split("(")[0]) or re.search(r"=\s*$", head): stack.append("o")
            else: consider(head); stack.append("f")
        else:
            if stack: stack.pop()
            if not opaque(): start = i + 1
    return set(out)
def loose7(text):
    """The cross-check, sharing no grammar with decls7: every identifier token followed by '(' at parenthesis depth 0
    of a transparent scope, keywords removed."""
    s = cstrip(text); out = set(); stack = []; depth = 0
    for m in re.finditer(r"[{}()]|~?[A-Za-z_]\w*(?=\s*\()", s):
        t = m.group(0)
        if depth > 0 and t in "{}": continue
        if t == "{":
            stack.append("t" if not any(x != "t" for x in stack) and re.search(r'(namespace\b[^;{}()]*|extern\s*"\s*")\s*$', s[max(0, m.start() - 200):m.start()]) else "o"); continue
        if t == "}": stack and stack.pop(); continue
        if t == "(": depth += 1; continue
        if t == ")": depth -= 1; continue
        if depth == 0 and not any(x != "t" for x in stack) and t not in KW7: out.add(t)
    return out
SITE7 = re.compile(r"(?:\b(\w+)\s*(?:->|\.)\s*)?\b(alloc\w*\s*(?:->|\.)\s*Reset|queue\w*\s*(?:->|\.)\s*ExecuteCommandLists)\s*\(")
def dev_kind(expr):
    expr = expr or ""
    if re.search(r"\bGetDevice\s*\(\s*\)", expr): return "singleton"
    if re.search(r"(?:->|\.)\s*device\b", expr): return "per-context"
    return None
def notsink_rows(text):                         # rev 10: s4.4's "not a sink" table, {(file, expression): count}
    out = {}
    for m in re.finditer(r"^\s*\| `([^`]+)` \| `([\w.]+)` \| (\d+) \| [^|]* \| not a sink \|\s*$", text, re.M):
        out[(m.group(2), re.sub(r"\s+", "", m.group(1)))] = int(m.group(3))
    return out
def notstate_rows(text):                        # rev 10: s4.4's "not the handle's state" table, {(file, expression): count}
    out = {}
    for m in re.finditer(r"^\s*\| `([^`]+)` \| `([\w.]+)` \| (\d+) \| [^|]* \| not the handle's state \|\s*$", text, re.M):
        out[(m.group(2), re.sub(r"\s+", "", m.group(1)))] = int(m.group(3))
    return out
def derive7(root):
    hdrs = tree_files(root, HDIR, {"h", "hpp", "inc", "def"}); srcs = tree_files(root, "src", {"cpp", "cc", "h", "hpp", "inl", "inc", "def"})   # every header, the X-macro files too
    D = {"hdrs": hdrs, "pop": {}, "missed": [], "defs": {}, "calls": {}, "sites": [], "sitekind": {}}
    for h in hdrs:
        t = open(os.path.join(root, h), encoding="utf-8", errors="replace").read()
        D["pop"][os.path.basename(h)] = (decls7(t), loose7(t))
    D["vars"] = {}; D["macros"] = {}; FNHEADS.clear()
    for f in srcs + hdrs:
        txt = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        for name, path, ln, body in cfunctions(txt, f):
            D["defs"].setdefault(name, []).append((path, ln, body))
        for name, path, ln, init in cvars(txt, f):              # rev 7.1: initializers outside any function body
            D["vars"].setdefault(name, []).append((path, ln, init))
        for name, path, ln, body in cmacros(txt, f):            # rev 8: macros (coverage audit round 6, F1)
            D["macros"].setdefault(name, []).append((path, ln, body))
    allnames = set(D["defs"]) | set(D["vars"]) | set(D["macros"])
    D["nodes"] = {n: list(D["defs"].get(n, [])) + list(D["vars"].get(n, [])) + list(D["macros"].get(n, [])) for n in allnames}
    # rev 8 (coverage audit round 6, F2 and F3): a store of a function's name, not a call, into any lvalue (a variable,
    # a field, an element) makes the lvalue's last name a node with an edge to the function; any body that names the
    # lvalue inherits the edge. The storer keeps its own token edge, as before.
    FN = set(D["defs"]); D["stores"] = {}
    STORE = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?=(?!=)\s*(&\s*)?((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*(?=[;,)}])")
    for name, lst in list(D["nodes"].items()):
        for path, ln, body in lst:
            for m in STORE.finditer(body):
                tgt = m.group(3).split("::")[-1]          # '&f', or a bare function name that no variable shares
                if tgt in FN and m.group(1) not in CKW and m.group(1) != tgt and (m.group(2) or tgt not in D["vars"] and "::" in m.group(3) or len(tgt) >= 12):
                    D["stores"].setdefault(m.group(1), set()).add(tgt)
    for x, tg in D["stores"].items():
        D["nodes"].setdefault(x, []).append(("(stores)", 0, " ".join(sorted(tg))))
    ID = re.compile(r"\b[A-Za-z_]\w*\b")
    for name, lst in D["nodes"].items():                       # an edge is any token naming a function or such a variable
        D["calls"][name] = {t for _, _, body in lst for t in set(ID.findall(body)) if t in D["nodes"] and t != name and t not in CKW}
    def params_of(fn):                                   # rev 8: the parameter names of fn, in order (its first definition)
        for head in FNHEADS.get(fn, [])[:1]:
            m = re.search(r"\((.*)\)[^()]*$", head, re.S)
            if not m: return []
            out, d, cur = [], 0, ""
            for ch in m.group(1):
                if ch in "(<[": d += 1
                elif ch in ")>]": d -= 1
                if ch == "," and d == 0: out.append(cur); cur = ""
                else: cur += ch
            out.append(cur)
            return [(re.findall(r"([A-Za-z_]\w*)\s*(?:=[^,]*)?$", a.strip()) or [""])[0] for a in out]
        return []
    def call_args(body, start):                          # the arguments of the call whose '(' ends at start
        out, d, cur = [], 0, ""
        for ch in body[start:]:
            if ch in "([{": d += 1
            elif ch in ")]}":
                if d == 0: break
                d -= 1
            if ch == "," and d == 0: out.append(cur); cur = ""
            else: cur += ch
        out.append(cur); return [a.strip() for a in out]
    def kinds_of(fn, recv, depth=0):
        for path, ln, body in D["nodes"].get(fn, []):
            if recv:
                m = re.search(r"\b" + re.escape(recv) + r"\s*=\s*([^;]*);", body)
                if m and dev_kind(m.group(1)): return {dev_kind(m.group(1))}
                if m:                                    # rev 8: an alias (a = d.alloc.Get()): trace its device object
                    mm = re.search(r"\b(\w+)\s*(?:->|\.)\s*(?:alloc|queue)\w*\b", m.group(1))
                    if mm and mm.group(1) != recv: return kinds_of(fn, mm.group(1), depth + 1)
                    return {"unclassified"}
        if depth > 4: return {"unclassified"}
        out = set(); ps = params_of(fn)
        idx = ps.index(recv) if recv and recv in ps else 0   # rev 8 (R6-N2): the argument at the parameter's position
        for caller, cs in D["calls"].items():
            if fn not in cs: continue
            for path, ln, body in D["nodes"].get(caller, []):
                for m in re.finditer(r"\b" + re.escape(fn) + r"\s*\(", body):
                    if not recv:                         # a harness::Device member: the receiver object
                        mm = re.search(r"([\w:]+\s*\(\s*\)|\w+(?:\s*(?:->|\.)\s*\w+)*)\s*(?:\.|->)\s*$", body[max(0, m.start() - 80):m.start()])
                        arg = mm.group(1) if mm else ""
                    else:                                # a parameter: the call's argument at its position
                        args = call_args(body, m.end()); arg = args[idx] if idx < len(args) else ""
                    k = dev_kind(arg)
                    if k: out.add(k)
                    elif re.fullmatch(r"\w+", arg.strip()): out |= kinds_of(caller, arg.strip(), depth + 1)
                    else: out.add("unclassified")
        return out or {"unclassified"}
    # rev 10 (prosecution round 5, F9-3; plan review round 8, W1; the coordinator's direction): a sink is recognised by
    # its method-name token, not by its receiver's syntax. Every code token Reset(, ExecuteCommandLists( or Signal( in
    # any body, initializer or macro is a sink site, whatever stands in front of it (->, ., a reference, a dereference,
    # an element, a call, a cast, or nothing). Rounds 3 to 5 each repaired the one receiver form they executed (a named
    # receiver, then .Get()->, then . and [i]->), and the next form passed; matching the name was rev 10's answer to that.
    # rev 11 (triage round 3, D1; prosecution round 7 s4.1): the census is an advisory lint on its named reasons. Rev 10's
    # "removes the enumeration" and "fails closed" are retracted as claims: it reports what it names and claims no more.
    # Over-matching is safe: a same-named method of another type (a ComPtr's Reset(), which releases a pointer; the
    # hash's Reset()) is a sink until s4.4's "not a sink" table names its site, and a row that names no site is stale.
    # Each site is classified by the device it runs on where the receiver is one of three traceable forms (a Device
    # member, an object's alloc/queue/list/fence member, or a local alias or reference to one); any other receiver
    # is a sink whose allocator is unclassified, which the census reports while still counting its reach.
    SINKTOK = re.compile(r"\b(Reset|ExecuteCommandLists|Signal|BeginRecording|Submit)\s*\(")
    # G3: the ledger's own D3D12 calls, reached only through BeginRecording() and Submit() (d3d12_harness.h); each is
    # recorded in D["ledger_inside"] and is not a sink of the graph (the entry points above are)
    LEDGER_INSIDE = {"ResetAllocatorAndList", "ExecuteRecordedList", "SignalFence", "CloseListConfirmed"}
    D["ledger_inside"] = []; D["raw_resets"] = []
    NS = notsink_rows(PLAN) if "PLAN" in globals() else {}
    D["notsink_seen"] = {}
    def receiver_of(body, pos):                  # the postfix expression in front of '.' or '->' ending at pos
        j = pos
        while j > 0 and body[j - 1].isspace(): j -= 1
        if body[max(0, j - 2):j] == "->": op = "->"; j -= 2
        elif body[j - 1:j] == ".": op = "."; j -= 1
        else: return "", ""
        end = j
        while True:
            while j > 0 and body[j - 1].isspace(): j -= 1
            if j > 0 and body[j - 1] in ")]":
                close = body[j - 1]; opn = "(" if close == ")" else "["; d = 0; k = j - 1
                while k >= 0:
                    if body[k] == close: d += 1
                    elif body[k] == opn:
                        d -= 1
                        if d == 0: break
                    k -= 1
                j = k
                while j > 0 and (body[j - 1].isalnum() or body[j - 1] == "_"): j -= 1   # a call's callee name
            elif j > 0 and (body[j - 1].isalnum() or body[j - 1] == "_"):
                while j > 0 and (body[j - 1].isalnum() or body[j - 1] == "_"): j -= 1
            else: break
            k = j
            while k > 0 and body[k - 1].isspace(): k -= 1
            if body[max(0, k - 2):k] in ("->", "::"): j = k - 2; continue
            if body[k - 1:k] == "." : j = k - 1; continue
            break
        return re.sub(r"\s+", "", body[j:end]), op
    for name, lst in D["nodes"].items():
        for path, ln, body in lst:
            for m in SINKTOK.finditer(body):
                recv, op = receiver_of(body, m.start())
                args = call_args(body, m.end()); meth = m.group(1)
                if meth in ("BeginRecording", "Submit") and (os.path.basename(path) == "submission_ledger.h" or re.search(r"(?:\bvoid|\bGpuSubmission|::)\s*$", body[:m.start()])):
                    continue                             # G3: the ledger's own definitions, not call sites
                if name in LEDGER_INSIDE and os.path.basename(path) == "d3d12_harness.h":
                    D["ledger_inside"].append((meth, f"{os.path.basename(path)}:{ln + body.count(chr(10), 0, m.start())}", name)); continue
                key = (os.path.basename(path), recv + op + meth + ("()" if args == [""] else "(...)"))
                if key in NS and D["notsink_seen"].get(key, 0) < NS[key]:
                    D["notsink_seen"][key] = D["notsink_seen"].get(key, 0) + 1; continue
                what = {"ExecuteCommandLists": "exec", "Signal": "signal", "BeginRecording": "reset", "Submit": "exec"}.get(meth, "reset" if args == [""] else "record")
                if meth == "Reset" and args == [""] and op == "->" and re.search(r"alloc\w*$", recv):    # G3: a raw allocator reset (a ComPtr's own .Reset() releases it)
                    D["raw_resets"].append((f"{os.path.basename(path)}:{ln + body.count(chr(10), 0, m.start())}", name))
                ps = params_of(name)
                if dev_kind(recv): k = {dev_kind(recv)}                            # G3: GetDevice().X, ctx->device->X
                elif meth in ("BeginRecording", "Submit") and recv == "": k = kinds_of(name, "")   # a Device member's own call
                elif re.match(r"\(?\*?(?:alloc|queue|list|fence)\w*\b", recv): k = kinds_of(name, "")            # a Device member
                elif re.search(r"\b(\w+)(?:->|\.)(?:alloc|queue|list|fence)\w*", recv):
                    k = kinds_of(name, re.search(r"\b(\w+)(?:->|\.)(?:alloc|queue|list|fence)\w*", recv).group(1))
                elif re.fullmatch(r"\(?\*?(\w+)\)?", recv) and (re.fullmatch(r"\(?\*?(\w+)\)?", recv).group(1) in ps or re.search(r"\b" + re.fullmatch(r"\(?\*?(\w+)\)?", recv).group(1) + r"\s*=", body)):
                    k = kinds_of(name, re.fullmatch(r"\(?\*?(\w+)\)?", recv).group(1))
                else: k = {"unclassified"}
                line = ln + body.count("\n", 0, m.start())
                D["sites"].append((what, f"{os.path.basename(path)}:{line}", name, tuple(sorted(k))))
                D["sitekind"].setdefault(name, set()).update(k)
    for h, (strict, loose) in D["pop"].items():            # a declaration of a defined function that the parser missed
        D["missed"] += [(h, n) for n in sorted(loose - strict) if n in D["defs"]]
    # rev 7 (the battery's C2; prosecution round 2 s7c): every write of the Submitted state, and whether the same
    # function raises the process count. A write in a function that raises no count is a Submitted state the guard
    # cannot see.
    # rev 8 (coverage audit round 6, F5 and F6): every use of the Submitted enumerator that is not a comparison
    # (after == or !=) or a case label counts as a write, so a write through a helper's argument is seen. A body is
    # uncounted when its net Submitted writes (Submitted uses minus writes of Idle) exceed its net count raises
    # (raises minus lowers), so a raise and a lower in the same body with no Idle restore is no raise.
    SUBU = re.compile(r"(==|!=|\bcase)?\s*(?:[\w]+::)*\bSubmitted\b"); IDLEW = re.compile(r"(?<![=!<>])=\s*(?:[\w]+::)*\bIdle\b")
    RAISE = re.compile(r"\+\+\s*g_process_wide_submitted_sequences\b|\bg_process_wide_submitted_sequences\s*(?:\+=|\+\+|\.fetch_add)"
                       r"|\bSetGpuSeqState\s*\([^;]*?\bSubmitted\b")       # G3: the counting accessor raises with the state
    LOWER = re.compile(r"--\s*g_process_wide_submitted_sequences\b|\bg_process_wide_submitted_sequences\s*(?:-=|--|\.fetch_sub)"
                       r"|\bSetGpuSeqState\s*\([^;]*?\b(?:Idle|Completed)\b")
    # rev 10 (prosecution round 5, F9-5; the coordinator's direction): the handle's state field is recognised by its
    # token, not by the syntax of a write. SslmGpuSequenceHandle is defined in one source file and is opaque in every
    # header, so only that file can name its members; the census checks that this is still so. In that file every
    # code token `state` used as a member (after . or ->) or as an offsetof designator is a use of the field, and a use
    # counts as a Submitted write unless it is one of three safe forms: an operand of == or != (or of switch), a
    # write of the named Idle or Completed enumerator, or a write of the named Submitted enumerator (which SUBU already
    # counts). So an integer, a cast, an address taken, a reference bound, an argument, a memcpy or an offsetof is a
    # write. Rev 9 listed the unsafe forms (state =, static_cast) and the next form passed; rev 10 lists the safe ones.
    # A member of another struct named `state` in that file is a write until s4.4's "not the handle's state" table
    # names it. rev 11 (triage round 3, D1 and D2): advisory. From G3 the compiler carries the property through a
    # private state and a counting accessor (s4.4 Three states). A value cast of the field, a read, is still counted
    # here (the battery's rev 10 must-accept M1(b)); its fix rides with L-res.
    D["handle_files"] = sorted(f for f in srcs + hdrs if re.search(r"\bstruct\s+SslmGpuSequenceHandle\b[^;{()]*\{", cstrip(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())))
    NSt = notstate_rows(PLAN) if "PLAN" in globals() else {}
    D["notstate_seen"] = {}
    STATEU = re.compile(r"(?:->|\.)\s*(state)\b|\boffsetof\s*\([^,()]*,\s*(state)\b")
    def state_writes(path, body):
        out = []
        if path not in D["handle_files"]: return out
        for m in STATEU.finditer(body):
            if m.group(2): out.append(m); continue
            recv, op = receiver_of(body, m.start(1))
            k = m.start(); left = len(recv)
            while k > 0 and left > 0:
                k -= 1
                if not body[k].isspace(): left -= 1
            prv = body[:k].rstrip(); nxt = body[m.end():m.end() + 120]
            if re.match(r"\s*(==|!=)", nxt) or prv.endswith(("==", "!=")) or re.search(r"\bswitch\s*\(\s*$", prv): continue
            if re.match(r"\s*=\s*(?:\w+::)*(?:Idle|Completed|Submitted)\s*;", nxt): continue
            if re.match(r"\s*\(\s*\)", nxt): continue          # G3: the state() reader, which returns a copy
            key = (os.path.basename(path), recv + op + "state")
            if key in NSt and D["notstate_seen"].get(key, 0) < NSt[key]:
                D["notstate_seen"][key] = D["notstate_seen"].get(key, 0) + 1; continue
            out.append(m)
        return out
    D["subw"] = {}; D["net"] = {}
    for name, lst in D["nodes"].items():
        for path, ln, body in lst:
            ws = [m for m in SUBU.finditer(body) if not m.group(1)] + state_writes(path, body)
            R_, L_ = len(RAISE.findall(body)), len(LOWER.findall(body)); I_ = len(IDLEW.findall(body))
            D["net"][name] = D["net"].get(name, 0) + R_ - L_
            for m in ws:
                D["subw"].setdefault(name, []).append((f"{os.path.basename(path)}:{ln + body.count(chr(10), 0, m.start())}", len(ws) - I_ <= R_ - L_))
    D["uncounted"] = {n for n, v in D["subw"].items() if not all(c for _, c in v)}
    # rev 8 (prosecution round 3, S-1; coverage audit round 6, F4): the asynchronous internal submitters are the bodies
    # that allocate the in-flight token; the consumers take the token by pointer and fence it. A holder is a body that
    # leaves the process count raised (net raise > 0), or one that calls a consumer itself, fencing its own window.
    D["producers"] = {n for n, lst in D["nodes"].items() for _, _, b in lst if re.search(r"\bnew\s+(?:\w+::)*GpuLayerLoopInFlight\b|make_unique\s*<\s*(?:\w+::)*GpuLayerLoopInFlight\b|\bGpuInFlightToken\s*::\s*Make\s*\(", b)}
    D["consumers"] = {n for n, hs in FNHEADS.items() if any(re.search(r"GpuLayerLoopInFlight\s*\*\s*(?!\*)\w+|GpuInFlightToken\s*&&\s*\w+", h) for h in hs)} - D["producers"]
    # rev 8 (coverage audit round 6, R6-G1): a call through a name of function-pointer type (a parameter, a field, a
    # variable, a std::function, or a name declared with a function-pointer alias) that no initializer or store
    # resolves is an unresolved call. A template's type parameter is not such a name.
    alias, fpn, tparams = set(), set(), set()
    texts = [cstrip(open(os.path.join(root, f), encoding="utf-8", errors="replace").read()) for f in srcs + hdrs]
    for t in texts:
        alias |= set(re.findall(r"\busing\s+(\w+)\s*=\s*[^;]*\(\s*[\w:]*\*\s*\)\s*\(", t)) | set(re.findall(r"\btypedef\s+[^;]*\(\s*[\w:]*\*\s*(\w+)\s*\)\s*\(", t))
        fpn |= set(re.findall(r"\(\s*[\w:]*\*\s*(\w+)\s*\)\s*\(", t)) | set(re.findall(r"std::function\s*<[^;{]*?>\s*[&*]?\s*(\w+)", t))
        tparams |= set(re.findall(r"\b(?:typename|class)\s+(\w+)\s*[,>=]", t))
    for t in texts:
        for al in alias: fpn |= set(re.findall(r"\b" + al + r"\b\s*[&*]?\s*(\w+)", t))
    # a template's non-type parameter of function-pointer type (template <..., R (*Hash)(K)> class Name) is resolved
    # by every instantiation Name<...> in the tree: each function named in its argument list becomes an edge
    for t in texts:
        for tm in re.finditer(r"template\s*<([^;{]*?)>\s*(?:class|struct)\s+(\w+)", t):
            fps = re.findall(r"\(\s*\*\s*(\w+)\s*\)\s*\(", tm.group(1))
            if not fps: continue
            for t2 in texts:
                for im in re.finditer(r"\b" + tm.group(2) + r"\s*<([^;{>]*)>", t2):
                    for tok in re.findall(r"[A-Za-z_]\w*", im.group(1)):
                        if tok in FN:
                            for x in fps: D["stores"].setdefault(x, set()).add(tok)
    for x, tg in D["stores"].items():
        if x not in D["nodes"] or all(e[0] != "(stores)" for e in D["nodes"][x]): D["nodes"].setdefault(x, []).append(("(stores)", 0, " ".join(sorted(tg))))
        D["calls"][x] = D["calls"].get(x, set()) | set(tg)
    fpn -= tparams | alias | set(CKW)
    D["fpnames"] = fpn; D["unresolved"] = {}
    CASTCALL = re.compile(r"\b(?:reinterpret_cast|static_cast|bit_cast)\s*<[^;]*?\(\s*\*\s*\)\s*\([^;]*?>\s*\(\s*([^;()]+?)\s*\)\s*\("
                          r"|\(\s*\(\s*[\w:]+\s*\(\s*\*\s*\)\s*\([^()]*\)\s*\)\s*([\w:.>-]+)\s*\)\s*\(")
    for n, lst in D["nodes"].items():
        for _, _, b in lst:
            for x in fpn:
                if x not in D["nodes"] and re.search(r"(?<![\w:])" + re.escape(x) + r"\s*\(", b): D["unresolved"].setdefault(n, set()).add(x)
            for m in CASTCALL.finditer(b):          # rev 9 (coverage audit round 7, N2, X1): a call through a cast
                D["unresolved"].setdefault(n, set()).add("cast call through " + (m.group(1) or m.group(2)).strip())
    handout = {n for n, hs in FNHEADS.items() if any(re.search(r"GpuLayerLoopInFlight\s*\*\s*\*|GpuInFlightToken\s*\*\s*\w", h) for h in hs)}
    D["sources"] = D["producers"] | handout             # a body that allocates the token or hands one out to its caller
    # rev 9 (prosecution round 4, F8-4; coverage audit round 7, N2, X2): a body holds the token by finishing it only
    # when the consumer call runs on every path that has a token. Every construct governing the call must be a try, or
    # an if whose every conjunct is implied by the token's presence: the token itself, the token != nullptr, or the
    # submitting call's status == Ok (the submitters hand out no token on failure). An else, a loop, a case, a catch, a
    # conditional operator, or any other conjunct (the error path only; one budget value) is no hold.
    def governed_ok(hdr, tok):
        hdr = hdr.strip()
        if hdr == "" or re.fullmatch(r"try", hdr): return True
        m = re.fullmatch(r"(?:.*\b)?if\s*\((.*)\)", hdr, re.S)
        if not m or re.search(r"\b(?:else|while|for|case|catch|do|switch)\b", hdr[:m.start(1)]): return False
        for cj in re.split(r"&&", m.group(1)):
            cj = cj.strip().strip("()").strip()
            if not (cj == tok or re.fullmatch(re.escape(tok) + r"\s*!=\s*nullptr", cj) or re.fullmatch(r"\w+\s*==\s*(?:\w+::)*Ok", cj)): return False
        return True
    def absent_cut(seg, tok):                    # rev 10: drop each `if (!tok) ...` branch, where no token exists to keep
        pat = re.compile(r"\bif\s*\(\s*(?:!\s*" + re.escape(tok) + r"|" + re.escape(tok) + r"\s*==\s*nullptr|nullptr\s*==\s*" + re.escape(tok) + r")\s*\)\s*") if tok else None
        while pat:
            mm = pat.search(seg)
            if not mm: break
            j = mm.end()
            if seg[j:j + 1] == "{":
                d = 0
                for k in range(j, len(seg)):
                    d += seg[k] == "{"; d -= seg[k] == "}"
                    if d == 0: break
                seg = seg[:mm.start()] + seg[k + 1:]
            else:
                k = seg.find(";", j); seg = seg[:mm.start()] + (seg[k + 1:] if k >= 0 else "")
        return seg
    def finishes_every_path(n):
        for _, _, b in D["nodes"].get(n, []):
            for c in D["consumers"]:
                for m in re.finditer(r"(?<![\w:])(?:\w+::)*" + re.escape(c) + r"\s*\(", b):
                    args = call_args(b, m.end()); tok = args[0] if args else ""
                    opens, ok = [], True
                    for t in re.finditer(r"[{}]", b[:m.start()]):
                        if t.group() == "{": opens.append(t.start())
                        elif opens: opens.pop()
                    for o in opens[1:]:                      # each enclosing block below the body's own brace
                        st = max(b.rfind(";", 0, o), b.rfind("{", 0, o), b.rfind("}", 0, o))
                        if not governed_ok(b[st + 1:o], tok): ok = False; break
                    st = max(b.rfind(";", 0, m.start()), b.rfind("{", 0, m.start()), b.rfind("}", 0, m.start()))
                    stmt = b[st + 1:m.start()]
                    if ok and re.search(r"\bif\s*\(", stmt):
                        mm = re.match(r"\s*(if\s*\(.*\))\s*(?:return\b|[\w:]+\s*=|const\b|$)", stmt, re.S)
                        ok = bool(mm) and governed_ok(mm.group(1), tok)
                    elif ok and re.search(r"\b(?:else|while|for|case|do)\b|\?|\|\|", stmt): ok = False
                    # rev 10 (prosecution round 5, F9-4; coverage audit round 8, N5): the pre-G3 stand-in for the RAII
                    # token. Between the call that produces the token and the finish, no statement may leave the body:
                    # a return, goto, throw, break, continue, co_return or longjmp there is a path that keeps the token.
                    # (Rev 9 read only the blocks enclosing the finish, so an early exit before an unconditional finish
                    # passed.) An exception thrown by a call between them is outside this grammar; G3's token type,
                    # whose destructor drains, covers it for control flow. rev 11 (triage round 3, D1 and D3): this
                    # stand-in is advisory and checks only that a finish exists, not where the token goes (prosecution
                    # round 6, F10-1); storage is the boundary token check's (s4.4 The two boundary checks).
                    if ok:
                        prods = [pm for pm in re.finditer(r"(?<![\w:])(?:\w+::)*(" + "|".join(map(re.escape, sorted(D["sources"]))) + r")\s*\(", b[:m.start()])] if D["sources"] else []
                        if prods and re.search(r"\b(?:return|goto|throw|break|continue|co_return|longjmp)\b", absent_cut(b[prods[0].end():m.start()], tok)): ok = False
                    if ok: return True
        return False
    D["holders"] = ({n for n, v in D["net"].items() if v > 0} | {n for n, cs in D["calls"].items() if cs & D["consumers"] and finishes_every_path(n)}) - handout
    return D
def reach7(D, fn, extra=None):                 # the set of allocator kinds fn's call graph reaches (empty: no sink)
    calls = dict(D["calls"], **(extra or {}))
    seen, st = set(), [fn]
    while st:
        f = st.pop()
        if f in seen: continue
        seen.add(f); st += list(calls.get(f, ()))
    return set().union(set(), *[D["sitekind"].get(f, set()) for f in seen])
def reach_inflight(D, fn, extra=None):          # rev 8: the token sources fn reaches without passing a holder
    calls = dict(D["calls"], **(extra or {}))
    seen, st = set(), [fn]
    while st:
        f = st.pop()
        if f in seen: continue
        seen.add(f)
        if f in D["holders"] or f in D["sources"]: continue
        st += list(calls.get(f, ()))
    return sorted(seen & D["sources"])
def reach_unresolved(D, fn, extra=None):        # rev 8: the unresolved calls fn's graph reaches, as (node, name)
    calls = dict(D["calls"], **(extra or {}))
    seen, st = set(), [fn]
    while st:
        f = st.pop()
        if f in seen: continue
        seen.add(f); st += list(calls.get(f, ()))
    return sorted((f, x) for f in seen for x in D["unresolved"].get(f, ()) if (f, x) not in UNRESOLVED_OK)
# s4.4, rev 8: the one exempt unresolved call on main. `run` is the host-supplied parallel-for (parallel_for.h:46), the
# host's own function; the only library code it runs is the task LogitsSiteParallel passes it, which is already an edge.
UNRESOLVED_OK = {("LogitsSiteParallel", "run")}
def reach_uncounted(D, fn, extra=None, uncounted=None):   # the uncounted Submitted writers fn's call graph reaches
    calls = dict(D["calls"], **(extra or {})); unc = D["uncounted"] if uncounted is None else uncounted
    seen, st = set(), [fn]
    while st:
        f = st.pop()
        if f in seen: continue
        seen.add(f); st += list(calls.get(f, ()))
    return sorted(seen & unc)
def gpu_defined(D, n): return any(p.startswith("src/gpu/") for p, _, _ in D["defs"].get(n, []))
def s44_table7(text):                          # s4.4's tables and "does not submit" lists
    body = text.split("4. **Barriers", 1)[1].split("5. **Per-sequence table buffer:**", 1)[0]
    rows = {}
    for m in re.finditer(r"^\s*\| `(\w+)`( \(arc\))? \| ([^|]*) \| [^|]* \| ([^|]*) \| ([^|]*) \| [^|]* \| (guarded|does not submit|internal) \|\s*$", body, re.M):
        rows[m.group(1)] = (bool(m.group(2)), re.findall(r"`(\w+)`", m.group(4)), m.group(6), m.group(5).strip(), m.group(3).strip().strip("`"))
    for m in re.finditer(r"^\s*- \*\*`([\w.]+)`, does not submit \((\d+)\):\*\*((?:.|\n(?!\s*-\s|\s*\n))*)", body, re.M):
        names = re.findall(r"`(\w+)`", m.group(3))
        if len(names) != int(m.group(2)): FAIL6.append(("s4.4 list count", m.group(1), len(names), int(m.group(2))))
        for n in names: rows.setdefault(n, (False, [], "does not submit", "", m.group(1)))
    prec, intable = set(), False
    for line in (body.split("**Status precedence", 1)[1].split("\n") if "**Status precedence" in body else []):
        if line.strip().startswith("|"):
            intable = True; m = re.match(r"^\s*\| `(\w+)`", line)
            if m: prec.add(m.group(1))
        elif intable: break
    return rows, prec
def arc_verbs(text):                           # every GPU verb s4 names that no public header declares
    sec4 = text.split("## 4. Design — GPU slice", 1)[1].split("## 5.", 1)[0]
    declared = set().union(*[v[0] for v in D7["pop"].values()]) if D7 else set()   # D7: the derivation (set by main)
    return sorted(set(re.findall(r"\b(sslm_gpu_\w+)\b", sec4)) - declared)
CABI_HDRS = {"gpu_1p0.h", "gpu_1p0_g5_bridge.h", "gpu_1p0_bench_bridge.h"}   # the C ABI surface's headers
PLANNED = {"UploadPoolRegionsSync": {"singleton"},   # s4.8: the pool upload, on harness::GetDevice()
           "ZeroFillResidentUavBufferSync": {"per-context"}}   # rev 15, s4 item 3: the pool zero-fill, on the per-context device
ALLOC = {"singleton": {"singleton"}, "per-context": {"per-context"}, "both": {"singleton", "per-context"}}
def census7(rows, prec, D, arc, drop=(), extra_calls=None, extra_decl=None, extra_kind=None, extra_uncounted=(), notsink=None, notstate=None):
    """Every error names its reason; the battery (blocking order step 3) grades by reason, not by exit code."""
    errs = []
    pop = {h: (set(v[0]) - set(drop), v[1]) for h, v in D["pop"].items()}
    if extra_decl: pop[extra_decl[0]] = (pop.get(extra_decl[0], (set(), set()))[0] | {extra_decl[1]}, pop.get(extra_decl[0], (set(), set()))[1] | {extra_decl[1]})
    Dx = dict(D, sitekind=dict(D["sitekind"], **(extra_kind or {})))
    def rk(n): return reach7(Dx, n, extra_calls)
    seen = set()
    for h, (strict, loose) in sorted(pop.items()):
        for n in sorted(loose - strict):
            if n in D["defs"]: errs.append(("declaration the parser missed", n, h))
        for n in sorted(strict):
            seen.add(n); r = rk(n); row = rows.get(n)
            u = reach_uncounted(Dx, n, extra_calls, D["uncounted"] | set(extra_uncounted))
            if u: errs.append(("uncounted Submitted state reachable from a public declaration", n, h, u))
            if h in CABI_HDRS:                           # rev 8: the C ABI surface only (the lint's property, P2)
                inf = reach_inflight(Dx, n, extra_calls)
                if inf: errs.append(("uncounted in-flight: reaches an asynchronous internal submitter without a held count", n, h, inf))
                ur = reach_unresolved(Dx, n, extra_calls)
                if ur: errs.append(("unresolved call reachable from a C ABI declaration", n, h, ur))
            if row is None:
                if r: errs.append(("reaches a sink and has no row", n, h))
                elif gpu_defined(D, n) or (extra_decl and n == extra_decl[1]): errs.append(("declaration defined under src/gpu with no row", n, h))
                continue
            d = row[2]
            if d == "internal" and h not in INTERNAL: errs.append(("'internal' row outside the internal layer", n, h))
            if r and d == "does not submit": errs.append(("false 'does not submit' row: reaches a sink", n, h))
            if not r and d == "guarded": errs.append(("stale 'guarded' row: reaches no sink", n, h))
            if not r and d == "internal": errs.append(("stale 'internal' row: reaches no sink", n, h))
            if r and row[3] and ALLOC.get(row[3]) != r: errs.append(("allocator column disagrees with the derivation", n, row[3], sorted(r)))
            if d == "guarded" and n not in prec: errs.append(("guarded entry point with no status-precedence row", n))
    for s in D["sites"]:
        if "unclassified" in s[3]: errs.append(("sink whose allocator is unclassified", s[1], s[2]))
    for key, c in (notsink_rows(PLAN) if notsink is None else notsink).items():          # rev 10: an over-match row whose site is gone
        if D.get("notsink_seen", {}).get(key, 0) < c: errs.append(("stale 'not a sink' row", key[0], key[1]))
    for key, c in (notstate_rows(PLAN) if notstate is None else notstate).items():
        if D.get("notstate_seen", {}).get(key, 0) < c: errs.append(("stale 'not the handle's state' row", key[0], key[1]))
    hf = D.get("handle_files", [])
    if len(hf) != 1 or not hf[0].endswith((".cpp", ".cc")):    # the state token's scope is the handle's one file
        errs.append(("handle's definition not confined to one source file", hf))
    for k, v in (extra_kind or {}).items():
        if "unclassified" in v: errs.append(("sink whose allocator is unclassified", "injected", k))
    for e in arc:
        if e not in rows: errs.append(("arc verb with no row", e)); continue
        cal = rows[e][1]
        bad = [c for c in cal if c not in D["defs"] and c not in PLANNED]
        if bad: errs.append(("planned callee not in the tree", e, bad)); continue
        r = set().union(set(), *[PLANNED.get(c) or rk(c) for c in cal]); d = rows[e][2]
        if r and d != "guarded": errs.append(("false 'does not submit' row: reaches a sink", e, "arc"))
        if not r and d == "guarded": errs.append(("stale 'guarded' row: reaches no sink", e, "arc"))
        if r and rows[e][3] and ALLOC.get(rows[e][3]) != r: errs.append(("allocator column disagrees with the derivation", e, rows[e][3], sorted(r)))
        if d == "guarded" and e not in prec: errs.append(("guarded entry point with no status-precedence row", e))
    for n, row in rows.items():
        if n not in seen and n not in arc: errs.append(("row names no declaration", n))
    return errs

# ---- G3: the token lint ----------------------------------------------------------------------------------------
LINT_TOK = re.compile(r"\b(ExecuteCommandLists|CreateCommandAllocator|CreateCommandQueue|D3D12CreateDevice)\b")
LINT_ALLOWED = {"src/gpu/d3d12_harness.h"}


def token_lint(root):
    out = []
    files = tree_files(root, "src", {"cpp", "cc", "h", "hpp", "inl", "inc", "def"}) + tree_files(root, "include", {"h", "hpp", "inc", "def"})
    for f in files:
        if f.replace(os.sep, "/") in LINT_ALLOWED: continue
        txt = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        stripped = cstrip(txt)
        for m in LINT_TOK.finditer(stripped):
            out.append(("sink call outside the harness", f.replace(os.sep, "/"), f"{os.path.basename(f)}:{stripped.count(chr(10), 0, m.start()) + 1}", m.group(1)))
    return out


# ---- G3: the built-tree reasons on the token and the reset sites ----------------------------------------------
TOKEN_HDR = "gpu_inflight_token.h"
RECORD_SRC = "superslm_gpu.cpp"


def g3_built_tree(root, D):
    errs = []
    # reset site without PoolSubmitAssert: from rev 8, any reset outside BeginRecording()
    for where, fn in D.get("raw_resets", []):
        errs.append(("reset site without `PoolSubmitAssert`", fn, where, "a raw allocator Reset() outside the ledger"))
    for where_meth in D.get("ledger_inside", []):
        meth, where, fn = where_meth
        if fn == "ResetAllocatorAndList" and meth == "Reset":
            callers = sorted(n for n, cs in D["calls"].items() if "ResetAllocatorAndList" in cs)
            bad = [c for c in callers if c not in ("BeginRecording", "ResetAllocatorAndList")]
            if bad: errs.append(("reset site without `PoolSubmitAssert`", fn, where, "ResetAllocatorAndList called from " + ", ".join(bad)))
    files = tree_files(root, "src", {"cpp", "cc", "h", "hpp", "inl", "inc", "def"}) + tree_files(root, "include", {"h", "hpp", "inc", "def"})
    for f in files:
        base = os.path.basename(f)
        txt = open(os.path.join(root, f), encoding="utf-8", errors="replace").read(); s = cstrip(txt)
        ln = lambda i: f"{base}:{s.count(chr(10), 0, i) + 1}"
        # T1: the raw record named outside the token's header and the record's source file. A forward declaration and
        # the token alias's template argument name the type without holding a record, and are not counted.
        if base not in (TOKEN_HDR, RECORD_SRC):
            for m in re.finditer(r"\bGpuLayerLoopInFlight\b", s):
                pre = s[max(0, m.start() - 40):m.start()]; post = s[m.end():m.end() + 3]
                if re.search(r"\bstruct\s*$", pre) and post.lstrip().startswith(";"): continue
                if re.search(r"BasicGpuInFlightToken\s*<\s*$", pre) and post.lstrip().startswith(">"): continue
                errs.append(("raw in-flight record named outside the token", "GpuLayerLoopInFlight", ln(m.start()), "T1"))
        # T2: a GpuInFlightToken, where the type is spelled, in static storage, on the heap, in a template argument or in
        # any member but the handle's in_flight
        if base != TOKEN_HDR:
            spans = FNSPANS.get(f, [])
            def in_fn(i): return any(a < i < b for a, b in spans)
            for m in re.finditer(r"\bGpuInFlightToken\b", s):
                i = m.start(); post = s[m.end():m.end() + 200]; pre = s[max(0, i - 60):i]
                if re.search(r"\bnew\s+(?:\w+::)*$", pre) or re.search(r"make_(?:unique|shared)\s*<\s*(?:\w+::)*$", pre):
                    errs.append(("in-flight token stored outside a local or the handle", "GpuInFlightToken", ln(i), "T2: on the heap")); continue
                if re.search(r"<[^<>;{}()]*$", pre) and re.match(r"[^<>;{}()]*>", post) and not re.search(r"BasicGpuInFlightToken\s*<\s*$", pre):
                    errs.append(("in-flight token stored outside a local or the handle", "GpuInFlightToken", ln(i), "T2: in a template argument")); continue
                dm = re.match(r"\s+(\w+)\s*(?:\{[^;]*\})?\s*[;=]", post)
                if not dm: continue                                      # a parameter, a cast, a call, Make, an alias
                if re.search(r"\bstatic\s+(?:\w+::)*$", pre) or re.search(r"\bthread_local\s+(?:\w+::)*$", pre):
                    errs.append(("in-flight token stored outside a local or the handle", "GpuInFlightToken", ln(i), "T2: in static storage")); continue
                if not in_fn(i):
                    if dm.group(1) == "in_flight": continue              # the handle's in_flight
                    errs.append(("in-flight token stored outside a local or the handle", "GpuInFlightToken", ln(i), f"T2: member or namespace variable {dm.group(1)}"))
    # T3: a store into the handle's in_flight in a body with no net count raise. Storing an empty token (a reset of the
    # slot) stores no work, and is not counted.
    for name, lst in D["nodes"].items():
        for path, l0, body in lst:
            for m in re.finditer(r"\bin_flight\s*=(?!=)\s*([^;]*);", body):
                if re.fullmatch(r"\s*(?:(?:\w+::)*GpuInFlightToken\s*(?:\(\s*\)|\{\s*\})|\{\s*\})\s*", m.group(1)): continue
                if D["net"].get(name, 0) <= 0:
                    errs.append(("token stored on the handle without a count raise", name, f"{os.path.basename(path)}:{l0 + body.count(chr(10), 0, m.start())}", "T3"))
    return errs


def emit(errs, as_json):
    for e in errs:
        reason, entry = e[0], e[1] if len(e) > 1 else ""
        rest = [x for x in e[2:]]
        where = rest[0] if rest and isinstance(rest[0], str) and re.search(r"\.(h|hpp|cpp|cc|inl)\b", rest[0]) else ""
        if where: rest = rest[1:]
        detail = "; ".join(x if isinstance(x, str) else json.dumps(x, default=list) for x in rest)
        if not isinstance(entry, str): entry = json.dumps(entry, default=list)
        if as_json: print(json.dumps({"reason": reason, "entry": entry, "where": where, "detail": detail}))
        else: print(f"CENSUS: {reason} | {entry} | {where} | {detail}")


def main():
    global PLAN, D7
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=os.path.normpath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--plan", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lint-only", action="store_true", help="run the token lint alone (no plan needed)")
    a = ap.parse_args()
    root = os.path.abspath(a.src)
    lint = token_lint(root)
    if a.lint_only or not a.plan:          # with no --plan only the lint can run (the census's rows are the plan's)
        emit(lint, a.json)
        print(f"token lint: {len(lint)} sink call(s) outside d3d12_harness.h under src/ and include/", file=sys.stderr)
        return 1 if lint else 0
    PLAN = open(a.plan, encoding="utf-8").read()
    globals()["PLAN"] = PLAN
    D7 = derive7(root); globals()["D7"] = D7
    rows, prec = s44_table7(PLAN); arc = arc_verbs(PLAN)
    errs = census7(rows, prec, D7, arc)
    errs += g3_built_tree(root, D7)
    errs += lint
    errs += [("plan table anomaly",) + tuple(x if isinstance(x, str) else str(x) for x in f) for f in FAIL6]
    emit(errs, a.json)
    guarded = sorted(e for e, r in rows.items() if r[2] == "guarded")
    print(f"census: {len(rows)} rows ({len(arc)} arc verbs), guarded {len(guarded)}; {len(D7['sites'])} sink sites "
          f"({len(D7['ledger_inside'])} inside the ledger); sources {sorted(D7['sources'])}; consumers {sorted(D7['consumers'])}; "
          f"{len(errs)} record(s), {len(lint)} from the token lint. Not implemented: \"state change before `EntrySubmitGuard`\", "
          f"\"pure helper writes outside the caller's locals\".", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
