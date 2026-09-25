"""Commissioning constructions for tools/ci/check_gpu_catch_totality.py (TE-437).

StandardsDocument.md Sec5.4: an instrument's verdicts are quarantined until a must-accept and a
must-reject, authored by a seat independent of its builder and blind to the builder's own
validation, fire correctly. The gate's builder (TE-435) cannot grade it; the test author authors
both sides here.

WHAT THE GATE CLAIMS (its own docstring). Every outermost `try` in a C++ source or header under
src/gpu ends its handlers with `catch (...)`; a `try` inside another try block's compound
statement is exempt; a `try` inside a handler is checked; comments and literals are not code; it
refuses (exit 2) rather than pass on a raw string literal, an unmatched bracket, or no `try`.

HOW EXPECTATIONS ARE DERIVED, independently of the gate's lexer and matcher.
  - Comments and literals are blanked by the site census's lexer
    (tests/ci/check_gpu_status_site_census.py strip_code, commissioned as TE425-SITE-CENSUS).
  - Each `catch (...)` clause's `try` is found by walking BACKWARD from the `catch` keyword over
    balanced handler bodies and headers until the `try` keyword; the gate matches forward.
  - A `try` is nested when one of its enclosing `{` is itself preceded by the `try` keyword; the gate
    compares block spans.

HOW A CONSTRUCTION RUNS. Each is a directory holding tools/ci/check_gpu_catch_totality.py (a copy
verified byte-identical to the instrument) and src/gpu, and the gate is run with no arguments --
exactly the CI step's invocation -- so its default root is the construction's own src/gpu.

  --mode accept   must-accept: exit 0 only when every accept leg reads OK (exit 0).
  --mode reject   must-reject: exit 1 only when every reject leg returns exactly its expected
                  verdict (exit 1 naming exactly the expected sites, or exit 2 for a refusal);
                  exit 0 on any other outcome, so a broken construction never reads as a rejection.
  --mode probe    shapes outside the registered constructions, printed for the record.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_GATE = REPO / "tools" / "ci" / "check_gpu_catch_totality.py"
_spec = importlib.util.spec_from_file_location("te437_census", HERE / "check_gpu_status_site_census.py")
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)

SUFFIXES = (".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inl")

# The ten ladders that did not end in catch (...) before TE-435 (c3b5412): the functions the R11 site
# list carried catch rows for and no `...` row at 042bd66/c3b5412, the test author's own census.
PRE_FIX_LADDERS = Counter({
    "CreateDeviceLogitsBuffers": 1, "RunDeviceLogits": 1, "SubmitAdmittedChunkForG5Bridge": 1,
    "sslm_gpu_adapter_mapImpl": 1, "sslm_gpu_model_mapImpl": 2, "sslm_gpu_seq_createImpl": 1,
    "sslm_gpu_seq_restoreImpl": 2, "RunLayerLoopGpuFinish": 1,
})


# ---- independent structure: backward walk over census-stripped code ----------------------------

def _skip_ws_back(code: str, i: int) -> int:
    while i >= 0 and code[i].isspace():
        i -= 1
    return i


def _match_back(code: str, close: int) -> int:
    """Offset of the bracket opening the one closing at `close` (a '}' or ')'), walking backward."""
    closer = code[close]
    opener = {"}": "{", ")": "("}[closer]
    depth = 0
    for k in range(close, -1, -1):
        if code[k] == closer:
            depth += 1
        elif code[k] == opener:
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unmatched {closer!r} at {close}")


def _match_fwd(code: str, open_: int) -> int:
    opener = code[open_]
    closer = {"{": "}", "(": ")"}[opener]
    depth = 0
    for k in range(open_, len(code)):
        if code[k] == opener:
            depth += 1
        elif code[k] == closer:
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unmatched {opener!r} at {open_}")


def _word_before(code: str, i: int) -> tuple[str, int]:
    """The identifier ending at or before offset i (after skipping whitespace), and its start."""
    j = _skip_ws_back(code, i)
    end = j + 1
    while j >= 0 and (code[j].isalnum() or code[j] == "_"):
        j -= 1
    return code[j + 1:end], j + 1


def try_of_catch(code: str, catch_pos: int) -> int:
    """Offset of the `try` keyword owning the handler whose `catch` keyword starts at catch_pos."""
    i = catch_pos - 1
    while True:
        j = _skip_ws_back(code, i)
        if code[j] != "}":
            raise ValueError(f"no block before catch at {catch_pos}")
        open_ = _match_back(code, j)
        k = _skip_ws_back(code, open_ - 1)
        if code[k] == ")":
            p = _match_back(code, k)
            word, start = _word_before(code, p - 1)
            if word != "catch":
                raise ValueError(f"a handler header not preceded by catch at {p}")
            i = start - 1
            continue
        word, start = _word_before(code, open_ - 1)
        if word == "try":
            return start
        raise ValueError(f"block before catch at {catch_pos} is not a try block")


def is_nested(code: str, try_pos: int) -> bool:
    """True when an enclosing '{' of try_pos is a try block's own opening brace."""
    depth = 0
    for k in range(try_pos - 1, -1, -1):
        c = code[k]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                if _word_before(code, k - 1)[0] == "try":
                    return True
            else:
                depth -= 1
    return False


def catch_all_sites(text: str) -> list[dict]:
    """Every `catch (...)` clause in `text`: its span, its try's line, nesting, and handler count."""
    code = census.strip_code(text)
    sites = []
    for m in re.finditer(r"\bcatch\s*\(\s*\.\.\.\s*\)", code):
        h = m.end()
        while code[h].isspace():
            h += 1
        assert code[h] == "{", "catch (...) without a compound statement"
        end = _match_fwd(code, h)
        t = try_of_catch(code, m.start())
        # Handler count of this ladder: walk forward from the try's block.
        b = t + 3
        while code[b].isspace():
            b += 1
        pos = _match_fwd(code, b) + 1
        n = 0
        while True:
            cm = re.compile(r"\s*catch\s*\(").match(code, pos)
            if not cm:
                break
            pc = _match_fwd(code, cm.end() - 1)
            hb = pc + 1
            while code[hb].isspace():
                hb += 1
            pos = _match_fwd(code, hb) + 1
            n += 1
        sites.append({
            "catch_start": m.start(), "header": (m.start(), m.end()), "handler_end": end,
            "try_line": code.count("\n", 0, t) + 1, "nested": is_nested(code, t), "handlers": n,
        })
    return sites


# ---- constructions ---------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_tree(work: Path, name: str, gate: Path, src_gpu: Path | None) -> Path:
    root = work / name
    (root / "tools" / "ci").mkdir(parents=True)
    copy = root / "tools" / "ci" / "check_gpu_catch_totality.py"
    shutil.copy2(gate, copy)
    if sha256(copy) != sha256(gate):
        raise RuntimeError("the gate copy is not byte-identical to the instrument")
    if src_gpu is not None:
        shutil.copytree(src_gpu, root / "src" / "gpu")
    else:
        (root / "src" / "gpu").mkdir(parents=True)
    return root


def run_gate(root: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(root / "tools" / "ci" / "check_gpu_catch_totality.py")],
                       capture_output=True, text=True, cwd=str(root))
    return p.returncode, p.stdout + p.stderr


def named_sites(out: str) -> Counter:
    return Counter(re.findall(r"^(src/gpu/[^:\s]+:\d+): outermost try ends in", out, re.M))


def gpu_files(src_gpu: Path) -> list[Path]:
    return sorted(p for p in src_gpu.rglob("*") if p.is_file() and p.suffix in SUFFIXES)


def site_legs(tip_gpu: Path) -> list[dict]:
    """One leg per catch (...) clause at the tip: the clause removed as a real edit would remove it.
    A ladder with other handlers loses the clause; a ladder whose only handler it is has it
    narrowed to `catch (const std::exception&)` (deleting it would not compile)."""
    legs = []
    for f in gpu_files(tip_gpu):
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(tip_gpu).as_posix()
        for s in catch_all_sites(text):
            if s["handlers"] > 1:
                mutated = text[:s["catch_start"]] + text[s["handler_end"] + 1:]
                how = "deleted"
            else:
                a, b = s["header"]
                mutated = text[:a] + "catch (const std::exception& te437_narrowed)" + text[b:]
                how = "narrowed"
            line = text.count("\n", 0, s["catch_start"]) + 1
            legs.append({
                "name": f"{rel}:{line} catch (...) {how}",
                "file": rel, "text": mutated,
                "expect": Counter() if s["nested"] else Counter({f"src/gpu/{rel}:{s['try_line']}": 1}),
                "nested": s["nested"],
            })
    return legs


def run_edit(work: Path, gate: Path, tip_gpu: Path, name: str, edits: dict[str, str]) -> tuple[int, str]:
    root = make_tree(work, name, gate, tip_gpu)
    for rel, text in edits.items():
        p = root / "src" / "gpu" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
    return run_gate(root)


_APPENDED_TYPED_TRY = (
    "\nint Te437ProbeTypedOnly() {\n\ttry {\n\t\treturn 1;\n\t} catch (const std::bad_alloc&) {\n"
    "\t\treturn 0;\n\t} catch (const std::exception&) {\n\t\treturn 2;\n\t}\n}\n")
_APPENDED_FUNCTION_TRY = (
    "\nint Te437ProbeFunctionTry() try {\n\treturn 1;\n} catch (const std::exception&) {\n\treturn 0;\n}\n")
_NEW_FILE = (
    "// A new translation unit under src/gpu.\n#include <new>\nnamespace superslm_gpu {\n"
    "int Te437NewSite(int* p) {\n\ttry {\n\t\t*p = 1;\n\t\treturn 0;\n\t} catch (const std::bad_alloc&) {\n"
    "\t\treturn 1;\n\t}\n}\n}  // namespace superslm_gpu\n")
_LEXER_NOISE = (
    "\n// try { x(); } catch (int) { }  -- a comment, not code\n/* try { } catch (long) { } */\n"
    "static const char* kTe437Str = \"try { y(); } catch (short) { }\";\n"
    "static const char kTe437Brace = '{';\nstatic const char kTe437Quote = '\\'';\n"
    "static const long long kTe437Sep = 1'000'000;\n"
    "int Te437ProbeNestedTyped() {\n\ttry {\n\t\ttry {\n\t\t\treturn 1;\n\t\t} catch (const std::bad_alloc&) {\n"
    "\t\t\treturn 2;\n\t\t}\n\t} catch (...) {\n\t\treturn 0;\n\t}\n}\n")
_RAW_STRING = "\nstatic const char* kTe437Raw = R\"(try { } catch (int) { })\";\n"


def line_of_append(base: str, appended: str, needle: str) -> int:
    return (base + appended).count("\n", 0, len(base) + appended.index(needle)) + 1


def reject_legs(tip_gpu: Path, pre_gpu: Path | None) -> list[dict]:
    legs = [l for l in site_legs(tip_gpu) if not l["nested"]]
    base = (tip_gpu / "gpu_1p0.cpp").read_text(encoding="utf-8")
    legs.append({"name": "gpu_1p0.cpp: appended function whose try ends in a typed catch",
                 "edits": {"gpu_1p0.cpp": base + _APPENDED_TYPED_TRY},
                 "expect": Counter({f"src/gpu/gpu_1p0.cpp:{line_of_append(base, _APPENDED_TYPED_TRY, 'try {')}": 1})})
    legs.append({"name": "gpu_1p0.cpp: appended function-try-block ending in a typed catch",
                 "edits": {"gpu_1p0.cpp": base + _APPENDED_FUNCTION_TRY},
                 "expect": Counter({f"src/gpu/gpu_1p0.cpp:{line_of_append(base, _APPENDED_FUNCTION_TRY, 'try {')}": 1})})
    legs.append({"name": "new file in a new subdirectory, try ends in a typed catch",
                 "edits": {"te437_probe/new_site.cpp": _NEW_FILE},
                 "expect": Counter({"src/gpu/te437_probe/new_site.cpp:5": 1})})
    legs.append({"name": "gpu_1p0.cpp: a raw string literal (the gate refuses to scan, exit 2)",
                 "edits": {"gpu_1p0.cpp": base + _RAW_STRING}, "expect_exit": 2})
    legs.append({"name": "src/gpu with no try at all (the gate refuses, exit 2)",
                 "empty": True, "expect_exit": 2})
    if pre_gpu is not None:
        legs.append({"name": "c3b5412, before TE-435: the ten ladders without catch (...)",
                     "pre": True, "expect_functions": PRE_FIX_LADDERS})
    return legs


def accept_legs(tip_gpu: Path) -> list[dict]:
    legs = [{"name": "91779ad tip, unmodified", "edits": {}}]
    legs += [l for l in site_legs(tip_gpu) if l["nested"]]
    base = (tip_gpu / "gpu_1p0.cpp").read_text(encoding="utf-8")
    legs.append({"name": "gpu_1p0.cpp: comments, literals, digit separators and a nested typed-only try",
                 "edits": {"gpu_1p0.cpp": base + _LEXER_NOISE}})
    return legs


def functions_named(pre_gpu: Path, out: str) -> Counter:
    got: Counter = Counter()
    for site in named_sites(out):
        rel, line = site[len("src/gpu/"):].rsplit(":", 1)
        text = (pre_gpu / rel).read_text(encoding="utf-8")
        clean = census.strip_code(text)
        offset = sum(len(l) + 1 for l in text.split("\n")[: int(line) - 1])
        got[census.enclosing(census.function_spans(clean), offset + 1).split("::")[-1]] += 1
    return got


def execute(leg: dict, work: Path, gate: Path, tip_gpu: Path, pre_gpu: Path | None, idx: int) -> tuple[int, str]:
    name = f"leg{idx:02d}"
    if leg.get("empty"):
        return run_gate(make_tree(work, name, gate, None))
    if leg.get("pre"):
        return run_gate(make_tree(work, name, gate, pre_gpu))
    edits = leg["edits"] if "edits" in leg else {leg["file"]: leg["text"]}
    return run_edit(work, gate, tip_gpu, name, edits)


def judge_reject(leg: dict, code: int, out: str, pre_gpu: Path | None) -> bool:
    if "expect_exit" in leg:
        return code == leg["expect_exit"]
    if code != 1:
        return False
    if "expect_functions" in leg:
        return functions_named(pre_gpu, out) == leg["expect_functions"]
    return named_sites(out) == leg["expect"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("accept", "reject", "probe"), required=True)
    ap.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    ap.add_argument("--tip", type=Path, default=REPO, help="directory holding src/gpu at the tip")
    ap.add_argument("--pre", type=Path, default=None, help="directory holding src/gpu before TE-435")
    ap.add_argument("--work", type=Path, default=None, help="parent directory for the constructions")
    args = ap.parse_args(argv)
    tip_gpu = args.tip / "src" / "gpu"
    pre_gpu = args.pre / "src" / "gpu" if args.pre else None
    if args.work is not None:
        args.work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="te437-", dir=args.work))
    try:
        if args.mode == "probe":
            return probe(work, args.gate, tip_gpu)
        legs = accept_legs(tip_gpu) if args.mode == "accept" else reject_legs(tip_gpu, pre_gpu)
        ok_all = True
        for i, leg in enumerate(legs):
            code, out = execute(leg, work, args.gate, tip_gpu, pre_gpu, i)
            if args.mode == "accept":
                ok = code == 0 and "OK:" in out
            else:
                ok = judge_reject(leg, code, out, pre_gpu)
            ok_all &= ok
            last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
            print(f"{'AS EXPECTED' if ok else 'UNEXPECTED '}  exit {code}  {leg['name']}  | {last}")
        print(f"{len(legs)} {args.mode} legs, {'every one as expected' if ok_all else 'NOT every one as expected'}")
        if args.mode == "accept":
            return 0 if ok_all else 1
        return 1 if ok_all else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def probe(work: Path, gate: Path, tip_gpu: Path) -> int:
    """Shapes a real edit can produce that the registered constructions do not include, printed
    for the record: what the gate reads on each, and what the property it claims requires."""
    base = (tip_gpu / "superslm_gpu.cpp").read_text(encoding="utf-8")
    finish = base.index("RunLayerLoopGpuFinish(GpuLayerLoopInFlight* inflight")
    catch_all = base.index("} catch (...) {", finish)
    guarded = (base[:catch_all] + "}\n#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)\n\tcatch (...) {"
               + base[catch_all + len("} catch (...) {"):])
    end_handler = guarded.index("\n\t}\n}", guarded.index("#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)"))
    guarded = guarded[:end_handler] + "\n\t}\n#endif\n}" + guarded[end_handler + len("\n\t}\n}"):]
    # The same clause guarded from inside the preceding handler: the directive then sits in a handler
    # body, not between handlers, and the braces balance in both configurations.
    sib = base.index("} catch (const std::exception&) {", finish)
    sib_end = base.index("\n\t}", base.index("return ClassifyForwardFault", sib))
    ca_end = base.index("\n\t}", base.index("return ClassifyForwardFault", catch_all))
    inside = (base[:sib_end] + "\n#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)"
              + base[sib_end:ca_end] + "\n#endif" + base[ca_end:])
    cases = [
        ("RunLayerLoopGpuFinish's catch (...) wrapped whole in #if SUPERSLM_GPU_ALLOC_FAULT_INJECTION; a "
         "release build has no catch-all",
         {"superslm_gpu.cpp": guarded}, "exit 1 (the release build's ladder is not total)"),
        ("RunLayerLoopGpuFinish's catch (...) behind an #if opened inside the preceding handler's body; a "
         "release build has no catch-all",
         {"superslm_gpu.cpp": inside}, "exit 1 (the release build's ladder is not total)"),
    ]
    for i, (_, edits, _) in enumerate(cases):
        tail = edits["superslm_gpu.cpp"][finish:]
        seg = tail[tail.index("} catch (const std::exception&) {"):]
        print(f"--- probe{i:02d} construction (RunLayerLoopGpuFinish's ladder):")
        print(seg[:seg.index("\n}\n") + 3])
    for i, (name, edits, required) in enumerate(cases):
        code, out = run_edit(work, gate, tip_gpu, f"probe{i:02d}", edits)
        last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        print(f"PROBE  exit {code}  {name}\n       required: {required}\n       gate: {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
