"""CI source check: every site in the GPU sources that turns an exception into a status, and every
code site that names a GPU forward status, is listed -- with its disposition -- in the committed site
list, and the list names no site the source no longer has.

WHY THIS EXISTS (TE-425; plan `te421-slm172-host-oom.md` Sec3.5 R11, from TE-422 S-2 and TE-423 F-4).
The SuperSLM 1.8.0 allocation-status fix is a classifier applied at every catch clause that produces a
status, and a status split (`GpuOperationFailed`) that every mapping and exclusion list must learn. Its
first draft named the sites from memory and missed three catch sites (`InvokeGpuApiBoundary`,
`SubmitAdmittedChunkForG5Bridge`'s swallowing `runtime_error` clause, `RunDeviceLogits`), both
guard-refusal exclusion lists, and the null-token return in `RunLayerLoopGpuFinish`. A census kept by
hand drifts the same way; this check keeps it complete by construction: a catch clause or a status
site added to the source without a row fails CI.

WHAT IS SCANNED.
  - catch sites: every `catch (...)` clause in the C++ sources under `src/gpu`.
  - status sites: every code statement naming `GpuAllocationFailed`, `GpuDeviceRemoved` or
    `GpuOperationFailed` in the C++ sources under `src/gpu`, plus `src/forward/checked_chain_funnel.cpp`
    and `src/sslm_abi.cpp` (the two CPU files plan Sec3.2 E-3 routes the new enumerator through).
Comments, string and character literals, and preprocessor lines are blanked before scanning, so prose
that mentions a status is not a site. A site is keyed by (kind, file, enclosing function, token): the
catch clause's caught type, or the status statement's normalized text. The enclosing function is found
by brace matching, not by line number, so an edit that shifts lines does not move a site; an edit that
changes a site's statement does, which is the point -- a changed site is re-dispositioned.

THE LIST. `tests/te425-gpu-alloc-status-red-suite/gpu_status_sites.txt`, tab-separated:
    kind  file  function  count  block  disposition  token
Every row carries a non-empty block (the plan's T-number, or E-3 for a status site) and disposition.

Exit status: 0 the list and the source agree; 1 they do not (each difference named); 2 usage.
`--emit` prints the source's census in list form (block and disposition empty) for authoring rows.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIST = REPO / "tests" / "te425-gpu-alloc-status-red-suite" / "gpu_status_sites.txt"
STATUS_NAMES = ("GpuAllocationFailed", "GpuDeviceRemoved", "GpuOperationFailed")
GPU_DIR = "src/gpu"
STATUS_EXTRA = ("src/forward/checked_chain_funnel.cpp", "src/sslm_abi.cpp")
CPP_SUFFIXES = {".cpp", ".h", ".hpp", ".cc", ".cxx"}
KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "sizeof", "decltype", "alignas",
            "static_assert", "alignof", "noexcept", "throw", "operator"}


def strip_code(text: str) -> str:
    """Blank comments, string/char literals (raw strings included) and preprocessor lines; keep every
    offset and newline so positions map back to the original."""
    out = list(text)
    n = len(text)
    i = 0

    def blank(a: int, b: int) -> None:
        for j in range(a, b):
            if out[j] != "\n":
                out[j] = " "

    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif c == "R" and i + 1 < n and text[i + 1] == '"' and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            k = text.find("(", i + 2)
            delim = text[i + 2:k]
            j = text.find(")" + delim + '"', k)
            j = n if j < 0 else j + len(delim) + 2
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'":
            # A digit separator (1'000) is not a character literal.
            if i > 0 and (text[i - 1].isalnum()):
                i += 1
                continue
            j = i + 1
            while j < n and text[j] != "'":
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        else:
            i += 1
    stripped = "".join(out)
    # Preprocessor lines, with their backslash continuations.
    lines = stripped.split("\n")
    in_directive = False
    for idx, line in enumerate(lines):
        if in_directive or line.lstrip().startswith("#"):
            in_directive = line.rstrip().endswith("\\")
            lines[idx] = " " * len(line)
        else:
            in_directive = False
    return "\n".join(lines)


def function_spans(clean: str) -> list[tuple[int, int, str]]:
    """(open-brace offset, close-brace offset, qualified name) of every function body, by brace
    matching. Everything nested in a function body -- lambdas, local structs, blocks -- belongs to it."""
    spans: list[tuple[int, int, str]] = []
    stack: list[tuple[str, str, int]] = []  # (kind, name, start)
    hstart = 0
    paren = 0
    for i, c in enumerate(clean):
        if c == "(":
            paren += 1
        elif c == ")":
            paren = max(0, paren - 1)
        elif c == ";" and paren == 0:
            hstart = i + 1
        elif c == "{":
            if any(k == "func" for k, _, _ in stack):
                stack.append(("block", "", i))
            else:
                header = " ".join(clean[hstart:i].split())
                stack.append(classify(header, stack) + (i,))
            hstart = i + 1
        elif c == "}":
            if stack:
                kind, name, start = stack.pop()
                if kind == "func":
                    spans.append((start, i, name))
            hstart = i + 1
    return spans


def classify(header: str, stack: list[tuple[str, str, int]]) -> tuple[str, str]:
    # `extern "C" {` is a linkage block; `extern "C" void f() {` is a function (the string is blanked).
    if re.match(r"^namespace\b", header) or (re.match(r"^extern\b", header) and "(" not in header):
        return ("ns", "")
    if "(" in header:
        before = header[: header.index("(")]
        if before.rstrip().endswith("=") or before.rstrip().endswith("]"):
            return ("block", "")
        names = re.findall(r"[A-Za-z_~][\w]*(?:::[A-Za-z_~][\w]*)*", before)
        if names and names[-1] not in KEYWORDS:
            owners = [n for k, n, _ in stack if k == "type" and n]
            return ("func", "::".join(owners + [names[-1]]))
        return ("block", "")
    m = re.search(r"\b(?:struct|class|union|enum(?:\s+class)?)\s+([A-Za-z_]\w*)", header)
    if m:
        return ("type", m.group(1))
    return ("block", "")


def enclosing(spans: list[tuple[int, int, str]], pos: int) -> str:
    best = None
    for start, end, name in spans:
        if start < pos < end and (best is None or start > best[0]):
            best = (start, name)
    return best[1] if best else "<file scope>"


def normalize_catch(param: str) -> str:
    p = " ".join(param.split())
    if p == "...":
        return "..."
    m = re.match(r"(?:const\s+)?([\w:]+(?:\s*<[^>]*>)?)", p)
    return m.group(1).replace(" ", "") if m else p


def normalize_statement(s: str) -> str:
    s = " ".join(s.split())
    s = s.replace("superslm::SslmForwardStatus::", "").replace("SslmForwardStatus::", "")
    return s


def census(root: Path) -> Counter:
    rows: Counter = Counter()
    gpu = root / GPU_DIR
    files = sorted(p for p in gpu.rglob("*") if p.suffix in CPP_SUFFIXES and p.is_file())
    status_files = files + [root / f for f in STATUS_EXTRA if (root / f).is_file()]
    for path in sorted(set(status_files)):
        rel = path.relative_to(root).as_posix()
        clean = strip_code(path.read_text(encoding="utf-8", errors="replace"))
        spans = function_spans(clean)
        if path in files:
            for m in re.finditer(r"\bcatch\s*\(", clean):
                depth, j = 1, m.end()
                while j < len(clean) and depth:
                    depth += {"(": 1, ")": -1}.get(clean[j], 0)
                    j += 1
                rows[("catch", rel, enclosing(spans, m.start()), normalize_catch(clean[m.end():j - 1]))] += 1
        seen: set[tuple[int, int]] = set()
        for m in re.finditer(r"\b(" + "|".join(STATUS_NAMES) + r")\b", clean):
            a = max(clean.rfind(";", 0, m.start()), clean.rfind("{", 0, m.start()), clean.rfind("}", 0, m.start())) + 1
            ends = [e for e in (clean.find(";", m.end()), clean.find("{", m.end())) if e >= 0]
            b = min(ends) if ends else len(clean)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows[("status", rel, enclosing(spans, m.start()), normalize_statement(clean[a:b]))] += 1
    return rows


def read_list(path: Path) -> tuple[Counter, list[str]]:
    rows: Counter = Counter()
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            problems.append(f"{path.name}:{lineno}: expected 7 tab-separated columns, found {len(parts)}")
            continue
        kind, rel, func, count, block, disposition, token = parts
        if kind not in ("catch", "status"):
            problems.append(f"{path.name}:{lineno}: kind must be catch or status, not {kind!r}")
            continue
        if not block.strip() or not disposition.strip():
            problems.append(f"{path.name}:{lineno}: a row without a block and a disposition is not a census row")
        try:
            rows[(kind, rel, func, token)] += int(count)
        except ValueError:
            problems.append(f"{path.name}:{lineno}: count {count!r} is not an integer")
    return rows, problems


def compare(source: Counter, listed: Counter) -> list[str]:
    diffs: list[str] = []
    for key in sorted(set(source) | set(listed)):
        s, l = source.get(key, 0), listed.get(key, 0)
        kind, rel, func, token = key
        if s > l:
            diffs.append(f"UNLISTED {kind} site: {rel} :: {func} :: {token} (source {s}, list {l})")
        elif l > s:
            diffs.append(f"STALE {kind} row: {rel} :: {func} :: {token} (source {s}, list {l})")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args(argv)
    source = census(args.root)
    if args.emit:
        for (kind, rel, func, token), count in sorted(source.items()):
            print("\t".join([kind, rel, func, str(count), "", "", token]))
        return 0
    if not args.list.is_file():
        print(f"FAIL: site list not found: {args.list}")
        return 2
    listed, problems = read_list(args.list)
    diffs = problems + compare(source, listed)
    n_catch = sum(c for k, c in source.items() if k[0] == "catch")
    n_status = sum(c for k, c in source.items() if k[0] == "status")
    if diffs:
        for d in diffs:
            print(d)
        print(f"FAIL: {len(diffs)} difference(s) between the GPU status-site census "
              f"({n_catch} catch, {n_status} status) and {args.list.name}")
        return 1
    print(f"OK: {n_catch} catch sites and {n_status} status sites, every one listed with a disposition")
    return 0


if __name__ == "__main__":
    sys.exit(main())
