"""check_fused_site_barriers.py -- GPU per-token overhead plan rev 3.1 §7 cell 7.7: every fused
site shader's `main` has a `DeviceMemoryBarrierWithGroupSync()` between consecutive body calls.

WHY A SOURCE CHECK. A fused shader (`fused_*_site.hlsl`, plan §3.2/§3.3) runs several site bodies
from `fused_site_bodies.hlsli` in one 256-thread group, in place of several dispatches with a global
UAV barrier between them. The in-group barrier between two bodies is what makes the later body see
the earlier one's UAV writes (its sticky word, its staged codes and scale). A mutant that deletes it
is not reliably killable by a GPU cell: the race it opens may pass on any given run, vendor or
driver. So this check reads the source instead and fails the build when the barrier is missing.

WHAT IS CHECKED, per shader. The population is every `src/gpu/shaders/*.hlsl` whose `main` calls a
body (an identifier ending in `SiteBody`), plus every `fused_*.hlsl` by name, so a fused shader whose
calls this check fails to recognize is a failure, not a silent skip. For each one, comments
stripped:
  1. `main` calls at least two bodies (a fused shader with one body is not fused);
  2. every body call is a top-level statement of `main` (brace depth 1), never inside a branch;
  3. between each pair of consecutive body calls the text is exactly one
     `DeviceMemoryBarrierWithGroupSync();` and nothing else -- so a barrier moved into a branch,
     replaced by a weaker `GroupMemoryBarrierWithGroupSync()`, or separated from the calls by other
     code also fails.
A wrapper (`o_proj_site.hlsl` ...) calls one body and is not in the population; the population
must be non-empty.

`--self-test` re-runs the check on in-memory mutants of every real fused shader (each barrier
deleted in turn, and each replaced by `GroupMemoryBarrierWithGroupSync()`) and fails unless every
mutant is reported, so the check's own vitality is executable rather than asserted.

Exit status: 0 OK; 1 a shader fails (each failure named) or the self-test finds a surviving mutant.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
SHADERS_DIR = os.path.join(_REPO_ROOT, "src", "gpu", "shaders")

BARRIER = "DeviceMemoryBarrierWithGroupSync"
_BODY_CALL = re.compile(r"\b([A-Za-z_]\w*SiteBody)\s*\(")
_MAIN = re.compile(r"\bvoid\s+main\s*\(")


def strip_comments(text: str) -> str:
    """// and /* */ comments replaced by spaces (newlines kept); HLSL here has no string literals
    that could contain comment markers, so no string handling is needed."""
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _main_body(code: str) -> str | None:
    """The text between `main`'s own braces, or None when the file has no `main`."""
    m = _MAIN.search(code)
    if not m:
        return None
    open_i = code.find("{", m.end())
    depth = 0
    for i in range(open_i, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_i + 1:i]
    return None


def _call_end(body: str, open_paren: int) -> int:
    """Index just past the `;` that ends the call whose `(` is at `open_paren`."""
    depth = 0
    for i in range(open_paren, len(body)):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < len(body) and body[j].isspace():
                    j += 1
                return j + 1 if j < len(body) and body[j] == ";" else i + 1
    return len(body)


def check_shader_text(name: str, text: str) -> tuple[bool, list[str]]:
    """(is_in_population, failures) for one shader's source text."""
    body = _main_body(strip_comments(text))
    if body is None:
        return (name.startswith("fused_"), [f"{name}: no `main` found"] if name.startswith("fused_") else [])
    calls = [(m.start(), m.end() - 1, m.group(1)) for m in _BODY_CALL.finditer(body)]
    in_population = bool(calls) and (len(calls) >= 2 or name.startswith("fused_"))
    if not in_population and not name.startswith("fused_"):
        return (False, [])
    failures: list[str] = []
    if len(calls) < 2:
        failures.append(f"{name}: `main` calls {len(calls)} site body(ies); a fused shader calls >= 2")
    for start, _paren, fn in calls:
        depth = body[:start].count("{") - body[:start].count("}")
        if depth != 0:
            failures.append(f"{name}: `{fn}(...)` is not a top-level statement of `main` (inside a block)")
    for (s0, p0, f0), (s1, _p1, f1) in zip(calls, calls[1:]):
        between = body[_call_end(body, p0):s1]
        if re.fullmatch(r"\s*" + BARRIER + r"\s*\(\s*\)\s*;\s*", between) is None:
            shown = " ".join(between.split()) or "(nothing)"
            failures.append(
                f"{name}: between `{f0}(...)` and `{f1}(...)` expected exactly one `{BARRIER}();`, "
                f"found: {shown}")
    return (True, failures)


def run_check(shaders_dir: str = SHADERS_DIR, overrides: dict[str, str] | None = None) -> tuple[int, list[str]]:
    """(population size, failures) over every `.hlsl` in `shaders_dir`; `overrides` maps a file name
    to replacement text (the self-test's mutants)."""
    population = 0
    failures: list[str] = []
    for fn in sorted(os.listdir(shaders_dir)):
        if not fn.endswith(".hlsl"):
            continue
        if overrides and fn in overrides:
            text = overrides[fn]
        else:
            with open(os.path.join(shaders_dir, fn), "r", encoding="utf-8") as f:
                text = f.read()
        in_pop, fails = check_shader_text(fn, text)
        population += int(in_pop)
        failures += fails
    if population == 0:
        failures.append(f"no fused shader found under {shaders_dir} (empty population)")
    return population, failures


def self_test(shaders_dir: str = SHADERS_DIR) -> list[str]:
    """Every barrier of every real fused shader, deleted or weakened in turn, must be reported."""
    survivors: list[str] = []
    mutants = 0
    for fn in sorted(os.listdir(shaders_dir)):
        if not (fn.startswith("fused_") and fn.endswith(".hlsl")):
            continue
        with open(os.path.join(shaders_dir, fn), "r", encoding="utf-8") as f:
            text = f.read()
        sites = [m.start() for m in re.finditer(BARRIER + r"\s*\(\s*\)\s*;", text)]
        for k, at in enumerate(sites):
            end = text.index(";", at) + 1
            for label, repl in (("deleted", ""), ("weakened", "GroupMemoryBarrierWithGroupSync();")):
                mutants += 1
                _, fails = run_check(shaders_dir, {fn: text[:at] + repl + text[end:]})
                if not fails:
                    survivors.append(f"{fn}: barrier #{k + 1} {label} -- mutant NOT reported")
    if mutants == 0:
        survivors.append("self-test built no mutants (no fused shader with a barrier found)")
    return survivors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shaders-dir", default=SHADERS_DIR)
    parser.add_argument("--self-test", action="store_true",
                        help="also prove every barrier-deletion/weakening mutant is reported")
    args = parser.parse_args(argv)
    population, failures = run_check(args.shaders_dir)
    if args.self_test:
        failures += self_test(args.shaders_dir)
    if failures:
        print("check_fused_site_barriers.py: FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"check_fused_site_barriers.py: OK -- {population} fused shader(s), every pair of consecutive "
          f"body calls in `main` separated by exactly one {BARRIER}()"
          + (" (self-test: every mutant reported)" if args.self_test else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
