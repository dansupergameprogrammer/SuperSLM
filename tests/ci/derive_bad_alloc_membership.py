"""Independent reference derivation of the "throws only std::bad_alloc"
contract's membership population (S-HARDEN-7, F5; Claude/Vitruvius/
SuperSLM_SHARDEN678_Bundle_Design-2026-07-23.md Sec3.1's corrected,
four-condition rule; T-411).

This is Curie's independent second read of Sec3.1's rule -- adapted from the
adversary's four probe scripts (Claude/Loki/superslm-sharden678-bundle-
strike-2026-07-23-probe-*), which implemented the rule's PRIOR, three-
condition form (the one the strike proved TU-dependent) and derived 12/13/14
depending on scan strategy. This module extends that same walk with the two
conditions the fold that closed the strike added:

  - condition 2 (new): exclude the enclosing class's own copy constructor,
    move constructor, copy-assignment operator, and move-assignment operator
    by a STRUCTURAL test (enclosing class name + the one parameter's stripped
    type), tested before the noexcept test -- this is what makes the
    population independent of which translation unit happens to be scanned
    (design Sec3.1, "fourth-pass" fold).
  - condition 4(b) (new): admit any function whose return type is
    `std::string` or `std::vector<T>`, regardless of parameter shape --
    closes the promise/rule scope mismatch the strike surveyed but did not
    prosecute (Sec6 of the strike record).

Per StandardsDocument Sec4: "a new structure is validated against an
independently-found population before it is trusted." The four Loki probes
ARE that independently-found population for the prior rule; this module
re-derives the population under the CORRECTED rule and this package's own
test suite (test_membership_check_population.py) confirms it reproduces the
design's stated eighteen, from the real headers on disk, under three
independent scan strategies -- the direct re-run of the property the strike
found false of the prior rule.

This is a REFERENCE tool, not the CI gate. The CI gate itself
(design Sec3.1: "tools/ci/check_bad_alloc_contract.py or the build seat's
equivalent placement") is the build seat's production infrastructure -- step
2 of the design's own three-step gate (derive membership; check wrapping;
fail naming every unwrapped member) is unbuildable until each site is
wrapped. This module implements only step 1 (derive membership), as the
oracle the CI gate's own step-1 output must match once built.

Invokes `clang++ -Xclang -ast-dump=json` as a subprocess per scan (no AST
dump is ever written to a tracked file -- a single header's dump is on the
order of 100+ MB of JSON, and the point of this module is that the dump is
reproducible from the headers on demand, not that a snapshot of it is
committed). Requires a clang++ on PATH capable of `-Xclang -ast-dump=json`
(pinned per design Sec3.1: ships pre-installed on GitHub's stock
`ubuntu-latest` runner image; locally, set SUPERSLM_CLANGXX to an explicit
path if `clang++` does not resolve to one).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_INCLUDE_DIR = os.path.join(_REPO_ROOT, "include")

_DEFAULT_CLANGXX = os.environ.get("SUPERSLM_CLANGXX", "clang++")

HEADERS = (
    "artifact",
    "model",
    "tokenizer",
    "proof_manifest",
    "intmath",
    "matmul",
    "sha256",
    "silu_lut",
    "silu_lut_canonical",
)

_FUNC_KINDS = {
    "FunctionDecl",
    "CXXMethodDecl",
    "CXXConstructorDecl",
    "CXXDestructorDecl",
    "CXXConversionDecl",
}

_VECTOR_RETURN_RE = re.compile(r"^std::vector<.*>$")


class ClangUnavailable(RuntimeError):
    """Raised when clang++ cannot run -Xclang -ast-dump=json at all -- the
    environment problem, distinct from a genuine population mismatch."""


def _strip(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


def _strip_ref_and_const(t: str) -> str:
    """Strip a trailing/leading const and any reference qualifier, matching
    condition 2's "the one parameter's stripped type" test and condition
    4(a)'s parameter-type comparisons."""
    t = _strip(t)
    t = t.replace("&&", "").replace("&", "").strip()
    if t.startswith("const "):
        t = t[len("const "):].strip()
    return t


def _condition4a(params: list[tuple[str, str]]) -> str | None:
    """At least one EXPLICIT parameter is SslmArtifact, SslmSectionView, a
    (const uint8_t*, size_t) adjacent pair, or `const char*` named `path`."""
    types = [_strip(t) for _, t in params]
    for i, (nm, t) in enumerate(params):
        t = _strip(t)
        base = _strip_ref_and_const(t)
        if base in ("SslmArtifact", "SslmSectionView"):
            return "4(a): param %d type %s" % (i, t)
        if t == "const char *" and nm == "path":
            return "4(a): param %d const char* named path" % i
        if t in ("const uint8_t *", "const unsigned char *") and i + 1 < len(types):
            nxt = types[i + 1]
            if nxt in ("size_t", "unsigned long long", "unsigned __int64", "std::size_t"):
                return "4(a): param %d/%d (const uint8_t*, size_t) pair" % (i, i + 1)
    return None


def _condition4b(return_type: str) -> str | None:
    """Return type is std::string or std::vector<T>, regardless of parameter
    shape (this fold's addition, closing the promise/rule scope mismatch)."""
    rt = _strip(return_type)
    if rt == "std::string":
        return "4(b): returns std::string"
    if _VECTOR_RETURN_RE.match(rt):
        return "4(b): returns %s" % rt
    return None


def _scan_ast_root(root: dict, header_name: str) -> list[dict]:
    """One parsed AST dump -> every member the four-condition rule admits,
    scanning ONLY declarations whose source file ends with `header_name` (so
    a single-TU scan that #includes several headers attributes each
    declaration to its own originating file, never double-counting)."""
    hits: list[dict] = []
    state = {"file": None}
    class_stack: list[str] = []

    def walk(n: dict) -> None:
        loc = n.get("loc") or {}
        if "file" in loc:
            state["file"] = loc["file"]
        cur_file = str(state["file"] or "").replace("\\", "/")

        kind = n.get("kind")
        pushed_class = False
        if kind == "CXXRecordDecl" and n.get("name"):
            class_stack.append(n["name"])
            pushed_class = True

        if kind in _FUNC_KINDS and cur_file.endswith(header_name) and not n.get("isImplicit"):
            params = [
                (c.get("name"), c.get("type", {}).get("qualType", ""))
                for c in (n.get("inner") or [])
                if c.get("kind") == "ParmVarDecl"
            ]
            qual_type = n.get("type", {}).get("qualType", "")
            name = n.get("name", "")
            enclosing = class_stack[-1] if class_stack else None

            # Condition 1: not =delete'd.
            deleted = bool(n.get("explicitlyDeleted"))
            if not deleted:
                # Condition 2: not the enclosing class's own copy/move special member.
                is_ctor = kind == "CXXConstructorDecl"
                is_assign_op = kind == "CXXMethodDecl" and name == "operator="
                is_special = False
                if (is_ctor or is_assign_op) and enclosing is not None and len(params) == 1:
                    param_type = _strip_ref_and_const(params[0][1])
                    if param_type == enclosing:
                        is_special = True

                if not is_special:
                    written_noexcept = "noexcept" in qual_type  # condition 3 (written specifier)
                    if not written_noexcept:
                        return_type = (
                            "" if kind == "CXXConstructorDecl" else qual_type.split("(", 1)[0].strip()
                        )
                        reason_a = _condition4a(params)
                        reason_b = _condition4b(return_type) if return_type else None
                        reason = reason_a or reason_b
                        if reason:
                            hits.append(
                                {
                                    "header": header_name,
                                    "line": loc.get("line"),
                                    "kind": kind,
                                    "name": name,
                                    "enclosing": enclosing,
                                    "type": qual_type,
                                    "reason": reason,
                                    "defaulted": n.get("explicitlyDefaulted"),
                                }
                            )

        for c in n.get("inner") or []:
            walk(c)

        if pushed_class:
            class_stack.pop()

    walk(root)
    return hits


def _dedup_sort(hits: list[dict]) -> list[dict]:
    seen = set()
    uniq = []
    for x in hits:
        key = (x["header"], x["line"], x["name"], x["type"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)
    uniq.sort(key=lambda x: (x["header"], x["line"] or 0))
    return uniq


def run_clang_ast_dump(source_path: str, include_dir: str = _INCLUDE_DIR,
                        clangxx: str = _DEFAULT_CLANGXX) -> dict:
    """Run `clangxx -Xclang -ast-dump=json` against `source_path` and return
    the parsed JSON root. Raises ClangUnavailable if clang++ cannot be
    invoked at all (missing binary, wrong flag support); a genuine parse
    error in the source itself raises RuntimeError with clang's stderr."""
    cmd = [
        clangxx,
        "-std=c++20",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-I",
        include_dir,
        "-Xclang",
        "-ast-dump=json",
        "-fsyntax-only",
        "-x",
        "c++",
        source_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except FileNotFoundError as e:
        raise ClangUnavailable(f"{clangxx} not found on PATH: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ClangUnavailable(f"{clangxx} timed out dumping {source_path}") from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"clang++ AST dump failed for {source_path} (exit {proc.returncode}):\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return json.loads(proc.stdout)


def derive_population_per_header(headers: tuple[str, ...] = HEADERS,
                                  include_dir: str = _INCLUDE_DIR,
                                  clangxx: str = _DEFAULT_CLANGXX) -> list[dict]:
    """Scan strategy 1: one translation unit per header (nine independent
    scans)."""
    all_hits: list[dict] = []
    for h in headers:
        header_path = os.path.join(include_dir, "superslm", h + ".h")
        root = run_clang_ast_dump(header_path, include_dir, clangxx)
        all_hits.extend(_scan_ast_root(root, h + ".h"))
    return _dedup_sort(all_hits)


def derive_population_single_tu(headers: tuple[str, ...] = HEADERS,
                                 include_dir: str = _INCLUDE_DIR,
                                 clangxx: str = _DEFAULT_CLANGXX,
                                 force_odr_use_artifact_moves: bool = False) -> list[dict]:
    """Scan strategy 2 (all nine headers in a single TU) and strategy 3 (the
    same TU, plus code that odr-uses SslmArtifact's move constructor and
    move-assignment operator -- the exact translation unit shape the strike
    showed collapses the prior rule's population to 12)."""
    body = "\n".join(f'#include "superslm/{h}.h"' for h in headers) + "\n"
    if force_odr_use_artifact_moves:
        body += (
            "\nnamespace {\n"
            "inline void _odr_use_artifact_moves() {\n"
            "  superslm::SslmArtifact a;\n"
            "  superslm::SslmArtifact b(static_cast<superslm::SslmArtifact&&>(a));\n"
            "  superslm::SslmArtifact c;\n"
            "  c = static_cast<superslm::SslmArtifact&&>(b);\n"
            "  (void)c;\n"
            "}\n"
            "}  // namespace\n"
        )
    fd, tu_path = tempfile.mkstemp(suffix=".cpp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        root = run_clang_ast_dump(tu_path, include_dir, clangxx)
        all_hits: list[dict] = []
        for h in headers:
            all_hits.extend(_scan_ast_root(root, h + ".h"))
        return _dedup_sort(all_hits)
    finally:
        os.remove(tu_path)


def derive_population_from_headers_dir(headers_root: str, headers: tuple[str, ...] = HEADERS,
                                        clangxx: str = _DEFAULT_CLANGXX) -> list[dict]:
    """Same as derive_population_per_header but against an arbitrary
    checkout root (headers_root/include/superslm/*.h), for use against a
    clean `git archive` extract rather than the working tree."""
    include_dir = os.path.join(headers_root, "include")
    return derive_population_per_header(headers, include_dir, clangxx)


def format_population(members: list[dict]) -> str:
    lines = ["DERIVED POPULATION: %d members" % len(members), ""]
    for m in members:
        d = " [explicitly-defaulted]" if m["defaulted"] == "default" else ""
        lines.append(
            "  %-24s :%-4s %-28s %s%s\n        %s"
            % (m["header"], m["line"], m["name"], m["type"], d, m["reason"])
        )
    return "\n".join(lines)


if __name__ == "__main__":
    pop = derive_population_per_header()
    print(format_population(pop))
