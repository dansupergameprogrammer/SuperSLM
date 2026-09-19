"""T-2791 (Curie) -- SuperSLM 1.6.0, plan Sec3.2 / Sec3.4 row 7: the two T-2106 fault-pin
environment reads leave the shipping GPU library.

Plan Sec3.2: "Both B5 reads move behind a compile definition, SUPERSLM_GPU_T2106_FAULT_PINS. It is
defined only for the test and bench targets that exercise the T-2106 violation classes ... The
installed superslm_gpu library then contains no environment read that changes a GPU output."
Plan Sec3.4 row 7: "a source scan of the installed library's translation units finds no
getenv/GetEnvironmentVariable for the two names outside #ifdef SUPERSLM_GPU_T2106_FAULT_PINS".

Cells (each red at v1.5.0, where the reads are unconditional and the macro exists nowhere):
  test_pin_names_only_inside_the_gate
      Every code occurrence (comments stripped, string literals kept) of either variable name in
      src/ and include/ -- every translation unit and header the installed library compiles --
      lies inside a preprocessor block conditioned on SUPERSLM_GPU_T2106_FAULT_PINS. Scanning for
      the NAME rather than for getenv( catches a read through a named constant or any other
      environment API (_dupenv_s, GetEnvironmentVariableA, ...).
  test_shipping_library_does_not_define_the_gate
      CMakeLists.txt defines the macro neither globally nor on the superslm_gpu target (a gate the
      shipping build defines is no gate).
  test_violation_pin_harness_still_receives_the_pins
      The harness that plants the pins (build.bat's tools\\t2113_b5_async_smoke.cpp line, the
      plant-and-revert protocol tests/t2112-gpu-1p0-red-suite/dim6_determinism_red.cpp cites)
      defines the macro, so gating the reads does not silently kill the T-2106 violation pins.

The scanner's own vitality (StandardsDocument Sec4: a check shown able to fail) is pinned by the
test_scanner_* cells over constructed snippets: an ungated read, a read in the #else of the gate,
and a read under #ifndef of the gate must each be flagged; a gated read and a comment must not.

Oracle: the plan's text; the scanned files are the code under test. The executed half of row 7
is tests/t2791-gpu-prefill-read-red-suite/cell_env_pins_shipping_leg.cpp.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = "SUPERSLM_GPU_T2106_FAULT_PINS"
PIN_NAMES = ("SSLM_B5_ASYNC_DROP_UAV_REBIND", "SSLM_B5_ASYNC_SWAP_SRV_REBIND")
SOURCE_SUFFIXES = {".cpp", ".cc", ".h", ".hpp", ".inc", ".def", ".hlsl", ".hlsli"}


def strip_comments(text: str) -> str:
    """Removes // and /* */ comments, keeping string and character literals and line breaks."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            q = c
            j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            chunk = text[i : (n if j < 0 else j + 2)]
            out.append("\n" * chunk.count("\n"))
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


_DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")


def _cond_gates(kind: str, cond: str) -> bool:
    """True when a branch's condition requires the gate macro to be defined."""
    if kind == "ifdef":
        return cond.strip() == GATE
    if kind in ("if", "elif"):
        uses = [m.group(1) for m in re.finditer(r"(!?)\s*defined\s*\(?\s*" + GATE + r"\b", cond)]
        return bool(uses) and all(neg == "" for neg in uses) and "||" not in cond
    return False


def ungated_pin_occurrences(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every code occurrence of a pin name outside the gate."""
    findings: list[tuple[int, str]] = []
    stack: list[dict] = []  # each frame: {"gated": bool, "kind": str, "cond": str}
    for lineno, line in enumerate(strip_comments(text).splitlines(), start=1):
        m = _DIRECTIVE.match(line)
        if m:
            kind, cond = m.group(1), m.group(2)
            if kind in ("ifdef", "ifndef", "if"):
                stack.append({"gated": _cond_gates(kind, cond), "kind": kind, "cond": cond.strip()})
            elif kind == "elif" and stack:
                stack[-1]["gated"] = _cond_gates("elif", cond)
            elif kind == "else" and stack:
                top = stack[-1]
                top["gated"] = top["kind"] == "ifndef" and top["cond"] == GATE
            elif kind == "endif" and stack:
                stack.pop()
            continue
        if any(name in line for name in PIN_NAMES) and not any(f["gated"] for f in stack):
            findings.append((lineno, line.strip()))
    return findings


def installed_library_sources() -> list[Path]:
    files = []
    for root in (REPO / "src", REPO / "include"):
        files += [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES]
    return sorted(files)


# --- the three cells -----------------------------------------------------------------------


def test_pin_names_only_inside_the_gate() -> None:
    findings = []
    for path in installed_library_sources():
        for lineno, line in ungated_pin_occurrences(path.read_text(encoding="utf-8", errors="replace")):
            findings.append(f"{path.relative_to(REPO)}:{lineno}: {line}")
    assert not findings, (
        "the shipping GPU library still reads a T-2106 fault pin outside #ifdef " + GATE + ":\n" + "\n".join(findings)
    )


def _cmake_commands(text: str) -> list[tuple[str, str]]:
    """(command name lower-cased, argument text) for every CMake command, comments stripped."""
    text = re.sub(r"#[^\n]*", "", text)
    return [(m.group(1).lower(), m.group(2)) for m in re.finditer(r"\b([A-Za-z_]+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text)]


def cmake_gate_definitions(text: str) -> list[str]:
    cmds = _cmake_commands(text)
    bad = []
    for name, args in cmds:
        if GATE not in args:
            continue
        if name in ("add_compile_definitions", "add_definitions", "add_compile_options"):
            bad.append(f"{name}({args.strip()})")
        elif name in ("target_compile_definitions", "target_compile_options") and re.match(r"\s*superslm_gpu\b", args):
            bad.append(f"{name}({args.strip()})")
        elif name == "set" and re.match(r"\s*CMAKE_CXX_FLAGS", args):
            bad.append(f"{name}({args.strip()})")
    return bad


def test_shipping_library_does_not_define_the_gate() -> None:
    # Green at v1.5.0 by construction (the macro exists nowhere yet); it guards the gate from being
    # made vacuous by the build. Its ability to fail is test_scanner_cmake_check_can_fail.
    bad = cmake_gate_definitions((REPO / "CMakeLists.txt").read_text(encoding="utf-8"))
    assert not bad, "the shipping superslm_gpu build defines " + GATE + ": " + "; ".join(bad)


@pytest.mark.parametrize(
    "snippet, flagged",
    [
        ("add_compile_definitions(" + GATE + ")\n", True),
        ("target_compile_definitions(superslm_gpu PRIVATE " + GATE + ")\n", True),
        ("target_compile_options(superslm_gpu PRIVATE /D" + GATE + ")\n", True),
        ('set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D' + GATE + '")\n', True),
        ("target_compile_definitions(t2113_b5_async_smoke PRIVATE " + GATE + ")\n", False),
        ("# add_compile_definitions(" + GATE + ")\n", False),
    ],
)
def test_scanner_cmake_check_can_fail(snippet: str, flagged: bool) -> None:
    assert bool(cmake_gate_definitions(snippet)) is flagged


def _bat_command_containing(text: str, needle: str) -> str:
    """The whole caret-continued `cl` command whose text names `needle` (comment lines skipped)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("rem") or low.startswith("::") or needle.lower() not in low:
            continue
        start = i
        while start > 0 and lines[start - 1].rstrip().endswith("^"):
            start -= 1
        end = i
        while end < len(lines) - 1 and lines[end].rstrip().endswith("^"):
            end += 1
        cmd = " ".join(l.strip() for l in lines[start : end + 1])
        if cmd.lower().startswith("cl "):
            return cmd
    return ""


def test_violation_pin_harness_still_receives_the_pins() -> None:
    cmd = _bat_command_containing((REPO / "build.bat").read_text(encoding="utf-8"), r"tools\t2113_b5_async_smoke.cpp")
    assert cmd, r"build.bat no longer builds tools\t2113_b5_async_smoke.cpp"
    assert re.search(r"/D" + GATE + r"\b", cmd), (
        r"build.bat's tools\t2113_b5_async_smoke.cpp compile line does not define " + GATE
        + ": the T-2106 violation pins would be compiled out of the only harness that plants them"
    )


# --- the scanner's own vitality ------------------------------------------------------------

UNGATED = 'static const bool x = std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND") != nullptr;\n'


@pytest.mark.parametrize(
    "snippet, flagged",
    [
        (UNGATED, True),
        ("#ifdef " + GATE + "\n" + UNGATED + "#endif\n", False),
        ("#if defined(" + GATE + ")\n" + UNGATED + "#endif\n", False),
        ("#ifdef " + GATE + "\nint a;\n#else\n" + UNGATED + "#endif\n", True),
        ("#ifndef " + GATE + "\n" + UNGATED + "#endif\n", True),
        ("#ifndef " + GATE + "\nint a;\n#else\n" + UNGATED + "#endif\n", False),
        ("#if !defined(" + GATE + ")\n" + UNGATED + "#endif\n", True),
        ("#ifdef OTHER\n#ifdef " + GATE + "\n" + UNGATED + "#endif\n#endif\n", False),
        ("#ifdef " + GATE + "\n#endif\n" + UNGATED, True),
        ('// std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND")\n', False),
        ('/* "SSLM_B5_ASYNC_DROP_UAV_REBIND" */\n', False),
        ('constexpr const char* kPin = "SSLM_B5_ASYNC_DROP_UAV_REBIND";\n', True),
        ('const char* u = "http://x"; const char* p = "SSLM_B5_ASYNC_DROP_UAV_REBIND";\n', True),
    ],
)
def test_scanner_flags_exactly_the_ungated_occurrences(snippet: str, flagged: bool) -> None:
    assert bool(ungated_pin_occurrences(snippet)) is flagged


def test_scanner_reaches_the_real_files() -> None:
    names = {p.name for p in installed_library_sources()}
    assert {"superslm_gpu.cpp", "gpu_1p0.cpp", "d3d12_harness.h", "gpu_1p0.h"} <= names


def test_scanner_harness_check_can_fail() -> None:
    text = "cl /nologo /Iinclude ^\n\tsrc\\gpu\\gpu_1p0.cpp ^\n\ttools\\t2113_b5_async_smoke.cpp /Fe:out\\x.exe\n"
    cmd = _bat_command_containing(text, r"tools\t2113_b5_async_smoke.cpp")
    assert cmd.startswith("cl ") and not re.search(r"/D" + GATE + r"\b", cmd)
    gated = text.replace("/Iinclude", "/Iinclude /D" + GATE)
    assert re.search(r"/D" + GATE + r"\b", _bat_command_containing(gated, r"tools\t2113_b5_async_smoke.cpp"))
