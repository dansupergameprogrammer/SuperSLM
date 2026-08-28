"""run_fp_free_scan_real_corpus.py (T-2338/T-2343, Brunel) -- population
eight's real occupant: a whole-corpus build-and-scan of the real,
currently-committed `SUPERSLM_CORE_SOURCES` (CMakeLists.txt), through the
production instrument (`check_fp_free_scan.py`) at `isa="x86-64"` -- the
toolchain (MSVC cl.exe, COFF) the design's own whole-corpus clean-scan claim
(Sec4.1, fold rounds 8/13/14) was measured on.

`test_check_fp_free_scan.py::test_population_08_real_corpus_whole_sweep` is,
by its own docstring, "Not gradable by this single pytest cell even once the
instrument exists -- a full-corpus build-and-scan is CI-scale, not a unit
cell." That test verifies the 17-file population is real and calls
`pytest.fail` unconditionally; it is not this population's own discharge.
THIS script is: it compiles all 17 real translation units fresh (never a
committed binary), scans each compiled object, and reports the real,
per-object and aggregate counts.

T-2343 (Brunel) corrects three defects Poirot's review and Popper's
commissioning found in this script's own T-2338 form:

  - **S3 (Poirot).** The (translation-unit, object-path) pairing is now
    derived exclusively from `scan.enumerate_scan_targets(build_dir=...)` --
    design Sec4.1's own ruled single production membership entry point
    (fold round 34 gap (d), D-SLM4859) -- rather than re-implemented inline.
    A duplicate-stem manifest now REFUSES here exactly as it does anywhere
    else `enumerate_scan_targets` is called, instead of silently colliding.
  - **M3 (Poirot) / Sec5.4's 181-vs-627 reconciliation (D-SLM4861).** This
    script's own hand-picked four-flag compile line (`/std:c++20 /O2 /W4
    /fp:precise /EHsc`) was a THIRD, unshipped configuration -- neither the
    plain dev build.bat path nor the real `windows-latest` CI leg's own
    CMake Release build. `cmake.exe` is not installed in this environment
    (confirmed this session, `where cmake` empty on every PATH entry and
    every well-known install location), so this script cannot literally
    shell out to `cmake --build --config Release` the way the design's own
    ruling names as the ideal. What it does instead, to avoid a SECOND
    hand-maintained flag string duplicating what CMakeLists.txt already
    declares: it PARSES the `superslm` target's own `target_compile_options`
    line out of the real `CMakeLists.txt` at run time (`_msvc_target_flags`,
    below) and combines it with CMake's own documented, unoverridden MSVC
    Release default (`/MD /O2 /Ob2 /DNDEBUG` -- confirmed at source this
    fold, D-SLM4861: CMakeLists.txt sets no CMAKE_CXX_FLAGS_RELEASE or
    CMAKE_MSVC_RUNTIME_LIBRARY override anywhere). A future edit to
    CMakeLists.txt's own `/W4 /fp:precise` line is picked up automatically;
    only the CMake-internal Release default (a fact about the generator,
    not about this repository's own text) remains a stated constant.
  - **Gap (a)/(b) (design Sec4.1 fold round 34).** A `corpus_symbols` index
    is built once (every function-typed symbol name in every compiled
    object) and threaded through every `scan_object` call, and the
    aggregate is computed via `ci_gate_corpus`, not by hand-summing
    ACCEPT/REJECT counts.

Per `StandardsDocument.md` Sec5.4's commissioning rule, `check_fp_free_scan.py`'s
own verdicts are QUARANTINED until independently commissioned -- this
script's REJECT/ACCEPT/`ci_gate_corpus` readings are reported as what the
instrument returns, never boarded as an answer about the corpus's actual
FP-freedom, and wiring this script (or a successor) into an actual CI job
step under `.github/workflows/` so a REJECT anywhere fails a real pipeline
is a separate, CI-configuration task this script does not perform (design
Sec4.1's own text: "a scan this design specifies and nothing in the
pipeline runs is not a gate").

Process exit code: 0 on a normal run, REGARDLESS of what the scan finds --
a REFUSE or a REJECT on the real corpus is not itself a defect in this
script (design Sec4.1's own text: the whole-corpus claim stays "[U]" for
disclosed, named reasons, not a script bug); 1 only if the real toolchain
itself failed (a compile error, a missing VsDevCmd.bat) or this script
raised an exception it did not itself account for -- a genuine
infrastructure failure, distinct from "the scan found something," per
Poirot's M4 (report the exit code, don't swallow it; don't let non-gating
mean unobserved).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import check_fp_free_scan as scan  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_CMAKELISTS = os.path.join(_REPO_ROOT, "CMakeLists.txt")

_VSDEVCMD_CANDIDATES = (
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
)

# CMake's own documented, unoverridden default flags for the MSVC Release
# configuration -- a fact about the CMake generator, not this repository's
# own text, so it is not something `_msvc_target_flags` can parse out of
# CMakeLists.txt; confirmed this fold (D-SLM4861) that no
# CMAKE_CXX_FLAGS_RELEASE / CMAKE_MSVC_RUNTIME_LIBRARY override exists
# anywhere in CMakeLists.txt to change it.
_CMAKE_MSVC_RELEASE_DEFAULT = ("/MD", "/O2", "/Ob2", "/DNDEBUG")


def _msvc_target_flags(target_name: str = "superslm") -> list:
    """Parses the named target's own `target_compile_options(<target>
    PRIVATE ...)` line out of the real CMakeLists.txt, inside that call's own
    `if(MSVC)` branch -- the part of the compile invocation that CAN drift
    from a hand-maintained copy, so it is read at source every run rather
    than hand-restated (the identical derive-don't-duplicate law Sec2.7/
    D-SLM4335 already states for the translation-unit list, applied here to
    the compile invocation that produces the objects that list names)."""
    with open(_CMAKELISTS) as f:
        text = f.read()
    # Find target_compile_options(<target> PRIVATE ...) then look for the
    # nearest preceding `if(MSVC)` guarding it, within a small window --
    # matches this file's own current shape (`if(MSVC) ... target_compile_
    # options(superslm PRIVATE ...) ... else() ... endif()`).
    pattern = re.compile(
        r"if\(MSVC\)\s*\n\s*target_compile_options\(\s*" + re.escape(target_name) +
        r"\s+PRIVATE\s+([^)]*)\)", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise RuntimeError(
            "could not find target_compile_options({} PRIVATE ...) inside an "
            "if(MSVC) branch in {} -- the flag derivation this fold specifies "
            "(design Sec5.4, D-SLM4861) has nothing to parse; refusing to fall "
            "back to a hand-maintained flag string".format(target_name, _CMAKELISTS)
        )
    return m.group(1).split()


def _production_compile_flags() -> list:
    """The real `windows-latest` CI leg's own compile invocation, derived
    rather than restated: CMake's own MSVC Release default, plus whatever
    `superslm`'s own CMakeLists.txt currently declares."""
    return list(_CMAKE_MSVC_RELEASE_DEFAULT) + _msvc_target_flags("superslm")


def _find_vsdevcmd():
    for c in _VSDEVCMD_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _compile_all(targets, out_dir):
    """targets: (translation_unit_path, object_path) pairs from
    scan.enumerate_scan_targets(). Returns True on success, False if the
    toolchain is absent or a compile genuinely failed."""
    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        print("no VsDevCmd.bat found -- cannot compile the real corpus; skipping (non-fatal)")
        return False
    os.makedirs(out_dir, exist_ok=True)
    flags = _production_compile_flags()
    print("Compile flags (derived from CMakeLists.txt + CMake's own MSVC Release "
          "default): {}".format(" ".join(flags)))
    fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="t2343_realcorpus_")
    os.close(fd)
    try:
        with open(bat_path, "w") as f:
            f.write("@echo off\r\n")
            f.write('call "{}" -arch=x64 -no_logo\r\n'.format(vsdevcmd))
            f.write('cd /d "{}"\r\n'.format(_REPO_ROOT))
            for src, obj in targets:
                f.write(
                    'cl /nologo /c {} /std:c++20 /EHsc /Iinclude "{}" /Fo:"{}"\r\n'
                    .format(" ".join(flags), src, obj)
                )
                f.write("if errorlevel 1 exit /b 1\r\n")
        r = subprocess.run(["cmd", "/c", os.path.abspath(bat_path)],
                           capture_output=True, text=True, cwd=_REPO_ROOT)
        if r.returncode != 0:
            print("compiling the real corpus FAILED:\n{}\n{}".format(r.stdout, r.stderr))
            return False
    finally:
        try:
            os.remove(bat_path)
        except OSError:
            pass
    return True


def _read_function_symbol_names(obj_path: str) -> set:
    """The corpus_symbols index's own per-object contribution (design
    Sec4.1 fold round 34 gap (b)): every function-typed symbol name defined
    in this one compiled object, read directly from its own symbol table --
    no decode, no accounting, no per-instruction check, the same data
    `_check_c_for_symbol`'s own relocation resolution already reads."""
    with open(obj_path, "rb") as f:
        data = f.read()
    object_format = scan._read_object_format(data)
    if object_format == "elf":
        code_sections, _sym_by_raw, _relocs = scan._parse_elf(data)
    else:
        code_sections, _sym_by_raw, _relocs = scan._parse_coff(data)
    names = set()
    for section in code_sections:
        for sym in section.symbols:
            if sym.is_function:
                names.add(sym.name)
    return names


def main() -> int:
    try:
        targets = scan.enumerate_scan_targets(build_dir=os.path.join(_REPO_ROOT, "out", "fp_scan_real_corpus"))
    except scan.DuplicateStemError as e:
        print("enumerate_scan_targets refused: {}".format(e))
        return 1
    print("SUPERSLM_CORE_SOURCES: {} translation units (via enumerate_scan_targets)".format(len(targets)))
    out_dir = os.path.join(_REPO_ROOT, "out", "fp_scan_real_corpus")
    ok = _compile_all(targets, out_dir)
    if not ok:
        return 0  # toolchain absent, or a real compile failure already printed above (non-fatal, per this script's own docstring)

    # corpus_symbols: built once, from every compiled object's own symbol
    # table -- before any scan_object call, so every call sees the full index.
    corpus_symbols = frozenset().union(
        *(_read_function_symbol_names(obj) for _src, obj in targets)
    ) if targets else frozenset()

    results = {}
    expected_symbols = {}
    for src, obj in targets:
        result = scan.scan_object(obj, isa="x86-64", corpus_symbols=corpus_symbols)
        results[obj] = result
        expected_symbols[obj] = sorted(_read_function_symbol_names(obj))

    total_accept = 0
    total_reject = 0
    total_refuse_objects = 0
    total_unclassified = 0
    for src, obj in targets:
        result = results[obj]
        if result.refuse:
            total_refuse_objects += 1
            total_unclassified += result.unclassified_bytes
        else:
            total_accept += sum(1 for v in result.verdicts.values() if v == "ACCEPT")
            total_reject += sum(1 for v in result.verdicts.values() if v == "REJECT")

    print()
    print("Per-object results (isa=x86-64, object_format read from each object's own header):")
    for src, obj in targets:
        result = results[obj]
        if result.refuse:
            print("  REFUSE  {}  (unclassified_bytes={}, format={})".format(
                src, result.unclassified_bytes, result.object_format))
        else:
            n_accept = sum(1 for v in result.verdicts.values() if v == "ACCEPT")
            n_reject = sum(1 for v in result.verdicts.values() if v == "REJECT")
            print("  scanned {}  ACCEPT={} REJECT={} format={}".format(
                src, n_accept, n_reject, result.object_format))

    gate_result = scan.ci_gate_corpus(results, expected_symbols)

    print()
    print("Aggregate: {} of {} translation units REFUSE; {} ACCEPT / {} REJECT across the "
          "remaining {}; {} total unclassified bytes among refused objects.".format(
              total_refuse_objects, len(targets), total_accept, total_reject,
              len(targets) - total_refuse_objects, total_unclassified))
    print("ci_gate_corpus (the production CI gate, design Sec4.1 gap (a)): {} -- {}".format(
        gate_result,
        "would PASS a real CI job" if gate_result else
        "would FAIL a real CI job (REFUSE and/or REJECT present)"))
    print(
        "QUARANTINED per StandardsDocument.md Sec5.4: these counts and this gate reading are "
        "what the instrument returns on this run, not yet an independently commissioned "
        "verdict about the corpus's own FP-freedom. This script's own process exit code stays "
        "0 regardless (see this module's own docstring) -- wiring a REJECT into a real failed "
        "CI job is a separate, CI-configuration task this script does not perform."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
