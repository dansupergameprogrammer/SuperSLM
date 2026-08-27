"""run_fp_free_scan_real_corpus.py (T-2338, Brunel) -- population eight's real
occupant: a whole-corpus build-and-scan of the real, currently-committed
`SUPERSLM_CORE_SOURCES` (CMakeLists.txt), through the production instrument
(`check_fp_free_scan.py`) at `isa="x86-64"` -- the toolchain (MSVC cl.exe,
COFF) the design's own whole-corpus clean-scan claim (Sec4.1, fold rounds 8/
13/14) was measured on.

`test_check_fp_free_scan.py::test_population_08_real_corpus_whole_sweep` is,
by its own docstring, "Not gradable by this single pytest cell even once the
instrument exists -- a full-corpus build-and-scan is CI-scale, not a unit
cell." That test verifies the 17-file population is real and calls
`pytest.fail` unconditionally; it is not this population's own discharge.
THIS script is: it compiles all 17 real translation units fresh (never a
committed binary), scans each compiled object, and reports the real,
per-object and aggregate counts -- run once per session on this ticket, not
wired as a pytest cell, because there is no unit-cell shape for "the whole
corpus, for real."

Per `StandardsDocument.md` Sec5.4's commissioning rule, `check_fp_free_scan.py`'s
own verdicts are QUARANTINED until independently commissioned (this ticket's
own build record states the current status) -- this script's REJECT/ACCEPT
counts are reported as what the instrument returns, never boarded as an
answer about the corpus's actual FP-freedom.

Exit code is always 0: a REFUSE or a REJECT on the real corpus is not itself
a defect in this script or a build regression (design Sec4.1's own text
states the whole-corpus claim stays "[U]" -- retracted as unevidenced -- for
exactly this reason, pending a padding-recognizer residual this build does
not close); it is what the instrument found, printed for a human to read.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import check_fp_free_scan as scan  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

_VSDEVCMD_CANDIDATES = (
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
)


def _find_vsdevcmd():
    for c in _VSDEVCMD_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _compile_all(sources, out_dir):
    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        print("no VsDevCmd.bat found -- cannot compile the real corpus; skipping (non-fatal)")
        return None
    os.makedirs(out_dir, exist_ok=True)
    fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="t2338_realcorpus_")
    os.close(fd)
    obj_paths = []
    try:
        with open(bat_path, "w") as f:
            f.write("@echo off\r\n")
            f.write('call "{}" -arch=x64 -no_logo\r\n'.format(vsdevcmd))
            f.write('cd /d "{}"\r\n'.format(_REPO_ROOT))
            for src in sources:
                stem = os.path.splitext(os.path.basename(src))[0]
                obj = os.path.join(out_dir, stem + ".obj")
                obj_paths.append(obj)
                f.write(
                    'cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /c "{}" /Fo:"{}"\r\n'
                    .format(src, obj)
                )
                f.write("if errorlevel 1 exit /b 1\r\n")
        r = subprocess.run(["cmd", "/c", os.path.abspath(bat_path)],
                           capture_output=True, text=True, cwd=_REPO_ROOT)
        if r.returncode != 0:
            print("compiling the real corpus FAILED:\n{}\n{}".format(r.stdout, r.stderr))
            return None
    finally:
        try:
            os.remove(bat_path)
        except OSError:
            pass
    return obj_paths


def main() -> int:
    sources = scan.derive_core_sources()
    print("SUPERSLM_CORE_SOURCES: {} translation units".format(len(sources)))
    out_dir = os.path.join(_REPO_ROOT, "out", "fp_scan_real_corpus")
    obj_paths = _compile_all(sources, out_dir)
    if obj_paths is None:
        return 0  # toolchain absent: loud, non-fatal skip

    total_accept = 0
    total_reject = 0
    total_refuse_objects = 0
    total_unclassified = 0
    per_object = []
    for src, obj in zip(sources, obj_paths):
        result = scan.scan_object(obj, isa="x86-64")
        if result.refuse:
            total_refuse_objects += 1
            total_unclassified += result.unclassified_bytes
        else:
            n_accept = sum(1 for v in result.verdicts.values() if v == "ACCEPT")
            n_reject = sum(1 for v in result.verdicts.values() if v == "REJECT")
            total_accept += n_accept
            total_reject += n_reject
        per_object.append((src, result))

    print()
    print("Per-object results (isa=x86-64, object_format read from each object's own header):")
    for src, result in per_object:
        if result.refuse:
            print("  REFUSE  {}  (unclassified_bytes={}, format={})".format(
                src, result.unclassified_bytes, result.object_format))
        else:
            n_accept = sum(1 for v in result.verdicts.values() if v == "ACCEPT")
            n_reject = sum(1 for v in result.verdicts.values() if v == "REJECT")
            print("  scanned {}  ACCEPT={} REJECT={} format={}".format(
                src, n_accept, n_reject, result.object_format))

    print()
    print("Aggregate: {} of {} translation units REFUSE; {} ACCEPT / {} REJECT across the "
          "remaining {}; {} total unclassified bytes among refused objects.".format(
              total_refuse_objects, len(sources), total_accept, total_reject,
              len(sources) - total_refuse_objects, total_unclassified))
    print(
        "QUARANTINED per StandardsDocument.md Sec5.4: these counts are what the instrument "
        "returns on this run, not yet an independently commissioned verdict about the "
        "corpus's own FP-freedom."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
