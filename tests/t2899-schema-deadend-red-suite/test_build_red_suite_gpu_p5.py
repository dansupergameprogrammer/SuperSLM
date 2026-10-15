"""P-5 pins for build_red_suite_gpu.bat's pre-SLM5 (gpu_fixed_noslm5) configuration (the 1.12.0 review):

  compile-fails  a failing compile of gpu_1p0_fixed.cpp clears PRISTINE_OK, so the slm4_dump link never runs, and it
                 leaves no gpu_fixed_noslm5 object and no slm4_blob.bin an earlier run wrote
  dump-fails     a failing compile of cell_gpu_slm4_dump.cpp leaves no obj_gpu\\slm4_dump.obj an earlier run wrote, so
                 the slm4_dump link cannot pick up a stale object
  control        with every compile passing, the slm4_dump link runs against gpu_fixed_noslm5's objects, which are
                 this run's (an earlier run's objects are replaced), and an earlier run's slm4_blob.bin is removed
                 before the dump step (the stub link writes no executable, so the dump writes no new blob)

The script runs for real under cmd, copied into a temporary tree with its one `call VsDevCmd.bat` line made a remark
(and, under Wine only, each `A || (B)` written as `A` then `if errorlevel 1 (B)`: Wine's cmd mis-runs `||`), so stub cl.exe, link.exe and python.exe stand in for the toolchain and the interpreter. Each stub appends its argv to
P5_STUB_LOG; the stub cl exits 2 when any argument contains P5_CL_FAIL and otherwise writes an object for each source
(at /Fo's file, or under /Fo's directory). The stubs are built from C at each run: with x86_64-w64-mingw32-gcc and run
under Wine off Windows; with MSVC (vcvars64.bat) on Windows. Without either, the cells fail, never skip.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "build_red_suite_gpu.bat"
SUITE = HERE.name
VCVARS = Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat")
VSDEVCMD_LINE = b'call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\VsDevCmd.bat"'
STALE = b"stale object from an earlier run"

STUB_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void write_object(const char* path) {
    FILE* f = fopen(path, "wb");
    if (f) { fputs("stub object", f); fclose(f); }
}
int main(int argc, char** argv) {
    const char* log = getenv("P5_STUB_LOG");
    FILE* f = log ? fopen(log, "a") : NULL;
    if (f) {
        fputs(ROLE, f);
        for (int i = 1; i < argc; ++i) { fputc(' ', f); fputs(argv[i], f); }
        fputc('\n', f);
        fclose(f);
    }
#ifdef ROLE_CL
    const char* fail = getenv("P5_CL_FAIL");
    const char* fo = NULL;
    for (int i = 1; i < argc; ++i) {
        if (fail && *fail && strstr(argv[i], fail)) { fprintf(stderr, "stub cl: failing on %s\n", argv[i]); return 2; }
        if (strncmp(argv[i], "/Fo", 3) == 0) fo = argv[i] + 3;
    }
    if (fo && *fo) {
        size_t n = strlen(fo);
        if (fo[n - 1] != '\\' && fo[n - 1] != '/') { write_object(fo); return 0; }
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            size_t len = strlen(a);
            if (a[0] == '/' || len < 5 || strcmp(a + len - 4, ".cpp") != 0) continue;
            const char* base = a;
            for (const char* p = a; *p; ++p) if (*p == '\\' || *p == '/') base = p + 1;
            char path[4096];
            snprintf(path, sizeof path, "%s%.*s.obj", fo, (int)(strlen(base) - 4), base);
            write_object(path);
        }
    }
#endif
    return 0;
}
"""
ROLES = {"cl": "ROLE_CL", "link": "ROLE_LINK", "python": "ROLE_PYTHON"}


def wine():
    for name in ("wine64", "wine"):
        found = shutil.which(name)
        if found:
            return found
    return "/usr/lib/wine/wine64" if Path("/usr/lib/wine/wine64").is_file() else None


def win_path(path):
    """`path` as the script's cmd sees it: itself on Windows, through Wine's Z: drive elsewhere."""
    return str(path) if os.name == "nt" else "Z:" + str(path).replace("/", "\\")


class BuildRedSuiteGpuP5(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t2899-gpu-p5-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.stubs = root / "stubs"
        self.stubs.mkdir()
        self.build_stubs()
        self.suite = root / "tests" / SUITE
        self.suite.mkdir(parents=True)
        text = SCRIPT.read_bytes()
        self.assertEqual(text.count(VSDEVCMD_LINE), 1, "the script's VsDevCmd line moved; update this harness")
        text = text.replace(VSDEVCMD_LINE, b"rem P-5 harness: " + VSDEVCMD_LINE)
        if os.name != "nt":
            # Wine's cmd runs the right-hand side of `A || (B)` whatever A exits with (A's errorlevel itself is
            # right), so under Wine each of the script's `A || (B)` becomes `A` then `if errorlevel 1 (B)`, which
            # native cmd runs the same way. Every `||` in the script is that form; Windows runs the script as it is.
            self.assertEqual(text.count(b"||"), text.count(b" || ("), "a `||` of another form; update this harness")
            text = text.replace(b" || (", b"\r\nif errorlevel 1 (")
        (self.suite / SCRIPT.name).write_bytes(text)
        self.shaders = root / "shaders"
        self.shaders.mkdir()
        (self.shaders / "stub.cso").write_bytes(b"cso")
        self.log = root / "stub.log"
        self.obj = self.suite / "obj_gpu"
        (self.obj / "gpu_fixed_noslm5").mkdir(parents=True)
        for name in ("gpu_1p0_fixed.obj", "superslm_gpu_pristine.obj"):
            (self.obj / "gpu_fixed_noslm5" / name).write_bytes(STALE)
        (self.obj / "slm4_blob.bin").write_bytes(STALE)
        (self.obj / "slm4_dump.obj").write_bytes(STALE)

    def build_stubs(self):
        source = self.stubs / "stub.c"
        source.write_text(STUB_C, encoding="ascii")
        if os.name == "nt":
            if not VCVARS.is_file():
                self.fail(f"{VCVARS} is missing: the stubs need MSVC on Windows")
            lines = ["@echo off", f'call "{VCVARS}" >nul 2>&1']
            for name, role in ROLES.items():
                lines.append(f'cl /nologo /D{role} /DROLE="\\"{name}\\"" "{source}" /Fe:"{self.stubs / name}.exe" '
                             f'/Fo:"{self.stubs / name}.obj" || exit /b 1')
            (self.stubs / "build.bat").write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
            built = subprocess.run(["cmd", "/c", str(self.stubs / "build.bat")], capture_output=True, text=True,
                                   timeout=300)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            return
        gcc = shutil.which("x86_64-w64-mingw32-gcc")
        if not gcc or not wine():
            self.fail("off Windows the stubs need x86_64-w64-mingw32-gcc and Wine to run the script")
        for name, role in ROLES.items():
            subprocess.run([gcc, "-O2", "-static", f"-D{role}", f'-DROLE="{name}"', str(source), "-o",
                            str(self.stubs / f"{name}.exe")], check=True, timeout=300)

    def run_script(self, cl_fail=""):
        env = os.environ.copy()
        env["P5_STUB_LOG"] = win_path(self.log)
        env["P5_CL_FAIL"] = cl_fail
        env["SSLM_PYTHON"] = win_path(self.stubs / "python.exe")
        args = [win_path(self.suite / SCRIPT.name), "model.sslm", "g5.sslm", win_path(self.shaders)]
        if os.name == "nt":
            env["PATH"] = str(self.stubs) + os.pathsep + env["PATH"]
            command = ["cmd", "/c"] + args
        else:
            env["WINEPATH"] = win_path(self.stubs)
            env.setdefault("WINEDEBUG", "-all")
            command = [wine(), "cmd", "/c"] + args
        result = subprocess.run(command, env=env, capture_output=True, timeout=900)
        output = result.stdout.decode("ascii", "replace") + result.stderr.decode("ascii", "replace")
        calls = self.log.read_text(encoding="ascii", errors="replace").splitlines() if self.log.exists() else []
        return result.returncode, output, calls

    def test_failed_fixed_compile_skips_the_slm4_dump_link_and_leaves_no_stale_output(self):
        code, output, calls = self.run_script(cl_fail="gpu_1p0_fixed.cpp")
        self.assertTrue([c for c in calls if c.startswith("cl ") and "gpu_1p0_fixed.cpp" in c],
                        "the gpu_fixed_noslm5 compile never ran:\n" + output)
        self.assertIn("BUILD FAILED: gpu_fixed_noslm5", output)
        self.assertEqual(code, 1, output)
        self.assertFalse([c for c in calls if c.startswith("link ") and "slm4_dump" in c],
                         "the slm4_dump link ran after its objects' compile failed")
        self.assertFalse(list((self.obj / "gpu_fixed_noslm5").glob("*.obj")) if (self.obj / "gpu_fixed_noslm5").exists()
                         else [], "an earlier run's gpu_fixed_noslm5 objects survived the failed compile")
        self.assertFalse((self.obj / "slm4_blob.bin").exists(), "an earlier run's slm4_blob.bin survived")

    def test_failed_slm4_dump_compile_leaves_no_stale_object(self):
        code, output, calls = self.run_script(cl_fail="cell_gpu_slm4_dump.cpp")
        self.assertTrue([c for c in calls if c.startswith("cl ") and "cell_gpu_slm4_dump.cpp" in c],
                        "the slm4_dump compile never ran:\n" + output)
        self.assertIn("BUILD FAILED: slm4_dump.obj", output)
        self.assertEqual(code, 1, output)
        self.assertFalse((self.obj / "slm4_dump.obj").exists(),
                         "an earlier run's slm4_dump.obj survived the failed compile, for the slm4_dump link to use")

    def test_control_links_slm4_dump_against_this_runs_objects(self):
        _, output, calls = self.run_script()
        self.assertNotIn("BUILD FAILED: gpu_fixed_noslm5", output)
        links = [c for c in calls if c.startswith("link ") and "slm4_dump.exe" in c]
        self.assertEqual(len(links), 1, "\n".join(calls))
        self.assertIn("gpu_fixed_noslm5\\gpu_1p0_fixed.obj", links[0])
        self.assertEqual((self.obj / "slm4_dump.obj").read_bytes(), b"stub object", "slm4_dump.obj is not this run's")
        for name in ("gpu_1p0_fixed.obj", "superslm_gpu_pristine.obj"):
            self.assertEqual((self.obj / "gpu_fixed_noslm5" / name).read_bytes(), b"stub object", name)
        self.assertFalse((self.obj / "slm4_blob.bin").exists(), "an earlier run's slm4_blob.bin survived")


if __name__ == "__main__":
    unittest.main()
