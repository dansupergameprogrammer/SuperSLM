"""79-13: production pair identity binds compile definitions and refuses seams."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3112/build-c6"))
SOURCE = ROOT / "tools/paged_kv_pair.cpp"
VCVARS = Path(os.environ.get(
    "PKV_C6_VCVARS64",
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"))


class PairBuildIdentity(unittest.TestCase):
    def compile(self, directory, definition, link):
        guard = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "if (Get-Process -Name smoke_adapter_driver -ErrorAction SilentlyContinue) { exit 86 }"],
            capture_output=True, text=True)
        self.assertEqual(guard.returncode, 0, "smoke_adapter_driver active")
        obj, exe = directory / "pair.obj", directory / "superslm_pkv_pair.exe"
        batch = directory / "compile-pair.bat"
        batch.write_text(
            f'@echo off\ncall "{VCVARS}" >nul 2>&1\n'
            f'if errorlevel 1 exit /b %errorlevel%\n'
            f'cl /nologo /c /std:c++20 /EHsc /MD /D{definition}=1 '
            f'/DPKV_REFERENCE_DIR=\\"{(ROOT / "tests/paged-kv/reference").as_posix()}\\" '
            f'/I"{ROOT / "include"}" /I"{ROOT / "src"}" '
            f'/I"{ROOT / "tests"}" /I"{ROOT / "tests/paged-kv"}" '
            f'/Fo"{obj}" "{SOURCE}"\n', encoding="utf-8")
        result = subprocess.run(["cmd", "/c", str(batch)],
                                capture_output=True, text=True, timeout=90)
        if not link:
            return result, exe
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        batch.write_text(
            f'@echo off\ncall "{VCVARS}" >nul 2>&1\n'
            f'if errorlevel 1 exit /b %errorlevel%\n'
            f'link /nologo /machine:x64 /subsystem:console /OUT:"{exe}" '
            f'"{obj}" "{BUILD / "superslm.lib"}" kernel32.lib user32.lib '
            f'gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib '
            f'uuid.lib comdlg32.lib advapi32.lib\n', encoding="utf-8")
        guard = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "if (Get-Process -Name smoke_adapter_driver -ErrorAction SilentlyContinue) { exit 86 }"],
            capture_output=True, text=True)
        self.assertEqual(guard.returncode, 0, "smoke_adapter_driver active")
        linked = subprocess.run(["cmd", "/c", str(batch)],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
        return result, exe

    def test_79_13_identity_changes_with_definition(self):
        base = subprocess.run([str(BUILD / "superslm_pkv_pair.exe"), "identity"],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(base.returncode, 0, base.stdout + base.stderr)
        with tempfile.TemporaryDirectory(prefix="pkv-pair-def-") as temp:
            _, changed = self.compile(Path(temp), "PKV_C6_IDENTITY_PROBE", link=True)
            probe = subprocess.run([str(changed), "identity"],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
            self.assertIn("defs=", base.stdout)
            self.assertIn("defs=", probe.stdout)
            self.assertNotEqual(base.stdout.strip(), probe.stdout.strip())

    def test_79_13_seam_definition_refuses_compile(self):
        with tempfile.TemporaryDirectory(prefix="pkv-pair-seam-") as temp:
            result, exe = self.compile(Path(temp), "SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING", link=False)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("A C6 production pair must not carry a seam or injection definition",
                          result.stdout + result.stderr)
            self.assertFalse(exe.exists())


if __name__ == "__main__":
    unittest.main()
