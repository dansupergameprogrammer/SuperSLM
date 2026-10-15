"""LC-7: each forbidden definition must refuse the production lifecycle TU."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("PKV_C6_LIFE_SOURCE",
                             ROOT / "tests/paged-kv/c6_lifecycle_timing.cpp"))
VCVARS = Path(os.environ.get(
    "PKV_C6_VCVARS64",
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"))
FORBIDDEN = (
    "SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS",
    "SUPERSLM_ENABLE_BAD_ALLOC_INJECTION",
    "SUPERSLM_ENABLE_MATVEC_TEST_SEAMS",
    "SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT",
    "SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT",
    "SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING",
    "SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION",
    "SUPERSLM_GPU_ALLOC_FAULT_INJECTION",
    "SUPERSLM_FORCE_SCALAR_MATMUL",
)


class LifecycleProductionGuards(unittest.TestCase):
    def test_lc7_each_definition_refuses_at_compilation(self):
        self.assertTrue(SOURCE.is_file(), SOURCE)
        self.assertTrue(VCVARS.is_file(), VCVARS)
        with tempfile.TemporaryDirectory(prefix="pkv-lifecycle-guards-") as temp:
            for definition in FORBIDDEN:
                with self.subTest(definition=definition):
                    guard = subprocess.run(
                        ["powershell", "-NoProfile", "-Command",
                         "if (Get-Process -Name smoke_adapter_driver -ErrorAction SilentlyContinue) { exit 86 }"],
                        capture_output=True, text=True)
                    self.assertEqual(guard.returncode, 0, "smoke_adapter_driver active")
                    obj = Path(temp) / f"{definition}.obj"
                    batch = Path(temp) / f"{definition}.bat"
                    batch.write_text(
                        f'@echo off\ncall "{VCVARS}" >nul 2>&1\n'
                        f'if errorlevel 1 exit /b %errorlevel%\n'
                        f'cl /nologo /c /std:c++20 /EHsc /MD '
                        f'/DPKV_LIFECYCLE_PRODUCTION_LEG=1 /D{definition}=1 '
                        f'/DPKV_REFERENCE_DIR=\\"{(ROOT / "tests/paged-kv/reference").as_posix()}\\" '
                        f'/I"{ROOT / "include"}" /I"{ROOT / "src"}" '
                        f'/I"{ROOT / "tests"}" /I"{ROOT / "tests/paged-kv"}" '
                        f'/Fo"{obj}" "{SOURCE}"\n', encoding="utf-8")
                    result = subprocess.run(["cmd", "/c", str(batch)],
                                            capture_output=True, text=True, timeout=90)
                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn("A paged lifecycle production leg with a seam or injection definition cannot publish a result", output)
                    self.assertFalse(obj.exists(), output)


if __name__ == "__main__":
    unittest.main()
