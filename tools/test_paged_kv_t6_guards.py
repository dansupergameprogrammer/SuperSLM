"""Cells for two T-3202 remedies the 7.9 driver's cells cannot see (SuperSLM 1478be7; the test author's record T-3189
§12). Each runs the real toolchain, so it needs MSVC (vcvars64) and, for F-11, a built pair executable and the
generated fixtures, as tools/test_paged_kv_pair_identity.py and test_paged_kv_74_refusals.py do.

  F-11  superslm_pkv_pair's row79 prints the paged arm's reserve as its second stdout line,
        `RESERVE seed=<kScatterReserveSeed> singles=<P - 1> source=tests/paged-kv/pkv_scatter_reserve.h`, and the
        driver's parse_reserve reads it (pkv_def: B 128, P 32, so 31 singles)
  F-4   the MR-OPT mutant definition is refused outside its target: at compile time by the #error in
        include/superslm/forward_sites.h and src/sslm_abi.cpp unless SUPERSLM_MUTANT_PKV_MR_OPT_TARGET is also defined
        (a definition CMake cannot see, such as the CL variable); and at configure time when CMAKE_CXX_FLAGS or a
        configuration's flags carry it

Environment: PKV_C6_BUILD (the build's Release directory, holding superslm_pkv_pair.exe), PKV_C6_FIXTURES (the
generated fixtures), PKV_C6_VCVARS64 (vcvars64.bat), PKV_T6_ROOT (the source tree under test; default this checkout,
the test author points it at a reverted copy to show each cell fails).
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get("PKV_T6_ROOT") or Path(__file__).resolve().parents[1])
BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3112/build-c6"))
FIXTURES = Path(os.environ.get("PKV_C6_FIXTURES", "D:/_scratch/T-3112/fixtures"))
VCVARS = Path(os.environ.get(
    "PKV_C6_VCVARS64", "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"))
GUARD_TEXT = "only the superslm_mr_opt target builds it"

_spec = importlib.util.spec_from_file_location("rows_driver", Path(__file__).with_name("run_paged_kv_rows.py"))
rows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rows)


def vs(directory, body, timeout=900):
    """Run `body` (batch lines) after vcvars64, in `directory`; returns the CompletedProcess."""
    batch = Path(directory) / "run.bat"
    batch.write_text(f'@echo off\ncall "{VCVARS}" >nul 2>&1\nif errorlevel 1 exit /b %errorlevel%\n{body}\n',
                     encoding="utf-8")
    return subprocess.run(["cmd", "/c", str(batch)], capture_output=True, text=True, timeout=timeout, cwd=directory)


class Reserve(unittest.TestCase):
    def test_f11_row79_prints_the_reserve_line(self):
        pair = BUILD / "superslm_pkv_pair.exe"
        model = FIXTURES / "pkv_def.sslm"
        self.assertTrue(pair.is_file() and model.is_file(), f"{pair}, {model}")
        env = dict(os.environ, SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR=str(FIXTURES))
        with tempfile.TemporaryDirectory(prefix="t3189-f11-") as temp:
            blob = Path(temp) / "r300.blob"
            made = subprocess.run([str(pair), "prefill", "pkv_def.sslm", "4096", "2", "2", "48", "812", "0", "300", "-",
                                   str(blob)], capture_output=True, text=True, timeout=300, env=env)
            self.assertEqual(made.returncode, 0, made.stdout + made.stderr)
            run = subprocess.run([str(pair), "row79", "4096", "2", "2", "48", "79", "--blob", str(blob), "--model",
                                  str(model), "--process", "1", "--pairs", "3", "--warmup", "1"],
                                 capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        lines = run.stdout.splitlines()
        self.assertEqual(lines[0], "PKV_PAIR_PRODUCTION_V2 defs=none")
        self.assertEqual(lines[1], "RESERVE seed=1 singles=31 source=tests/paged-kv/pkv_scatter_reserve.h")
        self.assertEqual(rows.parse_reserve(run.stdout),
                         {"seed": 1, "singles": 31, "source": "tests/paged-kv/pkv_scatter_reserve.h"})


# T-3206 N-3 (D-SLM8466 (2), the C++ half): each real pair build prints its construction's identity line, written here
# literally, and the driver admits it under its own --construction only. Needs a PKV_C6_BUILD from 4156b30 or later.
PAIR_IDENTITIES = {"superslm_pkv_pair.exe": ("production", "PKV_PAIR_PRODUCTION_V2 defs=none"),
                   "superslm_pkv_pair_mr_opt.exe": ("7.9-MR-OPT", "PKV_PAIR_MR_OPT_V1 defs=none")}


class PairIdentity(unittest.TestCase):
    def run_pair(self, exe, *args, env=None):
        done = subprocess.run([str(BUILD / exe), *args], capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout

    def test_n3_each_pair_build_prints_its_constructions_identity(self):
        for exe, (construction, line) in PAIR_IDENTITIES.items():
            with self.subTest(exe=exe):
                self.assertTrue((BUILD / exe).is_file(), f"{BUILD / exe} missing (PKV_C6_BUILD, 4156b30 or later)")
                self.assertEqual(self.run_pair(exe, "identity").splitlines()[0], line)
                for other, (other_construction, _) in PAIR_IDENTITIES.items():
                    if other != exe:
                        with self.assertRaises(rows.Refusal):
                            rows.check_build_identity(line, other_construction)
                self.assertIsNone(rows.check_build_identity(line, construction))

    def test_n3_mr_opt_row79_prints_its_identity_first(self):
        model = FIXTURES / "pkv_def.sslm"
        env = dict(os.environ, SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR=str(FIXTURES))
        with tempfile.TemporaryDirectory(prefix="t3189-n3-") as temp:
            blob = Path(temp) / "r300.blob"
            self.run_pair("superslm_pkv_pair.exe", "prefill", "pkv_def.sslm", "4096", "2", "2", "48", "812", "0", "300",
                          "-", str(blob), env=env)
            out = self.run_pair("superslm_pkv_pair_mr_opt.exe", "row79", "4096", "2", "2", "48", "79", "--blob",
                                str(blob), "--model", str(model), "--process", "1", "--pairs", "3", "--warmup", "1",
                                env=env)
        self.assertEqual(out.splitlines()[0], "PKV_PAIR_MR_OPT_V1 defs=none")
        identity, pairs, ids = rows.parse_process_output(out, "7.9-MR-OPT")
        self.assertEqual((identity, len(pairs)), ("PKV_PAIR_MR_OPT_V1 defs=none", 3))
        self.assertTrue(ids)
        with self.assertRaises(rows.Refusal):
            rows.parse_process_output(out, "production")


class MrOptGuard(unittest.TestCase):
    def test_f4_compile_time_error_without_the_target_acknowledgement(self):
        with tempfile.TemporaryDirectory(prefix="t3189-f4c-") as temp:
            tu = Path(temp) / "tu.cpp"
            tu.write_text('#include "superslm/forward_sites.h"\nint main() { return 0; }\n')
            inc = f'/I"{ROOT / "include"}" /I"{ROOT / "src"}"'
            bad = vs(temp, f'cl /nologo /c /std:c++20 /EHsc {inc} /DSUPERSLM_MUTANT_PKV_MR_OPT "{tu}"')
            self.assertNotEqual(bad.returncode, 0, bad.stdout + bad.stderr)
            self.assertIn(GUARD_TEXT, bad.stdout + bad.stderr)
            good = vs(temp, f'cl /nologo /c /std:c++20 /EHsc {inc} /DSUPERSLM_MUTANT_PKV_MR_OPT '
                            f'/DSUPERSLM_MUTANT_PKV_MR_OPT_TARGET "{tu}"')
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            abi = vs(temp, f'cl /nologo /c /std:c++20 /EHsc {inc} /DSUPERSLM_MUTANT_PKV_MR_OPT '
                           f'/Fo"{Path(temp) / "abi.obj"}" "{ROOT / "src" / "sslm_abi.cpp"}"')
            self.assertNotEqual(abi.returncode, 0, abi.stdout[-2000:] + abi.stderr[-2000:])
            self.assertIn(GUARD_TEXT, abi.stdout + abi.stderr)

    def test_f4_configure_time_refusal(self):
        for flag in ("CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory(prefix="t3189-f4k-") as temp:
                done = vs(temp, f'cmake -S "{ROOT}" -B "{Path(temp) / "b"}" -G "Visual Studio 17 2022" -A x64 '
                                f'"-D{flag}=/DSUPERSLM_MUTANT_PKV_MR_OPT=1"', timeout=1200)
                self.assertNotEqual(done.returncode, 0, done.stdout[-3000:])
                self.assertIn("a SUPERSLM_MUTANT_* definition reaches the production superslm library",
                              done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
