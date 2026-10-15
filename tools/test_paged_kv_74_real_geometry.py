"""Permanent P-5 pin for the real-artifact 7.4 geometry comparator."""

import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest
import hashlib
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get(
    "PKV_C6_SOURCE", ROOT / "tests/paged-kv/c6_dim7_reset_adopt_timing.cpp"))
ARTIFACTS = Path(os.environ.get(
    "PKV_C6_ARTIFACT_DIR", "D:/_artifacts/SuperSLM/t3114"))
# Required, with no default (the 1.12.0 review M-2): a default naming an old scratch build ran the cell against that
# stale build whenever it was present. Unset or missing fails the cell, never skips it, as .github/workflows/tests.yml
# says of the paged-KV tool tests it leaves to the build box.
BUILD_DIR = os.environ.get("PKV_C6_BUILD", "")
BUILD = Path(BUILD_DIR)
VCVARS = Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat")


def config(path):
    # Read CFG1 from the real artifact's section table, without loading weights.
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:4] != b"SSLM":
            raise AssertionError(f"{path}: bad artifact magic")
        count = struct.unpack_from("<I", header, 12)[0]
        for _ in range(count):
            row = stream.read(40)
            if struct.unpack_from("<I", row)[0] == 0:
                offset = struct.unpack_from("<Q", row, 8)[0]
                position = stream.tell()
                stream.seek(offset)
                cfg = stream.read(84)
                stream.seek(position)
                if cfg[:4] != b"CFG1":
                    raise AssertionError(f"{path}: bad CFG1 magic")
                values = struct.unpack_from("<16I", cfg, 4)
                return (values[2], values[4], values[5], values[1], values[11], values[8])
    raise AssertionError(f"{path}: no CFG1")


class RealGeometryCells(unittest.TestCase):
    def test_p5_real_artifact_block_mismatch_refuses_before_timer(self):
        if not BUILD_DIR:
            self.fail("PKV_C6_BUILD is unset: name the paged-KV C6 build's directory; there is no default")
        if not (BUILD / "superslm_pkv_c6.exe").is_file():
            self.fail(f"PKV_C6_BUILD names {BUILD_DIR}, which holds no superslm_pkv_c6.exe")
        small_name = "qwen2.5-0.5b-instruct-cap4096-aex.sslm"
        large_name = "qwen2.5-0.5b-instruct-cap32768-aex.sslm"
        with tempfile.TemporaryDirectory(prefix="pkv-p5-real-", dir="D:/_scratch/T-3112") as temp:
            folder = Path(temp)
            os.link(ARTIFACTS / small_name, folder / small_name)
            os.link(ARTIFACTS / large_name, folder / large_name)
            env = os.environ.copy()
            env["SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR"] = str(folder)
            env["SUPERSLM_PAGED_KV_74_FORCE_PIN_FAILURE"] = "1"

            def invoke():
                listing = subprocess.run(["tasklist", "/FI", "IMAGENAME eq smoke_adapter_driver.exe"],
                                         capture_output=True, text=True, timeout=15)
                self.assertNotIn("smoke_adapter_driver.exe", listing.stdout)
                return subprocess.run([str(BUILD / "superslm_pkv_c6.exe"), "=7.4/C6"],
                                      env=env, capture_output=True, text=True, timeout=90)

            control = invoke()
            self.assertEqual(control.returncode, 73, control.stdout + control.stderr)
            self.assertIn("cannot pin timing thread", control.stderr)

            (folder / large_name).unlink()  # break the hard link before modifying CFG1
            shutil.copyfile(ARTIFACTS / large_name, folder / large_name)
            with (folder / large_name).open("r+b") as stream:
                header = stream.read(104)
                config_offset = struct.unpack_from("<Q", header, 72)[0]
                stream.seek(config_offset + 48)  # CFG1 kv_block_size, after magic + 11 u32s
                old = struct.unpack("<I", stream.read(4))[0]
                self.assertEqual(old, 16)
                stream.seek(config_offset + 48)
                stream.write(struct.pack("<I", 17))
                stream.seek(32)
                stream.write(bytes(32))
                stream.seek(0)
                digest = hashlib.sha256()
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
                stream.seek(32)
                stream.write(digest.digest())

            refused = invoke()
            self.assertEqual(refused.returncode, 71, refused.stdout + refused.stderr)
            self.assertIn("geometry mismatch", refused.stderr)
            self.assertNotIn("cannot pin timing thread", refused.stderr)

    def test_p5_hidden_size_and_block_size_from_real_configs(self):
        source = SOURCE.read_text(encoding="utf-8")
        function = re.search(r"bool Same74Geometry\(.*?\n}\n(?=void Run74)", source, re.S)
        self.assertIsNotNone(function, "real-artifact geometry comparator is absent")
        run = source.split("void Run74(const Fixture& low,const Fixture& large){", 1)[1]
        run = run.split("void TimerLive", 1)[0]
        self.assertIn("SslmModel::Load(low.bytes.p", run)
        self.assertIn("SslmModel::Load(large.bytes.p", run)
        self.assertIn("Same74Geometry(small_config,large_config,low.geo,large.geo)", run)

        low = config(ARTIFACTS / "qwen2.5-0.5b-instruct-cap4096-aex.sslm")
        large = config(ARTIFACTS / "qwen2.5-0.5b-instruct-cap32768-aex.sslm")
        self.assertEqual(low[:5], large[:5], "control artifacts have different model geometry")
        self.assertEqual((low[5], large[5]), (4096, 32768))
        cfg = lambda values: "{" + ",".join(map(str, values[:5])) + "}"
        claim = lambda values: "{" + ",".join(map(str, values[:3] + (values[5],))) + "}"
        cpp = ("#include <cstdint>\n#include <iostream>\n"
               "namespace superslm {struct SslmModelConfig {"
               "uint32_t num_hidden_layers,num_key_value_heads,head_dim,hidden_size,kv_block_size;};}\n"
               "struct Geometry {uint32_t layers,kv_heads,head_dim;int64_t context_cap;};\n"
               + function.group() + "\nint main(){\n"
               + f"superslm::SslmModelConfig a={cfg(low)},b={cfg(large)};\n"
               + f"Geometry x={claim(low)},y={claim(large)};\n"
               + "std::cout<<Same74Geometry(a,b,x,y);\n"
               + "++b.hidden_size;std::cout<<Same74Geometry(a,b,x,y);\n"
               + "--b.hidden_size;++b.kv_block_size;std::cout<<Same74Geometry(a,b,x,y);\n}\n")
        with tempfile.TemporaryDirectory(prefix="pkv-p5-") as temp:
            folder = Path(temp)
            (folder / "pin.cpp").write_text(cpp, encoding="utf-8")
            (folder / "build.bat").write_text(
                f'@echo off\ncall "{VCVARS}" >nul 2>&1\n'
                f'cl /nologo /EHsc /Fe"{folder / "pin.exe"}" "{folder / "pin.cpp"}"\n',
                encoding="ascii")
            built = subprocess.run(["cmd", "/c", str(folder / "build.bat")],
                                   cwd=folder, capture_output=True, text=True, timeout=90)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(folder / "pin.exe")], capture_output=True,
                                     text=True, timeout=15)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout, "100", "equal / hidden-only / block-only")


if __name__ == "__main__":
    unittest.main()
