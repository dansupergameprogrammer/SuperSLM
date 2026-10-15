"""The production pair executable accepts the wrapper's full 32-bit seed range."""

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3114/build-c6-opt"))
FIXTURES = Path(os.environ.get("PKV_C6_FIXTURES", "D:/_scratch/T-3112/fixtures"))


class PairSeed(unittest.TestCase):
    def test_unsigned_32bit_seed_boundary(self):
        model = FIXTURES / "pkv_def.sslm"
        self.assertTrue(model.is_file())
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR"] = str(FIXTURES)
        with tempfile.TemporaryDirectory(prefix="pkv-seed-") as folder:
            for seed, expected in ((2147483648, 0), (4294967295, 0),
                                   (4294967296, 40), (-1, 40)):
                with self.subTest(seed=seed):
                    output = Path(folder) / f"{seed}.csv"
                    if os.name == "nt":
                        while True:
                            listing = subprocess.run([
                                "C:/Windows/System32/tasklist.exe", "/FI",
                                "IMAGENAME eq smoke_adapter_driver.exe", "/NH"],
                                capture_output=True, text=True, timeout=15)
                            self.assertEqual(listing.returncode, 0, listing.stderr)
                            if "smoke_adapter_driver.exe" not in listing.stdout.lower():
                                break
                            print("waiting for smoke_adapter_driver.exe before seed control", flush=True)
                            time.sleep(30)
                    result = subprocess.run([
                        str(BUILD / "superslm_pkv_pair.exe"), "pair79b", model.name,
                        "4096", "2", "2", "48", "1", str(seed), str(output)],
                        env=env, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, expected,
                                     result.stdout + result.stderr)
                    self.assertEqual(output.is_file(), expected == 0)


if __name__ == "__main__":
    unittest.main()
