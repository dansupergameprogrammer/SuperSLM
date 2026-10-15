"""RD-3: the cache identity is the hash of the executable actually invoked."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3112/build-c6"))
FIXTURES = Path(os.environ.get("PKV_C6_FIXTURES", "D:/_scratch/T-3112/fixtures"))


class CacheExecutableIdentity(unittest.TestCase):
    def run_cache_cell(self, binary):
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_FIXTURE_DIR"] = str(FIXTURES)
        result = subprocess.run([str(binary), "=cohort/C6:cache-version"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        prefix = "cohort executable sha256="
        hashes = [line.removeprefix(prefix) for line in result.stdout.splitlines()
                  if line.startswith(prefix)]
        self.assertEqual(len(hashes), 1, result.stdout)
        self.assertEqual(hashes[0], hashlib.sha256(binary.read_bytes()).hexdigest())
        return hashes[0]

    def test_rd3_changed_executable_bytes_change_cache_identity(self):
        source = BUILD / "superslm_pkv_c6.exe"
        original = self.run_cache_cell(source)
        with tempfile.TemporaryDirectory(prefix="pkv-cache-exe-") as temp:
            changed = Path(temp) / source.name
            shutil.copyfile(source, changed)
            with changed.open("ab") as output:
                output.write(b"T-3112-RD3-executable-overlay")
            mutated = self.run_cache_cell(changed)
            self.assertNotEqual(mutated, original)


if __name__ == "__main__":
    unittest.main()
