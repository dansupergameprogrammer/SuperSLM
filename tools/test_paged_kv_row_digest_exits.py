"""RD-4: executable row-digest exits at both reference record points."""

import os
from pathlib import Path
import subprocess
import unittest


BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3112/build-c6"))
FIXTURES = Path(os.environ.get("PKV_C6_FIXTURES", "D:/_scratch/T-3112/fixtures"))


class RowDigestExits(unittest.TestCase):
    def run_cell(self, cell, flip):
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_FIXTURE_DIR"] = str(FIXTURES)
        for key in ("SUPERSLM_PAGED_KV_COHORT_CACHE_DIR",
                    "SUPERSLM_PAGED_KV_COHORT_REQUIRE_CACHE",
                    "SUPERSLM_PAGED_KV_COHORT_SHARD"):
            env.pop(key, None)
        if flip:
            env["SUPERSLM_PAGED_KV_ROW_DIGEST_FLIP"] = flip
        else:
            env.pop("SUPERSLM_PAGED_KV_ROW_DIGEST_FLIP", None)
        return subprocess.run([str(BUILD / "superslm_pkv_c6.exe"), "=" + cell],
                              env=env, capture_output=True, text=True, timeout=180)

    def check_fixture(self, cell, expected_failures):
        healthy = self.run_cell(cell, None)
        self.assertEqual(healthy.returncode, 0, healthy.stdout + healthy.stderr)
        self.assertIn("1 cells, 0 red", healthy.stdout)
        for point in ("mid", "limit"):
            with self.subTest(cell=cell, point=point):
                changed = self.run_cell(cell, point)
                self.assertEqual(changed.returncode, 1, changed.stdout + changed.stderr)
                self.assertIn("K/V rows differ from both reference record points", changed.stderr)
                self.assertIn(f"1 cells, 1 red;", changed.stdout)
                self.assertIn(f"{expected_failures} failures", changed.stdout)

    def test_rd4_pkv_def_mid_and_limit(self):
        self.check_fixture("10.1/C6:pkv_def", 50)

    def test_rd4_pkv_32k_mid_and_limit(self):
        self.check_fixture("10.2/C6:pkv_32k", 8)


if __name__ == "__main__":
    unittest.main()
