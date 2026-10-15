"""Production-binary lifecycle publication and refusal constructions on tiny fixtures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3112/build-c6"))
FIXTURES = Path(os.environ.get("PKV_C6_FIXTURES", "D:/_scratch/T-3112/fixtures"))


class LifecyclePublication(unittest.TestCase):
    def call(self, binary="superslm_pkv_lifecycle_prod.exe", **settings):
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_FIXTURE_DIR"] = str(FIXTURES)
        for key in ("SUPERSLM_PAGED_KV_LIFE_OUT", "SUPERSLM_PAGED_KV_LIFE_BASELINE",
                    "SUPERSLM_PAGED_KV_LIFE_RESET_MUTANT", "SUPERSLM_PAGED_KV_LIFE_COMMISSIONED",
                    "SUPERSLM_PAGED_KV_LIFE_PUBLICATION_PROBE"):
            env.pop(key, None)
        for key, value in settings.items():
            env["SUPERSLM_PAGED_KV_LIFE_" + key] = str(value)
        return subprocess.run([str(BUILD / binary), "=lifecycle/C6:fixtures"], env=env,
                              capture_output=True, text=True, timeout=180)

    def test_4x_publication_path_and_real_reset_mutant(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "baseline.txt"
            first = self.call(OUT=baseline)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue(baseline.is_file())
            second = self.call(BASELINE=baseline)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("within load envelope; no change claimed", second.stdout)
            self.assertNotIn("published change", second.stdout)
            mutant = self.call(BASELINE=baseline, RESET_MUTANT="1")
            self.assertEqual(mutant.returncode, 0, mutant.stdout + mutant.stderr)
            reset_verdicts = [line for line in mutant.stdout.splitlines()
                              if "reset@" in line and "verdict" in line]
            self.assertTrue(reset_verdicts, mutant.stdout)
            self.assertTrue(all("RESOLVED" in line and "published change" in line
                                for line in reset_verdicts), mutant.stdout)

    def test_injection_binary_refuses_publication_path(self):
        result = self.call(binary="superslm_pkv_c6.exe", PUBLICATION_PROBE="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("publication refused", result.stderr)
        self.assertNotIn("published change", result.stdout)

    def test_truncated_baseline_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "truncated.txt"
            baseline.write_text("pkv_def:create 1000 0.05\ntruncated", encoding="utf-8")
            result = self.call(BASELINE=baseline)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("malformed baseline row", result.stderr)

    def test_lc4_duplicate_baseline_key_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "baseline.txt"
            captured = self.call(OUT=baseline)
            self.assertEqual(captured.returncode, 0, captured.stdout + captured.stderr)
            rows = baseline.read_text(encoding="utf-8").splitlines()
            self.assertTrue(rows)
            baseline.write_text("\n".join([*rows, rows[0]]) + "\n", encoding="utf-8")
            duplicate = self.call(BASELINE=baseline)
            self.assertNotEqual(duplicate.returncode, 0, duplicate.stdout + duplicate.stderr)
            self.assertIn("malformed baseline row", duplicate.stderr)
            self.assertNotIn("published change", duplicate.stdout)

    def test_lc6_all_near_boundary_envelope_arms_execute(self):
        result = subprocess.run([str(BUILD / "superslm_pkv_lifecycle_prod.exe"),
                                 "=lifecycle/C6:constructions"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines()
                 if "lifecycle boundary-no-change:" in line and "resolving power" in line]
        self.assertEqual(len(lines), 20, result.stdout)
        self.assertEqual(sum("paged 3.90" in line for line in lines), 4, result.stdout)
        self.assertEqual(sum("resolving power 100.00%" in line for line in lines), 5,
                         result.stdout)


if __name__ == "__main__":
    unittest.main()
