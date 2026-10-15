"""C-6: successor registration directions and complete 7.4 hook output."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3114/build-c6-opt"))


class ConfirmationDocs(unittest.TestCase):
    def test_registration_and_current_source_comments(self):
        readme = (ROOT / "tests/paged-kv/commissioning/README.md").read_text(encoding="utf-8")
        fields = readme.split("A successor receipt entry must include `source_binding` with ", 1)[1].split(". ", 1)[0]
        self.assertIn("`load_probe_sha256`", fields)
        self.assertIn("-Register", readme)
        self.assertIn("-SourceBindingJson PATH", readme)
        common = (ROOT / "tests/paged-kv/pkv_common.h").read_text(encoding="utf-8")
        lifecycle = (ROOT / "tests/paged-kv/c6_lifecycle_timing.cpp").read_text(encoding="utf-8")
        self.assertNotIn("SUPERSLM_PAGED_KV_TIMING_COMMISSIONED", common)
        self.assertNotIn("c6_dim7_throughput.cpp", lifecycle)

    def test_both_mutants_and_other_hooks_print_before_timing(self):
        env = os.environ.copy()
        env.update({"SUPERSLM_PAGED_KV_74_ADOPT_MUTANT": "1",
                    "SUPERSLM_PAGED_KV_74_RESET_MUTANT": "1",
                    "SUPERSLM_PAGED_KV_74_FLOOR_CONTROL_TICKS": "101"})
        result = subprocess.run([str(BUILD / "superslm_pkv_c6.exe"), "=7.4t/C6:hooks"],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("7.4 hook_state=adopt=1;reset=1;", result.stdout)
        self.assertIn("SUPERSLM_PAGED_KV_74_FLOOR_CONTROL_TICKS=101", result.stdout)
        self.assertIn("publication=quarantined", result.stdout)


if __name__ == "__main__":
    unittest.main()
