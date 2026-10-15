"""The production pair helper waits for the other box owner's smoke process."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


script = Path(__file__).with_name("run_paged_kv_pairs.py")
spec = importlib.util.spec_from_file_location("pair_box_rule", script)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class BoxRule(unittest.TestCase):
    def test_active_smoke_blocks_a_piece_until_absent(self):
        if pair.os.name != "nt":
            self.skipTest("Windows process owner rule")
        checks = []

        def tasklist(*args, **kwargs):
            checks.append(True)
            return SimpleNamespace(returncode=0, stdout=(
                "smoke_adapter_driver.exe 42 Console 1 1,000 K"
                if len(checks) == 1 else "INFO: No tasks are running"))

        def spawn(*args, **kwargs):
            self.assertEqual(len(checks), 2, "piece started while smoke was active")
            return SimpleNamespace(wait=lambda timeout: 0)

        with tempfile.TemporaryDirectory(prefix="pkv-box-rule-") as folder, \
             patch.object(pair.subprocess, "run", side_effect=tasklist), \
             patch.object(pair.subprocess, "Popen", side_effect=spawn), \
             patch.object(pair.time, "sleep"):
            self.assertEqual(pair.run_logged(["fixture"], Path(folder) / "piece.log", 1, {}), 0)
            self.assertEqual(len(checks), 2)


if __name__ == "__main__":
    unittest.main()
