"""508-pair mechanism cells through the production executable on tiny artifacts.

The output is deliberately MECHANISM_ONLY. Real-model C6 box timing and
commissioning are separate gates.
"""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_paged_kv_pairs as pair


BUILD = Path(os.environ.get("PKV_C6_BUILD", "D:/_scratch/T-3112/build-c6"))
FIXTURES = Path(os.environ.get("PKV_C6_FIXTURES", "D:/_scratch/T-3112/fixtures"))
BLOBS = Path(os.environ.get("PKV_C6_BLOBS", "D:/_scratch/T-3112"))


class ProductionPairs(unittest.TestCase):
    def test_79_and_79b_grade_measured_distributions(self):
        with tempfile.TemporaryDirectory(prefix="pkv-pair-mechanism-") as temp:
            root = Path(temp)
            for kind in ("79", "79b"):
                with self.subTest(kind=kind):
                    args = SimpleNamespace(
                        exe=BUILD / "superslm_pkv_pair.exe", kind=kind,
                        model_a=FIXTURES / "pkv_def.sslm",
                        model_b=FIXTURES / "pkv_odd.sslm" if kind == "79" else None,
                        manifest_a=BLOBS / "blob-paged" / "manifest.json" if kind == "79" else None,
                        manifest_b=BLOBS / "blob-one" / "manifest.json" if kind == "79" else None,
                        result=root / f"{kind}.json", cap_a=4096,
                        cap_b=4100 if kind == "79" else None,
                        layers=2, heads=2, dim=48, width=530 if kind == "79" else 1712,
                        pairs=508, seed=7904 if kind == "79" else 7905,
                        cap_seconds=300, mechanism=True, load_record=None)
                    with patch.dict(os.environ, {"SUPERSLM_PAGED_KV_79_SLOW_ARM": ""}):
                        record = pair.run(args)
                    self.assertEqual((record["status"], record["pair_count"]), ("MECHANISM_ONLY", 508))
                    values, token_sha = pair.parse_pairs(args.result.with_suffix(".pairs.csv"), 508)
                    self.assertEqual(record["token_sha256"], token_sha)
                    self.assertGreater(len({round(value, 5) for value in values}), 20)
                    ordered = sorted(values)
                    self.assertEqual(record["sign_interval"], [ordered[228], ordered[279]])
                    self.assertEqual(pair.read_result(args.result, {
                        "kind": kind, "exe_sha256": record["exe_sha256"],
                        "library_sha256": record["library_sha256"],
                        "model_hashes": record["model_hashes"],
                        "manifest_hashes": record["manifest_hashes"],
                        "pair_count": 508, "seed": record["seed"], "hook": "",
                        "load_sha256": None, "status": "MECHANISM_ONLY"})["status"], "MECHANISM_ONLY")


if __name__ == "__main__":
    unittest.main()
