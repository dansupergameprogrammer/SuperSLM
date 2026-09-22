"""Structural contract for the T-2932 red suite and its CI/CTest wiring."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class T2922SchemaQueryRedContract(unittest.TestCase):
    def test_cell_and_mutant_population_is_complete(self) -> None:
        manifest = json.loads((HERE / "t2922_schema_query_cells.json").read_text())
        self.assertEqual(len(manifest["cells"]), 24)
        self.assertEqual(set(manifest["mutants"]), set("abcdefghijkl"))
        self.assertEqual(len(set(manifest["cells"])), len(manifest["cells"]))

    def test_red_witness_names_all_three_new_public_symbols(self) -> None:
        source = (HERE / "t2922_gpu_schema_accepting_red.cpp").read_text()
        for symbol in ("SslmGpuSeqSchemaAcceptingForG5Bridge",
                       "SslmGpuSeqSchemaBoundForG5Bridge", "sslm_seq_schema_bound"):
            self.assertIn(symbol, source)

    def test_mutator_refuses_source_drift(self) -> None:
        result = subprocess.run(
            [sys.executable, str(HERE / "make_t2922_gpu_schema_accepting_mutants.py"),
             "--source", str(HERE / "t2922_gpu_schema_accepting_red.cpp"),
             "--out", str(HERE / "_t2922_should_not_exist.cpp"), "--mutant", "k"],
            text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REFUSED", result.stderr)
        self.assertFalse((HERE / "_t2922_should_not_exist.cpp").exists())


if __name__ == "__main__":
    unittest.main()
