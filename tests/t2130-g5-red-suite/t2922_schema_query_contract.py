"""T-2933 portable contract guard for plan §3.10.8."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "t2922_schema_query_cells.json"
QUERY_SOURCES = {
    "SslmGpuSeqSchemaAcceptingForG5Bridge": HERE / "t2922_gpu_schema_accepting_red.cpp",
    "SslmGpuSeqSchemaBoundForG5Bridge": HERE / "t2922_gpu_schema_bound_red.cpp",
    "sslm_seq_schema_bound": HERE / "t2922_cpu_schema_bound_red.cpp",
}
SURFACES = {
    "README.md": ("schema accepting", "schema bound"),
    "docs/api.md": ("schema accepting", "schema bound"),
    "include/superslm/gpu_1p0.h": (
        "SslmGpuSeqSchemaAcceptingForG5Bridge",
        "SslmGpuSeqSchemaBoundForG5Bridge",
    ),
    "include/superslm/schema_masks.h": ("accepting_le", "accepting_count"),
    "include/superslm/sslm_abi_functions.inc": ("sslm_seq_schema_bound",),
}


class T2933SchemaQueryContract(unittest.TestCase):
    def test_population_is_named_and_complete(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        ids = [row["id"] for row in data["cells"]]
        self.assertEqual(len(ids), 26)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(data["mutants"]), set("abcdefghijkl"))
        self.assertEqual(data["plan_commit"], "669d2ce24e")
        self.assertEqual(data["fixture"]["acceptance"],
                         {"t2922_accepts_q": 1, "t2922_rejects_q": 0})

    def test_new_apis_are_three_independent_executable_cells(self) -> None:
        for symbol, path in QUERY_SOURCES.items():
            source = path.read_text(encoding="utf-8")
            self.assertIn("int main()", source)
            self.assertIn(symbol, source)
            self.assertIn("unchanged", source)
            for other in QUERY_SOURCES:
                if other != symbol:
                    self.assertNotIn(other, source)

    def test_runner_executes_compilers_and_never_prints_synthetic_cell_results(self) -> None:
        runner = (HERE / "build_t2922_gpu_schema_accepting_red.bat").read_text(
            encoding="utf-8"
        )
        self.assertEqual(runner.lower().count("cl /nologo"), 3)
        self.assertNotIn("for %%C", runner)
        self.assertNotIn("echo CELL", runner)
        for source in QUERY_SOURCES.values():
            self.assertIn(source.name, runner)

    def test_surface_oracle_is_currently_red_for_the_new_contract(self) -> None:
        missing: list[str] = []
        for relative, needles in SURFACES.items():
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for needle in needles:
                if needle.lower() not in text:
                    missing.append(f"{relative}:{needle}")
        expected = {
            "README.md:schema accepting", "README.md:schema bound",
            "docs/api.md:schema accepting", "docs/api.md:schema bound",
            "include/superslm/gpu_1p0.h:SslmGpuSeqSchemaAcceptingForG5Bridge",
            "include/superslm/gpu_1p0.h:SslmGpuSeqSchemaBoundForG5Bridge",
            "include/superslm/sslm_abi_functions.inc:sslm_seq_schema_bound",
        }
        self.assertTrue(expected.issubset(set(missing)), sorted(missing))

    def test_real_model_manifest_is_pinned_and_gate_fails_closed(self) -> None:
        manifest = json.loads((HERE / "t2922_real_model_manifest.json").read_text())
        self.assertEqual(manifest["artifact_sha256"],
                         "696c2a4ac412f1c5866a446eb32aa6500db77e5ae8aff0018438103c607e61bd")
        self.assertEqual(manifest["prompt_ids_sha256"],
                         "207d0c68f9aa4de5bd6446c67df2226a6b582c1bff923138f08c7549d4d9ab81")
        prompt_path = ROOT / manifest["prompt_ids_file"]
        self.assertEqual(hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                         manifest["prompt_ids_sha256"])
        self.assertEqual(manifest["decode_budget"], 300)
        if os.environ.get("SUPERSLM_G5_REAL_MODEL_TESTS"):
            artifact = Path(manifest["artifact"])
            self.assertTrue(artifact.is_file(), f"required real artifact missing: {artifact}")

    def test_mutator_refuses_every_unlanded_future_anchor(self) -> None:
        for mutant in "abcdefhij":
            result = subprocess.run(
                [sys.executable, str(HERE / "make_t2922_gpu_schema_accepting_mutants.py"),
                 "--engine", str(ROOT), "--out", str(HERE / "_mutant_should_not_exist.cpp"),
                 "--mutant", mutant], text=True, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0, mutant)
            self.assertIn("REFUSED", result.stderr, mutant)
        self.assertFalse((HERE / "_mutant_should_not_exist.cpp").exists())

    def test_ctest_and_ci_name_this_contract(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
        self.assertIn("t2933_schema_query_contract", cmake)
        self.assertIn("t2933_schema_query_contract", workflow)


if __name__ == "__main__":
    unittest.main()
