"""T-2933 portable contract guard for plan §3.10.8."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
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
    "README.md": ("schema_accepting", "schema_bound"),
    "docs/api.md": ("schema_accepting", "schema_bound"),
    "include/superslm/gpu_1p0.h": (
        "SslmGpuSeqSchemaAcceptingForG5Bridge",
        "SslmGpuSeqSchemaBoundForG5Bridge",
    ),
    "include/superslm/schema_masks.h": ("accepting_le", "accepting_count"),
    "include/superslm/sslm_abi_functions_g5_comparable.inc": ("sslm_seq_schema_bound",),
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

    def test_public_surfaces_carry_the_landed_contract(self) -> None:
        missing: list[str] = []
        for relative, needles in SURFACES.items():
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for needle in needles:
                if needle.lower() not in text:
                    missing.append(f"{relative}:{needle}")
        self.assertEqual(missing, [])
        api = (ROOT / "docs/api.md").read_text(encoding="utf-8")
        self.assertNotIn("caller never needs to special-case", api)

    def test_real_model_manifest_is_pinned_and_gate_fails_closed(self) -> None:
        manifest = json.loads((HERE / "t2922_real_model_manifest.json").read_text())
        self.assertEqual(manifest["artifact_sha256"],
                         "6a41f87d3a48c91751b41fe716c4f8289b7ce97761850c3f1f2c53d01201205c")
        self.assertEqual(manifest["prompt_ids_sha256"],
                         "9a998103fffcac7bcdb0607ce83f456f889df3ff4c721ac7c67bbb6ae12863dc")
        prompt_path = ROOT / manifest["prompt_ids_file"]
        self.assertEqual(hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                         manifest["prompt_ids_sha256"])
        self.assertEqual(manifest["decode_budget"], 300)
        self.assertEqual(manifest["schema_name"], "prompt_result")
        self.assertEqual(manifest["expected_stop"], "budget")
        if os.environ.get("SUPERSLM_G5_REAL_MODEL_TESTS"):
            artifact = Path(manifest["artifact"])
            self.assertTrue(artifact.is_file(), f"required real artifact missing: {artifact}")

    def test_every_mutant_has_one_live_exact_anchor(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as scratch:
            for mutant in "abcdefghijkl":
                output = Path(scratch) / f"mutant-{mutant}.cpp"
                result = subprocess.run(
                    [sys.executable, str(HERE / "make_t2922_gpu_schema_accepting_mutants.py"),
                     "--engine", str(ROOT), "--out", str(output), "--mutant", mutant],
                    text=True, capture_output=True
                )
                self.assertEqual(result.returncode, 0, f"{mutant}: {result.stderr}")
                self.assertIn(f"MUTANT {mutant} READY", result.stdout)
                self.assertIn(f"MUT {mutant}", output.read_text(encoding="utf-8"))

    def test_ctest_and_ci_name_this_contract(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
        self.assertIn("t2933_schema_query_contract", cmake)
        self.assertIn("t2933_schema_query_contract", workflow)


if __name__ == "__main__":
    unittest.main()
