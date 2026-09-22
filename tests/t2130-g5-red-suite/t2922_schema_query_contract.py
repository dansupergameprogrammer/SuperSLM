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
REQUIRED_CELLS = [
    "BIND_FRESH_RESET_RESTORED_RESET_ACCEPT",
    "BIND_AFTER_PROMPT_PREFILL_REJECT",
    "BIND_AFTER_DECODE_REJECT",
    "BIND_AFTER_DEAD_END_REJECT",
    "BIND_AFTER_REFUSED_FIRST_CALL_REJECT",
    "RESET_REUSE",
    "RESET_AFTER_GUARD_REFUSAL",
    "SCHEMA_QUERY_REAL_DECODE",
    "ADAPTER_SWAP_RESET",
]
RETIRED = {
    "BIND_ORIGIN_MATRIX.strict_layer", "BIND_ORIGIN_MATRIX.full_depth",
    "BIND_ORIGIN_MATRIX.zero_commit", "RESTORED_BIND_ORIGIN_MATRIX.synthetic_sources",
    "BIND_NO_WRITE_ACCEPT", "BIND_EMPTY_ADOPT_ACCEPT", "BIND_POST_SUBMIT_NO_TOKEN",
    "queue_allocation_inflight_publication_mutations", "first_write_tables",
    "generation_history_field_censuses", "delegated_callback_placement",
    "ExecuteCommandLists_placement_claims", "TE380_TE384_historical_probe_reruns",
    "inverse_exit_apparatus",
}


class T2933SchemaQueryContract(unittest.TestCase):
    def test_population_is_named_and_complete(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        ids = [row["id"] for row in data["cells"]]
        self.assertEqual(ids, REQUIRED_CELLS)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(data["mutants"]), set("abcdefghijklm"))
        self.assertEqual(set(data["retired"]), RETIRED)
        self.assertEqual(data["plan_commit"], "38420c8a9a")
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
        runner = (HERE / "run_t2922_real_model.ps1").read_text(encoding="utf-8")
        self.assertEqual(manifest["artifact_sha256"],
                         "696c2a4ac412f1c5866a446eb32aa6500db77e5ae8aff0018438103c607e61bd")
        prompt_path = ROOT / manifest["prompt_ids_file"]
        prompt_bytes = prompt_path.read_text(encoding="utf-8").rstrip("\r\n").encode()
        self.assertEqual(hashlib.sha256(prompt_bytes).hexdigest(),
                         manifest["prompt_ids_text_without_final_newline_sha256"])
        self.assertEqual(manifest["decode_budget"], 300)
        self.assertEqual(manifest["schema_name"], "potion_shop_order")
        self.assertEqual(manifest["expected_stop"], "budget")
        self.assertNotIn("Get-FileHash", runner)
        self.assertNotIn("SHA256]::HashData", runner)
        if os.environ.get("SUPERSLM_G5_REAL_MODEL_TESTS"):
            artifact = Path(manifest["artifact"])
            self.assertTrue(artifact.is_file(), f"required real artifact missing: {artifact}")

    def test_every_mutant_has_one_live_exact_anchor(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as scratch:
            for mutant in "abcdefghijm":
                output = Path(scratch) / f"mutant-{mutant}.cpp"
                result = subprocess.run(
                    [sys.executable, str(HERE / "make_t2922_gpu_schema_accepting_mutants.py"),
                     "--engine", str(ROOT), "--out", str(output), "--mutant", mutant],
                    text=True, capture_output=True
                )
                self.assertEqual(result.returncode, 0, f"{mutant}: {result.stderr}")
                self.assertIn(f"MUTANT {mutant} READY", result.stdout)
                self.assertIn(f"MUT {mutant}", output.read_text(encoding="utf-8"))

    def test_unlanded_bind_mutants_have_exact_committed_recipes(self) -> None:
        for mutant in "kl":
            with tempfile.TemporaryDirectory(dir=ROOT / "build") as scratch:
                output = Path(scratch) / f"mutant-{mutant}.recipe"
                result = subprocess.run(
                    [sys.executable, str(HERE / "make_t2922_gpu_schema_accepting_mutants.py"),
                     "--engine", str(ROOT), "--out", str(output), "--mutant", mutant],
                    text=True, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"MUTANT {mutant} PREPARED", result.stdout)
                recipe = output.read_text(encoding="utf-8")
                self.assertIn("anchor=", recipe)
                self.assertIn("replacement=", recipe)

    def test_ctest_and_ci_name_this_contract(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
        self.assertIn("t2933_schema_query_contract", cmake)
        self.assertIn("t2933_schema_query_contract", workflow)

    def test_aggregate_runner_computes_the_delegated_query_cell(self) -> None:
        lifecycle = (HERE / "t2941_simple_bind_red.cpp").read_text(encoding="utf-8")
        aggregate = (HERE / "run_t2941_simple_bind_suite.ps1").read_text(encoding="utf-8")
        self.assertNotIn('Cell("SCHEMA_QUERY_REAL_DECODE"', lifecycle)
        self.assertIn("$query.ExitCode -eq 0 -and $real.ExitCode -eq 0", aggregate)
        self.assertIn("SUMMARY cells=9", aggregate)


if __name__ == "__main__":
    unittest.main()
