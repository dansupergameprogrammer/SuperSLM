"""P-1: a relink cannot renew an independent 7.9 source-bound receipt.

PKV_COMMISSION_TOOL (required; unset fails the cell, there is no default) names the registrar of the Wizard tree under
test. bound() grades the instrument-commission.ps1 beside the registry it is handed (SuperSLM cb7dd42, no fallback), so
the cell copies that file beside its temporary registry and claims its hash (T-3211 N-1; the test author's record,
T-3189 §20)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from verify_paged_kv_binding import bound, canonical, measurement_env, sha


ROOT = Path(__file__).resolve().parents[1]
REGISTRAR = os.environ.get("PKV_COMMISSION_TOOL", "")
RECEIPT = "PKV-TIMING-79-C6-R168"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def grader(path: Path):
    spec = importlib.util.spec_from_file_location("pkv_p1_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade([0.10] * 508)["bar"]


def self_certifying_manifest(binary: Path, repo: Path, registry: Path) -> bool:
    """Replay the reviewed P-1 rule: compare live bytes to refreshed link evidence."""
    manifest = json.loads(Path(str(binary) + ".source-set.json").read_text(encoding="utf-8"))
    receipt = json.loads(registry.read_text(encoding="utf-8"))[RECEIPT]
    source = Path(receipt["instrument"]).relative_to(repo).as_posix()
    return (manifest["sources"][source] == receipt["instrument_sha256"]
            and all(sha(repo / name, normalized=True) == digest
                    for name, digest in manifest["sources"].items())
            and all(sha(binary.parent / name if name.endswith(".exe") else repo / "tools" / name) == digest
                    for name, digest in manifest["invoked"].items()))


class RelinkedBar(unittest.TestCase):
    def test_p1_relinked_bar_refuses_old_successor_claim(self):
        if not REGISTRAR or not Path(REGISTRAR).is_file():
            self.fail("P-1 needs PKV_COMMISSION_TOOL: the instrument-commission.ps1 of the Wizard tree under test; "
                      "there is no default")
        with tempfile.TemporaryDirectory(prefix="pkv-p1-") as folder:
            temp = Path(folder)
            repo = temp / "source"
            subprocess.run(["git", "clone", "--quiet", "--shared", str(ROOT), str(repo)], check=True)
            build = temp / "build"
            build.mkdir()
            binary = build / "superslm_pkv_pair.exe"
            binary.write_bytes(b"7.9 linked pair fixture\n")
            instrument = repo / "tools/paged_kv_pair.cpp"
            runner = repo / "tools/run_paged_kv_pairs.py"
            original = runner.read_bytes()
            manifest_path = Path(str(binary) + ".source-set.json")
            registry_path = temp / "registry.json"
            verifier = temp / "instrument-commission.ps1"  # the registrar bound() grades: the registry's sibling
            verifier.write_bytes(Path(REGISTRAR).read_bytes())
            head, tree = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "HEAD^{tree}")
            recipe = {"commands": {binary.name: "cl /O2 /DNDEBUG paged_kv_pair.cpp"},
                      "compiler_sha256": "d" * 64}

            def relink() -> None:
                manifest = {"schema": "pkv-c6-source-set/2", "git_head": head, "git_tree": tree,
                            "build_recipe": recipe,
                            "sources": {"tools/paged_kv_pair.cpp": sha(instrument, normalized=True),
                                        "tools/run_paged_kv_pairs.py": sha(runner, normalized=True)},
                            "invoked": {binary.name: sha(binary), runner.name: sha(runner)}}
                manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            relink()
            claim = {"manifest_sha256": sha(manifest_path), "git_head": head, "git_tree": tree,
                     "build_recipe_sha256": canonical(recipe), "verifier_sha256": sha(verifier),
                     "measurement_env": measurement_env()}
            registry_path.write_text(json.dumps({RECEIPT: {
                "instrument": str(instrument), "instrument_sha256": sha(instrument, normalized=True),
                "source_binding": claim}}), encoding="utf-8")

            self.assertEqual(grader(runner), "FAIL", "10% slowdown exceeds the real 5% bar")
            self.assertTrue(bound(binary, repo, registry_path, RECEIPT)[0],
                            "the successor claim must accept its original committed bytes")
            text = original.decode("utf-8")
            self.assertEqual(text.count(".05"), 2, "the two 7.9 bar comparisons must be identified")
            runner.write_text(text.replace(".05", ".14"), encoding="utf-8")
            self.assertEqual(grader(runner), "PASS", "the mutation must change the grade")
            relink()  # POST_BUILD refreshed its own evidence; the receipt did not change.
            self.assertNotEqual(sha(manifest_path), claim["manifest_sha256"])
            self.assertTrue(self_certifying_manifest(binary, repo, registry_path),
                            "the reviewed self-certifying rule must accept the relinked mutant")
            self.assertFalse(bound(binary, repo, registry_path, RECEIPT)[0],
                             "a relinked 14% bar must be quarantined by the old 5% claim")
            runner.write_bytes(original)
            relink()
            self.assertEqual(grader(runner), "FAIL")
            self.assertTrue(bound(binary, repo, registry_path, RECEIPT)[0],
                            "the same claim must accept the restored, unmodified bytes")


if __name__ == "__main__":
    unittest.main()
