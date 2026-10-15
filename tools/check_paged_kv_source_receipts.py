"""RC-2/RC-3 source and invoked-byte binding controls for C6 instruments.

These controls exercise the builder-owned binding path. Instrument verdicts
remain quarantined until independent commissioning of final bytes.

Environment (all required; unset or missing fails the controls, never skips them; T-3211 N-1, the test author's
record T-3189 §20). The defaults were a T-3114 build and this checkout's head, which the build's manifests do not name,
so every positive member read unbound:
  PKV_BFR              the R9 build holding superslm_pkv_pair.exe and its manifest (RC-3): B-FR
  PKV_LIFECYCLE_BUILD  a build of the same head holding superslm_pkv_lifecycle_prod.exe and its manifest (RC-2); B-FR
                       builds only the three pair targets
  PKV_BFR_REPO         a clean checkout of SuperSLM at the manifests' git_head: the claims' head and tree, the
                       instrument paths, and the repository bound() reads
  PKV_COMMISSION_TOOL  the registrar of the Wizard tree under test, copied beside each fixture registry (bound() grades
                       the registrar beside the registry it is handed, SuperSLM cb7dd42, no fallback)
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from verify_paged_kv_binding import bound, canonical, measurement_env, sha


ROOT = Path(__file__).resolve().parents[1]
ENV = {name: os.environ.get(name, "") for name in
       ("PKV_BFR", "PKV_LIFECYCLE_BUILD", "PKV_BFR_REPO", "PKV_COMMISSION_TOOL")}
PROVENANCE = ROOT / "tools/paged_kv_c6_provenance.ps1"
PAIR_SCRIPT = ROOT / "tools/run_paged_kv_pairs.py"
spec = importlib.util.spec_from_file_location("pkv_receipt_pair", PAIR_SCRIPT)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class ReceiptBinding(unittest.TestCase):
    def setUp(self):
        missing = [name for name, value in ENV.items() if not value or not Path(value).exists()]
        if missing:
            self.fail(f"the RC controls need {', '.join(missing)} (see the module docstring); there is no default")
        self.repo = Path(ENV["PKV_BFR_REPO"])
        self.registrar = Path(ENV["PKV_COMMISSION_TOOL"])

    def build(self, target: str) -> Path:
        directory = Path(ENV["PKV_BFR"] if target == "superslm_pkv_pair.exe" else ENV["PKV_LIFECYCLE_BUILD"])
        if not (directory / target).is_file():
            self.fail(f"{directory} holds no {target}")
        return directory

    def registry(self, path: Path, receipt_id: str, instrument: Path, source_hash: str) -> None:
        target = "superslm_pkv_pair.exe" if "TIMING-79" in receipt_id else "superslm_pkv_lifecycle_prod.exe"
        manifest_path = self.build(target) / (target + ".source-set.json")
        manifest = json.loads(manifest_path.read_text())
        head = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
        self.assertEqual(manifest["git_head"], head, f"PKV_BFR_REPO is not at {manifest_path.name}'s head")
        verifier = path.parent / "instrument-commission.ps1"  # the registrar bound() grades: the registry's sibling
        verifier.write_bytes(self.registrar.read_bytes())
        path.write_text(json.dumps({receipt_id: {"instrument": str(instrument),
            "instrument_sha256": source_hash,
            "source_binding": {"manifest_sha256": sha(manifest_path), "git_head": head,
                               "git_tree": tree, "build_recipe_sha256": canonical(manifest["build_recipe"]),
                               "verifier_sha256": sha(verifier),
                               "measurement_env": measurement_env()}}}), encoding="utf-8")

    def powershell_binding(self, binary: Path, registry: Path, receipt_id: str) -> bool:
        code = (f'. "{PROVENANCE}"\n'
                'function Get-PkvSha256([string]$Path) { '
                '$hash = [Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::ReadAllBytes($Path)); '
                'return [BitConverter]::ToString($hash).Replace("-", "").ToLowerInvariant() }\n'
                f'Test-PkvSourceBinding -Binary "{binary}" -RegistryId "{receipt_id}" '
                f'-Repo "{self.repo}" -RegistryPath "{registry}"\n')
        script = registry.parent / "check.ps1"
        script.write_text(code, encoding="utf-8")
        result = subprocess.run(["powershell", "-NoProfile", "-File", str(script)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip().splitlines()[-1] == "True"

    def test_rc2_lifecycle_old_source_and_changed_bytes_quarantine(self):
        build = self.build("superslm_pkv_lifecycle_prod.exe")
        exe = build / "superslm_pkv_lifecycle_prod.exe"
        manifest = json.loads(exe.with_suffix(exe.suffix + ".source-set.json").read_text())
        source = self.repo / "tests/paged-kv/c6_lifecycle_timing.cpp"
        current_hash = manifest["sources"][source.relative_to(self.repo).as_posix()]
        receipt_id = "PKV-LIFECYCLE-C6-R168"
        with tempfile.TemporaryDirectory(prefix="pkv-rc2-") as folder:
            temp = Path(folder)
            registry = temp / "registry.json"
            self.registry(registry, receipt_id, source, current_hash)
            self.assertTrue(self.powershell_binding(exe, registry, receipt_id))
            self.registry(registry, receipt_id, source, "0" * 64)
            self.assertFalse(self.powershell_binding(exe, registry, receipt_id))
            self.registry(registry, receipt_id, source, current_hash)
            copied = temp / exe.name
            for name in (exe.name, "superslm.lib", exe.name + ".source-set.json"):
                shutil.copy2(build / name, temp / name)
            self.assertTrue(self.powershell_binding(copied, registry, receipt_id))
            with copied.open("ab") as output:
                output.write(b"changed executable")
            self.assertFalse(self.powershell_binding(copied, registry, receipt_id))
            old_id = "PKV-LIFECYCLE-C6"
            code = (f'. "{PROVENANCE}"; '
                    f'@(Get-PkvRunnerCommissionedIds -VerifyLines @("COMMISSIONED  {old_id}")).Count')
            result = subprocess.run(["powershell", "-NoProfile", "-Command", code],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "0")

    def test_rc3_pair_old_source_and_invoked_bytes_quarantine(self):
        build = self.build("superslm_pkv_pair.exe")
        exe = build / "superslm_pkv_pair.exe"
        manifest_path = exe.with_suffix(exe.suffix + ".source-set.json")
        manifest = json.loads(manifest_path.read_text())
        source = self.repo / "tools/paged_kv_pair.cpp"
        current_hash = manifest["sources"][source.relative_to(self.repo).as_posix()]
        receipt_id = "PKV-TIMING-79-C6-R168"
        with tempfile.TemporaryDirectory(prefix="pkv-rc3-") as folder:
            temp = Path(folder)
            registry = temp / "registry.json"
            verifier = temp / "verify.ps1"
            verifier.write_text(f'Write-Output "COMMISSIONED  {receipt_id}"\n', encoding="utf-8")
            self.registry(registry, receipt_id, source, current_hash)
            with patch.dict(os.environ, {"PKV_C6_REGISTRY": str(registry),
                                         "PKV_C6_RECEIPT_VERIFIER": str(verifier)}):
                self.assertFalse(pair.source_bound_receipt(exe, "79")[0], "redirected authority must quarantine")
                self.assertTrue(bound(exe, self.repo, registry, receipt_id)[0])
                old_source = Path("D:/SuperSLM/.worktrees/T-3093/tests/paged-kv/c6_dim7_throughput.cpp")
                self.registry(registry, receipt_id, old_source, current_hash)
                self.assertFalse(bound(exe, self.repo, registry, receipt_id)[0])
                self.registry(registry, receipt_id, source, current_hash)
                copied = temp / exe.name
                for name in (exe.name, "superslm.lib", exe.name + ".source-set.json"):
                    shutil.copy2(build / name, temp / name)
                self.assertTrue(bound(copied, self.repo, registry, receipt_id)[0])
                with copied.open("ab") as output:
                    output.write(b"changed executable")
                self.assertFalse(bound(copied, self.repo, registry, receipt_id)[0])
                shutil.copy2(exe, copied)
                changed = json.loads((temp / manifest_path.name).read_text())
                changed["invoked"][PAIR_SCRIPT.name] = "0" * 64
                (temp / manifest_path.name).write_text(json.dumps(changed), encoding="utf-8")
                self.assertFalse(bound(copied, self.repo, registry, receipt_id)[0])


if __name__ == "__main__":
    unittest.main()
