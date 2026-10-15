"""Fail-closed C6 receipt check shared by the Python and PowerShell runners.

The link manifest is evidence, never authority. A separately verified registry
entry must pin its exact bytes, Git tree, build recipe, and measurement hooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path: Path, normalized: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalized else data).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def measurement_env() -> dict[str, str]:
    prefixes = ("SUPERSLM_PAGED_KV_", "PKV_DEBUNK_")
    exact = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "CUDA_VISIBLE_DEVICES", "CL", "CXXFLAGS", "LDFLAGS"}
    excluded = {"PKV_C6_REGISTRY", "PKV_C6_RECEIPT_VERIFIER"}
    return dict(sorted((name, value) for name, value in os.environ.items()
                       if name not in excluded and (name.startswith(prefixes) or name in exact)))


def registrar_for(registry_path: Path) -> Path:
    """The registrar a registry is graded against: instrument-commission.ps1 beside the registry file."""
    return Path(registry_path).parent / "instrument-commission.ps1"


def bound(binary: Path, repo: Path, registry_path: Path, receipt_id: str) -> tuple[bool, str | None]:
    """(bound, the SHA-256 of `binary`'s source-set manifest or None when there is no manifest) for registry entry
    `receipt_id`. Bound requires every check below; any failure, missing file or parse error is unbound.

    The registrar the entry's `verifier_sha256` binds is `instrument-commission.ps1` in the directory of `registry_path`
    (`registrar_for`), never a fixed path. A caller that builds a registry anywhere else, a test fixture's temporary
    directory included, must place the registrar it claims beside that registry; with none there the call is unbound,
    and there is no fallback to another tree's registrar (T-3198 §23, T-3211 N-1)."""
    path = Path(str(binary) + ".source-set.json")
    if not path.is_file() or not registry_path.is_file():
        return False, sha(path) if path.is_file() else None
    digest = sha(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entry = json.loads(registry_path.read_text(encoding="utf-8"))[receipt_id]
        claim = entry["source_binding"]
        if manifest["schema"] != "pkv-c6-source-set/2" or claim["manifest_sha256"] != digest:
            return False, digest
        head, tree = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "HEAD^{tree}")
        if (not head or not tree or claim["git_head"] != head or claim["git_tree"] != tree or
            manifest["git_head"] != head or manifest["git_tree"] != tree):
            return False, digest
        if git(repo, "diff", "--name-only", "HEAD", "--"):
            return False, digest
        if (claim["build_recipe_sha256"] != canonical(manifest["build_recipe"]) or
            not manifest["build_recipe"]["compiler_sha256"] or
            any(not value for value in manifest["build_recipe"]["commands"].values())):
            return False, digest
        if claim["measurement_env"] != measurement_env():
            return False, digest
        # The registrar whose bytes verifier_sha256 binds is the sibling of the registry this call was handed (T-3198
        # §23, D-SLM8494 scope): at R0(c) the registry is D:/Wizard/Claude/Tooling/instrument-registry.json, so it is
        # that tree's instrument-commission.ps1; a worktree's registry grades that worktree's registrar. A missing
        # sibling is unbound; there is no fallback to another tree's registrar.
        verifier = registrar_for(registry_path)
        if not verifier.is_file() or sha(verifier) != claim["verifier_sha256"]:
            return False, digest
        instrument = Path(entry["instrument"]).resolve().relative_to(repo.resolve()).as_posix()
        if manifest["sources"][instrument] != entry["instrument_sha256"]:
            return False, digest
        for name, expected in manifest["sources"].items():
            live = (repo / name).resolve()
            live.relative_to(repo.resolve())
            committed = subprocess.run(["git", "-C", str(repo), "show", f"{head}:{name}"], capture_output=True)
            if (committed.returncode or sha(live, normalized=True) != expected or
                hashlib.sha256(committed.stdout.replace(b"\r\n", b"\n")).hexdigest() != expected):
                return False, digest
        for name, expected in manifest["invoked"].items():
            live = binary.parent / name if name.endswith((".exe", ".lib")) else repo / "tools" / name
            if sha(live) != expected:
                return False, digest
        return True, digest
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return False, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    okay, digest = bound(args.binary, args.repo, args.registry, args.id)
    print(json.dumps({"bound": okay, "manifest_sha256": digest}))


if __name__ == "__main__":
    main()
