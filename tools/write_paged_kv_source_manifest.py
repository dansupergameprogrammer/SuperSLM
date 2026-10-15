"""Write a C6 link artifact binding deciding sources and invoked bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path: Path, text: bool = False) -> str:
    data = path.read_bytes()
    if text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def build_recipe(executable: Path, library: Path) -> dict:
    build = executable.parent
    cache = build / "CMakeCache.txt"
    settings = {}
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if ":" in line and "=" in line and not line.startswith(("#", "//")):
                key, value = line.split("=", 1)
                settings[key.split(":", 1)[0]] = value
    ninja = settings.get("CMAKE_MAKE_PROGRAM", "ninja")
    commands = {}
    for target in (executable.name, library.name):
        try:
            result = subprocess.run([ninja, "-C", str(build), "-t", "commands", target],
                                    capture_output=True, text=True)
        except OSError:
            commands[target] = None
            continue
        commands[target] = result.stdout if result.returncode == 0 else None
    compiler = Path(settings["CMAKE_CXX_COMPILER"]) if settings.get("CMAKE_CXX_COMPILER") else None
    return {"commands": commands, "compiler_sha256": digest(compiler) if compiler and compiler.is_file() else None}


def manifest(repo: Path, executable: Path, library: Path, sources: list[Path],
             invoked: list[Path]) -> dict:
    repo = repo.resolve()
    source_hashes = {}
    for source in sources:
        path = source.resolve()
        source_hashes[path.relative_to(repo).as_posix()] = digest(path, text=True)
    byte_hashes = {}
    for path in [executable, library, *invoked]:
        resolved = path.resolve()
        byte_hashes[resolved.name] = digest(resolved)
    return {"schema": "pkv-c6-source-set/2", "git_head": git(repo, "rev-parse", "HEAD"),
            "git_tree": git(repo, "rev-parse", "HEAD^{tree}"),
            "build_recipe": build_recipe(executable, library),
            "sources": dict(sorted(source_hashes.items())), "invoked": dict(sorted(byte_hashes.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], type=Path)
    parser.add_argument("--invoked", action="append", default=[], type=Path)
    args = parser.parse_args()
    record = manifest(args.repo, args.exe, args.library, args.source, args.invoked)
    output = args.exe.with_suffix(args.exe.suffix + ".source-set.json")
    temporary = output.with_name(output.name + ".partial")
    try:
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
