"""Paged-KV plan rev 16.29 §30.2 and §31 (D-SLM8477, D-SLM8484): prove which engine library a C6 pair executable was
linked against, from the link step the build ran, not from the executable's word about itself.

The evidence is the executable's link manifest, `<exe>.source-set.json`, written at link time by
tools/write_paged_kv_source_manifest.py. Its `build_recipe.commands[<exe file name>]` holds `ninja -C <build> -t commands
<exe>`: every command ninja runs to produce the executable, from the build.ninja CMake generated out of the target's real
target_link_libraries. The command that writes the executable is the link step, and it names every library the linker
read. Ninja is required: under any other generator the commands are null, and check 2 refuses.

Usage:
    check_paged_kv_pair_link.py --exe E --library-name L --reference-manifest R [--source REL]
                                [--registry G --id I --slot S]
    check_paged_kv_pair_link.py --tie-only --exe E --registry G --id I --slot S

Checks, in order. Each failure prints `REFUSED check <n>: <reason>` and exits 3; success prints
`LINKED <exe sha256> <L> <library sha256>` (tie-only: `TIED <exe sha256> <S>`) and exits 0. Bad arguments exit 2.
  1. `<E>.source-set.json` exists, its schema is `pkv-c6-source-set/2`, and `invoked[<E's file name>]` equals the SHA-256
     of E.
  2. `build_recipe.commands[<E's file name>]` is a non-empty string.
  3. Exactly one command writes E: on MSVC a `link` or `lld-link` command with `/out:` naming E; elsewhere, a command
     with `-o E`. A link step that reads a response file (`@file`) refuses: its inputs are not in the record.
  4. Among that link step's library inputs (its non-option operands ending `.lib` or `.a`; `/out:`, `/implib:`, `/pdb:`
     and `/ilk:` are options, never inputs), the engine libraries (file names beginning `superslm`, case-insensitively)
     are exactly one, and it is L.
  5. That library file, resolved against the build directory (E's directory, where ninja runs), has a SHA-256 equal to
     `invoked[<L>]` in E's manifest and to `invoked[<L>]` in R, the production pair's manifest.
  6. No command in `commands[<E's file name>]` defines a `SUPERSLM_MUTANT` macro (`/D` or `-D`, joined or separate,
     quoted or not): MA's own compile and every library compile ninja would run for it. A command that reads a response
     file refuses, since a definition could hide in it.
  7. With --registry, --id and --slot: the SHA-256 of E equals the entry's
     `source_binding.construction_executable_sha256.<S>`, compared ordinally.
  8. Among the link step's non-option operands, the object files (`.obj` or `.o`) are exactly one; the compile command in
     the same list that writes that object compiles REL; and REL is the only `.c`, `.cc`, `.cpp` or `.cxx` file among E's
     manifest `sources`. REL defaults to `tools/paged_kv_pair_ma.cpp`, MA's source; --source names another executable's
     one source (the production pair's `tools/paged_kv_pair.cpp`), so the checker can be run on any pair the manifest
     writer records.
The tie-only mode runs check 7 alone. It is for 7.9-MR-OPT's executable, which writes no manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys

SCHEMA = "pkv-c6-source-set/2"
LINKERS = {"link", "link.exe", "lld-link", "lld-link.exe"}
COMPILE_SOURCES = (".c", ".cc", ".cpp", ".cxx")
# a SUPERSLM_MUTANT definition: /D or -D, the name joined to it or after whitespace, optionally quoted
MUTANT_DEFINE = re.compile(r'(?:^|[\s"])[/-]D\s*"?SUPERSLM_MUTANT', re.IGNORECASE)
RESPONSE_FILE = re.compile(r'(?:^|\s)"?@[^\s"]')


class Refused(Exception):
    def __init__(self, check: int, reason: str):
        super().__init__(f"check {check}: {reason}")
        self.check = check


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(command: str) -> list[str]:
    """Whitespace-split, each token's surrounding double quotes stripped. The link and compile operands the checks read
    carry no embedded spaces in a ninja command (CMake writes short or quoted-free paths for them)."""
    return [t.strip('"') for t in command.split()]


def segments(command: str) -> list[list[str]]:
    """The command's `&&`-chained parts, each a token list (`cmd /C "a && b"` included)."""
    out, current = [], []
    for token in tokens(command):
        if token == "&&":
            out.append(current)
            current = []
        else:
            current.append(token)
    out.append(current)
    return out


def base(token: str) -> str:
    return PureWindowsPath(token).name


def link_steps(commands: list[str]) -> list[tuple[str, list[str], str]]:
    """Every link step among the commands: (style, the arguments after the linker, the command line)."""
    steps = []
    for command in commands:
        for part in segments(command):
            for i, token in enumerate(part):
                name = base(token).lower()
                if name in LINKERS:
                    steps.append(("msvc", part[i + 1:], command))
                    break
                if name in {"c++", "g++", "clang++", "cc", "gcc", "clang", "ld", "ld.lld"} and "-o" in part[i + 1:]:
                    steps.append(("gnu", part[i + 1:], command))
                    break
    return steps


def output_of(style: str, args: list[str]) -> str | None:
    if style == "msvc":
        for a in args:
            if a.lower().startswith(("/out:", "-out:")):
                return a[5:]
        return None
    for i, a in enumerate(args):
        if a == "-o" and i + 1 < len(args):
            return args[i + 1]
    return None


def operands(style: str, args: list[str]) -> list[str]:
    """The link step's non-option operands."""
    if style == "msvc":
        return [a for a in args if a and not a.startswith(("/", "-"))]
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "-o":
            skip = True
            continue
        if not a.startswith("-"):
            out.append(a)
    return out


def same_file(build: Path, token: str, target: Path) -> bool:
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = build / candidate
    try:
        return candidate.resolve() == target.resolve()
    except OSError:
        return False


def load_json(path: Path, check: int, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refused(check, f"{what} {path} is missing or unreadable ({type(error).__name__})") from None
    if not isinstance(value, dict):
        raise Refused(check, f"{what} {path} is not a JSON object")
    return value


def registry_tie(exe: Path, exe_sha: str, registry: Path, entry_id: str, slot: str) -> None:
    """Check 7."""
    entries = load_json(registry, 7, "registry")
    entry = entries.get(entry_id)
    if not isinstance(entry, dict):
        raise Refused(7, f"registry {registry} has no entry {entry_id!r}")
    binding = entry.get("source_binding")
    mapping = binding.get("construction_executable_sha256") if isinstance(binding, dict) else None
    if not isinstance(mapping, dict):
        raise Refused(7, f"entry {entry_id} carries no source_binding.construction_executable_sha256 object")
    bound = mapping.get(slot)
    if not isinstance(bound, str) or bound != exe_sha:  # ordinal: no case folding
        raise Refused(7, f"entry {entry_id}'s construction_executable_sha256.{slot} is {bound!r}; {exe.name} is "
                         f"{exe_sha}")


def check(exe: Path, library_name: str, reference: Path, source: str,
          registry: Path | None, entry_id: str | None, slot: str | None) -> str:
    """Checks 1 to 8; returns the LINKED line."""
    exe = exe.resolve()
    build = exe.parent
    if not exe.is_file():
        raise Refused(1, f"{exe} does not exist")
    exe_sha = sha256(exe)
    # 1
    manifest_path = exe.with_name(exe.name + ".source-set.json")
    manifest = load_json(manifest_path, 1, "manifest")
    if manifest.get("schema") != SCHEMA:
        raise Refused(1, f"{manifest_path.name} schema is {manifest.get('schema')!r}, {SCHEMA!r} owed")
    invoked = manifest.get("invoked") if isinstance(manifest.get("invoked"), dict) else {}
    if invoked.get(exe.name) != exe_sha:
        raise Refused(1, f"{manifest_path.name} invoked[{exe.name}] is {invoked.get(exe.name)!r}; the file is {exe_sha}")
    # 2
    recipe = manifest.get("build_recipe") if isinstance(manifest.get("build_recipe"), dict) else {}
    recorded = (recipe.get("commands") or {}).get(exe.name) if isinstance(recipe.get("commands"), dict) else None
    if not isinstance(recorded, str) or not recorded.strip():
        raise Refused(2, f"build_recipe.commands[{exe.name}] is {'null' if recorded is None else 'empty'}: not a Ninja "
                         f"build, or ninja could not list the commands")
    commands = [line for line in recorded.splitlines() if line.strip()]
    # 3
    writers = [(style, args, line) for style, args, line in link_steps(commands)
               if (out := output_of(style, args)) is not None and same_file(build, out, exe)]
    if len(writers) != 1:
        raise Refused(3, f"{len(writers)} commands write {exe.name}; exactly one link step owed")
    style, args, link_line = writers[0]
    if any(a.startswith("@") for a in args):
        raise Refused(3, f"the link step of {exe.name} reads a response file; its inputs are not in the record")
    ops = operands(style, args)
    # 4
    libraries = [o for o in ops if o.lower().endswith((".lib", ".a"))]
    engine = [o for o in libraries if base(o).lower().startswith("superslm")]
    if len(engine) != 1 or base(engine[0]) != library_name:
        raise Refused(4, f"the link step's engine libraries are {[base(e) for e in engine]}; exactly [{library_name!r}] "
                         f"owed")
    # 5
    library_file = Path(engine[0]) if Path(engine[0]).is_absolute() else build / engine[0]
    if not library_file.is_file():
        raise Refused(5, f"{library_file} does not exist")
    library_sha = sha256(library_file)
    reference_manifest = load_json(reference, 5, "reference manifest")
    reference_invoked = reference_manifest.get("invoked") if isinstance(reference_manifest.get("invoked"), dict) else {}
    if invoked.get(library_name) != library_sha or reference_invoked.get(library_name) != library_sha:
        raise Refused(5, f"{library_name} is {library_sha}; {manifest_path.name} records {invoked.get(library_name)!r} "
                         f"and {reference.name} records {reference_invoked.get(library_name)!r}")
    # 6
    for line in commands:
        if RESPONSE_FILE.search(line):
            raise Refused(6, f"a command reads a response file, so its definitions are not in the record: {line[:160]}")
        if MUTANT_DEFINE.search(line):
            raise Refused(6, f"a command defines a SUPERSLM_MUTANT macro: {line[:200]}")
    # 7
    if registry is not None:
        registry_tie(exe, exe_sha, registry, entry_id, slot)
    # 8
    objects = [o for o in ops if o.lower().endswith((".obj", ".o"))]
    if len(objects) != 1:
        raise Refused(8, f"the link step reads {len(objects)} object files {[base(o) for o in objects]}; exactly one owed")
    wanted = source.replace("\\", "/").strip("/").lower()
    compiles = []
    for line in commands:
        toks = tokens(line)
        writes = any(same_file(build, t[3:], build / objects[0]) for t in toks if t.lower().startswith(("/fo", "-fo")))
        writes = writes or any(toks[i] == "-o" and i + 1 < len(toks) and same_file(build, toks[i + 1], build / objects[0])
                               for i in range(len(toks)))
        if writes:
            compiles.append([t for t in toks if t.lower().endswith(COMPILE_SOURCES) and not t.startswith(("/", "-"))])
    if len(compiles) != 1 or len(compiles[0]) != 1 or \
            not compiles[0][0].replace("\\", "/").lower().endswith("/" + wanted):
        raise Refused(8, f"the compile writing {base(objects[0])} compiles {compiles}; exactly {source} owed")
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), dict) else {}
    compiled_sources = sorted(s for s in sources if s.lower().endswith(COMPILE_SOURCES))
    if [s.lower() for s in compiled_sources] != [wanted]:
        raise Refused(8, f"{manifest_path.name} sources' compiled files are {compiled_sources}; exactly [{source}] owed")
    return f"LINKED {exe_sha} {library_name} {library_sha}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--library-name")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--source", default="tools/paged_kv_pair_ma.cpp")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--slot")
    parser.add_argument("--tie-only", action="store_true")
    args = parser.parse_args(argv)
    tie = (args.registry, args.id, args.slot)
    if any(x is not None for x in tie) and not all(x is not None for x in tie):
        parser.error("--registry, --id and --slot go together")
    try:
        if args.tie_only:
            if args.registry is None:
                parser.error("--tie-only needs --registry, --id and --slot")
            exe = args.exe.resolve()
            if not exe.is_file():
                raise Refused(7, f"{exe} does not exist")
            exe_sha = sha256(exe)
            registry_tie(exe, exe_sha, args.registry, args.id, args.slot)
            print(f"TIED {exe_sha} {args.slot}")
            return 0
        if not args.library_name or args.reference_manifest is None:
            parser.error("--library-name and --reference-manifest are required (or --tie-only)")
        print(check(args.exe, args.library_name, args.reference_manifest, args.source, *tie))
        return 0
    except Refused as refused:
        print(f"REFUSED {refused}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
