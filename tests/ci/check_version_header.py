"""Require the vendorable engine version header to match CMake's project version.

The independent must-accept and must-reject constructions are commissioned in E2.
"""

import argparse
from pathlib import Path
import re
import sys


def check(cmake: Path, header: Path) -> None:
    project = cmake.read_text(encoding="utf-8")
    declarations = re.findall(r"\bproject\s*\(\s*superslm\b([^)]*)\)", project, re.I | re.S)
    if len(declarations) != 1:
        raise ValueError("expected exactly one parseable project(superslm ...) declaration")
    versions = re.findall(r"\bVERSION\s+(\d+)\.(\d+)\.(\d+)\b", declarations[0], re.I)
    if len(versions) != 1:
        raise ValueError("project(superslm ...) needs exactly one parseable VERSION major.minor.patch")
    expected = tuple(map(int, versions[0]))
    source = header.read_text(encoding="utf-8")
    actual = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        found = re.findall(r"^\s*#\s*define\s+SUPERSLM_VERSION_" + part + r"\s+(\d+)\s*(?://.*)?$",
                           source, re.M)
        if len(found) != 1:
            raise ValueError(f"expected exactly one SUPERSLM_VERSION_{part} integer in {header}")
        actual.append(int(found[0]))
    if tuple(actual) != expected:
        raise ValueError(f"version header {tuple(actual)} differs from CMake {expected}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", type=Path, default=root / "CMakeLists.txt")
    parser.add_argument("--header", type=Path, default=root / "include/superslm/sslm_version.h")
    args = parser.parse_args()
    try:
        check(args.cmake, args.header)
    except (OSError, ValueError) as exc:
        print(f"check_version_header: {exc}", file=sys.stderr)
        return 1
    print("check_version_header: version matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
