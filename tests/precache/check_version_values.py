"""M1/M2 value receipt (plan §9 M1, rev 1.18.1 audit G2): the version read as values, not only as an equality.

check_version_header.py requires sslm_version.h to equal CMake's project version, and 7.12's independent parser
requires a saved blob's producer to equal it too. Neither sees both move together to a release number too early. This
receipt reads each value and requires it to be the expected version named on the command line: at M1, main's
pre-squash CMake version (1.13.1, PC-7); at E5's release commit and M2, 1.14.0.

usage: check_version_values.py --expect X.Y.Z [--cmake PATH] [--header PATH] [--blob SPX1 ...]

A blob is read independently of the engine's reader (struct at the SPX1 table's offsets: magic 0, backend 4, producer
major/minor/patch 8/12/16), as 7.12's parser reads it.
"""

import argparse
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]


def cmake_version(path: Path) -> tuple:
    text = path.read_text(encoding="utf-8")
    found = re.findall(r"\bproject\s*\(\s*superslm\b[^)]*?\bVERSION\s+(\d+)\.(\d+)\.(\d+)\b", text, re.I | re.S)
    if len(found) != 1:
        raise ValueError(f"{path}: expected one project(superslm VERSION x.y.z)")
    return tuple(map(int, found[0]))


def header_version(path: Path) -> tuple:
    text = path.read_text(encoding="utf-8")
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        found = re.findall(r"^\s*#\s*define\s+SUPERSLM_VERSION_" + part + r"\s+(\d+)\b", text, re.M)
        if len(found) != 1:
            raise ValueError(f"{path}: expected one SUPERSLM_VERSION_{part}")
        parts.append(int(found[0]))
    return tuple(parts)


def blob_producer(path: Path) -> tuple:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"SPX1":
        raise ValueError(f"{path}: not an SPX1 blob")
    if struct.unpack_from("<I", data, 4)[0] != 0:
        raise ValueError(f"{path}: not a CPU blob (backend {struct.unpack_from('<I', data, 4)[0]})")
    return struct.unpack_from("<III", data, 8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--cmake", type=Path, default=ROOT / "CMakeLists.txt")
    parser.add_argument("--header", type=Path, default=ROOT / "include/superslm/sslm_version.h")
    parser.add_argument("--blob", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.expect):
        parser.error("--expect takes X.Y.Z")
    expect = tuple(map(int, args.expect.split(".")))
    readings = [("CMake project version", cmake_version, args.cmake),
                ("sslm_version.h", header_version, args.header)]
    readings += [(f"SPX1 producer {blob}", blob_producer, blob) for blob in args.blob]
    failures = 0
    for label, read, path in readings:
        try:
            value = read(path)
        except (OSError, ValueError, struct.error) as error:
            print(f"FAIL {label}: {error}")
            failures += 1
            continue
        shown = ".".join(map(str, value))
        if value == expect:
            print(f"ok   {label} reads {shown}")
        else:
            print(f"FAIL {label} reads {shown}, expected {args.expect}")
            failures += 1
    print(f"check_version_values: {failures} failure(s) against {args.expect}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
