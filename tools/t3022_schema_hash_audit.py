"""List SCM1 schema names and engine-compatible FNV-1a hashes in SSLM artifacts.

Source at 5d5ca58: artifact.cpp:275-296, :358-365 (section table),
artifact.h:59 (section type 30), schema_masks.h:110-120, :237-264,
:281-305, :403 (SCM1 layout and hash). This checks the PC-12 hash condition,
not the artifact's full structural or SHA-256 validity.
"""

import argparse
import json
import struct
import sys
from pathlib import Path


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


def fnv1a64(name):
    value = 14695981039346656037
    for byte in name:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def audit(path):
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:4] != b"SSLM":
            raise ValueError("not an SSLM artifact")
        if u32(header, 4) != 2 or u32(header, 8) != 64:
            raise ValueError("unsupported artifact header")
        count = u32(header, 12)
        if count > 4096 or u64(header, 24) != path.stat().st_size:
            raise ValueError("invalid section count or file size")
        table = stream.read(count * 40)
        if len(table) != count * 40:
            raise ValueError("truncated section table")
        sections = [(u64(table, i * 40 + 8), u64(table, i * 40 + 16))
                    for i in range(count) if u32(table, i * 40) == 30]
        if len(sections) > 1:
            raise ValueError("multiple SchemaMasks sections")
        result = {"artifact": str(path), "schema_section": bool(sections), "schemas": [],
                  "duplicate_hashes": [], "zero_hashes": [], "status": "PASS"}
        if not sections:
            return result
        offset, size = sections[0]
        if offset + size > path.stat().st_size or size < 24:
            raise ValueError("SchemaMasks section out of bounds")
        stream.seek(offset)
        section = stream.read(size)
    if section[:4] != b"SCM1" or u32(section, 4) != 1:
        raise ValueError("unsupported SCM1 header")
    count = u32(section, 8)
    names_size = u32(section, 16)
    names_offset = 24 + count * 56
    if not 0 < count <= 65536 or names_offset + names_size > len(section):
        raise ValueError("SCM1 manifest out of bounds")
    seen = {}
    for index in range(count):
        descriptor = 24 + index * 56
        start = u32(section, descriptor)
        length = u32(section, descriptor + 4)
        if not length or start + length > names_size:
            raise ValueError(f"schema {index} name out of bounds")
        name = section[names_offset + start:names_offset + start + length]
        hash_value = fnv1a64(name)
        result["schemas"].append({"index": index, "name": name.decode("utf-8", "backslashreplace"),
                                  "name_hex": name.hex(), "fnv1a64": f"{hash_value:016x}"})
        if hash_value == 0:
            result["zero_hashes"].append(index)
        if hash_value in seen:
            result["duplicate_hashes"].append([seen[hash_value], index])
        else:
            seen[hash_value] = index
    if result["zero_hashes"] or result["duplicate_hashes"]:
        result["status"] = "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.artifacts:
        try:
            result = audit(path)
        except (OSError, ValueError, struct.error) as exc:
            result = {"artifact": str(path), "status": "ERROR", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        failed |= result["status"] != "PASS"
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
