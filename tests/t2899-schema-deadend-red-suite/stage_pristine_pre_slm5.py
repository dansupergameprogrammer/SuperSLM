"""Stage the pre-SLM5 GPU base for build_red_suite_gpu.bat.

The suite compiles a pristine copy of the GPU sources as they stood at commit ccf87c1, the tip before the SLM5 blob
format landed. A release archive carries no git history, so the five files are tracked here as data, each under its own
name with a `.ccf87c1` suffix (a name that tools/geometry_site_census.py, which reads .cpp, .h, .hlsl and .py, does not
count), listed in pristine_pre_slm5/manifest.txt.

manifest.txt holds one line per file, four whitespace-separated fields: the tracked name, the path at ccf87c1, the staged
name (relative to obj_gpu/pristine_pre_slm5/, with a backslash for a nested name) and the SHA-256 of the bytes.

Takes no arguments. Resolves the manifest and the output directory relative to this script. Every tracked copy is checked
against the manifest before anything is written; each staged file is checked again after it is written. Exit 0 when all
files are staged; exit 1, naming the tracked file, when a copy is missing, differs from its recorded SHA-256, or a staged
file does not read back as written.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "pristine_pre_slm5")
MANIFEST = os.path.join(SOURCE_DIR, "manifest.txt")
OUTPUT_DIR = os.path.join(HERE, "obj_gpu", "pristine_pre_slm5")


def refuse(tracked, why):
    print("stage_pristine_pre_slm5: REFUSED %s: %s" % (tracked, why))
    return 1


def read_manifest():
    rows = []
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("manifest.txt line %d holds %d fields, expected 4" % (number, len(fields)))
            rows.append(dict(tracked=fields[0], origin=fields[1], staged=fields[2], sha=fields[3].lower()))
    if not rows:
        raise ValueError("manifest.txt names no file")
    return rows


def main():
    try:
        rows = read_manifest()
    except (OSError, ValueError) as error:
        return refuse("manifest.txt", str(error))
    payloads = []
    for row in rows:
        path = os.path.join(SOURCE_DIR, row["tracked"])
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            return refuse(row["tracked"], "the tracked copy is missing")
        actual = hashlib.sha256(data).hexdigest()
        if actual != row["sha"]:
            return refuse(row["tracked"], "SHA-256 %s differs from the manifest's %s" % (actual, row["sha"]))
        payloads.append((row, data))
    for row, data in payloads:
        target = os.path.join(OUTPUT_DIR, *row["staged"].replace("\\", "/").split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(data)
        with open(target, "rb") as handle:
            written = hashlib.sha256(handle.read()).hexdigest()
        if written != row["sha"]:
            return refuse(row["tracked"], "the staged file %s reads back as %s" % (row["staged"], written))
    print("stage_pristine_pre_slm5: staged %d files into %s" % (len(payloads), OUTPUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
