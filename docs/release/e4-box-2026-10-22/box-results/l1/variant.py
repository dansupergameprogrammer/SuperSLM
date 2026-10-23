"""L1 loader constructions (debunker): build one scratch population + manifest + licence for the loader.

usage: python -I variant.py <population> <candidate-dir> <dest> accept|edit|v1 [<v1-dir>]

accept: an exact copy of the population, the candidate manifest's bytes, the candidate licence's bytes.
edit:   the copy with flat05 adopter 0's first token changed (+1) in both runs (so the pair still reproduces), the
        manifest's two file hashes updated, and the licence re-pointed at that manifest (re-licensed): the loader must
        grade it red ("differs from pinned prefill"), not refuse it on identity.
v1:     the exact copy and manifest, with the licence's record_hashes replaced by the 20 v1 records' hashes (the
        superseded records): the loader must refuse every reference for lacking its commissioned record.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def licence(path):
    return dict(l.split("=", 1) for l in Path(path).read_text(encoding="ascii").splitlines())


def write_licence(path, f):
    order = ["format", "status", "commissioning_verdict", "must_accept", "must_reject", "anchor",
             "population_manifest_sha256", "instrument_source_sha256", "record_hashes"]
    Path(path).write_bytes(("\n".join(f"{k}={f[k]}" for k in order) + "\n").encode("ascii"))


def main():
    pop, cand, dest, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    if dest.exists():
        shutil.rmtree(dest)
    for group in ("P1-real0p5-flat", "P2-real0p5-nested"):
        shutil.copytree(pop / group, dest / group)
    manifest = cand / "manifest.json"
    lic = licence(cand / "license.txt")
    if mode == "accept":
        shutil.copyfile(manifest, dest / "manifest.json")
    elif mode == "edit":
        m = json.loads(manifest.read_text(encoding="utf-8"))
        for run in ("run1", "run2"):
            p = dest / "P1-real0p5-flat" / f"flat05-{run}.ref"
            lines = p.read_bytes().decode("ascii").split("\n")
            for i, line in enumerate(lines):
                if " adopter=0 " in f" {line} " and "tokens=" in line:
                    head, toks = line.split("tokens=", 1)
                    t = toks.split(",")
                    t[0] = str(int(t[0]) + 1)
                    lines[i] = head + "tokens=" + ",".join(t)
                    break
            else:
                raise SystemExit("no adopter 0 row in flat05")
            p.write_bytes("\n".join(lines).encode("ascii"))
            for f in m["files"]:
                if f["path"] == f"P1-real0p5-flat/flat05-{run}.ref":
                    f["sha256"] = sha(p)
        (dest / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                            newline="\n")
        lic["population_manifest_sha256"] = sha(dest / "manifest.json")
    elif mode == "v1":
        shutil.copyfile(manifest, dest / "manifest.json")
        v1 = Path(sys.argv[5])
        lic["record_hashes"] = ",".join(
            sha(v1 / ("real0p5-" + k) / f"{k}{i:02d}.discrimination.json") for k in ("flat", "nested") for i in range(10))
    else:
        raise SystemExit(f"unknown mode {mode}")
    write_licence(dest / "license.txt", lic)
    print(mode, "->", dest, "manifest", sha(dest / "manifest.json"), "licence", sha(dest / "license.txt"))


if __name__ == "__main__":
    main()
