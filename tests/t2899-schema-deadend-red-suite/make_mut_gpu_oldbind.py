"""Restore one pre-D-SLM7625 bind behavior in a scratch GPU source.

Usage: python make_mut_gpu_oldbind.py LATE|RESTORE <gpu_1p0.cpp> <out.cpp>
Each variant changes one bind-eligibility clear and refuses source drift.
"""
import sys


ANCHORS = {
    "LATE": "\tseq->bind_eligible = false; // D-SLM7625 generation entry: gpu prompt prefill",
    "RESTORE": "\tfresh->bind_eligible = false;\n\t*out_seq = fresh;",
}


def main(argv: list[str]) -> None:
    if len(argv) != 4 or argv[1] not in ANCHORS:
        sys.exit("usage: make_mut_gpu_oldbind.py LATE|RESTORE <gpu_1p0.cpp> <out.cpp>")
    kind, src, dst = argv[1:]
    with open(src, encoding="utf-8") as f:
        source = f.read().replace("\r\n", "\n")
    anchor = ANCHORS[kind]
    if source.count(anchor) != 1:
        sys.exit(f"expected one {kind} anchor, found {source.count(anchor)}")
    if kind == "LATE":
        replacement = "\t// MUT_LATE: prompt prefill leaves late binding eligible."
    else:
        replacement = "\tfresh->bind_eligible = true; // MUT_RESTORE: restored copy can rebind.\n\t*out_seq = fresh;"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(source.replace(anchor, replacement, 1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv)
