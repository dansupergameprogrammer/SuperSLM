"""T-2916 (Curie), closing TE-370 M1: generates a GPU MUT_NOREARM variant of `gpu_1p0.cpp` FROM
THE LIVE TIP, replacing the out-of-repo `D:\\_t2900\\refs\\gpu_1p0_mut_norearm.cpp` (a static copy
of a since-superseded `gpu_1p0_v5.cpp` reference, predating the fault-injection seam
`make_mut_gpu_checkedreturn_seam.py` already regenerates the checked-return mutant against).

Same technique as that script (exact literal match, refuses unless exactly one occurrence),
applied to the sibling mutation Sec3.10.3 row 11 also names: the checked return to
`model->schemas.Transition(...)` stays intact (a genuine miss still yields `-2` at `SSLM_OK`),
but the `ready_for_logits` re-arm on that same miss is dropped -- the NEXT decode call then takes
the embed branch instead of the ready branch and drives a fresh forward, which must turn GPU
Cell 1's resumability assertion red while the FIRST dead-end call (Cell 1's own first assertion,
and Cell 2) stays green (`D:\\_t2900\\refs\\make_mut_norearm.py`'s own stated intent, reproduced
here against source that has not gone stale).

Usage: python make_mut_gpu_norearm_seam.py <path-to-gpu_1p0.cpp> <out-mutant.cpp>
"""
import sys

_ORIGINAL = (
    "\t\tif (!has_transition) {\n"
    "\t\t\t*out_token = -2;\n"
    "\t\t\tseq->ready_for_logits = true;\n"
    "\t\t\treturn SSLM_OK;\n"
    "\t\t}\n"
)

_REPLACEMENT = (
    "\t\tif (!has_transition) {\n"
    "\t\t\t// T-2916 MUT_NOREARM: the checked return stays (a genuine miss still yields -2 at\n"
    "\t\t\t// SSLM_OK), but the ready_for_logits re-arm is dropped.\n"
    "\t\t\t*out_token = -2;\n"
    "\t\t\treturn SSLM_OK;\n"
    "\t\t}\n"
)


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: make_mut_gpu_norearm_seam.py <path-to-gpu_1p0.cpp> <out-mutant.cpp>")
    src, dst = argv[1], argv[2]
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    count = text.count(_ORIGINAL)
    if count != 1:
        sys.exit(f"expected exactly 1 occurrence of the has_transition-miss block, found {count} "
                  "-- source may have moved, refusing to guess")
    out = text.replace(_ORIGINAL, _REPLACEMENT, 1)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv)
