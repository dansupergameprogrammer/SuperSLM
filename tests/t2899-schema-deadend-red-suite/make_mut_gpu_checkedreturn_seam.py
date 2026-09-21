"""T-2909 (Curie) -- generates a GPU MUT_CHECKEDRETURN variant of `gpu_1p0.cpp` FROM THE LIVE
TIP, for TE-365 C2's own guard-vitality proof on `cell_gpu_cell2_degenerate.cpp`.

The existing `D:\\_t2900\\refs\\gpu_1p0_mut_checkedreturn.cpp` (T-2900's own reference, copied
from `gpu_1p0_v5.cpp`) predates T-2905's own fault-injection seam
(`SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION`/`ArmGpuFinishDegenerateLogitRowInjection`)
entirely -- it link-fails against Cell 2 (`LNK2019: unresolved external symbol
ArmGpuFinishDegenerateLogitRowInjection`), because that symbol is defined only in the file this
mutant was never regenerated from. This script reverts the IDENTICAL single line
`D:\\_t2900\\refs\\make_mut_checkedreturn.py` reverts (the checked `has_transition` return,
Sec3.10.1), but starting from `src/gpu/gpu_1p0.cpp` as committed at the tip -- which already
carries the seam -- so the resulting mutant compiles and links against BOTH Cell 1 (unaffected
by this change; still the existing guard-vitality proof) and Cell 2 (this cell's own new proof).

Located structurally, not by line number: `const bool has_transition = ...Transition(...)`
through the immediately following `if (!has_transition) { ... }` block -- unique in this file.
Refuses (nonzero exit, no output written) unless exactly one occurrence is found.

Usage: python make_mut_gpu_checkedreturn_seam.py <path-to-gpu_1p0.cpp> <out-mutant.cpp>
"""
import re
import sys

_PAT = re.compile(
    r"(?P<i>[ \t]+)const bool has_transition = model->schemas\.Transition\(\n"
    r"[ \t]+\*entry, seq->dfa_walk_state, static_cast<uint32_t>\(produced\), &next_state\);\n"
    r"(?P=i)if \(!has_transition\) \{\n"
    r"(?P=i)\t\*out_token = -2;\n"
    r"(?P=i)\tseq->ready_for_logits = true;\n"
    r"(?P=i)\treturn SSLM_OK;\n"
    r"(?P=i)\}\n"
)


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: make_mut_gpu_checkedreturn_seam.py <path-to-gpu_1p0.cpp> <out-mutant.cpp>")
    src, dst = argv[1], argv[2]
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    hits = list(_PAT.finditer(text))
    if len(hits) != 1:
        sys.exit("expected exactly 1 occurrence of the checked-return block, found %d -- source "
                  "may have moved, refusing to guess" % len(hits))

    def repl(m):
        i = m.group("i")
        return (
            "%smodel->schemas.Transition(*entry, seq->dfa_walk_state, "
            "static_cast<uint32_t>(produced),\n"
            "%s                           &next_state);  "
            "// T-2909 MUT_CHECKEDRETURN: return value discarded.\n" % (i, i)
        )

    out = _PAT.sub(repl, text)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print("wrote %s" % dst)


if __name__ == "__main__":
    main(sys.argv)
