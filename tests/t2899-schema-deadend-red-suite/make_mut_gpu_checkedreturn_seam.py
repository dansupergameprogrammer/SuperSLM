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
    r"(?P<i>[ \t]+)if \(!has_transition\) \{\n"
    r"(?P<body>(?:(?!\n(?P=i)\}).|\n)*?)"
    r"\n(?P=i)\}"
)


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: make_mut_gpu_checkedreturn_seam.py <path-to-gpu_1p0.cpp> <out-mutant.cpp>")
    src, dst = argv[1], argv[2]
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    # T-2916 (TE-370 M1): normalize CRLF -> LF before matching. `.gitattributes` declares `*.cpp
    # text` (tool-native, not `eol=lf`), so a genuinely fresh checkout on Windows (this box's own
    # `core.autocrlf=true`) produces CRLF here even though the worktree this pattern was authored
    # against happened to hold LF -- confirmed by execution (clean-clone proof run, T-2916 M1):
    # the pattern found 0 occurrences against a fresh clone's own CRLF checkout. Output is written
    # LF-only, which the toolchain compiles identically either way.
    text = text.replace("\r\n", "\n")
    hits = list(_PAT.finditer(text))
    if len(hits) != 1:
        sys.exit("expected exactly 1 occurrence of the checked-return block, found %d -- source "
                  "may have moved, refusing to guess" % len(hits))

    block = hits[0].group(0)
    if "*out_token = -2;" not in block or "seq->ready_for_logits = true;" not in block:
        sys.exit("checked-return block drifted, refusing mutation")
    out = text[:hits[0].start()] + block.replace("if (!has_transition)", "if (false && !has_transition)", 1) + text[hits[0].end():]
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print("wrote %s" % dst)


if __name__ == "__main__":
    main(sys.argv)
