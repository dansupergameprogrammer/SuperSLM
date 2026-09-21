"""T-2916 (Curie), closing TE-370 M1's last residual: generates the GPU_FIXED_NOSLM5 reference
`build_red_suite_gpu.bat`'s `cell_gpu_slm4_dump` cell needs -- the checked-return +
`ready_for_logits` re-arm fix (TE-338/T-2903) ALONE, with no SLM5 blob format -- FROM TRACKED
GIT HISTORY, replacing the out-of-repo, hand-patched `D:\\_te338\\gpu_1p0_fixed.cpp`.

`git show ccf87c1:src/gpu/gpu_1p0.cpp` (a committed object -- the same commit
`build_red_suite_gpu.bat` already pulls `superslm_gpu.cpp`/`gpu_port.h`'s own pristine
pre-SLM5 pair from) is the PRISTINE base this fix predates: it carries the exact
`model->schemas.Transition(...)` unconditional-write bug the checked-return fix corrects.
Confirmed by execution: `D:\\_te338\\gpu_1p0_fixed.cpp` differs from this commit's own
`gpu_1p0.cpp` by exactly this one 10-line block (`diff --strip-trailing-cr`), the same
transformation `make_mut_gpu_checkedreturn_seam.py` reverts on the LIVE tip in the other
direction. This script applies it forward, onto the pristine base, anchored on the unique
`uint32_t next_state = seq->dfa_walk_state;` / unconditional-`Transition` / `seq->dfa_walk_state
= next_state;` sequence -- refuses (nonzero exit, no output written) unless it occurs exactly
once.

Usage: python make_gpu_fixed_noslm5.py <path-to-pristine-gpu_1p0.cpp> <out-fixed.cpp>
(the caller supplies the pristine file, already extracted via `git show ccf87c1:...` -- this
script never calls git itself, matching this suite's existing generator convention.)
"""
import sys

_ANCHOR = (
    "\t\tuint32_t next_state = seq->dfa_walk_state;\n"
    "\t\tmodel->schemas.Transition(*entry, seq->dfa_walk_state, static_cast<uint32_t>(produced),\n"
    "\t\t                           &next_state);\n"
    "\t\tseq->dfa_walk_state = next_state;\n"
)

_REPLACEMENT = (
    "\t\tuint32_t next_state = seq->dfa_walk_state;\n"
    "\t\t// T-2916 (TE-370 M1): TE-338/T-2903's own checked-return + ready-for-logits re-arm fix,\n"
    "\t\t// applied on top of the pristine pre-SLM5 base to regenerate the gpu_fixed_noslm5\n"
    "\t\t// reference in-repo, replacing the out-of-repo D:\\_te338\\gpu_1p0_fixed.cpp.\n"
    "\t\tif (!model->schemas.Transition(*entry, seq->dfa_walk_state, static_cast<uint32_t>(produced),\n"
    "\t\t                                &next_state)) {\n"
    "\t\t\t*out_token = -2;\n"
    "\t\t\tseq->ready_for_logits = true;\n"
    "\t\t\treturn SSLM_OK;\n"
    "\t\t}\n"
    "\t\tseq->dfa_walk_state = next_state;\n"
)


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: make_gpu_fixed_noslm5.py <path-to-pristine-gpu_1p0.cpp> <out-fixed.cpp>")
    src, dst = argv[1], argv[2]
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    count = text.count(_ANCHOR)
    if count != 1:
        sys.exit(f"expected exactly 1 occurrence of the unconditional-write anchor, found {count} "
                  "-- source may have moved, refusing to guess")
    out = text.replace(_ANCHOR, _REPLACEMENT, 1)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv)
