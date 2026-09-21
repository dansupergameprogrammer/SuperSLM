"""T-2909 (Curie) -- generates the MUT_NOSEAM guard-vitality mutant `run_mutants_cpu_deadend.bat`
builds: a scratch copy of `src/sslm_abi.cpp` with ONLY the seam's own consumption call deleted
(`MaybeInjectCpuFinishDegenerateLogitRow(logit_row, ...)`, the greedy branch's masked-argmax
site, T-2872 closing T-2870 F4). The armed flag itself, and every other line, is untouched: the
seam still ARMS under this mutant, but nothing ever reads it, so `CpuCell2DegenerateRowTwin`'s
own Cell 2(i)/(ii) assertions (`-2`/`SSLM_OK`, `dfa_walk_state` pinned at S_e) must turn red
while every other cell in the same binary (Cell 1's routes, D1, damped-greedy, save/restore)
stays green -- nothing else in this suite ever arms the flag the deleted call would have
consumed. This closes plan Sec3.10.3 row 11's own requirement for the GPU twin ("a single-point
mutant... must turn both Cell 1 and Cell 2 red") on the CPU side, where the analogous historical
defect does not exist (CPU's own masked-argmax `has_transition` check was already correct at
a3f89cb, Sec3.10.4's own cost table) -- the seam's consumption call is what stands in for a
"checked return" to revert here.

A single exact-line replacement, not a general patcher: refuses (nonzero exit, no output
written) if the source line has moved or changed, rather than silently mutating the wrong
line or leaving the seam still wired.

Usage: python make_mut_noseam.py <path-to-sslm_abi.cpp> <out-mutant.cpp>
"""
import sys

NEEDLE = "\t\t\tMaybeInjectCpuFinishDegenerateLogitRow(logit_row, static_cast<int32_t>(c.vocab_size));\r\n"
REPLACEMENT = "\t\t\t// T-2909 MUT_NOSEAM: seam consumption deleted (guard-vitality mutant).\r\n"


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: make_mut_noseam.py <path-to-sslm_abi.cpp> <out-mutant.cpp>")
    src, dst = argv[1], argv[2]
    with open(src, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    hits = [i for i, line in enumerate(lines) if line == NEEDLE]
    if len(hits) != 1:
        sys.exit("expected exactly one occurrence of the seam-consumption line, found %d "
                  "(source may have moved -- refusing to guess)" % len(hits))
    lines[hits[0]] = REPLACEMENT
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    print("wrote %s (seam-consumption call removed at original line %d)" % (dst, hits[0] + 1))


if __name__ == "__main__":
    main(sys.argv)
