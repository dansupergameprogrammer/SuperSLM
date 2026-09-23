"""Write a GPU miss-path mutant that corrupts the DFA walk before returning -2."""
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_mut_gpu_walkmiss.py <gpu_1p0.cpp> <output.cpp>")
    src = Path(sys.argv[1]).read_text(encoding="utf-8").replace("\r\n", "\n")
    anchor = "\t\tif (!has_transition) {\n\t\t\t// dead-end preserves dfa_walk_state\n\t\t\t*out_token = -2;"
    if src.count(anchor) != 1:
        raise SystemExit(f"walk-miss anchor count={src.count(anchor)}, expected 1")
    mutated = src.replace(anchor, anchor + "\n\t\t\tseq->dfa_walk_state = 0;", 1)
    Path(sys.argv[2]).write_text(mutated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
