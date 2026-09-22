"""Exact-anchor source mutants for §3.10.8; refuses absent or ambiguous anchors."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mutant:
    file: str
    anchor: str
    replacement: str


GPU = "src/gpu/gpu_1p0.cpp"
CPU = "src/sslm_abi.cpp"
MUTANTS = {
    "a": Mutant(GPU, "*out_schema_accepting = 0; // unbound contract",
                 "*out_schema_accepting = 1; // MUT a"),
    "b": Mutant(GPU, "*out_schema_bound = 0; // unbound contract",
                 "*out_schema_bound = 1; // MUT b"),
    "c": Mutant(GPU, "*out_schema_bound = restored_schema_index >= 0 ? 1 : 0;",
                 "*out_schema_bound = 0; // MUT c: hide SLM5 binding"),
    "d": Mutant(GPU, "IsAcceptingState(*entry, seq->dfa_walk_state)",
                 "(seq->dfa_walk_state != 0u) /* MUT d */"),
    "e": Mutant(GPU, "return std::binary_search(begin, end, state);",
                 "return false; // MUT e"),
    "f": Mutant(GPU, "// dead-end preserves dfa_walk_state",
                 "seq->dfa_walk_state = next_state; // MUT f"),
    "g": Mutant(CPU,
                 "seq->dfa_walk_state = seq->bound_schema ? 0u : kDfaWalkStateUnused;\n\t}",
                 "/* MUT g: stale prompt-only adoption walk */\n\t}"),
    "h": Mutant(GPU, "const bool accepting = IsAcceptingState(*entry, seq->dfa_walk_state);",
                 "const bool accepting = seq->cached_schema_accepting; // MUT h"),
    "i": Mutant(GPU, "const int32_t schema_index = seq->bound_schema_index;",
                 "const int32_t schema_index = seq->prior_bound_schema_index; // MUT i"),
    "j": Mutant(GPU, "IsAcceptingState(*entry, seq->dfa_walk_state)",
                 "true /* MUT j: universal acceptance */"),
    "k": Mutant(GPU, "if (seq->context_length != 0) {\n\t\treturn SSLM_SEQUENCE_REJECTED;\n\t}",
                 "if (false) { // MUT k: late-bind guard removed\n\t\treturn SSLM_SEQUENCE_REJECTED;\n\t}"),
    "l": Mutant(GPU,
                 "\tstd::fill(seq->hidden_codes.begin(), seq->hidden_codes.end(), 0);",
                 "\tif (seq->layer_index != 0) return SSLM_BUSY; // MUT l\n"
                 "\tstd::fill(seq->hidden_codes.begin(), seq->hidden_codes.end(), 0);"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mutant", choices=sorted(MUTANTS), required=True)
    args = parser.parse_args()
    mutant = MUTANTS[args.mutant]
    source_path = args.engine / mutant.file
    source = source_path.read_text(encoding="utf-8")
    count = source.count(mutant.anchor)
    if count != 1:
        raise SystemExit(
            f"MUTANT {args.mutant} REFUSED: {source_path} expected one exact anchor, found {count}"
        )
    mutated = source.replace(mutant.anchor, mutant.replacement, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(mutated, encoding="utf-8")
    print(f"MUTANT {args.mutant} READY source={mutant.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
