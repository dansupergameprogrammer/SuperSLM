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
    "c": Mutant(GPU, "*out_schema_bound = schema_index >= 0 ? 1 : 0;",
                 "*out_schema_bound = 0; // MUT c: hide SLM5 binding"),
    "d": Mutant(GPU, "IsAcceptingState(*entry, seq->dfa_walk_state)",
                 "(seq->dfa_walk_state != 0u) /* MUT d */"),
    "e": Mutant(GPU, "if (value == state) return true;",
                 "if (value == state) return false; // MUT e"),
    "f": Mutant(GPU, "// dead-end preserves dfa_walk_state",
                 "seq->dfa_walk_state = 0u; // MUT f: corrupt dead-end predecessor"),
    "g": Mutant(CPU,
                 "seq->dfa_walk_state = seq->bound_schema ? 0u : kDfaWalkStateUnused;\n\t}",
                 "/* MUT g: stale prompt-only adoption walk */\n\t}"),
    "h": Mutant(GPU, "const bool accepting = IsAcceptingState(*entry, seq->dfa_walk_state);",
                 "static const bool accepting = IsAcceptingState(*entry, seq->dfa_walk_state); // MUT h: stale cache"),
    "i": Mutant(GPU,
                 "const superslm::SchemaEntry* entry =\n"
                 "\t    seq->model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));",
                 "static const int32_t schema_index = seq->bound_schema_index; // MUT i: stale binding\n"
                 "\tconst superslm::SchemaEntry* entry =\n"
                 "\t    seq->model->schemas.ByIndex(static_cast<size_t>(schema_index));"),
    "j": Mutant(GPU, "IsAcceptingState(*entry, seq->dfa_walk_state)",
                 "true /* MUT j: universal acceptance */"),
    "m": Mutant(GPU,
                 "\tstd::fill(seq->hidden_codes.begin(), seq->hidden_codes.end(), 0);",
                 "\tif (seq->layer_index != 0) return SSLM_BUSY; // MUT m: reset refusal\n"
                 "\tstd::fill(seq->hidden_codes.begin(), seq->hidden_codes.end(), 0);"),
}

# D-SLM7625's flag and entry-clear anchors do not exist at the red pin. Their exact
# source-copy recipes are committed here before the builder writes either anchor.
PREPARED = {
    "k": Mutant(
        GPU,
        "\tif (!seq->bind_eligible) return SSLM_SEQUENCE_REJECTED; // D-SLM7625 bind gate",
        "\tif (false) return SSLM_SEQUENCE_REJECTED; // MUT k: bind flag deleted",
    ),
    "l": Mutant(
        GPU,
        "\tseq->bind_eligible = false; // D-SLM7625 generation entry: gpu prompt prefill",
        "\t// MUT l: omitted D-SLM7625 generation-entry clear",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mutant", choices=sorted(set(MUTANTS) | set(PREPARED)), required=True)
    args = parser.parse_args()
    if args.mutant in PREPARED:
        mutant = PREPARED[args.mutant]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            f"file={mutant.file}\nanchor={mutant.anchor}\nreplacement={mutant.replacement}\n",
            encoding="utf-8",
        )
        print(f"MUTANT {args.mutant} PREPARED source={mutant.file} exact_anchor_absent_at_red_pin")
        return 0
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
