"""Generate the T-2922 source-copy mutants without history or scratch-source inputs.

The builder's live source is the only input.  Every replacement is unique and the generated
copy carries an explicit marker; the batch runner refuses to score an unmarked variant.
At the red pin the API anchors are absent, which is a valid pre-build state rather than a
reason to substitute a historical implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MUTANTS = {
    "a": ("unbound accepting writes 1", "return 0;", "return 1;"),
    "b": ("unbound bound writes 1", "*out_schema_bound = 0;", "*out_schema_bound = 1;"),
    "c": ("SLM5 restore reports unbound", "bound_schema_index", "/* SLM5 binding removed */ -1"),
    "d": ("mid-walk uses state != 0", "accepting_le", "/* MUT d: state != 0 */"),
    "e": ("accepting search always returns 0", "return true;", "return false;"),
    "f": ("dead-end overwrites walk", "dfa_walk_state", "/* MUT f: overwrite walk */"),
    "g": ("CPU prompt-only adoption leaves stale walk", "dfa_walk_state", "/* MUT g: stale adoption walk */"),
    "h": ("GPU membership cache", "dfa_walk_state", "/* MUT h: cached membership */"),
    "i": ("GPU rebind prior schema", "bound_schema_index", "/* MUT i: prior schema */"),
    "j": ("contract accepts universal rule", "schema_accepting == 0", "/* MUT j: universal rule accepted */"),
    "k": ("late-bind guard removed", "context_length != 0", "false"),
    "l": ("Idle reset layer guard restored", "SslmGpuSeqReset", "/* MUT l: reset guard */"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mutant", choices=sorted(MUTANTS), required=True)
    args = parser.parse_args()
    label, anchor, replacement = MUTANTS[args.mutant]
    source = args.source.read_text(encoding="utf-8")
    count = source.count(anchor)
    if count != 1:
        raise SystemExit(f"MUTANT {args.mutant} REFUSED: expected one anchor {anchor!r}, found {count}")
    mutated = source.replace(anchor, replacement, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f'#pragma message("SSLM T2922 MUTANT {args.mutant} APPLIED: {label}")\n' + mutated,
                        encoding="utf-8")
    print(f"MUTANT {args.mutant} READY: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
