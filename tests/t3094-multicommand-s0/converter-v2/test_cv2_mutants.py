"""Reference control for item 6: every mutant in mutants.json names cells that exist in this suite."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_every_mutant_names_existing_cells():
    mutants = json.loads((HERE / "mutants.json").read_text(encoding="utf-8"))["mutants"]
    assert len({m["id"] for m in mutants}) == len(mutants)
    for limit in ("minItems", "recursion_depth", "nfa_states", "dfa_states", "token_transitions",
                  "serialized_mask_bytes", "tracked_temporary_bytes"):
        assert any(m["id"] == f"G-{limit}" for m in mutants), limit
    for m in mutants:
        assert m["killed_by"], m["id"]
        for cell in m["killed_by"]:
            module, _, rest = cell.partition("::")
            function = rest.split("[", 1)[0]
            mod = importlib.import_module(module[:-3])
            assert callable(getattr(mod, function, None)), f"{m['id']}: {cell} does not exist"
