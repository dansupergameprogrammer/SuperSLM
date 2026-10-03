"""One-time S0 capture of independent reference counts, before model output."""
import json
from pathlib import Path

from resource_pairs import DIAGNOSTICS, pair

data = {}
for guard in DIAGNOSTICS:
    _under, _over, below, above = pair(guard)
    data[guard] = {"under": below, "over": above}
(Path(__file__).parent / "resource-model-counts.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
