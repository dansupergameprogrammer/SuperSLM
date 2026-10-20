"""The keyword-by-node matrix on IntentDoc v2 (plan §6 row 2; T-3094 K1 re-pointed to the v2 nodes).

The keyword inventory and injected values are the commissioned independent catalogue
(`keyword_catalogue.py`, transcribed from the Draft 2020-12 Core and Validation vocabularies). The per-node
allowlist is §6 row 2's, by node kind (cv2_fixtures.KIND_ALLOWED). A placement is a must-refuse cell when
the keyword is outside its node's allowlist.

Excluded, with the reason: `enum` injected at a free-string leaf. The result `{"type": "string", "enum":
["fire"]}` is a valid enum node under the same allowlist, so no converter can refuse it as an unsupported
placement; the T-3094 matrix carries the same placement as a must-refuse (routed in the record).
"""
from __future__ import annotations

import cv2_fixtures as F
from keyword_catalogue import KEYWORDS

REFUSALS = []
EXCLUDED = []
for location, (_, kind) in F.NODES.items():
    for keyword in sorted(KEYWORDS - F.allowed(location)):
        if keyword == "enum" and kind == "string":
            EXCLUDED.append((location, keyword))
        else:
            REFUSALS.append((location, keyword))

ANNOTATION_PLACEMENTS = [(location, a) for location in F.NODES for a in sorted(F.ANNOTATIONS)]
ANNOTATION_VALUES = {"examples": ["sample"], "deprecated": False, "readOnly": False, "writeOnly": False}
