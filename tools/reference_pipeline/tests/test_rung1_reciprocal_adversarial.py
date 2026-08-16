"""Plan §6.8 C19 (D-SLM49) — CI-scale sampled adversarial reciprocal cell.

Audit item: Gap G-8a split into G-8(4a)/(4b); this packet authors G-8(4b).

G-8(4) named two artifacts under one charter line; the audit split them:
  * **(4a)** the exhaustive normalized-domain sweep as a **one-time certification** —
    the C++ per-device-tier gate (§6.8's "Golden gate"), not a pytest cell at all.
  * **(4b)** an **ongoing, every-push CI cell** at the scale of the reciprocal casebook's
    own adversarial fuzz set (~14,077 pairs) — the cell this packet authors.

This cell is distinct from packet 1's sampled cell (a few hundred points, chosen to
include domain boundaries). Packet 4's sample is an order of magnitude larger (~14,000)
and its purpose is CI adversarial coverage at scale, not boundary-focused spot-checking.

Test-design record: Claude/Curie/SuperSLM_Rung1_RedSuite_TestDesign-2026-07-16.md
"""

import random
import sys
from pathlib import Path

from conftest import api

# Import the oracle module — rung1_ref is in the same tests directory
sys.path.insert(0, str(Path(__file__).parent))
import rung1_ref


def _adversarial_sample(size=14077):
    """A deterministic pseudo-random sample of the normalized domain [2**30, 2**31).

    Uses the fixed seed from the reciprocal casebook's own held-out set
    (dynamic-pertoken-scale-reciprocal-solve-2026-07-16.md) so the cell is
    reproducible run to run.
    """
    rng = random.Random(31415926535)
    return [rng.randint(2 ** 30, 2 ** 31 - 1) for _ in range(size)]


DN_SAMPLE = _adversarial_sample()


def test_dynamic_scale_reciprocal_survives_a_ci_scale_adversarial_sample():
    """C19's output matches the arbitrary-precision oracle over a large adversarial sample.

    No tolerance — exact match required, same as packet 1. The sample (~14,000 points)
    is an order of magnitude larger than packet 1's boundary-focused sample, and its
    purpose is ongoing CI coverage against whatever construction ships.
    """
    dynamic_scale_reciprocal = api("reference_pipeline.intmath", "dynamic_scale_reciprocal")
    failures = [
        (dn, dynamic_scale_reciprocal(dn), rung1_ref.reciprocal_oracle(dn))
        for dn in DN_SAMPLE
        if dynamic_scale_reciprocal(dn) != rung1_ref.reciprocal_oracle(dn)
    ]
    assert not failures, (
        f"{len(failures)} of {len(DN_SAMPLE)} adversarial Dn values disagreed with the "
        f"arbitrary-precision oracle; first 5: {failures[:5]}"
    )
