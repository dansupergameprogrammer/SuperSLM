"""C19 dynamic scale reciprocal — output exactness vs arbitrary-precision oracle.

Construction C19 (SuperSLM_Plan.md §6.8, D-SLM49): R = round_half_up(2**62 / Dn)
for every Dn in [2**30, 2**31). This packet (packet-01) authors the achievement
claim cell over a sampled domain; the uncorrected-Newton negative control is
packet 2.

See: Claude/Curie/packets/packet-01.md.
"""

import random

import pytest

from conftest import api
from rung1_ref import reciprocal_oracle

MODULE = "reference_pipeline.intmath"


def _build_domain():
    """Sample Dn across the full normalized domain [2**30, 2**31).

    Includes:
    - The two domain endpoints: 2**30 and 2**31 - 1
    - Every power-of-two-plus-small-offset (2**30 + k and 2**31 - 1 - k for k in 0..8)
    - Domain midpoint and its neighbours
    - ~200 deterministic pseudo-random samples (seed 1729 for reproducibility)
    """
    values = set()

    # Domain endpoints
    values.add(2 ** 30)
    values.add(2 ** 31 - 1)

    # Powers of two near boundaries
    for k in range(9):
        values.add(2 ** 30 + k)
        values.add(2 ** 31 - 1 - k)

    # Domain midpoint and neighbours
    midpoint = 2 ** 30 + (2 ** 31 - 2 ** 30) // 2
    values.add(midpoint - 1)
    values.add(midpoint)
    values.add(midpoint + 1)

    # Deterministic pseudo-random sample (~200 values)
    rng = random.Random(1729)
    values.update(rng.sample(range(2 ** 30, 2 ** 31), 200))

    return sorted(values)


DOMAIN = _build_domain()


@pytest.mark.parametrize("dn", DOMAIN)
def test_dynamic_scale_reciprocal_matches_the_arbitrary_precision_oracle(dn):
    """The achievement claim: C19's output is pinned exactly.

    R = round_half_up(2**62 / Dn) for every Dn in [2**30, 2**31). The
    implementation (once it exists) must reproduce that exact integer over a
    sampled slice of the domain.
    """
    dynamic_scale_reciprocal = api(MODULE, "dynamic_scale_reciprocal")
    assert dynamic_scale_reciprocal(dn) == reciprocal_oracle(dn)


# ==============================================================================
# Packet 2 — uncorrected-Newton negative control
# ==============================================================================


def test_the_oracle_can_see_uncorrected_newton_is_wrong():
    """G-8(2): the suite's oracle must be able to see the uncorrected-Newton defect class,
    or a bare-Newton implementation would pass every C19 cell in this suite silently."""
    from rung1_ref import uncorrected_newton_reciprocal

    rng = random.Random(1729)
    sample = [rng.randint(2 ** 30, 2 ** 31 - 1) for _ in range(500)]
    disagreements = sum(
        1 for dn in sample
        if reciprocal_oracle(dn) != uncorrected_newton_reciprocal(dn)
    )
    # Not a tight bound -- the exact-Fraction reconstruction under-states the real int64
    # defect rate (see rung1_ref.uncorrected_newton_reciprocal's docstring). The claim under
    # test is only that the defect is NOT rare -- a materially wrong construction, not a
    # rounding curiosity at a handful of boundary points.
    assert disagreements / len(sample) > 0.05, (
        f"uncorrected Newton agreed with the exact oracle on "
        f"{100 * (1 - disagreements / len(sample)):.1f}% of the sample -- too close to see "
        f"the defect class D-SLM49 measured (54-62% wrong on the real int64 construction)"
    )


def test_dynamic_scale_reciprocal_does_not_reduce_to_uncorrected_newton():
    """The actual negative control: an implementation that dropped the two correction
    steps (bare 3-iteration Newton) must disagree with `dynamic_scale_reciprocal` on a
    material fraction of the domain -- if it agrees everywhere, either the correction
    steps are missing or this cell's sample is too small to see it."""
    from rung1_ref import uncorrected_newton_reciprocal

    dynamic_scale_reciprocal = api(MODULE, "dynamic_scale_reciprocal")
    rng = random.Random(1729)
    sample = [rng.randint(2 ** 30, 2 ** 31 - 1) for _ in range(500)]
    disagreements = sum(
        1 for dn in sample
        if dynamic_scale_reciprocal(dn) != uncorrected_newton_reciprocal(dn)
    )
    assert disagreements / len(sample) > 0.05, (
        "dynamic_scale_reciprocal agreed with bare uncorrected Newton on almost the whole "
        "sample -- the correction steps appear to be missing (D-SLM49's defect class)"
    )
