"""Plan §6.8 C21 — normalize_scale, the dynamic per-token scale reduction.

The construction is p = 63 - clz64(D'), s = 30 - p, Dn = D' << s if s >= 0 else D' >> 1,
normalizing a token's peak absolute value D' to a Dn lying in the normalized domain
[2**30, 2**31) (D-SLM51).

Citations: SuperSLM_Plan.md §6.8 C21, D-SLM51, and Curie Packet 07.

C21 rides along inside C19/C22 fixtures today only via probe helpers, asserted there with
a bare `assert`, not a test. A C21 defect that a C19 or C22 fixture doesn't happen to
trigger would ship undetected. This suite isolates C21's correctness independent of
whatever D' values C19/C22 packets picked.
"""

import pytest

from conftest import api
from rung1_ref import normalize_scale_oracle, ALL_D_PRIME_BIT_LENGTHS


def _d_prime_domain():
    """Inputs spanning the domain and the construction's own structure.

    C21 is a bit-length-dependent scale shift: each bit-length class from 1 to 31 spans
    a different range of D', and the shift amount depends on D''s bit length. This
    function constructs at least the two endpoints (2**(bl-1) and 2**bl - 1) plus a
    handful of interior values for each class.
    """
    values = set()
    for bl in ALL_D_PRIME_BIT_LENGTHS:      # 1..31
        lo, hi = 1 << (bl - 1), (1 << bl) - 1
        values.add(lo)
        values.add(hi)
        if hi > lo:
            values.add((lo + hi) // 2)
    return sorted(values)


D_PRIME_DOMAIN = _d_prime_domain()


def test_normalize_scale_postcondition_across_all_31_bit_length_classes():
    """C21 postcondition: Dn ∈ [2^30, 2^31) for every D' in the domain.

    This cell asserts the POSTCONDITION directly, stated without reference to the
    oracle's own internal computation — grounded in the definition, not a recode of
    p = 63 - clz64(D') etc.
    """
    normalize_scale = api("reference_pipeline.intmath", "normalize_scale")
    for d_prime in D_PRIME_DOMAIN:
        dn, s = normalize_scale(d_prime)
        assert 2 ** 30 <= dn < 2 ** 31, (
            f"C21 postcondition violated: D'={d_prime} -> Dn={dn}, not in [2**30, 2**31)"
        )


def test_normalize_scale_at_the_d_prime_2_31_negative_s_edge():
    """§6.8 C21: the only negative-s case is D'=2^31 (p=31, s=-1); it is even, so the
    single right-shift loses no bits (Dn = 2^30 exactly). This is the one case that MUST
    be held in int64 -- stored in int32, 2^31 wraps negative and every downstream arm
    returns silent wrong answers (D-SLM51/D-SLM50).
    """
    normalize_scale = api("reference_pipeline.intmath", "normalize_scale")
    dn, s = normalize_scale(2 ** 31)
    assert s == -1, f"expected s=-1 at D'=2**31 (p=31, s=30-31), got s={s}"
    assert dn == 2 ** 30, f"expected Dn=2**30 (2**31 >> 1, exact -- even, no bits lost), got {dn}"


def test_normalize_scale_matches_the_oracle_across_all_31_bit_length_classes():
    """C21's implementation matches the oracle (exact rational arithmetic) over all
    31 bit-length classes, plus the isolated 2^31 edge.
    """
    normalize_scale = api("reference_pipeline.intmath", "normalize_scale")
    for d_prime in D_PRIME_DOMAIN + [2 ** 31]:
        assert normalize_scale(d_prime) == normalize_scale_oracle(d_prime), f"D'={d_prime}"
