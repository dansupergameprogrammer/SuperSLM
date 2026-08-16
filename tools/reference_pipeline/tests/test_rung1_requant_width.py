"""C22 wide-intermediate boundary — joint-attainment and sibling points.

Plan §6.8 C22 width clause: "`|x_i · 127 · R|` reaches ~2^70 (127·2^63, jointly
attainable — a token containing INT32_MIN forces `R = 2^32` with `x_i = −2^31` present)
— int64 alone overflows; the implementation uses a wider intermediate or an exact
decomposition, and conformance is by output exactness against the pinned formula, not by
blessing any decomposition." (D-SLM52)

Audit item: S-3 / G-8(9). Reference: Claude/Curie/packets/packet-10.md
"""

from conftest import api
from rung1_ref import (
    normalize_scale_oracle, reciprocal_oracle, requant_code_oracle, INT32_MIN,
)


def test_requant_token_code_at_the_joint_attainment_width_boundary():
    """A token whose max-abs element is INT32_MIN forces D=2**31 -> D'=2**31 -> Dn=2**30
    -> R=2**32 through C20/C21/C19; C22 on x_i=-2**31 then produces |x_i*127*R| ==
    127*2**63 ~= 2**70, jointly attainable and int64-overflowing. Confirmed against the
    pinned formula's own arbitrary-precision oracle here, not against whatever the
    implementation's wide-intermediate CHOICE happens to produce."""
    requant_token_code = api("reference_pipeline.intmath", "requant_token_code")

    d_prime = 2 ** 31          # D = D' = 2**31, from an INT32_MIN-present token (C20)
    dn, s = normalize_scale_oracle(d_prime)
    assert (dn, s) == (2 ** 30, -1)               # C21, confirmed inline (see design doc §7)
    r = reciprocal_oracle(dn)
    assert r == 2 ** 32                            # C19, exact (2**62/2**30 == 2**32 exactly)

    x_i = INT32_MIN                                # -2**31
    intermediate_magnitude = abs(x_i * 127 * r)
    assert intermediate_magnitude == 127 * 2 ** 63  # confirms this really is the ~2**70 case
    assert intermediate_magnitude >= 2 ** 63        # confirms it genuinely exceeds int64 range

    expected = requant_code_oracle(x_i, r, s)
    assert expected == -127                        # confirmed inline (see design doc §7)
    assert requant_token_code(x_i, r, s) == expected
    assert type(requant_token_code(x_i, r, s)) is int


def test_requant_token_code_at_sibling_near_but_under_width_boundaries():
    requant_token_code = api("reference_pipeline.intmath", "requant_token_code")
    # (D', x_i) pairs chosen so |x_i * 127 * R| sits in a range that exercises the wide
    # intermediate without landing on the exact clamp boundary -- unlike the joint-attainment
    # cell, these produce a range of DISTINCT int8 codes, so a decomposition bug that only
    # breaks near the very top of the range would still be caught.
    for bit_len in [31, 30, 29, 20]:
        d_prime = (1 << bit_len) - 1   # the top of this bit-length class
        dn, s = normalize_scale_oracle(d_prime)
        r = reciprocal_oracle(dn)
        for frac in (0.25, 0.5, 0.75, 0.95):
            x_i = -int(d_prime * frac)   # negative side -- reaches larger |product| given R's
                                          # magnitude at these Dn values than the positive side
                                          # would at the same |x_i|, since s and r covary with D'
            expected = requant_code_oracle(x_i, r, s)
            got = requant_token_code(x_i, r, s)
            assert got == expected, (
                f"D'={d_prime} (bit_len={bit_len}) x_i={x_i}: got {got}, expected {expected} "
                f"(|x_i*127*R|={abs(x_i * 127 * r)}, vs 2**63={2 ** 63})"
            )
