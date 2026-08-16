"""Positive-side C22 tie corpus, wired into the suite (packet 5).

Promotes the 3M-point adversarial tie sweep from loki_probes/c22_adversarial_sweep.py
into a pytest cell using rung1_ref oracle, not to re-derive the sweep from scratch but to
wire in an already-proven construction as test coverage. Two-tier structure: full-corpus
oracle self-check (green, proves the fixture is correctly constructed — mirrors the Loki
strike's exact totals), then the implementation check on a representative sub-sample
(red-unimplemented until intmath.requant_token_code exists).

See SuperSLM_Plan.md §6.8 C22, D-SLM52, Tools/superslm_spike/loki_probes/c22_adversarial_sweep.py,
and Claude/Curie/packets/packet-05.md.
"""

import functools
import random
from conftest import api
import rung1_ref


# --- Fixture construction -----
#
# Every corpus builder below is lru_cache(maxsize=1) and called from inside the test
# bodies that need it — NOT at module scope. Building the ~3M-point corpus at import
# cost ~5.8 s, paid by every full-suite collection and every `pytest -k` run that never
# executes these cells.

@functools.lru_cache(maxsize=1)
def _positive_tie_sweep_d_primes():
    """Reproduce the exact D' set from c22_adversarial_sweep.py."""
    rng = random.Random(1729)   # same seed c22_adversarial_sweep.py uses
    d_primes = set()
    for bit_len in range(1, 32):
        lo = 1 << (bit_len - 1)
        hi = (1 << bit_len) - 1
        d_primes.add(lo)
        d_primes.add(hi)
        for _ in range(200):
            d_primes.add(rng.randint(lo, hi))
    d_primes.add(2 ** 31)   # C21's named negative-s edge
    return sorted(d_primes)


@functools.lru_cache(maxsize=1)
def _build_full_corpus():
    """Build the full ~3M-point corpus of (x, d_prime, dn, s, r) tuples.

    Returns: (all_points, divergent_points) where each point is a tuple
    (x, d_prime, dn, s, r).
    """
    all_points = []
    divergent_points = []
    divergent_bitlens = set()
    tested_count = 0
    divergence_count = 0

    d_primes = _positive_tie_sweep_d_primes()

    for d_prime in d_primes:
        # Compute dn, s, r from d_prime
        dn, s = rung1_ref.normalize_scale_oracle(d_prime)
        r = rung1_ref.reciprocal_oracle(dn)

        # For each k in [0, 127) and dx in [-2, -1, 0, 1, 2]
        for k in range(0, 127):
            for dx in (-2, -1, 0, 1, 2):
                x = rung1_ref.positive_tie_boundary_x(d_prime, k) + dx

                # Skip out-of-domain points
                if x < 0 or x > d_prime:
                    continue

                tested_count += 1

                # Check if this point diverges
                candidate_r_composed, candidate_single_rounding = rung1_ref.candidate_pair(
                    x, d_prime, dn, s, r
                )

                if candidate_r_composed != candidate_single_rounding:
                    divergence_count += 1
                    divergent_points.append((x, d_prime, dn, s, r))
                    divergent_bitlens.add(d_prime.bit_length())

                all_points.append((x, d_prime, dn, s, r))

    return all_points, divergent_points, tested_count, divergence_count, divergent_bitlens


@functools.lru_cache(maxsize=1)
def _divergence_points_plus_sample():
    """Build the deterministic sub-sample: all 5,696 divergent points +
    a 2,000-point random draw from non-divergent, deterministic seed.
    """
    all_points, divergent_points, _, _, _ = _build_full_corpus()

    # Start with all divergent points
    sample = list(divergent_points)

    # Get the non-divergent points
    divergent_set = set(divergent_points)
    non_divergent = [p for p in all_points if p not in divergent_set]

    # Draw 2,000 from non-divergent with deterministic seed
    rng = random.Random(42)  # deterministic seed for the sample
    sample_draw = rng.sample(non_divergent, min(2000, len(non_divergent)))
    sample.extend(sample_draw)

    return sample


# ==============================================================================
# Test cells
# ==============================================================================

def test_requant_oracle_reproduces_strike_totals():
    """Cell 1: Oracle self-check (green). Proves the corpus is correctly
    constructed by asserting the Loki strike's exact measured totals:
    tested == 3,014,429, divergences == 5,696, divergent bit-lengths {4..31}.

    This is the "wired into the suite as a fixture, not left probe-side"
    requirement in its strongest form: the exact numbers the strike measured
    are now asserted, not merely quoted.
    """
    _, _, tested_count, divergence_count, divergent_bitlens = _build_full_corpus()
    assert tested_count == 3_014_429
    assert divergence_count == 5_696
    assert divergent_bitlens == set(range(4, 32))


def test_requant_token_code_matches_r_composed_at_positive_side_ties():
    """Cell 2: Implementation check (red-unimplemented until intmath.requant_token_code
    exists). Asserts the pinned R-composed rule against the implementation on a
    representative sub-sample: all 5,696 divergent points (the discriminating set
    that fails for single-rounding) plus a 2,000-point random draw from the
    non-divergent remainder.
    """
    requant_token_code = api("reference_pipeline.intmath", "requant_token_code")

    for x, d_prime, dn, s, r in _divergence_points_plus_sample():
        actual = requant_token_code(x, r, s)
        expected = rung1_ref.requant_code_oracle(x, r, s)
        assert actual == expected, (
            f"Mismatch at x={x}, r={r}, s={s}: "
            f"expected {expected}, got {actual}"
        )


# ==============================================================================
# Packet 6 — negative-side C22 tie corpus (G-8(6b))
#
# §6.2: candidate tie rules coincide on positive composites and separate only on
# negative ones, so packet 5's positive-side fixtures cannot discriminate
# away-from-zero from toward-+∞ or half-even. Because away-from-zero is
# sign-symmetric, mirrored tie inputs are discriminating by construction: at a
# negative exact tie the pinned rule gives -(k+1) where toward-+∞ gives -k.
# This corpus is the one that actually protects C22's tie-direction pin —
# packet 5 alone would let a toward-+∞ misimplementation ship green.
# See Claude/Curie/packets/packet-06.md (amended 2026-07-16).
# ==============================================================================


@functools.lru_cache(maxsize=1)
def _build_negative_tie_sample():
    """The mirrored negative-side tie-adjacent corpus: for each swept D', each code
    boundary k in [0, 127), and each offset dx in (-2, -1, 0, 1, 2),
    x = negative_tie_boundary_x(d_prime, k) + dx (dx applied AFTER the negation,
    i.e. the boundary is mirrored first), keeping only x in [-d_prime, 0]."""
    sample = []
    for d_prime in _positive_tie_sweep_d_primes():
        dn, s = rung1_ref.normalize_scale_oracle(d_prime)
        r = rung1_ref.reciprocal_oracle(dn)
        for k in range(0, 127):
            for dx in (-2, -1, 0, 1, 2):
                x = rung1_ref.negative_tie_boundary_x(d_prime, k) + dx
                if x > 0 or x < -d_prime:
                    continue
                sample.append((x, d_prime, dn, s, r))
    return sample


@functools.lru_cache(maxsize=1)
def _build_exact_tie_points():
    """The mathematically exact negative ties, filtered by rung1_ref.is_exact_composite_tie
    — a property of the INPUT (where the un-rounded composite ratio sits), independent of
    both candidate rounding rules. NEVER filtered by the discrimination outcome: that would
    make the discriminating assertion below a tautology (packet 6, amended 2026-07-16).

    Most boundary points are tie-ADJACENT, not exact (negative_tie_boundary_x returns the
    nearest integer to a usually non-integer-attainable boundary); measured over the pinned
    sweep, exactly 151 of 603,504 in-domain dx == 0 points are exact ties."""
    exact = []
    for d_prime in _positive_tie_sweep_d_primes():
        dn, s = rung1_ref.normalize_scale_oracle(d_prime)
        r = rung1_ref.reciprocal_oracle(dn)
        for k in range(0, 127):
            x = rung1_ref.negative_tie_boundary_x(d_prime, k)
            if x > 0 or x < -d_prime:
                continue
            if rung1_ref.is_exact_composite_tie(x, r, s):
                exact.append((d_prime, k, x, dn, s, r))
    return exact


def round_half_toward_positive_infinity_ratio(numerator, denominator):
    """The C2 rule (SaturatingRoundingDoublingHighMul's tie direction) applied to C22's
    own ratio -- the misimplementation this cell must catch, computed independently of
    requant_code_oracle so the comparison is real."""
    return (2 * numerator + denominator) // (2 * denominator)


def test_the_negative_tie_corpus_is_nonempty_and_discriminating():
    """Oracle self-check for the negative-side corpus (no api(), green immediately) —
    the negative control for the conformance cell below, made explicit: a toward-+∞ or
    half-even misimplementation of requant_token_code would pass packet 5 in full and
    fail only at the exact points this cell proves are discriminating.
    """
    exact_tie_points = _build_exact_tie_points()
    # (a) The control cannot pass vacuously. The pinned sweep (seed 1729) yields exactly
    #     151 exact ties, measured 2026-07-16; a shrunk count means the sweep or the
    #     predicate changed and the corpus must be re-derived, loudly.
    assert len(exact_tie_points) >= 151, (
        f"only {len(exact_tie_points)} exact negative ties found; the pinned sweep yields "
        f"exactly 151 -- the corpus construction has changed or broken"
    )
    # (b) Every shift class is covered structurally, independent of the random draw: the
    #     constructive family (D' = 2**(bl-1), x = -(2**(bl-2))) is an exact tie in every
    #     class (R = 2**32 exactly at the class floor; v2(x) = bl-2 meets 61-s-32).
    covered = {d_prime.bit_length() for (d_prime, k, x, dn, s, r) in exact_tie_points}
    assert set(range(2, 33)) <= covered, (
        f"shift classes without an exact negative tie: {sorted(set(range(2, 33)) - covered)}"
    )
    # (c) Discrimination at EVERY exact tie -- the sign-symmetry argument made executable.
    for d_prime, k, x, dn, s, r in exact_tie_points:
        num, den = x * 127 * r, 1 << (62 - s)
        pinned = rung1_ref.requant_code_oracle(x, r, s)
        toward_pos_inf = max(-127, min(127, round_half_toward_positive_infinity_ratio(num, den)))
        assert pinned != toward_pos_inf, (
            f"D'={d_prime} k={k} x={x}: away-from-zero and toward-+infinity agree at an "
            f"exact negative tie -- the corpus construction is wrong"
        )


def test_requant_token_code_matches_r_composed_at_negative_side_ties():
    """Conformance cell (via api(), red-unimplemented until built): the implementation
    matches the pinned R-composed oracle at every sampled negative-side tie-adjacent
    point — exact ties plus the dx neighbourhood, same shape as packet 5's positive side.
    """
    requant_token_code = api("reference_pipeline.intmath", "requant_token_code")
    failures = [
        (x, d_prime, s, r, got, want)
        for (x, d_prime, dn, s, r) in _build_negative_tie_sample()
        for got in [requant_token_code(x, r, s)]
        for want in [rung1_ref.requant_code_oracle(x, r, s)]
        if got != want
    ]
    assert not failures, f"{len(failures)} negative-tie disagreements; first 5: {failures[:5]}"
