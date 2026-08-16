"""§14 rung-1 cost / data-independence cells.

Audit item: S-2 / G-8(8). Packets authored independently verify the four fixed,
data-independent op-count terms in SuperSLM_Plan.md §14's rung-1 bullet. See
Claude/Curie/packets/packet-09.md for the design.

Failure this catches: a compiled implementation that is output-correct on every other
rung-1 cell but has a DATA-DEPENDENT branch (e.g. a C22 fast-path that skips the wide
intermediate when |x_i| is small) would pass every numeric cell in the whole roster and
still violate the bounded-work ceiling (G1) by construction — invisible to every cell in
packets 1–8 and 10.
"""

import pytest

from conftest import api, require


# ==============================================================================
# Cell 1 — C19: exactly 5 iterates (3 Newton + 2 correction), data-independent
# ==============================================================================


def test_dynamic_scale_reciprocal_trace_length_is_always_five_and_data_independent():
    dynamic_scale_reciprocal_trace, newton_iters, corr_steps = api(
        "reference_pipeline.intmath", "dynamic_scale_reciprocal_trace",
        "DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS", "DYNAMIC_RECIPROCAL_CORRECTION_STEPS")
    assert newton_iters == 3
    assert corr_steps == 2
    domain_sample = [2 ** 30, 2 ** 30 + 1, 2 ** 31 - 1, 2 ** 31 - 2,
                      2 ** 30 + (2 ** 31 - 2 ** 30) // 2]   # floor, floor+1, ceil, ceil-1, mid
    lengths = {dn: len(dynamic_scale_reciprocal_trace(dn)) for dn in domain_sample}
    distinct = sorted(set(lengths.values()))
    assert distinct == [5], f"trip count varies with data: observed lengths {lengths}"
    dynamic_scale_reciprocal = api("reference_pipeline.intmath", "dynamic_scale_reciprocal")
    for dn in domain_sample:
        trace = dynamic_scale_reciprocal_trace(dn)
        assert trace[-1] == dynamic_scale_reciprocal(dn), (
            f"trace's last iterate does not match the returned value at Dn={dn} -- the trace "
            f"may be a decoy of the right length beside a data-dependent real computation"
        )


# ==============================================================================
# Cell 2 — C20: visits every element exactly once, regardless of value
# ==============================================================================


class _CountingSequence:
    """Wraps a list, counting __getitem__/iteration accesses -- a reduction that short-
    circuits on an early large value visits fewer elements than len(x); this catches that
    without needing max_abs_reduce to expose an internal trace."""
    def __init__(self, values):
        self._values = list(values)
        self.access_count = 0

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        self.access_count += 1
        return self._values[idx]

    def __iter__(self):
        for v in self._values:
            self.access_count += 1
            yield v


def test_max_abs_reduce_visits_every_element_exactly_once_regardless_of_value():
    max_abs_reduce = api("reference_pipeline.intmath", "max_abs_reduce")
    # The extreme value FIRST -- an early-exit-on-INT32_MIN implementation would stop here.
    fixture = _CountingSequence([-(2 ** 31), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = max_abs_reduce(fixture)
    assert result == 2 ** 31
    assert fixture.access_count == len(fixture), (
        f"max_abs_reduce visited {fixture.access_count} of {len(fixture)} elements -- the "
        f"reduction is data-dependent (an early-exit shortcut), which §14 forbids"
    )
    # And the extreme value LAST -- symmetric check, an implementation that only short-
    # circuits when the max comes early would pass the first fixture and fail this one.
    fixture2 = _CountingSequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -(2 ** 31)])
    max_abs_reduce(fixture2)
    assert fixture2.access_count == len(fixture2)


# ==============================================================================
# Cell 3 — C21: fixed op count (one bit-scan + one shift + one select) on both branches
# ==============================================================================


def test_normalize_scale_calls_clz64_exactly_once_on_both_branches(monkeypatch):
    """§14: C21's op count is fixed -- one bit-scan + one shift + one select -- across
    both the s>=0 and s<0 branches. Instrumented via the pinned requirement that
    normalize_scale calls clz64 through the module-global reference (see the design doc's
    API pin, §1.2/§6 item 3)."""
    module = require("reference_pipeline.intmath")
    normalize_scale = api("reference_pipeline.intmath", "normalize_scale")
    real_clz64 = module.clz64
    calls = []

    def counting_clz64(n):
        calls.append(n)
        return real_clz64(n)

    monkeypatch.setattr(module, "clz64", counting_clz64)

    for d_prime, branch in [(2 ** 20, "s>=0"), (2 ** 31, "s<0 (the named edge)")]:
        calls.clear()
        normalize_scale(d_prime)
        assert len(calls) == 1, (
            f"normalize_scale called clz64 {len(calls)} times on the {branch} branch "
            f"(D'={d_prime}); §14 pins exactly one bit-scan per call"
        )


# ==============================================================================
# Cell 4 — C22: decomposition op count is constant for a fixed implementation
# ==============================================================================


def test_requant_token_code_trace_length_is_constant_across_sampled_inputs():
    requant_token_code_trace, requant_token_code = api(
        "reference_pipeline.intmath", "requant_token_code_trace", "requant_token_code")
    from rung1_ref import normalize_scale_oracle, reciprocal_oracle
    sample_points = []
    for d_prime in [1, 2 ** 15, 2 ** 30, 2 ** 31 - 1, 2 ** 31]:
        dn, s = normalize_scale_oracle(d_prime)
        r = reciprocal_oracle(dn)
        for x in [0, 1, d_prime // 2, d_prime, -1 if d_prime > 1 else 0, -(d_prime // 2)]:
            if -d_prime <= x <= d_prime:
                sample_points.append((x, r, s))
    lengths = {(x, r, s): len(requant_token_code_trace(x, r, s)) for (x, r, s) in sample_points}
    distinct = sorted(set(lengths.values()))
    assert len(distinct) == 1, (
        f"C22's decomposition op count varies with the input: observed lengths {distinct} "
        f"-- §14 requires a constant count for any FIXED implementation choice"
    )
